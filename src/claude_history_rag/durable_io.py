"""Held-authority durable file I/O for POSIX and Windows."""

from __future__ import annotations

import contextlib
import ctypes
import errno
import hashlib
import os
import secrets
import shutil
import stat
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# Fixed reasons a pinned root refuses to serve. They are distinct because they
# demand different operator responses: an unbound root may simply not exist yet,
# while a changed identity means the watched object was substituted.
ROOT_UNBOUND = "root_unbound"
ROOT_IDENTITY_CHANGED = "root_identity_changed"
ROOT_NOT_A_DIRECTORY = "root_not_a_directory"
ROOT_IS_LINK_OR_REPARSE = "root_is_link_or_reparse"
ROOT_UNAVAILABLE = "root_unavailable"
PATH_OUTSIDE_ROOT = "path_outside_root"
DESCENDANT_NOT_TRAVERSABLE = "descendant_not_traversable"
SOURCE_TOO_LARGE = "source_too_large"
SOURCE_NOT_DECODABLE = "source_not_decodable"


class UnsafeDurablePathError(OSError):
    """A durable path could traverse or alias an object outside its authority."""


class DurableRootUnavailableError(OSError):
    """A pinned root cannot currently serve as an authority.

    ``reason`` is one of the fixed reason constants above so callers emit a code
    that says which failure occurred instead of one indiscriminate message.
    """

    def __init__(self, message: str, *, reason: str):
        super().__init__(message)
        self.reason = reason


class DurableSizeLimitExceeded(ValueError):
    """A durable source is larger than its caller's bounded snapshot limit."""


class DurableCommitUncertainError(OSError):
    """A mutation committed, but its durability confirmation failed."""

    def __init__(self, message: str, *, committed: bool):
        super().__init__(message)
        self.committed = committed


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _is_reparse(metadata: os.stat_result) -> bool:
    attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(getattr(metadata, "st_file_attributes", 0) & attribute)


def _validate_directory_metadata(path: Path, metadata: os.stat_result) -> None:
    if _is_reparse(metadata) or stat.S_ISLNK(metadata.st_mode):
        raise UnsafeDurablePathError(f"durable directory is a link or reparse point: {path.name}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise UnsafeDurablePathError(f"durable directory is not a directory: {path.name}")


def _validate_regular_metadata(path: Path, metadata: os.stat_result) -> None:
    if _is_reparse(metadata) or stat.S_ISLNK(metadata.st_mode):
        raise UnsafeDurablePathError(f"durable file is a link or reparse point: {path.name}")
    if not stat.S_ISREG(metadata.st_mode):
        raise UnsafeDurablePathError(f"durable file is not regular: {path.name}")
    if metadata.st_nlink != 1:
        raise UnsafeDurablePathError(f"durable file has multiple hard links: {path.name}")


def _target_name(path: Path, root: Path) -> tuple[Path, str]:
    target = _absolute(path)
    if target.parent != root or target.name in {"", ".", ".."}:
        raise UnsafeDurablePathError("durable file must be a direct child of its durable root")
    return target, target.name


@dataclass
class _HeldDirectory:
    path: Path
    descriptor: int | None = None
    windows_handles: tuple[int, ...] = ()


if os.name == "nt":
    import msvcrt
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    _FILE_READ_ATTRIBUTES = 0x0080
    _FILE_LIST_DIRECTORY = 0x0001
    _FILE_ADD_SUBDIRECTORY = 0x0004
    _SYNCHRONIZE = 0x00100000
    _DELETE = 0x00010000
    _GENERIC_WRITE = 0x40000000
    _GENERIC_READ = 0x80000000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _FILE_SHARE_DELETE = 0x00000004
    _OPEN_EXISTING = 3
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_ATTRIBUTE_DIRECTORY = 0x00000010
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
    _DUPLICATE_SAME_ACCESS = 0x00000002
    _MOVEFILE_REPLACE_EXISTING = 0x00000001
    _NT_FILE_RENAME_INFORMATION = 10
    _NT_FILE_DISPOSITION_INFORMATION = 13
    _FILE_OPEN = 1
    _FILE_CREATE = 2
    _FILE_DIRECTORY_FILE = 0x00000001
    _FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
    _FILE_NON_DIRECTORY_FILE = 0x00000040
    _FILE_OPEN_REPARSE_POINT_NT = 0x00200000
    _OBJ_CASE_INSENSITIVE = 0x00000040
    _FILE_ATTRIBUTE_NORMAL = 0x00000080
    _FILE_ATTRIBUTE_DIRECTORY_NT = 0x00000010

    class _FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

    class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", _FILETIME),
            ("ftLastAccessTime", _FILETIME),
            ("ftLastWriteTime", _FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    _kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION),
    ]
    _kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    _kernel32.GetCurrentProcess.argtypes = []
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.DuplicateHandle.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    _kernel32.DuplicateHandle.restype = wintypes.BOOL
    _ntdll.RtlNtStatusToDosError.argtypes = [ctypes.c_long]
    _ntdll.RtlNtStatusToDosError.restype = wintypes.ULONG

    class _IO_STATUS_BLOCK(ctypes.Structure):
        _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]

    class _UNICODE_STRING(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT),
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", wintypes.LPWSTR),
        ]

    class _OBJECT_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.ULONG),
            ("RootDirectory", wintypes.HANDLE),
            ("ObjectName", ctypes.POINTER(_UNICODE_STRING)),
            ("Attributes", wintypes.ULONG),
            ("SecurityDescriptor", wintypes.LPVOID),
            ("SecurityQualityOfService", wintypes.LPVOID),
        ]

    _ntdll.NtCreateFile.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(_OBJECT_ATTRIBUTES),
        ctypes.POINTER(_IO_STATUS_BLOCK),
        wintypes.LPVOID,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.LPVOID,
        wintypes.ULONG,
    ]
    _ntdll.NtCreateFile.restype = ctypes.c_long

    _ntdll.NtSetInformationFile.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_IO_STATUS_BLOCK),
        wintypes.LPVOID,
        wintypes.ULONG,
        ctypes.c_int,
    ]
    _ntdll.NtSetInformationFile.restype = ctypes.c_long

    _nt_flush_buffers_file_ex = getattr(_ntdll, "NtFlushBuffersFileEx", None)
    if _nt_flush_buffers_file_ex is not None:
        _nt_flush_buffers_file_ex.argtypes = [
            wintypes.HANDLE,
            wintypes.ULONG,
            wintypes.LPVOID,
            wintypes.ULONG,
            ctypes.POINTER(_IO_STATUS_BLOCK),
        ]
        _nt_flush_buffers_file_ex.restype = ctypes.c_long

    class _FILE_RENAME_INFO_HEADER(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", wintypes.WCHAR * 1),
        ]


def _windows_handle_information(handle: int) -> tuple[int, int, int]:
    if os.name != "nt":  # pragma: no cover - platform guard
        raise NotImplementedError
    information = _BY_HANDLE_FILE_INFORMATION()
    if not _kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
        raise ctypes.WinError(ctypes.get_last_error())
    identity = (
        (information.dwVolumeSerialNumber << 64)
        | (information.nFileIndexHigh << 32)
        | information.nFileIndexLow
    )
    return information.dwFileAttributes, information.nNumberOfLinks, identity


def _windows_consume_handle_stat(handle: int) -> os.stat_result:
    """Stat one owned handle and release it in the same operation.

    ``open_osfhandle`` takes ownership, so closing the descriptor closes the
    handle. Use this only where the handle would be closed anyway.
    """
    if os.name != "nt":  # pragma: no cover - platform guard
        raise NotImplementedError
    descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    try:
        return os.fstat(descriptor)
    finally:
        os.close(descriptor)


def _windows_handle_stat(handle: int) -> os.stat_result:
    """Stat exactly the object a live handle holds, leaving that handle open.

    A pathname can be re-pointed between two opens, so identity must come from
    the held object itself. The handle is duplicated because ``open_osfhandle``
    would otherwise consume the caller's handle.
    """
    if os.name != "nt":  # pragma: no cover - platform guard
        raise NotImplementedError
    duplicated = wintypes.HANDLE()
    process = _kernel32.GetCurrentProcess()
    if not _kernel32.DuplicateHandle(
        process,
        wintypes.HANDLE(handle),
        process,
        ctypes.byref(duplicated),
        0,
        False,
        _DUPLICATE_SAME_ACCESS,
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return _windows_consume_handle_stat(int(duplicated.value))


def _windows_open_directory(path: Path, *, allow_create: bool = False) -> int:
    """Open and validate a non-reparse directory as a handle authority."""
    if os.name != "nt":  # pragma: no cover - platform guard
        raise NotImplementedError
    handle = _kernel32.CreateFileW(
        str(path),
        _FILE_READ_ATTRIBUTES
        | _FILE_LIST_DIRECTORY
        | (_FILE_ADD_SUBDIRECTORY if allow_create else 0)
        | _SYNCHRONIZE,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        attributes, _, _ = _windows_handle_information(handle)
        if not attributes & _FILE_ATTRIBUTE_DIRECTORY:
            raise UnsafeDurablePathError(f"durable path is not a directory: {path.name}")
        if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise UnsafeDurablePathError(
                f"durable directory is a link or reparse point: {path.name}"
            )
        return int(handle)
    except BaseException:
        _kernel32.CloseHandle(handle)
        raise


def _windows_open_relative_directory(
    held_parent: _HeldDirectory,
    name: str,
    *,
    allow_create: bool = False,
) -> int:
    """Open one directory component beneath an already-held parent authority."""
    handle = _windows_open_relative_handle(
        held_parent,
        name,
        access=(
            _FILE_READ_ATTRIBUTES
            | _FILE_LIST_DIRECTORY
            | (_FILE_ADD_SUBDIRECTORY if allow_create else 0)
        ),
        directory=True,
    )
    try:
        attributes, _, _ = _windows_handle_information(handle)
        if not attributes & _FILE_ATTRIBUTE_DIRECTORY:
            raise UnsafeDurablePathError(f"durable path is not a directory: {name}")
        if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise UnsafeDurablePathError(f"durable directory is a link or reparse point: {name}")
        return handle
    except BaseException:
        _windows_close_handle(handle)
        raise


def _windows_close_handle(handle: int) -> None:
    if os.name == "nt" and not _kernel32.CloseHandle(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_flush_metadata(handle: int, *, directory: bool) -> None:
    """Synchronize one held file object's data and metadata to storage."""
    if os.name != "nt":  # pragma: no cover - platform guard
        raise NotImplementedError
    if _nt_flush_buffers_file_ex is None:
        raise OSError("object-bound Windows metadata flush is unavailable")
    io_status = _IO_STATUS_BLOCK()
    status = _nt_flush_buffers_file_ex(
        handle,
        0,
        None,
        0,
        ctypes.byref(io_status),
    )
    if status < 0:
        object_kind = "directory" if directory else "file"
        raise OSError(
            _ntdll.RtlNtStatusToDosError(status),
            f"held Windows {object_kind} metadata flush failed",
        )


def _windows_assert_held_path_identity(held: _HeldDirectory) -> None:
    if not held.windows_handles:
        raise OSError("Windows held directory authority is unavailable")
    observed = _windows_open_directory(held.path)
    try:
        if (
            _windows_handle_information(observed)[2]
            != _windows_handle_information(held.windows_handles[-1])[2]
        ):
            raise UnsafeDurablePathError("Windows durable directory pathname changed")
    finally:
        _windows_close_handle(observed)


def _windows_open_relative_handle(
    held: _HeldDirectory,
    name: str,
    *,
    access: int,
    create: bool = False,
    directory: bool = False,
) -> int:
    if not held.windows_handles or Path(name).name != name or name in {"", ".", ".."}:
        raise UnsafeDurablePathError("invalid Windows relative durable name")
    name_buffer = ctypes.create_unicode_buffer(name)
    encoded_length = len(name.encode("utf-16-le"))
    unicode_name = _UNICODE_STRING(
        encoded_length,
        encoded_length,
        ctypes.cast(name_buffer, wintypes.LPWSTR),
    )
    attributes = _OBJECT_ATTRIBUTES(
        ctypes.sizeof(_OBJECT_ATTRIBUTES),
        held.windows_handles[-1],
        ctypes.pointer(unicode_name),
        _OBJ_CASE_INSENSITIVE,
        None,
        None,
    )
    handle = wintypes.HANDLE()
    io_status = _IO_STATUS_BLOCK()
    options = (
        (_FILE_DIRECTORY_FILE if directory else _FILE_NON_DIRECTORY_FILE)
        | _FILE_SYNCHRONOUS_IO_NONALERT
        | _FILE_OPEN_REPARSE_POINT_NT
    )
    status = _ntdll.NtCreateFile(
        ctypes.byref(handle),
        access | _SYNCHRONIZE,
        ctypes.byref(attributes),
        ctypes.byref(io_status),
        None,
        _FILE_ATTRIBUTE_DIRECTORY_NT if directory else _FILE_ATTRIBUTE_NORMAL,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        _FILE_CREATE if create else _FILE_OPEN,
        options,
        None,
        0,
    )
    if status < 0:
        raise ctypes.WinError(_ntdll.RtlNtStatusToDosError(status))
    return int(handle.value)


def _windows_delete_relative(
    held: _HeldDirectory,
    name: str,
    *,
    directory: bool = False,
) -> None:
    handle = _windows_open_relative_handle(
        held,
        name,
        access=_DELETE | _FILE_READ_ATTRIBUTES,
        directory=directory,
    )
    try:
        io_status = _IO_STATUS_BLOCK()
        delete_file = ctypes.c_ubyte(1)
        status = _ntdll.NtSetInformationFile(
            handle,
            ctypes.byref(io_status),
            ctypes.byref(delete_file),
            ctypes.sizeof(delete_file),
            _NT_FILE_DISPOSITION_INFORMATION,
        )
        if status < 0:
            raise ctypes.WinError(_ntdll.RtlNtStatusToDosError(status))
    finally:
        _windows_close_handle(handle)


def _windows_rename_relative(
    held: _HeldDirectory,
    source_name: str,
    target_name: str,
    *,
    replace: bool,
    directory: bool = False,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    """Rename through object and directory handles, never a mutable ancestor path.

    ``expected_identity`` binds the rename to one specific object: the source
    name is reopened here, so a caller that created the object must be able to
    prove the reopened handle is still that object.
    """
    if not held.windows_handles:
        raise OSError("Windows held directory authority is unavailable")
    handle = _windows_open_relative_handle(
        held,
        source_name,
        access=_DELETE | _FILE_READ_ATTRIBUTES | _GENERIC_WRITE,
        directory=directory,
    )
    committed = False
    try:
        if (
            expected_identity is not None
            and _identity(_windows_handle_stat(handle)) != expected_identity
        ):
            raise UnsafeDurablePathError("durable rename source identity changed")
        encoded_name = target_name.encode("utf-16-le")
        name_offset = _FILE_RENAME_INFO_HEADER.FileName.offset
        information_size = max(
            ctypes.sizeof(_FILE_RENAME_INFO_HEADER),
            name_offset + len(encoded_name),
        )
        buffer = ctypes.create_string_buffer(information_size)
        header = _FILE_RENAME_INFO_HEADER.from_buffer(buffer)
        header.Flags = _MOVEFILE_REPLACE_EXISTING if replace else 0
        header.RootDirectory = held.windows_handles[-1]
        header.FileNameLength = len(encoded_name)
        ctypes.memmove(ctypes.addressof(buffer) + name_offset, encoded_name, len(encoded_name))
        io_status = _IO_STATUS_BLOCK()
        status = _ntdll.NtSetInformationFile(
            handle,
            ctypes.byref(io_status),
            buffer,
            information_size,
            _NT_FILE_RENAME_INFORMATION,
        )
        if status < 0:
            raise ctypes.WinError(_ntdll.RtlNtStatusToDosError(status))
        committed = True
        _windows_flush_metadata(handle, directory=directory)
        _windows_assert_held_path_identity(held)
    except OSError as error:
        if committed:
            raise DurableCommitUncertainError(
                "held Windows rename committed without durability confirmation",
                committed=True,
            ) from error
        raise
    finally:
        _windows_close_handle(handle)


def _windows_create_directory(
    held_parent: _HeldDirectory,
    path: Path,
) -> tuple[int, int] | None:
    """Publish a new directory name relative to a held parent authority.

    Returns the created object's identity so the caller can prove the name it
    reopens afterwards still resolves to this object. Returns ``None`` when the
    name was published by someone else, because nothing was created to bind.
    """
    for _ in range(128):
        temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmpdir"
        try:
            temporary_handle = _windows_open_relative_handle(
                held_parent,
                temporary.name,
                access=_FILE_LIST_DIRECTORY | _FILE_READ_ATTRIBUTES,
                create=True,
                directory=True,
            )
        except FileExistsError:
            continue
        # Identity comes from the handle this call created, never from a later
        # stat of a name that could already have been re-pointed.
        created = _identity(_windows_consume_handle_stat(temporary_handle))
        try:
            _windows_rename_relative(
                held_parent,
                temporary.name,
                path.name,
                replace=False,
                directory=True,
                expected_identity=created,
            )
            return created
        except BaseException as error:
            committed = False
            temporary_absent = False
            try:
                _relative_stat(held_parent, temporary.name)
            except FileNotFoundError:
                temporary_absent = True
            with contextlib.suppress(OSError):
                committed = (
                    _identity(_relative_stat(held_parent, path.name)) == created
                    and temporary_absent
                )
            if committed:
                raise DurableCommitUncertainError(
                    "durable directory creation committed without confirmation",
                    committed=True,
                ) from error
            with contextlib.suppress(OSError):
                if _identity(_relative_stat(held_parent, temporary.name)) == created:
                    _windows_delete_relative(
                        held_parent,
                        temporary.name,
                        directory=True,
                    )
            if isinstance(error, FileExistsError) or getattr(error, "winerror", None) == 183:
                # Someone else published this name. Nothing of ours was created,
                # so there is no created object to bind the reopen against.
                return None
            raise
    raise FileExistsError("could not allocate an exclusive durable directory")


def _assert_published_directory_identity(
    handle: int,
    created_identity: tuple[int, int] | None,
) -> None:
    """Prove a reopened published name still resolves to the created directory.

    A mismatch means the create committed and the name was then re-pointed at a
    different object, so the outcome is committed but unconfirmed: the caller
    must never write through the substituted directory.
    """
    if created_identity is None:
        return
    try:
        observed = _identity(_windows_handle_stat(handle))
        if observed != created_identity:
            raise DurableCommitUncertainError(
                "durable directory creation committed but its published name "
                "no longer resolves to the created object",
                committed=True,
            )
    except BaseException:
        _windows_close_handle(handle)
        raise


@contextmanager
def _hold_windows_directory(path: Path) -> Iterator[_HeldDirectory]:
    absolute = _absolute(path)
    anchor = Path(absolute.anchor)
    if not absolute.anchor:
        raise UnsafeDurablePathError("Windows durable path must be absolute")
    current = anchor
    handles: list[int] = []
    try:
        handles.append(_windows_open_directory(current))
        for component in absolute.parts[1:]:
            held_parent = _HeldDirectory(
                path=current,
                windows_handles=tuple(handles),
            )
            child_path = current / component
            try:
                handle = _windows_open_relative_directory(
                    held_parent,
                    component,
                    allow_create=True,
                )
            except FileNotFoundError:
                created_identity = _windows_create_directory(held_parent, child_path)
                handle = _windows_open_relative_directory(
                    held_parent,
                    component,
                    allow_create=True,
                )
                _assert_published_directory_identity(handle, created_identity)
            except PermissionError:
                handle = _windows_open_relative_directory(held_parent, component)
            handles.append(handle)
            current = child_path
        yield _HeldDirectory(path=absolute, windows_handles=tuple(handles))
    finally:
        first_error: OSError | None = None
        for handle in reversed(handles):
            try:
                _windows_close_handle(handle)
            except OSError as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error


def _close_handles(handles: Sequence[int]) -> None:
    """Release held authorities in reverse order, surfacing the first failure."""
    first_error: OSError | None = None
    for handle in reversed(list(handles)):
        try:
            if os.name == "nt":
                _windows_close_handle(handle)
            else:
                os.close(handle)
        except OSError as error:
            first_error = first_error or error
    if first_error is not None:
        raise first_error


def held_identity(held: _HeldDirectory) -> tuple[int, int]:
    """Return the identity of the object a held authority actually holds.

    This never restats a pathname, so it cannot be answered by whatever object
    the name happens to point at now.
    """
    if os.name == "nt":
        if not held.windows_handles:
            raise OSError("Windows held directory authority is unavailable")
        return _identity(_windows_handle_stat(held.windows_handles[-1]))
    if held.descriptor is None:
        raise OSError("POSIX held directory authority is unavailable")
    return _identity(os.fstat(held.descriptor))


@contextmanager
def hold_existing_directory(path: Path) -> Iterator[_HeldDirectory]:
    """Hold an existing directory chain, creating nothing.

    ``_hold_directory`` publishes missing components, which is correct for a
    state file the process owns and wrong for a watch root the process only
    observes: a watch root that does not exist must stay absent.
    """
    absolute = _absolute(path)
    if not absolute.anchor:
        raise UnsafeDurablePathError("durable path must be absolute")
    anchor = Path(absolute.anchor)
    if os.name == "nt":
        handles: list[int] = [_windows_open_directory(anchor)]
        try:
            current = anchor
            for component in absolute.parts[1:]:
                held_parent = _HeldDirectory(path=current, windows_handles=tuple(handles))
                handles.append(_windows_open_relative_directory(held_parent, component))
                current = current / component
            yield _HeldDirectory(path=absolute, windows_handles=tuple(handles))
        finally:
            _close_handles(handles)
        return
    flags = _posix_directory_flags()
    descriptors: list[int] = [os.open(anchor, flags)]
    try:
        for component in absolute.parts[1:]:
            descriptors.append(_posix_open_relative_directory(descriptors[-1], component, flags))
        yield _HeldDirectory(path=absolute, descriptor=descriptors[-1])
    finally:
        _close_handles(descriptors)


@contextmanager
def hold_descendant_directory(
    root: _HeldDirectory,
    components: Sequence[str],
) -> Iterator[_HeldDirectory]:
    """Descend from a held root one component at a time, relative to handles.

    Every component is opened against the directory handle above it and refused
    if it is a link or reparse object, so no ancestor pathname can be re-pointed
    partway through the descent.
    """
    opened: list[int] = []
    current = root.path
    try:
        if os.name == "nt":
            handles = list(root.windows_handles)
            for component in components:
                held_parent = _HeldDirectory(path=current, windows_handles=tuple(handles))
                handle = _windows_open_relative_directory(held_parent, component)
                handles.append(handle)
                opened.append(handle)
                current = current / component
            yield _HeldDirectory(path=current, windows_handles=tuple(handles))
            return
        flags = _posix_directory_flags()
        descriptor = root.descriptor
        if descriptor is None:
            raise OSError("POSIX held directory authority is unavailable")
        for component in components:
            try:
                child = _posix_open_relative_directory(descriptor, component, flags)
            except NotADirectoryError as error:
                raise UnsafeDurablePathError(
                    f"durable descendant is not traversable: {component}"
                ) from error
            opened.append(child)
            descriptor = child
            current = current / component
        yield _HeldDirectory(path=current, descriptor=descriptor)
    finally:
        _close_handles(opened)


def copy_held_file(
    held: _HeldDirectory,
    name: str,
    destination: Path,
    *,
    max_bytes: int,
) -> int:
    """Stream one validated regular file into a private destination path.

    The directory entry, the opened descriptor and the entry observed after the
    copy must all identify one regular, single-linked object, so the bytes
    written to ``destination`` come from exactly one source object. The source
    modification time is preserved so callers deriving identity from mtime are
    unaffected by the relocation.
    """
    if max_bytes < 0:
        raise ValueError("durable copy bound must be nonnegative")
    before = _relative_stat(held, name)
    _validate_regular_metadata(Path(name), before)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    if os.name == "nt":
        handle = _windows_open_relative_handle(
            held,
            name,
            access=_GENERIC_READ | _FILE_READ_ATTRIBUTES,
        )
        descriptor = msvcrt.open_osfhandle(handle, flags)
    else:
        assert held.descriptor is not None
        descriptor = os.open(name, flags, dir_fd=held.descriptor)
    try:
        opened = os.fstat(descriptor)
        current = _relative_stat(held, name)
        _validate_regular_metadata(Path(name), opened)
        _validate_regular_metadata(Path(name), current)
        if not (_identity(before) == _identity(opened) == _identity(current)):
            raise UnsafeDurablePathError("durable file identity changed before snapshot")

        written = 0
        target_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        target = os.open(destination, target_flags, 0o600)
        try:
            while True:
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise DurableSizeLimitExceeded("durable file exceeds bounded snapshot limit")
                os.write(target, chunk)
        finally:
            os.close(target)
        # Applied to the private destination after close because a descriptor is
        # not a valid utime target on every platform. Not suppressed: callers
        # derive chunk identity from modification time, so a snapshot that lost
        # it would silently change identity on every pass.
        os.utime(destination, ns=(opened.st_atime_ns, opened.st_mtime_ns))

        final = _relative_stat(held, name)
        if _identity(opened) != _identity(final):
            raise UnsafeDurablePathError("durable file identity changed during snapshot")
        return written
    finally:
        os.close(descriptor)


def lexical_relative_parts(candidate: Path, root: Path) -> tuple[str, ...] | None:
    """Return the components of ``candidate`` beneath ``root``, lexically only.

    ``Path.resolve()`` is deliberately not used: it answers "where does this name
    point right now", which a substituted root makes true for content outside the
    watched subtree. Containment is a question about names; identity is answered
    separately by the held root handle.
    """
    target = _absolute(candidate)
    base = _absolute(root)
    target_parts = target.parts
    base_parts = base.parts
    if len(target_parts) <= len(base_parts):
        return None
    normalized_target = tuple(os.path.normcase(part) for part in target_parts)
    normalized_base = tuple(os.path.normcase(part) for part in base_parts)
    if normalized_target[: len(base_parts)] != normalized_base:
        return None
    tail = target_parts[len(base_parts) :]
    if any(part in {"", ".", ".."} for part in tail):
        return None
    return tail


class PinnedRoot:
    """Bind one directory object as the sole authority over a watched subtree.

    A configured root path is mutable: it can be replaced by a junction, a
    symlink, or another ordinary directory of the same name, after which every
    pathname beneath it resolves to content the operator never authorized. This
    binds the root's filesystem object identity once and refuses to serve any
    other object under that name for the rest of the process lifetime.
    """

    def __init__(self, path: Path):
        self._path = _absolute(path)
        self._identity: tuple[int, int] | None = None

    @property
    def path(self) -> Path:
        """The configured root pathname."""
        return self._path

    @property
    def identity(self) -> tuple[int, int] | None:
        """The bound object identity, or None while still unbound."""
        return self._identity

    @property
    def is_bound(self) -> bool:
        """Whether a root object has been bound."""
        return self._identity is not None

    def bind(self) -> bool:
        """Bind the root object on first availability; re-verify once bound.

        Returns True when a bound root is currently present and unchanged, and
        False when the root is not yet available to bind. Rebinding is
        structurally impossible: the only transitions are unbound to bound and
        bound to error.
        """
        try:
            with hold_existing_directory(self._path) as held:
                observed = held_identity(held)
        except FileNotFoundError:
            return False
        except UnsafeDurablePathError as error:
            raise DurableRootUnavailableError(
                "durable watch root is a link or reparse object",
                reason=ROOT_IS_LINK_OR_REPARSE,
            ) from error
        except NotADirectoryError as error:
            raise DurableRootUnavailableError(
                "durable watch root is not a directory",
                reason=ROOT_NOT_A_DIRECTORY,
            ) from error
        except OSError as error:
            if getattr(error, "errno", None) == errno.ELOOP:
                raise DurableRootUnavailableError(
                    "durable watch root is a link or reparse object",
                    reason=ROOT_IS_LINK_OR_REPARSE,
                ) from error
            raise DurableRootUnavailableError(
                "durable watch root could not be opened",
                reason=ROOT_UNAVAILABLE,
            ) from error
        if self._identity is None:
            self._identity = observed
            return True
        if observed != self._identity:
            raise DurableRootUnavailableError(
                "durable watch root object identity changed",
                reason=ROOT_IDENTITY_CHANGED,
            )
        return True

    def classify(self, candidate: Path) -> str:
        """Return "ok" or the fixed reason this candidate cannot be served."""
        if lexical_relative_parts(candidate, self._path) is None:
            return PATH_OUTSIDE_ROOT
        try:
            if not self.bind():
                return ROOT_UNBOUND
        except DurableRootUnavailableError as error:
            return error.reason
        except OSError:
            return ROOT_UNAVAILABLE
        return "ok"

    @contextmanager
    def hold(self) -> Iterator[_HeldDirectory]:
        """Hold the bound root, proving the held object is the bound object."""
        if self._identity is None:
            raise DurableRootUnavailableError(
                "durable watch root is not bound",
                reason=ROOT_UNBOUND,
            )
        # ``entered`` keeps the reason translation below scoped to acquisition
        # failures. Once the body runs, an identical exception type raised by
        # the caller must propagate unchanged rather than be relabelled a root
        # failure.
        entered = False
        try:
            with hold_existing_directory(self._path) as held:
                if held_identity(held) != self._identity:
                    raise DurableRootUnavailableError(
                        "durable watch root object identity changed",
                        reason=ROOT_IDENTITY_CHANGED,
                    )
                entered = True
                yield held
        except DurableRootUnavailableError:
            raise
        except UnsafeDurablePathError as error:
            if entered:
                raise
            raise DurableRootUnavailableError(
                "durable watch root is a link or reparse object",
                reason=ROOT_IS_LINK_OR_REPARSE,
            ) from error
        except FileNotFoundError as error:
            if entered:
                raise
            raise DurableRootUnavailableError(
                "durable watch root is missing",
                reason=ROOT_UNBOUND,
            ) from error
        except NotADirectoryError as error:
            if entered:
                raise
            raise DurableRootUnavailableError(
                "durable watch root is not a directory",
                reason=ROOT_NOT_A_DIRECTORY,
            ) from error

    @contextmanager
    def snapshot(self, candidate: Path, *, max_bytes: int) -> Iterator[Path]:
        """Yield an immutable private copy of one descendant regular file.

        Every read of the live source happens exactly once, through held
        handles, so line counting, digesting and parsing downstream all observe
        identical bytes instead of racing separate pathname opens.
        """
        parts = lexical_relative_parts(candidate, self._path)
        if parts is None:
            raise DurableRootUnavailableError(
                "durable candidate is outside its watch root",
                reason=PATH_OUTSIDE_ROOT,
            )
        directory = Path(tempfile.mkdtemp(prefix="history-snapshot-"))
        failed = False
        try:
            destination = directory / parts[-1]
            with self.hold() as root, hold_descendant_directory(root, parts[:-1]) as parent:
                copy_held_file(parent, parts[-1], destination, max_bytes=max_bytes)
            yield destination
        except BaseException:
            # Tracked with an explicit flag rather than sys.exc_info(), which
            # reports any ambient handled exception and would therefore swallow
            # a cleanup failure whenever the caller happens to sit inside an
            # except block.
            failed = True
            raise
        finally:
            shutil.rmtree(directory, ignore_errors=True)
            # Surfaced only when nothing else is already propagating. Raising
            # over an in-flight failure would replace a typed, reasoned error
            # with a bare OSError and lose the caller's fail-closed
            # classification of what actually went wrong.
            if directory.exists() and not failed:
                raise OSError("durable snapshot directory could not be removed")


def _posix_directory_flags() -> int:
    required = getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    if not required or not os.supports_dir_fd:
        raise OSError("held POSIX directory authority is unsupported")
    return os.O_RDONLY | required


def _posix_open_relative_directory(
    descriptor: int,
    component: str,
    flags: int,
) -> int:
    """Open one directory component and normalize POSIX link refusal.

    Linux reports an ``O_NOFOLLOW | O_DIRECTORY`` open of a directory symlink
    as ``ENOTDIR`` rather than ``ELOOP``.  Classify the entry through the same
    held parent descriptor after either result so links have one stable public
    failure while ordinary non-directories retain their distinct classification.
    """
    try:
        return os.open(component, flags, dir_fd=descriptor)
    except OSError as error:
        if error.errno not in {errno.ELOOP, errno.ENOTDIR}:
            raise
        try:
            metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError as missing:
            # The entry changed while it was being classified. Refuse the
            # original open result; importantly, never retry with link following.
            raise missing from error
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            raise UnsafeDurablePathError(
                f"durable directory is a link or reparse point: {component}"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise NotADirectoryError(
                errno.ENOTDIR,
                "durable path component is not a directory",
                component,
            ) from error
        # A directory that failed a no-follow directory open changed or became
        # otherwise unsafe between observations. Preserve fail-closed behavior
        # with a bounded error instead of exposing a platform-specific errno.
        raise UnsafeDurablePathError(
            f"durable directory could not be safely traversed: {component}"
        ) from error


def _posix_replace_relative(
    descriptor: int,
    source: str,
    target: str,
) -> None:
    """Atomically replace one name relative to a held directory authority."""
    os.replace(
        source,
        target,
        src_dir_fd=descriptor,
        dst_dir_fd=descriptor,
    )


@contextmanager
def _hold_posix_directory(path: Path) -> Iterator[_HeldDirectory]:
    absolute = _absolute(path)
    anchor = Path(absolute.anchor)
    flags = _posix_directory_flags()
    descriptors: list[int] = []
    current = anchor
    try:
        descriptor = os.open(anchor, flags)
        descriptors.append(descriptor)
        for component in absolute.parts[1:]:
            current = current / component
            try:
                child = _posix_open_relative_directory(descriptor, component, flags)
            except NotADirectoryError as error:
                raise UnsafeDurablePathError(
                    f"durable directory is not a directory: {component}"
                ) from error
            except FileNotFoundError:
                created_identity: tuple[int, int] | None = None
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    # Someone else published this name; nothing of ours exists
                    # to bind the reopen against.
                    pass
                else:
                    created = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                    _validate_directory_metadata(Path(component), created)
                    created_identity = _identity(created)
                try:
                    os.fsync(descriptor)
                except OSError as error:
                    # Only OUR creation can be committed-but-unconfirmed. When
                    # another process published the name, this call created
                    # nothing, so claiming a commit here would tell the caller a
                    # write it never made is durable.
                    if created_identity is None:
                        raise
                    raise DurableCommitUncertainError(
                        "durable directory creation committed without confirmation",
                        committed=True,
                    ) from error
                child = _posix_open_relative_directory(descriptor, component, flags)
                if created_identity is not None:
                    try:
                        opened = os.fstat(child)
                        entry = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                        _validate_directory_metadata(Path(component), opened)
                        _validate_directory_metadata(Path(component), entry)
                        # POSIX has no atomic create-and-open for a directory,
                        # so the created object is bound by the stat taken
                        # immediately after mkdir. That binding, the surviving
                        # directory entry and the descriptor now in use must all
                        # be one object. The Windows branch binds the created
                        # handle itself, which is strictly stronger.
                        if not (created_identity == _identity(entry) == _identity(opened)):
                            raise DurableCommitUncertainError(
                                "durable directory creation committed but its published "
                                "name no longer resolves to the created object",
                                committed=True,
                            )
                    except BaseException:
                        os.close(child)
                        raise
            descriptors.append(child)
            descriptor = child
        yield _HeldDirectory(path=absolute, descriptor=descriptor)
    finally:
        first_error: OSError | None = None
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error


def _hold_directory(path: Path):
    return _hold_windows_directory(path) if os.name == "nt" else _hold_posix_directory(path)


def ensure_durable_directory(path: Path) -> Path:
    """Create, publish, and validate a durable directory through held authority."""
    with _hold_directory(path) as held:
        return held.path


def _relative_stat(held: _HeldDirectory, name: str) -> os.stat_result:
    if os.name == "nt":
        try:
            handle = _windows_open_relative_handle(
                held,
                name,
                access=_FILE_READ_ATTRIBUTES,
            )
        except PermissionError:
            handle = _windows_open_relative_handle(
                held,
                name,
                access=_FILE_READ_ATTRIBUTES,
                directory=True,
            )
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY)
        try:
            return os.fstat(descriptor)
        finally:
            os.close(descriptor)
    assert held.descriptor is not None
    return os.stat(name, dir_fd=held.descriptor, follow_symlinks=False)


def _relative_unlink(
    held: _HeldDirectory,
    name: str,
    *,
    expected_identity: tuple[int, int],
) -> None:
    tombstone_name = f".{name}.{secrets.token_hex(16)}.deleted"
    if os.name == "nt":
        _windows_rename_relative(held, name, tombstone_name, replace=False)
        moved_identity = _identity(_relative_stat(held, tombstone_name))
        if moved_identity != expected_identity:
            _windows_rename_relative(held, tombstone_name, name, replace=False)
            raise UnsafeDurablePathError("durable delete target identity changed")
        try:
            _windows_delete_relative(held, tombstone_name)
        except OSError as error:
            raise DurableCommitUncertainError(
                "durable deletion committed without cleanup confirmation", committed=True
            ) from error
        return
    assert held.descriptor is not None
    _posix_rename_noreplace(held.descriptor, name, tombstone_name)
    moved_identity = _identity(_relative_stat(held, tombstone_name))
    if moved_identity != expected_identity:
        _posix_rename_noreplace(held.descriptor, tombstone_name, name)
        raise UnsafeDurablePathError("durable delete target identity changed")
    try:
        os.fsync(held.descriptor)
    except OSError as error:
        raise DurableCommitUncertainError(
            "durable deletion committed without confirmation", committed=True
        ) from error
    os.unlink(tombstone_name, dir_fd=held.descriptor)
    try:
        os.fsync(held.descriptor)
    except OSError as error:
        raise DurableCommitUncertainError(
            "durable deletion cleanup committed without confirmation", committed=True
        ) from error


def _posix_rename_noreplace(descriptor: int, source: str, target: str) -> None:
    """Rename relative to one directory without ever replacing another name."""
    library = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_target = os.fsencode(target)
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(descriptor, encoded_source, descriptor, encoded_target, 1)
    elif sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        function = library.renameatx_np
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(descriptor, encoded_source, descriptor, encoded_target, 0x00000004)
    else:
        raise OSError("exclusive relative rename is unsupported on this platform")
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def durable_file_exists(path: Path, *, durable_root: Path) -> bool:
    """Return existence while the verified parent authority remains held."""
    root = _absolute(durable_root)
    _, name = _target_name(path, root)
    with _hold_directory(root) as held:
        try:
            metadata = _relative_stat(held, name)
        except FileNotFoundError:
            return False
        _validate_regular_metadata(Path(name), metadata)
        return True


def _read_descriptor(descriptor: int, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = max_bytes + 1
    while remaining:
        chunk = os.read(descriptor, min(remaining, 64 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if len(payload) > max_bytes:
        raise ValueError("durable file exceeds bounded read limit")
    return payload


def read_bytes(path: Path, *, durable_root: Path, max_bytes: int) -> bytes:
    """Read a bounded file relative to a held, non-reparse directory authority."""
    if max_bytes < 0:
        raise ValueError("durable read bound must be nonnegative")
    root = _absolute(durable_root)
    _, name = _target_name(path, root)
    with _hold_directory(root) as held:
        return _read_held_bytes(held, name, max_bytes)


def _read_held_bytes(held: _HeldDirectory, name: str, max_bytes: int) -> bytes:
    before = _relative_stat(held, name)
    _validate_regular_metadata(Path(name), before)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    if os.name == "nt":
        handle = _windows_open_relative_handle(
            held,
            name,
            access=_GENERIC_READ | _FILE_READ_ATTRIBUTES,
        )
        descriptor = msvcrt.open_osfhandle(handle, flags)
    else:
        assert held.descriptor is not None
        descriptor = os.open(name, flags, dir_fd=held.descriptor)
    try:
        opened = os.fstat(descriptor)
        current = _relative_stat(held, name)
        _validate_regular_metadata(Path(name), opened)
        _validate_regular_metadata(Path(name), current)
        if not (_identity(before) == _identity(opened) == _identity(current)):
            raise UnsafeDurablePathError("durable file identity changed before read")
        payload = _read_descriptor(descriptor, max_bytes)
        final = _relative_stat(held, name)
        if _identity(opened) != _identity(final):
            raise UnsafeDurablePathError("durable file identity changed during read")
        return payload
    finally:
        os.close(descriptor)


def read_text(
    path: Path,
    *,
    durable_root: Path,
    max_bytes: int,
    encoding: str = "utf-8",
) -> str:
    """Read and decode bounded durable text through the shared authority."""
    return read_bytes(path, durable_root=durable_root, max_bytes=max_bytes).decode(encoding)


def _create_exclusive_temporary(held: _HeldDirectory, target_name: str) -> tuple[int, str, Path]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _ in range(128):
        name = f".{target_name}.{secrets.token_hex(16)}.tmp"
        path = held.path / name
        try:
            if os.name == "nt":
                handle = _windows_open_relative_handle(
                    held,
                    name,
                    access=_GENERIC_WRITE | _DELETE | _FILE_READ_ATTRIBUTES,
                    create=True,
                )
                descriptor = msvcrt.open_osfhandle(handle, flags)
            else:
                assert held.descriptor is not None
                descriptor = os.open(name, flags, 0o600, dir_fd=held.descriptor)
            return descriptor, name, path
        except FileExistsError:
            continue
    raise FileExistsError("could not allocate an exclusive durable temporary file")


def _cleanup_created(held: _HeldDirectory, name: str, identity: tuple[int, int]) -> None:
    with contextlib.suppress(OSError):
        if _identity(_relative_stat(held, name)) == identity:
            if os.name == "nt":
                _windows_delete_relative(held, name)
            else:
                assert held.descriptor is not None
                os.unlink(name, dir_fd=held.descriptor)


def atomic_write_bytes(path: Path, payload: bytes, *, durable_root: Path) -> None:
    """Atomically replace a file without releasing its directory authority."""
    root = _absolute(durable_root)
    target, target_name = _target_name(path, root)
    with _hold_directory(root) as held:
        try:
            existing = _relative_stat(held, target_name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            _validate_regular_metadata(Path(target_name), existing)

        descriptor, temporary_name, temporary = _create_exclusive_temporary(held, target_name)
        created = os.fstat(descriptor)
        created_identity = _identity(created)
        try:
            with contextlib.suppress(AttributeError, OSError):
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            staged = _relative_stat(held, temporary_name)
            _validate_regular_metadata(Path(temporary_name), staged)
            if _identity(staged) != created_identity:
                raise UnsafeDurablePathError("durable temporary file identity changed")

            try:
                if os.name == "nt":
                    _windows_rename_relative(
                        held,
                        temporary_name,
                        target_name,
                        replace=True,
                    )
                else:
                    assert held.descriptor is not None
                    _posix_replace_relative(
                        held.descriptor,
                        temporary_name,
                        target_name,
                    )
                    try:
                        os.fsync(held.descriptor)
                    except OSError as error:
                        raise DurableCommitUncertainError(
                            "durable replacement committed without confirmation",
                            committed=True,
                        ) from error
            except BaseException as error:
                committed = False
                temporary_absent = False
                try:
                    _relative_stat(held, temporary_name)
                except FileNotFoundError:
                    temporary_absent = True
                with contextlib.suppress(OSError):
                    committed = (
                        _identity(_relative_stat(held, target_name)) == created_identity
                        and temporary_absent
                    )
                if committed and not isinstance(error, DurableCommitUncertainError):
                    raise DurableCommitUncertainError(
                        "durable replacement committed without confirmation",
                        committed=True,
                    ) from error
                raise

            final = _relative_stat(held, target_name)
            _validate_regular_metadata(Path(target_name), final)
            if _identity(final) != created_identity:
                raise UnsafeDurablePathError("durable replacement identity mismatch")
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            _cleanup_created(held, temporary_name, created_identity)
            raise


def atomic_write_text(
    path: Path,
    text: str,
    *,
    durable_root: Path,
    encoding: str = "utf-8",
) -> None:
    """Encode and atomically replace durable text through the shared authority."""
    atomic_write_bytes(path, text.encode(encoding), durable_root=durable_root)


def delete_file(
    path: Path,
    *,
    durable_root: Path,
    expected_sha256: str | None = None,
    missing_ok: bool = True,
) -> bool:
    """Durably remove exactly the validated file bound by optional content identity."""
    root = _absolute(durable_root)
    _, name = _target_name(path, root)
    with _hold_directory(root) as held:
        try:
            metadata = _relative_stat(held, name)
        except FileNotFoundError:
            if missing_ok:
                return False
            raise
        _validate_regular_metadata(Path(name), metadata)
        if expected_sha256 is not None:
            payload = _read_held_bytes(held, name, 16 * 1024 * 1024)
            if hashlib.sha256(payload).hexdigest() != expected_sha256:
                raise UnsafeDurablePathError("durable delete content identity mismatch")
        _relative_unlink(held, name, expected_identity=_identity(metadata))
        return True


def remove_empty_directory(path: Path, *, durable_parent: Path) -> bool:
    """Durably retire an empty child directory through its held parent authority."""
    parent = _absolute(durable_parent)
    directory, name = _target_name(path, parent)
    with _hold_directory(parent) as held:
        try:
            metadata = _relative_stat(held, name)
        except FileNotFoundError:
            return False
        _validate_directory_metadata(Path(name), metadata)
        expected_identity = _identity(metadata)
        tombstone_name = f".{name}.{secrets.token_hex(16)}.deleted"
        if os.name == "nt":
            # Windows cannot flush a directory handle. Keeping an empty directory is safer
            # than acknowledging an unconfirmed removal or leaving a tombstone behind.
            return False
        assert held.descriptor is not None
        _posix_rename_noreplace(held.descriptor, name, tombstone_name)
        moved = _relative_stat(held, tombstone_name)
        if _identity(moved) != expected_identity:
            _posix_rename_noreplace(held.descriptor, tombstone_name, name)
            raise UnsafeDurablePathError("durable directory identity changed")
        try:
            os.fsync(held.descriptor)
        except OSError as error:
            raise DurableCommitUncertainError(
                "durable directory removal committed without confirmation", committed=True
            ) from error
        try:
            os.rmdir(tombstone_name, dir_fd=held.descriptor)
        except OSError:
            _posix_rename_noreplace(held.descriptor, tombstone_name, name)
            raise
        try:
            os.fsync(held.descriptor)
        except OSError as error:
            raise DurableCommitUncertainError(
                "durable directory cleanup committed without confirmation", committed=True
            ) from error
        return True
