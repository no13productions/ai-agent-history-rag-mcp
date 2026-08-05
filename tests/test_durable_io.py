"""Security and durability proofs for the shared durable-file authority."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path

import pytest

from claude_history_rag import durable_io


def test_atomic_write_read_and_replace_round_trip(tmp_path: Path):
    target = tmp_path / "state.json"

    durable_io.atomic_write_bytes(target, b"first", durable_root=tmp_path)
    assert durable_io.read_bytes(target, durable_root=tmp_path, max_bytes=5) == b"first"

    durable_io.atomic_write_bytes(target, b"second", durable_root=tmp_path)
    assert durable_io.read_bytes(target, durable_root=tmp_path, max_bytes=6) == b"second"
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not enforced on Windows")
def test_atomic_write_restricts_file_mode(tmp_path: Path):
    target = tmp_path / "state.json"

    durable_io.atomic_write_bytes(target, b"private", durable_root=tmp_path)

    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_deterministic_temp_name_preoccupation_is_never_opened(tmp_path: Path):
    target = tmp_path / "state.json"
    preoccupied = tmp_path / "state.json.tmp"
    preoccupied.write_bytes(b"outside-sentinel")

    durable_io.atomic_write_bytes(target, b"durable", durable_root=tmp_path)

    assert preoccupied.read_bytes() == b"outside-sentinel"
    assert target.read_bytes() == b"durable"


def test_random_temp_collision_uses_a_new_exclusive_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "state.json"
    occupied = tmp_path / ".state.json.occupied.tmp"
    occupied.write_bytes(b"outside-sentinel")
    names = iter(["occupied", "fresh"])
    monkeypatch.setattr(durable_io.secrets, "token_hex", lambda length: next(names))

    durable_io.atomic_write_bytes(target, b"durable", durable_root=tmp_path)

    assert occupied.read_bytes() == b"outside-sentinel"
    assert target.read_bytes() == b"durable"


def test_atomic_replace_failure_preserves_old_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "state.json"
    durable_io.atomic_write_bytes(target, b"old-state", durable_root=tmp_path)
    if os.name == "nt":
        original_rename = durable_io._windows_rename_relative

        def fail_target_replace(held, source, destination, *, replace, directory=False):
            if destination == target.name:
                raise OSError("injected replace failure")
            return original_rename(
                held,
                source,
                destination,
                replace=replace,
                directory=directory,
            )

        monkeypatch.setattr(durable_io, "_windows_rename_relative", fail_target_replace)
    else:
        original_replace = durable_io.os.replace

        def fail_target_replace(source, destination, **kwargs):
            if Path(destination) == target.name:
                raise OSError("injected replace failure")
            return original_replace(source, destination, **kwargs)

        monkeypatch.setattr(durable_io.os, "replace", fail_target_replace)
    with pytest.raises(OSError, match="replace failure"):
        durable_io.atomic_write_bytes(target, b"new-state", durable_root=tmp_path)

    assert target.read_bytes() == b"old-state"
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_bounded_read_rejects_hardlinked_file(tmp_path: Path):
    target = tmp_path / "state.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside")
    os.link(outside, target)

    with pytest.raises(durable_io.UnsafeDurablePathError, match="multiple hard links"):
        durable_io.read_bytes(target, durable_root=tmp_path, max_bytes=128)


def test_atomic_write_rejects_hardlinked_target_without_overwrite(tmp_path: Path):
    target = tmp_path / "state.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside-sentinel")
    os.link(outside, target)

    with pytest.raises(durable_io.UnsafeDurablePathError, match="multiple hard links"):
        durable_io.atomic_write_bytes(target, b"replacement", durable_root=tmp_path)

    assert outside.read_bytes() == b"outside-sentinel"


def test_bounded_read_rejects_symlink_without_consuming_target(tmp_path: Path):
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside-secret")
    linked = tmp_path / "state.json"
    try:
        linked.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"file symlinks unavailable: {type(error).__name__}")

    with pytest.raises(durable_io.UnsafeDurablePathError, match="link or reparse"):
        durable_io.read_bytes(linked, durable_root=tmp_path, max_bytes=128)


def test_durable_directory_rejects_symlink_or_reparse_point(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "linked-root"
    try:
        linked_root.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {type(error).__name__}")

    with pytest.raises(durable_io.UnsafeDurablePathError, match="link or reparse"):
        durable_io.atomic_write_bytes(
            linked_root / "state.json",
            b"must-not-write",
            durable_root=linked_root,
        )

    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-pinning proof")
def test_windows_new_root_avoids_absolute_mutation_after_authority_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "new-root"
    authority_acquired = False
    original_open = durable_io._windows_open_directory

    def observe_anchor(path, **kwargs):
        nonlocal authority_acquired
        handle = original_open(path, **kwargs)
        authority_acquired = True
        return handle

    def reject_absolute_mutation(*args, **kwargs):
        if authority_acquired:
            pytest.fail("absolute-path mutation followed held-authority acquisition")

    monkeypatch.setattr(durable_io, "_windows_open_directory", observe_anchor)
    monkeypatch.setattr(durable_io._kernel32, "MoveFileExW", reject_absolute_mutation)
    monkeypatch.setattr(os, "replace", reject_absolute_mutation)
    monkeypatch.setattr(os, "rename", reject_absolute_mutation)
    monkeypatch.setattr(os, "remove", reject_absolute_mutation)
    monkeypatch.setattr(os, "unlink", reject_absolute_mutation)
    monkeypatch.setattr(os, "mkdir", reject_absolute_mutation)

    durable_io.atomic_write_bytes(root / "state.json", b"inside", durable_root=root)

    assert authority_acquired is True
    assert (root / "state.json").read_bytes() == b"inside"


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-pinning proof")
def test_windows_replace_avoids_absolute_mutation_after_authority_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "state.json"
    target.write_bytes(b"old")
    authority_acquired = False
    original_open = durable_io._windows_open_directory

    def observe_anchor(path, **kwargs):
        nonlocal authority_acquired
        handle = original_open(path, **kwargs)
        authority_acquired = True
        return handle

    def reject_absolute_mutation(*args, **kwargs):
        if authority_acquired:
            pytest.fail("absolute-path mutation followed held-authority acquisition")

    monkeypatch.setattr(durable_io, "_windows_open_directory", observe_anchor)
    monkeypatch.setattr(durable_io._kernel32, "MoveFileExW", reject_absolute_mutation)
    monkeypatch.setattr(os, "replace", reject_absolute_mutation)
    monkeypatch.setattr(os, "rename", reject_absolute_mutation)
    monkeypatch.setattr(os, "remove", reject_absolute_mutation)
    monkeypatch.setattr(os, "unlink", reject_absolute_mutation)
    monkeypatch.setattr(os, "mkdir", reject_absolute_mutation)

    durable_io.atomic_write_bytes(target, b"new", durable_root=tmp_path)

    assert authority_acquired is True
    assert target.read_bytes() == b"new"


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-pinning proof")
def test_windows_directory_open_failure_precedes_file_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    target = root / "state.json"
    original_open = durable_io._windows_open_relative_directory

    def reject_root(held_parent, name, **kwargs):
        if held_parent.path == root.parent and name == root.name:
            raise PermissionError("injected directory authority failure")
        return original_open(held_parent, name, **kwargs)

    monkeypatch.setattr(durable_io, "_windows_open_relative_directory", reject_root)
    with pytest.raises(PermissionError, match="authority failure"):
        durable_io.atomic_write_bytes(target, b"must-not-write", durable_root=root)

    assert not target.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-relative descent proof")
def test_windows_descendant_junction_swap_cannot_redirect_directory_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    trusted = tmp_path / "trusted"
    trusted_leaf = trusted / "leaf"
    trusted_leaf.mkdir(parents=True)
    parked = tmp_path / "parked"
    outside = tmp_path / "outside"
    outside_leaf = outside / "leaf"
    outside_leaf.mkdir(parents=True)
    sentinel = outside_leaf / "state.json"
    sentinel.write_bytes(b"outside-sentinel")

    original_absolute_open = durable_io._windows_open_directory
    original_relative_open = durable_io._windows_open_relative_handle
    attacked = False

    def swap_trusted_ancestor() -> None:
        nonlocal attacked
        if attacked:
            return
        os.replace(trusted, parked)
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(trusted), str(outside)],
            check=True,
            capture_output=True,
        )
        attacked = True

    def attack_absolute_open(path, **kwargs):
        if Path(path) == trusted_leaf:
            swap_trusted_ancestor()
        return original_absolute_open(path, **kwargs)

    def attack_relative_open(held, name, **kwargs):
        if held.path == trusted and name == trusted_leaf.name:
            swap_trusted_ancestor()
        return original_relative_open(held, name, **kwargs)

    monkeypatch.setattr(durable_io, "_windows_open_directory", attack_absolute_open)
    monkeypatch.setattr(durable_io, "_windows_open_relative_handle", attack_relative_open)
    try:
        with pytest.raises(durable_io.DurableCommitUncertainError) as raised:
            durable_io.atomic_write_bytes(
                trusted_leaf / "state.json",
                b"inside",
                durable_root=trusted_leaf,
            )

        assert attacked is True
        assert raised.value.committed is True
        assert sentinel.read_bytes() == b"outside-sentinel"
        assert (parked / "leaf" / "state.json").read_bytes() == b"inside"
    finally:
        if os.path.isjunction(trusted):
            os.rmdir(trusted)
        if parked.exists():
            os.replace(parked, trusted)


@pytest.mark.skipif(os.name != "nt", reason="Windows write-through publication proof")
def test_windows_new_directory_is_published_write_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    original_flush = durable_io._windows_flush_metadata
    directory_publications = 0

    def observe_flush(handle, *, directory):
        nonlocal directory_publications
        if directory:
            directory_publications += 1
        return original_flush(handle, directory=directory)

    monkeypatch.setattr(durable_io, "_windows_flush_metadata", observe_flush)
    durable_io.atomic_write_bytes(root / "state.json", b"published", durable_root=root)

    assert directory_publications == 1
    assert (root / "state.json").read_bytes() == b"published"


@pytest.mark.skipif(os.name != "nt", reason="Windows directory publication uncertainty proof")
def test_windows_directory_publication_failure_is_uncertain_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    target = root / "state.json"
    original_flush = durable_io._windows_flush_metadata
    injected = False

    def publish_then_fail(handle, *, directory):
        nonlocal injected
        result = original_flush(handle, directory=directory)
        if directory and not injected:
            injected = True
            raise OSError("injected directory flush failure")
        return result

    monkeypatch.setattr(durable_io, "_windows_flush_metadata", publish_then_fail)
    with pytest.raises(durable_io.DurableCommitUncertainError) as raised:
        durable_io.atomic_write_bytes(target, b"published", durable_root=root)

    assert raised.value.committed is True
    assert root.is_dir()
    durable_io.atomic_write_bytes(target, b"published", durable_root=root)
    assert target.read_bytes() == b"published"


@pytest.mark.skipif(os.name != "nt", reason="Windows write-through uncertainty proof")
def test_windows_committed_but_unconfirmed_replace_is_idempotent_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "state.json"
    original_flush = durable_io._windows_flush_metadata
    injected = False

    def commit_then_fail(handle, *, directory):
        nonlocal injected
        result = original_flush(handle, directory=directory)
        if not directory and not injected:
            injected = True
            raise OSError("injected lost confirmation")
        return result

    monkeypatch.setattr(durable_io, "_windows_flush_metadata", commit_then_fail)
    with pytest.raises(durable_io.DurableCommitUncertainError) as raised:
        durable_io.atomic_write_bytes(target, b"new-state", durable_root=tmp_path)

    assert raised.value.committed is True
    assert target.read_bytes() == b"new-state"
    durable_io.atomic_write_bytes(target, b"new-state", durable_root=tmp_path)
    assert target.read_bytes() == b"new-state"


@pytest.mark.skipif(os.name != "nt", reason="Windows fail-closed durability proof")
def test_windows_missing_object_bound_flush_fails_closed_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "state.json"
    monkeypatch.setattr(durable_io, "_nt_flush_buffers_file_ex", None)

    with pytest.raises(durable_io.DurableCommitUncertainError) as raised:
        durable_io.atomic_write_bytes(target, b"committed", durable_root=tmp_path)

    assert raised.value.committed is True
    assert target.read_bytes() == b"committed"


@pytest.mark.skipif(os.name != "nt", reason="Windows published-directory identity proof")
def test_windows_published_directory_replacement_is_rejected_before_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    parked = tmp_path / "parked"
    outside_sentinel = root / "outside.json"
    original_create = durable_io._windows_create_directory
    replaced = False

    def replace_after_publication(held_parent, path):
        nonlocal replaced
        created_identity = original_create(held_parent, path)
        if path == root and not replaced:
            # The directory is published under its final name but not yet
            # reopened: swap it for a different ordinary directory.
            os.replace(root, parked)
            os.mkdir(root)
            outside_sentinel.write_bytes(b"outside-sentinel")
            replaced = True
        return created_identity

    monkeypatch.setattr(durable_io, "_windows_create_directory", replace_after_publication)

    with pytest.raises(durable_io.DurableCommitUncertainError) as raised:
        durable_io.atomic_write_bytes(root / "state.json", b"inside", durable_root=root)

    assert replaced is True
    assert raised.value.committed is True
    assert outside_sentinel.read_bytes() == b"outside-sentinel"
    assert not (root / "state.json").exists()
    assert not (parked / "state.json").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX published-directory identity proof")
def test_posix_published_directory_replacement_is_rejected_before_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    parked = tmp_path / "parked"
    outside_sentinel = root / "outside.json"
    original_fsync = durable_io.os.fsync
    replaced = False

    def replace_after_publication(descriptor):
        nonlocal replaced
        result = original_fsync(descriptor)
        if not replaced and root.is_dir():
            # The parent fsync sits between mkdir and the reopen, so this is the
            # exact publication window the identity binding has to survive.
            os.replace(root, parked)
            os.mkdir(root)
            outside_sentinel.write_bytes(b"outside-sentinel")
            replaced = True
        return result

    monkeypatch.setattr(durable_io.os, "fsync", replace_after_publication)

    with pytest.raises(durable_io.DurableCommitUncertainError) as raised:
        durable_io.atomic_write_bytes(root / "state.json", b"inside", durable_root=root)

    assert replaced is True
    assert raised.value.committed is True
    assert outside_sentinel.read_bytes() == b"outside-sentinel"
    assert not (root / "state.json").exists()
    assert not (parked / "state.json").exists()


def test_atomic_write_refuses_a_replacement_that_is_not_the_staged_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The published name must resolve to the object this call staged.

    Without the post-publication comparison, a name re-pointed between the
    rename and the check is accepted as a successful durable write.
    """
    target = tmp_path / "state.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside-sentinel")

    original_stat = durable_io._relative_stat
    diverted = False

    def report_another_object_as_the_result(held, name):
        nonlocal diverted
        if name == target.name and not diverted:
            observed = original_stat(held, name)
            # Only the FINAL post-rename observation is diverted.
            if observed.st_size == len(b"published"):
                diverted = True
                return os.stat(outside)
        return original_stat(held, name)

    monkeypatch.setattr(durable_io, "_relative_stat", report_another_object_as_the_result)

    with pytest.raises(durable_io.UnsafeDurablePathError, match="identity mismatch"):
        durable_io.atomic_write_bytes(target, b"published", durable_root=tmp_path)

    assert diverted is True
    assert outside.read_bytes() == b"outside-sentinel"


def test_hold_refuses_a_substituted_root_without_reclassifying_first(
    tmp_path: Path,
):
    """hold() is the last defence on the read path and must check identity itself.

    classify()/bind() are not called before a read, so a hold() that trusted its
    pathname would serve a substituted root's content while every containment
    and reparse check still reported success.
    """
    root = tmp_path / "history"
    project = root / "project"
    project.mkdir(parents=True)
    source = project / "session.jsonl"
    source.write_bytes(b"trusted\n")

    pinned = durable_io.PinnedRoot(root)
    assert pinned.bind() is True

    parked = tmp_path / "parked"
    os.replace(root, parked)
    substitute = root / "project"
    substitute.mkdir(parents=True)
    (substitute / "session.jsonl").write_bytes(b"outside-sentinel\n")

    # An ordinary directory: no symlink bit, no reparse bit.
    assert not os.path.islink(root)
    assert root.is_dir()

    # Straight to hold()/snapshot() with no intervening classify() or bind().
    with pytest.raises(durable_io.DurableRootUnavailableError) as held, pinned.hold():
        pass
    assert held.value.reason == durable_io.ROOT_IDENTITY_CHANGED

    with (
        pytest.raises(durable_io.DurableRootUnavailableError) as served,
        pinned.snapshot(source, max_bytes=1024) as snapshot,
    ):
        assert snapshot.read_bytes() != b"outside-sentinel\n"
    assert served.value.reason == durable_io.ROOT_IDENTITY_CHANGED


def test_snapshot_refuses_when_the_entry_changes_after_the_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The entry observed AFTER the copy must still be the object opened."""
    root = tmp_path / "history"
    root.mkdir()
    source = root / "session.jsonl"
    source.write_bytes(b"trusted\n")
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"outside-sentinel\n")

    pinned = durable_io.PinnedRoot(root)
    assert pinned.bind() is True

    original_stat = durable_io._relative_stat
    observations = 0

    def diverge_on_the_final_observation(held, name):
        nonlocal observations
        if name == source.name:
            observations += 1
            # Calls 1 and 2 bracket the open; call 3 is the post-copy check.
            if observations >= 3:
                return os.stat(outside)
        return original_stat(held, name)

    monkeypatch.setattr(durable_io, "_relative_stat", diverge_on_the_final_observation)

    with (
        pytest.raises(durable_io.UnsafeDurablePathError, match="during snapshot"),
        pinned.snapshot(source, max_bytes=1024),
    ):
        pass

    assert observations >= 3


def test_snapshot_cleanup_failure_surfaces_even_inside_an_except_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A cleanup failure must not be swallowed by the caller's ambient context.

    Deciding this from `sys.exc_info()` reports whatever exception the caller is
    already handling, so a snapshot directory that could not be removed would
    silently leak whenever the call happened to sit inside an except block.
    """
    root = tmp_path / "history"
    root.mkdir()
    source = root / "session.jsonl"
    source.write_bytes(b"line\n")

    pinned = durable_io.PinnedRoot(root)
    assert pinned.bind() is True

    original_rmtree = durable_io.shutil.rmtree

    def refuse_removal(target, *args, **kwargs):
        del args, kwargs
        # Leave the directory in place so the guard has something to detect.
        assert Path(target).exists()

    monkeypatch.setattr(durable_io.shutil, "rmtree", refuse_removal)
    try:
        raise ValueError("caller is already handling something")
    except ValueError:
        with (
            pytest.raises(OSError, match="could not be removed"),
            pinned.snapshot(source, max_bytes=1024) as snapshot,
        ):
            assert snapshot.read_bytes() == b"line\n"

    monkeypatch.setattr(durable_io.shutil, "rmtree", original_rmtree)


def test_snapshot_refuses_when_the_entry_and_descriptor_disagree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The directory entry and the opened descriptor must be one object."""
    root = tmp_path / "history"
    root.mkdir()
    source = root / "session.jsonl"
    source.write_bytes(b"trusted\n")
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"outside-sentinel\n")

    pinned = durable_io.PinnedRoot(root)
    assert pinned.bind() is True

    original_stat = durable_io._relative_stat
    swapped = False

    def report_a_different_object(held, name):
        nonlocal swapped
        if name == source.name and not swapped:
            # The directory entry now names an object other than the one the
            # descriptor will open. Both are ordinary regular files, so only the
            # identity comparison can tell them apart.
            swapped = True
            return os.stat(outside)
        return original_stat(held, name)

    monkeypatch.setattr(durable_io, "_relative_stat", report_a_different_object)

    with (
        pytest.raises(durable_io.UnsafeDurablePathError, match="identity changed"),
        pinned.snapshot(source, max_bytes=1024),
    ):
        pass

    assert swapped is True


@pytest.mark.skipif(os.name == "nt", reason="POSIX creation-race durability proof")
def test_posix_lost_creation_race_does_not_claim_a_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Losing the create race means this call committed nothing.

    Reporting committed=True here would tell the caller a write it never made is
    durable, and the watcher marks its in-memory positions durable on exactly
    that signal.
    """
    root = tmp_path / "root"
    original_mkdir = durable_io.os.mkdir
    original_fsync = durable_io.os.fsync
    raced = False

    def lose_the_race(path, mode=0o777, *args, **kwargs):
        nonlocal raced
        # Publish the name first, so our own mkdir loses with FileExistsError.
        if not raced and Path(path).name == root.name:
            raced = True
            original_mkdir(root)
        return original_mkdir(path, mode, *args, **kwargs)

    def fail_fsync(descriptor):
        if raced:
            raise OSError("injected parent fsync failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(durable_io.os, "mkdir", lose_the_race)
    monkeypatch.setattr(durable_io.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="injected parent fsync failure") as raised:
        durable_io.atomic_write_bytes(root / "state.json", b"inside", durable_root=root)

    assert raced is True
    assert not isinstance(raised.value, durable_io.DurableCommitUncertainError)


@pytest.mark.skipif(os.name != "nt", reason="Windows created-object rename binding proof")
def test_windows_created_directory_rename_rejects_substituted_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    substituted = False
    original_open = durable_io._windows_open_relative_handle

    def substitute_temporary_directory(held, name, **kwargs):
        nonlocal substituted
        if not substituted and name.endswith(".tmpdir") and not kwargs.get("create"):
            # Between creation and the publishing rename, replace the temporary
            # directory with a different object carrying the same name.
            os.rmdir(tmp_path / name)
            os.mkdir(tmp_path / name)
            substituted = True
        return original_open(held, name, **kwargs)

    monkeypatch.setattr(durable_io, "_windows_open_relative_handle", substitute_temporary_directory)

    with pytest.raises(durable_io.UnsafeDurablePathError, match="rename source identity changed"):
        durable_io.atomic_write_bytes(root / "state.json", b"inside", durable_root=root)

    assert substituted is True
    assert not root.exists()


def test_delete_rejects_substituted_object_without_removing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "payload.json"
    outside = tmp_path / "outside.json"
    durable_io.atomic_write_bytes(target, b"expected", durable_root=tmp_path)
    outside.write_bytes(b"outside-sentinel")
    expected = hashlib.sha256(b"expected").hexdigest()

    if os.name == "nt":
        original_rename = durable_io._windows_rename_relative

        def substitute_then_move(held, source, destination, *, replace, directory=False):
            if source == target.name and ".deleted" in destination:
                target.unlink()
                os.link(outside, target)
            return original_rename(
                held,
                source,
                destination,
                replace=replace,
                directory=directory,
            )

        monkeypatch.setattr(durable_io, "_windows_rename_relative", substitute_then_move)
    else:
        original_rename = durable_io._posix_rename_noreplace

        def substitute_then_move(descriptor, source, destination):
            if source == target.name and ".deleted" in str(destination):
                target.unlink()
                os.link(outside, target)
            return original_rename(descriptor, source, destination)

        monkeypatch.setattr(durable_io, "_posix_rename_noreplace", substitute_then_move)

    with pytest.raises(durable_io.UnsafeDurablePathError, match="identity changed"):
        durable_io.delete_file(
            target,
            durable_root=tmp_path,
            expected_sha256=expected,
        )

    assert target.read_bytes() == b"outside-sentinel"
    assert outside.read_bytes() == b"outside-sentinel"
