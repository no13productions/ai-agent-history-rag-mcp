//go:build windows

package watch

import (
	"errors"
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

type platformRoot struct {
	file         *os.File
	info         os.FileInfo
	rootIdentity RootIdentity
}

func (r *platformRoot) isBound() bool { return r.file != nil }

func (r *platformRoot) identity() RootIdentity { return r.rootIdentity }

func windowsLstat(path string, directory bool) (os.FileInfo, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	pointer, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}
	attributes, err := windows.GetFileAttributes(pointer)
	if err != nil {
		return nil, err
	}
	if info.Mode()&os.ModeSymlink != 0 || attributes&windows.FILE_ATTRIBUTE_REPARSE_POINT != 0 || info.IsDir() != directory {
		return nil, ErrUnsafeRoot
	}
	return info, nil
}

func (r *platformRoot) bind(path string) (bool, error) {
	if r.file != nil {
		if err := r.verify(path); err != nil {
			return false, err
		}
		return true, nil
	}
	before, err := windowsLstat(path, true)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	pointer, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return false, err
	}
	handle, err := windows.CreateFile(pointer, windows.FILE_GENERIC_READ, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE, nil, windows.OPEN_EXISTING, windows.FILE_FLAG_BACKUP_SEMANTICS|windows.FILE_FLAG_OPEN_REPARSE_POINT, 0)
	if err != nil {
		return false, fmt.Errorf("hold watch root: %w", err)
	}
	held := os.NewFile(uintptr(handle), path)
	if held == nil {
		_ = windows.CloseHandle(handle)
		return false, fmt.Errorf("hold watch root")
	}
	opened, err := held.Stat()
	if err != nil || !os.SameFile(before, opened) {
		_ = held.Close()
		return false, ErrRootChanged
	}
	var handleInfo windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(handle, &handleInfo); err != nil || handleInfo.FileAttributes&windows.FILE_ATTRIBUTE_REPARSE_POINT != 0 {
		_ = held.Close()
		return false, ErrUnsafeRoot
	}
	r.file = held
	r.info = opened
	r.rootIdentity = RootIdentity{
		Volume: uint64(handleInfo.VolumeSerialNumber),
		Object: uint64(handleInfo.FileIndexHigh)<<32 | uint64(handleInfo.FileIndexLow),
	}
	return true, nil
}

func (r *platformRoot) verify(path string) error {
	if r.file == nil {
		return ErrRootUnbound
	}
	held, err := r.file.Stat()
	if err != nil {
		return ErrRootChanged
	}
	named, err := windowsLstat(path, true)
	if err != nil || !os.SameFile(r.info, held) || !os.SameFile(held, named) {
		return ErrRootChanged
	}
	return nil
}

func openWindowsRelative(parent windows.Handle, name string, directory bool) (*os.File, windows.ByHandleFileInformation, error) {
	objectName, err := windows.NewNTUnicodeString(name)
	if err != nil {
		return nil, windows.ByHandleFileInformation{}, err
	}
	attributes := &windows.OBJECT_ATTRIBUTES{
		Length:        uint32(unsafe.Sizeof(windows.OBJECT_ATTRIBUTES{})),
		RootDirectory: parent,
		ObjectName:    objectName,
		Attributes:    windows.OBJ_CASE_INSENSITIVE | windows.OBJ_DONT_REPARSE,
	}
	options := uint32(windows.FILE_SYNCHRONOUS_IO_NONALERT | windows.FILE_OPEN_REPARSE_POINT)
	if directory {
		options |= windows.FILE_DIRECTORY_FILE
	} else {
		options |= windows.FILE_NON_DIRECTORY_FILE
	}
	var handle windows.Handle
	var status windows.IO_STATUS_BLOCK
	err = windows.NtCreateFile(&handle, windows.FILE_GENERIC_READ, attributes, &status, nil, 0, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE, windows.FILE_OPEN, options, 0, 0)
	if err != nil {
		return nil, windows.ByHandleFileInformation{}, err
	}
	file := os.NewFile(uintptr(handle), name)
	if file == nil {
		_ = windows.CloseHandle(handle)
		return nil, windows.ByHandleFileInformation{}, ErrUnsafeSource
	}
	var information windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(handle, &information); err != nil {
		_ = file.Close()
		return nil, information, err
	}
	if information.FileAttributes&windows.FILE_ATTRIBUTE_REPARSE_POINT != 0 || directory != (information.FileAttributes&windows.FILE_ATTRIBUTE_DIRECTORY != 0) || (!directory && information.NumberOfLinks != 1) {
		_ = file.Close()
		return nil, information, ErrUnsafeSource
	}
	return file, information, nil
}

func sameWindowsIdentity(left, right windows.ByHandleFileInformation) bool {
	return left.VolumeSerialNumber == right.VolumeSerialNumber && left.FileIndexHigh == right.FileIndexHigh && left.FileIndexLow == right.FileIndexLow
}

func reopenWindowsRelative(parent windows.Handle, name string, directory bool, expected windows.ByHandleFileInformation) error {
	current, information, err := openWindowsRelative(parent, name, directory)
	if err != nil {
		return ErrUnsafeSource
	}
	defer current.Close()
	if !sameWindowsIdentity(expected, information) {
		return ErrUnsafeSource
	}
	return nil
}

func (r *platformRoot) snapshot(rootPath, sourcePath string, parts []string, maxBytes int64) (*Snapshot, error) {
	if err := r.verify(rootPath); err != nil {
		return nil, err
	}
	parent := windows.Handle(r.file.Fd())
	var openedDirectories []*os.File
	defer func() {
		for index := len(openedDirectories) - 1; index >= 0; index-- {
			_ = openedDirectories[index].Close()
		}
	}()
	for _, component := range parts[:len(parts)-1] {
		opened, identity, err := openWindowsRelative(parent, component, true)
		if err != nil {
			return nil, ErrUnsafeSource
		}
		if err := reopenWindowsRelative(parent, component, true, identity); err != nil {
			_ = opened.Close()
			return nil, err
		}
		openedDirectories = append(openedDirectories, opened)
		parent = windows.Handle(opened.Fd())
	}

	name := parts[len(parts)-1]
	source, identity, err := openWindowsRelative(parent, name, false)
	if err != nil {
		return nil, ErrUnsafeSource
	}
	defer source.Close()
	if err := reopenWindowsRelative(parent, name, false, identity); err != nil {
		return nil, err
	}
	before, err := source.Stat()
	if err != nil {
		return nil, ErrUnsafeSource
	}
	snapshot, err := newPrivateSnapshot(sourcePath, source, before.Size(), before.ModTime(), maxBytes)
	if err != nil {
		return nil, err
	}
	refuse := func(failure error) (*Snapshot, error) {
		_ = snapshot.Close()
		return nil, failure
	}
	after, err := source.Stat()
	if err != nil || before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) {
		return refuse(ErrSourceChanged)
	}
	if err := reopenWindowsRelative(parent, name, false, identity); err != nil {
		return refuse(ErrSourceChanged)
	}
	if err := r.verify(rootPath); err != nil {
		return refuse(err)
	}
	return snapshot, nil
}

func (r *platformRoot) close() error {
	if r.file == nil {
		return nil
	}
	err := r.file.Close()
	r.file = nil
	return err
}
