//go:build darwin || linux

package watch

import (
	"errors"
	"fmt"
	"os"

	"golang.org/x/sys/unix"
)

type platformRoot struct {
	file *os.File
	stat unix.Stat_t
}

func (r *platformRoot) isBound() bool { return r.file != nil }

func (r *platformRoot) identity() RootIdentity {
	return RootIdentity{Volume: uint64(r.stat.Dev), Object: uint64(r.stat.Ino)}
}

func (r *platformRoot) bind(path string) (bool, error) {
	if r.file != nil {
		if _, err := r.verify(path); err != nil {
			return false, err
		}
		return true, nil
	}
	var before unix.Stat_t
	if err := unix.Lstat(path, &before); errors.Is(err, unix.ENOENT) {
		return false, nil
	} else if err != nil {
		return false, fmt.Errorf("inspect watch root: %w", err)
	}
	if before.Mode&unix.S_IFMT != unix.S_IFDIR {
		return false, ErrUnsafeRoot
	}
	descriptor, err := unix.Open(path, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return false, fmt.Errorf("hold watch root: %w", err)
	}
	held := os.NewFile(uintptr(descriptor), path)
	if held == nil {
		_ = unix.Close(descriptor)
		return false, fmt.Errorf("hold watch root")
	}
	var opened unix.Stat_t
	if err := unix.Fstat(descriptor, &opened); err != nil {
		_ = held.Close()
		return false, fmt.Errorf("inspect held watch root: %w", err)
	}
	if !sameUnixIdentity(before, opened) || opened.Mode&unix.S_IFMT != unix.S_IFDIR {
		_ = held.Close()
		return false, ErrRootChanged
	}
	r.file = held
	r.stat = opened
	return true, nil
}

func (r *platformRoot) verify(path string) (int, error) {
	if r.file == nil {
		return -1, ErrRootUnbound
	}
	descriptor := int(r.file.Fd())
	var held unix.Stat_t
	if err := unix.Fstat(descriptor, &held); err != nil {
		return -1, ErrRootChanged
	}
	var named unix.Stat_t
	if err := unix.Lstat(path, &named); err != nil {
		return -1, ErrRootChanged
	}
	if !sameUnixIdentity(r.stat, held) || !sameUnixIdentity(held, named) || named.Mode&unix.S_IFMT != unix.S_IFDIR {
		return -1, ErrRootChanged
	}
	return descriptor, nil
}

func sameUnixIdentity(left, right unix.Stat_t) bool {
	return left.Dev == right.Dev && left.Ino == right.Ino
}

func validateUnixDirectory(value unix.Stat_t) error {
	if value.Mode&unix.S_IFMT != unix.S_IFDIR {
		return ErrUnsafeSource
	}
	return nil
}

func validateUnixFile(value unix.Stat_t) error {
	if value.Mode&unix.S_IFMT != unix.S_IFREG || value.Nlink != 1 {
		return ErrUnsafeSource
	}
	return nil
}

func unixStatAt(parent int, name string) (unix.Stat_t, error) {
	var value unix.Stat_t
	err := unix.Fstatat(parent, name, &value, unix.AT_SYMLINK_NOFOLLOW)
	return value, err
}

func (r *platformRoot) snapshot(rootPath, sourcePath string, parts []string, maxBytes int64) (*Snapshot, error) {
	rootDescriptor, err := r.verify(rootPath)
	if err != nil {
		return nil, err
	}
	parent := rootDescriptor
	var openedDirectories []*os.File
	defer func() {
		for index := len(openedDirectories) - 1; index >= 0; index-- {
			_ = openedDirectories[index].Close()
		}
	}()
	for _, component := range parts[:len(parts)-1] {
		before, statErr := unixStatAt(parent, component)
		if statErr != nil || validateUnixDirectory(before) != nil {
			return nil, ErrUnsafeSource
		}
		descriptor, openErr := unix.Openat(parent, component, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
		if openErr != nil {
			return nil, ErrUnsafeSource
		}
		opened := os.NewFile(uintptr(descriptor), component)
		if opened == nil {
			_ = unix.Close(descriptor)
			return nil, ErrUnsafeSource
		}
		var held unix.Stat_t
		current, currentErr := unixStatAt(parent, component)
		if unix.Fstat(descriptor, &held) != nil || currentErr != nil || validateUnixDirectory(held) != nil || !sameUnixIdentity(before, held) || !sameUnixIdentity(held, current) {
			_ = opened.Close()
			return nil, ErrUnsafeSource
		}
		openedDirectories = append(openedDirectories, opened)
		parent = descriptor
	}

	name := parts[len(parts)-1]
	before, err := unixStatAt(parent, name)
	if err != nil || validateUnixFile(before) != nil {
		return nil, ErrUnsafeSource
	}
	descriptor, err := unix.Openat(parent, name, unix.O_RDONLY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, ErrUnsafeSource
	}
	source := os.NewFile(uintptr(descriptor), name)
	if source == nil {
		_ = unix.Close(descriptor)
		return nil, ErrUnsafeSource
	}
	defer source.Close()
	var held unix.Stat_t
	current, currentErr := unixStatAt(parent, name)
	if unix.Fstat(descriptor, &held) != nil || currentErr != nil || validateUnixFile(held) != nil || !sameUnixIdentity(before, held) || !sameUnixIdentity(held, current) {
		return nil, ErrUnsafeSource
	}
	beforeInfo, err := source.Stat()
	if err != nil {
		return nil, ErrUnsafeSource
	}
	snapshot, err := newPrivateSnapshot(sourcePath, source, beforeInfo.Size(), beforeInfo.ModTime(), maxBytes)
	if err != nil {
		return nil, err
	}
	refuse := func(failure error) (*Snapshot, error) {
		_ = snapshot.Close()
		return nil, failure
	}
	afterInfo, err := source.Stat()
	final, finalErr := unixStatAt(parent, name)
	if err != nil || finalErr != nil || validateUnixFile(final) != nil || !sameUnixIdentity(held, final) || beforeInfo.Size() != afterInfo.Size() || !beforeInfo.ModTime().Equal(afterInfo.ModTime()) {
		return refuse(ErrSourceChanged)
	}
	if _, err := r.verify(rootPath); err != nil {
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
