//go:build darwin || linux

package durable

import (
	"crypto/rand"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	"golang.org/x/sys/unix"
)

var (
	ErrUnsafePath      = errors.New("unsafe durable path")
	ErrRootChanged     = errors.New("durable root identity changed")
	ErrNotOwnerOnly    = errors.New("durable object is not owner-only")
	ErrReadLimit       = errors.New("durable read exceeds limit")
	ErrIdentityChanged = errors.New("durable object identity changed")
)

type Root struct {
	path string
	file *os.File
	stat unix.Stat_t
	mu   sync.Mutex
}

func OpenRoot(path string) (*Root, error) {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, ErrUnsafePath
	}
	var before unix.Stat_t
	if err := unix.Lstat(path, &before); err != nil {
		return nil, fmt.Errorf("open durable root: %w", err)
	}
	if before.Mode&unix.S_IFMT != unix.S_IFDIR {
		return nil, ErrUnsafePath
	}
	if before.Mode&0o777 != 0o700 {
		return nil, ErrNotOwnerOnly
	}
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("hold durable root: %w", err)
	}
	file := os.NewFile(uintptr(fd), path)
	if file == nil {
		_ = unix.Close(fd)
		return nil, errors.New("hold durable root")
	}
	var opened unix.Stat_t
	if err := unix.Fstat(fd, &opened); err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("inspect durable root: %w", err)
	}
	if !sameIdentity(before, opened) {
		_ = file.Close()
		return nil, ErrIdentityChanged
	}
	return &Root{path: path, file: file, stat: opened}, nil
}

func (r *Root) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.file == nil {
		return nil
	}
	err := r.file.Close()
	r.file = nil
	return err
}

func (r *Root) descriptor() (int, error) {
	if r.file == nil {
		return -1, os.ErrClosed
	}
	return int(r.file.Fd()), nil
}

func (r *Root) verify() (int, error) {
	fd, err := r.descriptor()
	if err != nil {
		return -1, err
	}
	var held unix.Stat_t
	if err := unix.Fstat(fd, &held); err != nil {
		return -1, err
	}
	var named unix.Stat_t
	if err := unix.Lstat(r.path, &named); err != nil {
		return -1, ErrRootChanged
	}
	if !sameIdentity(r.stat, held) || !sameIdentity(held, named) || named.Mode&unix.S_IFMT != unix.S_IFDIR || named.Mode&0o777 != 0o700 {
		return -1, ErrRootChanged
	}
	return fd, nil
}

func validateName(name string) error {
	if name == "" || name == "." || name == ".." || filepath.Base(name) != name || filepath.Clean(name) != name {
		return ErrUnsafePath
	}
	return nil
}

func statAt(fd int, name string) (unix.Stat_t, error) {
	var value unix.Stat_t
	err := unix.Fstatat(fd, name, &value, unix.AT_SYMLINK_NOFOLLOW)
	return value, err
}

func validateFile(value unix.Stat_t) error {
	if value.Mode&unix.S_IFMT != unix.S_IFREG || value.Nlink != 1 {
		return ErrUnsafePath
	}
	if value.Mode&0o777 != 0o600 {
		return ErrNotOwnerOnly
	}
	return nil
}

func sameIdentity(a, b unix.Stat_t) bool {
	return a.Dev == b.Dev && a.Ino == b.Ino
}

func (r *Root) Exists(name string) (bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := validateName(name); err != nil {
		return false, err
	}
	fd, err := r.verify()
	if err != nil {
		return false, err
	}
	value, err := statAt(fd, name)
	if errors.Is(err, unix.ENOENT) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if err := validateFile(value); err != nil {
		return false, err
	}
	return true, nil
}

func (r *Root) Read(name string, maxBytes int64) ([]byte, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.readLocked(name, maxBytes)
}

func (r *Root) readLocked(name string, maxBytes int64) ([]byte, error) {
	if err := validateName(name); err != nil || maxBytes < 0 {
		return nil, ErrUnsafePath
	}
	fd, err := r.verify()
	if err != nil {
		return nil, err
	}
	before, err := statAt(fd, name)
	if err != nil {
		return nil, err
	}
	if err := validateFile(before); err != nil {
		return nil, err
	}
	openedFD, err := unix.Openat(fd, name, unix.O_RDONLY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	opened := os.NewFile(uintptr(openedFD), name)
	if opened == nil {
		_ = unix.Close(openedFD)
		return nil, errors.New("open durable file")
	}
	defer opened.Close()
	var held unix.Stat_t
	if err := unix.Fstat(openedFD, &held); err != nil {
		return nil, err
	}
	current, err := statAt(fd, name)
	if err != nil {
		return nil, err
	}
	if err := validateFile(held); err != nil {
		return nil, err
	}
	if err := validateFile(current); err != nil {
		return nil, err
	}
	if !sameIdentity(before, held) || !sameIdentity(held, current) {
		return nil, ErrIdentityChanged
	}
	payload, err := io.ReadAll(io.LimitReader(opened, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(payload)) > maxBytes {
		return nil, ErrReadLimit
	}
	final, err := statAt(fd, name)
	if err != nil || !sameIdentity(held, final) {
		return nil, ErrIdentityChanged
	}
	return payload, nil
}

func randomName(target, suffix string) (string, error) {
	var token [16]byte
	if _, err := io.ReadFull(rand.Reader, token[:]); err != nil {
		return "", err
	}
	return fmt.Sprintf(".%s.%x.%s", target, token, suffix), nil
}

func (r *Root) WriteAtomic(name string, payload []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := validateName(name); err != nil {
		return err
	}
	fd, err := r.verify()
	if err != nil {
		return err
	}
	if existing, statErr := statAt(fd, name); statErr == nil {
		if err := validateFile(existing); err != nil {
			return err
		}
	} else if !errors.Is(statErr, unix.ENOENT) {
		return statErr
	}
	var temp string
	var tempFD int
	for attempt := 0; attempt < 128; attempt++ {
		temp, err = randomName(name, "tmp")
		if err != nil {
			return err
		}
		tempFD, err = unix.Openat(fd, temp, unix.O_WRONLY|unix.O_CREAT|unix.O_EXCL|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0o600)
		if errors.Is(err, unix.EEXIST) {
			continue
		}
		break
	}
	if err != nil {
		return err
	}
	created := os.NewFile(uintptr(tempFD), temp)
	if created == nil {
		_ = unix.Close(tempFD)
		return errors.New("create durable temporary")
	}
	committed := false
	defer func() {
		_ = created.Close()
		if !committed {
			_ = unix.Unlinkat(fd, temp, 0)
		}
	}()
	if err := unix.Fchmod(tempFD, 0o600); err != nil {
		return err
	}
	if _, err := created.Write(payload); err != nil {
		return err
	}
	if err := created.Sync(); err != nil {
		return err
	}
	var staged unix.Stat_t
	if err := unix.Fstat(tempFD, &staged); err != nil {
		return err
	}
	if err := validateFile(staged); err != nil {
		return err
	}
	if err := unix.Renameat(fd, temp, fd, name); err != nil {
		return err
	}
	committed = true
	if err := unix.Fsync(fd); err != nil {
		return fmt.Errorf("durable replacement committed without confirmation: %w", err)
	}
	final, err := statAt(fd, name)
	if err != nil || !sameIdentity(staged, final) {
		return ErrIdentityChanged
	}
	return validateFile(final)
}

func (r *Root) Remove(name string, expectedSHA256 [32]byte) (bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := validateName(name); err != nil {
		return false, err
	}
	fd, err := r.verify()
	if err != nil {
		return false, err
	}
	value, err := statAt(fd, name)
	if errors.Is(err, unix.ENOENT) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if err := validateFile(value); err != nil {
		return false, err
	}
	payload, err := r.readLockedWithoutMutex(fd, name, 16<<20)
	if err != nil {
		return false, err
	}
	if sha256.Sum256(payload) != expectedSHA256 {
		return false, ErrIdentityChanged
	}
	tombstone, err := randomName(name, "deleted")
	if err != nil {
		return false, err
	}
	if err := unix.Renameat(fd, name, fd, tombstone); err != nil {
		return false, err
	}
	moved, err := statAt(fd, tombstone)
	if err != nil || !sameIdentity(value, moved) {
		return false, ErrIdentityChanged
	}
	if err := unix.Fsync(fd); err != nil {
		return false, fmt.Errorf("durable delete committed without confirmation: %w", err)
	}
	if err := unix.Unlinkat(fd, tombstone, 0); err != nil {
		return false, fmt.Errorf("durable delete cleanup uncertain: %w", err)
	}
	if err := unix.Fsync(fd); err != nil {
		return false, fmt.Errorf("durable delete cleanup unconfirmed: %w", err)
	}
	return true, nil
}

func (r *Root) readLockedWithoutMutex(fd int, name string, maxBytes int64) ([]byte, error) {
	before, err := statAt(fd, name)
	if err != nil {
		return nil, err
	}
	if err := validateFile(before); err != nil {
		return nil, err
	}
	openedFD, err := unix.Openat(fd, name, unix.O_RDONLY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	opened := os.NewFile(uintptr(openedFD), name)
	if opened == nil {
		_ = unix.Close(openedFD)
		return nil, errors.New("open durable file")
	}
	defer opened.Close()
	var held unix.Stat_t
	if err := unix.Fstat(openedFD, &held); err != nil || !sameIdentity(before, held) {
		return nil, ErrIdentityChanged
	}
	payload, err := io.ReadAll(io.LimitReader(opened, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(payload)) > maxBytes {
		return nil, ErrReadLimit
	}
	final, err := statAt(fd, name)
	if err != nil || !sameIdentity(held, final) {
		return nil, ErrIdentityChanged
	}
	return payload, nil
}
