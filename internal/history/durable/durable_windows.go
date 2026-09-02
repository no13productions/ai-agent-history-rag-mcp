//go:build windows

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

	"golang.org/x/sys/windows"
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
	info os.FileInfo
	mu   sync.Mutex
}

func OpenRoot(path string) (*Root, error) {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, ErrUnsafePath
	}
	before, err := safeLstat(path, true)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	opened, err := file.Stat()
	if err != nil || !os.SameFile(before, opened) {
		_ = file.Close()
		return nil, ErrIdentityChanged
	}
	return &Root{path: path, file: file, info: opened}, nil
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

func validateName(name string) error {
	if name == "" || name == "." || name == ".." || filepath.Base(name) != name || filepath.Clean(name) != name {
		return ErrUnsafePath
	}
	return nil
}

func safeLstat(path string, directory bool) (os.FileInfo, error) {
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
	if info.Mode()&os.ModeSymlink != 0 || attributes&windows.FILE_ATTRIBUTE_REPARSE_POINT != 0 || directory != info.IsDir() {
		return nil, ErrUnsafePath
	}
	return info, nil
}

func (r *Root) verify() error {
	if r.file == nil {
		return os.ErrClosed
	}
	held, err := r.file.Stat()
	if err != nil {
		return err
	}
	named, err := safeLstat(r.path, true)
	if err != nil || !os.SameFile(r.info, held) || !os.SameFile(held, named) {
		return ErrRootChanged
	}
	return nil
}

func validateOpenedFile(path string, file *os.File, before os.FileInfo) (os.FileInfo, error) {
	opened, err := file.Stat()
	if err != nil || !opened.Mode().IsRegular() || !os.SameFile(before, opened) {
		return nil, ErrIdentityChanged
	}
	var handleInfo windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(windows.Handle(file.Fd()), &handleInfo); err != nil {
		return nil, err
	}
	if handleInfo.NumberOfLinks != 1 || handleInfo.FileAttributes&windows.FILE_ATTRIBUTE_REPARSE_POINT != 0 {
		return nil, ErrUnsafePath
	}
	current, err := safeLstat(path, false)
	if err != nil || !os.SameFile(opened, current) {
		return nil, ErrIdentityChanged
	}
	return opened, nil
}

func (r *Root) Exists(name string) (bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := validateName(name); err != nil {
		return false, err
	}
	if err := r.verify(); err != nil {
		return false, err
	}
	path := filepath.Join(r.path, name)
	info, err := safeLstat(path, false)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	file, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer file.Close()
	_, err = validateOpenedFile(path, file, info)
	return err == nil, err
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
	if err := r.verify(); err != nil {
		return nil, err
	}
	path := filepath.Join(r.path, name)
	before, err := safeLstat(path, false)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	opened, err := validateOpenedFile(path, file, before)
	if err != nil {
		return nil, err
	}
	payload, err := io.ReadAll(io.LimitReader(file, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(payload)) > maxBytes {
		return nil, ErrReadLimit
	}
	after, err := safeLstat(path, false)
	if err != nil || !os.SameFile(opened, after) {
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

func moveFile(source, target string, replace bool) error {
	sourcePointer, err := windows.UTF16PtrFromString(source)
	if err != nil {
		return err
	}
	targetPointer, err := windows.UTF16PtrFromString(target)
	if err != nil {
		return err
	}
	flags := uint32(windows.MOVEFILE_WRITE_THROUGH)
	if replace {
		flags |= windows.MOVEFILE_REPLACE_EXISTING
	}
	return windows.MoveFileEx(sourcePointer, targetPointer, flags)
}

func (r *Root) WriteAtomic(name string, payload []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := validateName(name); err != nil {
		return err
	}
	if err := r.verify(); err != nil {
		return err
	}
	target := filepath.Join(r.path, name)
	if info, err := safeLstat(target, false); err == nil {
		file, openErr := os.Open(target)
		if openErr != nil {
			return openErr
		}
		_, validateErr := validateOpenedFile(target, file, info)
		_ = file.Close()
		if validateErr != nil {
			return validateErr
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	tempName, err := randomName(name, "tmp")
	if err != nil {
		return err
	}
	temp := filepath.Join(r.path, tempName)
	file, err := os.OpenFile(temp, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	committed := false
	defer func() {
		_ = file.Close()
		if !committed {
			_ = os.Remove(temp)
		}
	}()
	if _, err := file.Write(payload); err != nil {
		return err
	}
	if err := file.Sync(); err != nil {
		return err
	}
	before, err := file.Stat()
	if err != nil {
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	if err := moveFile(temp, target, true); err != nil {
		return err
	}
	committed = true
	after, err := safeLstat(target, false)
	if err != nil || !os.SameFile(before, after) {
		return ErrIdentityChanged
	}
	return nil
}

func (r *Root) Remove(name string, expectedSHA256 [32]byte) (bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := validateName(name); err != nil {
		return false, err
	}
	if err := r.verify(); err != nil {
		return false, err
	}
	path := filepath.Join(r.path, name)
	before, err := safeLstat(path, false)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	file, err := os.Open(path)
	if err != nil {
		return false, err
	}
	opened, err := validateOpenedFile(path, file, before)
	if err != nil {
		_ = file.Close()
		return false, err
	}
	payload, err := io.ReadAll(io.LimitReader(file, (16<<20)+1))
	_ = file.Close()
	if err != nil || len(payload) > 16<<20 || sha256.Sum256(payload) != expectedSHA256 {
		return false, ErrIdentityChanged
	}
	tombstoneName, err := randomName(name, "deleted")
	if err != nil {
		return false, err
	}
	tombstone := filepath.Join(r.path, tombstoneName)
	if err := moveFile(path, tombstone, false); err != nil {
		return false, err
	}
	moved, err := safeLstat(tombstone, false)
	if err != nil || !os.SameFile(opened, moved) {
		return false, ErrIdentityChanged
	}
	if err := os.Remove(tombstone); err != nil {
		return false, fmt.Errorf("durable delete cleanup uncertain: %w", err)
	}
	return true, nil
}
