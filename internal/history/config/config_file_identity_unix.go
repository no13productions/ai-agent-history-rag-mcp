//go:build darwin || linux

package config

import (
	"errors"
	"os"

	"golang.org/x/sys/unix"
)

func requireOwnerOnlyConfigMode(info os.FileInfo) error {
	if info.Mode().Perm() != 0o600 {
		return errors.New("config mode must be owner-only")
	}
	return nil
}

func requireOwnerOnlyStateDirectoryMode(info os.FileInfo) error {
	if info.Mode().Perm() != 0o700 {
		return errors.New("directory mode must be owner-only")
	}
	return nil
}

func requireSingleLink(path string) error {
	var identity unix.Stat_t
	if err := unix.Lstat(path, &identity); err != nil {
		return err
	}
	if identity.Nlink != 1 {
		return unix.EMLINK
	}
	return nil
}
