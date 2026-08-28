//go:build windows

package config

import (
	"os"

	"golang.org/x/sys/windows"
)

func requireOwnerOnlyConfigMode(os.FileInfo) error { return nil }

// Windows synthesizes os.FileMode permission bits from attributes rather than
// the file's ACL. Comparing those bits to POSIX 0700 rejects ordinary private
// directories without proving ownership, so the Windows path retains its
// handle identity and single-link checks and does not interpret them as ACLs.
func requireOwnerOnlyStateDirectoryMode(os.FileInfo) error { return nil }

func requireSingleLink(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	var information windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(windows.Handle(file.Fd()), &information); err != nil {
		return err
	}
	if information.NumberOfLinks != 1 {
		return windows.ERROR_TOO_MANY_LINKS
	}
	return nil
}
