//go:build !aix && !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd && !solaris && !windows

package gcpauth

import (
	"fmt"
	"os"
)

func validateCredentialFileAccess(_ *os.File, _ os.FileInfo) error {
	return fmt.Errorf("credential file access validation is unsupported on this operating system")
}
