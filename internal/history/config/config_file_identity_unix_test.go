//go:build !windows

package config

import (
	"os"
	"testing"
	"time"
)

type syntheticFileInfo struct{ mode os.FileMode }

func (s syntheticFileInfo) Name() string       { return "state" }
func (s syntheticFileInfo) Size() int64        { return 0 }
func (s syntheticFileInfo) Mode() os.FileMode  { return s.mode }
func (s syntheticFileInfo) ModTime() time.Time { return time.Time{} }
func (s syntheticFileInfo) IsDir() bool        { return true }
func (s syntheticFileInfo) Sys() any           { return nil }

func TestOwnerOnlyStateDirectoryModeIsExactOnUnix(t *testing.T) {
	if err := requireOwnerOnlyStateDirectoryMode(syntheticFileInfo{mode: os.ModeDir | 0o700}); err != nil {
		t.Fatalf("0700 rejected: %v", err)
	}
	if err := requireOwnerOnlyStateDirectoryMode(syntheticFileInfo{mode: os.ModeDir | 0o755}); err == nil {
		t.Fatal("0755 accepted")
	}
}
