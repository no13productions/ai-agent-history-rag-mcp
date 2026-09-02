//go:build windows

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

func TestOwnerOnlyStateDirectoryModeDoesNotUseSynthesizedWindowsBits(t *testing.T) {
	for _, mode := range []os.FileMode{os.ModeDir | 0o777, os.ModeDir | 0o555} {
		if err := requireOwnerOnlyStateDirectoryMode(syntheticFileInfo{mode: mode}); err != nil {
			t.Fatalf("synthesized mode %v rejected: %v", mode, err)
		}
	}
}
