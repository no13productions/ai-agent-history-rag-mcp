package watch

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func newBoundRoot(t *testing.T) (*PinnedRoot, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "history")
	if err := os.Mkdir(path, 0o755); err != nil {
		t.Fatal(err)
	}
	root, err := NewPinnedRoot(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = root.Close() })
	bound, err := root.Bind()
	if err != nil || !bound {
		t.Fatalf("Bind() = %v, %v", bound, err)
	}
	return root, path
}

func TestPinnedRootBindsOnlyOnceAndRefusesReplacement(t *testing.T) {
	base := t.TempDir()
	path := filepath.Join(base, "history")
	root, err := NewPinnedRoot(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = root.Close() })

	if bound, err := root.Bind(); err != nil || bound {
		t.Fatalf("missing Bind() = %v, %v", bound, err)
	}
	if err := os.Mkdir(path, 0o755); err != nil {
		t.Fatal(err)
	}
	if bound, err := root.Bind(); err != nil || !bound {
		t.Fatalf("first Bind() = %v, %v", bound, err)
	}
	if !root.IsBound() {
		t.Fatal("root did not report bound")
	}
	identity, ok := root.Identity()
	if !ok || identity == (RootIdentity{}) {
		t.Fatalf("Identity() = %#v, %v", identity, ok)
	}
	if bound, err := root.Bind(); err != nil || !bound {
		t.Fatalf("repeat Bind() = %v, %v", bound, err)
	}
	if repeated, ok := root.Identity(); !ok || repeated != identity {
		t.Fatalf("repeated Identity() = %#v, %v; want %#v", repeated, ok, identity)
	}

	moved := path + "-moved"
	if err := os.Rename(path, moved); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(path, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := root.Bind(); !errors.Is(err, ErrRootChanged) {
		t.Fatalf("replacement Bind() error = %v", err)
	}
}

func TestPinnedRootAndSnapshotRequireClosedAbsoluteCoordinates(t *testing.T) {
	for _, path := range []string{"relative", t.TempDir() + string(filepath.Separator) + ".." + string(filepath.Separator) + "unclean"} {
		if _, err := NewPinnedRoot(path); !errors.Is(err, ErrUnsafeRoot) {
			t.Fatalf("NewPinnedRoot(%q) error = %v", path, err)
		}
	}
	base := filepath.Join(t.TempDir(), "history")
	root, err := NewPinnedRoot(base)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := root.Snapshot(filepath.Join(base, "session.jsonl"), 64); !errors.Is(err, ErrRootUnbound) {
		t.Fatalf("unbound Snapshot() error = %v", err)
	}
}

func TestPinnedRootRefusesNonDirectoryAndSymlink(t *testing.T) {
	base := t.TempDir()
	nonDirectory := filepath.Join(base, "file")
	if err := os.WriteFile(nonDirectory, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	root, err := NewPinnedRoot(nonDirectory)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := root.Bind(); !errors.Is(err, ErrUnsafeRoot) {
		t.Fatalf("non-directory Bind() error = %v", err)
	}

	target := filepath.Join(base, "target")
	if err := os.Mkdir(target, 0o755); err != nil {
		t.Fatal(err)
	}
	linked := filepath.Join(base, "linked")
	if err := os.Symlink(target, linked); err != nil {
		t.Skipf("directory symlink unavailable: %v", err)
	}
	root, err = NewPinnedRoot(linked)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := root.Bind(); !errors.Is(err, ErrUnsafeRoot) {
		t.Fatalf("symlink Bind() error = %v", err)
	}
}

func TestSnapshotIsBoundedPrivateAndPreservesProvenance(t *testing.T) {
	root, path := newBoundRoot(t)
	nested := filepath.Join(path, "project")
	if err := os.Mkdir(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(nested, "session.jsonl")
	payload := []byte("first\nsecond\n")
	if err := os.WriteFile(source, payload, 0o644); err != nil {
		t.Fatal(err)
	}
	wantModTime := time.Unix(1_700_000_000, 123_000_000)
	if err := os.Chtimes(source, wantModTime, wantModTime); err != nil {
		t.Fatal(err)
	}

	snapshot, err := root.Snapshot(source, MaxSourceSnapshotBytes)
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	if snapshot.SourcePath() != source {
		t.Fatalf("SourcePath() = %q", snapshot.SourcePath())
	}
	if snapshot.Size() != int64(len(payload)) {
		t.Fatalf("Size() = %d", snapshot.Size())
	}
	if snapshot.SHA256() != sha256.Sum256(payload) {
		t.Fatalf("SHA256() = %x", snapshot.SHA256())
	}
	if !snapshot.ModTime().Equal(wantModTime) {
		t.Fatalf("ModTime() = %v, want %v", snapshot.ModTime(), wantModTime)
	}

	if err := os.WriteFile(source, []byte("replacement\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := io.ReadAll(snapshot.Reader())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("snapshot bytes = %q", got)
	}

	if err := snapshot.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := io.ReadAll(snapshot.Reader()); !errors.Is(err, ErrSnapshotClosed) {
		t.Fatalf("read after Close() error = %v", err)
	}
}

func TestSnapshotRefusesUnsafeDescendantsAndLimits(t *testing.T) {
	root, path := newBoundRoot(t)
	outside := filepath.Join(t.TempDir(), "outside.jsonl")
	if err := os.WriteFile(outside, []byte("outside"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := root.Snapshot(outside, 64); !errors.Is(err, ErrOutsideRoot) {
		t.Fatalf("outside Snapshot() error = %v", err)
	}
	if _, err := root.Snapshot(filepath.Join(path, "missing"), -1); !errors.Is(err, ErrInvalidLimit) {
		t.Fatalf("negative limit error = %v", err)
	}
	if _, err := root.Snapshot(filepath.Join(path, "missing"), MaxSourceSnapshotBytes+1); !errors.Is(err, ErrInvalidLimit) {
		t.Fatalf("oversized limit error = %v", err)
	}

	oversize := filepath.Join(path, "oversize")
	if err := os.WriteFile(oversize, bytes.Repeat([]byte("x"), 65), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := root.Snapshot(oversize, 64); !errors.Is(err, ErrSnapshotLimit) {
		t.Fatalf("oversize Snapshot() error = %v", err)
	}

	symlink := filepath.Join(path, "symlink")
	if err := os.Symlink(outside, symlink); err == nil {
		if _, err := root.Snapshot(symlink, 64); !errors.Is(err, ErrUnsafeSource) {
			t.Fatalf("symlink Snapshot() error = %v", err)
		}
	}

	hardlink := filepath.Join(path, "hardlink")
	if err := os.Link(outside, hardlink); err == nil {
		if _, err := root.Snapshot(hardlink, 64); !errors.Is(err, ErrUnsafeSource) {
			t.Fatalf("hardlink Snapshot() error = %v", err)
		}
	}

	directory := filepath.Join(path, "directory")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := root.Snapshot(directory, 64); !errors.Is(err, ErrUnsafeSource) {
		t.Fatalf("directory Snapshot() error = %v", err)
	}
}

func TestSnapshotRefusesNestedLinkAndReplacedRoot(t *testing.T) {
	root, path := newBoundRoot(t)
	outsideDir := filepath.Join(t.TempDir(), "outside")
	if err := os.Mkdir(outsideDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outsideDir, "session.jsonl"), []byte("outside"), 0o600); err != nil {
		t.Fatal(err)
	}
	linkedDir := filepath.Join(path, "linked")
	if err := os.Symlink(outsideDir, linkedDir); err == nil {
		if _, err := root.Snapshot(filepath.Join(linkedDir, "session.jsonl"), 64); !errors.Is(err, ErrUnsafeSource) {
			t.Fatalf("nested symlink Snapshot() error = %v", err)
		}
	}

	moved := path + "-moved"
	if err := os.Rename(path, moved); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(path, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, "session.jsonl"), []byte("substitute"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := root.Snapshot(filepath.Join(path, "session.jsonl"), 64); !errors.Is(err, ErrRootChanged) {
		t.Fatalf("replaced-root Snapshot() error = %v", err)
	}
}

func TestWindowsSourceUsesHeldRelativeHandles(t *testing.T) {
	_, current, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test source")
	}
	payload, err := os.ReadFile(filepath.Join(filepath.Dir(current), "root_windows.go"))
	if err != nil {
		t.Fatal(err)
	}
	source := string(payload)
	for _, required := range []string{
		"windows.NtCreateFile",
		"RootDirectory: parent",
		"windows.OBJ_DONT_REPARSE",
		"windows.FILE_OPEN_REPARSE_POINT",
		"information.NumberOfLinks != 1",
	} {
		if !strings.Contains(source, required) {
			t.Fatalf("Windows held-relative source missing %q", required)
		}
	}
	if strings.Contains(source, "filepath.Join(r.path") || strings.Contains(source, "filepath.Join(rootPath") {
		t.Fatal("Windows descendant access fell back to mutable absolute pathnames")
	}
}

func TestUnixSourceUsesHeldNoFollowHandles(t *testing.T) {
	_, current, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test source")
	}
	payload, err := os.ReadFile(filepath.Join(filepath.Dir(current), "root_unix.go"))
	if err != nil {
		t.Fatal(err)
	}
	source := string(payload)
	for _, required := range []string{
		"unix.Openat(parent, component, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC",
		"unix.Openat(parent, name, unix.O_RDONLY|unix.O_NOFOLLOW|unix.O_CLOEXEC",
		"unix.Fstatat(parent, name, &value, unix.AT_SYMLINK_NOFOLLOW)",
		"validateUnixFile(final) != nil || !sameUnixIdentity(held, final)",
	} {
		if !strings.Contains(source, required) {
			t.Fatalf("Unix held-relative source missing %q", required)
		}
	}
	if strings.Contains(source, "os.Open(sourcePath)") {
		t.Fatal("Unix descendant access fell back to mutable absolute pathnames")
	}
}
