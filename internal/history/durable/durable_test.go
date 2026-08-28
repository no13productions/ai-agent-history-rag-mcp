package durable

import (
	"crypto/sha256"
	"os"
	"path/filepath"
	"testing"
)

func openTestRoot(t *testing.T) (*Root, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "state")
	if err := os.Mkdir(path, 0o700); err != nil {
		t.Fatal(err)
	}
	root, err := OpenRoot(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = root.Close() })
	return root, path
}

func TestAtomicWriteReadAndOwnerOnlyMode(t *testing.T) {
	root, path := openTestRoot(t)
	if err := root.WriteAtomic("state.json", []byte("first")); err != nil {
		t.Fatal(err)
	}
	if err := root.WriteAtomic("state.json", []byte("second")); err != nil {
		t.Fatal(err)
	}
	got, err := root.Read("state.json", 6)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "second" {
		t.Fatalf("Read() = %q", got)
	}
	info, err := os.Stat(filepath.Join(path, "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("mode = %#o", info.Mode().Perm())
	}
	if _, err := root.Read("state.json", 5); err == nil {
		t.Fatal("Read() ignored bound")
	}
}

func TestRefusesSymlinkHardlinkNonRegularAndUnsafeName(t *testing.T) {
	root, path := openTestRoot(t)
	outside := filepath.Join(t.TempDir(), "outside")
	if err := os.WriteFile(outside, []byte("secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(path, "link")); err != nil {
		t.Fatal(err)
	}
	if _, err := root.Read("link", 64); err == nil {
		t.Fatal("Read() followed symlink")
	}
	if err := os.Link(outside, filepath.Join(path, "hard")); err != nil {
		t.Fatal(err)
	}
	if _, err := root.Read("hard", 64); err == nil {
		t.Fatal("Read() accepted hardlink")
	}
	if err := os.Mkdir(filepath.Join(path, "directory"), 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := root.Read("directory", 64); err == nil {
		t.Fatal("Read() accepted directory")
	}
	for _, name := range []string{"", ".", "..", "../escape", "a/b"} {
		if err := root.WriteAtomic(name, []byte("x")); err == nil {
			t.Fatalf("WriteAtomic(%q) accepted unsafe name", name)
		}
	}
}

func TestRootReplacementFailsClosed(t *testing.T) {
	root, path := openTestRoot(t)
	if err := root.WriteAtomic("state", []byte("original")); err != nil {
		t.Fatal(err)
	}
	moved := path + "-moved"
	if err := os.Rename(path, moved); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(path, 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := root.Read("state", 64); err == nil {
		t.Fatal("Read() used replaced root pathname")
	}
}

func TestDeleteIsContentBound(t *testing.T) {
	root, path := openTestRoot(t)
	payload := []byte("bound payload")
	if err := root.WriteAtomic("payload", payload); err != nil {
		t.Fatal(err)
	}
	wrong := sha256.Sum256([]byte("wrong"))
	if _, err := root.Remove("payload", wrong); err == nil {
		t.Fatal("Remove() ignored content identity")
	}
	if _, err := os.Stat(filepath.Join(path, "payload")); err != nil {
		t.Fatalf("mismatch removed file: %v", err)
	}
	right := sha256.Sum256(payload)
	removed, err := root.Remove("payload", right)
	if err != nil || !removed {
		t.Fatalf("Remove() = %v, %v", removed, err)
	}
	if _, err := os.Stat(filepath.Join(path, "payload")); !os.IsNotExist(err) {
		t.Fatalf("file remains: %v", err)
	}
}
