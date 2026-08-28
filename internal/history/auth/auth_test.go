package auth

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/no13productions/ai-agent-history-rag-mcp/internal/history/durable"
)

func newManager(t *testing.T) (*Manager, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "state")
	if err := os.Mkdir(path, 0o700); err != nil {
		t.Fatal(err)
	}
	root, err := durable.OpenRoot(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = root.Close() })
	random := bytes.NewReader(bytes.Repeat([]byte{0x42}, 256))
	manager, err := NewManager(root, "auth.json", random)
	if err != nil {
		t.Fatal(err)
	}
	return manager, path
}

func TestInitializeVerifyRotatePromoteWithoutPlaintextPersistence(t *testing.T) {
	manager, path := newManager(t)
	active := "active-secret-with-enough-entropy"
	pending := "pending-secret-with-enough-entropy"
	if err := manager.Initialize(active); err != nil {
		t.Fatal(err)
	}
	if kind, err := manager.VerifyBearer("Bearer " + active); err != nil || kind != ActiveKey {
		t.Fatalf("VerifyBearer(active) = %q, %v", kind, err)
	}
	if _, err := manager.VerifyBearer("Bearer wrong-secret-same-length-value"); !errors.Is(err, ErrInvalidCredential) {
		t.Fatalf("wrong credential error = %v", err)
	}
	if err := manager.BeginRotation(pending); err != nil {
		t.Fatal(err)
	}
	if kind, err := manager.VerifyBearer("Bearer " + pending); err != nil || kind != PendingKey {
		t.Fatalf("VerifyBearer(pending) = %q, %v", kind, err)
	}
	if err := manager.PromotePending(); err != nil {
		t.Fatal(err)
	}
	if _, err := manager.VerifyBearer("Bearer " + active); !errors.Is(err, ErrInvalidCredential) {
		t.Fatalf("old credential error = %v", err)
	}
	if kind, err := manager.VerifyBearer("Bearer " + pending); err != nil || kind != ActiveKey {
		t.Fatalf("promoted credential = %q, %v", kind, err)
	}
	stored, err := os.ReadFile(filepath.Join(path, "auth.json"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(stored, []byte(active)) || bytes.Contains(stored, []byte(pending)) {
		t.Fatal("durable auth state contains plaintext credential")
	}
	if !bytes.Contains(stored, []byte(`"rounds":200000`)) {
		t.Fatalf("durable state lacks exact round count: %s", stored)
	}
}

func TestBearerParsingIsClosedAndBounded(t *testing.T) {
	manager, _ := newManager(t)
	if err := manager.Initialize("active-secret-with-enough-entropy"); err != nil {
		t.Fatal(err)
	}
	bad := []string{
		"", "Basic abc", "Bearer", "bearer secret", "Bearer  secret", "Bearer secret extra",
		"Bearer " + strings.Repeat("x", MaxCredentialBytes+1),
	}
	for _, header := range bad {
		if _, err := manager.VerifyBearer(header); err == nil {
			t.Fatalf("VerifyBearer(%q) accepted malformed input", header)
		}
	}
}

func TestCorruptOrSymlinkStateFailsClosed(t *testing.T) {
	manager, path := newManager(t)
	if err := os.WriteFile(filepath.Join(path, "auth.json"), []byte(`{"version":1,"unknown":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := manager.VerifyBearer("Bearer secret"); err == nil {
		t.Fatal("corrupt state was accepted")
	}
	if err := os.Remove(filepath.Join(path, "auth.json")); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(t.TempDir(), "auth.json")
	if err := os.WriteFile(outside, []byte(`{}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(path, "auth.json")); err != nil {
		t.Fatal(err)
	}
	other, err := NewManager(manager.root, "auth.json", strings.NewReader(strings.Repeat("r", 128)))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := other.VerifyBearer("Bearer secret"); err == nil {
		t.Fatal("symlink state was accepted")
	}
}

func TestDuplicateAuthStateFailsClosed(t *testing.T) {
	manager, path := newManager(t)
	if err := manager.Initialize("active-secret-with-enough-entropy"); err != nil {
		t.Fatal(err)
	}
	payload, err := os.ReadFile(filepath.Join(path, "auth.json"))
	if err != nil {
		t.Fatal(err)
	}
	duplicate := strings.Replace(string(payload), `"version":1`, `"version":1,"version":1`, 1)
	if err := os.WriteFile(filepath.Join(path, "auth.json"), []byte(duplicate), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := manager.VerifyBearer("Bearer active-secret-with-enough-entropy"); !errors.Is(err, ErrStateInvalid) {
		t.Fatalf("duplicate state error = %v", err)
	}
}
