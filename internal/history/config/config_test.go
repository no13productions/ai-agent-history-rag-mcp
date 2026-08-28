package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func validConfig(t *testing.T) (string, string) {
	t.Helper()
	base := t.TempDir()
	state := filepath.Join(base, "state")
	checkout := filepath.Join(base, "checkout")
	if err := os.Mkdir(state, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(checkout, 0o700); err != nil {
		t.Fatal(err)
	}
	executable := filepath.Join(checkout, "history-ragd")
	if err := os.WriteFile(executable, []byte("candidate"), 0o700); err != nil {
		t.Fatal(err)
	}
	body := `{"state_dir":` + quote(state) + `,"listen":"127.0.0.1:4680","pid_file":` + quote(filepath.Join(state, "daemon.pid")) + `,"auth_state_file":` + quote(filepath.Join(state, "auth.json")) + `,"auth_enabled":true,"checkout_root":` + quote(checkout) + `,"executable":` + quote(executable) + `}`
	path := filepath.Join(base, "config.json")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path, body
}

func quote(value string) string {
	return `"` + strings.ReplaceAll(value, `\`, `\\`) + `"`
}

func TestLoadClosedAndSafe(t *testing.T) {
	path, _ := validConfig(t)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.Listen != "127.0.0.1:4680" || !cfg.AuthEnabled {
		t.Fatalf("unexpected config: %#v", cfg)
	}
}

func TestLoadRejectsUnknownDuplicateOversizedAndTrailingData(t *testing.T) {
	path, body := validConfig(t)
	cases := map[string]string{
		"unknown":   strings.Replace(body, `}`, `,"surprise":true}`, 1),
		"duplicate": strings.Replace(body, `"listen":"127.0.0.1:4680"`, `"listen":"127.0.0.1:4680","listen":"127.0.0.1:4681"`, 1),
		"trailing":  body + `{}`,
		"oversized": body + strings.Repeat(" ", int(MaxConfigBytes)),
	}
	for name, content := range cases {
		t.Run(name, func(t *testing.T) {
			if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := Load(path); err == nil {
				t.Fatal("Load() accepted ambiguous or unbounded config")
			}
		})
	}
}

func TestLoadRejectsUnsafePathsNonLoopbackAndSymlinkConfig(t *testing.T) {
	path, body := validConfig(t)
	stateDir := filepath.Dir(filepath.Join(filepath.Dir(path), "state", "daemon.pid"))
	cases := map[string]string{
		"pid outside state": strings.Replace(body, quote(filepath.Join(stateDir, "daemon.pid")), quote(filepath.Join(filepath.Dir(stateDir), "outside.pid")), 1),
		"non loopback":      strings.Replace(body, "127.0.0.1:4680", "0.0.0.0:4680", 1),
		"relative state":    strings.Replace(body, quote(stateDir), quote("relative-state"), 1),
	}
	for name, content := range cases {
		t.Run(name, func(t *testing.T) {
			if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := Load(path); err == nil {
				t.Fatal("Load() accepted unsafe config")
			}
		})
	}

	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(filepath.Dir(path), "config-link.json")
	if err := os.Symlink(path, link); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(link); err == nil {
		t.Fatal("Load() followed a config symlink")
	}
	hardlink := filepath.Join(filepath.Dir(path), "config-hardlink.json")
	if err := os.Link(path, hardlink); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("Load() accepted a multiply-linked config")
	}
}
