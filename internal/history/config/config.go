package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
)

const MaxConfigBytes int64 = 64 << 10

type Config struct {
	StateDir      string `json:"state_dir"`
	Listen        string `json:"listen"`
	PIDFile       string `json:"pid_file"`
	AuthStateFile string `json:"auth_state_file"`
	AuthEnabled   bool   `json:"auth_enabled"`
	CheckoutRoot  string `json:"checkout_root"`
	Executable    string `json:"executable"`
}

func Load(path string) (Config, error) {
	payload, err := readClosedFile(path, MaxConfigBytes)
	if err != nil {
		return Config{}, fmt.Errorf("read config: %w", err)
	}
	if err := rejectDuplicateKeys(payload); err != nil {
		return Config{}, fmt.Errorf("decode config: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var cfg Config
	if err := decoder.Decode(&cfg); err != nil {
		return Config{}, fmt.Errorf("decode config: %w", err)
	}
	if err := expectEOF(decoder); err != nil {
		return Config{}, fmt.Errorf("decode config: %w", err)
	}
	if err := cfg.validate(); err != nil {
		return Config{}, fmt.Errorf("validate config: %w", err)
	}
	return cfg, nil
}

func readClosedFile(path string, maxBytes int64) ([]byte, error) {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, errors.New("config path must be absolute and clean")
	}
	before, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !before.Mode().IsRegular() || before.Mode()&os.ModeSymlink != 0 || requireOwnerOnlyConfigMode(before) != nil {
		return nil, errors.New("config file must be owner-only regular file")
	}
	if err := requireSingleLink(path); err != nil {
		return nil, errors.New("config file must have one name")
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	opened, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if !os.SameFile(before, opened) || !opened.Mode().IsRegular() || requireOwnerOnlyConfigMode(opened) != nil {
		return nil, errors.New("config file identity changed")
	}
	payload, err := io.ReadAll(io.LimitReader(file, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(payload)) > maxBytes {
		return nil, errors.New("config exceeds size limit")
	}
	after, err := os.Lstat(path)
	if err != nil || !os.SameFile(opened, after) {
		return nil, errors.New("config file identity changed")
	}
	return payload, nil
}

func rejectDuplicateKeys(payload []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	if err := walkJSON(decoder); err != nil {
		return err
	}
	return expectEOF(decoder)
}

func walkJSON(decoder *json.Decoder) error {
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]struct{})
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return err
			}
			key, ok := keyToken.(string)
			if !ok {
				return errors.New("object key is not a string")
			}
			if _, exists := seen[key]; exists {
				return errors.New("duplicate object key")
			}
			seen[key] = struct{}{}
			if err := walkJSON(decoder); err != nil {
				return err
			}
		}
		closing, err := decoder.Token()
		if err != nil || closing != json.Delim('}') {
			return errors.New("unterminated object")
		}
	case '[':
		for decoder.More() {
			if err := walkJSON(decoder); err != nil {
				return err
			}
		}
		closing, err := decoder.Token()
		if err != nil || closing != json.Delim(']') {
			return errors.New("unterminated array")
		}
	default:
		return errors.New("unexpected delimiter")
	}
	return nil
}

func expectEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("trailing JSON value")
		}
		return err
	}
	return nil
}

func (cfg *Config) validate() error {
	if cfg.Listen != "127.0.0.1:4680" {
		return errors.New("listen must be the fixed loopback endpoint")
	}
	host, port, err := net.SplitHostPort(cfg.Listen)
	if err != nil || port != "4680" || net.ParseIP(host) == nil || !net.ParseIP(host).IsLoopback() {
		return errors.New("listen must be loopback port 4680")
	}
	for label, value := range map[string]string{
		"state_dir": cfg.StateDir, "pid_file": cfg.PIDFile, "auth_state_file": cfg.AuthStateFile,
		"checkout_root": cfg.CheckoutRoot, "executable": cfg.Executable,
	} {
		if !filepath.IsAbs(value) || filepath.Clean(value) != value {
			return fmt.Errorf("%s must be absolute and clean", label)
		}
	}
	if filepath.Dir(cfg.PIDFile) != cfg.StateDir || filepath.Dir(cfg.AuthStateFile) != cfg.StateDir || cfg.PIDFile == cfg.AuthStateFile {
		return errors.New("durable files must be distinct direct children of state_dir")
	}
	stateDir, err := requireCanonicalDirectory(cfg.StateDir, 0o700)
	if err != nil {
		return fmt.Errorf("state_dir: %w", err)
	}
	checkoutRoot, err := requireCanonicalDirectory(cfg.CheckoutRoot, 0)
	if err != nil {
		return fmt.Errorf("checkout_root: %w", err)
	}
	executable, err := filepath.EvalSymlinks(cfg.Executable)
	if err != nil {
		return errors.New("executable must exist")
	}
	relative, err := filepath.Rel(checkoutRoot, executable)
	if err != nil || relative == "." || relative == ".." || filepath.IsAbs(relative) || len(relative) >= 3 && relative[:3] == ".."+string(filepath.Separator) {
		return errors.New("executable must be beneath checkout_root")
	}
	info, err := os.Lstat(executable)
	if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return errors.New("executable must be a regular file")
	}
	cfg.StateDir = stateDir
	cfg.PIDFile = filepath.Join(stateDir, filepath.Base(cfg.PIDFile))
	cfg.AuthStateFile = filepath.Join(stateDir, filepath.Base(cfg.AuthStateFile))
	cfg.CheckoutRoot = checkoutRoot
	cfg.Executable = executable
	return nil
}

func requireCanonicalDirectory(path string, mode os.FileMode) (string, error) {
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", errors.New("directory must exist")
	}
	info, err := os.Lstat(resolved)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", errors.New("directory must be a non-link directory")
	}
	if mode != 0 && info.Mode().Perm() != mode {
		return "", errors.New("directory mode is not owner-only")
	}
	return resolved, nil
}
