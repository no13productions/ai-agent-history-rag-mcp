package auth

import (
	"bytes"
	"crypto/pbkdf2"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/no13productions/ai-agent-history-rag-mcp/internal/history/durable"
)

const (
	PBKDF2Rounds       = 200_000
	MaxCredentialBytes = 256
	maxStateBytes      = 16 << 10
	derivedKeyBytes    = 32
	saltBytes          = 16
)

var (
	ErrInvalidCredential = errors.New("invalid credential")
	ErrMalformedBearer   = errors.New("malformed bearer credential")
	ErrStateInvalid      = errors.New("authentication state invalid")
	ErrRotationPending   = errors.New("authentication rotation already pending")
	ErrNoPendingKey      = errors.New("authentication rotation has no pending key")
)

type KeyKind string

const (
	ActiveKey  KeyKind = "active"
	PendingKey KeyKind = "pending"
)

type credential struct {
	ID     string `json:"id"`
	Salt   string `json:"salt"`
	Digest string `json:"digest"`
	Rounds int    `json:"rounds"`
}

type state struct {
	Version int         `json:"version"`
	Active  credential  `json:"active"`
	Pending *credential `json:"pending,omitempty"`
}

type Manager struct {
	root   *durable.Root
	name   string
	random io.Reader
	mu     sync.Mutex
}

func NewManager(root *durable.Root, name string, random io.Reader) (*Manager, error) {
	if root == nil || random == nil || name == "" || name != filepathBase(name) {
		return nil, ErrStateInvalid
	}
	return &Manager{root: root, name: name, random: random}, nil
}

func filepathBase(name string) string {
	if strings.ContainsAny(name, `/\`) || name == "." || name == ".." {
		return ""
	}
	return name
}

func (m *Manager) Initialize(secret string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := validateSecret(secret); err != nil {
		return err
	}
	exists, err := m.root.Exists(m.name)
	if err != nil {
		return fmt.Errorf("inspect authentication state: %w", err)
	}
	if exists {
		current, err := m.loadLocked()
		if err != nil {
			return err
		}
		match, err := verify(secret, current.Active)
		if err != nil || !match {
			return ErrInvalidCredential
		}
		return nil
	}
	active, err := m.derive(secret)
	if err != nil {
		return fmt.Errorf("initialize authentication state: %w", err)
	}
	return m.saveLocked(state{Version: 1, Active: active})
}

func (m *Manager) VerifyBearer(header string) (KeyKind, error) {
	secret, err := parseBearer(header)
	if err != nil {
		return "", err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	current, err := m.loadLocked()
	if err != nil {
		return "", err
	}
	activeMatch, activeErr := verify(secret, current.Active)
	pendingMatch := false
	var pendingErr error
	if current.Pending != nil {
		pendingMatch, pendingErr = verify(secret, *current.Pending)
	} else {
		_, pendingErr = pbkdf2.Key(sha256.New, secret, make([]byte, saltBytes), PBKDF2Rounds, derivedKeyBytes)
	}
	if activeErr != nil || pendingErr != nil {
		return "", ErrStateInvalid
	}
	if activeMatch {
		return ActiveKey, nil
	}
	if pendingMatch {
		return PendingKey, nil
	}
	return "", ErrInvalidCredential
}

func (m *Manager) BeginRotation(secret string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := validateSecret(secret); err != nil {
		return err
	}
	current, err := m.loadLocked()
	if err != nil {
		return err
	}
	if current.Pending != nil {
		return ErrRotationPending
	}
	pending, err := m.derive(secret)
	if err != nil {
		return fmt.Errorf("derive pending credential: %w", err)
	}
	current.Pending = &pending
	return m.saveLocked(current)
}

func (m *Manager) PromotePending() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	current, err := m.loadLocked()
	if err != nil {
		return err
	}
	if current.Pending == nil {
		return ErrNoPendingKey
	}
	current.Active = *current.Pending
	current.Pending = nil
	return m.saveLocked(current)
}

func (m *Manager) derive(secret string) (credential, error) {
	salt := make([]byte, saltBytes)
	if _, err := io.ReadFull(m.random, salt); err != nil {
		return credential{}, err
	}
	digest, err := pbkdf2.Key(sha256.New, secret, salt, PBKDF2Rounds, derivedKeyBytes)
	if err != nil {
		return credential{}, err
	}
	idHash := sha256.Sum256(append(append([]byte{}, salt...), digest...))
	return credential{
		ID:     base64.RawURLEncoding.EncodeToString(idHash[:9]),
		Salt:   base64.RawStdEncoding.EncodeToString(salt),
		Digest: base64.RawStdEncoding.EncodeToString(digest),
		Rounds: PBKDF2Rounds,
	}, nil
}

func verify(secret string, value credential) (bool, error) {
	if err := validateCredential(value); err != nil {
		return false, err
	}
	salt, _ := base64.RawStdEncoding.DecodeString(value.Salt)
	want, _ := base64.RawStdEncoding.DecodeString(value.Digest)
	got, err := pbkdf2.Key(sha256.New, secret, salt, PBKDF2Rounds, derivedKeyBytes)
	if err != nil {
		return false, err
	}
	return subtle.ConstantTimeCompare(got, want) == 1, nil
}

func validateSecret(secret string) error {
	if len(secret) < 16 || len(secret) > MaxCredentialBytes || !utf8.ValidString(secret) || strings.TrimSpace(secret) != secret || strings.ContainsAny(secret, "\r\n\t ") {
		return ErrInvalidCredential
	}
	for _, character := range secret {
		if character < 0x21 || character > 0x7e {
			return ErrInvalidCredential
		}
	}
	return nil
}

func parseBearer(header string) (string, error) {
	if len(header) > len("Bearer ")+MaxCredentialBytes || !strings.HasPrefix(header, "Bearer ") {
		return "", ErrMalformedBearer
	}
	secret := strings.TrimPrefix(header, "Bearer ")
	if err := validateSecret(secret); err != nil || strings.Contains(secret, " ") {
		return "", ErrMalformedBearer
	}
	return secret, nil
}

func (m *Manager) loadLocked() (state, error) {
	payload, err := m.root.Read(m.name, maxStateBytes)
	if err != nil {
		return state{}, fmt.Errorf("read authentication state: %w", err)
	}
	if err := rejectDuplicateJSONKeys(payload); err != nil {
		return state{}, ErrStateInvalid
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var current state
	if err := decoder.Decode(&current); err != nil {
		return state{}, ErrStateInvalid
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return state{}, ErrStateInvalid
	}
	if current.Version != 1 || validateCredential(current.Active) != nil {
		return state{}, ErrStateInvalid
	}
	if current.Pending != nil && validateCredential(*current.Pending) != nil {
		return state{}, ErrStateInvalid
	}
	return current, nil
}

func validateCredential(value credential) error {
	if value.Rounds != PBKDF2Rounds || value.ID == "" {
		return ErrStateInvalid
	}
	salt, err := base64.RawStdEncoding.DecodeString(value.Salt)
	if err != nil || len(salt) != saltBytes {
		return ErrStateInvalid
	}
	digest, err := base64.RawStdEncoding.DecodeString(value.Digest)
	if err != nil || len(digest) != derivedKeyBytes {
		return ErrStateInvalid
	}
	return nil
}

func (m *Manager) saveLocked(current state) error {
	payload, err := json.Marshal(current)
	if err != nil {
		return ErrStateInvalid
	}
	if len(payload) > maxStateBytes {
		return ErrStateInvalid
	}
	if err := m.root.WriteAtomic(m.name, payload); err != nil {
		return fmt.Errorf("write authentication state: %w", err)
	}
	return nil
}
