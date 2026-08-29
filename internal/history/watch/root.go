package watch

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const MaxSourceSnapshotBytes int64 = 512 << 20

var (
	ErrUnsafeRoot       = errors.New("unsafe watch root")
	ErrRootUnbound      = errors.New("watch root is unbound")
	ErrRootChanged      = errors.New("watch root identity changed")
	ErrOutsideRoot      = errors.New("source is outside watch root")
	ErrUnsafeSource     = errors.New("unsafe watch source")
	ErrSourceChanged    = errors.New("watch source changed during snapshot")
	ErrSnapshotLimit    = errors.New("watch snapshot exceeds limit")
	ErrInvalidLimit     = errors.New("invalid watch snapshot limit")
	ErrSnapshotClosed   = errors.New("watch snapshot is closed")
	ErrUnsupportedWatch = errors.New("secure watch roots are unsupported on this platform")
)

type PinnedRoot struct {
	path     string
	mu       sync.Mutex
	closed   bool
	platform platformRoot
}

type RootIdentity struct {
	Volume uint64
	Object uint64
}

func NewPinnedRoot(path string) (*PinnedRoot, error) {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, ErrUnsafeRoot
	}
	return &PinnedRoot{path: path}, nil
}

func (r *PinnedRoot) Path() string {
	return r.path
}

func (r *PinnedRoot) IsBound() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return !r.closed && r.platform.isBound()
}

func (r *PinnedRoot) Identity() (RootIdentity, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed || !r.platform.isBound() {
		return RootIdentity{}, false
	}
	return r.platform.identity(), true
}

func (r *PinnedRoot) Bind() (bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return false, os.ErrClosed
	}
	return r.platform.bind(r.path)
}

func (r *PinnedRoot) Snapshot(candidate string, maxBytes int64) (*Snapshot, error) {
	if maxBytes < 0 || maxBytes > MaxSourceSnapshotBytes {
		return nil, ErrInvalidLimit
	}
	parts, err := relativeParts(r.path, candidate)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, os.ErrClosed
	}
	if !r.platform.isBound() {
		return nil, ErrRootUnbound
	}
	return r.platform.snapshot(r.path, candidate, parts, maxBytes)
}

func (r *PinnedRoot) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	return r.platform.close()
}

func relativeParts(root, candidate string) ([]string, error) {
	if !filepath.IsAbs(candidate) || filepath.Clean(candidate) != candidate {
		return nil, ErrOutsideRoot
	}
	relative, err := filepath.Rel(root, candidate)
	if err != nil || relative == "." || filepath.IsAbs(relative) {
		return nil, ErrOutsideRoot
	}
	parts := strings.Split(relative, string(filepath.Separator))
	if len(parts) == 0 {
		return nil, ErrOutsideRoot
	}
	for _, part := range parts {
		if part == "" || part == "." || part == ".." {
			return nil, ErrOutsideRoot
		}
	}
	return parts, nil
}

type Snapshot struct {
	mu         sync.Mutex
	file       *os.File
	path       string
	directory  string
	sourcePath string
	size       int64
	modTime    time.Time
	digest     [sha256.Size]byte
	closed     bool
	closeErr   error
}

func (s *Snapshot) SourcePath() string { return s.sourcePath }
func (s *Snapshot) Size() int64        { return s.size }
func (s *Snapshot) ModTime() time.Time { return s.modTime }
func (s *Snapshot) SHA256() [sha256.Size]byte {
	return s.digest
}

type errorReader struct{ err error }

func (r errorReader) Read(_ []byte) (int, error) { return 0, r.err }

func (s *Snapshot) Reader() io.Reader {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.file == nil {
		return errorReader{err: ErrSnapshotClosed}
	}
	return io.NewSectionReader(s.file, 0, s.size)
}

func (s *Snapshot) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return s.closeErr
	}
	s.closed = true
	var failures []error
	if s.file != nil {
		if err := s.file.Close(); err != nil {
			failures = append(failures, err)
		}
		s.file = nil
	}
	if err := os.Remove(s.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		failures = append(failures, fmt.Errorf("remove private snapshot: %w", err))
	}
	if err := os.Remove(s.directory); err != nil && !errors.Is(err, os.ErrNotExist) {
		failures = append(failures, fmt.Errorf("remove private snapshot directory: %w", err))
	}
	s.closeErr = errors.Join(failures...)
	return s.closeErr
}

func newPrivateSnapshot(sourcePath string, source *os.File, sourceSize int64, modTime time.Time, maxBytes int64) (*Snapshot, error) {
	if sourceSize < 0 || sourceSize > maxBytes {
		return nil, ErrSnapshotLimit
	}
	directory, err := os.MkdirTemp("", "history-rag-snapshot-")
	if err != nil {
		return nil, fmt.Errorf("create private snapshot directory: %w", err)
	}
	cleanup := true
	path := filepath.Join(directory, "source")
	defer func() {
		if cleanup {
			_ = os.Remove(path)
			_ = os.Remove(directory)
		}
	}()
	if err := os.Chmod(directory, 0o700); err != nil {
		return nil, fmt.Errorf("secure private snapshot directory: %w", err)
	}
	destination, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return nil, fmt.Errorf("create private snapshot: %w", err)
	}
	failed := true
	defer func() {
		if failed {
			_ = destination.Close()
		}
	}()
	if _, err := source.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("seek watch source: %w", err)
	}
	hasher := sha256.New()
	written, err := io.CopyN(io.MultiWriter(destination, hasher), source, sourceSize)
	if err != nil || written != sourceSize {
		return nil, ErrSourceChanged
	}
	var extra [1]byte
	if count, readErr := source.Read(extra[:]); count != 0 || (readErr != nil && !errors.Is(readErr, io.EOF)) {
		return nil, ErrSourceChanged
	}
	if err := destination.Sync(); err != nil {
		return nil, fmt.Errorf("sync private snapshot: %w", err)
	}
	if err := os.Chtimes(path, modTime, modTime); err != nil {
		return nil, fmt.Errorf("preserve watch source modification time: %w", err)
	}
	if err := destination.Chmod(0o400); err != nil {
		return nil, fmt.Errorf("seal private snapshot: %w", err)
	}
	if _, err := destination.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("rewind private snapshot: %w", err)
	}
	var digest [sha256.Size]byte
	copy(digest[:], hasher.Sum(nil))
	failed = false
	cleanup = false
	return &Snapshot{
		file:       destination,
		path:       path,
		directory:  directory,
		sourcePath: sourcePath,
		size:       sourceSize,
		modTime:    modTime,
		digest:     digest,
	}, nil
}
