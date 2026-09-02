package process

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/no13productions/ai-agent-history-rag-mcp/internal/history/durable"
)

var (
	ErrProcessNotFound     = errors.New("process not found")
	ErrUncertainTarget     = errors.New("process target identity is uncertain")
	ErrTerminationUnproven = errors.New("process termination could not be proven")
	ErrPIDRecordInvalid    = errors.New("pid record invalid")
)

const maxPIDRecordBytes int64 = 8 << 10

type Mode int

const (
	Start Mode = iota
	Supervise
)

type Record struct {
	PID           int    `json:"pid"`
	Executable    string `json:"executable"`
	CheckoutRoot  string `json:"checkout_root"`
	StartIdentity string `json:"start_identity"`
}

type Snapshot Record

type Ops interface {
	Snapshot(pid int) (Snapshot, error)
	Signal(pid int, signal os.Signal) error
}

type Sleep func(context.Context, time.Duration) error

type Controller struct {
	root       *durable.Root
	name       string
	ops        Ops
	sleep      Sleep
	termSignal os.Signal
}

func NewController(root *durable.Root, name string, ops Ops, sleep Sleep) (*Controller, error) {
	if root == nil || ops == nil || sleep == nil || filepath.Base(name) != name || name == "." || name == ".." {
		return nil, ErrPIDRecordInvalid
	}
	return &Controller{root: root, name: name, ops: ops, sleep: sleep, termSignal: terminationSignal()}, nil
}

func (c *Controller) WriteRecord(record Record) error {
	if err := validateRecord(record); err != nil {
		return err
	}
	payload, err := json.Marshal(record)
	if err != nil {
		return ErrPIDRecordInvalid
	}
	return c.root.WriteAtomic(c.name, payload)
}

func (c *Controller) RemoveRecord(record Record) error {
	payload, err := json.Marshal(record)
	if err != nil {
		return ErrPIDRecordInvalid
	}
	_, err = c.root.Remove(c.name, sha256.Sum256(payload))
	return err
}

func (c *Controller) Prepare(ctx context.Context, mode Mode, current Record, termTimeout, killTimeout time.Duration) (bool, error) {
	if mode != Start && mode != Supervise {
		return false, ErrPIDRecordInvalid
	}
	if err := validateRecord(current); err != nil {
		return false, err
	}
	existing, payload, exists, err := c.readRecord()
	if err != nil {
		return false, err
	}
	if !exists {
		return true, nil
	}
	snapshot, err := c.ops.Snapshot(existing.PID)
	if errors.Is(err, ErrProcessNotFound) {
		if _, removeErr := c.root.Remove(c.name, sha256.Sum256(payload)); removeErr != nil {
			return false, fmt.Errorf("retire stale pid record: %w", removeErr)
		}
		return true, nil
	}
	if err != nil || !matches(existing, snapshot) {
		return false, ErrUncertainTarget
	}
	if mode == Start {
		return false, nil
	}
	exited := false
	if c.termSignal == os.Kill {
		signalErr := c.ops.Signal(existing.PID, os.Kill)
		if errors.Is(signalErr, ErrProcessNotFound) {
			exited = true
		} else if signalErr != nil {
			return false, fmt.Errorf("kill supervised process: %w", signalErr)
		}
		if !exited {
			exited, err = c.waitForExit(ctx, existing, killTimeout)
			if err != nil {
				return false, err
			}
		}
	} else {
		signalErr := c.ops.Signal(existing.PID, c.termSignal)
		if errors.Is(signalErr, ErrProcessNotFound) {
			exited = true
		} else if signalErr == nil {
			exited, err = c.waitForExit(ctx, existing, termTimeout)
			if err != nil {
				return false, err
			}
		}
	}
	if !exited && c.termSignal != os.Kill {
		snapshot, inspectErr := c.ops.Snapshot(existing.PID)
		if inspectErr != nil || !matches(existing, snapshot) {
			return false, ErrUncertainTarget
		}
		if err := c.ops.Signal(existing.PID, os.Kill); err != nil {
			return false, fmt.Errorf("kill supervised process: %w", err)
		}
		exited, err = c.waitForExit(ctx, existing, killTimeout)
		if err != nil {
			return false, err
		}
	}
	if !exited {
		return false, ErrTerminationUnproven
	}
	if _, err := c.root.Remove(c.name, sha256.Sum256(payload)); err != nil {
		return false, fmt.Errorf("retire replaced pid record: %w", err)
	}
	return true, nil
}

func (c *Controller) waitForExit(ctx context.Context, target Record, timeout time.Duration) (bool, error) {
	deadline := time.Now().Add(timeout)
	for {
		snapshot, err := c.ops.Snapshot(target.PID)
		if errors.Is(err, ErrProcessNotFound) {
			return true, nil
		}
		if err != nil {
			return false, ErrUncertainTarget
		}
		if !matches(target, snapshot) {
			return false, ErrUncertainTarget
		}
		if timeout <= 0 || !time.Now().Before(deadline) {
			return false, nil
		}
		remaining := time.Until(deadline)
		if remaining > 50*time.Millisecond {
			remaining = 50 * time.Millisecond
		}
		if err := c.sleep(ctx, remaining); err != nil {
			return false, err
		}
	}
}

func (c *Controller) readRecord() (Record, []byte, bool, error) {
	exists, err := c.root.Exists(c.name)
	if err != nil {
		return Record{}, nil, false, err
	}
	if !exists {
		return Record{}, nil, false, nil
	}
	payload, err := c.root.Read(c.name, maxPIDRecordBytes)
	if err != nil {
		return Record{}, nil, false, err
	}
	if hasDuplicateTopLevelKey(payload) {
		return Record{}, nil, false, ErrPIDRecordInvalid
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var record Record
	if err := decoder.Decode(&record); err != nil {
		return Record{}, nil, false, ErrPIDRecordInvalid
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return Record{}, nil, false, ErrPIDRecordInvalid
	}
	if err := validateRecord(record); err != nil {
		return Record{}, nil, false, err
	}
	return record, payload, true, nil
}

func hasDuplicateTopLevelKey(payload []byte) bool {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	token, err := decoder.Token()
	if err != nil || token != json.Delim('{') {
		return true
	}
	seen := make(map[string]struct{})
	for decoder.More() {
		keyToken, err := decoder.Token()
		if err != nil {
			return true
		}
		key, ok := keyToken.(string)
		if !ok {
			return true
		}
		if _, exists := seen[key]; exists {
			return true
		}
		seen[key] = struct{}{}
		var value any
		if err := decoder.Decode(&value); err != nil {
			return true
		}
	}
	closing, err := decoder.Token()
	return err != nil || closing != json.Delim('}')
}

func validateRecord(record Record) error {
	if record.PID <= 1 || record.StartIdentity == "" || !filepath.IsAbs(record.Executable) || filepath.Clean(record.Executable) != record.Executable || !filepath.IsAbs(record.CheckoutRoot) || filepath.Clean(record.CheckoutRoot) != record.CheckoutRoot {
		return ErrPIDRecordInvalid
	}
	relative, err := filepath.Rel(record.CheckoutRoot, record.Executable)
	if err != nil || relative == "." || relative == ".." || filepath.IsAbs(relative) || len(relative) >= 3 && relative[:3] == ".."+string(filepath.Separator) {
		return ErrPIDRecordInvalid
	}
	return nil
}

func matches(record Record, snapshot Snapshot) bool {
	return record.PID == snapshot.PID && record.Executable == snapshot.Executable && record.CheckoutRoot == snapshot.CheckoutRoot && record.StartIdentity == snapshot.StartIdentity
}

func DefaultSleep(ctx context.Context, duration time.Duration) error {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
