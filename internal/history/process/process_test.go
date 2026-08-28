package process

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/no13productions/ai-agent-history-rag-mcp/internal/history/durable"
)

type fakeOps struct {
	mu        sync.Mutex
	snapshots map[int]Snapshot
	signals   []os.Signal
	termStops bool
	killStops bool
}

func (f *fakeOps) Snapshot(pid int) (Snapshot, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	snapshot, ok := f.snapshots[pid]
	if !ok {
		return Snapshot{}, ErrProcessNotFound
	}
	return snapshot, nil
}

func (f *fakeOps) Signal(pid int, signal os.Signal) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.signals = append(f.signals, signal)
	if (signal == os.Interrupt && f.termStops) || (signal == os.Kill && f.killStops) {
		delete(f.snapshots, pid)
	}
	return nil
}

func testController(t *testing.T, ops *fakeOps) (*Controller, *durable.Root) {
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
	controller, err := NewController(root, "daemon.pid", ops, func(context.Context, time.Duration) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	controller.termSignal = os.Interrupt
	return controller, root
}

func record(pid int) Record {
	return Record{PID: pid, Executable: "/checkout/bin/history-ragd", CheckoutRoot: "/checkout", StartIdentity: "start-1"}
}

func snapshot(value Record) Snapshot {
	return Snapshot{PID: value.PID, Executable: value.Executable, CheckoutRoot: value.CheckoutRoot, StartIdentity: value.StartIdentity}
}

func TestStartIsIdempotentForExactRunningOwner(t *testing.T) {
	r := record(123)
	ops := &fakeOps{snapshots: map[int]Snapshot{123: snapshot(r)}}
	controller, root := testController(t, ops)
	if err := controller.WriteRecord(r); err != nil {
		t.Fatal(err)
	}
	shouldRun, err := controller.Prepare(context.Background(), Start, record(999), time.Second, time.Second)
	if err != nil || shouldRun {
		t.Fatalf("Prepare(start) = %v, %v", shouldRun, err)
	}
	if exists, err := root.Exists("daemon.pid"); err != nil || !exists {
		t.Fatalf("pid file exists = %v, %v", exists, err)
	}
}

func TestPIDReuseAndSameCheckoutMismatchFailClosed(t *testing.T) {
	for name, mutate := range map[string]func(*Snapshot){
		"executable": func(s *Snapshot) { s.Executable = "/usr/bin/other" },
		"checkout":   func(s *Snapshot) { s.CheckoutRoot = "/other" },
		"start":      func(s *Snapshot) { s.StartIdentity = "reused" },
	} {
		t.Run(name, func(t *testing.T) {
			r := record(123)
			s := snapshot(r)
			mutate(&s)
			ops := &fakeOps{snapshots: map[int]Snapshot{123: s}}
			controller, _ := testController(t, ops)
			if err := controller.WriteRecord(r); err != nil {
				t.Fatal(err)
			}
			if _, err := controller.Prepare(context.Background(), Supervise, record(999), time.Second, time.Second); !errors.Is(err, ErrUncertainTarget) {
				t.Fatalf("Prepare() error = %v", err)
			}
			if len(ops.signals) != 0 {
				t.Fatalf("mismatched PID was signaled: %v", ops.signals)
			}
		})
	}
}

func TestSuperviseTermThenKillAndRefusesSurvivor(t *testing.T) {
	r := record(123)
	ops := &fakeOps{snapshots: map[int]Snapshot{123: snapshot(r)}, killStops: true}
	controller, _ := testController(t, ops)
	if err := controller.WriteRecord(r); err != nil {
		t.Fatal(err)
	}
	shouldRun, err := controller.Prepare(context.Background(), Supervise, record(999), 0, 0)
	if err != nil || !shouldRun {
		t.Fatalf("Prepare() = %v, %v", shouldRun, err)
	}
	if len(ops.signals) != 2 || ops.signals[0] != os.Interrupt || ops.signals[1] != os.Kill {
		t.Fatalf("signals = %v", ops.signals)
	}

	r = record(456)
	ops = &fakeOps{snapshots: map[int]Snapshot{456: snapshot(r)}}
	controller, _ = testController(t, ops)
	if err := controller.WriteRecord(r); err != nil {
		t.Fatal(err)
	}
	if _, err := controller.Prepare(context.Background(), Supervise, record(999), 0, 0); !errors.Is(err, ErrTerminationUnproven) {
		t.Fatalf("survivor error = %v", err)
	}
}

func TestStaleAbsentPIDIsCleanedButMalformedFileFailsClosed(t *testing.T) {
	r := record(123)
	ops := &fakeOps{snapshots: map[int]Snapshot{}}
	controller, root := testController(t, ops)
	if err := controller.WriteRecord(r); err != nil {
		t.Fatal(err)
	}
	shouldRun, err := controller.Prepare(context.Background(), Start, record(999), time.Second, time.Second)
	if err != nil || !shouldRun {
		t.Fatalf("stale Prepare() = %v, %v", shouldRun, err)
	}
	if err := root.WriteAtomic("daemon.pid", []byte("123")); err != nil {
		t.Fatal(err)
	}
	if _, err := controller.Prepare(context.Background(), Start, record(999), time.Second, time.Second); err == nil {
		t.Fatal("malformed pid file was treated as stale")
	}
	duplicate := `{"pid":123,"pid":123,"executable":"/checkout/bin/history-ragd","checkout_root":"/checkout","start_identity":"start-1"}`
	if err := root.WriteAtomic("daemon.pid", []byte(duplicate)); err != nil {
		t.Fatal(err)
	}
	if _, err := controller.Prepare(context.Background(), Start, record(999), time.Second, time.Second); !errors.Is(err, ErrPIDRecordInvalid) {
		t.Fatalf("duplicate pid record error = %v", err)
	}
}
