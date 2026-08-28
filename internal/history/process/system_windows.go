//go:build windows

package process

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"sync"

	"golang.org/x/sys/windows"
)

type SystemOps struct {
	mu                 sync.RWMutex
	expectedExecutable string
	expectedCheckout   string
}

func NewSystemOps() *SystemOps { return &SystemOps{} }

func (ops *SystemOps) Signal(pid int, signal os.Signal) error {
	process, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	if signal == os.Kill {
		if err := process.Kill(); errors.Is(err, os.ErrProcessDone) {
			return ErrProcessNotFound
		} else {
			return err
		}
	}
	if err := windows.GenerateConsoleCtrlEvent(windows.CTRL_BREAK_EVENT, uint32(pid)); err != nil {
		return err
	}
	return nil
}

func (ops *SystemOps) Snapshot(pid int) (Snapshot, error) {
	handle, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		if errors.Is(err, windows.ERROR_INVALID_PARAMETER) {
			return Snapshot{}, ErrProcessNotFound
		}
		return Snapshot{}, err
	}
	defer windows.CloseHandle(handle)
	buffer := make([]uint16, windows.MAX_PATH*4)
	size := uint32(len(buffer))
	if err := windows.QueryFullProcessImageName(handle, 0, &buffer[0], &size); err != nil {
		return Snapshot{}, err
	}
	executable := filepath.Clean(windows.UTF16ToString(buffer[:size]))
	var creation, exit, kernel, user windows.Filetime
	if err := windows.GetProcessTimes(handle, &creation, &exit, &kernel, &user); err != nil {
		return Snapshot{}, err
	}
	ops.mu.RLock()
	expectedExecutable := ops.expectedExecutable
	expectedCheckout := ops.expectedCheckout
	ops.mu.RUnlock()
	if expectedExecutable == "" || executable != expectedExecutable {
		return Snapshot{}, ErrUncertainTarget
	}
	return Snapshot{PID: pid, Executable: executable, CheckoutRoot: expectedCheckout, StartIdentity: strconv.FormatInt(creation.Nanoseconds(), 10)}, nil
}

func terminationSignal() os.Signal { return os.Interrupt }

func CurrentRecord(generic Ops, executable, checkoutRoot string) (Record, error) {
	ops, ok := generic.(*SystemOps)
	if !ok {
		return Record{}, ErrUncertainTarget
	}
	resolvedExecutable, err := filepath.EvalSymlinks(executable)
	if err != nil {
		return Record{}, ErrUncertainTarget
	}
	resolvedCheckout, err := filepath.EvalSymlinks(checkoutRoot)
	if err != nil {
		return Record{}, ErrUncertainTarget
	}
	relative, err := filepath.Rel(resolvedCheckout, resolvedExecutable)
	if err != nil || relative == "." || relative == ".." || filepath.IsAbs(relative) {
		return Record{}, ErrUncertainTarget
	}
	ops.mu.Lock()
	ops.expectedExecutable = resolvedExecutable
	ops.expectedCheckout = resolvedCheckout
	ops.mu.Unlock()
	snapshot, err := ops.Snapshot(os.Getpid())
	if err != nil {
		return Record{}, fmt.Errorf("inspect current process: %w", err)
	}
	return Record(snapshot), nil
}
