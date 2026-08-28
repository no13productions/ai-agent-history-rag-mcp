//go:build darwin || linux

package process

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
)

type SystemOps struct{}

func NewSystemOps() *SystemOps { return &SystemOps{} }

func (SystemOps) Signal(pid int, signal os.Signal) error {
	process, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	if err := process.Signal(signal); errors.Is(err, os.ErrProcessDone) {
		return ErrProcessNotFound
	} else {
		return err
	}
}

func (SystemOps) Snapshot(pid int) (Snapshot, error) {
	if err := syscall.Kill(pid, 0); err != nil {
		if errors.Is(err, syscall.ESRCH) {
			return Snapshot{}, ErrProcessNotFound
		}
		return Snapshot{}, err
	}
	if runtime.GOOS == "linux" {
		return linuxSnapshot(pid)
	}
	return darwinSnapshot(pid)
}

func linuxSnapshot(pid int) (Snapshot, error) {
	base := filepath.Join("/proc", strconv.Itoa(pid))
	executable, err := filepath.EvalSymlinks(filepath.Join(base, "exe"))
	if err != nil {
		return Snapshot{}, err
	}
	checkout, err := filepath.EvalSymlinks(filepath.Join(base, "cwd"))
	if err != nil {
		return Snapshot{}, err
	}
	stat, err := os.ReadFile(filepath.Join(base, "stat"))
	if err != nil {
		return Snapshot{}, err
	}
	closing := strings.LastIndexByte(string(stat), ')')
	if closing < 0 {
		return Snapshot{}, ErrUncertainTarget
	}
	fields := strings.Fields(string(stat[closing+1:]))
	if len(fields) < 20 {
		return Snapshot{}, ErrUncertainTarget
	}
	return Snapshot{PID: pid, Executable: executable, CheckoutRoot: checkout, StartIdentity: fields[19]}, nil
}

func darwinSnapshot(pid int) (Snapshot, error) {
	commandOutput, err := exec.Command("ps", "-ww", "-p", strconv.Itoa(pid), "-o", "lstart=", "-o", "command=").Output()
	if err != nil {
		return Snapshot{}, ErrProcessNotFound
	}
	fields := strings.Fields(string(commandOutput))
	if len(fields) < 6 {
		return Snapshot{}, ErrUncertainTarget
	}
	executable := fields[5]
	if !filepath.IsAbs(executable) {
		return Snapshot{}, ErrUncertainTarget
	}
	executable, err = filepath.EvalSymlinks(executable)
	if err != nil {
		return Snapshot{}, ErrUncertainTarget
	}
	lsofOutput, err := exec.Command("lsof", "-a", "-p", strconv.Itoa(pid), "-d", "cwd", "-Fn").Output()
	if err != nil {
		return Snapshot{}, ErrUncertainTarget
	}
	checkout := ""
	scanner := bufio.NewScanner(strings.NewReader(string(lsofOutput)))
	for scanner.Scan() {
		if strings.HasPrefix(scanner.Text(), "n/") {
			checkout = strings.TrimPrefix(scanner.Text(), "n")
			break
		}
	}
	if checkout == "" || scanner.Err() != nil {
		return Snapshot{}, ErrUncertainTarget
	}
	checkout, err = filepath.EvalSymlinks(checkout)
	if err != nil {
		return Snapshot{}, ErrUncertainTarget
	}
	return Snapshot{PID: pid, Executable: executable, CheckoutRoot: checkout, StartIdentity: strings.Join(fields[:5], " ")}, nil
}

func terminationSignal() os.Signal { return syscall.SIGTERM }

func CurrentRecord(ops Ops, executable, checkoutRoot string) (Record, error) {
	snapshot, err := ops.Snapshot(os.Getpid())
	if err != nil {
		return Record{}, fmt.Errorf("inspect current process: %w", err)
	}
	resolvedExecutable, err := filepath.EvalSymlinks(executable)
	if err != nil || resolvedExecutable != snapshot.Executable {
		return Record{}, ErrUncertainTarget
	}
	resolvedCheckout, err := filepath.EvalSymlinks(checkoutRoot)
	if err != nil || resolvedCheckout != snapshot.CheckoutRoot {
		return Record{}, ErrUncertainTarget
	}
	return Record(snapshot), nil
}
