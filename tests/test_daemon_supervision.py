"""Tests for daemon lifecycle ownership commands."""

from types import SimpleNamespace

from claude_history_rag import daemon


def test_stale_reused_pid_file_is_removed_without_reporting_daemon(monkeypatch, tmp_path):
    """A PID file pointing at another process must not become a kill target."""
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text("1234")
    monkeypatch.setattr(daemon, "PID_FILE", pid_file)
    monkeypatch.setattr(daemon.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(daemon, "_pid_is_history_daemon", lambda pid: False)

    assert daemon.is_daemon_running() == (False, None)
    assert not pid_file.exists()


def test_start_is_idempotent_and_does_not_replace_running_daemon(monkeypatch, capsys):
    """Manual start should report an existing daemon without taking ownership."""
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: (True, 1234))

    def fail_run():
        raise AssertionError("start must not run a replacement daemon")

    monkeypatch.setattr(daemon, "_run_foreground_daemon", fail_run)

    assert daemon.cmd_start(SimpleNamespace()) == 0
    assert "Daemon is already running (PID 1234)" in capsys.readouterr().out


def test_supervise_replaces_pid_file_daemon_before_foreground_run(monkeypatch, capsys):
    """Service managers should become the lifecycle owner instead of wrapping an orphan."""
    calls: list[tuple[str, int | None]] = []

    monkeypatch.setattr(daemon, "is_daemon_running", lambda: (True, 1234))

    def fake_terminate(pid):
        calls.append(("terminate", pid))
        return True

    def fake_run():
        calls.append(("run", None))
        return 0

    monkeypatch.setattr(daemon, "terminate_daemon_process", fake_terminate)
    monkeypatch.setattr(daemon, "_run_foreground_daemon", fake_run)

    assert daemon.cmd_supervise(SimpleNamespace()) == 0
    assert calls == [("terminate", 1234), ("run", None)]
    assert "Replacing existing daemon (PID 1234)" in capsys.readouterr().out


def test_supervise_fails_closed_when_existing_daemon_cannot_stop(monkeypatch, capsys):
    """A supervisor must not start a second daemon if the old owner survives."""
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: (True, 1234))
    monkeypatch.setattr(daemon, "terminate_daemon_process", lambda pid: False)

    def fail_run():
        raise AssertionError("supervise must not run while the old daemon is alive")

    monkeypatch.setattr(daemon, "_run_foreground_daemon", fail_run)

    assert daemon.cmd_supervise(SimpleNamespace()) == 1
    assert "Failed to stop existing daemon (PID 1234)" in capsys.readouterr().out


def test_stop_uses_shared_daemon_termination_path(monkeypatch, capsys):
    """Stop and supervise should use the same PID-file shutdown primitive."""
    calls: list[int] = []
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: (True, 1234))

    def fake_terminate(pid, **kwargs):
        calls.append(pid)
        assert kwargs["timeout_seconds"] == 15.0
        assert kwargs["kill_timeout_seconds"] == 5.0
        return True

    monkeypatch.setattr(daemon, "terminate_daemon_process", fake_terminate)

    assert daemon.cmd_stop(SimpleNamespace()) == 0
    assert calls == [1234]
    assert "Daemon stopped" in capsys.readouterr().out


def test_service_manager_configs_use_supervise():
    """Generated service files must not invoke the human idempotent start wrapper."""
    root = daemon.Path(__file__).resolve().parents[1]
    files = [
        root / "scripts/com.ai-agent-history-rag.daemon.plist",
        root / "scripts/com.ai-agent-history-rag.daemon.plist.template",
        root / "scripts/install-launchd.sh",
        root / "scripts/install-systemd.sh",
        root / "scripts/install-windows.ps1",
        root / "scripts/ai-agent-history-rag.service",
        root / "src/claude_history_rag/installer.py",
    ]

    for path in files:
        content = path.read_text()
        assert "ai-agent-history-rag-daemon supervise" in content or (
            "ai-agent-history-rag-daemon" in content and "<string>supervise</string>" in content
        )
