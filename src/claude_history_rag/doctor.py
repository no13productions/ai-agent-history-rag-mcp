"""Diagnostic tool for ai-agent-history-rag.

Checks system health, configuration, connectivity, and provides troubleshooting info.
"""

import os
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Import config - this will fail early if there are config issues
# Type annotations for module-level variables
CONFIG_LOADED: bool = False
CONFIG_ERROR: str | None = None

try:
    from claude_history_rag.config import Settings
    from claude_history_rag.config import settings as _settings

    settings: Settings | None = _settings
    CONFIG_LOADED = True
    CONFIG_ERROR = None
except Exception as e:
    settings = None
    CONFIG_LOADED = False
    CONFIG_ERROR = str(e)


# ANSI colors
def supports_color() -> bool:
    """Check if terminal supports ANSI colors."""
    # Respect NO_COLOR standard (https://no-color.org/)
    if os.environ.get("NO_COLOR"):
        return False
    # Respect FORCE_COLOR for CI environments
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


if supports_color():
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
else:
    GREEN = YELLOW = RED = BLUE = BOLD = DIM = RESET = ""


def print_header(text: str) -> None:
    print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}{BLUE}{text:^60}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 60}{RESET}\n")


def print_ok(text: str) -> None:
    print(f"  {GREEN}✓{RESET} {text}")


def print_warn(text: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {text}")


def print_fail(text: str) -> None:
    print(f"  {RED}✗{RESET} {text}")


def print_info(text: str) -> None:
    print(f"  {DIM}→{RESET} {text}")


def check_port_in_use(port: int, host: str = "127.0.0.1") -> tuple[bool, str | None]:
    """Check if a port is in use and try to identify the process."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex((host, port))
        if result == 0:
            # Port is in use, try to find what's using it
            try:
                if sys.platform == "darwin":
                    output = subprocess.check_output(
                        ["lsof", "-i", f":{port}", "-t"], stderr=subprocess.DEVNULL, text=True
                    )
                    pids = output.strip().split("\n")
                    if pids and pids[0]:
                        # Get process name
                        ps_output = subprocess.check_output(
                            ["ps", "-p", pids[0], "-o", "comm="],
                            stderr=subprocess.DEVNULL,
                            text=True,
                        )
                        return True, ps_output.strip()
                elif sys.platform == "linux":
                    output = subprocess.check_output(
                        ["ss", "-tlnp", f"sport = :{port}"], stderr=subprocess.DEVNULL, text=True
                    )
                    return True, output.strip() if output else None
                elif sys.platform == "win32":
                    # Windows: use netstat
                    output = subprocess.check_output(
                        ["netstat", "-ano", "-p", "TCP"], stderr=subprocess.DEVNULL, text=True
                    )
                    for line in output.split("\n"):
                        if f":{port}" in line and "LISTENING" in line:
                            parts = line.split()
                            if parts:
                                pid = parts[-1]
                                # Get process name from PID
                                try:
                                    tasklist = subprocess.check_output(
                                        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"],
                                        stderr=subprocess.DEVNULL,
                                        text=True,
                                    )
                                    lines = tasklist.strip().split("\n")
                                    if len(lines) > 1:
                                        # Parse CSV: "process.exe","PID",...
                                        return True, lines[1].split(",")[0].strip('"')
                                except Exception:
                                    return True, f"PID {pid}"
                    return True, None
            except Exception:
                pass
            return True, None
        return False, None
    except Exception:
        return False, None
    finally:
        sock.close()


def check_process_running(pid: int) -> bool:
    """Check if a process with given PID is running (cross-platform)."""
    if sys.platform == "win32":
        try:
            output = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            # Parse CSV to avoid false positives (PID 123 matching 1234)
            for line in output.strip().split("\n")[1:]:  # Skip header
                parts = line.split(",")
                if len(parts) >= 2:
                    # CSV format: "process.exe","PID",...
                    csv_pid = parts[1].strip('"')
                    if csv_pid == str(pid):
                        return True
            return False
        except Exception:
            return False
    else:
        # Unix: use kill -0
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # PermissionError means process exists but we can't signal it
            return True


def check_url_reachable(url: str, timeout: int = 5) -> tuple[bool, int | None, str | None]:
    """Check if a URL is reachable. Returns (reachable, status_code, error)."""
    if not HTTPX_AVAILABLE:
        return False, None, "httpx not installed"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            return True, response.status_code, None
    except httpx.ConnectError as e:
        return False, None, f"Connection refused: {e}"
    except httpx.ConnectTimeout:
        return False, None, "Connection timed out"
    except Exception as e:
        return False, None, str(e)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return datetime.fromisoformat(value)
    except Exception:
        return None


def get_client_state_summary() -> dict[str, object]:
    """Read client_state.json and return summary info."""
    if not CONFIG_LOADED or settings is None:
        return {}
    state_path = settings.db_path.parent / "client_state.json"
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text())
    except Exception:
        return {}

    pending = data.get("pending_uploads", []) or []
    last_sync_raw = data.get("last_server_sync")
    last_sync = _parse_iso_datetime(last_sync_raw)
    connected = data.get("connected")

    summary: dict[str, object] = {
        "pending_uploads": len(pending),
        "last_server_sync": last_sync_raw,
        "connected": connected,
    }

    if last_sync:
        summary["last_server_sync_age_min"] = int(
            (datetime.now(timezone.utc) - last_sync).total_seconds() / 60
        )

    return summary


def get_recent_logs(log_file: Path, lines: int = 20) -> list[str]:
    """Get recent log lines."""
    if not log_file.exists():
        return []
    try:
        with open(log_file) as f:
            all_lines = f.readlines()
            return all_lines[-lines:]
    except Exception:
        return []


def check_daemon_status() -> tuple[bool, int | None]:
    """Check if daemon is running (cross-platform)."""
    if not CONFIG_LOADED or settings is None:
        return False, None

    pid_file = settings.db_path.parent / "daemon.pid"
    if not pid_file.exists():
        return False, None

    try:
        pid = int(pid_file.read_text().strip())
        if check_process_running(pid):
            return True, pid
        return False, None
    except ValueError:
        return False, None


def get_daemon_env_from_service() -> dict[str, str]:
    """Get daemon environment variables from the service configuration (cross-platform)."""
    daemon_env: dict[str, str] = {}

    if sys.platform == "darwin":
        # macOS: read launchd plist
        plist_path = Path.home() / "Library/LaunchAgents/com.ai-agent-history-rag.daemon.plist"
        if plist_path.exists():
            try:
                import plistlib

                with open(plist_path, "rb") as f:
                    plist = plistlib.load(f)
                daemon_env = plist.get("EnvironmentVariables", {})
            except Exception:
                pass

    elif sys.platform == "linux":
        # Linux: read systemd service file
        service_path = Path.home() / ".config/systemd/user/ai-agent-history-rag.service"
        if service_path.exists():
            try:
                content = service_path.read_text()
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("Environment="):
                        # Format: Environment="KEY=value" or Environment=KEY=value
                        env_part = line[len("Environment=") :]
                        # Remove quotes if present
                        env_part = env_part.strip('"')
                        if "=" in env_part:
                            key, value = env_part.split("=", 1)
                            daemon_env[key] = value
            except Exception:
                pass

    elif sys.platform == "win32":
        # Windows: scheduled tasks don't store env vars, check user env
        # We can't easily read scheduled task env, so just note this
        pass

    return daemon_env


def run_doctor() -> int:
    """Run all diagnostic checks."""
    print_header("AI Agent History RAG Doctor")

    all_ok = True

    # ==========================================================================
    # Configuration Check
    # ==========================================================================
    print(f"{BOLD}Configuration{RESET}")

    if not CONFIG_LOADED or settings is None:
        print_fail(f"Failed to load configuration: {CONFIG_ERROR}")
        print_info("Fix configuration issues before proceeding")
        return 1

    # Type narrowing for type checker - settings is not None after this point
    assert settings is not None
    print_ok("Configuration loaded successfully")

    # Show mode
    if settings.server_url:
        print_info(f"Mode: CLIENT (connecting to {settings.server_url})")
        print_info(f"Machine ID: {settings.machine_id}")
    else:
        print_info("Mode: SERVER/STANDALONE (local processing)")
        print_info(f"Embedding URL: {settings.embedding_base_url}")
        print_info(f"Embedding Model: {settings.embedding_model}")

    print_info(f"Storage Backend: {settings.storage_backend}")
    if settings.storage_backend == "sqlite":
        print_info(f"SQLite DB: {settings.sqlite_db_path}")
    elif settings.storage_backend == "qdrant":
        print_info(f"Qdrant URL: {settings.qdrant_url}")
        print_info(f"Qdrant Collection: {settings.qdrant_collection}")

    print_info(f"Projects: {settings.projects_path}")
    print_info(f"Codex Sessions: {settings.codex_sessions_path}")
    print_info(f"Gemini Sessions: {settings.gemini_sessions_path}")
    print_info(f"Antigravity Sessions: {settings.antigravity_sessions_path}")
    print_info(f"Status Server: {settings.status_server_host}:{settings.status_server_port}")

    # ==========================================================================
    # Daemon Status
    # ==========================================================================
    print(f"\n{BOLD}Daemon Status{RESET}")

    running, pid = check_daemon_status()
    if running:
        print_ok(f"Daemon is running (PID {pid})")
    else:
        print_fail("Daemon is NOT running")
        print_info("Start with: uv run ai-agent-history-rag-daemon start")
        all_ok = False

    # Check PID file
    pid_file = settings.db_path.parent / "daemon.pid"
    if pid_file.exists() and not running:
        print_warn(f"Stale PID file exists: {pid_file}")
        print_info("Remove with: rm ~/.claude-history-rag/daemon.pid")

    # ==========================================================================
    # Port Status
    # ==========================================================================
    # Determine if daemon is in client mode (from service config or env)
    daemon_env = get_daemon_env_from_service()
    daemon_is_client = "CLAUDE_HISTORY_RAG_SERVER_URL" in daemon_env
    # Also check current env
    if not daemon_is_client:
        daemon_is_client = bool(os.environ.get("CLAUDE_HISTORY_RAG_SERVER_URL"))

    # Determine daemon status server host (service config overrides current shell)
    daemon_status_host = daemon_env.get("CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST")
    if not daemon_status_host:
        daemon_status_host = os.environ.get("CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST")
    if not daemon_status_host:
        daemon_status_host = settings.status_server_host

    # Warn if current shell config differs from daemon service config
    daemon_host_env = daemon_env.get("CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST")
    shell_host_env = os.environ.get("CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST")
    if daemon_host_env and daemon_host_env != settings.status_server_host:
        print_warn(
            "Status server host differs between current shell and daemon service configuration"
        )
        print_info(f"Current shell: {settings.status_server_host} | Daemon service: {daemon_host_env}")

    # Warn if shell has host override but daemon service does not
    if not daemon_host_env and shell_host_env:
        print_warn(
            "Daemon service does not set CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST (current shell does)"
        )
        print_info(
            "The daemon will default to 127.0.0.1 unless the service config is updated"
        )

    print(f"\n{BOLD}Port Status{RESET}")

    port = settings.status_server_port

    if daemon_is_client:
        print_info(f"Port {port} check skipped (client mode doesn't run status server)")
    else:
        if daemon_status_host in ("127.0.0.1", "localhost"):
            print_warn("Status server is bound to localhost (127.0.0.1)")
            print_info(
                "If this is a CENTRAL SERVER, set CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST=0.0.0.0"
            )
            print_info(
                "If this is a standalone install, 127.0.0.1 is expected for security"
            )
        in_use, process = check_port_in_use(port)
        if in_use:
            if running and process and "claude" in process.lower():
                print_ok(f"Port {port} is in use by daemon ({process})")
            elif running:
                print_ok(f"Port {port} is in use (likely daemon)")
            else:
                print_fail(f"Port {port} is in use by another process: {process or 'unknown'}")
                # Platform-specific commands
                if sys.platform == "darwin":
                    print_info(f"Find process: lsof -i :{port}")
                    print_info(f"Kill process: kill $(lsof -t -i :{port})")
                elif sys.platform == "linux":
                    print_info(f"Find process: ss -tlnp 'sport = :{port}'")
                    print_info(f"Kill process: fuser -k {port}/tcp")
                elif sys.platform == "win32":
                    print_info(f"Find process: netstat -ano | findstr :{port}")
                    print_info("Kill process: taskkill /PID <pid> /F")
                all_ok = False
        else:
            if running:
                print_warn(f"Port {port} is not listening (daemon may still be starting)")
            else:
                print_info(f"Port {port} is available")

    # ==========================================================================
    # Service Connectivity
    # ==========================================================================
    print(f"\n{BOLD}Service Connectivity{RESET}")

    # Get central server URL from daemon config if in client mode
    central_server_url = daemon_env.get("CLAUDE_HISTORY_RAG_SERVER_URL")
    if not central_server_url:
        central_server_url = os.environ.get("CLAUDE_HISTORY_RAG_SERVER_URL")

    if daemon_is_client and central_server_url:
        # Client mode - check central server
        print(f"  Checking central server at {central_server_url}...")
        reachable, status, error = check_url_reachable(f"{central_server_url}/health")
        if reachable:
            print_ok(f"Central server is reachable (HTTP {status})")
        else:
            print_fail(f"Cannot reach central server: {error}")
            print_info("Make sure the central server is running and accessible")
            all_ok = False
        print_info("(Client mode: no local embedding server needed)")
    else:
        # Server mode - check local status server
        status_url = f"http://127.0.0.1:{port}/health"
        print(f"  Checking status server at {status_url}...")
        reachable, status, error = check_url_reachable(status_url)
        if reachable:
            print_ok(f"Status server is responding (HTTP {status})")
        else:
            if running:
                print_warn(f"Status server not responding: {error}")
                print_info("The daemon may still be initializing")
                print_info("If this persists, check logs: ~/.claude-history-rag/daemon.log")
            else:
                print_info("Status server not running (daemon not started)")

        # Check embedding server
        embedding_url = f"{settings.embedding_base_url}/models"
        print(f"  Checking embedding server at {settings.embedding_base_url}...")
        reachable, status, error = check_url_reachable(embedding_url)
        if reachable:
            print_ok(f"Embedding server is responding (HTTP {status})")
        else:
            print_fail(f"Cannot reach embedding server: {error}")
            print_info("Start Ollama with: ollama serve")
            print_info("Or check your CLAUDE_HISTORY_RAG_EMBEDDING_BASE_URL setting")
            all_ok = False
    if settings.storage_backend == "qdrant" and settings.qdrant_url:
        print(f"  Checking Qdrant at {settings.qdrant_url}...")
        reachable, status, error = check_url_reachable(settings.qdrant_url)
        if reachable:
            print_ok(f"Qdrant is reachable (HTTP {status})")
        else:
            print_fail(f"Cannot reach Qdrant: {error}")
            print_info("Make sure Qdrant is running")
            all_ok = False
    # ==========================================================================
    # Client Queue (client mode only)
    # ==========================================================================
    if daemon_is_client:
        print(f"\n{BOLD}Client Queue{RESET}")
        summary = get_client_state_summary()
        if not summary:
            print_info("No client state found")
        else:
            pending_count = summary.get("pending_uploads", 0)
            last_sync = summary.get("last_server_sync")
            age_min = summary.get("last_server_sync_age_min")
            connected = summary.get("connected")

            print_info(f"Pending uploads: {pending_count}")
            if last_sync:
                if age_min is not None:
                    print_info(f"Last server sync: {last_sync} ({age_min} min ago)")
                else:
                    print_info(f"Last server sync: {last_sync}")
            else:
                print_info("Last server sync: unknown")

            if connected is True:
                print_ok("Connection state: connected")
            elif connected is False:
                print_warn("Connection state: disconnected")
            else:
                print_info("Connection state: unknown")

            if pending_count and age_min is not None and age_min > 30:
                print_warn("Pending uploads are not clearing; retry loop may not be running")

    # ==========================================================================
    # File System
    # ==========================================================================
    print(f"\n{BOLD}File System{RESET}")

    # Check data directory
    data_dir = settings.db_path.parent
    if data_dir.exists():
        print_ok(f"Data directory exists: {data_dir}")
    else:
        print_info(f"Data directory will be created: {data_dir}")

    # Check projects directory
    if settings.projects_path.exists():
        print_ok(f"Projects directory exists: {settings.projects_path}")
        # Count JSONL files
        jsonl_files = list(settings.projects_path.rglob("*.jsonl"))
        print_info(f"Found {len(jsonl_files)} conversation files")
    else:
        print_fail(f"Projects directory not found: {settings.projects_path}")
        print_info("Claude Code stores conversations in ~/.claude/projects/")
        all_ok = False

    # Check Codex sessions directory
    if settings.codex_sessions_path.exists():
        print_ok(f"Codex sessions directory exists: {settings.codex_sessions_path}")
        codex_files = list(settings.codex_sessions_path.rglob("*.jsonl"))
        print_info(f"Found {len(codex_files)} Codex session files")
    else:
        print_fail(f"Codex sessions directory not found: {settings.codex_sessions_path}")
        print_info("Codex stores sessions in ~/.codex/sessions/")
        all_ok = False

    if settings.gemini_sessions_path.exists():
        print_ok(f"Gemini sessions directory exists: {settings.gemini_sessions_path}")
        gemini_files = list(settings.gemini_sessions_path.rglob("*.json"))
        print_info(f"Found {len(gemini_files)} Gemini session files")
    else:
        print_fail(f"Gemini sessions directory not found: {settings.gemini_sessions_path}")
        print_info("Gemini stores sessions in ~/.gemini/tmp/")
        all_ok = False

    if settings.antigravity_sessions_path.exists():
        print_ok(f"Antigravity sessions directory exists: {settings.antigravity_sessions_path}")
        antigravity_files = list(settings.antigravity_sessions_path.rglob("*.pb"))
        print_info(f"Found {len(antigravity_files)} Antigravity history files")
    else:
        print_fail(
            f"Antigravity sessions directory not found: {settings.antigravity_sessions_path}"
        )
        print_info("Antigravity stores sessions in ~/.gemini/antigravity/conversations/")
        all_ok = False

    # Check database
    if settings.storage_backend == "sqlite":
        if settings.sqlite_db_path.exists():
            print_ok(f"SQLite DB exists: {settings.sqlite_db_path}")
            # Try to get size
            try:
                size_mb = settings.sqlite_db_path.stat().st_size / (1024 * 1024)
                print_info(f"Database size: {size_mb:.1f} MB")
            except Exception:
                pass
        else:
            print_info(f"SQLite DB not yet created: {settings.sqlite_db_path}")
    elif settings.storage_backend == "qdrant":
        print_info(f"Storage is remote (Qdrant): {settings.qdrant_url}")

    # ==========================================================================
    # Logs
    # ==========================================================================
    print(f"\n{BOLD}Recent Logs{RESET}")

    log_file = settings.db_path.parent / "daemon.log"
    if log_file.exists():
        print_info(f"Log file: {log_file}")
        lines = get_recent_logs(log_file, 10)
        if lines:
            print_info("Last 10 log entries:")
            for line in lines:
                line = line.strip()
                truncated = line[:100]
                suffix = "..." if len(line) > 100 else ""
                if "ERROR" in line:
                    print(f"    {RED}{truncated}{suffix}{RESET}")
                elif "WARNING" in line:
                    print(f"    {YELLOW}{truncated}{suffix}{RESET}")
                else:
                    print(f"    {DIM}{truncated}{suffix}{RESET}")
        else:
            print_info("Log file is empty")
    else:
        print_info("No log file yet (daemon hasn't run)")

    # Check launchd stderr log (macOS)
    launchd_log = data_dir / "launchd-stderr.log"
    if launchd_log.exists():
        lines = get_recent_logs(launchd_log, 5)
        if lines:
            print_info(f"\nLaunchd stderr ({launchd_log}):")
            for line in lines:
                line = line.strip()
                if line:
                    truncated = line[:100]
                    suffix = "..." if len(line) > 100 else ""
                    print(f"    {RED}{truncated}{suffix}{RESET}")

    # ==========================================================================
    # Environment Variables (current shell)
    # ==========================================================================
    print(f"\n{BOLD}Environment Variables (current shell){RESET}")

    env_vars = [
        "CLAUDE_HISTORY_RAG_SERVER_URL",
        "CLAUDE_HISTORY_RAG_MACHINE_ID",
        "CLAUDE_HISTORY_RAG_EMBEDDING_BASE_URL",
        "CLAUDE_HISTORY_RAG_EMBEDDING_MODEL",
        "CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST",
        "CLAUDE_HISTORY_RAG_STATUS_SERVER_PORT",
    ]

    found_any = False
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print_info(f"{var}={value}")
            found_any = True

    if not found_any:
        print_info("No environment variables set in current shell")
        print_info("(Note: daemon may have different env vars via launchd/systemd)")

    # Show daemon's configured env vars from service file (cross-platform)
    if daemon_env:
        if sys.platform == "darwin":
            source = "launchd plist"
        elif sys.platform == "linux":
            source = "systemd service"
        else:
            source = "service config"

        print(f"\n{BOLD}Daemon Environment (from {source}){RESET}")
        for k, v in daemon_env.items():
            print_info(f"{k}={v}")
        # Determine actual mode
        if "CLAUDE_HISTORY_RAG_SERVER_URL" in daemon_env:
            print_ok(
                f"Daemon is configured as CLIENT → {daemon_env['CLAUDE_HISTORY_RAG_SERVER_URL']}"
            )
        else:
            print_info("Daemon is configured as SERVER/STANDALONE")

    # ==========================================================================
    # Service Installation
    # ==========================================================================
    print(f"\n{BOLD}Service Installation{RESET}")

    # Check launchd (macOS)
    if sys.platform == "darwin":
        plist_path = Path.home() / "Library/LaunchAgents/com.ai-agent-history-rag.daemon.plist"
        if plist_path.exists():
            print_ok(f"Launchd plist installed: {plist_path}")
            # Check if loaded
            try:
                result = subprocess.run(
                    ["launchctl", "list", "com.ai-agent-history-rag.daemon"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print_ok("Service is loaded in launchd")
                else:
                    print_warn("Service plist exists but not loaded")
                    print_info(f"Load with: launchctl load {plist_path}")
            except Exception:
                pass
        else:
            print_info("Launchd service not installed")
            print_info("Install with: uv run ai-agent-history-rag-install")

    # Check systemd (Linux)
    elif sys.platform == "linux":
        service_path = Path.home() / ".config/systemd/user/ai-agent-history-rag.service"
        if service_path.exists():
            print_ok(f"Systemd service installed: {service_path}")
            try:
                result = subprocess.run(
                    ["systemctl", "--user", "is-active", "ai-agent-history-rag"],
                    capture_output=True,
                    text=True,
                )
                status = result.stdout.strip()
                if status == "active":
                    print_ok("Service is active")
                elif status == "inactive":
                    print_warn("Service is inactive")
                    print_info("Start with: systemctl --user start ai-agent-history-rag")
                elif status == "failed":
                    print_fail("Service has failed")
                    print_info("Check logs: journalctl --user -u ai-agent-history-rag -n 50")
                else:
                    print_warn(f"Service status: {status}")
            except Exception:
                pass

            # Check if enabled
            try:
                result = subprocess.run(
                    ["systemctl", "--user", "is-enabled", "ai-agent-history-rag"],
                    capture_output=True,
                    text=True,
                )
                if result.stdout.strip() == "enabled":
                    print_ok("Service is enabled (starts on boot)")
                else:
                    print_info("Service is not enabled for auto-start")
                    print_info("Enable with: systemctl --user enable ai-agent-history-rag")
            except Exception:
                pass
        else:
            print_info("Systemd service not installed")
            print_info("Install with: uv run ai-agent-history-rag-install")

    # Check Windows scheduled task
    elif sys.platform == "win32":
        try:
            result = subprocess.run(
                ["schtasks", "/Query", "/TN", "AIAgentHistoryRAG", "/FO", "CSV"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print_ok("Windows scheduled task installed: AIAgentHistoryRAG")
                if "Running" in result.stdout:
                    print_ok("Task is running")
                elif "Ready" in result.stdout:
                    print_info("Task is ready (will run on next logon)")
                else:
                    print_warn("Task status unknown")
            else:
                print_info("Windows scheduled task not installed")
                print_info("Install with: uv run ai-agent-history-rag-install")
        except Exception:
            print_info("Could not check Windows scheduled task")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print_header("Summary")

    if all_ok:
        print(f"{GREEN}All checks passed!{RESET}")
        if not running:
            print(f"\n{YELLOW}Daemon is not running. Start it with:{RESET}")
            print("  uv run ai-agent-history-rag-daemon start")
        return 0
    else:
        print(f"{RED}Some checks failed. See above for details.{RESET}")
        print(f"\n{YELLOW}Common fixes:{RESET}")
        print("  1. Start the daemon: uv run ai-agent-history-rag-daemon start")
        print("  2. Start Ollama: ollama serve")
        # Platform-specific log and port commands
        if sys.platform == "win32":
            print("  3. Check logs: type %USERPROFILE%\\.claude-history-rag\\daemon.log")
            print("  4. Kill stuck port: netstat -ano | findstr :4680, then taskkill /PID <pid> /F")
        else:
            print("  3. Check logs: cat ~/.claude-history-rag/daemon.log")
            if sys.platform == "darwin":
                print("  4. Kill stuck port: kill $(lsof -t -i :4680)")
            else:  # Linux
                print("  4. Kill stuck port: fuser -k 4680/tcp")
        print("  5. Re-run installer: uv run ai-agent-history-rag-install")
        return 1


def main() -> int:
    """Entry point."""
    try:
        return run_doctor()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 1
    except Exception as e:
        print(f"{RED}Doctor failed: {e}{RESET}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
