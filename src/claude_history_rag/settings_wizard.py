"""Interactive settings wizard for ai-agent-history-rag."""

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from claude_history_rag.config import Settings

ENV_PREFIX = "CLAUDE_HISTORY_RAG_"
MANAGED_FIELDS = [
    "db_path",
    "state_path",
    "projects_path",
    "codex_sessions_path",
    "codex_state_path",
    "gemini_sessions_path",
    "gemini_sessions_path",
    "gemini_state_path",
    "antigravity_sessions_path",
    "antigravity_state_path",
    "server_url",
    "machine_id",
    "client_name",
    "upload_interval_seconds",
    "upload_retry_count",
    "upload_retry_delay_seconds",
    "embedding_base_url",
    "embedding_model",
    "embedding_api_key",
    "log_level",
    "debounce_delay",
    "batch_size",
    "max_file_batch_size",
    "max_chunks_per_file",
    "gc_after_files",
    "defer_startup_indexing",
    "startup_indexing_delay_ms",
    "status_server_enabled",
    "status_server_host",
    "status_server_port",
    "status_refresh_interval",
    "auth_enabled",
    "server_psk",
    "client_psk",
    "auth_state_path",
    "client_auth_path",
]


def env_key(field_name: str) -> str:
    """Build the environment variable name for a field."""
    return f"{ENV_PREFIX}{field_name.upper()}"


def print_header(text: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n{text.center(60)}\n{line}\n")


def print_ok(text: str) -> None:
    print(f"[OK] {text}")


def print_warn(text: str) -> None:
    print(f"[WARN] {text}")


def print_error(text: str) -> None:
    print(f"[ERROR] {text}")


def prompt_string(prompt: str, default: str = "") -> str:
    """Prompt user for a string value."""
    default_display = f" (default: {default})" if default else ""
    response = input(f"{prompt}{default_display}: ").strip()
    return response if response else default


def validate_url(url: str) -> bool:
    """Validate that a string is a valid HTTP(S) URL."""
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def prompt_url(prompt: str, default: str = "", allow_empty: bool = True) -> str:
    """Prompt user for a URL with validation."""
    while True:
        url = prompt_string(prompt, default)
        if not url and allow_empty:
            return ""
        if validate_url(url):
            return url
        print_error("Please enter a valid URL (e.g., http://localhost:4680)")


def validate_machine_id(machine_id: str) -> bool:
    """Validate that a machine ID contains only safe characters."""
    return bool(re.match(r"^[a-zA-Z0-9._-]+$", machine_id))


def prompt_machine_id(prompt: str, default: str = "") -> str:
    """Prompt user for a machine ID with validation."""
    while True:
        machine_id = prompt_string(prompt, default)
        if not machine_id:
            return ""
        if validate_machine_id(machine_id):
            return machine_id
        print_error("Machine ID can only contain letters, numbers, dashes, underscores, and dots")


def prompt_int(prompt: str, default: int) -> int:
    """Prompt user for an integer value."""
    while True:
        response = prompt_string(prompt, str(default))
        try:
            return int(response)
        except ValueError:
            print_error("Please enter a valid integer")


def prompt_bool(prompt: str, default: bool) -> bool:
    """Prompt user for a yes/no answer."""
    default_str = "Y/n" if default else "y/N"
    while True:
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not response:
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print_error("Please enter 'y' or 'n'")


def prompt_choice(prompt: str, choices: list[str], default: str) -> str:
    """Prompt user to select a choice."""
    print(f"\n{prompt}")
    for idx, choice in enumerate(choices, start=1):
        marker = ">" if choice == default else " "
        print(f"  {marker} [{idx}] {choice}")
    while True:
        response = input(f"\nEnter choice [1-{len(choices)}] (default: {default}): ").strip()
        if not response:
            return default
        try:
            idx = int(response) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except ValueError:
            pass
        print_error("Please enter a valid choice number")


def get_project_dir() -> Path:
    """Get the project directory (where pyproject.toml is)."""
    current = Path(__file__).parent
    for _ in range(5):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def get_uv_path() -> str | None:
    """Find the uv binary path. Returns None if not found."""
    uv_path = shutil.which("uv")
    if uv_path:
        return uv_path

    home = Path.home()
    candidates = [
        home / ".local" / "bin" / "uv",
        home / ".cargo" / "bin" / "uv",
        Path("/usr/local/bin/uv"),
    ]

    if platform.system() == "Windows":
        candidates = [
            home / ".local" / "bin" / "uv.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "uv" / "uv.exe",
        ]

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    return None


def load_default_settings() -> Settings:
    """Load settings, falling back to defaults if validation fails."""
    try:
        return Settings()
    except Exception as e:
        print_warn(f"Existing configuration is invalid: {e}")
        return Settings.model_validate({})


def get_daemon_env_from_service() -> dict[str, str]:
    """Get daemon environment variables from the service configuration."""
    daemon_env: dict[str, str] = {}

    if sys.platform == "darwin":
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
        service_path = Path.home() / ".config/systemd/user/ai-agent-history-rag.service"
        if service_path.exists():
            try:
                content = service_path.read_text()
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("Environment="):
                        env_part = line[len("Environment=") :].strip().strip('"')
                        if "=" in env_part:
                            key, value = env_part.split("=", 1)
                            daemon_env[key] = value
            except Exception:
                pass

    return daemon_env


def parse_env_override(field_name: str, raw_value: str, current_value: Any) -> Any:
    """Parse a raw env var string into a typed value."""
    if raw_value is None:
        return current_value

    raw_value = raw_value.strip()
    if raw_value == "":
        return None if field_name in ("server_url",) else raw_value

    if isinstance(current_value, bool):
        return raw_value.lower() in ("1", "true", "yes", "on")
    if isinstance(current_value, int):
        try:
            return int(raw_value)
        except ValueError:
            return current_value
    if isinstance(current_value, Path):
        return Path(raw_value)
    return raw_value


def to_env_string(value: Any) -> str | None:
    """Convert a typed value to an env string."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def format_env_value(value: str) -> str:
    """Format a value for .env output."""
    if value == "":
        return ""
    if any(ch.isspace() for ch in value) or "#" in value or '"' in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def update_env_file(path: Path, env_vars: dict[str, str], managed_keys: set[str]) -> None:
    """Update or create a .env file with the managed keys."""
    lines: list[str] = []
    if path.exists():
        lines = path.read_text().splitlines()

    updated_lines: list[str] = []
    seen_keys: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            updated_lines.append(line)
            continue

        key = stripped.split("=", 1)[0].strip()
        if key in managed_keys:
            if key in env_vars:
                value = format_env_value(env_vars[key])
                updated_lines.append(f"{key}={value}")
            seen_keys.add(key)
        else:
            updated_lines.append(line)

    for key in sorted(managed_keys):
        if key in seen_keys or key not in env_vars:
            continue
        value = format_env_value(env_vars[key])
        updated_lines.append(f"{key}={value}")

    path.write_text("\n".join(updated_lines) + "\n")


def update_launchd_env(env_vars: dict[str, str], managed_keys: set[str]) -> bool:
    plist_path = Path.home() / "Library/LaunchAgents/com.ai-agent-history-rag.daemon.plist"
    if not plist_path.exists():
        return False
    try:
        import plistlib

        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)
        current_env = plist.get("EnvironmentVariables", {})
        for key in list(current_env.keys()):
            if key in managed_keys:
                current_env.pop(key, None)
        for key, value in env_vars.items():
            current_env[key] = value
        if current_env:
            plist["EnvironmentVariables"] = current_env
        else:
            plist.pop("EnvironmentVariables", None)
        with open(plist_path, "wb") as f:
            plistlib.dump(plist, f)
        return True
    except Exception as e:
        print_warn(f"Failed to update launchd plist: {e}")
        return False


def update_systemd_env(env_vars: dict[str, str], managed_keys: set[str]) -> bool:
    service_path = Path.home() / ".config/systemd/user/ai-agent-history-rag.service"
    if not service_path.exists():
        return False
    try:
        lines = service_path.read_text().splitlines()
        filtered: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Environment="):
                env_part = stripped[len("Environment=") :].strip().strip('"')
                key = env_part.split("=", 1)[0] if "=" in env_part else env_part
                if key in managed_keys:
                    continue
            filtered.append(line)

        env_lines = []
        for key, value in env_vars.items():
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            env_lines.append(f'Environment="{key}={escaped}"')

        service_start = None
        service_end = None
        for idx, line in enumerate(filtered):
            stripped = line.strip()
            if stripped == "[Service]":
                service_start = idx
                continue
            if service_start is not None and stripped.startswith("[") and stripped.endswith("]"):
                service_end = idx
                break
        if service_start is None:
            filtered.append("[Service]")
            service_start = len(filtered) - 1
            service_end = len(filtered)
        if service_end is None:
            service_end = len(filtered)

        insertion_index = service_end
        filtered[insertion_index:insertion_index] = env_lines
        service_path.write_text("\n".join(filtered) + "\n")
        return True
    except Exception as e:
        print_warn(f"Failed to update systemd service: {e}")
        return False


def update_windows_env(env_vars: dict[str, str], managed_keys: set[str]) -> bool:
    if sys.platform != "win32":
        return False
    any_updates = False
    for key in sorted(managed_keys):
        value = env_vars.get(key, "")
        result = subprocess.run(["setx", key, value], capture_output=True, text=True)
        if result.returncode != 0:
            print_warn(f"Failed to set {key}: {result.stderr.strip()}")
        else:
            any_updates = True
    return any_updates


def update_daemon_env(env_vars: dict[str, str], managed_keys: set[str]) -> None:
    if sys.platform == "darwin":
        if update_launchd_env(env_vars, managed_keys):
            print_ok("Updated launchd daemon environment")
        else:
            print_warn("launchd service not found; daemon env not updated")
    elif sys.platform == "linux":
        if update_systemd_env(env_vars, managed_keys):
            print_ok("Updated systemd daemon environment")
        else:
            print_warn("systemd service not found; daemon env not updated")
    elif sys.platform == "win32":
        if update_windows_env(env_vars, managed_keys):
            print_ok("Updated user environment variables")
        else:
            print_warn("Windows environment update skipped")
    else:
        print_warn("Unsupported platform for service updates")


def restart_daemon(project_dir: Path) -> None:
    if sys.platform == "darwin":
        plist_path = Path.home() / "Library/LaunchAgents/com.ai-agent-history-rag.daemon.plist"
        if plist_path.exists():
            subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
            result = subprocess.run(["launchctl", "load", str(plist_path)], check=False)
            if result.returncode == 0:
                print_ok("Restarted launchd daemon")
            else:
                print_warn("Failed to restart launchd daemon")
            return
    elif sys.platform == "linux":
        service_path = Path.home() / ".config/systemd/user/ai-agent-history-rag.service"
        if service_path.exists():
            result = subprocess.run(
                ["systemctl", "--user", "restart", "ai-agent-history-rag"],
                check=False,
            )
            if result.returncode == 0:
                print_ok("Restarted systemd daemon")
            else:
                print_warn("Failed to restart systemd daemon")
            return
    elif sys.platform == "win32":
        task_name = "AIAgentHistoryRAG"
        query = subprocess.run(["schtasks", "/Query", "/TN", task_name], capture_output=True)
        if query.returncode == 0:
            subprocess.run(["schtasks", "/End", "/TN", task_name], check=False)
            result = subprocess.run(["schtasks", "/Run", "/TN", task_name], check=False)
            if result.returncode == 0:
                print_ok("Restarted Windows scheduled task")
            else:
                print_warn("Failed to restart Windows scheduled task")
            return

    uv_path = get_uv_path()
    if uv_path:
        result = subprocess.run(
            [
                uv_path,
                "--directory",
                str(project_dir),
                "run",
                "ai-agent-history-rag-daemon",
                "restart",
            ],
            check=False,
        )
        if result.returncode == 0:
            print_ok("Restarted daemon via uv")
            return

    result = subprocess.run(["ai-agent-history-rag-daemon", "restart"], check=False)
    if result.returncode == 0:
        print_ok("Restarted daemon")
    else:
        print_warn("Failed to restart daemon; restart it manually if needed")


def run_wizard() -> int:
    """Run the settings wizard."""
    print_header("AI Agent History RAG Settings")

    project_dir = get_project_dir()
    env_file = project_dir / ".env"

    base_settings = load_default_settings()
    current_values = base_settings.model_dump()
    daemon_env = get_daemon_env_from_service()

    for field in MANAGED_FIELDS:
        key = env_key(field)
        if key in daemon_env:
            current_values[field] = parse_env_override(
                field, daemon_env[key], current_values[field]
            )

    print_header("Paths")
    db_path = Path(prompt_string("Database path", str(current_values["db_path"])))
    state_path = Path(prompt_string("State file path", str(current_values["state_path"])))
    projects_path = Path(prompt_string("Projects path", str(current_values["projects_path"])))
    codex_sessions_path = Path(
        prompt_string("Codex sessions path", str(current_values["codex_sessions_path"]))
    )
    codex_state_path = Path(
        prompt_string("Codex state file path", str(current_values["codex_state_path"]))
    )
    gemini_sessions_path = Path(
        prompt_string("Gemini sessions path", str(current_values["gemini_sessions_path"]))
    )
    gemini_state_path = Path(
        prompt_string("Gemini state file path", str(current_values["gemini_state_path"]))
    )
    antigravity_sessions_path = Path(
        prompt_string("Antigravity sessions path", str(current_values["antigravity_sessions_path"]))
    )
    antigravity_state_path = Path(
        prompt_string("Antigravity state file path", str(current_values["antigravity_state_path"]))
    )

    print_header("Client/Server Mode")
    server_url = prompt_url(
        "Central server URL (leave empty for server/standalone)",
        str(current_values.get("server_url") or ""),
    )
    machine_id = prompt_machine_id("Machine ID", str(current_values["machine_id"]))
    client_name = prompt_string(
        "Client name (optional label)", str(current_values.get("client_name") or "")
    )
    upload_interval_seconds = prompt_int(
        "Upload interval seconds", int(current_values["upload_interval_seconds"])
    )
    upload_retry_count = prompt_int("Upload retry count", int(current_values["upload_retry_count"]))
    upload_retry_delay_seconds = prompt_int(
        "Upload retry delay seconds", int(current_values["upload_retry_delay_seconds"])
    )

    print_header("Embedding Settings")
    embedding_base_url = prompt_url(
        "Embedding base URL", str(current_values["embedding_base_url"]), allow_empty=False
    )
    embedding_model = prompt_string("Embedding model", str(current_values["embedding_model"]))
    embedding_api_key = prompt_string(
        "Embedding API key (optional)", str(current_values.get("embedding_api_key") or "")
    )

    print_header("General Settings")
    log_level = prompt_choice(
        "Log level",
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        str(current_values["log_level"]),
    )
    debounce_delay = prompt_int("Debounce delay (ms)", int(current_values["debounce_delay"]))
    batch_size = prompt_int("Embedding batch size", int(current_values["batch_size"]))
    max_file_batch_size = prompt_int(
        "Max file batch size", int(current_values["max_file_batch_size"])
    )
    max_chunks_per_file = prompt_int(
        "Max chunks per file", int(current_values["max_chunks_per_file"])
    )
    gc_after_files = prompt_bool(
        "Run garbage collection after file batches", bool(current_values["gc_after_files"])
    )
    defer_startup_indexing = prompt_bool(
        "Defer startup indexing", bool(current_values["defer_startup_indexing"])
    )
    startup_indexing_delay_ms = prompt_int(
        "Startup indexing delay (ms)", int(current_values["startup_indexing_delay_ms"])
    )

    print_header("Status Server")
    status_server_enabled = prompt_bool(
        "Enable status server", bool(current_values["status_server_enabled"])
    )
    status_server_host = prompt_string(
        "Status server host", str(current_values["status_server_host"])
    )
    status_server_port = prompt_int("Status server port", int(current_values["status_server_port"]))
    status_refresh_interval = prompt_int(
        "Status refresh interval (seconds)", int(current_values["status_refresh_interval"])
    )

    print_header("Auth (PSK)")
    auth_enabled = prompt_bool("Require PSK authentication", bool(current_values["auth_enabled"]))
    server_psk = prompt_string(
        "Server PSK (optional override)",
        str(current_values.get("server_psk") or ""),
    )
    client_psk = prompt_string(
        "Client PSK (optional override)",
        str(current_values.get("client_psk") or ""),
    )
    auth_state_path = Path(
        prompt_string("Auth state path", str(current_values["auth_state_path"]))
    )
    client_auth_path = Path(
        prompt_string("Client auth path", str(current_values["client_auth_path"]))
    )

    new_values = {
        "db_path": db_path,
        "state_path": state_path,
        "projects_path": projects_path,
        "codex_sessions_path": codex_sessions_path,
        "codex_state_path": codex_state_path,
        "gemini_sessions_path": gemini_sessions_path,
        "gemini_state_path": gemini_state_path,
        "antigravity_sessions_path": antigravity_sessions_path,
        "antigravity_state_path": antigravity_state_path,
        "server_url": server_url or None,
        "machine_id": machine_id,
        "client_name": client_name,
        "upload_interval_seconds": upload_interval_seconds,
        "upload_retry_count": upload_retry_count,
        "upload_retry_delay_seconds": upload_retry_delay_seconds,
        "embedding_base_url": embedding_base_url,
        "embedding_model": embedding_model,
        "embedding_api_key": embedding_api_key,
        "log_level": log_level,
        "debounce_delay": debounce_delay,
        "batch_size": batch_size,
        "max_file_batch_size": max_file_batch_size,
        "max_chunks_per_file": max_chunks_per_file,
        "gc_after_files": gc_after_files,
        "defer_startup_indexing": defer_startup_indexing,
        "startup_indexing_delay_ms": startup_indexing_delay_ms,
        "status_server_enabled": status_server_enabled,
        "status_server_host": status_server_host,
        "status_server_port": status_server_port,
        "status_refresh_interval": status_refresh_interval,
        "auth_enabled": auth_enabled,
        "server_psk": server_psk,
        "client_psk": client_psk,
        "auth_state_path": auth_state_path,
        "client_auth_path": client_auth_path,
    }

    try:
        validated = Settings.model_validate(new_values)
    except ValidationError as e:
        print_error("Settings validation failed:")
        print(e)
        return 1

    env_vars: dict[str, str] = {}
    for field in MANAGED_FIELDS:
        value = getattr(validated, field)
        env_value = to_env_string(value)
        if env_value is None or env_value == "":
            continue
        env_vars[env_key(field)] = env_value

    managed_keys = {env_key(field) for field in MANAGED_FIELDS}
    update_env_file(env_file, env_vars, managed_keys)
    print_ok(f"Updated {env_file}")

    update_daemon_env(env_vars, managed_keys)
    restart_daemon(project_dir)

    print_ok("Settings applied")
    print("If you use an MCP client, restart it to pick up new settings.")
    return 0


def main() -> int:
    """Entry point."""
    try:
        return run_wizard()
    except KeyboardInterrupt:
        print("\nSettings update cancelled.")
        return 1
    except Exception as e:
        print_error(f"Settings update failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
