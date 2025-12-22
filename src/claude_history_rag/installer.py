"""Cross-platform installation wizard for ai-agent-history-rag.

Supports:
- MCP server installation into Claude Desktop, Claude Code, Cursor, VS Code, Gemini, Codex
- Daemon service installation (launchd, systemd, Windows Task Scheduler)
- Both server mode (local embeddings) and client mode (remote server)
"""

import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from claude_history_rag.settings_wizard import get_daemon_env_from_service

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# ANSI colors (disabled on Windows without ANSI support)
def supports_color() -> bool:
    """Check if terminal supports ANSI colors."""
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if platform.system() == "Windows":
        # Windows 10+ supports ANSI in cmd/powershell
        return os.environ.get("TERM") or os.environ.get("WT_SESSION")
    return True


if supports_color():
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
else:
    GREEN = YELLOW = RED = BLUE = BOLD = RESET = ""


def print_header(text: str) -> None:
    """Print a styled header."""
    print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}{BLUE}{text:^60}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 60}{RESET}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def prompt_choice(prompt: str, choices: list[str], default: int = 0) -> int:
    """Prompt user to choose from a list of options."""
    print(f"\n{prompt}")
    for i, choice in enumerate(choices):
        marker = ">" if i == default else " "
        print(f"  {marker} [{i + 1}] {choice}")

    while True:
        try:
            response = input(
                f"\nEnter choice [1-{len(choices)}] (default: {default + 1}): "
            ).strip()
            if not response:
                return default
            idx = int(response) - 1
            if 0 <= idx < len(choices):
                return idx
            print_error(f"Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print_error("Please enter a valid number")


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt user for yes/no answer."""
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


def validate_machine_id(machine_id: str) -> bool:
    """Validate that a machine ID contains only safe characters."""
    # Allow alphanumeric, dash, underscore, and dot
    return bool(re.match(r"^[a-zA-Z0-9._-]+$", machine_id))


def prompt_url(prompt: str, default: str = "") -> str:
    """Prompt user for a URL with validation."""
    while True:
        url = prompt_string(prompt, default)
        if not url:
            return ""
        if validate_url(url):
            return url
        print_error("Please enter a valid URL (e.g., http://localhost:4680)")


def check_url_reachable(url: str, timeout: int = 5) -> tuple[bool, int | None, str | None]:
    """Check if a URL is reachable. Returns (reachable, status_code, error)."""
    if not HTTPX_AVAILABLE:
        return False, None, "httpx not installed"
    try:
        response = httpx.get(url, timeout=timeout)
        if response.status_code == 200:
            return True, response.status_code, None
        return False, response.status_code, f"HTTP {response.status_code}"
    except Exception as e:
        return False, None, str(e)


def prompt_machine_id(prompt: str, default: str = "") -> str:
    """Prompt user for a machine ID with validation."""
    while True:
        machine_id = prompt_string(prompt, default)
        if not machine_id:
            return ""
        if validate_machine_id(machine_id):
            return machine_id
        print_error("Machine ID can only contain letters, numbers, dashes, underscores, and dots")


def get_project_dir() -> Path:
    """Get the project directory (where pyproject.toml is)."""
    # Walk up from this file to find pyproject.toml
    current = Path(__file__).parent
    for _ in range(5):  # Max 5 levels up
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to cwd
    return Path.cwd()


def get_uv_path() -> str | None:
    """Find the uv binary path. Returns None if not found."""
    # Check PATH first
    uv_path = shutil.which("uv")
    if uv_path:
        return uv_path

    # Common locations
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


def validate_uv_executable(uv_path: str) -> bool:
    """Validate that uv exists and is executable."""
    if not uv_path:
        return False
    path = Path(uv_path)
    return path.exists() and os.access(path, os.X_OK)


# =============================================================================
# MCP Configuration Detection
# =============================================================================


class MCPTarget:
    """Represents an MCP-compatible application."""

    def __init__(
        self,
        name: str,
        config_paths: dict[str, Path],
        config_key: str = "mcpServers",
        wrapper_key: str | None = None,
        config_format: str = "json",
        require_existing: bool = False,
    ):
        self.name = name
        self.config_paths = config_paths  # platform -> path
        self.config_key = config_key  # Key for servers dict
        self.wrapper_key = wrapper_key  # Optional wrapper (e.g., "mcp" for VS Code)
        self.config_format = config_format  # json or toml
        self.require_existing = require_existing  # Only consider installed if file exists

    def get_config_path(self) -> Path | None:
        """Get config path for current platform."""
        system = platform.system()
        path_template = self.config_paths.get(system)
        if not path_template:
            return None
        # Expand ~ and env vars
        return Path(os.path.expandvars(os.path.expanduser(str(path_template))))

    def is_installed(self) -> bool:
        """Check if the application appears to be installed."""
        config_path = self.get_config_path()
        if not config_path:
            return False
        # Check if config exists or parent dir exists (app installed but not configured)
        if self.require_existing:
            return config_path.exists()
        return config_path.exists() or config_path.parent.exists()

    def config_exists(self) -> bool:
        """Check if config file already exists."""
        config_path = self.get_config_path()
        return config_path is not None and config_path.exists()


def get_mcp_targets(project_dir: Path | None = None) -> list[MCPTarget]:
    """Get all known MCP-compatible applications."""
    home = str(Path.home())
    targets: list[MCPTarget] = []

    def detect_mcp_json_format(path: Path) -> tuple[str | None, str]:
        """Detect MCP JSON config structure for a given file."""
        if not path.exists():
            return None, "servers"
        try:
            data = json.loads(path.read_text())
        except Exception:
            return None, "servers"
        if isinstance(data, dict):
            if "servers" in data:
                return None, "servers"
            if "mcpServers" in data:
                return None, "mcpServers"
            if isinstance(data.get("mcp"), dict) and "servers" in data.get("mcp", {}):
                return "mcp", "servers"
        return None, "servers"

    def find_mcp_json_files(roots: list[Path]) -> list[Path]:
        """Find existing mcp.json files under the given roots."""
        results: list[Path] = []
        skip_dirs = {
            ".git",
            ".cache",
            ".local",
            ".npm",
            ".gradle",
            "node_modules",
            ".vscode",
            "Library/Caches",
            "Library/Developer",
            "Library/Logs",
            "Library/Containers",
            "Library/Application Support/Code/User/workspaceStorage",
            "Library/Application Support/Code/User/globalStorage",
            "Library/Application Support/Code/User/History",
        }
        for root in roots:
            if not root.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                try:
                    rel = Path(dirpath).relative_to(root)
                except ValueError:
                    rel = Path(dirpath)
                # Skip heavy or irrelevant directories
                if any(str(rel).startswith(skip) for skip in skip_dirs):
                    dirnames[:] = []
                    continue
                if "mcp.json" in filenames:
                    results.append(Path(dirpath) / "mcp.json")
        return results

    targets.extend(
        [
            MCPTarget(
                name="Claude Desktop",
                config_paths={
                    "Darwin": Path(
                        f"{home}/Library/Application Support/Claude/claude_desktop_config.json"
                    ),
                    "Windows": Path(
                        os.path.expandvars(r"%APPDATA%\Claude\claude_desktop_config.json")
                    ),
                    "Linux": Path(f"{home}/.config/Claude/claude_desktop_config.json"),
                },
                require_existing=True,
            ),
            MCPTarget(
                name="Claude Code",
                config_paths={
                    "Darwin": Path(f"{home}/.claude.json"),
                    "Windows": Path(f"{home}/.claude.json"),
                    "Linux": Path(f"{home}/.claude.json"),
                },
                require_existing=True,
            ),
            MCPTarget(
                name="Cursor",
                config_paths={
                    "Darwin": Path(f"{home}/.cursor/mcp.json"),
                    "Windows": Path(f"{home}/.cursor/mcp.json"),
                    "Linux": Path(f"{home}/.cursor/mcp.json"),
                },
                require_existing=True,
            ),
            MCPTarget(
                name="VS Code",
                config_paths={
                    "Darwin": Path(f"{home}/Library/Application Support/Code/User/mcp.json"),
                    "Windows": Path(os.path.expandvars(r"%APPDATA%\\Code\\User\\mcp.json")),
                    "Linux": Path(f"{home}/.config/Code/User/mcp.json"),
                },
                config_key="servers",
                require_existing=True,
            ),
            MCPTarget(
                name="Gemini CLI",
                config_paths={
                    "Darwin": Path(f"{home}/.gemini/settings.json"),
                    "Windows": Path(f"{home}/.gemini/settings.json"),
                    "Linux": Path(f"{home}/.gemini/settings.json"),
                },
                require_existing=True,
            ),
            MCPTarget(
                name="OpenAI Codex",
                config_paths={
                    "Darwin": Path(f"{home}/.codex/config.toml"),
                    "Windows": Path(f"{home}/.codex/config.toml"),
                    "Linux": Path(f"{home}/.codex/config.toml"),
                },
                config_format="toml",
                require_existing=True,
            ),
            MCPTarget(
                name="Google Antigravity",
                config_paths={
                    "Darwin": Path(f"{home}/.gemini/antigravity/mcp_config.json"),
                    "Windows": Path(f"{home}/.gemini/antigravity/mcp_config.json"),
                    "Linux": Path(f"{home}/.gemini/antigravity/mcp_config.json"),
                },
                require_existing=True,
            ),
        ]
    )

    # VS Code profile-specific configs (only add if they exist)
    vscode_profiles: dict[str, Path] = {}
    system = platform.system()
    if system == "Darwin":
        vscode_profiles = {
            "profiles_dir": Path(f"{home}/Library/Application Support/Code/User/profiles")
        }
    elif system == "Windows":
        vscode_profiles = {
            "profiles_dir": Path(os.path.expandvars(r"%APPDATA%\\Code\\User\\profiles"))
        }
    elif system == "Linux":
        vscode_profiles = {"profiles_dir": Path(f"{home}/.config/Code/User/profiles")}

    profiles_dir = vscode_profiles.get("profiles_dir")
    if profiles_dir and profiles_dir.exists():
        for profile_path in sorted(profiles_dir.glob("*/mcp.json")):
            targets.append(
                MCPTarget(
                    name=f"VS Code (Profile: {profile_path.parent.name})",
                    config_paths={
                        "Darwin": profile_path,
                        "Windows": profile_path,
                        "Linux": profile_path,
                    },
                    config_key="servers",
                    require_existing=True,
                )
            )

    # Generic mcp.json discovery across the user's profile
    print(
        "\nScanning your home directory for existing mcp.json files. "
        "macOS may prompt for folder access."
    )
    discovered = find_mcp_json_files([Path.home()])
    for path in sorted(set(discovered)):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        wrapper_key, config_key = detect_mcp_json_format(path)
        targets.append(
            MCPTarget(
                name=f"Found MCP JSON ({path})",
                config_paths={
                    "Darwin": path,
                    "Windows": path,
                    "Linux": path,
                },
                config_key=config_key,
                wrapper_key=wrapper_key,
                require_existing=True,
            )
        )

    if project_dir:
        gemini_project_config = project_dir / ".gemini" / "settings.json"
        if gemini_project_config.exists():
            targets.append(
                MCPTarget(
                    name="Gemini CLI (Project)",
                    config_paths={
                        "Darwin": gemini_project_config,
                        "Windows": gemini_project_config,
                        "Linux": gemini_project_config,
                    },
                    require_existing=True,
                )
            )

        claude_project_config = project_dir / ".claude.json"
        if claude_project_config.exists():
            targets.append(
                MCPTarget(
                    name="Claude Code (Project)",
                    config_paths={
                        "Darwin": claude_project_config,
                        "Windows": claude_project_config,
                        "Linux": claude_project_config,
                    },
                    require_existing=True,
                )
            )

    return targets


def backup_config(path: Path) -> Path | None:
    """Create a backup of a config file if it exists. Returns backup path or None."""
    if not path.exists():
        return None
    backup_path = path.with_suffix(path.suffix + ".backup")
    try:
        shutil.copy2(path, backup_path)
        return backup_path
    except OSError as e:
        print_warning(f"Could not create backup of {path}: {e}")
        return None


def read_json_config(path: Path) -> dict[str, Any]:
    """Read a JSON config file, returning empty dict if not exists."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print_warning(f"Could not read {path}: {e}")
        return {}


def write_json_config(path: Path, config: dict[str, Any]) -> bool:
    """Write a JSON config file, creating parent dirs if needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2))
        return True
    except OSError as e:
        print_error(f"Could not write {path}: {e}")
        return False


def _toml_escape_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_serialize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _toml_escape_string(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_serialize_value(v) for v in value) + "]"
    return _toml_escape_string(str(value))


def _build_toml_mcp_block(server_name: str, server_config: dict[str, Any]) -> str:
    lines: list[str] = [f"[mcp_servers.{server_name}]"]

    if "command" in server_config:
        lines.append(f"command = {_toml_serialize_value(server_config['command'])}")
    if "args" in server_config:
        lines.append(f"args = {_toml_serialize_value(server_config['args'])}")

    env = server_config.get("env")
    if isinstance(env, dict) and env:
        lines.append("")
        lines.append(f"[mcp_servers.{server_name}.env]")
        for key, value in env.items():
            lines.append(f"{key} = {_toml_serialize_value(value)}")

    return "\n".join(lines).rstrip() + "\n"


def _remove_toml_mcp_sections(toml_text: str, server_name: str) -> str:
    if not toml_text.strip():
        return toml_text

    prefix = f"mcp_servers.{server_name}"
    lines = toml_text.splitlines()
    output: list[str] = []
    skip = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            header = stripped.strip("[]").strip()
            if header.startswith(prefix):
                skip = True
                continue
            skip = False
        if not skip:
            output.append(line)

    result = "\n".join(output).rstrip()
    if toml_text.endswith("\n"):
        result += "\n"
    return result


def add_mcp_to_toml_config(
    path: Path,
    server_config: dict[str, Any],
    server_name: str = "ai-agent-history-rag",
) -> bool:
    """Add MCP server config to a TOML file."""
    try:
        existing = path.read_text() if path.exists() else ""
        cleaned = _remove_toml_mcp_sections(existing, server_name).rstrip()
        block = _build_toml_mcp_block(server_name, server_config).rstrip()
        new_text = f"{cleaned}\n\n{block}\n" if cleaned else f"{block}\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_text)
        return True
    except OSError as e:
        print_error(f"Could not write {path}: {e}")
        return False


def remove_mcp_from_toml_config(path: Path, server_name: str) -> bool:
    """Remove MCP server config from a TOML file."""
    if not path.exists():
        return False
    try:
        existing = path.read_text()
        updated = _remove_toml_mcp_sections(existing, server_name)
        if updated == existing:
            return False
        path.write_text(updated)
        return True
    except OSError as e:
        print_error(f"Could not write {path}: {e}")
        return False


def build_mcp_server_config(
    project_dir: Path,
    uv_path: str,
    server_url: str | None = None,
    machine_id: str | None = None,
    client_name: str | None = None,
    embedding_url: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Build the MCP server configuration dict."""
    config: dict[str, Any] = {
        "command": uv_path,
        "args": ["--directory", str(project_dir), "run", "ai-agent-history-rag"],
    }

    env: dict[str, str] = {}

    # Client mode
    if server_url:
        env["CLAUDE_HISTORY_RAG_SERVER_URL"] = server_url
        if machine_id:
            env["CLAUDE_HISTORY_RAG_MACHINE_ID"] = machine_id
        if client_name:
            env["CLAUDE_HISTORY_RAG_CLIENT_NAME"] = client_name
    else:
        # Server mode - add embedding config
        if embedding_url:
            env["CLAUDE_HISTORY_RAG_EMBEDDING_BASE_URL"] = embedding_url
        if embedding_model:
            env["CLAUDE_HISTORY_RAG_EMBEDDING_MODEL"] = embedding_model

    if env:
        config["env"] = env

    return config


def add_mcp_to_target(
    target: MCPTarget,
    server_config: dict[str, Any],
    server_name: str = "ai-agent-history-rag",
) -> bool:
    """Add MCP server configuration to a target application."""
    config_path = target.get_config_path()
    if not config_path:
        print_error(f"No config path for {target.name} on this platform")
        return False
    if target.require_existing and not config_path.exists():
        print_warning(f"Skipping {target.name}: config does not exist at {config_path}")
        return False

    if target.config_format == "toml":
        backup_path = backup_config(config_path)
        if backup_path:
            print(f"  Backed up existing config to: {backup_path}")
        if add_mcp_to_toml_config(config_path, server_config, server_name):
            print_success(f"Added to {target.name}: {config_path}")
            return True
        return False

    # Backup existing config before modification
    backup_path = backup_config(config_path)
    if backup_path:
        print(f"  Backed up existing config to: {backup_path}")

    # Read existing config
    config = read_json_config(config_path)
    if not isinstance(config, dict):
        print_warning(f"Skipping {target.name}: config is not a JSON object")
        return False

    def normalize_legacy_entrypoint(servers: dict[str, Any]) -> None:
        """Upgrade legacy args that referenced 'claude-history-rag'."""
        for key in ("claude-history-rag", "ai-agent-history-rag"):
            entry = servers.get(key)
            if not isinstance(entry, dict):
                continue
            args = entry.get("args")
            if isinstance(args, list) and "claude-history-rag" in args:
                entry["args"] = ["ai-agent-history-rag" if a == "claude-history-rag" else a for a in args]
                servers[key] = entry

    # Handle Claude Code specially - it has a different structure
    if target.name == "Claude Code":
        # Claude Code uses mcpServers at top level in ~/.claude.json
        if "mcpServers" not in config:
            config["mcpServers"] = {}
        normalize_legacy_entrypoint(config["mcpServers"])
        config["mcpServers"][server_name] = server_config

    # Handle VS Code (wrapped in "mcp" key)
    elif target.wrapper_key:
        if target.wrapper_key not in config:
            config[target.wrapper_key] = {}
        if target.config_key not in config[target.wrapper_key]:
            config[target.wrapper_key][target.config_key] = {}
        normalize_legacy_entrypoint(config[target.wrapper_key][target.config_key])
        config[target.wrapper_key][target.config_key][server_name] = server_config

    # Standard format (Claude Desktop, Cursor)
    else:
        if target.config_key not in config:
            config[target.config_key] = {}
        normalize_legacy_entrypoint(config[target.config_key])
        config[target.config_key][server_name] = server_config

    # Write back
    if write_json_config(config_path, config):
        print_success(f"Added to {target.name}: {config_path}")
        return True
    return False


def discover_project_mcp_targets(root_paths: list[Path]) -> list[MCPTarget]:
    """Discover project-scoped mcp.json files under provided roots."""
    targets: list[MCPTarget] = []
    if root_paths:
        roots_display = ", ".join(str(p) for p in root_paths)
        print(
            f"\nScanning for mcp.json under: {roots_display}. "
            "macOS may prompt for folder access."
        )

    def detect_mcp_json_format(path: Path) -> tuple[str | None, str]:
        if not path.exists():
            return None, "servers"
        try:
            data = json.loads(path.read_text())
        except Exception:
            return None, "servers"
        if isinstance(data, dict):
            if "servers" in data:
                return None, "servers"
            if "mcpServers" in data:
                return None, "mcpServers"
            if isinstance(data.get("mcp"), dict) and "servers" in data.get("mcp", {}):
                return "mcp", "servers"
        return None, "servers"

    def find_mcp_json_files(roots: list[Path]) -> list[Path]:
        results: list[Path] = []
        skip_dirs = {
            ".git",
            ".cache",
            ".local",
            ".npm",
            ".gradle",
            "node_modules",
            "Library/Caches",
            "Library/Developer",
            "Library/Logs",
            "Library/Containers",
        }
        for root in roots:
            if not root.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                try:
                    rel = Path(dirpath).relative_to(root)
                except ValueError:
                    rel = Path(dirpath)
                if any(str(rel).startswith(skip) for skip in skip_dirs):
                    dirnames[:] = []
                    continue
                if "mcp.json" in filenames:
                    results.append(Path(dirpath) / "mcp.json")
        return results

    discovered = find_mcp_json_files(root_paths)
    for path in sorted(set(discovered)):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        wrapper_key, config_key = detect_mcp_json_format(path)
        targets.append(
            MCPTarget(
                name=f"Found MCP JSON ({path})",
                config_paths={
                    "Darwin": path,
                    "Windows": path,
                    "Linux": path,
                },
                config_key=config_key,
                wrapper_key=wrapper_key,
                require_existing=True,
            )
        )

    return targets


# =============================================================================
# Post-Installation Verification
# =============================================================================


def get_pid_file_path() -> Path:
    """Get the daemon PID file path."""
    return Path.home() / ".claude-history-rag" / "daemon.pid"


def wait_for_daemon_start(timeout: int = 30) -> tuple[bool, int | None]:
    """Wait for daemon to start and return (success, pid).

    Args:
        timeout: Maximum seconds to wait

    Returns:
        Tuple of (started successfully, pid or None)
    """
    pid_file = get_pid_file_path()
    start_time = time.time()

    print(f"\nWaiting for daemon to start (timeout: {timeout}s)...")

    while time.time() - start_time < timeout:
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                # Check if process is actually running
                os.kill(pid, 0)
                return True, pid
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        time.sleep(0.5)
        print(".", end="", flush=True)

    print()  # Newline after dots
    return False, None


def check_daemon_health(
    is_client_mode: bool,
    server_url: str | None = None,
    local_port: int = 4680,
    timeout: int = 10,
) -> tuple[bool, str]:
    """Check if daemon is healthy by hitting its endpoints.

    Args:
        is_client_mode: Whether running in client mode
        server_url: Central server URL (for client mode)
        local_port: Local status server port (for server/standalone mode)
        timeout: Request timeout in seconds

    Returns:
        Tuple of (healthy, status message)
    """
    if not HTTPX_AVAILABLE:
        return True, "Health check skipped (httpx not available)"

    try:
        with httpx.Client(timeout=timeout) as client:
            if is_client_mode:
                # Client mode: check if central server is reachable
                if server_url:
                    try:
                        response = client.get(f"{server_url}/health")
                        if response.status_code == 200:
                            return True, f"Central server at {server_url} is reachable"
                        return False, f"Central server returned status {response.status_code}"
                    except httpx.ConnectError:
                        return False, f"Cannot connect to central server at {server_url}"
                    except httpx.ConnectTimeout:
                        return False, f"Connection to {server_url} timed out"
            else:
                # Server/Standalone mode: check local status server
                try:
                    response = client.get(f"http://127.0.0.1:{local_port}/health")
                    if response.status_code == 200:
                        return True, f"Local status server is running on port {local_port}"
                    return False, f"Status server returned status {response.status_code}"
                except httpx.ConnectError:
                    return False, f"Status server not responding on port {local_port}"
                except httpx.ConnectTimeout:
                    return False, "Status server connection timed out"
    except Exception as e:
        return False, f"Health check failed: {type(e).__name__}: {e}"

    return True, "Health check passed"


def verify_installation(
    daemon_installed: bool,
    deployment_mode: str,  # "central_server", "client", or "standalone"
    server_url: str | None = None,
) -> bool:
    """Run post-installation verification.

    Args:
        daemon_installed: Whether daemon was installed
        deployment_mode: One of "central_server", "client", or "standalone"
        server_url: Central server URL (for client mode)

    Returns:
        True if all checks passed
    """
    if not daemon_installed:
        return True  # Nothing to verify

    print_header("Installation Verification")

    mode_names = {
        "central_server": "Central Server",
        "client": "Client",
        "standalone": "Standalone",
    }
    print(f"Verifying {mode_names.get(deployment_mode, deployment_mode)} installation...\n")

    all_passed = True

    # Step 1: Wait for daemon to start
    started, pid = wait_for_daemon_start(timeout=30)
    if started:
        print_success(f"Daemon started (PID {pid})")
    else:
        print_error("Daemon failed to start within 30 seconds")
        print("  Check logs: ~/.claude-history-rag/daemon.log")
        all_passed = False

    # Step 2: Health check (only if daemon started)
    if started:
        # Give daemon a moment to initialize its servers
        print("\nWaiting for services to initialize...")
        time.sleep(3)

        is_client_mode = deployment_mode == "client"
        healthy, message = check_daemon_health(
            is_client_mode=is_client_mode,
            server_url=server_url,
        )

        if healthy:
            print_success(message)
        else:
            print_warning(message)
            # Provide mode-specific guidance
            if deployment_mode == "client":
                print("  The daemon is running but cannot reach the central server yet.")
                print("  Make sure the central server is running and accessible.")
            elif deployment_mode == "central_server":
                print("  The status server may still be initializing.")
                print("  Check if your embedding server (e.g., Ollama) is running.")
            else:  # standalone
                print("  The status server may still be initializing.")
                print("  Check if your embedding server (e.g., Ollama) is running.")

    # Mode-specific additional checks
    if started and deployment_mode == "central_server":
        # For central server, also verify it's listening on 0.0.0.0
        print("\nCentral server should be accessible from other machines on port 4680.")
        print("  Verify firewall allows incoming connections on port 4680.")

    # Summary
    if all_passed:
        print_success("\nAll verification checks passed!")
    else:
        print_warning("\nSome verification checks failed. Check the logs for details.")

    return all_passed


# =============================================================================
# Daemon Service Installation
# =============================================================================


def install_daemon_macos(
    project_dir: Path,
    uv_path: str,
    env_vars: dict[str, str],
) -> bool:
    """Install launchd service on macOS."""
    home = Path.home()
    plist_path = home / "Library/LaunchAgents/com.ai-agent-history-rag.daemon.plist"
    log_dir = home / ".claude-history-rag"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build environment dict for plist (escape XML special chars)
    def escape_xml(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    env_xml_lines = []
    for k, v in env_vars.items():
        env_xml_lines.append(f"        <key>{escape_xml(k)}</key>")
        env_xml_lines.append(f"        <string>{escape_xml(v)}</string>")
    env_xml = "\n".join(env_xml_lines) if env_xml_lines else ""

    # Build EnvironmentVariables section only if there are env vars
    env_section = ""
    if env_xml:
        env_section = f"""
    <key>EnvironmentVariables</key>
    <dict>
{env_xml}
    </dict>
"""

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ai-agent-history-rag.daemon</string>

    <key>ProgramArguments</key>
    <array>
        <string>{escape_xml(uv_path)}</string>
        <string>--directory</string>
        <string>{escape_xml(str(project_dir))}</string>
        <string>run</string>
        <string>ai-agent-history-rag-daemon</string>
        <string>start</string>
    </array>
{env_section}
    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>{escape_xml(str(log_dir))}/launchd-stdout.log</string>

    <key>StandardErrorPath</key>
    <string>{escape_xml(str(log_dir))}/launchd-stderr.log</string>

    <key>WorkingDirectory</key>
    <string>{escape_xml(str(project_dir))}</string>
</dict>
</plist>
"""

    # Remove existing service to ensure updates are applied
    if plist_path.exists():
        with contextlib.suppress(Exception):
            subprocess.run(
                ["launchctl", "unload", str(plist_path)],
                capture_output=True,
                check=False,
            )
        with contextlib.suppress(Exception):
            plist_path.unlink()

    # Write plist
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content)

    # Load service
    result = subprocess.run(
        ["launchctl", "load", str(plist_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print_success(f"Daemon installed: {plist_path}")
        print(f"  Logs: {log_dir}/daemon.log")
        print(f"  Stop: launchctl unload {plist_path}")
        return True
    else:
        print_error(f"Failed to load service: {result.stderr}")
        return False


def install_daemon_linux(
    project_dir: Path,
    uv_path: str,
    env_vars: dict[str, str],
) -> bool:
    """Install systemd user service on Linux."""
    home = Path.home()
    service_dir = home / ".config/systemd/user"
    service_path = service_dir / "ai-agent-history-rag.service"
    log_dir = home / ".claude-history-rag"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Escape spaces with backslash for systemd paths
    def escape_systemd_path(path: str) -> str:
        return path.replace(" ", r"\ ")

    uv_escaped = escape_systemd_path(uv_path)
    project_escaped = escape_systemd_path(str(project_dir))

    # Build environment lines (use quotes around values to handle special chars)
    env_lines = "\n".join(f'Environment="{k}={v}"' for k, v in env_vars.items())

    # Build service content - only include env lines if present
    env_section = f"\n{env_lines}\n" if env_lines else ""

    service_content = f"""[Unit]
Description=AI Agent History RAG Daemon
After=network.target

[Service]
Type=simple
ExecStart={uv_escaped} --directory {project_escaped} run ai-agent-history-rag-daemon start
Restart=on-failure
RestartSec=10
WorkingDirectory={project_escaped}
{env_section}
[Install]
WantedBy=default.target
"""

    # Remove existing service to ensure updates are applied
    subprocess.run(
        ["systemctl", "--user", "stop", "ai-agent-history-rag"],
        capture_output=True,
        check=False,
    )
    subprocess.run(
        ["systemctl", "--user", "disable", "ai-agent-history-rag"],
        capture_output=True,
        check=False,
    )
    if service_path.exists():
        with contextlib.suppress(OSError):
            service_path.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)

    # Write service file
    service_dir.mkdir(parents=True, exist_ok=True)
    service_path.write_text(service_content)

    # Reload systemd to pick up new/changed service file
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)

    # Enable and start the service
    result = subprocess.run(
        ["systemctl", "--user", "enable", "--now", "ai-agent-history-rag"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print_success(f"Daemon installed: {service_path}")
        print("  Status: systemctl --user status ai-agent-history-rag")
        print("  Logs:   journalctl --user -u ai-agent-history-rag -f")
        return True
    else:
        print_error(f"Failed to enable service: {result.stderr}")
        return False


def install_daemon_windows(
    project_dir: Path,
    uv_path: str,
    env_vars: dict[str, str],
) -> bool:
    """Install Windows scheduled task."""
    task_name = "AIAgentHistoryRAG"

    # Remove existing task
    subprocess.run(
        ["schtasks", "/Delete", "/TN", task_name, "/F"],
        capture_output=True,
        check=False,
    )

    # Build the full command - schtasks /TR needs the whole thing as one string
    # Escape paths properly for Windows
    full_command = f'"{uv_path}" --directory "{project_dir}" run ai-agent-history-rag-daemon start'

    # Create task using schtasks
    # Note: Environment variables need to be set in user environment
    result = subprocess.run(
        [
            "schtasks",
            "/Create",
            "/TN",
            task_name,
            "/TR",
            full_command,
            "/SC",
            "ONLOGON",
            "/RL",
            "LIMITED",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        # Start the task
        subprocess.run(["schtasks", "/Run", "/TN", task_name], check=False)
        print_success(f"Daemon installed as scheduled task: {task_name}")

        if env_vars:
            print_warning("Environment variables need to be set manually:")
            for k, v in env_vars.items():
                print(f"  {k}={v}")
            print("Set these in System Properties > Environment Variables")
            print(
                "Then restart the task: schtasks /End /TN AIAgentHistoryRAG && schtasks /Run /TN AIAgentHistoryRAG"
            )

        return True
    else:
        print_error(f"Failed to create task: {result.stderr}")
        return False


def install_daemon(
    project_dir: Path,
    uv_path: str,
    env_vars: dict[str, str],
) -> bool:
    """Install daemon service for current platform."""
    system = platform.system()

    if system == "Darwin":
        return install_daemon_macos(project_dir, uv_path, env_vars)
    elif system == "Linux":
        return install_daemon_linux(project_dir, uv_path, env_vars)
    elif system == "Windows":
        return install_daemon_windows(project_dir, uv_path, env_vars)
    else:
        print_error(f"Unsupported platform: {system}")
        return False


# =============================================================================
# Main Wizard
# =============================================================================


def run_wizard() -> int:
    """Run the installation wizard."""
    print_header("AI Agent History RAG Installer")

    project_dir = get_project_dir()
    uv_path = get_uv_path()

    print(f"Project directory: {project_dir}")

    if not uv_path:
        print_error("uv package manager not found!")
        print("\nPlease install uv first:")
        if platform.system() == "Windows":
            print("  irm https://astral.sh/uv/install.ps1 | iex")
        else:
            print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        return 1

    if not validate_uv_executable(uv_path):
        print_error(f"uv found at {uv_path} but is not executable")
        return 1

    print(f"UV path: {uv_path}")

    # Step 1: Choose deployment mode
    print_header("Deployment Mode")

    deployment_modes = [
        "Update - Reinstall services using existing config (no prompts)",
        "Central Server - Accept connections from other machines (multi-machine hub)",
        "Client - Connect to an existing central server (multi-machine client)",
        "Standalone - Single machine setup (everything local, recommended for most users)",
    ]

    deployment_choice = prompt_choice("How will this machine be used?", deployment_modes, default=3)

    is_update = deployment_choice == 0
    is_central_server = deployment_choice == 1
    is_client_mode = deployment_choice == 2
    # is_standalone = deployment_choice == 3

    # Collect configuration based on mode
    env_vars: dict[str, str] = {}
    server_url: str | None = None
    machine_id: str | None = None
    client_name: str | None = None
    embedding_url: str | None = None
    embedding_model: str | None = None
    existing_env = get_daemon_env_from_service()

    def load_env_file(path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        env: dict[str, str] = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
        return env

    # Set install defaults based on deployment mode
    if is_update:
        print_header("Update With Existing Config")

        daemon_env = get_daemon_env_from_service()

        if not daemon_env:
            print_error("No existing daemon configuration found to update.")
            print_error("Re-run the full installer to configure this machine first.")
            return 1

        env_vars.update(daemon_env)
        server_url = env_vars.get("CLAUDE_HISTORY_RAG_SERVER_URL")
        machine_id = env_vars.get("CLAUDE_HISTORY_RAG_MACHINE_ID")
        client_name = env_vars.get("CLAUDE_HISTORY_RAG_CLIENT_NAME")
        embedding_url = env_vars.get("CLAUDE_HISTORY_RAG_EMBEDDING_BASE_URL")
        embedding_model = env_vars.get("CLAUDE_HISTORY_RAG_EMBEDDING_MODEL")

        is_client_mode = bool(server_url)
        is_central_server = (
            not is_client_mode
            and env_vars.get("CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST") == "0.0.0.0"
        )

        install_mcp = True
        install_daemon_service = True
        print_success("Loaded existing configuration; reinstalling services...")

    elif is_central_server:
        # Central server: daemon only (no MCP server needed on the server itself)
        install_mcp = False
        install_daemon_service = True

        print_header("Central Server Configuration")
        print_success("This machine will accept connections from client machines")
        env_vars["CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST"] = "0.0.0.0"
        print_warning(
            "Make sure port 4680 is accessible from client machines (firewall/port forwarding)"
        )

        default_embedding_url = (
            existing_env.get("CLAUDE_HISTORY_RAG_EMBEDDING_BASE_URL") if existing_env else None
        )
        embedding_url = prompt_url(
            "Enter embedding server URL (Ollama, vLLM, OpenAI-compatible)",
            default_embedding_url or "http://localhost:11434/v1",
        )
        if embedding_url:
            env_vars["CLAUDE_HISTORY_RAG_EMBEDDING_BASE_URL"] = embedding_url

        default_embedding_model = (
            existing_env.get("CLAUDE_HISTORY_RAG_EMBEDDING_MODEL") if existing_env else None
        )
        embedding_model = prompt_string(
            "Enter embedding model name",
            default_embedding_model or "nomic-embed-text",
        )
        if embedding_model:
            env_vars["CLAUDE_HISTORY_RAG_EMBEDDING_MODEL"] = embedding_model

    elif is_client_mode:
        # Client: MCP server + daemon, connecting to remote
        install_mcp = True
        install_daemon_service = True

        print_header("Client Configuration")
        default_server_url = (
            existing_env.get("CLAUDE_HISTORY_RAG_SERVER_URL") if existing_env else None
        )
        server_url = prompt_url(
            "Enter the central server URL",
            default_server_url or "http://192.168.1.100:4680",
        )
        if not server_url:
            print_error("Server URL is required for client mode")
            return 1
        print(f"Checking central server at {server_url}...")
        reachable, status, error = check_url_reachable(f"{server_url}/health")
        if not reachable:
            print_error(f"Cannot reach central server: {error}")
            print_error("Please verify the server is running and reachable, then retry.")
            return 1
        env_vars["CLAUDE_HISTORY_RAG_SERVER_URL"] = server_url

        default_machine_id = (
            existing_env.get("CLAUDE_HISTORY_RAG_MACHINE_ID") if existing_env else None
        )
        machine_id = prompt_machine_id(
            "Enter a unique machine ID",
            default_machine_id or platform.node(),
        )
        if machine_id:
            env_vars["CLAUDE_HISTORY_RAG_MACHINE_ID"] = machine_id

        default_client_name = (
            existing_env.get("CLAUDE_HISTORY_RAG_CLIENT_NAME") if existing_env else None
        )
        client_name = prompt_string(
            "Enter an optional client name (optional)",
            default_client_name or "",
        )
        if client_name:
            env_vars["CLAUDE_HISTORY_RAG_CLIENT_NAME"] = client_name

    else:
        # Standalone: MCP server + daemon, local processing
        install_mcp = True
        install_daemon_service = True

        print_header("Standalone Configuration")

        default_embedding_url = (
            existing_env.get("CLAUDE_HISTORY_RAG_EMBEDDING_BASE_URL") if existing_env else None
        )
        embedding_url = prompt_url(
            "Enter embedding server URL (Ollama, vLLM, OpenAI-compatible)",
            default_embedding_url or "http://localhost:11434/v1",
        )
        if embedding_url:
            env_vars["CLAUDE_HISTORY_RAG_EMBEDDING_BASE_URL"] = embedding_url

        default_embedding_model = (
            existing_env.get("CLAUDE_HISTORY_RAG_EMBEDDING_MODEL") if existing_env else None
        )
        embedding_model = prompt_string(
            "Enter embedding model name",
            default_embedding_model or "nomic-embed-text",
        )
        if embedding_model:
            env_vars["CLAUDE_HISTORY_RAG_EMBEDDING_MODEL"] = embedding_model

    # Auth configuration (PSK)
    if not is_update:
        print_header("Auth (PSK)")
        default_auth_enabled = (
            existing_env.get("CLAUDE_HISTORY_RAG_AUTH_ENABLED") if existing_env else "true"
        )
        auth_enabled = prompt_yes_no(
            "Require PSK authentication for dashboard/API?",
            default=default_auth_enabled.lower() not in ("0", "false", "no"),
        )
        env_vars["CLAUDE_HISTORY_RAG_AUTH_ENABLED"] = "true" if auth_enabled else "false"

        if is_client_mode:
            default_client_psk = (
                existing_env.get("CLAUDE_HISTORY_RAG_CLIENT_PSK") if existing_env else ""
            )
            client_psk = prompt_string(
                "Client PSK (optional override)",
                default_client_psk or "",
            )
            if client_psk:
                env_vars["CLAUDE_HISTORY_RAG_CLIENT_PSK"] = client_psk

            default_client_auth_path = (
                existing_env.get("CLAUDE_HISTORY_RAG_CLIENT_AUTH_PATH") if existing_env else ""
            )
            client_auth_path = prompt_string(
                "Client auth path (optional)",
                default_client_auth_path,
            )
            if client_auth_path:
                env_vars["CLAUDE_HISTORY_RAG_CLIENT_AUTH_PATH"] = client_auth_path
        else:
            default_server_psk = (
                existing_env.get("CLAUDE_HISTORY_RAG_SERVER_PSK") if existing_env else ""
            )
            server_psk = prompt_string(
                "Server PSK (optional override)",
                default_server_psk or "",
            )
            if server_psk:
                env_vars["CLAUDE_HISTORY_RAG_SERVER_PSK"] = server_psk

            default_auth_state_path = (
                existing_env.get("CLAUDE_HISTORY_RAG_AUTH_STATE_PATH") if existing_env else ""
            )
            auth_state_path = prompt_string(
                "Auth state path (optional)",
                default_auth_state_path,
            )
            if auth_state_path:
                env_vars["CLAUDE_HISTORY_RAG_AUTH_STATE_PATH"] = auth_state_path

    # Step 3: MCP installation
    if install_mcp:
        print_header("MCP Server Installation")

        targets = get_mcp_targets(project_dir)
        extra_scan_roots: list[Path] = []
        if not is_update:
            print(
                "\nOptional: Add a parent folder to scan for project-scoped mcp.json files."
            )
            print("Example: ~/develop (we'll add to any mcp.json under that folder).")
            raw_paths = input("Additional scan folder(s), comma-separated [skip]: ").strip()
            if raw_paths:
                for part in raw_paths.split(","):
                    candidate = Path(os.path.expandvars(os.path.expanduser(part.strip())))
                    if candidate.exists() and candidate.is_dir():
                        extra_scan_roots.append(candidate)
                    else:
                        print_warning(f"Skipping invalid folder: {candidate}")
        if extra_scan_roots:
            targets.extend(discover_project_mcp_targets(extra_scan_roots))
        available_targets = [t for t in targets if t.is_installed()]

        if not available_targets:
            print_warning("No supported MCP applications detected.")
            print(
                "Supported applications: Claude Desktop, Claude Code, Cursor, VS Code, Gemini CLI, OpenAI Codex"
            )
        else:
            print("Detected MCP-compatible applications:")
            for i, target in enumerate(available_targets):
                status = "configured" if target.config_exists() else "not configured"
                print(f"  [{i + 1}] {target.name} ({status})")

        print(
            "\nNote: ChatGPT connectors (OpenAI) are configured in-app (Developer mode) "
            "and are not managed by this installer."
        )

        # Let user select which to configure
        if is_update:
            selected_targets = available_targets
        else:
            print("\nEnter numbers to configure (comma-separated), or 'all':")
            print("Tip: Use 'found' to add to every detected mcp.json in your profile.")
            selection = input("Selection [all]: ").strip().lower()

            if selection == "" or selection == "all":
                selected_targets = available_targets
            elif selection in {"found", "all-found"}:
                selected_targets = [t for t in available_targets if t.name.startswith("Found MCP JSON")]
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in selection.split(",")]
                    selected_targets = [
                        available_targets[i] for i in indices if 0 <= i < len(available_targets)
                    ]
                except (ValueError, IndexError):
                    print_error("Invalid selection")
                    selected_targets = []

        # Build and install MCP config
        if selected_targets:
            server_config = build_mcp_server_config(
                project_dir=project_dir,
                uv_path=uv_path,
                server_url=server_url,
                machine_id=machine_id,
                client_name=client_name,
                embedding_url=embedding_url,
                embedding_model=embedding_model,
            )

            for target in selected_targets:
                add_mcp_to_target(target, server_config)

    # Step 4: Daemon installation
    daemon_was_installed = False
    if install_daemon_service:
        print_header("Daemon Service Installation")

        if is_update:
            daemon_was_installed = install_daemon(project_dir, uv_path, env_vars)
        else:
            if prompt_yes_no("Install daemon to start automatically on boot?", default=True):
                daemon_was_installed = install_daemon(project_dir, uv_path, env_vars)
            else:
                print("\nTo start the daemon manually:")
                print(f"  cd {project_dir}")
                print("  uv run ai-agent-history-rag-daemon start")

    # Step 5: Verify installation
    if daemon_was_installed:
        # Determine deployment mode string
        if is_central_server:
            deployment_mode = "central_server"
        elif is_client_mode:
            deployment_mode = "client"
        else:
            deployment_mode = "standalone"

        verify_installation(
            daemon_installed=True,
            deployment_mode=deployment_mode,
            server_url=server_url,
        )

    # Done
    print_header("Installation Complete")

    if install_mcp:
        print("MCP Server: Restart your AI application to load the new server.")

    if install_daemon_service:
        if is_client_mode:
            print(f"Daemon: Connecting to {server_url}")
        else:
            print("Daemon: Make sure your embedding server is running (e.g., ollama serve)")

    # Show correct dashboard URL based on configuration
    dashboard_host = env_vars.get("CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST", "127.0.0.1")
    if not is_client_mode:
        print(f"\nDashboard: http://{dashboard_host}:4680/dashboard")
    print("Logs: ~/.claude-history-rag/daemon.log")

    return 0


def main() -> int:
    """Entry point."""
    try:
        return run_wizard()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled.")
        return 1
    except Exception as e:
        print_error(f"Installation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
