"""Uninstall wizard for ai-agent-history-rag."""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

from claude_history_rag.installer import (
    backup_config,
    get_mcp_targets,
    print_error,
    print_header,
    print_success,
    print_warning,
    prompt_yes_no,
    remove_mcp_from_toml_config,
    read_json_config,
    write_json_config,
)

SERVER_NAME = "ai-agent-history-rag"


def get_project_dir() -> Path:
    """Get the project directory (where pyproject.toml is)."""
    current = Path(__file__).parent
    for _ in range(5):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def remove_mcp_from_target(target, server_name: str) -> bool:
    """Remove MCP server configuration from a target application."""
    config_path = target.get_config_path()
    if not config_path or not config_path.exists():
        return False

    if getattr(target, "config_format", "json") == "toml":
        backup_path = backup_config(config_path)
        if backup_path:
            print(f"  Backed up existing config to: {backup_path}")
        if remove_mcp_from_toml_config(config_path, server_name):
            print_success(f"Removed from {target.name}: {config_path}")
            return True
        return False

    backup_path = backup_config(config_path)
    if backup_path:
        print(f"  Backed up existing config to: {backup_path}")

    config = read_json_config(config_path)
    modified = False

    if target.name == "Claude Code":
        servers = config.get("mcpServers", {})
        if server_name in servers:
            servers.pop(server_name, None)
            modified = True
        if modified:
            if servers:
                config["mcpServers"] = servers
            else:
                config.pop("mcpServers", None)

    elif target.wrapper_key:
        wrapper = config.get(target.wrapper_key, {})
        servers = wrapper.get(target.config_key, {})
        if server_name in servers:
            servers.pop(server_name, None)
            modified = True
        if modified:
            if servers:
                wrapper[target.config_key] = servers
            else:
                wrapper.pop(target.config_key, None)
            if wrapper:
                config[target.wrapper_key] = wrapper
            else:
                config.pop(target.wrapper_key, None)

    else:
        servers = config.get(target.config_key, {})
        if server_name in servers:
            servers.pop(server_name, None)
            modified = True
        if modified:
            if servers:
                config[target.config_key] = servers
            else:
                config.pop(target.config_key, None)

    if modified and write_json_config(config_path, config):
        print_success(f"Removed from {target.name}: {config_path}")
        return True
    return False


def remove_mcp_configs(server_name: str) -> None:
    """Remove MCP configs from all detected targets."""
    print_header("MCP Configuration Removal")
    project_dir = get_project_dir()
    targets = get_mcp_targets(project_dir)
    available_targets = [t for t in targets if t.is_installed()]

    if not available_targets:
        print_warning("No MCP-compatible applications detected.")
        print_warning(
            "ChatGPT connectors are configured in-app (Developer mode) and must be removed manually."
        )
        return

    any_removed = False
    for target in available_targets:
        removed = remove_mcp_from_target(target, server_name)
        if removed:
            any_removed = True

    if not any_removed:
        print_warning("No MCP configurations were removed.")
    print_warning(
        "ChatGPT connectors are configured in-app (Developer mode) and must be removed manually."
    )


def uninstall_daemon_service() -> None:
    """Stop and remove the daemon service."""
    print_header("Daemon Service Removal")
    system = platform.system()

    if system == "Darwin":
        plist_path = Path.home() / "Library/LaunchAgents/com.ai-agent-history-rag.daemon.plist"
        if plist_path.exists():
            subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
            plist_path.unlink(missing_ok=True)
            print_success("Removed launchd service")
        else:
            print_warning("launchd service not found")
    elif system == "Linux":
        service_path = Path.home() / ".config/systemd/user/ai-agent-history-rag.service"
        if service_path.exists():
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", "ai-agent-history-rag"], check=False
            )
            service_path.unlink(missing_ok=True)
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
            print_success("Removed systemd service")
        else:
            print_warning("systemd service not found")
    elif system == "Windows":
        result = subprocess.run(
            ["schtasks", "/Delete", "/TN", "AIAgentHistoryRAG", "/F"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print_success("Removed Windows scheduled task")
        else:
            print_warning("Windows scheduled task not found or could not be removed")
    else:
        print_warning(f"Unsupported platform: {system}")


def remove_data_dir() -> None:
    """Remove the data directory."""
    data_dir = Path.home() / ".claude-history-rag"
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)
        print_success(f"Removed data directory: {data_dir}")
    else:
        print_warning("Data directory not found")


def remove_env_file(project_dir: Path) -> None:
    """Remove the .env file if present."""
    env_path = project_dir / ".env"
    if env_path.exists():
        env_path.unlink(missing_ok=True)
        print_success(f"Removed {env_path}")
    else:
        print_warning("No .env file found")


def run_uninstall() -> int:
    """Run the uninstall wizard."""
    print_header("AI Agent History RAG Uninstall")

    project_dir = get_project_dir()

    if prompt_yes_no("Remove MCP server configuration from clients?", default=True):
        remove_mcp_configs(SERVER_NAME)

    if prompt_yes_no("Stop and remove the daemon service?", default=True):
        uninstall_daemon_service()

    if prompt_yes_no("Delete data directory (~/.claude-history-rag)?", default=True):
        remove_data_dir()

    if prompt_yes_no("Remove project .env file?", default=True):
        remove_env_file(project_dir)

    print_success("Uninstall complete")
    return 0


def main() -> int:
    """Entry point."""
    try:
        return run_uninstall()
    except KeyboardInterrupt:
        print("\nUninstall cancelled.")
        return 1
    except Exception as e:
        print_error(f"Uninstall failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
