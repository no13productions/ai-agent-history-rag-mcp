"""Docker deployment wizard for the central server."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from claude_history_rag.installer import (
    print_error,
    print_header,
    print_success,
    print_warning,
    prompt_yes_no,
)

CONFIG_PATH = Path.home() / ".claude-history-rag" / "docker.json"
DEFAULT_IMAGE_TAG = "ai-agent-history-rag:local"
DEFAULT_PORT = 4680


def prompt_string(prompt: str, default: str | None = None) -> str:
    """Prompt user for a string value."""
    default_display = f" (default: {default})" if default else ""
    response = input(f"{prompt}{default_display}: ").strip()
    return response if response else (default or "")


def prompt_int(prompt: str, default: int) -> int:
    """Prompt user for an integer value."""
    while True:
        response = prompt_string(prompt, str(default))
        try:
            return int(response)
        except ValueError:
            print_error("Please enter a valid integer")


def prompt_choice(prompt: str, choices: list[str], default_index: int = 0) -> str:
    """Prompt user to choose from a list of options."""
    print(f"\n{prompt}")
    for idx, choice in enumerate(choices, start=1):
        marker = ">" if idx - 1 == default_index else " "
        print(f"  {marker} [{idx}] {choice}")

    while True:
        response = input(
            f"\nEnter choice [1-{len(choices)}] (default: {default_index + 1}): "
        ).strip()
        if not response:
            return choices[default_index]
        try:
            selected = int(response) - 1
            if 0 <= selected < len(choices):
                return choices[selected]
        except ValueError:
            pass
    print_error("Please enter a valid choice number")


def parse_label_list(raw: str) -> dict[str, str]:
    """Parse comma-separated key=value pairs into a dict."""
    labels: dict[str, str] = {}
    if not raw:
        return labels
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Missing '=' in label: {chunk}")
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError("Label key cannot be empty")
        labels[key] = value
    return labels


def prompt_labels(existing: dict[str, str] | None = None) -> dict[str, str]:
    """Prompt for optional Docker labels."""
    labels = dict(existing or {})
    if not prompt_yes_no("Add Docker labels?", default=bool(labels)):
        return labels

    method = prompt_choice(
        "How do you want to enter labels?",
        ["Paste comma-separated key=value list", "Add labels one-by-one"],
        default_index=0,
    )

    if method.startswith("Paste"):
        while True:
            raw = prompt_string("Labels (comma-separated key=value)", "" if not labels else "")
            try:
                parsed = parse_label_list(raw)
            except ValueError as exc:
                print_error(str(exc))
                continue
            labels.update(parsed)
            break
    else:
        while True:
            key = prompt_string("Label key")
            value = prompt_string("Label value")
            labels[key] = value
            if not prompt_yes_no("Add another label?", default=False):
                break

    return labels


def prompt_additional_networks(existing: list[str] | None = None) -> list[str]:
    """Prompt for additional existing Docker networks to attach."""
    networks = get_docker_networks()
    if not networks:
        return []
    defaults = ",".join(existing or [])
    if not prompt_yes_no("Attach to additional existing networks?", default=bool(existing)):
        return existing or []

    print("Available networks:")
    print("  " + ", ".join(networks))

    while True:
        raw = prompt_string("Extra network names (comma-separated, blank for none)", defaults)
        if not raw:
            return []
        selected = [name.strip() for name in raw.split(",") if name.strip()]
        unknown = [name for name in selected if name not in networks]
        if unknown:
            print_error(f"Unknown networks: {', '.join(unknown)}")
            continue
        deduped: list[str] = []
        for name in selected:
            if name not in deduped:
                deduped.append(name)
        return deduped


def prompt_required_url(prompt: str, example: str, default: str | None = None) -> str:
    """Prompt for a non-empty URL."""
    while True:
        value = prompt_string(f"{prompt} (e.g., {example})", default)
        if value:
            return value
        print_error("This value is required.")


def ensure_writable(path: Path) -> bool:
    """Check if a path is writable."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".write-test"
        test_file.write_text("ok")
        test_file.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def check_docker_available() -> bool:
    """Validate Docker engine is running and user can access it."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print_error("Docker CLI not found. Please install Docker first.")
        return False

    if result.returncode != 0:
        print_error("Docker is not accessible or the daemon is not running.")
        print_warning(result.stderr.strip() or result.stdout.strip())
        return False

    return True


def get_docker_networks() -> list[str]:
    """Return a list of Docker network names."""
    result = subprocess.run(
        ["docker", "network", "ls", "--format", "{{.Name}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def create_docker_network(
    name: str,
    driver: str,
    subnet: str | None = None,
    gateway: str | None = None,
    parent: str | None = None,
) -> bool:
    """Create a Docker network."""
    command = ["docker", "network", "create", "-d", driver]
    if subnet:
        command.extend(["--subnet", subnet])
    if gateway:
        command.extend(["--gateway", gateway])
    if parent:
        command.extend(["-o", f"parent={parent}"])
    command.append(name)

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print_success(f"Created Docker network: {name}")
        return True
    print_error(f"Failed to create network: {result.stderr.strip()}")
    return False


def get_used_docker_ports() -> list[int]:
    """Return a list of host ports published by Docker containers."""
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Ports}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []

    ports: set[int] = set()
    for line in result.stdout.splitlines():
        for part in line.split(","):
            part = part.strip()
            if "->" in part and ":" in part:
                left = part.split("->", 1)[0]
                if ":" in left:
                    host_port = left.rsplit(":", 1)[-1]
                    if host_port.isdigit():
                        ports.add(int(host_port))
    return sorted(ports)


def port_is_available(port: int) -> bool:
    """Check if a host port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def format_ports_line(ports: list[int]) -> str:
    """Format ports in a single line."""
    if not ports:
        return "None"
    return " _ ".join(str(port) for port in ports)


def normalize_embedding_base_url(url: str) -> str:
    """Append /v1 for Ollama-style base URLs if missing."""
    cleaned = url.strip().rstrip("/")
    parsed = urlparse(cleaned)
    if (
        parsed.scheme in ("http", "https")
        and parsed.netloc
        and parsed.netloc.endswith(":11434")
        and parsed.path in ("", "/")
    ):
        parsed = parsed._replace(path="/v1")
        return urlunparse(parsed)
    return cleaned


def format_mount(host_path: str, container_path: str) -> str:
    """Format a volume mount for compose."""
    mount = f"{host_path}:{container_path}"
    if " " in mount:
        return f'"{mount}"'
    return mount


def load_config() -> dict[str, Any] | None:
    """Load wizard configuration if present."""
    if not CONFIG_PATH.exists():
        return None
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_config(config: dict[str, Any]) -> None:
    """Save wizard configuration."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def summarize_config(config: dict[str, Any]) -> None:
    """Print a summary of the current config."""
    print("\nConfiguration summary:")
    print(f"  Image tag: {config['image_tag']}")
    print(f"  Network mode: {config['network_mode']}")
    if config.get("network_name"):
        print(f"  Network name: {config['network_name']}")
    if config.get("extra_networks"):
        print(f"  Extra networks: {', '.join(config['extra_networks'])}")
    if config.get("publish_port") is not None:
        print(f"  Published port: {config['publish_port']}")
    else:
        print("  Published port: none")
    print(f"  Embedding base URL: {config['embedding_base_url']}")
    print(f"  Embedding model: {config['embedding_model']}")
    if config.get("volume_mode"):
        print(f"  Volume mode: {config['volume_mode']}")
    if config.get("db_dir"):
        print(f"  DB dir: {config['db_dir']}")
    if config.get("state_dir"):
        print(f"  State dir: {config['state_dir']}")
    if config.get("labels"):
        print(f"  Labels: {len(config['labels'])} configured")


def write_compose_file(config: dict[str, Any], compose_path: Path) -> None:
    """Write docker-compose.yml based on config."""
    env_lines = [
        f"      - CLAUDE_HISTORY_RAG_EMBEDDING_BASE_URL={config['embedding_base_url']}",
        f"      - CLAUDE_HISTORY_RAG_EMBEDDING_MODEL={config['embedding_model']}",
        f"      - CLAUDE_HISTORY_RAG_EMBEDDING_API_KEY={config.get('embedding_api_key', '')}",
        "      - CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST=0.0.0.0",
        f"      - CLAUDE_HISTORY_RAG_STATUS_SERVER_PORT={DEFAULT_PORT}",
        "      - CLAUDE_HISTORY_RAG_DB_PATH=/data/db/lancedb",
        "      - CLAUDE_HISTORY_RAG_STATE_PATH=/data/state/state.json",
        "      - CLAUDE_HISTORY_RAG_AUTH_STATE_PATH=/data/state/auth.json",
    ]

    volume_lines = []
    volume_defs = []

    if config["volume_mode"] == "docker":
        volume_lines = [
            "      - ai-agent-history-rag-db:/data/db",
            "      - ai-agent-history-rag-state:/data/state",
        ]
        volume_defs = [
            "  ai-agent-history-rag-db:",
            "  ai-agent-history-rag-state:",
        ]
    else:
        db_dir = config["db_dir"]
        state_dir = config["state_dir"]
        volume_lines = [
            f"      - {format_mount(db_dir, '/data/db')}",
            f"      - {format_mount(state_dir, '/data/state')}",
        ]

    ports_section = ""
    if config.get("publish_port") is not None:
        ports_section = f'    ports:\n      - "{config["publish_port"]}:{DEFAULT_PORT}"\n'

    extra_hosts_section = ""
    if "host.docker.internal" in config["embedding_base_url"]:
        extra_hosts_section = '    extra_hosts:\n      - "host.docker.internal:host-gateway"\n'

    networks_section = ""
    networks_def = ""
    network_name = config.get("network_name")
    extra_networks = config.get("extra_networks") or []
    if network_name or extra_networks:
        networks_section = "    networks:\n"
        if network_name:
            networks_section += "      - ai-agent-history-rag-net\n"
        for name in extra_networks:
            networks_section += f"      - {name}\n"
        networks_def = "networks:\n"
        if network_name:
            networks_def += "  ai-agent-history-rag-net:\n"
            if config.get("network_external"):
                networks_def += f"    external: true\n    name: {network_name}\n"
            else:
                networks_def += f"    name: {network_name}\n"
                networks_def += f"    driver: {config['network_mode']}\n"
        for name in extra_networks:
            networks_def += f"  {name}:\n    external: true\n"

    labels_section = ""
    labels = config.get("labels") or {}
    if labels:
        labels_section = "    labels:\n"
        for key, value in labels.items():
            labels_section += f'      {key}: "{value}"\n'

    compose = f"""# AI Agent History RAG Server - Docker Compose
# Generated by ai-agent-history-rag-docker

services:
  ai-agent-history-rag-server:
    image: {config["image_tag"]}
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ai-agent-history-rag-server
    environment:
{os.linesep.join(env_lines)}
    volumes:
{os.linesep.join(volume_lines)}
{ports_section}{extra_hosts_section}{labels_section}    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{DEFAULT_PORT}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    restart: unless-stopped
{networks_section}
"""

    if volume_defs:
        compose += "\nvolumes:\n" + "\n".join(volume_defs) + "\n"
    if networks_def:
        compose += "\n" + networks_def

    compose_path.write_text(compose)
    print_success(f"Wrote {compose_path}")


def run_health_check(port: int) -> None:
    """Run a basic health check against the status server."""
    url = f"http://127.0.0.1:{port}/health"
    print(f"Running health check: {url}")
    start = time.time()
    timeout = 60
    while time.time() - start < timeout:
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                print_success("Health check passed")
                return
        except httpx.RequestError:
            pass
        time.sleep(2)
    print_warning("Health check did not pass within 60s")


def run_wizard() -> int:
    """Run the Docker deployment wizard."""
    print_header("AI Agent History RAG Docker Wizard")

    if not check_docker_available():
        return 1

    project_dir = Path.cwd()
    config = load_config()
    reuse_config = None
    if config:
        print_warning(f"Found existing config at {CONFIG_PATH}")
        summarize_config(config)
        action = prompt_choice(
            "Use existing configuration?",
            ["Use and deploy", "Modify and deploy", "Start fresh"],
            default_index=0,
        )
        if action == "Use and deploy":
            reuse_config = config
            use_config = config
        elif action == "Modify and deploy":
            use_config = config
        else:
            use_config = {}
    else:
        use_config = {}

    print_header("Networking")
    if reuse_config:
        defaults = {
            "image_tag": DEFAULT_IMAGE_TAG,
            "network_mode": "bridge",
            "network_name": None,
            "network_external": False,
            "extra_networks": [],
            "publish_port": None,
            "volume_mode": "docker",
            "db_dir": "",
            "state_dir": "",
            "embedding_base_url": "",
            "embedding_model": "bge-m3",
            "embedding_api_key": "",
            "labels": {},
            "build_now": True,
            "no_cache": False,
        }
        config = {**defaults, **reuse_config}
        if not config.get("embedding_base_url"):
            print_warning("Saved config is missing the embedding base URL.")
            reuse_config = None
        if reuse_config and config.get("volume_mode") in ("local", "custom"):
            db_dir = Path(str(config.get("db_dir", ""))).expanduser()
            state_dir = Path(str(config.get("state_dir", ""))).expanduser()
            if not db_dir.exists() or not state_dir.exists():
                print_warning("Saved config has missing storage paths.")
                reuse_config = None
    else:
        network_mode = prompt_choice(
            "Select network mode",
            ["bridge", "macvlan"],
            default_index=0,
        )

        network_name = None
        network_external = False
        publish_port: int | None = None

        if network_mode == "bridge":
            use_existing = prompt_yes_no(
                "Use or create a custom bridge network?",
                default=bool(use_config.get("network_name")),
            )
            if use_existing:
                join_existing = prompt_yes_no(
                    "Join an existing bridge network?",
                    default=bool(use_config.get("network_external", True)),
                )
                if join_existing:
                    networks = get_docker_networks()
                    if networks:
                        network_name = prompt_choice("Select a network", networks, default_index=0)
                        network_external = True
                    else:
                        print_warning("No networks found. Using default bridge.")
                else:
                    network_name = prompt_string(
                        "New network name",
                        use_config.get("network_name", "ai-agent-history-rag-bridge"),
                    )
                    created = create_docker_network(network_name, "bridge")
                    if created:
                        network_external = True
            publish = prompt_yes_no(
                "Publish the status server port?",
                default=use_config.get("publish_port") is not None or use_config == {},
            )
            if publish:
                used_ports = get_used_docker_ports()
                print(f"Unavailable Docker ports: {format_ports_line(used_ports)}")
                default_port = DEFAULT_PORT
                if not port_is_available(DEFAULT_PORT) or DEFAULT_PORT in used_ports:
                    default_port = use_config.get("publish_port") or 5000
                while True:
                    publish_port = prompt_int("Host port to publish", default_port)
                    if not port_is_available(publish_port):
                        print_error(f"Port {publish_port} is already in use on this host.")
                        continue
                    break
        else:
            join_existing = prompt_yes_no(
                "Join an existing macvlan network?", default=bool(use_config.get("network_name"))
            )
            if join_existing:
                networks = get_docker_networks()
                if networks:
                    network_name = prompt_choice("Select a network", networks, default_index=0)
                    network_external = True
                else:
                    print_warning("No networks found. Creating a new macvlan network.")
                    join_existing = False
            if not join_existing:
                network_name = prompt_string(
                    "New macvlan network name",
                    use_config.get("network_name", "ai-agent-history-rag-macvlan"),
                )
                subnet = prompt_string("Subnet (CIDR, e.g., 192.168.4.0/24)")
                gateway = prompt_string("Gateway (e.g., 192.168.4.1)")
                parent = prompt_string("Parent interface (e.g., eth0)")
                created = create_docker_network(
                    network_name,
                    "macvlan",
                    subnet=subnet,
                    gateway=gateway,
                    parent=parent,
                )
                if created:
                    network_external = True

        extra_networks = prompt_additional_networks(use_config.get("extra_networks"))
        if network_name and network_name in extra_networks:
            extra_networks = [name for name in extra_networks if name != network_name]

        print_header("Storage")
        storage_choices = ["docker", "local", "custom"]
        default_storage = use_config.get("volume_mode", "docker")
        default_index = (
            storage_choices.index(default_storage) if default_storage in storage_choices else 0
        )
        volume_mode = prompt_choice(
            "Select storage strategy",
            storage_choices,
            default_index=default_index,
        )

        db_dir = ""
        state_dir = ""
        if volume_mode == "docker":
            print_success("Using Docker-managed volumes for database and state.")
        else:
            if volume_mode == "local":
                base_dir = project_dir / "docker-data"
                db_dir = str(base_dir / "db")
                state_dir = str(base_dir / "state")
            else:
                db_dir = prompt_string("Database folder path", use_config.get("db_dir"))
                state_dir = prompt_string("State folder path", use_config.get("state_dir"))

            for label, path_str in [("Database", db_dir), ("State", state_dir)]:
                path = Path(path_str).expanduser()
                if not path.exists():
                    create = prompt_yes_no(
                        f"{label} folder does not exist. Create it?",
                        default=True,
                    )
                    if create:
                        try:
                            path.mkdir(parents=True, exist_ok=True)
                        except OSError as e:
                            print_error(f"Failed to create {label} folder: {e}")
                            return 1
                    else:
                        print_error(f"{label} folder is required.")
                        return 1
                if not ensure_writable(path):
                    print_error(f"{label} folder is not writable: {path}")
                    return 1
                if label == "Database":
                    db_dir = str(path)
                else:
                    state_dir = str(path)

        print_header("Embedding Settings")
        embedding_base_url = prompt_required_url(
            "Embedding base URL",
            "http://192.168.4.204:11434/v1",
            use_config.get("embedding_base_url") or None,
        )
        embedding_base_url = normalize_embedding_base_url(embedding_base_url)
        embedding_model = prompt_string(
            "Embedding model", use_config.get("embedding_model", "bge-m3")
        )
        embedding_api_key = prompt_string(
            "Embedding API key (optional)",
            use_config.get("embedding_api_key", ""),
        )

        print_header("Container Labels")
        labels = prompt_labels(use_config.get("labels"))

        print_header("Image Build")
        image_tag = prompt_string("Image tag", use_config.get("image_tag", DEFAULT_IMAGE_TAG))
        build_now = prompt_yes_no("Build the image now?", default=use_config.get("build_now", True))
        no_cache = False
        if build_now:
            no_cache = prompt_yes_no(
                "Build with --no-cache?", default=use_config.get("no_cache", False)
            )

        config = {
            "image_tag": image_tag,
            "network_mode": network_mode,
            "network_name": network_name,
            "network_external": network_external,
            "extra_networks": extra_networks,
            "publish_port": publish_port,
            "volume_mode": volume_mode,
            "db_dir": db_dir,
            "state_dir": state_dir,
            "embedding_base_url": embedding_base_url,
            "embedding_model": embedding_model,
            "embedding_api_key": embedding_api_key,
            "labels": labels,
            "build_now": build_now,
            "no_cache": no_cache,
        }

    build_now = config.get("build_now", False)
    no_cache = config.get("no_cache", False)
    image_tag = config["image_tag"]
    publish_port = config.get("publish_port")
    network_name = config.get("network_name")
    extra_networks = config.get("extra_networks") or []
    volume_mode = config.get("volume_mode")
    db_dir = config.get("db_dir")
    state_dir = config.get("state_dir")
    embedding_base_url = normalize_embedding_base_url(config["embedding_base_url"])
    embedding_model = config["embedding_model"]
    embedding_api_key = config.get("embedding_api_key", "")
    labels = config.get("labels") or {}

    save_config(config)
    summarize_config(config)

    compose_path = project_dir / "docker-compose.yml"
    write_compose_file(config, compose_path)

    print_header("Deploy")
    deploy_method = prompt_choice(
        "Deploy using",
        ["docker compose", "docker run", "Skip deploy"],
        default_index=0,
    )

    if deploy_method == "Skip deploy":
        print_success("Compose file updated. Skipping deployment.")
        return 0

    if build_now:
        if deploy_method == "docker compose":
            command = ["docker", "compose", "build"]
            if no_cache:
                command.append("--no-cache")
        else:
            command = ["docker", "build", "-t", image_tag, "."]
            if no_cache:
                command.insert(2, "--no-cache")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print_error("Image build failed.")
            return 1

    if deploy_method == "docker compose":
        up_cmd = ["docker", "compose", "up", "-d"]
        if not build_now:
            up_cmd.append("--no-build")
        result = subprocess.run(up_cmd, check=False)
    else:
        run_cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            "ai-agent-history-rag-server",
            "--restart",
            "unless-stopped",
        ]
        if publish_port is not None:
            run_cmd.extend(["-p", f"{publish_port}:{DEFAULT_PORT}"])
        if network_name:
            run_cmd.extend(["--network", network_name])
        if volume_mode == "docker":
            run_cmd.extend(["-v", "ai-agent-history-rag-db:/data/db"])
            run_cmd.extend(["-v", "ai-agent-history-rag-state:/data/state"])
        else:
            run_cmd.extend(["-v", f"{db_dir}:/data/db"])
            run_cmd.extend(["-v", f"{state_dir}:/data/state"])

        run_cmd.extend(["-e", f"CLAUDE_HISTORY_RAG_EMBEDDING_BASE_URL={embedding_base_url}"])
        run_cmd.extend(["-e", f"CLAUDE_HISTORY_RAG_EMBEDDING_MODEL={embedding_model}"])
        if embedding_api_key:
            run_cmd.extend(["-e", f"CLAUDE_HISTORY_RAG_EMBEDDING_API_KEY={embedding_api_key}"])
        run_cmd.extend(["-e", "CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST=0.0.0.0"])
        run_cmd.extend(["-e", f"CLAUDE_HISTORY_RAG_STATUS_SERVER_PORT={DEFAULT_PORT}"])
        run_cmd.extend(["-e", "CLAUDE_HISTORY_RAG_DB_PATH=/data/db/lancedb"])
        run_cmd.extend(["-e", "CLAUDE_HISTORY_RAG_STATE_PATH=/data/state/state.json"])
        run_cmd.extend(["-e", "CLAUDE_HISTORY_RAG_AUTH_STATE_PATH=/data/state/auth.json"])
        for key, value in labels.items():
            run_cmd.extend(["--label", f"{key}={value}"])

        if "host.docker.internal" in embedding_base_url and sys.platform.startswith("linux"):
            run_cmd.extend(["--add-host", "host.docker.internal:host-gateway"])

        result = subprocess.run(run_cmd + [image_tag], check=False)

    if result.returncode != 0:
        print_error("Deployment failed.")
        return 1

    if deploy_method == "docker run" and extra_networks:
        for name in extra_networks:
            connect_cmd = ["docker", "network", "connect", name, "ai-agent-history-rag-server"]
            connect_result = subprocess.run(connect_cmd, check=False)
            if connect_result.returncode != 0:
                print_error(f"Failed to connect to network: {name}")
                return 1

    print_success("Deployment complete.")
    if publish_port is not None:
        run_health_check(publish_port)
    else:
        print_warning("No published port, skipping health check.")
    return 0


def main() -> int:
    """Entry point."""
    try:
        return run_wizard()
    except KeyboardInterrupt:
        print("\nDocker wizard cancelled.")
        return 1
    except Exception as e:
        print_error(f"Docker wizard failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
