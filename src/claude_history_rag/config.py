"""Configuration management."""

import logging
import socket
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from pydantic import ConfigDict, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Optimization interval in seconds (15 minutes)
OPTIMIZE_INTERVAL = 900


def _validate_path_safe(path: Path, allowed_prefixes: list[Path]) -> Path:
    """Validate path doesn't contain traversal and is under allowed prefixes.

    Returns the resolved (canonical) path to prevent TOCTOU vulnerabilities.
    """
    try:
        # Resolve to absolute path to normalize symlinks and .. sequences
        resolved = path.resolve()
    except (OSError, RuntimeError) as e:
        # Handle errors from symlink resolution (e.g., circular symlinks, permission issues)
        raise ValueError(f"Failed to resolve path {path}: {e}") from e

    # Check if under allowed prefix using relative_to (raises ValueError if not)
    for prefix in allowed_prefixes:
        try:
            prefix_resolved = prefix.resolve()
        except (OSError, RuntimeError) as e:
            logger.warning(f"Failed to resolve allowed prefix {prefix}: {e}")
            continue

        try:
            # This will raise ValueError if resolved is not under prefix_resolved
            resolved.relative_to(prefix_resolved)
            # Return resolved path to prevent TOCTOU attacks
            return resolved
        except ValueError:
            # Not under this prefix, try next one
            continue

    raise ValueError(f"Path not under allowed directories: {path}")


def _get_default_machine_id() -> str:
    """Get default machine ID from hostname."""
    try:
        return socket.gethostname()
    except Exception:
        return "unknown-machine"


class Settings(BaseSettings):
    """Application settings from environment variables."""

    db_path: Path = Path.home() / ".claude-history-rag" / "lancedb"
    state_path: Path = Path.home() / ".claude-history-rag" / "state.json"
    projects_path: Path = Path.home() / ".claude" / "projects"
    codex_sessions_path: Path = Path.home() / ".codex" / "sessions"
    codex_state_path: Path = Path.home() / ".claude-history-rag" / "codex_state.json"
    gemini_sessions_path: Path = Path.home() / ".gemini" / "tmp"
    gemini_state_path: Path = Path.home() / ".claude-history-rag" / "gemini_state.json"
    antigravity_sessions_path: Path = Path.home() / ".gemini" / "antigravity" / "conversations"
    antigravity_state_path: Path = Path.home() / ".claude-history-rag" / "antigravity_state.json"

    # ============================================================
    # Client/Server Mode Settings
    # ============================================================
    # If server_url is set, run in CLIENT mode (upload chunks to server)
    # If server_url is None, run in SERVER mode (local processing)
    server_url: str | None = None  # e.g., "http://192.168.1.100:4680"
    machine_id: str = ""  # Will default to hostname if empty
    client_name: str = ""  # Optional human-friendly label for this client

    # Upload settings (client mode only)
    upload_interval_seconds: int = 300  # 5 minutes batch upload
    upload_retry_count: int = 3  # Retry 3 times with delay
    upload_retry_delay_seconds: int = 30  # 30 seconds between retries
    client_heartbeat_interval_seconds: int = 60  # Client heartbeat interval

    # ============================================================
    # Embedding Settings (OpenAI-compatible API)
    # ============================================================
    # Works with: Ollama, vLLM, text-embeddings-inference, OpenAI, etc.
    # Example URLs:
    #   - Ollama: http://localhost:11434/v1
    #   - vLLM: http://localhost:8000/v1
    #   - OpenAI: https://api.openai.com/v1
    embedding_base_url: str = "http://localhost:11434/v1"  # Default to Ollama
    embedding_model: str = "nomic-embed-text"
    embedding_api_key: str = ""  # Optional, for OpenAI or auth-required endpoints

    # ============================================================
    # General Settings
    # ============================================================
    log_level: str = "INFO"
    debounce_delay: int = 5000  # Debounce delay in milliseconds
    batch_size: int = 32  # Embedding batch size
    max_file_batch_size: int = 50  # Process this many files before GC
    max_chunks_per_file: int = 100  # Split large files into smaller batches
    gc_after_files: bool = True  # Enable garbage collection after file batches
    defer_startup_indexing: bool = False  # If True, skip initial indexing on startup
    startup_indexing_delay_ms: int = 0  # Delay between files during startup (ms, 0=no delay)

    # Optimization Settings
    optimization_cleanup_older_than_seconds: int = 3600  # Default 1 hour
    optimization_delete_unverified: bool = True  # Default to True to reclaim space aggressively

    # ============================================================
    # Status Server Settings
    # ============================================================
    status_server_enabled: bool = True
    status_server_host: str = "127.0.0.1"  # Localhost only for security
    status_server_port: int = 4680  # Dashboard/API port
    status_refresh_interval: int = 5  # seconds for dashboard auto-refresh

    # ============================================================
    # Auth Settings
    # ============================================================
    auth_enabled: bool = True
    server_psk: str = ""  # Optional env override for server PSK
    client_psk: str = ""  # Optional env override for client PSK
    auth_state_path: Path = Path.home() / ".claude-history-rag" / "auth.json"
    client_auth_path: Path = Path.home() / ".claude-history-rag" / "client_auth.json"

    model_config = ConfigDict(
        env_prefix="CLAUDE_HISTORY_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @field_validator("machine_id", mode="after")
    @classmethod
    def validate_machine_id(cls, v: str) -> str:
        """Default machine_id to hostname if empty."""
        if not v or not v.strip():
            return _get_default_machine_id()
        return v.strip()

    @field_validator("client_name", mode="after")
    @classmethod
    def validate_client_name(cls, v: str) -> str:
        """Normalize optional client name."""
        if not v:
            return ""
        return v.strip()

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, v: str) -> str:
        """Validate embedding model name format."""
        if not v or not v.strip():
            raise ValueError("embedding_model cannot be empty")
        return v.strip()

    @field_validator("embedding_base_url")
    @classmethod
    def validate_embedding_base_url(cls, v: str) -> str:
        """Validate embedding base URL format."""
        if not v or not v.strip():
            raise ValueError("embedding_base_url cannot be empty")
        cleaned = v.strip().rstrip("/")
        parsed = urlparse(cleaned)
        if (
            parsed.scheme in ("http", "https")
            and parsed.netloc
            and parsed.netloc.endswith(":11434")
            and parsed.path in ("", "/")
        ):
            parsed = parsed._replace(path="/v1")
            cleaned = urlunparse(parsed)
        return cleaned

    @field_validator("server_url", mode="after")
    @classmethod
    def validate_server_url(cls, v: str | None) -> str | None:
        """Validate server URL format if provided."""
        if v is None or not v.strip():
            return None
        url = v.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError("server_url must start with http:// or https://")
        return url

    @field_validator("upload_interval_seconds")
    @classmethod
    def validate_upload_interval(cls, v: int) -> int:
        """Validate upload interval is reasonable."""
        if v < 30:
            raise ValueError("upload_interval_seconds must be at least 30")
        if v > 3600:
            raise ValueError("upload_interval_seconds must be at most 3600 (1 hour)")
        return v

    @field_validator("upload_retry_count")
    @classmethod
    def validate_upload_retry_count(cls, v: int) -> int:
        """Validate retry count is reasonable."""
        if v < 1:
            raise ValueError("upload_retry_count must be at least 1")
        if v > 10:
            raise ValueError("upload_retry_count must be at most 10")
        return v

    @field_validator("client_heartbeat_interval_seconds")
    @classmethod
    def validate_client_heartbeat_interval(cls, v: int) -> int:
        """Validate heartbeat interval is reasonable."""
        if v < 30:
            raise ValueError("client_heartbeat_interval_seconds must be at least 30")
        if v > 3600:
            raise ValueError("client_heartbeat_interval_seconds must be at most 3600 (1 hour)")
        return v

    @field_validator(
        "db_path",
        "state_path",
        "codex_state_path",
        "gemini_state_path",
        "antigravity_state_path",
        mode="after",
    )
    @classmethod
    def validate_data_paths(cls, v: Path) -> Path:
        """Validate data paths are under ~/.claude-history-rag/."""
        # Allow /data for container deployments
        allowed = [Path.home() / ".claude-history-rag", Path("/data")]
        return _validate_path_safe(v, allowed)

    @field_validator("auth_state_path", "client_auth_path", mode="after")
    @classmethod
    def validate_auth_paths(cls, v: Path) -> Path:
        """Validate auth paths are under ~/.claude-history-rag/."""
        allowed = [Path.home() / ".claude-history-rag", Path("/data")]
        return _validate_path_safe(v, allowed)

    @field_validator("projects_path", mode="after")
    @classmethod
    def validate_projects_path(cls, v: Path) -> Path:
        """Validate projects path is under ~/.claude/."""
        allowed = [Path.home() / ".claude"]
        return _validate_path_safe(v, allowed)

    @field_validator("codex_sessions_path", mode="after")
    @classmethod
    def validate_codex_sessions_path(cls, v: Path) -> Path:
        """Validate Codex sessions path is under ~/.codex/."""
        allowed = [Path.home() / ".codex"]
        return _validate_path_safe(v, allowed)

    @field_validator("gemini_sessions_path", mode="after")
    @classmethod
    def validate_gemini_sessions_path(cls, v: Path) -> Path:
        """Validate Gemini sessions path is under ~/.gemini/."""
        allowed = [Path.home() / ".gemini"]
        return _validate_path_safe(v, allowed)

    @field_validator("antigravity_sessions_path", mode="after")
    @classmethod
    def validate_antigravity_sessions_path(cls, v: Path) -> Path:
        """Validate Antigravity sessions path is under ~/.gemini/."""
        allowed = [Path.home() / ".gemini"]
        return _validate_path_safe(v, allowed)

    @field_validator("debounce_delay")
    @classmethod
    def validate_debounce_delay(cls, v: int) -> int:
        if v < 0:
            raise ValueError("debounce_delay must be non-negative")
        if v > 300000:  # 5 minutes max
            raise ValueError("debounce_delay must be <= 300000ms")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("batch_size must be positive")
        if v > 1000:
            raise ValueError("batch_size must be <= 1000")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper  # Normalize to uppercase

    @model_validator(mode="after")
    def validate_paths_unique(self) -> "Settings":
        """Validate that db_path and state_path are different."""
        if self.db_path == self.state_path:
            raise ValueError(
                f"db_path and state_path must be different: both are set to {self.db_path}"
            )
        if self.codex_state_path == self.state_path:
            raise ValueError(
                "codex_state_path must be different from state_path: "
                f"both are set to {self.state_path}"
            )
        if self.gemini_state_path == self.state_path:
            raise ValueError(
                "gemini_state_path must be different from state_path: "
                f"both are set to {self.state_path}"
            )
        if self.antigravity_state_path == self.state_path:
            raise ValueError(
                "antigravity_state_path must be different from state_path: "
                f"both are set to {self.state_path}"
            )
        return self

    @property
    def is_client_mode(self) -> bool:
        """Check if running in client mode (uploading to remote server)."""
        return self.server_url is not None

    @property
    def is_server_mode(self) -> bool:
        """Check if running in server mode (local processing)."""
        return self.server_url is None


settings = Settings()
