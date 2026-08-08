"""Server-side registry of client connections and reindex acknowledgements."""

import json
import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from claude_history_rag import durable_io
from claude_history_rag.config import settings

logger = logging.getLogger(__name__)
MAX_CLIENT_REGISTRY_BYTES = 8 * 1024 * 1024


def _is_canonical_reindex_identity(value: object) -> bool:
    """Return whether value is an exact server-issued UTC request identity."""
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return (
        parsed.tzinfo is not None
        and parsed.utcoffset() == timedelta(0)
        and parsed.astimezone(timezone.utc).isoformat() == value
    )


def _sanitize_reason(value: str, default: str = "error") -> str:
    """Return a bounded status reason suitable for dashboard/API payloads."""
    reason = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or default)).strip("_")
    return (reason or default)[:120]


def _safe_scalar(value: Any) -> Any:
    """Return small scalar values; redact arbitrary strings."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        return _sanitize_reason(value, "value")
    return type(value).__name__


def _safe_client_name(value: object) -> str:
    """Preserve display names while removing log/control injection characters.

    The value arrives from request JSON, so it is coerced before substitution. A
    sanitizer that raises on an unexpected type turns a rejectable request into
    a server error, and leaves the partially applied identity writes above it
    committed while the writes below it never run.
    """
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(value)).strip()[:256]


def _safe_heartbeat_section(value: Any) -> dict[str, Any] | None:
    """Summarize heartbeat sections without retaining arbitrary diagnostics."""
    if not isinstance(value, dict):
        return None
    summary: dict[str, Any] = {}
    for key in (
        "size",
        "queued",
        "pending",
        "failed",
        "failed_count",
        "queue_size",
        "queue_max_size",
        "files_indexed",
        "files_pending",
        "files_failed",
        "total",
        "count",
        "memory_mb",
        "cpu_percent",
        "pending_uploads",
        "pending_uploads_oldest_age_sec",
        "active_claims",
        "blocked_records",
        "retry_total",
        "failed_files_count",
        "debounce_ms",
        "generation",
        "catchup_failure_count",
        "catchup_failure_oldest_age_sec",
        "reindex_generation",
    ):
        raw = value.get(key)
        if isinstance(raw, int | float | bool):
            summary[key] = raw
    for key in (
        "status",
        "last_indexed_file_hash",
        "required_at",
        "ack_at",
        "reindex_status",
        "source_name",
        "projects_path_hash",
        "storage_backend",
        "embedding_mode",
    ):
        raw = value.get(key)
        if isinstance(raw, str):
            summary[key] = (
                raw
                if re.fullmatch(r"[A-Za-z0-9_.:+-]{1,256}", raw)
                else _sanitize_reason(raw, "unknown")
            )
    reasons = value.get("catchup_failure_reasons")
    if isinstance(reasons, dict):
        summary["catchup_failure_reasons"] = {
            _sanitize_reason(reason, "unknown"): count
            for reason, count in reasons.items()
            if isinstance(reason, str) and isinstance(count, int) and count >= 0
        }
    client_state = value.get("client_state")
    if isinstance(client_state, dict):
        safe_client_state = _safe_heartbeat_section(client_state)
        if safe_client_state:
            summary["client_state"] = safe_client_state
    return summary or None


class ClientRegistry:
    """Track client activity and reindex acknowledgements."""

    def __init__(self, path: Path | None = None):
        self.path = path or (settings.state_path.parent / "client_registry.json")
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {
            "reindex_requested_at": None,
            "clients": {},
        }
        self._loaded = False

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {"reindex_requested_at": None, "clients": {}}

    @classmethod
    def _parse_state(cls, raw: str) -> dict[str, Any]:
        loaded = json.loads(raw)
        if not isinstance(loaded, dict) or not isinstance(loaded.get("clients", {}), dict):
            raise ValueError("client registry state must contain a clients object")
        return loaded

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            if durable_io.durable_file_exists(self.path, durable_root=self.path.parent):
                self._state = self._parse_state(
                    durable_io.read_text(
                        self.path,
                        durable_root=self.path.parent,
                        max_bytes=MAX_CLIENT_REGISTRY_BYTES,
                    )
                )
        except Exception as e:
            logger.error(
                "Failed to load client registry: reason=load_failed error_type=%s",
                type(e).__name__,
            )
            raise RuntimeError("durable client registry could not be loaded") from e
        self._loaded = True

    def _save(self) -> None:
        try:
            durable_io.atomic_write_text(
                self.path,
                json.dumps(self._state, indent=2, default=str),
                durable_root=self.path.parent,
            )
        except Exception as e:
            try:
                self._state = (
                    self._parse_state(
                        durable_io.read_text(
                            self.path,
                            durable_root=self.path.parent,
                            max_bytes=MAX_CLIENT_REGISTRY_BYTES,
                        )
                    )
                    if durable_io.durable_file_exists(self.path, durable_root=self.path.parent)
                    else self._empty_state()
                )
            except Exception:
                self._state = self._empty_state()
            logger.error(
                "Failed to save client registry: reason=save_failed error_type=%s",
                type(e).__name__,
            )
            raise RuntimeError("durable client registry could not be saved") from e

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def register_client(self, machine_id: str, client_name: str | None = None) -> None:
        with self._lock:
            self._load()
            clients = self._state.setdefault("clients", {})
            entry = clients.get(machine_id) or {}
            if "first_seen" not in entry:
                entry["first_seen"] = self._now()
            if client_name:
                entry["client_name"] = _safe_client_name(client_name)
            entry["last_seen"] = self._now()
            clients[machine_id] = entry
            self._save()

    def set_client_key_hash(self, machine_id: str, key_hash: str, key_id: str | None) -> None:
        with self._lock:
            self._load()
            clients = self._state.setdefault("clients", {})
            entry = clients.get(machine_id) or {}
            entry["key_hash"] = key_hash
            entry["key_id"] = key_id
            entry["key_status"] = "current"
            entry["last_key_update_at"] = self._now()
            clients[machine_id] = entry
            self._save()

    def set_client_identity_hash(self, machine_id: str, identity_hash: str) -> None:
        with self._lock:
            self._load()
            clients = self._state.setdefault("clients", {})
            entry = clients.get(machine_id) or {}
            entry["identity_hash"] = identity_hash
            entry["last_identity_update_at"] = self._now()
            clients[machine_id] = entry
            self._save()

    def get_client_identity_hash(self, machine_id: str) -> str | None:
        with self._lock:
            self._load()
            entry = self._state.get("clients", {}).get(machine_id) or {}
            return entry.get("identity_hash")

    def get_client_key_hash(self, machine_id: str) -> str | None:
        with self._lock:
            self._load()
            entry = self._state.get("clients", {}).get(machine_id) or {}
            return entry.get("key_hash")

    def get_client_key_id(self, machine_id: str) -> str | None:
        with self._lock:
            self._load()
            entry = self._state.get("clients", {}).get(machine_id) or {}
            return entry.get("key_id")

    def set_key_status(self, machine_id: str, status: str, message: str | None = None) -> None:
        with self._lock:
            self._load()
            clients = self._state.setdefault("clients", {})
            entry = clients.get(machine_id) or {}
            entry["key_status"] = status
            entry["last_key_status_at"] = self._now()
            if message:
                entry["key_status_message"] = message
            clients[machine_id] = entry
            self._save()

    def record_key_rotation_error(self, machine_id: str, error: str) -> None:
        with self._lock:
            self._load()
            clients = self._state.setdefault("clients", {})
            entry = clients.get(machine_id) or {}
            entry["key_rotation_error"] = _sanitize_reason(error, "rotation_failed")
            entry["key_status"] = "error"
            entry["last_key_error_at"] = self._now()
            clients[machine_id] = entry
            self._save()

    def mark_key_rotated(self, machine_id: str, key_id: str | None) -> None:
        with self._lock:
            self._load()
            clients = self._state.setdefault("clients", {})
            entry = clients.get(machine_id) or {}
            entry["key_status"] = "current"
            entry["key_id"] = key_id
            entry["last_key_rotated_at"] = self._now()
            entry.pop("key_rotation_error", None)
            clients[machine_id] = entry
            self._save()

    def set_rotation_awaiting(self, machine_id: str) -> None:
        with self._lock:
            self._load()
            clients = self._state.setdefault("clients", {})
            entry = clients.get(machine_id) or {}
            entry["key_status"] = "awaiting"
            entry["last_key_status_at"] = self._now()
            clients[machine_id] = entry
            self._save()

    def set_key_old(self, machine_id: str) -> None:
        with self._lock:
            self._load()
            clients = self._state.setdefault("clients", {})
            entry = clients.get(machine_id) or {}
            entry["key_status"] = "old"
            entry["last_key_status_at"] = self._now()
            clients[machine_id] = entry
            self._save()

    def record_upload(self, machine_id: str, client_name: str | None = None) -> None:
        with self._lock:
            self._load()
            clients = self._state.setdefault("clients", {})
            entry = clients.get(machine_id) or {}
            if "first_seen" not in entry:
                entry["first_seen"] = self._now()
            if client_name:
                entry["client_name"] = _safe_client_name(client_name)
            entry["last_seen"] = self._now()
            entry["last_upload_at"] = self._now()
            clients[machine_id] = entry
            self._save()

    def mark_reindex_requested(self) -> str:
        with self._lock:
            self._load()
            timestamp = self._now()
            previous = self._state.get("reindex_requested_at")
            if _is_canonical_reindex_identity(previous):
                previous_time = datetime.fromisoformat(previous)
                timestamp_time = datetime.fromisoformat(timestamp)
                if timestamp_time <= previous_time:
                    timestamp = (previous_time + timedelta(microseconds=1)).isoformat()
            self._state["reindex_requested_at"] = timestamp
            clients = self._state.setdefault("clients", {})
            for entry in clients.values():
                if not isinstance(entry, dict):
                    continue
                entry.pop("reindex_ack_for", None)
                entry.pop("reindex_ack_status", None)
                entry.pop("reindex_ack_reason", None)
                entry.pop("last_reindex_ack", None)
            self._save()
            return timestamp

    def ack_reindex(
        self,
        machine_id: str,
        reindex_requested_at: str | None = None,
        status: str | None = None,
        reason: str | None = None,
    ) -> None:
        ack_status = "queued" if status is None else status
        if not isinstance(ack_status, str) or ack_status not in {"queued", "completed"}:
            raise ValueError("reindex acknowledgement status must be queued or completed")
        if reindex_requested_at is not None and not _is_canonical_reindex_identity(
            reindex_requested_at
        ):
            raise ValueError("reindex acknowledgement must carry a canonical request identity")
        if ack_status == "completed" and reindex_requested_at is None:
            raise ValueError("completed reindex acknowledgement must carry a request identity")

        with self._lock:
            self._load()
            clients = self._state.setdefault("clients", {})
            entry = clients.get(machine_id) or {}
            if "first_seen" not in entry:
                entry["first_seen"] = self._now()
            entry["last_seen"] = self._now()
            entry["last_reindex_ack"] = self._now()
            if reindex_requested_at:
                entry["reindex_ack_for"] = reindex_requested_at
            entry["reindex_ack_status"] = ack_status
            if reason:
                entry["reindex_ack_reason"] = _sanitize_reason(reason, "reason")
            clients[machine_id] = entry
            self._save()

    def mark_purged(self, machine_id: str, client_name: str | None = None) -> None:
        with self._lock:
            self._load()
            clients = self._state.setdefault("clients", {})
            entry = clients.get(machine_id) or {}
            if "first_seen" not in entry:
                entry["first_seen"] = self._now()
            if client_name:
                entry["client_name"] = _safe_client_name(client_name)
            entry["last_seen"] = self._now()
            entry["last_purged_at"] = self._now()
            clients[machine_id] = entry
            self._save()

    def record_heartbeat(
        self,
        machine_id: str,
        client_name: str | None = None,
        heartbeat: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._load()
            clients = self._state.setdefault("clients", {})
            entry = clients.get(machine_id) or {}
            if "first_seen" not in entry:
                entry["first_seen"] = self._now()
            if client_name:
                entry["client_name"] = _safe_client_name(client_name)
            entry["last_seen"] = self._now()
            entry["last_heartbeat_at"] = self._now()

            if heartbeat:
                for field in (
                    "client_version",
                    "os",
                    "arch",
                    "python_version",
                    "hostname",
                    "timezone",
                    "heartbeat_interval_s",
                    "status",
                    "last_upload_at",
                    "last_indexed_at",
                ):
                    if field in heartbeat and heartbeat[field] is not None:
                        entry[field] = _safe_scalar(heartbeat[field])

                for field in (
                    "queue",
                    "watcher",
                    "reindex",
                    "errors",
                    "config",
                    "doctor",
                    "resources",
                ):
                    if field in heartbeat and heartbeat[field] is not None:
                        safe_section = _safe_heartbeat_section(heartbeat[field])
                        if safe_section is not None:
                            entry[field] = safe_section

            clients[machine_id] = entry
            self._save()

    def get_reindex_status(self, machine_id: str) -> tuple[bool, str | None]:
        with self._lock:
            self._load()
            request_is_absent = (
                "reindex_requested_at" not in self._state
                or self._state["reindex_requested_at"] is None
            )
            if request_is_absent:
                return False, None
            reindex_requested_at = self._state["reindex_requested_at"]
            if not _is_canonical_reindex_identity(reindex_requested_at):
                # A malformed persisted request must never become an implicit
                # completion authority. Replace it with a fresh server-issued
                # identity and durably require every client to replay it.
                reindex_requested_at = self._now()
                self._state["reindex_requested_at"] = reindex_requested_at
                clients = self._state.get("clients", {})
                if isinstance(clients, dict):
                    for entry in clients.values():
                        if isinstance(entry, dict):
                            entry.pop("reindex_ack_for", None)
                            entry.pop("reindex_ack_status", None)
                self._save()
                return True, reindex_requested_at

            clients = self._state.get("clients", {})
            entry = clients.get(machine_id, {}) if isinstance(clients, dict) else {}
            if not isinstance(entry, dict):
                return True, reindex_requested_at
            ack_for = entry.get("reindex_ack_for")
            ack_status = entry.get("reindex_ack_status")

            if (
                _is_canonical_reindex_identity(ack_for)
                and ack_for == reindex_requested_at
                and ack_status == "completed"
            ):
                return False, reindex_requested_at
            return True, reindex_requested_at

    def get_client_status(self, stale_days: int = 7) -> dict[str, Any]:
        with self._lock:
            self._load()
            reindex_requested_at = self._state.get("reindex_requested_at")
            clients = self._state.get("clients", {})
            now = datetime.now(timezone.utc)
            stale_delta = timedelta(days=stale_days)
            heartbeat_delta = timedelta(minutes=5)
            entries: list[dict[str, Any]] = []

            for machine_id, entry in clients.items():
                last_seen_raw = entry.get("last_seen")
                last_heartbeat_raw = entry.get("last_heartbeat_at") or last_seen_raw
                stale = False
                if last_seen_raw:
                    try:
                        last_seen = datetime.fromisoformat(last_seen_raw)
                        stale = now - last_seen > stale_delta
                    except Exception:
                        stale = False

                disconnected = False
                if last_heartbeat_raw:
                    try:
                        last_heartbeat = datetime.fromisoformat(last_heartbeat_raw)
                        disconnected = now - last_heartbeat > heartbeat_delta
                    except Exception:
                        disconnected = False
                else:
                    disconnected = True

                heartbeat_status = entry.get("status") or "ok"
                if disconnected:
                    status_label = "Disconnected"
                elif heartbeat_status == "degraded":
                    status_label = "Degraded"
                elif heartbeat_status == "ok":
                    status_label = "Healthy"
                else:
                    status_label = "Unknown"

                reindex_pending, _ = self.get_reindex_status(machine_id)

                entries.append(
                    {
                        "machine_id": machine_id,
                        "first_seen": entry.get("first_seen"),
                        "client_name": entry.get("client_name"),
                        "last_seen": last_seen_raw,
                        "last_upload_at": entry.get("last_upload_at"),
                        "last_heartbeat_at": entry.get("last_heartbeat_at"),
                        "disconnected": disconnected,
                        "status_label": status_label,
                        "last_reindex_ack": entry.get("last_reindex_ack"),
                        "reindex_ack_for": entry.get("reindex_ack_for"),
                        "reindex_ack_status": entry.get("reindex_ack_status"),
                        "reindex_ack_reason": entry.get("reindex_ack_reason"),
                        "last_purged_at": entry.get("last_purged_at"),
                        "client_version": entry.get("client_version"),
                        "os": entry.get("os"),
                        "arch": entry.get("arch"),
                        "python_version": entry.get("python_version"),
                        "hostname": entry.get("hostname"),
                        "timezone": entry.get("timezone"),
                        "heartbeat_interval_s": entry.get("heartbeat_interval_s"),
                        "status": entry.get("status"),
                        "last_indexed_at": entry.get("last_indexed_at"),
                        "queue": entry.get("queue"),
                        "watcher": entry.get("watcher"),
                        "reindex": entry.get("reindex"),
                        "errors": entry.get("errors"),
                        "config": entry.get("config"),
                        "doctor": entry.get("doctor"),
                        "resources": entry.get("resources"),
                        "stale": stale,
                        "reindex_pending": reindex_pending,
                        "key_status": entry.get("key_status"),
                        "key_id": entry.get("key_id"),
                        "key_status_message": entry.get("key_status_message"),
                        "key_rotation_error": entry.get("key_rotation_error"),
                        "last_key_status_at": entry.get("last_key_status_at"),
                        "last_key_rotated_at": entry.get("last_key_rotated_at"),
                    }
                )

            entries.sort(key=lambda e: e.get("last_seen") or "", reverse=True)

            return {
                "total": len(entries),
                "stale_after_days": stale_days,
                "reindex_requested_at": reindex_requested_at,
                "clients": entries,
            }


_client_registry: ClientRegistry | None = None


def get_client_registry() -> ClientRegistry:
    global _client_registry
    if _client_registry is None:
        _client_registry = ClientRegistry()
    return _client_registry
