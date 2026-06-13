"""Server-side registry of client connections and reindex acknowledgements."""

import json
import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from claude_history_rag.config import settings

logger = logging.getLogger(__name__)


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
    ):
        raw = value.get(key)
        if isinstance(raw, int | float | bool):
            summary[key] = raw
    status = value.get("status")
    if isinstance(status, str):
        summary["status"] = _sanitize_reason(status, "unknown")
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

    def _load(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            try:
                self._state = json.loads(self.path.read_text())
            except Exception as e:
                logger.warning(f"Failed to load client registry: {e}")
        self._loaded = True

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._state, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save client registry: {e}")

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
                entry["client_name"] = client_name
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
                entry["client_name"] = client_name
            entry["last_seen"] = self._now()
            entry["last_upload_at"] = self._now()
            clients[machine_id] = entry
            self._save()

    def mark_reindex_requested(self) -> str:
        with self._lock:
            self._load()
            timestamp = self._now()
            self._state["reindex_requested_at"] = timestamp
            self._save()
            return timestamp

    def ack_reindex(
        self,
        machine_id: str,
        reindex_requested_at: str | None = None,
        status: str | None = None,
        reason: str | None = None,
    ) -> None:
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
            if status:
                entry["reindex_ack_status"] = _sanitize_reason(status, "queued")
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
                entry["client_name"] = client_name
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
                entry["client_name"] = client_name
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
            reindex_requested_at = self._state.get("reindex_requested_at")
            if not reindex_requested_at:
                return False, None

            entry = self._state.get("clients", {}).get(machine_id, {})
            ack_for = entry.get("reindex_ack_for")
            last_ack = entry.get("last_reindex_ack")

            if ack_for == reindex_requested_at:
                return False, reindex_requested_at
            if last_ack and reindex_requested_at and last_ack > reindex_requested_at:
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
                    status_label = "Healthy"

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
