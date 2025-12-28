"""Authentication and PSK rotation management."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from claude_history_rag.config import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_secure_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2))
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _hash_key(raw_key: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        raw_key.encode("utf-8"),
        salt.encode("utf-8"),
        200_000,
    )
    return digest.hex()


def _secure_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)


def _generate_key() -> str:
    return secrets.token_urlsafe(32)


def _generate_salt() -> str:
    return secrets.token_urlsafe(16)


@dataclass
class AuthCheckResult:
    ok: bool
    key_type: str | None = None  # "active", "pending"
    rotation_required: bool = False
    rotate_to: str | None = None
    rotate_id: str | None = None
    error: str | None = None


class AuthManager:
    def __init__(self, path: Path | None = None):
        self.path = path or settings.auth_state_path
        self._state: dict[str, Any] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        data = _read_json(self.path)
        if data:
            self._state = data
        else:
            self._state = {}
        self._loaded = True

    def _save(self) -> None:
        _write_secure_json(self.path, self._state)

    def ensure_initialized(self) -> None:
        self._load()
        if not settings.auth_enabled:
            self._state.setdefault("auth_enabled", False)
            self._save()
            return

        self._state.setdefault("auth_enabled", True)
        if not self._state.get("dashboard_hash"):
            self._state["dashboard_hash"] = _generate_salt()
            self._save()

        if settings.server_psk.get_secret_value():
            self._state["env_override"] = True
            active = self._state.get("active") or {}
            if not active.get("key_salt"):
                active["key_salt"] = _generate_salt()
            active["key_hash"] = _hash_key(settings.server_psk.get_secret_value(), active["key_salt"])
            active["created_at"] = active.get("created_at") or _now()
            active["key_id"] = active.get("key_id") or active["key_hash"][:12]
            active["key_plain"] = settings.server_psk.get_secret_value()
            self._state["active"] = active
            self._save()
            return

        self._state["env_override"] = False

        if "active" not in self._state:
            raw = _generate_key()
            salt = _generate_salt()
            self._state["active"] = {
                "key_hash": _hash_key(raw, salt),
                "key_salt": salt,
                "key_plain": raw,
                "created_at": _now(),
                "key_id": _hash_key(raw, salt)[:12],
            }
            self._save()

    def is_env_override(self) -> bool:
        self._load()
        return bool(self._state.get("env_override"))

    def auth_enabled(self) -> bool:
        self._load()
        return bool(self._state.get("auth_enabled", True))

    def get_active_key_plain(self) -> str | None:
        self._load()
        active = self._state.get("active") or {}
        return active.get("key_plain")

    def get_pending_key_plain(self) -> str | None:
        self._load()
        pending = self._state.get("pending") or {}
        return pending.get("key_plain")

    def get_rotation_state(self) -> dict[str, Any]:
        self._load()
        return {
            "active": self._state.get("active"),
            "pending": self._state.get("pending"),
            "rotation": self._state.get("rotation"),
            "auth_enabled": self._state.get("auth_enabled", True),
            "env_override": self._state.get("env_override", False),
            "dashboard_hash": self._state.get("dashboard_hash"),
        }

    def _allowlist_expired(self) -> bool:
        pending = self._state.get("pending") or {}
        expires_at = _parse_time(pending.get("allowlist_expires_at"))
        if not expires_at:
            return True
        return datetime.now(timezone.utc) > expires_at

    def _is_allowlisted(self, machine_id: str) -> bool:
        pending = self._state.get("pending") or {}
        allowlist = pending.get("allowlist") or []
        if machine_id in allowlist and not self._allowlist_expired():
            return True
        return False

    def _derive_client_hash(self, raw_key: str, client_id: str, base_salt: str) -> str:
        salt = f"{client_id}:{base_salt}"
        return _hash_key(raw_key, salt)

    def validate_key(
        self,
        raw_key: str,
        machine_id: str | None,
        registry: Any | None = None,
    ) -> AuthCheckResult:
        self.ensure_initialized()
        if not self.auth_enabled():
            return AuthCheckResult(ok=True, key_type="active")

        active = self._state.get("active") or {}
        pending = self._state.get("pending") or {}

        active_salt = active.get("key_salt") or ""
        pending_salt = pending.get("key_salt") or ""

        base_active_hash = _hash_key(raw_key, active_salt) if active_salt else ""
        base_pending_hash = _hash_key(raw_key, pending_salt) if pending_salt else ""

        is_pending = pending and _secure_compare(base_pending_hash, pending.get("key_hash", ""))
        is_active = _secure_compare(base_active_hash, active.get("key_hash", ""))

        rotation_active = bool(pending)
        if is_pending:
            if machine_id and registry:
                derived = self._derive_client_hash(raw_key, machine_id, pending_salt)
                stored = registry.get_client_key_hash(machine_id)
                stored_key_id = registry.get_client_key_id(machine_id)
                if stored and stored_key_id == pending.get("key_id"):
                    if not _secure_compare(derived, stored):
                        return AuthCheckResult(ok=False, error="invalid_client_key")
                registry.set_client_key_hash(machine_id, derived, pending.get("key_id"))
            if machine_id and pending.get("allowlist"):
                if machine_id in pending.get("allowlist", []):
                    self.remove_from_allowlist(machine_id)
                    pending = self._state.get("pending") or {}
                    if not pending.get("allowlist"):
                        self.promote_pending()
            return AuthCheckResult(
                ok=True,
                key_type="pending",
                rotation_required=False,
                rotate_id=pending.get("key_id"),
            )

        if is_active:
            if rotation_active:
                if machine_id and registry and self._is_allowlisted(machine_id):
                    derived = self._derive_client_hash(raw_key, machine_id, active_salt)
                    stored = registry.get_client_key_hash(machine_id)
                    stored_key_id = registry.get_client_key_id(machine_id)
                    if stored and stored_key_id == active.get("key_id"):
                        if not _secure_compare(derived, stored):
                            return AuthCheckResult(ok=False, error="invalid_client_key")
                    registry.set_client_key_hash(machine_id, derived, active.get("key_id"))
                    return AuthCheckResult(
                        ok=True,
                        key_type="active",
                        rotation_required=True,
                        rotate_to=pending.get("key_plain"),
                        rotate_id=pending.get("key_id"),
                    )
                return AuthCheckResult(
                    ok=False,
                    error="old_key_not_allowed",
                )

            if machine_id and registry:
                derived = self._derive_client_hash(raw_key, machine_id, active_salt)
                stored = registry.get_client_key_hash(machine_id)
                stored_key_id = registry.get_client_key_id(machine_id)
                if stored and stored_key_id == active.get("key_id"):
                    if not _secure_compare(derived, stored):
                        return AuthCheckResult(ok=False, error="invalid_client_key")
                registry.set_client_key_hash(machine_id, derived, active.get("key_id"))
            return AuthCheckResult(ok=True, key_type="active")

        return AuthCheckResult(ok=False, error="invalid_key")

    def rotate_key(self, allowlist: list[str], allow_days: int, revoke_old: bool) -> dict[str, Any]:
        self.ensure_initialized()
        if self.is_env_override():
            raise RuntimeError("env_override")
        pending_key = _generate_key()
        pending_salt = _generate_salt()
        allow_days = max(0, int(allow_days))
        expires_at = (
            datetime.now(timezone.utc) + timedelta(days=allow_days)
            if allow_days > 0
            else datetime.now(timezone.utc)
        )

        if revoke_old:
            allowlist = []
            expires_at = datetime.now(timezone.utc)

        self._state["pending"] = {
            "key_hash": _hash_key(pending_key, pending_salt),
            "key_salt": pending_salt,
            "key_plain": pending_key,
            "created_at": _now(),
            "key_id": _hash_key(pending_key, pending_salt)[:12],
            "allowlist": allowlist,
            "allowlist_days": allow_days,
            "allowlist_expires_at": expires_at.isoformat(),
        }
        self._state["rotation"] = {
            "started_at": _now(),
            "completed_at": None,
        }
        self._save()
        return {
            "pending_key_id": self._state["pending"]["key_id"],
            "allowlist_expires_at": self._state["pending"]["allowlist_expires_at"],
        }

    def promote_pending(self) -> None:
        self._load()
        pending = self._state.get("pending")
        if not pending:
            return
        self._state["active"] = {
            k: pending[k]
            for k in ("key_hash", "key_salt", "key_plain", "created_at", "key_id")
        }
        self._state["pending"] = None
        rotation = self._state.get("rotation") or {}
        rotation["completed_at"] = _now()
        self._state["rotation"] = rotation
        self._save()

    def remove_from_allowlist(self, machine_id: str) -> None:
        self._load()
        pending = self._state.get("pending") or {}
        allowlist = [m for m in pending.get("allowlist") or [] if m != machine_id]
        pending["allowlist"] = allowlist
        self._state["pending"] = pending
        self._save()

    def keep_on_allowlist(self, machine_id: str) -> bool:
        self._load()
        pending = self._state.get("pending")
        if not pending:
            return False
        if self._allowlist_expired():
            return False
        allowlist = pending.get("allowlist") or []
        if machine_id not in allowlist:
            allowlist.append(machine_id)
        pending["allowlist"] = allowlist
        self._state["pending"] = pending
        self._save()
        return True

    def cleanup_allowlist(self) -> list[str]:
        self._load()
        pending = self._state.get("pending")
        if not pending:
            return []
        if self._allowlist_expired():
            removed = list(pending.get("allowlist") or [])
            pending["allowlist"] = []
            self._state["pending"] = pending
            self._save()
            return removed
        return []

    def get_dashboard_hash(self) -> str | None:
        self._load()
        return self._state.get("dashboard_hash")


_auth_manager: AuthManager | None = None


def get_auth_manager() -> AuthManager:
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
        _auth_manager.ensure_initialized()
    return _auth_manager
