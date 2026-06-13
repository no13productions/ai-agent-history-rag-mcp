from pathlib import Path

import pytest

from claude_history_rag.auth import AuthManager
from claude_history_rag.client_registry import ClientRegistry
from claude_history_rag.config import settings


@pytest.fixture
def auth_manager(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "server_psk", "")
    manager = AuthManager(path=tmp_path / "auth.json")
    manager.ensure_initialized()
    return manager


def test_rotation_promotes_after_allowlist_use(auth_manager: AuthManager, tmp_path: Path):
    registry = ClientRegistry(path=tmp_path / "client_registry.json")
    active_key = auth_manager.get_active_key_plain()
    assert active_key

    auth_manager.rotate_key(["client-1"], allow_days=1, revoke_old=False)
    result = auth_manager.validate_key(active_key, "client-1", registry)
    assert result.ok
    assert result.rotation_required
    assert result.rotate_to

    pending_key = auth_manager.get_pending_key_plain()
    assert pending_key
    result_pending = auth_manager.validate_key(pending_key, "client-1", registry)
    assert result_pending.ok
    state = auth_manager.get_rotation_state()
    assert state.get("pending") is None


def test_client_hash_validation(auth_manager: AuthManager, tmp_path: Path):
    registry = ClientRegistry(path=tmp_path / "client_registry.json")
    active_key = auth_manager.get_active_key_plain()
    assert active_key

    first = auth_manager.validate_key(active_key, "client-1", registry)
    assert first.ok

    # Bad hash should be rejected when key_id matches
    registry.set_client_key_hash(
        "client-1", "bad-hash", auth_manager.get_rotation_state()["active"]["key_id"]
    )
    second = auth_manager.validate_key(active_key, "client-1", registry)
    assert not second.ok
    assert second.error == "invalid_client_key"

    # Mismatched key_id should not block (rotation path)
    registry.set_client_key_hash("client-1", "bad-hash", "old-key")
    third = auth_manager.validate_key(active_key, "client-1", registry)
    assert third.ok
