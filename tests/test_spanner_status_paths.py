"""Regression tests for Spanner backend status/search call paths."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from claude_history_rag import server as server_module
from claude_history_rag.api_client import _redact_url
from claude_history_rag.config import settings
from claude_history_rag.models import (
    Chunk,
    ChunkUploadRequest,
    ClientHeartbeatRequest,
    SearchRequest,
)
from claude_history_rag.status import StatusCollector
from claude_history_rag.status_server import (
    StatusServer,
    _clear_machine_positions,
    _client_chunk_identity,
    _machine_positions,
    _scope_uploaded_chunk_ids,
    _server_chunk_id,
    _validate_upload_chunks,
)
from claude_history_rag.watcher import (
    HistoryWatcher,
    _count_file_lines,
    _machine_scoped_chunk_id,
)


class FakeEmbedder:
    is_initialized = True

    async def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeWatcher:
    is_running = True
    source_name = "Claude Code"
    projects_path = "/tmp/history"
    state = SimpleNamespace(get_all_files=lambda: [])
    queue = SimpleNamespace(qsize=lambda: 0)
    failed_files_count = 0

    def get_watched_files(self):
        return []


class FakeSpannerStore:
    def __init__(self, fts_available: bool = True):
        self.fts_available = fts_available
        self.hybrid_calls = 0
        self.vector_calls = 0
        self.native_embed_queries: list[str] = []
        self.last_hybrid_kwargs = {}
        self.last_vector_kwargs = {}

    def has_fts_index(self) -> bool:
        return self.fts_available

    async def get_stats_async(self) -> dict:
        return self.get_stats()

    def get_stats(self) -> dict:
        return {
            "total_chunks": 7,
            "backend": "spanner",
            "project": "project",
            "instance": "instance",
            "database": "database",
            "dimension": 3072,
            "fts_index_available": self.fts_available,
            "vector_index_available": True,
            "vector_search_mode": "ann",
            "embedding_mode": settings.spanner_embedding_mode,
            "embedding_model_id": "ConversationEmbeddingModel",
        }

    async def embed_query_text_async(self, query: str) -> list[float]:
        self.native_embed_queries.append(query)
        return [0.1, 0.2, 0.3]

    async def hybrid_search_async(self, **kwargs):
        self.hybrid_calls += 1
        self.last_hybrid_kwargs = kwargs
        return [
            {
                "id": "chunk-1",
                "content": "content",
                "score": 0.1,
                "_search_type": "vector",
            }
        ]

    async def search_async(self, **kwargs):
        self.vector_calls += 1
        self.last_vector_kwargs = kwargs
        return [{"id": "chunk-1", "content": "content", "score": 0.1}]


class CapturingStore:
    def __init__(self):
        self.chunks: list[dict] = []

    async def add_chunks_async(self, chunks: list[dict]) -> None:
        self.chunks.extend(chunks)


async def _no_daemon_status(*args, **kwargs):
    return None


def _daemon_status_snapshot() -> dict:
    return {
        "server": {"pid": 999999},
        "health": {"status": "healthy"},
        "database": {
            "total_chunks": 703496,
            "backend": "spanner",
            "project": "jeeves-486102",
            "instance": "jeeves-rg-spanner-prod-4d0e4c43",
            "database": "ai-agent-history-rag",
            "dimension": 3072,
            "fts_index_available": True,
            "vector_index_available": True,
            "vector_search_mode": "ann",
            "embedding_mode": "spanner",
            "embedding_model_id": "ConversationEmbeddingModel",
            "awaiting_embedding": 3,
        },
        "embedder": {"loaded": True},
        "cache": {"size": 0},
        "indexing": {
            "sources": {
                "Codex": {
                    "files_indexed": 112,
                    "files_pending": 0,
                    "files_failed": 0,
                    "is_running": True,
                    "watch_path": "/Users/brandon/.codex/sessions",
                },
                "Claude Code": {
                    "files_indexed": 4532,
                    "files_pending": 0,
                    "files_failed": 0,
                    "is_running": True,
                    "watch_path": "/Users/brandon/.claude/projects",
                },
            }
        },
        "file_watcher": {"all_sources_running": True},
    }


def _single_chunker(file_path, start_line=0):
    del start_line
    yield Chunk(
        id="chunk-1",
        content="content",
        chunk_type="turn",
        session_id="session",
        project_path="/project",
        project_name="project",
        timestamp="2026-06-13T00:00:00Z",
        source_file=str(file_path),
        source_line=1,
        parent_chunk_id="parent-1",
        child_chunk_ids=["child-1"],
    )


@pytest.mark.asyncio
async def test_local_direct_indexing_stamps_machine_id(monkeypatch, tmp_path):
    """Direct Spanner/local indexing must keep rows attributable by machine."""
    monkeypatch.setattr(settings, "machine_id", "test-machine")
    monkeypatch.setattr(settings, "storage_backend", "spanner")
    monkeypatch.setattr(settings, "spanner_embedding_mode", "spanner")
    history_file = tmp_path / "session.jsonl"
    history_file.write_text("{}\n")
    watcher = HistoryWatcher(
        projects_path=tmp_path,
        state_path=tmp_path / "state.json",
        chunker=_single_chunker,
        source_name="Test",
    )
    store = CapturingStore()

    await watcher._index_file(history_file, embedder=None, store=store)

    stored = store.chunks[0]
    assert stored["machine_id"] == "test-machine"
    assert stored["id"] == _machine_scoped_chunk_id("test-machine", "chunk-1")
    assert stored["parent_chunk_id"] == _machine_scoped_chunk_id("test-machine", "parent-1")
    assert stored["child_chunk_ids"] == [_machine_scoped_chunk_id("test-machine", "child-1")]


@pytest.mark.asyncio
async def test_get_index_status_supports_spanner_stats(monkeypatch):
    """Index status should not assume LanceDB-only db_path stats."""
    fake_store = FakeSpannerStore()
    monkeypatch.setattr("claude_history_rag.store.store", fake_store)
    monkeypatch.setattr(settings, "storage_backend", "spanner")
    monkeypatch.setattr(server_module, "_get_status_server_snapshot", _no_daemon_status)
    monkeypatch.setattr(server_module, "get_all_watchers", lambda: [FakeWatcher()])
    monkeypatch.setattr("claude_history_rag.embedder.get_embedder", lambda: FakeEmbedder())

    result = await server_module.get_index_status()

    assert result["status"] == "healthy"
    assert result["total_chunks"] == 7
    assert result["storage_backend"] == "spanner"
    assert result["database"]["vector_search_mode"] == "ann"
    assert "db_path" not in result


@pytest.mark.asyncio
async def test_get_index_status_uses_external_daemon_status(monkeypatch):
    """Lightweight MCP status should report the always-on daemon watcher when present."""

    async def fake_daemon_status(*args, **kwargs):
        return _daemon_status_snapshot()

    monkeypatch.setattr(server_module, "_get_status_server_snapshot", fake_daemon_status)

    result = await server_module.get_index_status()

    assert result["source"] == "daemon"
    assert result["daemon_pid"] == 999999
    assert result["watcher_running"] is True
    assert result["watched_files"] == 4644
    assert result["sources"]["Codex"]["running"] is True
    assert result["database"]["awaiting_embedding"] == 3


@pytest.mark.asyncio
async def test_get_server_status_uses_external_daemon_status(monkeypatch):
    """Full MCP server status should prefer daemon health over idle MCP process state."""

    async def fake_daemon_status(*args, **kwargs):
        return _daemon_status_snapshot()

    monkeypatch.setattr(server_module, "_get_status_server_snapshot", fake_daemon_status)

    result = await server_module.get_server_status(detail_level="full")

    assert result["source"] == "daemon"
    assert result["server"]["pid"] == 999999
    assert result["file_watcher"]["all_sources_running"] is True


@pytest.mark.asyncio
async def test_status_collector_spanner_database_stats_are_backend_specific(monkeypatch):
    """Full status should report Spanner index state without fake LanceDB path data."""
    fake_store = FakeSpannerStore()
    monkeypatch.setattr("claude_history_rag.status.store", fake_store)
    monkeypatch.setattr(settings, "storage_backend", "spanner")

    stats = await StatusCollector()._get_database_stats()

    assert stats["backend"] == "spanner"
    assert stats["vector_index_available"] is True
    assert stats["vector_search_mode"] == "ann"
    assert "database_path" not in stats
    assert "database_size_bytes" not in stats


@pytest.mark.asyncio
async def test_search_conversations_honors_vector_only_with_analysis(monkeypatch):
    """Decision-engine path should obey use_hybrid=False."""
    fake_store = FakeSpannerStore(fts_available=True)
    monkeypatch.setattr("claude_history_rag.store.store", fake_store)
    monkeypatch.setattr("claude_history_rag.embedder.get_embedder", lambda: FakeEmbedder())

    result = await server_module.search_conversations(
        query="oauth",
        limit=2,
        use_hybrid=False,
        enable_analysis=True,
    )

    assert result["search_type"] == "vector"
    assert fake_store.vector_calls >= 1
    assert fake_store.hybrid_calls == 0


@pytest.mark.asyncio
async def test_search_conversations_reports_runtime_vector_fallback(monkeypatch):
    """Backend fallback marker should make response search_type truthful."""
    fake_store = FakeSpannerStore(fts_available=True)
    monkeypatch.setattr("claude_history_rag.store.store", fake_store)
    monkeypatch.setattr("claude_history_rag.embedder.get_embedder", lambda: FakeEmbedder())

    result = await server_module.search_conversations(
        query="oauth",
        limit=2,
        use_hybrid=True,
        enable_analysis=False,
    )

    assert result["search_type"] == "vector"
    assert "_search_type" not in result["results"][0]


@pytest.mark.asyncio
async def test_search_conversations_passes_timeframe_to_store(monkeypatch):
    """MCP search should pass parsed timeframe filters into backend search."""
    fake_store = FakeSpannerStore(fts_available=True)
    monkeypatch.setattr("claude_history_rag.store.store", fake_store)
    monkeypatch.setattr("claude_history_rag.embedder.get_embedder", lambda: FakeEmbedder())

    result = await server_module.search_conversations(
        query="multi-table sql gql",
        date_from="2026-06-13T04:12:25Z",
        date_to="2026-06-15T04:12:25Z",
        limit=2,
        use_hybrid=True,
        enable_analysis=False,
    )

    assert result["date_from"] == "2026-06-13T04:12:25Z"
    assert result["date_to"] == "2026-06-15T04:12:25Z"
    assert fake_store.last_hybrid_kwargs["date_from"].isoformat() == ("2026-06-13T04:12:25+00:00")
    assert fake_store.last_hybrid_kwargs["date_to"].isoformat() == ("2026-06-15T04:12:25+00:00")


@pytest.mark.asyncio
async def test_search_file_changes_uses_spanner_native_query_embedding(monkeypatch):
    """File-change MCP search should not instantiate the app embedder in native mode."""
    fake_store = FakeSpannerStore(fts_available=True)
    monkeypatch.setattr(settings, "storage_backend", "spanner")
    monkeypatch.setattr(settings, "spanner_embedding_mode", "spanner")
    monkeypatch.setattr("claude_history_rag.store.store", fake_store)

    def fail_embedder():
        raise AssertionError("app embedder should not be used")

    monkeypatch.setattr("claude_history_rag.embedder.get_embedder", fail_embedder)

    result = await server_module.search_file_changes(query="auth", limit=2)

    assert result["count"] == 1
    assert fake_store.native_embed_queries == ["auth"]
    assert fake_store.vector_calls == 1


@pytest.mark.asyncio
async def test_get_session_summary_uses_spanner_native_query_embedding(monkeypatch):
    """Session summary MCP search should use the backend-aware embedding path."""
    fake_store = FakeSpannerStore(fts_available=True)
    monkeypatch.setattr(settings, "storage_backend", "spanner")
    monkeypatch.setattr(settings, "spanner_embedding_mode", "spanner")
    monkeypatch.setattr("claude_history_rag.store.store", fake_store)

    def fail_embedder():
        raise AssertionError("app embedder should not be used")

    monkeypatch.setattr("claude_history_rag.embedder.get_embedder", fail_embedder)

    result = await server_module.get_session_summary(count=1)

    assert result["count"] == 1
    assert fake_store.native_embed_queries == ["session summary overview"]


def test_server_chunk_id_is_scoped_by_machine_id():
    """Remote uploads cannot collide across machines by reusing a client chunk id."""
    assert _server_chunk_id("machine-a", "chunk-1") != _server_chunk_id("machine-b", "chunk-1")
    assert _server_chunk_id("machine-a", "chunk-1") == _server_chunk_id("machine-a", "chunk-1")


def test_missing_client_chunk_ids_get_stable_distinct_fallbacks():
    """Older clients that omit chunk id should not collapse rows on upload."""
    first = {
        "content": "first",
        "source_file": "/tmp/history.jsonl",
        "source_line": 1,
        "chunk_type": "turn",
        "session_id": "session",
    }
    second = {
        "content": "second",
        "source_file": "/tmp/history.jsonl",
        "source_line": 2,
        "chunk_type": "turn",
        "session_id": "session",
    }

    assert _client_chunk_identity(first) != _client_chunk_identity(second)
    assert _server_chunk_id("machine-a", _client_chunk_identity(first)) != _server_chunk_id(
        "machine-a", _client_chunk_identity(second)
    )


def test_uploaded_chunk_id_scoping_rewrites_batch_references():
    """Server-side id namespacing should not leave intra-batch links dangling."""
    chunks = [
        {
            "id": "parent",
            "content": "parent content",
            "chunk_type": "turn",
            "session_id": "session",
            "project_path": "/project",
            "project_name": "project",
            "source_file": "/tmp/history.jsonl",
            "source_line": 1,
            "child_chunk_ids": ["child"],
        },
        {
            "id": "child",
            "content": "child content",
            "chunk_type": "turn",
            "session_id": "session",
            "project_path": "/project",
            "project_name": "project",
            "source_file": "/tmp/history.jsonl",
            "source_line": 2,
            "parent_chunk_id": "parent",
            "child_chunk_ids": [],
        },
    ]

    _scope_uploaded_chunk_ids("machine-a", chunks)

    assert chunks[0]["id"] == _server_chunk_id("machine-a", "parent")
    assert chunks[1]["id"] == _server_chunk_id("machine-a", "child")
    assert chunks[0]["child_chunk_ids"] == [chunks[1]["id"]]
    assert chunks[1]["parent_chunk_id"] == chunks[0]["id"]


def test_uploaded_chunk_id_scoping_rewrites_cross_batch_references():
    """References outside the current upload still stay in the same machine namespace."""
    chunks = [
        {
            "id": "child",
            "content": "child content",
            "chunk_type": "turn",
            "session_id": "session",
            "project_path": "/project",
            "project_name": "project",
            "source_file": "/tmp/history.jsonl",
            "source_line": 2,
            "parent_chunk_id": "parent-from-previous-batch",
            "child_chunk_ids": ["child-from-next-batch"],
        }
    ]

    _scope_uploaded_chunk_ids("machine-a", chunks)

    assert chunks[0]["parent_chunk_id"] == _server_chunk_id(
        "machine-a", "parent-from-previous-batch"
    )
    assert chunks[0]["child_chunk_ids"] == [_server_chunk_id("machine-a", "child-from-next-batch")]


def test_upload_chunk_validation_rejects_missing_required_fields():
    """Malformed upload chunks should fail before embedding/storage."""
    error = _validate_upload_chunks([{"content": "missing shape"}])

    assert error == "chunks[0].chunk_type is required"


def test_upload_chunk_validation_requires_project_name():
    """Storage schemas require project_name, so reject it before storage."""
    error = _validate_upload_chunks(
        [
            {
                "content": "content",
                "chunk_type": "turn",
                "session_id": "session",
                "project_path": "/project",
                "source_file": "/tmp/history.jsonl",
            }
        ]
    )

    assert error == "chunks[0].project_name is required"


def test_request_models_bound_limits_and_positions():
    """Authenticated clients should not be able to request unbounded work or bad positions."""
    with pytest.raises(ValidationError):
        SearchRequest(query="x", limit=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="x", limit=1000)
    with pytest.raises(ValidationError):
        ChunkUploadRequest(
            machine_id="machine",
            chunks=[],
            source_file="/tmp/history.jsonl",
            file_position=-1,
        )
    with pytest.raises(ValidationError):
        ChunkUploadRequest(
            machine_id="machine;DROP",
            chunks=[],
            source_file="/tmp/history.jsonl",
            file_position=1,
        )
    with pytest.raises(ValidationError):
        ChunkUploadRequest(
            machine_id="machine",
            chunks=[{}] * 501,
            source_file="/tmp/history.jsonl",
            file_position=1,
        )
    with pytest.raises(ValidationError):
        ClientHeartbeatRequest(
            machine_id="machine",
            errors={f"k{i}": i for i in range(65)},
        )


def test_api_client_url_redaction_removes_userinfo():
    """Client mode connection logs should not expose URL credentials."""
    assert _redact_url("https://user:pass@example.com:8443/api") == "https://example.com:8443/api"


def test_binary_history_line_count_does_not_raise(tmp_path):
    """Legacy Antigravity protobuf files may contain invalid UTF-8."""
    pb_file = tmp_path / "conversation.pb"
    pb_file.write_bytes(b"\xff\xfe\x00\x80")

    assert _count_file_lines(pb_file) == 0


def test_machine_position_clear_helpers_remove_stale_positions():
    """Purges and full reindex use this helper to avoid stale server positions."""
    _machine_positions.clear()
    _machine_positions["machine-a"] = {"/tmp/a.jsonl": 10}
    _machine_positions["machine-b"] = {"/tmp/b.jsonl": 20}

    _clear_machine_positions("machine-a")
    assert "machine-a" not in _machine_positions
    assert "machine-b" in _machine_positions

    _clear_machine_positions()
    assert _machine_positions == {}


@pytest.mark.asyncio
async def test_dashboard_html_bootstraps_without_bearer_auth(monkeypatch):
    """The dashboard page must load so its JavaScript unlock flow can attach auth."""
    monkeypatch.setattr(settings, "auth_enabled", True)
    response = await StatusServer().handle_dashboard(SimpleNamespace())

    assert response.status == 200
    assert "text/html" in response.content_type


def test_local_status_auth_headers_uses_shared_psk(monkeypatch, tmp_path):
    """The in-process status snapshot must Bearer the shared daemon PSK when auth is on.

    Without this header the daemon returns 401 and the snapshot falls back to the idle
    MCP process's own watchers, reporting watcher_running False even while the daemon runs.
    """
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"active": {"key_plain": "shared-psk-123"}}))
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "server_psk", "")
    monkeypatch.setattr(settings, "auth_state_path", auth_file)

    assert server_module._local_status_auth_headers() == {
        "Authorization": "Bearer shared-psk-123"
    }


def test_local_status_auth_headers_prefers_env_override(monkeypatch, tmp_path):
    """An explicit server_psk env override should win over the auth-state file."""
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"active": {"key_plain": "file-psk"}}))
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "server_psk", "env-psk")
    monkeypatch.setattr(settings, "auth_state_path", auth_file)

    assert server_module._local_status_auth_headers() == {"Authorization": "Bearer env-psk"}


def test_local_status_auth_headers_empty_when_auth_disabled(monkeypatch):
    """No Authorization header should be sent when auth is disabled."""
    monkeypatch.setattr(settings, "auth_enabled", False)

    assert server_module._local_status_auth_headers() == {}
