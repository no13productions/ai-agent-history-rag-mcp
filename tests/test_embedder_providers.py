"""Tests for pluggable embedding providers."""

import pytest

from claude_history_rag import embedder as embedder_module
from claude_history_rag.config import settings
from claude_history_rag.embedder import (
    AsyncEmbedder,
    OpenAICompatibleEmbeddingProvider,
    create_embedding_provider,
    redact_url,
)


class FakeProvider:
    """Minimal provider used to verify AsyncEmbedder routing."""

    provider_name = "vertex"
    model_name = "gemini-embedding-001"
    dimension = 3072

    def __init__(self):
        self.initialized = False
        self.calls: list[tuple[list[str], str | None]] = []

    def initialize(self) -> None:
        self.initialized = True

    def embed_sync(self, texts: list[str], task_type: str | None = None) -> list[list[float]]:
        self.calls.append((texts, task_type))
        return [[0.1] * self.dimension for _ in texts]

    def shutdown(self) -> None:
        self.initialized = False


@pytest.mark.asyncio
async def test_async_embedder_routes_vertex_task_types(monkeypatch):
    """Vertex query/document embeddings receive the configured task type."""
    monkeypatch.setattr(settings, "vertex_query_task_type", "RETRIEVAL_QUERY")
    monkeypatch.setattr(settings, "vertex_document_task_type", "RETRIEVAL_DOCUMENT")

    provider = FakeProvider()
    embedder = AsyncEmbedder(provider=provider)

    query_vector = await embedder.embed_query("find auth changes")
    document_vectors = await embedder.embed_documents(["changed auth.py"])

    assert len(query_vector) == 3072
    assert len(document_vectors[0]) == 3072
    assert provider.calls == [
        (["find auth changes"], "RETRIEVAL_QUERY"),
        (["changed auth.py"], "RETRIEVAL_DOCUMENT"),
    ]

    embedder.shutdown()


def test_create_embedding_provider_supports_vertex(monkeypatch):
    """The provider factory can construct Vertex without importing SDK classes yet."""
    monkeypatch.setattr(settings, "embedding_provider", "vertex")
    monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")
    monkeypatch.setattr(settings, "embedding_dimension", 3072)

    provider = create_embedding_provider()

    assert provider.provider_name == "vertex"
    assert provider.model_name == "gemini-embedding-001"
    assert provider.dimension == 3072


def test_get_embedder_uses_configured_provider(monkeypatch):
    """The global embedder factory honors the configured provider setting."""
    monkeypatch.setattr(settings, "embedding_provider", "vertex")
    monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")
    monkeypatch.setattr(settings, "embedding_dimension", 3072)
    monkeypatch.setattr(embedder_module, "embedder", None)

    created = embedder_module.get_embedder()

    assert created.provider_name == "vertex"
    assert created.model_name == "gemini-embedding-001"
    assert created.dimension == 3072

    created.shutdown()
    monkeypatch.setattr(embedder_module, "embedder", None)


def test_redact_url_removes_userinfo():
    """Credentials embedded in endpoint URLs are removed before logging/status."""
    assert redact_url("https://user:pass@example.com:8443/v1") == "https://example.com:8443/v1"
    assert redact_url("http://localhost:11434/v1") == "http://localhost:11434/v1"


def test_openai_provider_does_not_send_dimensions_by_default(monkeypatch):
    """Storage dimensions should not break OpenAI-compatible local endpoints."""
    monkeypatch.setattr(settings, "embedding_dimension", 3072)
    monkeypatch.setattr(settings, "openai_embedding_send_dimensions", False)
    provider = OpenAICompatibleEmbeddingProvider(model_name="nomic-embed-text")

    captured = {}

    class FakeClient:
        pass

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    def fake_post(url, json, headers):
        del url, headers
        captured.update(json)
        return FakeResponse()

    provider._client = FakeClient()
    provider._client.post = fake_post

    assert provider.embed_sync(["hello"]) == [[0.1, 0.2]]
    assert "dimensions" not in captured


def test_openai_provider_rejects_bad_embedding_indexes(monkeypatch):
    """Malformed OpenAI-compatible responses must not reorder embeddings silently."""
    monkeypatch.setattr(settings, "openai_embedding_send_dimensions", False)
    provider = OpenAICompatibleEmbeddingProvider(model_name="nomic-embed-text")

    class FakeClient:
        pass

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"index": 0, "embedding": [0.1]},
                    {"index": 0, "embedding": [0.2]},
                ]
            }

    def fake_post(url, json, headers):
        del url, json, headers
        return FakeResponse()

    provider._client = FakeClient()
    provider._client.post = fake_post

    with pytest.raises(RuntimeError, match="Invalid embedding response format"):
        provider.embed_sync(["first", "second"])


def test_openai_provider_requires_indexes_for_batches(monkeypatch):
    """Batch responses without indexes are ambiguous and rejected."""
    monkeypatch.setattr(settings, "openai_embedding_send_dimensions", False)
    provider = OpenAICompatibleEmbeddingProvider(model_name="nomic-embed-text")

    class FakeClient:
        pass

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"embedding": [0.1]},
                    {"embedding": [0.2]},
                ]
            }

    def fake_post(url, json, headers):
        del url, json, headers
        return FakeResponse()

    provider._client = FakeClient()
    provider._client.post = fake_post

    with pytest.raises(RuntimeError, match="Invalid embedding response format"):
        provider.embed_sync(["first", "second"])
