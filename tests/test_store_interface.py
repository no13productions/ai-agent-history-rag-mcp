"""Tests for storage backend configuration seams."""

from datetime import UTC, datetime

import pytest

from claude_history_rag import store as store_module
from claude_history_rag.config import Settings, settings
from claude_history_rag.store import (
    SPANNER_CONTENT_SEARCH_INDEX,
    SPANNER_CONTENT_TOKENS_COLUMN,
    SPANNER_TABLE_NAME,
    SPANNER_VECTOR_INDEX,
    SpannerStore,
    VectorStore,
    create_store,
    get_conversation_chunk_schema,
    get_spanner_embedding_model_ddl,
    get_spanner_schema_ddl,
    get_spanner_vector_index_ddl,
    get_vector_dim,
)


class FakeBatch:
    def __init__(self):
        self.insert_or_update_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def insert_or_update(self, table, columns, values):
        self.insert_or_update_calls.append({"table": table, "columns": columns, "values": values})


class FakeSnapshot:
    def __init__(self, database):
        self.database = database

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute_sql(self, sql, params=None, param_types=None):
        self.database.sql_calls.append(
            {"sql": sql, "params": params or {}, "param_types": param_types or {}}
        )
        if "information_schema.tables" in sql:
            return []
        if "ML.PREDICT" in sql and "SELECT" in sql:
            return [[[0.1, 0.2, 0.3]]]
        if "COSINE_DISTANCE" in sql:
            return [
                [
                    "chunk-1",
                    "content",
                    "turn",
                    "session-1",
                    "/project",
                    "project",
                    datetime(2026, 1, 1, tzinfo=UTC),
                    None,
                    None,
                    "machine",
                    0.123,
                ]
            ]
        if "COUNTIF" in sql:
            return [[0, 0]]  # (total, embedded) for _embedding_counts
        if "COUNT(*)" in sql:
            return [[0]]
        return []


class FakeOperation:
    def __init__(self, database):
        self.database = database

    def result(self):
        self.database.ddl_completed = True


class FakeDatabase:
    def __init__(self):
        self.sql_calls = []
        self.ddl = []
        self.ddl_completed = False
        self.batch_obj = FakeBatch()
        self.pdml_calls = []

    def execute_partitioned_dml(self, sql):
        self.pdml_calls.append(sql)
        return 7

    def snapshot(self):
        return FakeSnapshot(self)

    def update_ddl(self, ddl):
        self.ddl.extend(ddl)
        return FakeOperation(self)

    def batch(self):
        return self.batch_obj

    def run_in_transaction(self, callback):
        return callback(self)

    def execute_update(self, sql, params=None, param_types=None):
        self.sql_calls.append(
            {"sql": sql, "params": params or {}, "param_types": param_types or {}}
        )
        return 1


def test_explicit_embedding_dimension_controls_vector_schema(monkeypatch):
    """Configured embedding dimension is used for validation and LanceDB schema."""
    monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")
    monkeypatch.setattr(settings, "embedding_dimension", 3072)
    monkeypatch.setattr(store_module, "_vector_dim", None)

    schema = get_conversation_chunk_schema()

    assert get_vector_dim() == 3072
    assert schema.field("vector").type.list_size == 3072


def test_known_vertex_model_defaults_to_3072(monkeypatch):
    """gemini-embedding-001 maps to the requested 3072-dimensional output."""
    monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")
    monkeypatch.setattr(settings, "embedding_dimension", None)
    monkeypatch.setattr(store_module, "_vector_dim", None)

    assert get_vector_dim() == 3072


def test_spanner_native_embedding_mode_defaults_to_gemini_3072():
    """Spanner-native embeddings use the Google publisher Gemini embedding model."""
    configured = Settings(spanner_embedding_mode="spanner")

    assert configured.embedding_model == "gemini-embedding-001"
    assert configured.embedding_dimension == 3072


def test_spanner_native_embedding_mode_rejects_non_gemini_model():
    """Spanner-native embeddings cannot register arbitrary app embedding models."""
    with pytest.raises(ValueError, match="gemini-embedding-001"):
        Settings(
            spanner_embedding_mode="spanner",
            embedding_model="text-embedding-3-large",
            embedding_dimension=3072,
        )


def test_vertex_task_type_rejects_sql_literal_injection():
    """Task types are constrained to expected Vertex embedding tokens."""
    with pytest.raises(ValueError, match="Vertex task type"):
        Settings(vertex_query_task_type='RETRIEVAL_QUERY"; SELECT 1; --')


def test_create_store_uses_lancedb_backend(monkeypatch):
    """Storage factory keeps LanceDB as the default working backend."""
    monkeypatch.setattr(settings, "storage_backend", "lancedb")

    assert isinstance(create_store(), VectorStore)


def test_create_store_uses_spanner_backend(monkeypatch):
    """Storage factory can select Spanner."""
    monkeypatch.setattr(settings, "storage_backend", "spanner")

    assert isinstance(create_store(), SpannerStore)


def test_spanner_schema_ddl_uses_configured_vector_dimension(monkeypatch):
    """Spanner table DDL uses ARRAY<FLOAT32> with configured vector length."""
    monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")
    monkeypatch.setattr(settings, "embedding_dimension", 3072)
    monkeypatch.setattr(store_module, "_vector_dim", None)

    ddl = "\n".join(get_spanner_schema_ddl())

    # Vector is nullable so deferred-embedding mode can land rows before vectors exist.
    assert "Vector ARRAY<FLOAT32>(vector_length=>3072)" in ddl
    assert "Vector ARRAY<FLOAT32>(vector_length=>3072) NOT NULL" not in ddl
    assert f"CREATE TABLE {SPANNER_TABLE_NAME}" in ddl
    assert f"{SPANNER_CONTENT_TOKENS_COLUMN} TOKENLIST" in ddl
    assert "TOKENIZE_FULLTEXT(Content)" in ddl
    assert f"CREATE SEARCH INDEX {SPANNER_CONTENT_SEARCH_INDEX}" in ddl


def test_spanner_vector_index_ddl_is_separate_from_base_schema(monkeypatch):
    """Spanner ANN index DDL can be run after bulk ingestion."""
    monkeypatch.setattr(settings, "spanner_vector_index_leaves", 2048)

    ddl = get_spanner_vector_index_ddl()

    assert f"CREATE VECTOR INDEX {SPANNER_VECTOR_INDEX}" in ddl
    assert f"ON {SPANNER_TABLE_NAME}(Vector)" in ddl
    # Partial index over embedded rows only (Vector is nullable in deferred mode).
    assert "WHERE Vector IS NOT NULL" in ddl
    assert "distance_type = 'COSINE'" in ddl
    assert "num_leaves = 2048" in ddl


def test_spanner_embedding_model_ddl_registers_vertex_endpoint(monkeypatch):
    """Spanner embedding model DDL points at the configured Vertex embedding model."""
    monkeypatch.setattr(settings, "vertex_project", "vertex-project")
    monkeypatch.setattr(settings, "vertex_location", "us-central1")
    monkeypatch.setattr(settings, "embedding_model", "gemini-embedding-001")
    monkeypatch.setattr(settings, "spanner_embedding_model_id", "ConversationEmbeddingModel")

    ddl = get_spanner_embedding_model_ddl("spanner-project")

    assert "CREATE MODEL IF NOT EXISTS ConversationEmbeddingModel" in ddl
    assert "INPUT(" in ddl
    assert "OUTPUT(" in ddl
    assert (
        "endpoint = '//aiplatform.googleapis.com/projects/vertex-project/locations/"
        "us-central1/publishers/google/models/gemini-embedding-001'"
    ) in ddl


def test_spanner_store_add_chunks_writes_insert_or_update(monkeypatch):
    """SpannerStore persists chunks using insert_or_update mutations."""
    monkeypatch.setattr(settings, "embedding_dimension", 3)
    monkeypatch.setattr(store_module, "_vector_dim", None)
    database = FakeDatabase()
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = database

    store.add_chunks(
        [
            {
                "id": "chunk-1",
                "content": "content",
                "vector": [0.1, 0.2, 0.3],
                "chunk_type": "turn",
                "session_id": "session-1",
                "project_path": "/project",
                "project_name": "project",
                "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
                "source_file": "/source.jsonl",
                "source_line": 1,
                "machine_id": "machine",
            }
        ]
    )

    call = database.batch_obj.insert_or_update_calls[0]
    assert call["table"] == SPANNER_TABLE_NAME
    assert call["values"][0][0] == "chunk-1"
    assert call["values"][0][2] == [0.1, 0.2, 0.3]


def test_spanner_store_add_chunks_can_generate_embeddings_in_spanner(monkeypatch):
    """Spanner-native embedding mode stores raw chunks with ML.PREDICT DML."""
    monkeypatch.setattr(settings, "embedding_dimension", 3)
    monkeypatch.setattr(settings, "spanner_embedding_mode", "spanner")
    monkeypatch.setattr(settings, "spanner_embedding_model_id", "ConversationEmbeddingModel")
    monkeypatch.setattr(settings, "vertex_document_task_type", "RETRIEVAL_DOCUMENT")
    monkeypatch.setattr(store_module, "_vector_dim", None)
    database = FakeDatabase()
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = database
    monkeypatch.setattr(store, "ensure_embedding_model", lambda: None)

    store.add_chunks(
        [
            {
                "id": "chunk-1",
                "content": "content",
                "chunk_type": "turn",
                "session_id": "session-1",
                "project_path": "/project",
                "project_name": "project",
                "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
                "source_file": "/source.jsonl",
                "source_line": 1,
                "machine_id": "machine",
            },
            {
                "id": "chunk-2",
                "content": "more content",
                "chunk_type": "turn",
                "session_id": "session-1",
                "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
                "source_file": "/source.jsonl",
                "source_line": 2,
                "machine_id": "machine",
            },
        ]
    )

    # A whole batch is embedded in ONE batched INSERT ... SELECT, not one DML per chunk.
    assert len(database.sql_calls) == 1
    call = database.sql_calls[-1]
    assert "INSERT OR UPDATE INTO ConversationChunks" in call["sql"]
    assert "ML.PREDICT" in call["sql"]
    assert "MODEL ConversationEmbeddingModel" in call["sql"]
    assert "FROM UNNEST(@rows)" in call["sql"]
    assert "remote_udf_max_rows_per_rpc=" in call["sql"]
    assert "RETRIEVAL_DOCUMENT" not in call["sql"]
    assert "CAST(value AS FLOAT32)" in call["sql"]
    assert call["params"]["task_type"] == "RETRIEVAL_DOCUMENT"
    rows = call["params"]["rows"]
    assert len(rows) == 2
    # Struct field order: id is first, content second (see _CHUNK_STRUCT_FIELDS).
    assert rows[0][0] == "chunk-1"
    assert rows[0][1] == "content"
    assert rows[1][0] == "chunk-2"


def test_chunk_struct_value_matches_declared_field_order(monkeypatch):
    """The positional STRUCT value lines up with _CHUNK_STRUCT_FIELDS and omits Vector."""
    store = SpannerStore(project="p", instance="i", database="d")
    chunk = {
        "id": "c1",
        "content": "hello",
        "chunk_type": "turn",
        "session_id": "s1",
        "project_path": "/p",
        "project_name": "p",
        "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
        "source_file": "/f.jsonl",
        "source_line": 4,
        "child_chunk_ids": ["a", "b"],
        "machine_id": "m",
    }
    value = store._chunk_struct_value(chunk)
    fields = store._CHUNK_STRUCT_FIELDS
    assert len(value) == len(fields)
    assert "vector" not in fields
    assert value[fields.index("id")] == "c1"
    assert value[fields.index("content")] == "hello"
    assert value[fields.index("source_line")] == 4
    assert value[fields.index("child_chunk_ids")] == ["a", "b"]


def test_spanner_store_defer_embeddings_inserts_without_vector(monkeypatch):
    """Deferred mode lands rows via mutations with the Vector column omitted (NULL)."""
    monkeypatch.setattr(settings, "embedding_dimension", 3)
    monkeypatch.setattr(settings, "spanner_embedding_mode", "spanner")
    monkeypatch.setattr(settings, "spanner_defer_embeddings", True)
    monkeypatch.setattr(store_module, "_vector_dim", None)
    database = FakeDatabase()
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = database

    store.add_chunks(
        [
            {
                "id": "chunk-1",
                "content": "content",
                "chunk_type": "turn",
                "session_id": "session-1",
                "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
                "source_file": "/source.jsonl",
                "source_line": 1,
                "machine_id": "machine",
            }
        ]
    )

    # No ML.PREDICT DML on the write path; rows are inserted vector-less via mutations.
    assert database.sql_calls == []
    call = database.batch_obj.insert_or_update_calls[0]
    assert call["table"] == SPANNER_TABLE_NAME
    assert "Vector" not in call["columns"]
    assert call["values"][0][0] == "chunk-1"


def test_spanner_store_backfill_embeddings_shards_and_embeds(monkeypatch):
    """backfill_embeddings drains Id-prefix shards via the batched embed path."""
    monkeypatch.setattr(settings, "spanner_backfill_concurrency", 4)
    monkeypatch.setattr(settings, "spanner_backfill_batch_size", 2)
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = FakeDatabase()
    monkeypatch.setattr(store, "ensure_embedding_model", lambda: None)

    # One shard ('00') yields a single batch then drains; every other prefix is empty.
    served: dict[str, bool] = {}

    def fake_read(prefix, limit):
        if prefix == "00" and not served.get(prefix):
            served[prefix] = True
            return [{"id": "00aaa", "content": "c1"}, {"id": "00bbb", "content": "c2"}]
        return []

    embedded_batches: list[list[dict]] = []
    monkeypatch.setattr(store, "_read_unembedded_batch", fake_read)
    monkeypatch.setattr(
        store,
        "_add_chunks_with_spanner_embeddings",
        lambda rows: embedded_batches.append(list(rows)),
    )

    total = store.backfill_embeddings()

    assert total == 2
    assert embedded_batches == [
        [{"id": "00aaa", "content": "c1"}, {"id": "00bbb", "content": "c2"}]
    ]


def test_backfill_shard_failure_does_not_abort_the_pass(monkeypatch):
    """A failed batch stops only its shard; backfill_embeddings completes without raising."""
    monkeypatch.setattr(settings, "spanner_backfill_concurrency", 4)
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = FakeDatabase()
    monkeypatch.setattr(store, "ensure_embedding_model", lambda: None)

    # Every shard reads one batch; embedding always fails (e.g. Vertex 409 quota).
    monkeypatch.setattr(store, "_read_unembedded_batch", lambda prefix, limit: [{"id": prefix}])

    def boom(rows):
        raise RuntimeError("429 quota exceeded")

    monkeypatch.setattr(store, "_add_chunks_with_spanner_embeddings", boom)

    # Must not raise, and embeds 0 (all batches failed) — rows stay NULL for the next pass.
    total = store.backfill_embeddings()
    assert total == 0


def test_spanner_store_get_stats_includes_backfill_progress(monkeypatch):
    """get_stats exposes embedded/awaiting counts that drive the dashboard backfill section."""
    monkeypatch.setattr(settings, "storage_backend", "spanner")
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = FakeDatabase()

    stats = store.get_stats()

    assert "embedded_chunks" in stats
    assert "awaiting_embedding" in stats
    assert stats["awaiting_embedding"] == max(stats["total_chunks"] - stats["embedded_chunks"], 0)


def test_embedding_counts_are_ttl_cached(monkeypatch):
    """The expensive count scan is cached — repeated polls don't re-scan within the TTL."""
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = FakeDatabase()
    scans = []
    monkeypatch.setattr(store, "_embedding_counts_uncached", lambda: (scans.append(1), (10, 4))[1])

    first = store._embedding_counts()
    second = store._embedding_counts()

    assert first == (10, 4)
    assert second == (10, 4)
    assert len(scans) == 1  # cached — the O(table) scan ran exactly once


def test_read_unembedded_batch_filters_by_id_prefix(monkeypatch):
    """_read_unembedded_batch targets one Id shard with a bounded SELECT."""
    store = SpannerStore(project="p", instance="i", database="d")
    database = FakeDatabase()
    store._database = database

    rows = store._read_unembedded_batch("ab", 50)

    call = database.sql_calls[-1]
    assert "WHERE Vector IS NULL AND STARTS_WITH(Id, @prefix)" in call["sql"]
    assert "LIMIT @limit" in call["sql"]
    assert "Vector" not in call["sql"].split("FROM")[0]  # Vector is recomputed, not read
    assert call["params"]["prefix"] == "ab"
    assert call["params"]["limit"] == 50
    assert rows == []  # fake DB has no matching rows


def test_row_to_chunk_dict_maps_columns_in_order():
    """A positional Spanner read row maps to the app chunk-dict shape."""
    store = SpannerStore(project="p", instance="i", database="d")
    row = [
        "id1",
        "content1",
        "turn",
        "s1",
        "/p",
        "p",
        datetime(2026, 1, 1, tzinfo=UTC),
        "u",
        "a",
        "/f",
        "edit",
        "m",
        "/src.jsonl",
        7,
        "parent",
        ["c1", "c2"],
        "machine",
    ]
    mapped = store._row_to_chunk_dict(row)
    assert len(mapped) == len(store._BACKFILL_DICT_KEYS)
    assert mapped["id"] == "id1"
    assert mapped["content"] == "content1"
    assert mapped["source_line"] == 7
    assert mapped["child_chunk_ids"] == ["c1", "c2"]
    assert mapped["machine_id"] == "machine"


def test_spanner_store_treats_none_vector_as_unembedded(monkeypatch):
    """Spanner-native mode accepts API chunks with vector=None as unembedded."""
    monkeypatch.setattr(settings, "embedding_dimension", 3)
    monkeypatch.setattr(settings, "spanner_embedding_mode", "spanner")
    monkeypatch.setattr(store_module, "_vector_dim", None)
    database = FakeDatabase()
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = database
    monkeypatch.setattr(store, "ensure_embedding_model", lambda: None)

    store.add_chunks(
        [
            {
                "id": "chunk-1",
                "content": "content",
                "vector": None,
                "chunk_type": "turn",
                "session_id": "session-1",
                "project_path": "/project",
                "project_name": "project",
                "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
                "source_file": "/source.jsonl",
                "source_line": 1,
                "machine_id": "machine",
            }
        ]
    )

    assert "ML.PREDICT" in database.sql_calls[-1]["sql"]


def test_spanner_store_rejects_mixed_embedded_unembedded_native_batch(monkeypatch):
    """Mixed raw/vector chunks are rejected instead of partially changing semantics."""
    monkeypatch.setattr(settings, "embedding_dimension", 3)
    monkeypatch.setattr(settings, "spanner_embedding_mode", "spanner")
    monkeypatch.setattr(store_module, "_vector_dim", None)
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = FakeDatabase()

    with pytest.raises(ValueError, match="Cannot mix"):
        store.add_chunks(
            [
                {"id": "raw", "content": "raw"},
                {"id": "embedded", "content": "embedded", "vector": [0.1, 0.2, 0.3]},
            ]
        )


def test_spanner_store_query_embedding_uses_ml_predict(monkeypatch):
    """SpannerStore can generate query embeddings inside Spanner."""
    monkeypatch.setattr(settings, "embedding_dimension", 3)
    monkeypatch.setattr(settings, "spanner_embedding_mode", "spanner")
    monkeypatch.setattr(settings, "spanner_embedding_model_id", "ConversationEmbeddingModel")
    monkeypatch.setattr(settings, "vertex_query_task_type", "RETRIEVAL_QUERY")
    monkeypatch.setattr(store_module, "_vector_dim", None)
    database = FakeDatabase()
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = database
    monkeypatch.setattr(store, "ensure_embedding_model", lambda: None)

    vector = store.embed_query_text("find auth")

    assert vector == [0.1, 0.2, 0.3]
    call = database.sql_calls[-1]
    assert "ML.PREDICT" in call["sql"]
    assert "RETRIEVAL_QUERY" not in call["sql"]
    assert call["params"]["query"] == "find auth"
    assert call["params"]["task_type"] == "RETRIEVAL_QUERY"


def test_spanner_store_search_uses_cosine_distance(monkeypatch):
    """SpannerStore vector search issues a COSINE_DISTANCE query."""
    monkeypatch.setattr(settings, "embedding_dimension", 3)
    monkeypatch.setattr(store_module, "_vector_dim", None)
    database = FakeDatabase()
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = database

    results = store.search([0.1, 0.2, 0.3], project_filter="/project")

    assert results[0]["id"] == "chunk-1"
    assert results[0]["score"] == 0.123
    search_call = database.sql_calls[-1]
    assert "COSINE_DISTANCE(Vector, @query_vector)" in search_call["sql"]
    assert search_call["params"]["query_vector"] == [0.1, 0.2, 0.3]
    assert search_call["params"]["project_filter"] == "/project"


def test_spanner_store_search_uses_ann_when_vector_index_exists(monkeypatch):
    """SpannerStore switches to indexed ANN distance when the vector index exists."""
    monkeypatch.setattr(settings, "embedding_dimension", 3)
    monkeypatch.setattr(settings, "spanner_use_approx_vector_search", True)
    monkeypatch.setattr(settings, "spanner_num_leaves_to_search", 25)
    monkeypatch.setattr(store_module, "_vector_dim", None)
    database = FakeDatabase()
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = database
    monkeypatch.setattr(store, "_vector_index_exists", lambda: True)

    store.search([0.1, 0.2, 0.3])

    search_call = database.sql_calls[-1]
    assert (
        "FROM ConversationChunks@{FORCE_INDEX=ConversationChunksVectorIndex}" in search_call["sql"]
    )
    assert "APPROX_COSINE_DISTANCE(Vector, @query_vector" in search_call["sql"]
    assert '"num_leaves_to_search": 25' in search_call["sql"]


def test_spanner_store_hybrid_search_uses_full_text_and_rrf(monkeypatch):
    """Spanner hybrid search combines vector and text candidate ranks."""
    monkeypatch.setattr(settings, "embedding_dimension", 3)
    monkeypatch.setattr(settings, "spanner_hybrid_candidate_limit", 100)
    monkeypatch.setattr(store_module, "_vector_dim", None)
    database = FakeDatabase()
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = database
    monkeypatch.setattr(store, "has_fts_index", lambda: True)

    results = store.hybrid_search("oauth token", [0.1, 0.2, 0.3], project_filter="/project")

    assert results[0]["id"] == "chunk-1"
    search_call = database.sql_calls[-1]
    assert f"SEARCH({SPANNER_CONTENT_TOKENS_COLUMN}, @query)" in search_call["sql"]
    assert f"SCORE({SPANNER_CONTENT_TOKENS_COLUMN}, @query)" in search_call["sql"]
    assert "SUM(1.0 / (@rrf_k + rank + 1)) AS Score" in search_call["sql"]
    assert "1.0 - LEAST(1.0, f.Score /" in search_call["sql"]
    assert "ProjectPath = @project_filter" in search_call["sql"]
    assert search_call["params"]["query"] == "oauth token"


def test_spanner_store_hybrid_search_marks_vector_fallback_without_fts(monkeypatch):
    """When FTS is unavailable, hybrid search should tell callers it used vector search."""
    monkeypatch.setattr(settings, "embedding_dimension", 3)
    monkeypatch.setattr(store_module, "_vector_dim", None)
    database = FakeDatabase()
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = database
    monkeypatch.setattr(store, "has_fts_index", lambda: False)

    results = store.hybrid_search("oauth token", [0.1, 0.2, 0.3])

    assert results[0]["_search_type"] == "vector"


def test_spanner_store_filtered_search_stays_exact_when_ann_filter_columns_unstored(monkeypatch):
    """Project/file/operation filters avoid ANN because those columns are not stored in the index."""
    monkeypatch.setattr(settings, "embedding_dimension", 3)
    monkeypatch.setattr(settings, "spanner_use_approx_vector_search", True)
    monkeypatch.setattr(store_module, "_vector_dim", None)
    database = FakeDatabase()
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = database
    monkeypatch.setattr(store, "_vector_index_exists", lambda: True)

    store.search([0.1, 0.2, 0.3], project_filter="/project")

    search_call = database.sql_calls[-1]
    assert "COSINE_DISTANCE(Vector, @query_vector)" in search_call["sql"]
    assert "APPROX_COSINE_DISTANCE" not in search_call["sql"]


def test_vertex_location_rejects_host_injection():
    """Vertex location is interpolated into a URL host, so it must be region-shaped."""
    with pytest.raises(ValueError, match="vertex_location"):
        Settings(vertex_location="attacker.example/x#")


def test_spanner_store_rejects_zero_query_vector(monkeypatch):
    """Bad query vectors are rejected before Spanner COSINE_DISTANCE."""
    monkeypatch.setattr(settings, "embedding_dimension", 3)
    monkeypatch.setattr(store_module, "_vector_dim", None)
    store = SpannerStore(project="p", instance="i", database="d")
    store._database = FakeDatabase()

    with pytest.raises(ValueError, match="zero vector"):
        store.search([0.0, 0.0, 0.0])
