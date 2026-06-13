"""LanceDB vector store operations."""

import asyncio
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Protocol

import lancedb
import pyarrow as pa
from lancedb.rerankers import RRFReranker

from claude_history_rag.config import settings

logger = logging.getLogger(__name__)


class ConversationStore(Protocol):
    """Storage backend interface for searchable conversation chunks."""

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Persist embedded conversation chunks."""
        ...

    async def add_chunks_async(self, chunks: list[dict[str, Any]]) -> None:
        """Persist embedded conversation chunks asynchronously."""
        ...

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        project_filter: str | None = None,
        chunk_type_filter: str | None = None,
        file_path_filter: str | None = None,
        operation_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search by vector similarity with optional filters."""
        ...

    def vector_search(
        self,
        query_vector: list[float],
        limit: int = 5,
        project_filter: str | None = None,
        chunk_type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search by vector similarity."""
        ...

    async def search_async(
        self,
        query_vector: list[float],
        limit: int = 10,
        project_filter: str | None = None,
        chunk_type_filter: str | None = None,
        file_path_filter: str | None = None,
        operation_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search asynchronously by vector similarity with optional filters."""
        ...

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        limit: int = 5,
        project_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search by combined lexical and vector relevance."""
        ...

    async def hybrid_search_async(
        self,
        query: str,
        query_vector: list[float],
        limit: int = 5,
        project_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search asynchronously by combined lexical and vector relevance."""
        ...

    async def optimize_async(self) -> None:
        """Optimize backend storage."""
        ...

    def has_fts_index(self) -> bool:
        """Return whether full-text search is available."""
        ...

    def get_stats(self) -> dict[str, Any]:
        """Return backend statistics."""
        ...

    async def get_stats_async(self) -> dict[str, Any]:
        """Return backend statistics asynchronously."""
        ...

    def clear_all(self) -> int:
        """Clear all chunks."""
        ...

    async def clear_all_async(self) -> int:
        """Clear all chunks asynchronously."""
        ...

    def delete_by_machine_id(self, machine_id: str) -> int:
        """Delete chunks for one machine."""
        ...

    async def delete_by_machine_id_async(self, machine_id: str) -> int:
        """Delete chunks for one machine asynchronously."""
        ...

    def close(self) -> None:
        """Release backend resources."""
        ...

    async def close_async(self) -> None:
        """Release backend resources asynchronously."""
        ...


def _escape_sql_string(value: str) -> str:
    """Escape single quotes in SQL string values to prevent injection."""
    return value.replace("'", "''")


def _sanitize_filter_value(value: str) -> str:
    """Sanitize filter value for safe SQL interpolation."""
    # Remove any SQL metacharacters and escape quotes
    sanitized = _escape_sql_string(value)
    # Remove semicolons and comments
    had_suspicious = ";" in sanitized or "--" in sanitized
    sanitized = sanitized.replace(";", "").replace("--", "")
    if had_suspicious:
        logger.warning(f"Suspicious characters removed from filter value: {value[:50]}")
    return sanitized


def _escape_like_pattern(value: str) -> str:
    """Escape LIKE wildcards in user input to prevent pattern injection."""
    return value.replace("%", r"\%").replace("_", r"\_")


# Model name to vector dimension mapping
# https://ollama.com/search?c=embedding
MODEL_DIMENSIONS: dict[str, int] = {
    "gemini-embedding-001": 3072,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "bge-m3": 1024,
    "snowflake-arctic-embed": 1024,  # Default size (also has 768, 384, 256 variants)
    "all-minilm": 384,
}


def get_vector_dim() -> int:
    """Get vector dimension based on configured embedding model."""
    if settings.embedding_dimension is not None:
        logger.info(
            "Vector dimension from explicit setting for model '%s': %s",
            settings.embedding_model,
            settings.embedding_dimension,
        )
        return settings.embedding_dimension
    model = settings.embedding_model
    # Strip tag if present (e.g., "nomic-embed-text:latest" -> "nomic-embed-text")
    base_model = model.split(":")[0]
    dim = MODEL_DIMENSIONS.get(base_model)
    if dim is None:
        logger.warning(f"Unknown model '{base_model}', defaulting to 768 dimensions")
        return 768
    logger.info(f"Vector dimension for model '{model}': {dim}")
    return dim


# Vector dimension - computed lazily to ensure settings are loaded
_vector_dim: int | None = None


def get_current_vector_dim() -> int:
    """Get the current vector dimension (lazy initialization)."""
    global _vector_dim
    if _vector_dim is None:
        _vector_dim = get_vector_dim()
    return _vector_dim


# For backward compatibility - but use get_current_vector_dim() for runtime checks
VECTOR_DIM = 1024  # Legacy default only; new tables use a dynamic schema.

# Threshold for creating vector indexes (only needed for large collections)
VECTOR_INDEX_THRESHOLD = 10000

# Vector index parameters (IVF_HNSW_SQ configuration)
# num_partitions: Controls IVF clustering. Rule of thumb: sqrt(num_rows) to num_rows/100
# For 10K rows: 256 partitions is reasonable (sqrt(10000) = 100, 10000/100 = 100)
# For 100K rows: can increase to 512-1024 partitions
# The create_vector_index method calculates this dynamically based on row count
VECTOR_INDEX_BASE_PARTITIONS = 256
VECTOR_INDEX_MIN_PARTITIONS = 64
VECTOR_INDEX_MAX_PARTITIONS = 2048

SPANNER_TABLE_NAME = "ConversationChunks"
SPANNER_CONTENT_TOKENS_COLUMN = "ContentTokens"
SPANNER_CONTENT_SEARCH_INDEX = "ConversationChunksContentSearch"
SPANNER_VECTOR_INDEX = "ConversationChunksVectorIndex"
SPANNER_COLUMNS = [
    "Id",
    "Content",
    "Vector",
    "ChunkType",
    "SessionId",
    "ProjectPath",
    "ProjectName",
    "Timestamp",
    "UserUuid",
    "AssistantUuid",
    "FilePath",
    "Operation",
    "Model",
    "SourceFile",
    "SourceLine",
    "ParentChunkId",
    "ChildChunkIds",
    "MachineId",
]


def get_spanner_schema_ddl() -> list[str]:
    """Return DDL for the Spanner storage backend."""
    dim = get_current_vector_dim()
    ddl = [
        f"""
        CREATE TABLE {SPANNER_TABLE_NAME} (
            Id STRING(64) NOT NULL,
            Content STRING(MAX) NOT NULL,
            {SPANNER_CONTENT_TOKENS_COLUMN} TOKENLIST
                AS (TOKENIZE_FULLTEXT(Content)) HIDDEN,
            Vector ARRAY<FLOAT32>(vector_length=>{dim}),
            ChunkType STRING(32) NOT NULL,
            SessionId STRING(128) NOT NULL,
            ProjectPath STRING(MAX) NOT NULL,
            ProjectName STRING(256) NOT NULL,
            Timestamp TIMESTAMP NOT NULL,
            UserUuid STRING(128),
            AssistantUuid STRING(128),
            FilePath STRING(MAX),
            Operation STRING(32),
            Model STRING(256),
            SourceFile STRING(MAX) NOT NULL,
            SourceLine INT64 NOT NULL,
            ParentChunkId STRING(64),
            ChildChunkIds ARRAY<STRING(64)>,
            MachineId STRING(256)
        ) PRIMARY KEY (Id)
        """,
        f"CREATE INDEX {SPANNER_TABLE_NAME}ByProject ON {SPANNER_TABLE_NAME}(ProjectPath)",
        f"CREATE INDEX {SPANNER_TABLE_NAME}ByMachine ON {SPANNER_TABLE_NAME}(MachineId)",
    ]
    if settings.spanner_enable_full_text:
        ddl.append(
            f"""
            CREATE SEARCH INDEX {SPANNER_CONTENT_SEARCH_INDEX}
            ON {SPANNER_TABLE_NAME}({SPANNER_CONTENT_TOKENS_COLUMN})
            """
        )
    return ddl


def get_spanner_vector_index_ddl() -> str:
    """Return DDL for the Spanner ANN vector index."""
    return f"""
        CREATE VECTOR INDEX {SPANNER_VECTOR_INDEX}
        ON {SPANNER_TABLE_NAME}(Vector)
        STORING (ChunkType, SessionId, ProjectName, MachineId)
        WHERE Vector IS NOT NULL
        OPTIONS (
            distance_type = 'COSINE',
            tree_depth = 2,
            num_leaves = {settings.spanner_vector_index_leaves}
        )
    """


def _spanner_embedding_endpoint(project: str, location: str, model: str) -> str:
    """Return Agent Platform model endpoint path for Spanner CREATE MODEL."""
    return (
        f"//aiplatform.googleapis.com/projects/{_escape_sql_string(project)}"
        f"/locations/{_escape_sql_string(location)}"
        f"/publishers/google/models/{_escape_sql_string(model)}"
    )


def get_spanner_embedding_model_ddl(project: str) -> str:
    """Return DDL for registering the configured embedding model in Spanner."""
    model_project = settings.vertex_project or project
    endpoint = _spanner_embedding_endpoint(
        project=model_project,
        location=settings.vertex_location,
        model=settings.embedding_model,
    )
    return f"""
        CREATE MODEL IF NOT EXISTS {settings.spanner_embedding_model_id}
        INPUT(
            content STRING(MAX),
            task_type STRING(MAX)
        )
        OUTPUT(
            embeddings STRUCT<
                statistics STRUCT<truncated BOOL, token_count FLOAT64>,
                values ARRAY<FLOAT64>
            >
        )
        REMOTE OPTIONS (
            endpoint = '{endpoint}',
            default_batch_size = {settings.spanner_embedding_rpc_batch_size}
        )
    """


def _validate_vector(vector: list[float], chunk_id: str = "unknown") -> None:
    """Validate vector content for NaN, Inf, and zero vectors.

    Args:
        vector: The embedding vector to validate
        chunk_id: Chunk identifier for error messages

    Raises:
        ValueError: If vector contains NaN, Inf, or is a zero vector
    """
    if not vector:
        raise ValueError(f"Chunk {chunk_id}: Vector is empty")

    # Check for NaN or Inf values
    for i, val in enumerate(vector):
        if not math.isfinite(val):
            raise ValueError(
                f"Chunk {chunk_id}: Vector contains non-finite value at index {i}: {val}"
            )

    # Check for zero vector (all values are zero)
    if all(abs(v) < 1e-10 for v in vector):
        raise ValueError(f"Chunk {chunk_id}: Vector is a zero vector")


def _normalize_timestamp(value: Any) -> dt:
    """Normalize timestamp values for storage backends."""
    if isinstance(value, dt):
        timestamp = value
    elif isinstance(value, str):
        timestamp = dt.fromisoformat(value.replace("Z", "+00:00"))
    else:
        timestamp = dt.now(UTC)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _format_timestamp(value: Any) -> str | None:
    """Format backend timestamp values for API responses."""
    if isinstance(value, dt):
        return value.isoformat()
    if value is None:
        return None
    return str(value)


def _relevance_to_distance(value: Any) -> float:
    """Convert higher-is-better relevance into lower-is-better distance."""
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return 1.0
    return 1.0 - max(0.0, min(1.0, float(value)))


def get_conversation_chunk_schema() -> pa.Schema:
    """Create the LanceDB schema for the configured embedding dimension."""
    dim = get_current_vector_dim()
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dim), nullable=False),
            pa.field("chunk_type", pa.string(), nullable=False),
            pa.field("session_id", pa.string(), nullable=False),
            pa.field("project_path", pa.string(), nullable=False),
            pa.field("project_name", pa.string(), nullable=False),
            pa.field("timestamp", pa.timestamp("us"), nullable=False),
            pa.field("user_uuid", pa.string()),
            pa.field("assistant_uuid", pa.string()),
            pa.field("file_path", pa.string()),
            pa.field("operation", pa.string()),
            pa.field("model", pa.string()),
            pa.field("source_file", pa.string(), nullable=False),
            pa.field("source_line", pa.int64(), nullable=False),
            pa.field("parent_chunk_id", pa.string()),
            pa.field("child_chunk_ids", pa.list_(pa.string())),
            pa.field("machine_id", pa.string()),
        ]
    )


class VectorStore:
    """LanceDB vector store for conversation chunks."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.db_path
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None
        self._db_lock = threading.Lock()
        self._table_lock = threading.Lock()

    def _ensure_db_path(self) -> None:
        """Ensure database directory exists."""
        self.db_path.mkdir(parents=True, exist_ok=True)

    def connect(self) -> lancedb.DBConnection:
        """Get or create database connection with thread-safe lazy initialization."""
        # Double-check locking pattern for thread safety
        if self._db is None:
            with self._db_lock:
                if self._db is None:
                    self._ensure_db_path()
                    self._db = lancedb.connect(str(self.db_path))
        return self._db

    def get_table(self) -> lancedb.table.Table:
        """Get or create the conversations table with thread-safe lazy initialization."""
        # Double-check locking pattern for thread safety
        if self._table is not None:
            return self._table
        with self._table_lock:
            if self._table is not None:
                return self._table

            db = self.connect()

            if "conversations" in db.table_names():
                self._table = db.open_table("conversations")
            else:
                # Create empty table with schema
                self._table = db.create_table(
                    "conversations",
                    schema=get_conversation_chunk_schema(),
                    mode="overwrite",
                )
                logger.info("Created new conversations table")

            return self._table

    def reset_connections(self) -> None:
        """Reset cached DB/table handles (forces reopen on next use)."""
        with self._table_lock:
            self._table = None
        with self._db_lock:
            self._db = None

    def _should_retry_on_lance_error(self, error: Exception) -> bool:
        """Detect errors caused by stale table files after reindex/clear."""
        message = str(error)
        return "LanceError(IO)" in message or "Not found:" in message

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Add chunks to the table."""
        if not chunks:
            return

        # Validate vector dimensions and types before adding
        expected_dim = get_current_vector_dim()
        for chunk in chunks:
            vec = chunk.get("vector")
            chunk_id = chunk.get("id", "unknown")
            if vec is not None:
                if not isinstance(vec, list):
                    raise ValueError(f"Vector must be a list, got {type(vec).__name__}")
                if len(vec) != expected_dim:
                    raise ValueError(
                        f"Vector dimension mismatch: expected {expected_dim}, got {len(vec)}"
                    )
                if not all(isinstance(v, (int, float)) for v in vec):
                    raise ValueError("Vector must contain only numeric values")
                # Validate vector content for NaN, Inf, and zero vectors
                _validate_vector(vec, chunk_id)

        table = self.get_table()
        try:
            table.add(chunks)
            logger.info(f"Added {len(chunks)} chunks to store")
        except Exception as e:
            chunk_ids = [c.get("id", "unknown")[:8] for c in chunks[:5]]
            logger.error(
                f"Failed to add {len(chunks)} chunks (first IDs: {chunk_ids}): {type(e).__name__}",
                exc_info=True,
            )
            raise

    async def add_chunks_async(self, chunks: list[dict[str, Any]]) -> None:
        """Add chunks asynchronously."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.add_chunks, chunks)

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        project_filter: str | None = None,
        chunk_type_filter: str | None = None,
        file_path_filter: str | None = None,
        operation_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks.

        Args:
            query_vector: Query embedding vector
            limit: Maximum results to return
            project_filter: Filter by project path
            chunk_type_filter: Filter by chunk type
            file_path_filter: Filter by file path (supports LIKE)
            operation_filter: Filter by operation type (edit/write)

        Returns:
            List of matching chunks with scores
        """
        table = self.get_table()

        # Build query
        query = table.search(query_vector)

        # Apply filters (sanitized to prevent SQL injection)
        filters = []
        if project_filter:
            safe_project = _sanitize_filter_value(project_filter)
            if safe_project:  # Only add filter if non-empty after sanitization
                filters.append(f"project_path = '{safe_project}'")
            else:
                logger.warning("project_filter became empty after sanitization, ignoring")
        if chunk_type_filter:
            safe_chunk_type = _sanitize_filter_value(chunk_type_filter)
            if safe_chunk_type:
                filters.append(f"chunk_type = '{safe_chunk_type}'")
            else:
                logger.warning("chunk_type_filter became empty after sanitization, ignoring")
        if file_path_filter:
            safe_file_path = _sanitize_filter_value(file_path_filter)
            if safe_file_path:
                safe_file_path = _escape_like_pattern(safe_file_path)
                filters.append(f"file_path LIKE '%{safe_file_path}%' ESCAPE '\\'")
            else:
                logger.warning("file_path_filter became empty after sanitization, ignoring")
        if operation_filter:
            safe_operation = _sanitize_filter_value(operation_filter)
            if safe_operation:
                filters.append(f"operation = '{safe_operation}'")
            else:
                logger.warning("operation_filter became empty after sanitization, ignoring")

        if filters:
            query = query.where(" AND ".join(filters))

        try:
            results = query.limit(limit).to_list()
        except RuntimeError as e:
            if self._should_retry_on_lance_error(e):
                logger.warning("Search hit stale table state, reopening store and retrying")
                self.reset_connections()
                table = self.get_table()
                query = table.search(query_vector)
                if filters:
                    query = query.where(" AND ".join(filters))
                results = query.limit(limit).to_list()
            else:
                raise

        # Validate results before accessing dict keys. LanceDB hybrid returns a
        # higher-is-better relevance score; normalize to the store contract where
        # score is distance-like and lower is better.
        return [
            {
                "id": r["id"],
                "content": r["content"],
                "chunk_type": r["chunk_type"],
                "session_id": r["session_id"],
                "project_path": r["project_path"],
                "project_name": r["project_name"],
                "timestamp": r["timestamp"].isoformat()
                if isinstance(r.get("timestamp"), dt)
                else None,
                "file_path": r.get("file_path"),
                "operation": r.get("operation"),
                "machine_id": r.get("machine_id"),
                "score": r.get("_distance", 0),
            }
            for r in results
            if isinstance(r, dict) and "id" in r and "content" in r
        ]

    def vector_search(
        self,
        query_vector: list[float],
        limit: int = 5,
        project_filter: str | None = None,
        chunk_type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Protocol alias for vector search."""
        return self.search(
            query_vector=query_vector,
            limit=limit,
            project_filter=project_filter,
            chunk_type_filter=chunk_type_filter,
        )

    async def search_async(
        self,
        query_vector: list[float],
        limit: int = 10,
        project_filter: str | None = None,
        chunk_type_filter: str | None = None,
        file_path_filter: str | None = None,
        operation_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(
                query_vector,
                limit,
                project_filter,
                chunk_type_filter,
                file_path_filter,
                operation_filter,
            ),
        )

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        limit: int = 10,
        project_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining vector and full-text search with RRF reranking.

        Falls back to vector-only search if FTS index is not available.
        """
        table = self.get_table()

        try:
            # Try hybrid search with both vector and text components
            search_query = table.search(query_type="hybrid").vector(query_vector).text(query)

            # Apply project filter if specified (sanitized)
            if project_filter:
                safe_project = _sanitize_filter_value(project_filter)
                if safe_project:
                    search_query = search_query.where(f"project_path = '{safe_project}'")
                else:
                    logger.warning("project_filter became empty after sanitization, ignoring")

            # Use RRF reranker for fusion
            try:
                results = search_query.rerank(reranker=RRFReranker()).limit(limit).to_list()
            except RuntimeError as e:
                if self._should_retry_on_lance_error(e):
                    logger.warning(
                        "Hybrid search hit stale table state, reopening store and retrying"
                    )
                    self.reset_connections()
                    table = self.get_table()
                    search_query = (
                        table.search(query_type="hybrid").vector(query_vector).text(query)
                    )
                    if project_filter:
                        safe_project = _sanitize_filter_value(project_filter)
                        if safe_project:
                            search_query = search_query.where(f"project_path = '{safe_project}'")
                    results = search_query.rerank(reranker=RRFReranker()).limit(limit).to_list()
                else:
                    raise
        except RuntimeError as e:
            if "INVERTED index" in str(e) or "full text search" in str(e).lower():
                # FTS index not available, fall back to vector-only search
                logger.warning(f"FTS index not available, falling back to vector search: {e}")
                return self.search(
                    query_vector=query_vector,
                    limit=limit,
                    project_filter=project_filter,
                )
            else:
                raise

        # Validate results before accessing dict keys
        return [
            {
                "id": r["id"],
                "content": r["content"],
                "chunk_type": r["chunk_type"],
                "session_id": r["session_id"],
                "project_path": r["project_path"],
                "project_name": r["project_name"],
                "timestamp": r["timestamp"].isoformat()
                if isinstance(r.get("timestamp"), dt)
                else None,
                "file_path": r.get("file_path"),
                "operation": r.get("operation"),
                "machine_id": r.get("machine_id"),
                "score": _relevance_to_distance(r.get("_relevance_score")),
            }
            for r in results
            if isinstance(r, dict) and "id" in r and "content" in r
        ]

    async def hybrid_search_async(
        self,
        query: str,
        query_vector: list[float],
        limit: int = 10,
        project_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.hybrid_search(query, query_vector, limit, project_filter),
        )

    def create_fts_index(self) -> None:
        """Create full-text search index on content field."""
        table = self.get_table()
        try:
            table.create_fts_index(
                "content",
                use_tantivy=True,
                with_position=True,
                replace=True,
            )
            logger.info("Created FTS index on content field")
        except Exception as e:
            logger.warning(f"Failed to create FTS index: {e}")

    def create_vector_index(self) -> None:
        """Create vector index for large collections.

        Dynamically calculates num_partitions based on row count for optimal performance.
        Uses IVF_HNSW_SQ index type which provides a good balance of speed and accuracy.
        """
        table = self.get_table()

        try:
            row_count = table.count_rows()
        except Exception as e:
            logger.error(f"Failed to get row count for vector index creation: {type(e).__name__}")
            return

        if row_count < VECTOR_INDEX_THRESHOLD:
            logger.info(f"Skipping vector index creation (only {row_count} rows)")
            return

        # Calculate optimal num_partitions based on row count
        # Rule of thumb: sqrt(num_rows) to num_rows/100, clamped to reasonable bounds
        import math

        calculated_partitions = int(math.sqrt(row_count))
        num_partitions = max(
            VECTOR_INDEX_MIN_PARTITIONS, min(calculated_partitions, VECTOR_INDEX_MAX_PARTITIONS)
        )

        # Ensure num_partitions doesn't exceed row_count
        num_partitions = min(num_partitions, row_count)

        try:
            table.create_index(
                metric="cosine",
                num_partitions=num_partitions,
                index_type="IVF_HNSW_SQ",
            )
            logger.info(
                f"Created IVF_HNSW_SQ vector index with {num_partitions} partitions for {row_count} rows"
            )
        except Exception as e:
            logger.warning(f"Failed to create vector index: {e}")

    def has_fts_index(self) -> bool:
        """Check if FTS index exists on the table by attempting a FTS query."""
        table = self.get_table()
        try:
            # The most reliable way to check if FTS is available is to try a query
            # FTS indexes are stored separately and may not appear in list_indices()
            table.search("test", query_type="fts").limit(1).to_list()
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "inverted index" in error_msg or "fts" in error_msg or "full text" in error_msg:
                return False
            # Other errors might indicate FTS exists but query failed for other reasons
            logger.debug(f"FTS check error (assuming not available): {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        # If the DB directory is empty, report zero chunks to avoid stale counts.
        try:
            if not self.db_path.exists():
                return {
                    "total_chunks": 0,
                    "db_path": str(self.db_path),
                    "fts_index_available": False,
                }
            has_files = any(p.is_file() for p in self.db_path.rglob("*"))
            if not has_files:
                return {
                    "total_chunks": 0,
                    "db_path": str(self.db_path),
                    "fts_index_available": False,
                }
        except OSError:
            pass

        error_message = None
        try:
            table = self.get_table()
            row_count = table.count_rows()
        except Exception as e:
            error_message = f"Stats unavailable: {type(e).__name__}"
            logger.error(
                f"Failed to get row count for stats: {type(e).__name__}",
                exc_info=True,
            )
            # Retry once after resetting connections (handles stale handles)
            try:
                self.reset_connections()
                table = self.get_table()
                row_count = table.count_rows()
                error_message = None
            except Exception as retry_error:
                error_message = f"Stats unavailable after retry: {type(retry_error).__name__}"
                logger.error(
                    f"Retry failed to get row count: {type(retry_error).__name__}",
                    exc_info=True,
                )
                row_count = 0

        result = {
            "total_chunks": row_count,
            "db_path": str(self.db_path),
            "fts_index_available": self.has_fts_index(),
        }
        if error_message:
            result["error"] = error_message
        return result

    async def get_stats_async(self) -> dict[str, Any]:
        """Get store statistics asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_stats)

    def clear_all(self) -> int:
        """Clear all chunks from the database.

        This is useful for forcing a complete re-index with a new embedding model.
        Deletes the entire database directory to ensure clean schema recreation.

        Returns:
            Number of chunks deleted.
        """
        import gc
        import shutil
        import time

        row_count = 0

        # Get row count before clearing (for logging)
        try:
            table = self.get_table()
            row_count = table.count_rows()
        except Exception:
            pass  # Table might not exist

        # Clear all cached references
        with self._table_lock:
            self._table = None
        with self._db_lock:
            self._db = None

        # Force garbage collection to release file handles
        gc.collect()
        time.sleep(0.5)  # Brief delay for OS to release handles

        # Delete the entire database directory
        db_path = self.db_path
        if db_path.exists():
            try:
                shutil.rmtree(db_path)
                logger.info(f"Deleted database directory: {db_path}")
            except OSError as e:
                # Retry once after more aggressive cleanup
                logger.warning(f"First rmtree attempt failed: {e}, retrying...")
                gc.collect()
                time.sleep(1.0)
                shutil.rmtree(db_path)
                logger.info(f"Deleted database directory on retry: {db_path}")

        logger.info(f"Cleared {row_count} chunks from database (deleted {db_path})")
        return row_count

    async def clear_all_async(self) -> int:
        """Clear all chunks from the database asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.clear_all)

    def delete_by_machine_id(self, machine_id: str) -> int:
        """Delete all chunks for a specific machine_id."""
        if not machine_id:
            return 0

        table = self.get_table()
        safe_machine_id = _sanitize_filter_value(machine_id)
        count = 0

        try:
            matches = table.search().where(f"machine_id = '{safe_machine_id}'").to_list()
            count = len(matches)
        except Exception as e:
            logger.warning(f"Failed to estimate rows for purge: {type(e).__name__}: {e}")

        try:
            table.delete(f"machine_id = '{safe_machine_id}'")
            logger.info(f"Deleted {count} chunks for machine_id={machine_id}")
        except Exception as e:
            logger.error(f"Failed to delete chunks for machine_id={machine_id}: {e}")
            raise

        return count

    async def delete_by_machine_id_async(self, machine_id: str) -> int:
        """Delete all chunks for a specific machine_id asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.delete_by_machine_id, machine_id)

    def optimize(self) -> None:
        """Optimize the database by compacting and creating indexes if needed."""
        table = self.get_table()

        try:
            row_count = table.count_rows()
        except Exception as e:
            logger.error(f"Failed to get row count for optimization: {type(e).__name__}")
            row_count = 0

        logger.info(f"Optimizing store with {row_count} chunks...")

        # Create FTS index for hybrid search
        self.create_fts_index()

        # Create vector index for large collections
        if row_count >= VECTOR_INDEX_THRESHOLD:
            self.create_vector_index()

        # Compact files and clean up old versions
        try:
            from datetime import timedelta

            # Use new optimize API which handles both compaction and cleanup
            # This replaces the deprecated compact_files() and cleanup_old_versions()
            cleanup_seconds = settings.optimization_cleanup_older_than_seconds
            delete_unverified = settings.optimization_delete_unverified

            logger.info(
                f"Running optimization (cleanup_older_than={cleanup_seconds}s, "
                f"delete_unverified={delete_unverified})"
            )

            table.optimize(
                cleanup_older_than=timedelta(seconds=cleanup_seconds),
                delete_unverified=delete_unverified,
            )
            logger.info("Database optimization complete")
        except Exception as e:
            logger.warning(
                f"Validation during optimization failed (this is expected if files are changing): {e}"
            )
            # Fallback to simple compaction if full optimization fails
            try:
                table.compact_files()
                logger.info("Fallback: Compacted table files")
            except Exception as compact_err:
                logger.warning(f"Fallback compaction also failed: {compact_err}")

        logger.info("Store optimization routine finished")

    async def optimize_async(self) -> None:
        """Optimize the database asynchronously."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.optimize)

    def chunk_exists(self, chunk_id: str) -> bool:
        """Check if a chunk already exists.

        Note on SQL safety: String interpolation is safe here because chunk_id is
        rigorously validated to contain ONLY hexadecimal characters [0-9a-f] with
        exact length 16. This makes SQL injection impossible as there are no quotes,
        semicolons, or other SQL metacharacters that could escape the WHERE clause.
        The validation ensures chunk_id matches the pattern: ^[0-9a-f]{16}$
        """
        try:
            # Validate chunk_id is hex (SHA256 hash prefix) and exactly 16 chars
            if not chunk_id or len(chunk_id) != 16:
                logger.warning(
                    f"Invalid chunk_id length: {len(chunk_id) if chunk_id else 0}, expected 16"
                )
                return False
            if not all(c in "0123456789abcdef" for c in chunk_id):
                logger.warning("Invalid chunk_id format: non-hex characters")
                return False

            table = self.get_table()
            # After validation, chunk_id is guaranteed to be hex-only, making string
            # interpolation safe (no SQL metacharacters possible in [0-9a-f])
            results = table.search().where(f"id = '{chunk_id}'").limit(1).to_list()
            return len(results) > 0
        except Exception as e:
            logger.warning(f"Failed to check chunk existence: {type(e).__name__}")
            return False

    def close(self) -> None:
        """Close database connection and release resources.

        LanceDB connections in Python use automatic resource management via garbage
        collection. There is no explicit close() method on DBConnection objects.
        Setting references to None allows the garbage collector to clean up
        underlying resources (file handles, memory buffers, etc.) when the
        connection is no longer referenced.

        Tables created from the connection remain independent and valid even after
        the connection is released, as per LanceDB's design.

        References:
        - LanceDB Python API: Connections are garbage-collected automatically
        - Safe to call multiple times (idempotent operation)
        """
        self._db = None
        self._table = None
        logger.debug("Released database connection references for garbage collection")

    async def close_async(self) -> None:
        """Close database connection asynchronously.

        This async wrapper ensures connection cleanup happens off the event loop,
        though in practice setting references to None is a synchronous operation.
        The executor ensures consistency with other async methods and prevents
        any potential blocking during object finalization.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.close)


class SpannerStore:
    """Cloud Spanner storage backend with vector and native hybrid search."""

    def __init__(
        self,
        project: str | None = None,
        instance: str | None = None,
        database: str | None = None,
        create_schema: bool = True,
    ):
        self.project = project if project is not None else settings.spanner_project
        self.instance_id = instance if instance is not None else settings.spanner_instance
        self.database_id = database if database is not None else settings.spanner_database
        self.create_schema = create_schema
        self._client = None
        self._instance = None
        self._database = None
        self._lock = threading.Lock()

    def _resolve_project(self) -> str:
        """Resolve the Spanner project from config or ADC."""
        if self.project:
            return self.project
        try:
            from claude_history_rag.gcp_auth import default_project_and_credentials
        except ImportError as e:
            raise RuntimeError(
                "Spanner storage backend requires google-cloud-spanner. "
                "Install with: uv sync --extra server"
            ) from e
        project, _ = default_project_and_credentials(
            ["https://www.googleapis.com/auth/cloud-platform"]
        )
        if not project:
            raise RuntimeError(
                "Spanner project is not configured. Set CLAUDE_HISTORY_RAG_SPANNER_PROJECT "
                "or configure Application Default Credentials with a project."
            )
        self.project = project
        return project

    def connect(self):
        """Get or create a Spanner database handle."""
        if self._database is not None:
            return self._database
        with self._lock:
            if self._database is not None:
                return self._database
            if not self.instance_id or not self.database_id:
                raise RuntimeError(
                    "Spanner storage requires CLAUDE_HISTORY_RAG_SPANNER_INSTANCE and "
                    "CLAUDE_HISTORY_RAG_SPANNER_DATABASE."
                )
            try:
                from google.cloud import spanner
            except ImportError as e:
                raise RuntimeError(
                    "Spanner storage backend requires google-cloud-spanner. "
                    "Install with: uv sync --extra server"
                ) from e

            project, credentials = self._project_and_credentials()
            self._client = spanner.Client(
                project=project, credentials=credentials, disable_builtin_metrics=True
            )
            self._instance = self._client.instance(self.instance_id)
            self._database = self._instance.database(self.database_id)
            if self.create_schema:
                self.ensure_schema()
            return self._database

    def ensure_database(self) -> None:
        """Create the configured Spanner database when it does not exist."""
        if not self.instance_id or not self.database_id:
            raise RuntimeError(
                "Spanner database creation requires spanner_instance and spanner_database."
            )
        try:
            from google.api_core.exceptions import AlreadyExists
            from google.cloud import spanner
        except ImportError as e:
            raise RuntimeError(
                "Spanner storage backend requires google-cloud-spanner. "
                "Install with: uv sync --extra server"
            ) from e

        if self._client is None:
            project, credentials = self._project_and_credentials()
            self._client = spanner.Client(
                project=project, credentials=credentials, disable_builtin_metrics=True
            )
        instance = self._client.instance(self.instance_id)
        database = instance.database(self.database_id, ddl_statements=get_spanner_schema_ddl())
        try:
            operation = database.create()
            operation.result()
            logger.info("Created Spanner database %s", self.database_id)
        except AlreadyExists:
            logger.info("Spanner database %s already exists", self.database_id)
        self._instance = instance
        self._database = instance.database(self.database_id)

    def _project_and_credentials(self):
        """Resolve project and credentials for Spanner clients."""
        from claude_history_rag.gcp_auth import default_project_and_credentials

        resolved_project, credentials = default_project_and_credentials(
            ["https://www.googleapis.com/auth/cloud-platform"]
        )
        if credentials is not None and not credentials.valid:
            from google.auth.transport.requests import Request

            credentials.refresh(Request())
        project = self.project or resolved_project
        if not project:
            raise RuntimeError(
                "Spanner project is not configured. Set CLAUDE_HISTORY_RAG_SPANNER_PROJECT "
                "or configure a gcloud/ADC project."
            )
        self.project = project
        return project, credentials

    def ensure_schema(self) -> None:
        """Create Spanner tables/indexes required by this backend if missing."""
        database = self._database
        if database is None:
            return
        if self._table_exists(SPANNER_TABLE_NAME):
            self.ensure_search_schema()
            self.ensure_embedding_model()
            return
        operation = database.update_ddl(get_spanner_schema_ddl())
        operation.result()
        logger.info("Created Spanner schema for %s", SPANNER_TABLE_NAME)
        self.ensure_embedding_model()

    def _run_ddl(self, ddl: str, description: str) -> bool:
        """Run one DDL statement and treat already-existing objects as success."""
        database = self._database
        if database is None:
            return False
        try:
            from google.api_core.exceptions import (
                AlreadyExists,
                FailedPrecondition,
                InvalidArgument,
            )
        except ImportError as e:
            raise RuntimeError("google-api-core is required") from e
        try:
            operation = database.update_ddl([ddl])
            operation.result()
            logger.info("Created Spanner %s", description)
            return True
        except AlreadyExists:
            logger.info("Spanner %s already exists", description)
            return True
        except (FailedPrecondition, InvalidArgument) as e:
            if "already exists" in str(e).lower():
                logger.info("Spanner %s already exists", description)
                return True
            logger.warning("Failed to create Spanner %s: %s", description, e)
            return False

    def ensure_search_schema(self) -> None:
        """Ensure generated token column and full-text search index exist."""
        if not settings.spanner_enable_full_text:
            return
        if not self._column_exists(SPANNER_TABLE_NAME, SPANNER_CONTENT_TOKENS_COLUMN):
            self._run_ddl(
                f"""
                ALTER TABLE {SPANNER_TABLE_NAME}
                ADD COLUMN {SPANNER_CONTENT_TOKENS_COLUMN} TOKENLIST
                    AS (TOKENIZE_FULLTEXT(Content)) HIDDEN
                """,
                f"generated token column {SPANNER_CONTENT_TOKENS_COLUMN}",
            )
        if not self._index_exists(SPANNER_CONTENT_SEARCH_INDEX):
            self._run_ddl(
                f"""
                CREATE SEARCH INDEX {SPANNER_CONTENT_SEARCH_INDEX}
                ON {SPANNER_TABLE_NAME}({SPANNER_CONTENT_TOKENS_COLUMN})
                """,
                f"search index {SPANNER_CONTENT_SEARCH_INDEX}",
            )

    def ensure_embedding_model(self) -> None:
        """Register the configured Gemini embedding model in Spanner when enabled."""
        if settings.spanner_embedding_mode != "spanner":
            return
        self._run_ddl(
            get_spanner_embedding_model_ddl(self.project or self._resolve_project()),
            f"embedding model {settings.spanner_embedding_model_id}",
        )

    def _table_exists(self, table_name: str) -> bool:
        """Return whether a table exists in the configured Spanner database."""
        database = self._database
        if database is None:
            return False
        sql = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_catalog = '' AND table_schema = '' AND table_name = @table_name
        """
        try:
            from google.cloud.spanner_v1 import param_types
        except ImportError as e:
            raise RuntimeError("google-cloud-spanner is required") from e
        with database.snapshot() as snapshot:
            rows = list(
                snapshot.execute_sql(
                    sql,
                    params={"table_name": table_name},
                    param_types={"table_name": param_types.STRING},
                )
            )
        return bool(rows)

    def _column_exists(self, table_name: str, column_name: str) -> bool:
        """Return whether a column exists in the configured Spanner database."""
        database = self._database
        if database is None:
            return False
        sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_catalog = ''
              AND table_schema = ''
              AND table_name = @table_name
              AND column_name = @column_name
        """
        try:
            from google.cloud.spanner_v1 import param_types
        except ImportError as e:
            raise RuntimeError("google-cloud-spanner is required") from e
        with database.snapshot() as snapshot:
            rows = list(
                snapshot.execute_sql(
                    sql,
                    params={"table_name": table_name, "column_name": column_name},
                    param_types={
                        "table_name": param_types.STRING,
                        "column_name": param_types.STRING,
                    },
                )
            )
        return bool(rows)

    def _index_exists(self, index_name: str) -> bool:
        """Return whether any Spanner index-like object exists by name."""
        database = self._database
        if database is None:
            return False
        sql = """
            SELECT index_name
            FROM information_schema.indexes
            WHERE table_catalog = '' AND table_schema = '' AND index_name = @index_name
        """
        try:
            from google.cloud.spanner_v1 import param_types
        except ImportError as e:
            raise RuntimeError("google-cloud-spanner is required") from e
        try:
            with database.snapshot() as snapshot:
                rows = list(
                    snapshot.execute_sql(
                        sql,
                        params={"index_name": index_name},
                        param_types={"index_name": param_types.STRING},
                    )
                )
            return bool(rows)
        except Exception as e:
            logger.debug("Failed to inspect Spanner indexes: %s", e)
            return False

    def _vector_index_exists(self) -> bool:
        """Return whether the configured Spanner vector index exists."""
        return self._index_exists(SPANNER_VECTOR_INDEX)

    def _row_count(self) -> int:
        """Return the current number of stored chunks."""
        database = self.connect()
        with database.snapshot() as snapshot:
            rows = list(snapshot.execute_sql(f"SELECT COUNT(*) FROM {SPANNER_TABLE_NAME}"))
        return int(rows[0][0]) if rows else 0

    def _embedding_counts(self) -> tuple[int, int]:
        """Return (total_chunks, embedded_chunks) in one scan — drives backfill progress."""
        database = self.connect()
        sql = f"SELECT COUNT(*), COUNTIF(Vector IS NOT NULL) FROM {SPANNER_TABLE_NAME}"
        with database.snapshot() as snapshot:
            rows = list(snapshot.execute_sql(sql))
        if rows:
            return int(rows[0][0]), int(rows[0][1])
        return 0, 0

    def create_vector_index(self, force: bool = False) -> bool:
        """Create Spanner ANN vector index when enabled and useful."""
        if not settings.spanner_enable_vector_index:
            return False
        database = self.connect()
        del database
        if self._vector_index_exists():
            return True
        row_count = self._row_count()
        if not force and row_count < VECTOR_INDEX_THRESHOLD:
            logger.info("Skipping Spanner vector index creation (only %s rows)", row_count)
            return False
        return self._run_ddl(get_spanner_vector_index_ddl(), f"vector index {SPANNER_VECTOR_INDEX}")

    def _chunk_values(self, chunk: dict[str, Any]) -> list[Any]:
        """Convert an embedded chunk dict to Spanner mutation values."""
        return [
            chunk.get("id"),
            chunk.get("content"),
            [float(v) for v in chunk.get("vector", [])],
            chunk.get("chunk_type"),
            chunk.get("session_id"),
            chunk.get("project_path"),
            chunk.get("project_name"),
            _normalize_timestamp(chunk.get("timestamp")),
            chunk.get("user_uuid"),
            chunk.get("assistant_uuid"),
            chunk.get("file_path"),
            chunk.get("operation"),
            chunk.get("model"),
            chunk.get("source_file"),
            int(chunk.get("source_line", 0)),
            chunk.get("parent_chunk_id"),
            chunk.get("child_chunk_ids"),
            chunk.get("machine_id"),
        ]

    # Field order for the ARRAY<STRUCT> ML.PREDICT input. MUST match _chunk_struct_value()
    # and the column list in _batched_insert_with_spanner_embedding_sql(). Vector is omitted
    # because ML.PREDICT computes it; "content"/"task_type" are the model's INPUT columns and
    # every other field is passed through to the prediction output relation.
    _CHUNK_STRUCT_FIELDS = [
        "id",
        "content",
        "chunk_type",
        "session_id",
        "project_path",
        "project_name",
        "timestamp",
        "user_uuid",
        "assistant_uuid",
        "file_path",
        "operation",
        "model",
        "source_file",
        "source_line",
        "parent_chunk_id",
        "child_chunk_ids",
        "machine_id",
    ]

    def _chunk_struct_param_type(self) -> Any:
        """Return the ARRAY<STRUCT> Spanner parameter type for a batch of chunks."""
        try:
            from google.cloud.spanner_v1 import param_types
        except ImportError as e:
            raise RuntimeError("google-cloud-spanner is required") from e
        field_types = {
            "timestamp": param_types.TIMESTAMP,
            "source_line": param_types.INT64,
            "child_chunk_ids": param_types.Array(param_types.STRING),
        }
        struct = param_types.Struct(
            [
                param_types.StructField(name, field_types.get(name, param_types.STRING))
                for name in self._CHUNK_STRUCT_FIELDS
            ]
        )
        return param_types.Array(struct)

    def _chunk_struct_value(self, chunk: dict[str, Any]) -> list[Any]:
        """Convert a chunk dict to a positional STRUCT value (order = _CHUNK_STRUCT_FIELDS)."""
        return [
            chunk.get("id"),
            chunk.get("content"),
            chunk.get("chunk_type"),
            chunk.get("session_id"),
            chunk.get("project_path"),
            chunk.get("project_name"),
            _normalize_timestamp(chunk.get("timestamp")),
            chunk.get("user_uuid"),
            chunk.get("assistant_uuid"),
            chunk.get("file_path"),
            chunk.get("operation"),
            chunk.get("model"),
            chunk.get("source_file"),
            int(chunk.get("source_line", 0)),
            chunk.get("parent_chunk_id"),
            chunk.get("child_chunk_ids"),
            chunk.get("machine_id"),
        ]

    def _chunk_values_no_vector(self, chunk: dict[str, Any]) -> list[Any]:
        """Mutation values for a chunk with the Vector column omitted (left NULL)."""
        values = self._chunk_values(chunk)
        # SPANNER_COLUMNS index 2 is "Vector"; drop it for deferred-embedding inserts.
        return values[:2] + values[3:]

    def _embedding_vector_sql(self, content_sql: str, task_type_sql: str) -> str:
        """Return SQL expression that generates a FLOAT32 embedding via ML.PREDICT."""
        return f"""
            (
                SELECT ARRAY(
                    SELECT CAST(value AS FLOAT32)
                    FROM UNNEST(embeddings.values) AS value
                )
                FROM ML.PREDICT(
                    MODEL {settings.spanner_embedding_model_id},
                    (SELECT {content_sql} AS content, {task_type_sql} AS task_type),
                    STRUCT({get_current_vector_dim()} AS outputDimensionality)
                )
            )
        """

    def _batched_insert_with_spanner_embedding_sql(self) -> str:
        """DML that upserts a whole batch of chunks, embedding them in ONE ML.PREDICT.

        The batch is passed as the @rows ARRAY<STRUCT> parameter and UNNESTed into the
        model input. ML.PREDICT passes every input column through to its output relation
        (pred.*), so the outer SELECT reads the passthrough columns and the computed
        embedding. The @{remote_udf_max_rows_per_rpc=N} hint makes Spanner fan the batch
        into N-row Vertex RPCs instead of one-call-per-row.
        """
        model_id = settings.spanner_embedding_model_id
        dim = get_current_vector_dim()
        rpc = settings.spanner_embedding_rpc_batch_size
        return f"""
            INSERT OR UPDATE INTO {SPANNER_TABLE_NAME} (
                Id, Content, Vector, ChunkType, SessionId, ProjectPath, ProjectName,
                Timestamp, UserUuid, AssistantUuid, FilePath, Operation, Model,
                SourceFile, SourceLine, ParentChunkId, ChildChunkIds, MachineId
            )
            SELECT
                pred.id, pred.content,
                ARRAY(SELECT CAST(value AS FLOAT32) FROM UNNEST(pred.embeddings.values) AS value),
                pred.chunk_type, pred.session_id, pred.project_path, pred.project_name,
                pred.timestamp, pred.user_uuid, pred.assistant_uuid, pred.file_path,
                pred.operation, pred.model, pred.source_file, pred.source_line,
                pred.parent_chunk_id, pred.child_chunk_ids, pred.machine_id
            FROM ML.PREDICT(
                MODEL {model_id},
                (
                    SELECT
                        id, content, chunk_type, session_id, project_path, project_name,
                        timestamp, user_uuid, assistant_uuid, file_path, operation, model,
                        source_file, source_line, parent_chunk_id, child_chunk_ids, machine_id,
                        @task_type AS task_type
                    FROM UNNEST(@rows)
                ),
                STRUCT({dim} AS outputDimensionality)
            ) @{{remote_udf_max_rows_per_rpc={rpc}}} AS pred
        """

    def _add_chunks_with_spanner_embeddings(self, chunks: list[dict[str, Any]]) -> None:
        """Add unembedded chunks, generating all vectors in ONE batched ML.PREDICT.

        Replaces the legacy one-DML-per-chunk loop (which made one serial Vertex RPC per
        chunk) with a single INSERT ... SELECT over the whole batch, fanned into N-row RPCs
        by the remote_udf_max_rows_per_rpc hint.
        """
        try:
            from google.cloud.spanner_v1 import param_types
        except ImportError as e:
            raise RuntimeError("google-cloud-spanner is required") from e

        self.ensure_embedding_model()
        database = self.connect()
        sql = self._batched_insert_with_spanner_embedding_sql()
        params = {
            "rows": [self._chunk_struct_value(chunk) for chunk in chunks],
            "task_type": settings.vertex_document_task_type,
        }
        spanner_param_types = {
            "rows": self._chunk_struct_param_type(),
            "task_type": param_types.STRING,
        }

        def insert_batch(transaction):
            return transaction.execute_update(
                sql,
                params=params,
                param_types=spanner_param_types,
            )

        row_count = database.run_in_transaction(insert_batch)
        logger.debug(
            "Added %s chunks to Spanner store via batched Spanner-native embeddings (rpc_batch=%s)",
            row_count,
            settings.spanner_embedding_rpc_batch_size,
        )

    def add_chunks_without_embeddings(self, chunks: list[dict[str, Any]]) -> None:
        """Insert chunks with the Vector column left NULL (deferred-embedding mode).

        Ingest is decoupled from embedding: rows land immediately via the mutation API and
        are filled in later by backfill_embeddings(). Search already filters Vector IS NOT
        NULL, so un-embedded rows are simply invisible until backfilled.
        """
        if not chunks:
            return
        database = self.connect()
        columns = [column for column in SPANNER_COLUMNS if column != "Vector"]
        values = [self._chunk_values_no_vector(chunk) for chunk in chunks]
        with database.batch() as batch:
            batch.insert_or_update(table=SPANNER_TABLE_NAME, columns=columns, values=values)
        logger.info(
            "Inserted %s chunks to Spanner store without embeddings (deferred backfill)",
            len(chunks),
        )

    # Spanner columns needed to re-embed a row (everything except the recomputed Vector),
    # in the order their values map to _BACKFILL_DICT_KEYS below.
    _BACKFILL_READ_COLUMNS = [
        "Id",
        "Content",
        "ChunkType",
        "SessionId",
        "ProjectPath",
        "ProjectName",
        "Timestamp",
        "UserUuid",
        "AssistantUuid",
        "FilePath",
        "Operation",
        "Model",
        "SourceFile",
        "SourceLine",
        "ParentChunkId",
        "ChildChunkIds",
        "MachineId",
    ]
    _BACKFILL_DICT_KEYS = [
        "id",
        "content",
        "chunk_type",
        "session_id",
        "project_path",
        "project_name",
        "timestamp",
        "user_uuid",
        "assistant_uuid",
        "file_path",
        "operation",
        "model",
        "source_file",
        "source_line",
        "parent_chunk_id",
        "child_chunk_ids",
        "machine_id",
    ]

    def _row_to_chunk_dict(self, row: Any) -> dict[str, Any]:
        """Map a positional Spanner backfill-read row to the app chunk-dict shape."""
        return {key: row[index] for index, key in enumerate(self._BACKFILL_DICT_KEYS)}

    def _read_unembedded_batch(self, prefix: str, limit: int) -> list[dict[str, Any]]:
        """Read up to `limit` un-embedded rows whose Id starts with `prefix`."""
        try:
            from google.cloud.spanner_v1 import param_types
        except ImportError as e:
            raise RuntimeError("google-cloud-spanner is required") from e
        database = self.connect()
        columns = ", ".join(self._BACKFILL_READ_COLUMNS)
        sql = (
            f"SELECT {columns} FROM {SPANNER_TABLE_NAME} "
            "WHERE Vector IS NULL AND STARTS_WITH(Id, @prefix) LIMIT @limit"
        )
        with database.snapshot() as snapshot:
            rows = list(
                snapshot.execute_sql(
                    sql,
                    params={"prefix": prefix, "limit": limit},
                    param_types={"prefix": param_types.STRING, "limit": param_types.INT64},
                )
            )
        return [self._row_to_chunk_dict(row) for row in rows]

    def _backfill_shard(self, prefix: str) -> int:
        """Drain one Id-prefix shard: read NULL-vector rows in batches and re-embed them.

        Resilient by design: a failed batch (e.g. Vertex quota/throttle, or a transient
        Spanner Aborted) stops THIS shard but never propagates — the affected rows stay NULL
        and are retried on the next pass, and the other shards keep going. A single batch
        failure must not abort the whole backfill pass (the bug the PDML version avoided via
        SAFE.ML.PREDICT + auto-retry).
        """
        batch_size = settings.spanner_backfill_batch_size
        embedded = 0
        while True:
            try:
                rows = self._read_unembedded_batch(prefix, batch_size)
                if not rows:
                    break
                self._add_chunks_with_spanner_embeddings(rows)
                embedded += len(rows)
            except Exception as exc:
                logger.warning(
                    "Backfill shard %s stopped after %s rows: %s; remaining rows stay NULL "
                    "for the next pass",
                    prefix,
                    embedded,
                    exc,
                )
                break
        return embedded

    def backfill_embeddings(self) -> int:
        """Fill NULL vectors with app-controlled concurrency (not PDML split-bound).

        NULL-vector rows are sharded by Id hex-prefix (Id is a sha256 hex digest) into 256
        disjoint slices, drained by a pool of `spanner_backfill_concurrency` workers. Each
        worker re-embeds its rows through the batched INSERT ... SELECT FROM ML.PREDICT path,
        so aggregate throughput is bounded by the Vertex quota rather than the table's Spanner
        split count (which caps partitioned DML on a freshly loaded table). Idempotent and
        re-runnable (WHERE Vector IS NULL). Returns the rows embedded; 0 when none need it.
        """
        self.ensure_embedding_model()
        self.connect()
        prefixes = [f"{value:02x}" for value in range(256)]
        concurrency = settings.spanner_backfill_concurrency
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            counts = list(executor.map(self._backfill_shard, prefixes))
        total = sum(counts)
        if total:
            logger.info(
                "Backfilled embeddings for %s rows via %s sharded workers", total, concurrency
            )
        return total

    async def backfill_embeddings_async(self) -> int:
        """Run backfill_embeddings() off the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.backfill_embeddings)

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Add embedded chunks to Spanner."""
        if not chunks:
            return
        if settings.spanner_embedding_mode == "spanner":
            unembedded_chunks = [chunk for chunk in chunks if not chunk.get("vector")]
            if len(unembedded_chunks) == len(chunks):
                if settings.spanner_defer_embeddings:
                    # Fast path: land rows now (Vector NULL), embed later via partitioned DML.
                    self.add_chunks_without_embeddings(chunks)
                else:
                    self._add_chunks_with_spanner_embeddings(chunks)
                return
            if unembedded_chunks:
                raise ValueError("Cannot mix embedded and unembedded chunks in one Spanner batch")
        expected_dim = get_current_vector_dim()
        for chunk in chunks:
            vector = chunk.get("vector")
            chunk_id = chunk.get("id", "unknown")
            if not isinstance(vector, list):
                raise ValueError(f"Vector must be a list, got {type(vector).__name__}")
            if len(vector) != expected_dim:
                raise ValueError(
                    f"Vector dimension mismatch: expected {expected_dim}, got {len(vector)}"
                )
            if not all(isinstance(v, (int, float)) for v in vector):
                raise ValueError("Vector must contain only numeric values")
            _validate_vector(vector, chunk_id)

        database = self.connect()
        values = [self._chunk_values(chunk) for chunk in chunks]
        with database.batch() as batch:
            batch.insert_or_update(table=SPANNER_TABLE_NAME, columns=SPANNER_COLUMNS, values=values)
        logger.info("Added %s chunks to Spanner store", len(chunks))

    async def add_chunks_async(self, chunks: list[dict[str, Any]]) -> None:
        """Add chunks asynchronously."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.add_chunks, chunks)

    def _vector_distance_sql(self, filters_use_unstored_columns: bool = False) -> str:
        """Return the best available Spanner vector distance expression."""
        if self._can_use_ann(filters_use_unstored_columns):
            return (
                "APPROX_COSINE_DISTANCE(Vector, @query_vector, "
                f'options => JSON \'{{"num_leaves_to_search": '
                f"{settings.spanner_num_leaves_to_search}}}')"
            )
        return "COSINE_DISTANCE(Vector, @query_vector)"

    def _can_use_ann(self, filters_use_unstored_columns: bool = False) -> bool:
        """Return whether current settings/query shape can use Spanner ANN."""
        return (
            settings.spanner_use_approx_vector_search
            and not filters_use_unstored_columns
            and self._vector_index_exists()
        )

    def _vector_table_source_sql(self, filters_use_unstored_columns: bool = False) -> str:
        """Return table source with vector index hint when ANN is selected."""
        if self._can_use_ann(filters_use_unstored_columns):
            return f"{SPANNER_TABLE_NAME}@{{FORCE_INDEX={SPANNER_VECTOR_INDEX}}}"
        return SPANNER_TABLE_NAME

    def embed_query_text(self, query: str) -> list[float]:
        """Generate a query embedding inside Spanner via ML.PREDICT."""
        if not query.strip():
            raise ValueError("Query cannot be empty")
        self.ensure_embedding_model()
        database = self.connect()
        sql = f"""
            SELECT {self._embedding_vector_sql("@query", "@task_type")} AS Vector
        """
        try:
            from google.cloud.spanner_v1 import param_types
        except ImportError as e:
            raise RuntimeError("google-cloud-spanner is required") from e
        with database.snapshot() as snapshot:
            rows = list(
                snapshot.execute_sql(
                    sql,
                    params={"query": query, "task_type": settings.vertex_query_task_type},
                    param_types={
                        "query": param_types.STRING,
                        "task_type": param_types.STRING,
                    },
                )
            )
        if not rows:
            raise RuntimeError("Spanner embedding query returned no rows")
        vector = [float(v) for v in rows[0][0]]
        _validate_vector(vector, "query")
        return vector

    async def embed_query_text_async(self, query: str) -> list[float]:
        """Generate a query embedding inside Spanner asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_query_text, query)

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        project_filter: str | None = None,
        chunk_type_filter: str | None = None,
        file_path_filter: str | None = None,
        operation_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """KNN vector search via Spanner exact or indexed ANN cosine distance."""
        if len(query_vector) != get_current_vector_dim():
            raise ValueError(
                f"Query vector dimension mismatch: expected {get_current_vector_dim()}, "
                f"got {len(query_vector)}"
            )
        _validate_vector(query_vector, "query")
        database = self.connect()
        filters = ["Vector IS NOT NULL"]
        params: dict[str, Any] = {
            "query_vector": [float(v) for v in query_vector],
            "limit": limit,
        }
        try:
            from google.cloud.spanner_v1 import param_types
        except ImportError as e:
            raise RuntimeError("google-cloud-spanner is required") from e
        types: dict[str, Any] = {
            "query_vector": param_types.Array(param_types.FLOAT32),
            "limit": param_types.INT64,
        }
        if project_filter:
            filters.append("ProjectPath = @project_filter")
            params["project_filter"] = project_filter
            types["project_filter"] = param_types.STRING
        if chunk_type_filter:
            filters.append("ChunkType = @chunk_type_filter")
            params["chunk_type_filter"] = chunk_type_filter
            types["chunk_type_filter"] = param_types.STRING
        if file_path_filter:
            filters.append("FilePath LIKE @file_path_filter")
            params["file_path_filter"] = f"%{file_path_filter}%"
            types["file_path_filter"] = param_types.STRING
        if operation_filter:
            filters.append("Operation = @operation_filter")
            params["operation_filter"] = operation_filter
            types["operation_filter"] = param_types.STRING

        filters_use_unstored_columns = bool(project_filter or file_path_filter or operation_filter)
        distance_expr = self._vector_distance_sql(
            filters_use_unstored_columns=filters_use_unstored_columns
        )
        table_source = self._vector_table_source_sql(
            filters_use_unstored_columns=filters_use_unstored_columns
        )
        sql = f"""
            SELECT
                Id, Content, ChunkType, SessionId, ProjectPath, ProjectName, Timestamp,
                FilePath, Operation, MachineId, {distance_expr} AS Distance
            FROM {table_source}
            WHERE {" AND ".join(filters)}
            ORDER BY Distance ASC
            LIMIT @limit
        """
        with database.snapshot() as snapshot:
            rows = list(snapshot.execute_sql(sql, params=params, param_types=types))
        return [self._row_to_result(row) for row in rows]

    def vector_search(
        self,
        query_vector: list[float],
        limit: int = 5,
        project_filter: str | None = None,
        chunk_type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Protocol alias for vector search."""
        return self.search(
            query_vector=query_vector,
            limit=limit,
            project_filter=project_filter,
            chunk_type_filter=chunk_type_filter,
        )

    async def search_async(
        self,
        query_vector: list[float],
        limit: int = 10,
        project_filter: str | None = None,
        chunk_type_filter: str | None = None,
        file_path_filter: str | None = None,
        operation_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(
                query_vector,
                limit,
                project_filter,
                chunk_type_filter,
                file_path_filter,
                operation_filter,
            ),
        )

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        limit: int = 10,
        project_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search using Spanner full-text SCORE plus vector RRF fusion."""
        if not query.strip() or not self.has_fts_index():
            results = self.search(
                query_vector=query_vector, limit=limit, project_filter=project_filter
            )
            for result in results:
                result["_search_type"] = "vector"
            return results
        if len(query_vector) != get_current_vector_dim():
            raise ValueError(
                f"Query vector dimension mismatch: expected {get_current_vector_dim()}, "
                f"got {len(query_vector)}"
            )
        _validate_vector(query_vector, "query")
        database = self.connect()
        vector_filters = ["Vector IS NOT NULL"]
        text_filters = [f"SEARCH({SPANNER_CONTENT_TOKENS_COLUMN}, @query)"]
        params: dict[str, Any] = {
            "query": query,
            "query_vector": [float(v) for v in query_vector],
            "limit": limit,
            "candidate_limit": max(limit, settings.spanner_hybrid_candidate_limit),
            "rrf_k": float(settings.spanner_rrf_k),
        }
        try:
            from google.cloud.spanner_v1 import param_types
        except ImportError as e:
            raise RuntimeError("google-cloud-spanner is required") from e
        types: dict[str, Any] = {
            "query": param_types.STRING,
            "query_vector": param_types.Array(param_types.FLOAT32),
            "limit": param_types.INT64,
            "candidate_limit": param_types.INT64,
            "rrf_k": param_types.FLOAT64,
        }
        if project_filter:
            vector_filters.append("ProjectPath = @project_filter")
            text_filters.append("ProjectPath = @project_filter")
            params["project_filter"] = project_filter
            types["project_filter"] = param_types.STRING

        filters_use_unstored_columns = bool(project_filter)
        distance_expr = self._vector_distance_sql(
            filters_use_unstored_columns=filters_use_unstored_columns
        )
        vector_table_source = self._vector_table_source_sql(
            filters_use_unstored_columns=filters_use_unstored_columns
        )
        max_rrf_score_expr = "2.0 / (@rrf_k + 1)"
        sql = f"""
            WITH VectorCandidates AS (
                SELECT rank, chunk_id AS Id
                FROM UNNEST(ARRAY(
                    SELECT Id
                    FROM {vector_table_source}
                    WHERE {" AND ".join(vector_filters)}
                    ORDER BY {distance_expr} ASC
                    LIMIT @candidate_limit
                )) AS chunk_id WITH OFFSET AS rank
            ),
            TextCandidates AS (
                SELECT rank, chunk_id AS Id
                FROM UNNEST(ARRAY(
                    SELECT Id
                    FROM {SPANNER_TABLE_NAME}
                    WHERE {" AND ".join(text_filters)}
                    ORDER BY SCORE({SPANNER_CONTENT_TOKENS_COLUMN}, @query) DESC
                    LIMIT @candidate_limit
                )) AS chunk_id WITH OFFSET AS rank
            ),
            FusedCandidates AS (
                SELECT Id, SUM(1.0 / (@rrf_k + rank + 1)) AS Score
                FROM (
                    SELECT Id, rank FROM VectorCandidates
                    UNION ALL
                    SELECT Id, rank FROM TextCandidates
                )
                GROUP BY Id
            )
            SELECT
                c.Id, c.Content, c.ChunkType, c.SessionId, c.ProjectPath,
                c.ProjectName, c.Timestamp, c.FilePath, c.Operation, c.MachineId,
                1.0 - LEAST(1.0, f.Score / ({max_rrf_score_expr})) AS Distance
            FROM FusedCandidates f
            JOIN {SPANNER_TABLE_NAME} c ON c.Id = f.Id
            ORDER BY f.Score DESC
            LIMIT @limit
        """
        try:
            with database.snapshot() as snapshot:
                rows = list(snapshot.execute_sql(sql, params=params, param_types=types))
        except Exception as e:
            logger.warning(
                "Spanner hybrid search failed; falling back to vector search: "
                "error_type=%s limit=%s project_filter_present=%s candidate_limit=%s vector_mode=%s",
                type(e).__name__,
                limit,
                bool(project_filter),
                settings.spanner_hybrid_candidate_limit,
                "ann" if self._can_use_ann(False) else "exact",
            )
            fallback_results = self.search(
                query_vector=query_vector, limit=limit, project_filter=project_filter
            )
            for result in fallback_results:
                result["_search_type"] = "vector"
            return fallback_results
        return [self._row_to_result(row) for row in rows]

    def has_fts_index(self) -> bool:
        """Return whether Spanner full-text search is configured and indexed."""
        if not settings.spanner_enable_full_text:
            return False
        database = self.connect()
        del database
        return self._column_exists(
            SPANNER_TABLE_NAME, SPANNER_CONTENT_TOKENS_COLUMN
        ) and self._index_exists(SPANNER_CONTENT_SEARCH_INDEX)

    async def hybrid_search_async(
        self,
        query: str,
        query_vector: list[float],
        limit: int = 10,
        project_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.hybrid_search(query, query_vector, limit, project_filter)
        )

    def _row_to_result(self, row: Any) -> dict[str, Any]:
        """Convert a Spanner result row to the API shape expected by MCP tools."""
        (
            chunk_id,
            content,
            chunk_type,
            session_id,
            project_path,
            project_name,
            timestamp,
            file_path,
            operation,
            machine_id,
            distance,
        ) = row
        return {
            "id": chunk_id,
            "content": content,
            "chunk_type": chunk_type,
            "session_id": session_id,
            "project_path": project_path,
            "project_name": project_name,
            "timestamp": _format_timestamp(timestamp),
            "file_path": file_path,
            "operation": operation,
            "machine_id": machine_id,
            "score": distance,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get Spanner store statistics."""
        total, embedded = self._embedding_counts()
        return {
            "total_chunks": total,
            "embedded_chunks": embedded,
            "awaiting_embedding": max(total - embedded, 0),
            "backend": "spanner",
            "project": self.project,
            "instance": self.instance_id,
            "database": self.database_id,
            "dimension": get_current_vector_dim(),
            "fts_index_available": self.has_fts_index(),
            "vector_index_available": self._vector_index_exists(),
            "vector_search_mode": "ann"
            if settings.spanner_use_approx_vector_search and self._vector_index_exists()
            else "exact",
            "embedding_mode": settings.spanner_embedding_mode,
            "embedding_model_id": settings.spanner_embedding_model_id
            if settings.spanner_embedding_mode == "spanner"
            else None,
        }

    async def get_stats_async(self) -> dict[str, Any]:
        """Get store statistics asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_stats)

    def clear_all(self) -> int:
        """Delete all chunks."""
        count = self.get_stats()["total_chunks"]
        database = self.connect()
        database.execute_partitioned_dml(f"DELETE FROM {SPANNER_TABLE_NAME} WHERE TRUE")
        return int(count)

    async def clear_all_async(self) -> int:
        """Clear chunks asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.clear_all)

    def delete_by_machine_id(self, machine_id: str) -> int:
        """Delete all chunks for a specific machine_id."""
        if not machine_id:
            return 0
        database = self.connect()
        try:
            from google.cloud.spanner_v1 import param_types
        except ImportError as e:
            raise RuntimeError("google-cloud-spanner is required") from e
        with database.snapshot() as snapshot:
            rows = list(
                snapshot.execute_sql(
                    f"SELECT COUNT(*) FROM {SPANNER_TABLE_NAME} WHERE MachineId = @machine_id",
                    params={"machine_id": machine_id},
                    param_types={"machine_id": param_types.STRING},
                )
            )
        deleted = int(rows[0][0]) if rows else 0
        database.execute_partitioned_dml(
            f"DELETE FROM {SPANNER_TABLE_NAME} WHERE MachineId = @machine_id",
            params={"machine_id": machine_id},
            param_types={"machine_id": param_types.STRING},
        )
        return deleted

    async def delete_by_machine_id_async(self, machine_id: str) -> int:
        """Delete all chunks for a specific machine_id asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.delete_by_machine_id, machine_id)

    def optimize(self) -> None:
        """Create deferred Spanner ANN index once enough rows have been ingested."""
        self.ensure_search_schema()
        created = self.create_vector_index(force=False)
        if created:
            logger.info("Spanner vector index is available")
        else:
            logger.info("Spanner optimize completed without vector index creation")

    async def optimize_async(self) -> None:
        """Optimize asynchronously."""
        self.optimize()

    def chunk_exists(self, chunk_id: str) -> bool:
        """Check whether a chunk exists."""
        database = self.connect()
        try:
            from google.cloud.spanner_v1 import param_types
        except ImportError as e:
            raise RuntimeError("google-cloud-spanner is required") from e
        with database.snapshot() as snapshot:
            rows = list(
                snapshot.execute_sql(
                    f"SELECT Id FROM {SPANNER_TABLE_NAME} WHERE Id = @chunk_id LIMIT 1",
                    params={"chunk_id": chunk_id},
                    param_types={"chunk_id": param_types.STRING},
                )
            )
        return bool(rows)

    def close(self) -> None:
        """Release cached Spanner handles."""
        self._database = None
        self._instance = None
        self._client = None

    async def close_async(self) -> None:
        """Close asynchronously."""
        self.close()


def create_store() -> ConversationStore:
    """Create the configured conversation storage backend."""
    if settings.storage_backend == "lancedb":
        return VectorStore()
    if settings.storage_backend == "spanner":
        return SpannerStore()
    raise ValueError(f"Unsupported storage backend: {settings.storage_backend}")


# Global store instance
store = create_store()
