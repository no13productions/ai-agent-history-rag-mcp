"""LanceDB vector store operations."""

import asyncio
import logging
import math
import threading
from datetime import datetime as dt
from pathlib import Path
from typing import Any

import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import RRFReranker

from claude_history_rag.config import settings

logger = logging.getLogger(__name__)


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
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "bge-m3": 1024,
    "snowflake-arctic-embed": 1024,  # Default size (also has 768, 384, 256 variants)
    "all-minilm": 384,
}


def get_vector_dim() -> int:
    """Get vector dimension based on configured embedding model."""
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
VECTOR_DIM = 1024  # Default to largest common dimension for schema

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


class ConversationChunkModel(LanceModel):
    """LanceDB schema for conversation chunks."""

    id: str
    content: str
    vector: Vector(VECTOR_DIM)  # type: ignore
    chunk_type: str
    session_id: str
    project_path: str
    project_name: str
    timestamp: dt
    user_uuid: str | None = None
    assistant_uuid: str | None = None
    file_path: str | None = None
    operation: str | None = None
    model: str | None = None
    source_file: str
    source_line: int
    parent_chunk_id: str | None = None
    child_chunk_ids: list[str] | None = None
    machine_id: str | None = None  # For multi-machine support


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
                    schema=ConversationChunkModel,
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
            # Convert dicts to Pydantic models to ensure proper schema handling
            models = [ConversationChunkModel(**chunk) for chunk in chunks]
            table.add(models)
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
                "score": r.get("_distance", 0),
            }
            for r in results
            if isinstance(r, dict) and "id" in r and "content" in r
        ]

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
                    search_query = table.search(query_type="hybrid").vector(query_vector).text(query)
                    if project_filter:
                        safe_project = _sanitize_filter_value(project_filter)
                        if safe_project:
                            search_query = search_query.where(
                                f"project_path = '{safe_project}'"
                            )
                    results = (
                        search_query.rerank(reranker=RRFReranker()).limit(limit).to_list()
                    )
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
                "score": r.get("_relevance_score", 0),
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
            error_message = str(e)
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
                error_message = str(retry_error)
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

        # Compact the table to reclaim space
        try:
            table.compact_files()
            logger.info("Compacted table files")
        except Exception as e:
            logger.warning(f"Failed to compact files: {type(e).__name__}")

        logger.info("Store optimization complete")

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


# Global store instance
store = VectorStore()
