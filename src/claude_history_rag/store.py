"""Vector store options (SQLite/Qdrant) abstraction."""

import asyncio
import logging
import math
import threading
from typing import Any

from claude_history_rag.config import settings
from claude_history_rag.storage.protocol import StorageBackend

logger = logging.getLogger(__name__)


# Model name to vector dimension mapping
# https://ollama.com/search?c=embedding
MODEL_DIMENSIONS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "bge-m3": 1024,
    "snowflake-arctic-embed": 1024,
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
    return dim


# Vector dimension - computed lazily to ensure settings are loaded
_vector_dim: int | None = None


def get_current_vector_dim() -> int:
    """Get the current vector dimension (lazy initialization)."""
    global _vector_dim
    if _vector_dim is None:
        _vector_dim = get_vector_dim()
    return _vector_dim


class VectorStore:
    """Vector store abstraction delegating to configured backend."""

    def __init__(self) -> None:
        """Initialize appropriate backend based on settings."""
        self.backend: StorageBackend | None = None
        self._lock = threading.Lock()
        
    def _get_backend(self) -> StorageBackend:
        """Lazy load backend."""
        if self.backend:
            return self.backend
            
        with self._lock:
            if self.backend:
                return self.backend
                
            backend_type = settings.storage_backend.lower()
            logger.info(f"Initializing storage backend: {backend_type}")
            
            if backend_type == "qdrant":
                from claude_history_rag.storage.qdrant import QdrantBackend
                self.backend = QdrantBackend()
            elif backend_type == "sqlite":
                from claude_history_rag.storage.sqlite import SQLiteBackend
                self.backend = SQLiteBackend()
            else:
                logger.warning(f"Unknown backend '{backend_type}', defaulting to sqlite")
                from claude_history_rag.storage.sqlite import SQLiteBackend
                self.backend = SQLiteBackend()
                
            return self.backend

    # --------------------------------------------------------------------------
    # Async API (Primary)
    # --------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize backend connection."""
        await self._get_backend().initialize()

    async def close_async(self) -> None:
        """Close backend connection."""
        if self.backend:
            await self.backend.close()

    async def add_chunks_async(self, chunks: list[dict[str, Any]]) -> None:
        """Add chunks asynchronously."""
        if not chunks:
            return
        await self._get_backend().add_chunks(chunks)

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
        filters = {}
        if project_filter:
            filters["project_path"] = project_filter
        if chunk_type_filter:
            filters["chunk_type"] = chunk_type_filter
        if file_path_filter:
            filters["file_path"] = file_path_filter
        if operation_filter:
            filters["operation"] = operation_filter
            
        return await self._get_backend().search(query_vector, limit, filters)

    async def hybrid_search_async(
        self,
        query: str,
        query_vector: list[float],
        limit: int = 10,
        project_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search asynchronously.
        
        Note: Currently maps to vector search only until full hybrid support 
        is implemented in protocols/backends.
        """
        # TODO: Implement true hybrid search in backends using FTS + Vector
        # For now, fallback to vector search
        return await self.search_async(
            query_vector, 
            limit, 
            project_filter=project_filter
        )

    async def optimize_async(self) -> None:
        """Optimize database asynchronously."""
        await self._get_backend().optimize()

    async def get_stats_async(self) -> dict[str, Any]:
        """Get stats asynchronously."""
        return await self._get_backend().get_stats()

    async def clear_all_async(self) -> int:
        """Clear all data asynchronously."""
        return await self._get_backend().clear_all()

    async def delete_by_machine_id_async(self, machine_id: str) -> int:
        """Delete by machine ID asynchronously."""
        return await self._get_backend().delete({"machine_id": machine_id})




# Global store instance
store = VectorStore()
