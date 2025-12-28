"""Protocol definition for storage backends."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol defining the contract for storage backends."""

    async def initialize(self) -> None:
        """Initialize the backend connection and schema."""
        ...

    async def close(self) -> None:
        """Close connections and release resources."""
        ...

    async def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Add conversation chunks to storage.
        
        Args:
            chunks: List of dictionaries containing chunk data and 'vector' field.
        """
        ...

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks using vector similarity.

        Args:
            query_vector: The query embedding vector.
            limit: Maximum number of results.
            filters: Optional dictionary of exact match filters (e.g., {"project_path": "..."}).
        
        Returns:
            List of chunks compatible with the application schema.
        """
        ...

    async def delete(self, filters: dict[str, Any]) -> int:
        """Delete chunks matching the filters.
        
        Args:
            filters: Dictionary of filters to identify chunks to delete (e.g. {"machine_id": "..."}).
            
        Returns:
            Number of chunks deleted.
        """
        ...

    async def optimize(self) -> None:
        """Perform backend-specific optimization (compression, vacuum, indexing)."""
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary containing stats like "total_chunks", "backend_type", etc.
        """
        ...
        
    async def clear_all(self) -> int:
        """Clear all data from the storage.
        
        Returns:
            Number of items cleared.
        """
        ...
