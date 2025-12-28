"""Qdrant storage backend implementation."""

import logging
import asyncio
from typing import Any
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from claude_history_rag.config import settings
from claude_history_rag.store import get_current_vector_dim
from claude_history_rag.storage.protocol import StorageBackend

logger = logging.getLogger(__name__)


class QdrantBackend(StorageBackend):
    """Qdrant storage backend."""

    def __init__(self) -> None:
        """Initialize Qdrant backend."""
        self.client: AsyncQdrantClient | None = None
        self.collection_name = settings.qdrant_collection
        # Lazy init of vector dim
        self._vector_dim: int | None = None
        self._init_lock = asyncio.Lock()

    @property
    def vector_dim(self) -> int:
        """Get vector dimension."""
        if self._vector_dim is None:
            self._vector_dim = get_current_vector_dim()
        return self._vector_dim

    async def initialize(self) -> None:
        """Initialize the backend connection and schema."""
        if self.client:
            return

        async with self._init_lock:
            if self.client:
                return

            if not settings.qdrant_url:
                raise ValueError("Qdrant URL is not configured")

            logger.info(f"Connecting to Qdrant at {settings.qdrant_url}")
            self.client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            )

        try:
            # Check if collection exists
            exists = await self.client.collection_exists(self.collection_name)
            if not exists:
                logger.info(f"Creating Qdrant collection '{self.collection_name}'")
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dim,
                        distance=models.Distance.COSINE,
                    ),
                )
                
                # Create filter indexes
                await self._create_indexes()
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise

    async def _create_indexes(self) -> None:
        """Create payload indexes for filtering."""
        if not self.client:
            return
            
        fields = ["project_path", "chunk_type", "file_path", "machine_id", "session_id"]
        for field in fields:
            try:
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                logger.warning(f"Failed to create index for {field}: {e}")

    async def close(self) -> None:
        """Close connections and release resources."""
        if self.client:
            await self.client.close()
            self.client = None

    async def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Add conversation chunks to storage."""
        if not self.client:
            await self.initialize()
        
        points = []
        for chunk in chunks:
            chunk_id = chunk.get("id")
            if not chunk_id:
                # Need valid UUID for Qdrant if not provided? 
                # Qdrant supports UUID or integer IDs. Our IDs are hex strings (SHA256 usually).
                # We can store our ID in payload and generate a UUID for Qdrant, or use our ID if it's UUID-compatible.
                # Actually, our IDs are likely not UUIDs if they are SHA256 hashes.
                # Qdrant IDs can be UUIDs or unsigned integers.
                # Solution: Generate a deterministic UUID from the chunk ID string.
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "unknown"))
            
            # Validate vector presence
            vector = chunk.get("vector")
            if not vector:
                continue

            # Qdrant ID must be UUID or int. 
            # Our chunk['id'] is a string (hex). 
            # We will generate a UUID from it for the Point ID, and store the original ID in payload.
            original_id = chunk.get("id", "unknown")
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, original_id))

            # Prepare payload (exclude vector)
            payload = chunk.copy()
            del payload["vector"]
            # Ensure timestamp is serialized
            if "timestamp" in payload and not isinstance(payload["timestamp"], str):
                 if hasattr(payload["timestamp"], "isoformat"):
                    payload["timestamp"] = payload["timestamp"].isoformat()

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        if points:
            try:
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                logger.info(f"Added {len(points)} chunks to Qdrant")
            except Exception as e:
                logger.error(f"Failed to add chunks to Qdrant: {e}")
                raise

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks."""
        if not self.client:
            await self.initialize()

        # Build filter conditions
        must_conditions = []
        if filters:
            for key, value in filters.items():
                if key == "file_path" and "%" in value:
                     # Handle LIKE logic manually or roughly? 
                     # Qdrant supports 'MatchText' for full text but 'MatchValue' for keywords.
                     # For basic support, we might fallback or try regex if supported?
                     # Qdrant doesn't have standard SQL LIKE.
                     # For now, if exact match isn't possible, we might log warning or just attempt simple match.
                     # "file_path LIKE ..."
                     pass 
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )

        q_filter = models.Filter(must=must_conditions) if must_conditions else None

        try:
            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=q_filter,
                limit=limit,
            )
        except UnexpectedResponse as e:
            # Handle 404 if collection deleted etc
            logger.warning(f"Qdrant search error: {e}")
            return []

        # Convert back to our dict format
        output = []
        for hit in results:
            payload = hit.payload or {}
            item = payload.copy()
            item["score"] = hit.score
            output.append(item)
            
        return output

    async def delete(self, filters: dict[str, Any]) -> int:
        """Delete chunks matching the filters."""
        if not self.client:
            await self.initialize()
            
        must_conditions = []
        for key, value in filters.items():
            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )
            
        if not must_conditions:
            return 0
            
        q_filter = models.Filter(must=must_conditions)
        
        try:
            # Delete points by filter
            # Qdrant delete_points returns UpdateResult, not count.
            # We can't easily get the count of deleted items without searching first.
            # For now return 0 or approximate.
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=q_filter),
            )
            # Todo: find a way to return count
            return 1 
        except Exception as e:
            logger.error(f"Failed to delete from Qdrant: {e}")
            return 0

    async def optimize(self) -> None:
        """Perform optimization."""
        # Qdrant auto-optimizes, but we can force it if needed?
        # Typically not required.
        pass

    async def get_stats(self) -> dict[str, Any]:
        """Get stats."""
        if not self.client:
            try:
                await self.initialize()
            except Exception:
                return {"backend": "qdrant", "status": "disconnected"}
                
        try:
            info = await self.client.get_collection(self.collection_name)
            return {
                "backend": "qdrant",
                "total_chunks": info.points_count,
                "status": "connected",
                "collection": self.collection_name,
            }
        except Exception as e:
            return {"backend": "qdrant", "error": str(e)}

    async def clear_all(self) -> int:
        """Clear all chunks."""
        if not self.client:
            await self.initialize()
            
        try:
            # Recreate collection is fastest way to clear
            await self.client.delete_collection(self.collection_name)
            await self.initialize()
            return 0 # Sentinal
        except Exception as e:
            logger.error(f"Failed to clear Qdrant: {e}")
            return 0
