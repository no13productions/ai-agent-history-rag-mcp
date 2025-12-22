"""Search result caching with LRU eviction and TTL support.

Provides fast in-memory caching for search results to reduce redundant
computation and improve response times for repeated queries.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import threading
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from typing import Any

from claude_history_rag.decision_engine.models import CacheEntry

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache implementation.

    Uses OrderedDict for O(1) access and LRU eviction. Thread safety is
    provided through asyncio.Lock for async contexts.
    """

    def __init__(self, maxsize: int = 100):
        """Initialize LRU cache.

        Args:
            maxsize: Maximum number of entries to store
        """
        if maxsize <= 0:
            logger.error(
                f"Invalid maxsize for LRUCache: {maxsize} "
                f"(type={type(maxsize).__name__}). Expected: positive integer."
            )
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> CacheEntry | None:
        """Get entry from cache.

        Moves accessed entry to end (most recently used) if found.
        Returns a copy to prevent caller modifications from corrupting
        the cache.

        Args:
            key: Cache key to look up

        Returns:
            Copy of CacheEntry if found and not expired, None otherwise
        """
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if datetime.now(UTC) > entry.expires_at:
                del self._cache[key]
                self._misses += 1
                query_text = entry.query[:50] if len(entry.query) > 50 else entry.query
                logger.debug(f"Cache entry expired: {key[:16]}... (query: {query_text})")
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1

            # BUG FIX: Return a copy to maintain cache isolation.
            # This prevents callers from mutating entry.results and
            # corrupting the cache. Using model_copy(deep=True) ensures
            # nested structures (results list) are copied.
            return entry.model_copy(deep=True)

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set entry in cache.

        Evicts least recently used entry if at capacity.

        Args:
            key: Cache key
            entry: CacheEntry to store
        """
        async with self._lock:
            # If key exists, update it
            if key in self._cache:
                self._cache[key] = entry
                self._cache.move_to_end(key)
                return

            # Evict oldest when at capacity to make room for new entry
            # Use while for defensive programming in case maxsize is reduced.
            # WARNING: Changing maxsize at runtime is unsafe and may cause
            # multiple evictions. The LRUCache does not protect against this.
            while len(self._cache) >= self.maxsize:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug(f"Evicted cache entry: {evicted_key[:16]}...")

            self._cache[key] = entry

    async def delete(self, key: str) -> bool:
        """Delete entry from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if entry was deleted, False if not found
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """Clear all entries from cache.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    async def clear_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of expired entries removed
        """
        async with self._lock:
            now = datetime.now(UTC)
            expired_keys = [k for k, v in self._cache.items() if now > v.expires_at]
            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                logger.debug(f"Cleared {len(expired_keys)} expired cache entries")

            return len(expired_keys)

    @property
    def size(self) -> int:
        """Current cache entry count (approximate, not locked)."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate 0.0-1.0 (approximate, not locked)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics (approximate, not locked).

        Returns:
            Dictionary with cache stats
        """
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


class SearchCache:
    """High-level search result caching manager.

    Wraps LRUCache with query-specific functionality including
    cache key generation, TTL management, and result serialization.
    """

    def __init__(
        self,
        maxsize: int = 100,
        default_ttl_seconds: int = 300,
        enable_stats: bool = True,
        auto_start_maintenance: bool = False,
        maintenance_interval_seconds: int = 300,
    ):
        """Initialize search cache.

        Args:
            maxsize: Maximum cached entries
            default_ttl_seconds: Default TTL in seconds (default: 300)
            enable_stats: Whether to track statistics
            auto_start_maintenance: Whether to auto-start maintenance
                (Default: False. Caller should call start_maintenance())
            maintenance_interval_seconds: Maintenance interval (default: 300)
        """
        if default_ttl_seconds <= 0:
            logger.error(
                f"Invalid default_ttl_seconds for SearchCache: {default_ttl_seconds} "
                f"(type={type(default_ttl_seconds).__name__}). Expected: positive integer."
            )
            raise ValueError("default_ttl_seconds must be positive")

        self._lru = LRUCache(maxsize=maxsize)
        self.default_ttl = default_ttl_seconds
        self.enable_stats = enable_stats
        self._maintenance_task: asyncio.Task | None = None
        self._maintenance_interval = maintenance_interval_seconds

        logger.info(f"SearchCache initialized: maxsize={maxsize}, ttl={default_ttl_seconds}s")

        # BUG FIX: Removed auto_start_maintenance from __init__.
        # Creating tasks from __init__ is fragile - the task reference
        # isn't tracked until after start_maintenance() runs, and
        # __init__ may be called outside an async context. Callers
        # should explicitly call start_maintenance() when ready.

    def generate_cache_key(
        self,
        query: str,
        project_filter: str | None = None,
        search_type: str = "hybrid",
        limit: int = 5,
        **extra_params: Any,
    ) -> str:
        """Generate a normalized cache key from query parameters.

        Args:
            query: Search query
            project_filter: Optional project filter
            search_type: Type of search (hybrid, vector)
            limit: Result limit
            **extra_params: Additional parameters to include in key

        Returns:
            SHA-256 hash prefix as cache key
        """
        # Normalize query
        normalized_query = query.lower().strip()

        # Build key components
        key_parts = [
            normalized_query,
            project_filter or "",
            search_type,
            str(limit),
        ]

        # Add sorted extra params
        if extra_params:
            sorted_params = json.dumps(extra_params, sort_keys=True)
            key_parts.append(sorted_params)

        key_string = "|".join(key_parts)

        # Reject invalid UTF-8 in cache keys to prevent hash collisions
        try:
            hash_bytes = key_string.encode("utf-8")
        except UnicodeEncodeError as e:
            logger.error(
                f"Invalid UTF-8 in cache key at position {e.start}-{e.end}: {e.reason}. "
                f"Query: {query[:50]}..."
            )
            # Use fallback encoding (don't double-hash)
            hash_bytes = key_string.encode("utf-8", errors="replace")

        return hashlib.sha256(hash_bytes).hexdigest()[:32]

    async def get_cached_results(
        self,
        query: str,
        project_filter: str | None = None,
        search_type: str = "hybrid",
        limit: int = 5,
        **extra_params: Any,
    ) -> list[dict[str, Any]] | None:
        """Get cached search results.

        Args:
            query: Search query
            project_filter: Optional project filter
            search_type: Type of search
            limit: Result limit
            **extra_params: Additional parameters

        Returns:
            Cached results if found and valid, None otherwise
        """
        cache_key = self.generate_cache_key(
            query, project_filter, search_type, limit, **extra_params
        )

        entry = await self._lru.get(cache_key)
        if entry is None:
            logger.debug(f"Cache miss for query: {query[:50]}...")
            return None

        logger.debug(f"Cache hit for query: {query[:50]}...")
        return entry.results

    async def cache_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        project_filter: str | None = None,
        search_type: str = "hybrid",
        limit: int = 5,
        ttl_seconds: int | None = None,
        **extra_params: Any,
    ) -> None:
        """Cache search results.

        Args:
            query: Search query
            results: Results to cache
            project_filter: Optional project filter
            search_type: Type of search
            limit: Result limit
            ttl_seconds: Custom TTL (uses default if None)
            **extra_params: Additional parameters
        """
        logger.debug(
            f"cache_results() called: query_length={len(query)}, "
            f"result_count={len(results)}, search_type={search_type}"
        )

        cache_key = self.generate_cache_key(
            query, project_filter, search_type, limit, **extra_params
        )

        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        now = datetime.now(UTC)

        entry = CacheEntry(
            cache_key=cache_key,
            query=query,
            results=results,
            result_count=len(results),
            created_at=now,
            expires_at=now + timedelta(seconds=ttl),
            search_type=search_type,
            project_filter=project_filter,
        )

        await self._lru.set(cache_key, entry)
        logger.debug(f"Cached {len(results)} results for query: {query[:50]}...")

    async def invalidate(
        self,
        query: str,
        project_filter: str | None = None,
        search_type: str = "hybrid",
        limit: int = 5,
        **extra_params: Any,
    ) -> bool:
        """Invalidate a specific cache entry.

        Args:
            query: Search query
            project_filter: Optional project filter
            search_type: Type of search
            limit: Result limit
            **extra_params: Additional parameters

        Returns:
            True if entry was invalidated, False if not found
        """
        cache_key = self.generate_cache_key(
            query, project_filter, search_type, limit, **extra_params
        )
        return await self._lru.delete(cache_key)

    async def clear_all(self) -> int:
        """Clear all cached entries.

        Returns:
            Number of entries cleared
        """
        count = await self._lru.clear()
        logger.info(f"Cleared {count} cache entries")
        return count

    async def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of expired entries removed
        """
        return await self._lru.clear_expired()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = self._lru.get_stats()
        stats["default_ttl_seconds"] = self.default_ttl
        return stats

    async def start_maintenance(self, interval_seconds: int = 300) -> None:
        """Start background maintenance task for cleanup.

        Args:
            interval_seconds: Cleanup interval in seconds (default: 300)
        """
        if self._maintenance_task is not None:
            return  # Already running

        async def maintenance_loop():
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    expired = await self.cleanup_expired()
                    if expired > 0:
                        logger.debug(f"Maintenance: cleared {expired} expired entries")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    # L2 fix: Add exc_info for better debugging
                    logger.warning(f"Cache maintenance error: {e}", exc_info=True)

        self._maintenance_task = asyncio.create_task(maintenance_loop())
        logger.info(f"Started cache maintenance task (interval: {interval_seconds}s)")

    async def stop_maintenance(self) -> None:
        """Stop background maintenance task."""
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._maintenance_task
            self._maintenance_task = None
            logger.info("Stopped cache maintenance task")


# Global cache instance (lazy initialization with thread safety)
_global_cache: SearchCache | None = None
_global_cache_lock = threading.Lock()


def get_search_cache(
    maxsize: int = 100,
    default_ttl_seconds: int = 300,
) -> SearchCache:
    """Get or create the global search cache instance.

    Uses double-check locking for thread-safe lazy initialization.

    Args:
        maxsize: Maximum cached entries (only used on first call)
        default_ttl_seconds: Default TTL in seconds (first call only)

    Returns:
        Global SearchCache instance
    """
    global _global_cache
    if _global_cache is None:
        with _global_cache_lock:
            # Double-check after acquiring lock
            if _global_cache is None:
                _global_cache = SearchCache(
                    maxsize=maxsize,
                    default_ttl_seconds=default_ttl_seconds,
                )
    return _global_cache


def reset_search_cache() -> None:
    """Reset global search cache for testing."""
    global _global_cache
    with _global_cache_lock:
        _global_cache = None
