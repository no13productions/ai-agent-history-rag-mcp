"""Tests for search cache functionality."""

from datetime import UTC, datetime, timedelta

import pytest

from claude_history_rag.decision_engine.cache import LRUCache, SearchCache
from claude_history_rag.decision_engine.models import CacheEntry


class TestLRUCache:
    """Tests for LRUCache implementation."""

    @pytest.fixture
    def cache(self):
        """Create a fresh LRU cache for each test."""
        return LRUCache(maxsize=3)

    async def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        entry = CacheEntry(
            cache_key="test_key",
            query="test query",
            results=[{"id": "1", "content": "test"}],
            result_count=1,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        await cache.set("test_key", entry)
        result = await cache.get("test_key")

        assert result is not None
        assert result.query == "test query"
        assert result.result_count == 1

    async def test_get_nonexistent_key(self, cache):
        """Test getting a key that doesn't exist."""
        result = await cache.get("nonexistent")
        assert result is None
        assert cache._misses == 1

    async def test_lru_eviction(self, cache):
        """Test that oldest items are evicted when cache is full."""
        # Fill cache to capacity
        for i in range(3):
            entry = CacheEntry(
                cache_key=f"key_{i}",
                query=f"query {i}",
                results=[],
                result_count=0,
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
            await cache.set(f"key_{i}", entry)

        # Add one more to trigger eviction
        new_entry = CacheEntry(
            cache_key="key_3",
            query="query 3",
            results=[],
            result_count=0,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        await cache.set("key_3", new_entry)

        # First key should be evicted
        assert await cache.get("key_0") is None
        # Other keys should still exist
        assert await cache.get("key_1") is not None
        assert await cache.get("key_2") is not None
        assert await cache.get("key_3") is not None

    async def test_expired_entry_returns_none(self, cache):
        """Test that expired entries are not returned."""
        expired_entry = CacheEntry(
            cache_key="expired",
            query="expired query",
            results=[],
            result_count=0,
            expires_at=datetime.now(UTC) - timedelta(hours=1),  # Already expired
        )
        await cache.set("expired", expired_entry)

        result = await cache.get("expired")
        assert result is None

    async def test_hit_rate_calculation(self, cache):
        """Test hit rate is calculated correctly."""
        entry = CacheEntry(
            cache_key="test",
            query="test",
            results=[],
            result_count=0,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        await cache.set("test", entry)

        # One hit
        await cache.get("test")
        # Two misses
        await cache.get("miss1")
        await cache.get("miss2")

        # 1 hit / 3 total = 0.333...
        assert abs(cache.hit_rate - 1 / 3) < 0.01

    async def test_clear_expired(self, cache):
        """Test clearing expired entries."""
        valid_entry = CacheEntry(
            cache_key="valid",
            query="valid",
            results=[],
            result_count=0,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        expired_entry = CacheEntry(
            cache_key="expired",
            query="expired",
            results=[],
            result_count=0,
            expires_at=datetime.now(UTC) - timedelta(hours=1),
        )

        await cache.set("valid", valid_entry)
        await cache.set("expired", expired_entry)

        cleared = await cache.clear_expired()
        assert cleared == 1
        assert cache.size == 1

    async def test_access_updates_lru_order(self, cache):
        """Test that accessing an entry moves it to most recently used."""
        for i in range(3):
            entry = CacheEntry(
                cache_key=f"key_{i}",
                query=f"query {i}",
                results=[],
                result_count=0,
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
            await cache.set(f"key_{i}", entry)

        # Access key_0 to make it most recently used
        await cache.get("key_0")

        # Add new entry to trigger eviction
        new_entry = CacheEntry(
            cache_key="key_3",
            query="query 3",
            results=[],
            result_count=0,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        await cache.set("key_3", new_entry)

        # key_1 should be evicted (oldest after key_0 was accessed)
        assert await cache.get("key_0") is not None  # Was accessed, not evicted
        assert await cache.get("key_1") is None  # Oldest, was evicted
        assert await cache.get("key_2") is not None
        assert await cache.get("key_3") is not None


class TestSearchCache:
    """Tests for SearchCache functionality."""

    @pytest.fixture
    def search_cache(self):
        """Create a fresh SearchCache for each test."""
        return SearchCache(maxsize=10, default_ttl_seconds=3600)

    def test_generate_cache_key_consistent(self, search_cache):
        """Test that cache keys are generated consistently."""
        key1 = search_cache.generate_cache_key("test query", project_filter="/test")
        key2 = search_cache.generate_cache_key("test query", project_filter="/test")
        assert key1 == key2

    def test_generate_cache_key_different_for_different_queries(self, search_cache):
        """Test that different queries generate different keys."""
        key1 = search_cache.generate_cache_key("query one")
        key2 = search_cache.generate_cache_key("query two")
        assert key1 != key2

    def test_generate_cache_key_normalized(self, search_cache):
        """Test that queries are normalized for caching."""
        key1 = search_cache.generate_cache_key("Test Query")
        key2 = search_cache.generate_cache_key("test query")
        key3 = search_cache.generate_cache_key("  test query  ")
        assert key1 == key2 == key3

    def test_generate_cache_key_includes_filters(self, search_cache):
        """Test that different filters generate different keys."""
        key1 = search_cache.generate_cache_key("query", project_filter="/project1")
        key2 = search_cache.generate_cache_key("query", project_filter="/project2")
        assert key1 != key2

    async def test_cache_and_retrieve_results(self, search_cache):
        """Test caching and retrieving search results."""
        results = [
            {"id": "1", "content": "Result 1", "score": 0.9},
            {"id": "2", "content": "Result 2", "score": 0.8},
        ]

        await search_cache.cache_results(
            query="test query",
            results=results,
            project_filter="/test",
        )

        cached = await search_cache.get_cached_results(
            query="test query",
            project_filter="/test",
        )

        assert cached is not None
        assert len(cached) == 2
        assert cached[0]["id"] == "1"

    async def test_cache_miss_returns_none(self, search_cache):
        """Test that cache miss returns None."""
        result = await search_cache.get_cached_results(query="uncached query")
        assert result is None

    async def test_invalidate_removes_entry(self, search_cache):
        """Test that invalidation removes the cached entry."""
        await search_cache.cache_results(
            query="test query",
            results=[{"id": "1"}],
        )

        # Verify it's cached
        assert await search_cache.get_cached_results(query="test query") is not None

        # Invalidate
        removed = await search_cache.invalidate(query="test query")
        assert removed is True

        # Verify it's gone
        assert await search_cache.get_cached_results(query="test query") is None

    async def test_clear_all_removes_everything(self, search_cache):
        """Test that clear_all removes all entries."""
        for i in range(5):
            await search_cache.cache_results(
                query=f"query {i}",
                results=[{"id": str(i)}],
            )

        cleared = await search_cache.clear_all()
        assert cleared == 5

        # Verify all are gone
        for i in range(5):
            assert await search_cache.get_cached_results(query=f"query {i}") is None

    def test_get_stats(self, search_cache):
        """Test getting cache statistics."""
        stats = search_cache.get_stats()

        assert "size" in stats
        assert "maxsize" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "default_ttl_seconds" in stats

    def test_invalid_maxsize_raises(self):
        """Test that invalid maxsize raises ValueError."""
        with pytest.raises(ValueError, match="maxsize must be positive"):
            LRUCache(maxsize=0)

    def test_invalid_ttl_raises(self):
        """Test that invalid TTL raises ValueError."""
        with pytest.raises(ValueError, match="default_ttl_seconds must be positive"):
            SearchCache(default_ttl_seconds=0)
