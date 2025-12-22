"""Tests for decision engine orchestration."""

import pytest

from claude_history_rag.decision_engine.cache import SearchCache
from claude_history_rag.decision_engine.engine import DecisionEngine
from claude_history_rag.decision_engine.evaluator import ResultEvaluator
from claude_history_rag.decision_engine.query_analyzer import QueryAnalyzer
from claude_history_rag.decision_engine.synthesizer import ResultSynthesizer


class TestDecisionEngine:
    """Tests for DecisionEngine orchestration."""

    @pytest.fixture
    def engine(self):
        """Create a fresh engine for each test."""
        return DecisionEngine(
            cache=SearchCache(maxsize=10, default_ttl_seconds=3600),
            analyzer=QueryAnalyzer(),
            evaluator=ResultEvaluator(),
            synthesizer=ResultSynthesizer(),
            enable_cache=True,
            enable_refinement=True,
            enable_synthesis=True,
        )

    @pytest.fixture
    def engine_no_features(self):
        """Create an engine with all features disabled."""
        return DecisionEngine(
            enable_cache=False,
            enable_refinement=False,
            enable_synthesis=False,
        )

    @pytest.fixture
    def mock_results(self):
        """Mock search results."""
        return [
            {
                "id": "1",
                "content": "Python authentication using Django. Here's how to set it up properly.",
                "score": 0.2,
                "chunk_type": "turn",
                "session_id": "session-1",
                "timestamp": "2024-01-01T10:00:00Z",
            },
            {
                "id": "2",
                "content": "Authentication best practices for Python applications.",
                "score": 0.3,
                "chunk_type": "turn",
                "session_id": "session-2",
                "timestamp": "2024-01-02T10:00:00Z",
            },
        ]

    async def test_search_basic(self, engine, mock_results):
        """Test basic search functionality."""

        async def mock_search_func(query, vector, limit, project_filter):
            return mock_results

        async def mock_embed_func(query):
            return [0.1] * 768  # Mock embedding

        result = await engine.search(
            query="python authentication",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
            limit=5,
        )

        assert "results" in result
        assert "count" in result
        assert "query" in result
        assert result["count"] == 2

    async def test_search_includes_analysis(self, engine, mock_results):
        """Test that search includes query analysis."""

        async def mock_search_func(query, vector, limit, project_filter):
            return mock_results

        async def mock_embed_func(query):
            return [0.1] * 768

        result = await engine.search(
            query="python authentication",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
        )

        assert "analysis" in result
        assert "intent" in result["analysis"]
        assert "detected_technologies" in result["analysis"]

    async def test_search_includes_evaluation(self, engine, mock_results):
        """Test that search includes result evaluation."""

        async def mock_search_func(query, vector, limit, project_filter):
            return mock_results

        async def mock_embed_func(query):
            return [0.1] * 768

        result = await engine.search(
            query="python authentication",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
        )

        assert "evaluation" in result
        assert "relevance_score" in result["evaluation"]
        assert "completeness_score" in result["evaluation"]

    async def test_search_includes_synthesis(self, engine, mock_results):
        """Test that search includes synthesis when enabled."""
        engine.enable_synthesis = True

        async def mock_search_func(query, vector, limit, project_filter):
            return mock_results

        async def mock_embed_func(query):
            return [0.1] * 768

        result = await engine.search(
            query="python authentication",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
        )

        assert "synthesis" in result
        assert "primary_content" in result["synthesis"]

    async def test_search_includes_metrics(self, engine, mock_results):
        """Test that search includes timing metrics."""

        async def mock_search_func(query, vector, limit, project_filter):
            return mock_results

        async def mock_embed_func(query):
            return [0.1] * 768

        result = await engine.search(
            query="python authentication",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
            include_debug=True,
        )

        assert "metrics" in result
        metrics = result["metrics"]
        assert "query_analysis_ms" in metrics
        assert "search_ms" in metrics
        assert "total_ms" in metrics
        assert "decisions_made" in metrics

    async def test_cache_hit(self, mock_results):
        """Test that cache hits work correctly."""
        # Create engine without refinement to test cache behavior in isolation
        engine = DecisionEngine(
            cache=SearchCache(maxsize=10, default_ttl_seconds=3600),
            analyzer=QueryAnalyzer(),
            evaluator=ResultEvaluator(),
            synthesizer=ResultSynthesizer(),
            enable_cache=True,
            enable_refinement=False,  # Disable to test cache in isolation
            enable_synthesis=False,
        )

        call_count = 0

        async def mock_search_func(query, vector, limit, project_filter):
            nonlocal call_count
            call_count += 1
            return mock_results

        async def mock_embed_func(query):
            return [0.1] * 768

        # First call - cache miss
        result1 = await engine.search(
            query="test query",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
        )
        assert result1["cache_hit"] is False
        assert call_count == 1

        # Second call - cache hit
        result2 = await engine.search(
            query="test query",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
        )
        assert result2["cache_hit"] is True
        assert call_count == 1  # Search not called again

        # Bug 3 fix verification: Cache hits must include evaluation for schema consistency
        assert "evaluation" in result2
        assert "relevance_score" in result2["evaluation"]
        assert "completeness_score" in result2["evaluation"]

    async def test_cache_disabled(self, engine_no_features, mock_results):
        """Test search with cache disabled."""

        call_count = 0

        async def mock_search_func(query, vector, limit, project_filter):
            nonlocal call_count
            call_count += 1
            return mock_results

        async def mock_embed_func(query):
            return [0.1] * 768

        # First call
        await engine_no_features.search(
            query="test query",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
        )

        # Second call - should still call search (no cache)
        await engine_no_features.search(
            query="test query",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
        )

        assert call_count == 2  # Called both times

    async def test_refinement_triggered_on_poor_results(self, engine):
        """Test that refinement is triggered for poor results."""
        poor_results = [
            {
                "id": "1",
                "content": "Completely unrelated content.",
                "score": 0.9,  # High distance = low similarity
                "chunk_type": "turn",
                "session_id": "session-1",
            }
        ]

        async def mock_search_func(query, vector, limit, project_filter):
            return poor_results

        async def mock_embed_func(query):
            return [0.1] * 768

        result = await engine.search(
            query="python authentication",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
            include_debug=True,
        )

        metrics = result["metrics"]
        # Refinement should be triggered due to poor results
        assert (
            metrics["refinement_triggered"] or "refinement:triggered" in metrics["decisions_made"]
        )

    async def test_empty_results_handling(self, engine):
        """Test handling of empty search results."""

        async def mock_search_func(query, vector, limit, project_filter):
            return []

        async def mock_embed_func(query):
            return [0.1] * 768

        result = await engine.search(
            query="nonexistent topic",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
        )

        assert result["count"] == 0
        assert result["results"] == []
        assert result["evaluation"]["relevance_score"] == 0.0

    async def test_project_filter_passed(self, engine, mock_results):
        """Test that project filter is passed to search function."""
        received_filter = None

        async def mock_search_func(query, vector, limit, project_filter):
            nonlocal received_filter
            received_filter = project_filter
            return mock_results

        async def mock_embed_func(query):
            return [0.1] * 768

        await engine.search(
            query="test",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
            project_filter="/my/project",
        )

        assert received_filter == "/my/project"

    async def test_get_cache_stats(self, engine, mock_results):
        """Test getting cache statistics."""

        async def mock_search_func(query, vector, limit, project_filter):
            return mock_results

        async def mock_embed_func(query):
            return [0.1] * 768

        # Perform a search to populate cache
        await engine.search(
            query="test",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
        )

        stats = await engine.get_cache_stats()
        assert stats is not None
        assert "size" in stats
        assert stats["size"] >= 1

    async def test_clear_cache(self, engine, mock_results):
        """Test clearing the cache."""

        async def mock_search_func(query, vector, limit, project_filter):
            return mock_results

        async def mock_embed_func(query):
            return [0.1] * 768

        # Perform searches to populate cache
        await engine.search(query="test1", search_func=mock_search_func, embed_func=mock_embed_func)
        await engine.search(query="test2", search_func=mock_search_func, embed_func=mock_embed_func)

        # Clear cache
        cleared = await engine.clear_cache()
        assert cleared >= 2

        # Verify cache is empty
        stats = await engine.get_cache_stats()
        assert stats["size"] == 0

    async def test_decisions_tracked_in_metrics(self, engine, mock_results):
        """Test that decisions are tracked in metrics."""

        async def mock_search_func(query, vector, limit, project_filter):
            return mock_results

        async def mock_embed_func(query):
            return [0.1] * 768

        result = await engine.search(
            query="python django authentication",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
            include_debug=True,
        )

        decisions = result["metrics"]["decisions_made"]
        assert len(decisions) > 0
        # Should include intent decision
        assert any("intent:" in d for d in decisions)

    async def test_search_type_passed_correctly(self, engine, mock_results):
        """Test that search type is included in response."""

        async def mock_search_func(query, vector, limit, project_filter):
            return mock_results

        async def mock_embed_func(query):
            return [0.1] * 768

        result = await engine.search(
            query="test",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
            search_type="vector",
        )

        assert result["search_type"] == "vector"

    async def test_empty_query_validation(self, engine):
        """Test E1 fix: empty query returns error at engine level."""

        async def mock_search_func(query, vector, limit, project_filter):
            return []

        async def mock_embed_func(query):
            return [0.1] * 768

        # Test completely empty query
        result = await engine.search(
            query="",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
        )

        assert result["count"] == 0
        assert "error" in result
        assert "empty" in result["error"].lower()

        # Test whitespace-only query
        result = await engine.search(
            query="   ",
            search_func=mock_search_func,
            embed_func=mock_embed_func,
        )

        assert result["count"] == 0
        assert "error" in result
        assert "empty" in result["error"].lower()

    async def test_query_length_validation(self, engine):
        """Test E2 fix: query length limit enforced at engine level."""

        async def mock_search_func(query, vector, limit, project_filter):
            return []

        async def mock_embed_func(query):
            return [0.1] * 768

        # Test query exceeding 10,000 chars
        long_query = "x" * 10001

        result = await engine.search(
            query=long_query,
            search_func=mock_search_func,
            embed_func=mock_embed_func,
        )

        assert result["count"] == 0
        assert "error" in result
        assert "too long" in result["error"].lower()
        assert "10,000" in result["error"]
