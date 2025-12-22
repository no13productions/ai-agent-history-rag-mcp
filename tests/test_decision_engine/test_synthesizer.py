"""Tests for result synthesizer functionality."""

import pytest

from claude_history_rag.decision_engine.models import QueryAnalysis, QueryIntent
from claude_history_rag.decision_engine.synthesizer import ResultSynthesizer


class TestResultSynthesizer:
    """Tests for ResultSynthesizer."""

    @pytest.fixture
    def synthesizer(self):
        """Create a fresh synthesizer for each test."""
        return ResultSynthesizer(
            similarity_threshold=0.7,
            max_key_points=5,
            max_code_snippets=3,
        )

    @pytest.fixture
    def sample_results(self):
        """Sample search results for testing."""
        return [
            {
                "id": "1",
                "content": "Authentication should use bcrypt for password hashing. This is important for security.",
                "score": 0.2,
                "chunk_type": "turn",
                "session_id": "session-1",
                "timestamp": "2024-01-01T10:00:00Z",
            },
            {
                "id": "2",
                "content": "Here's an example:\n```python\nimport bcrypt\nhash = bcrypt.hashpw(password, bcrypt.gensalt())\n```",
                "score": 0.3,
                "chunk_type": "turn",
                "session_id": "session-2",
                "timestamp": "2024-01-02T10:00:00Z",
            },
            {
                "id": "3",
                "content": "File path /src/auth.py was modified to add password validation.",
                "score": 0.4,
                "chunk_type": "file_change",
                "session_id": "session-3",
                "file_path": "/src/auth.py",
                "operation": "edit",
                "timestamp": "2024-01-03T10:00:00Z",
            },
        ]

    @pytest.fixture
    def sample_analysis(self):
        """Sample query analysis for testing."""
        return QueryAnalysis(
            original_query="python password hashing",
            normalized_query="python password hashing",
            intent=QueryIntent.HOW_TO,
            detected_technologies=["python"],
            key_terms=["password", "hashing"],
            confidence=0.7,
        )

    def test_synthesize_empty_results(self, synthesizer):
        """Test synthesis with empty results."""
        synthesis = synthesizer.synthesize("test query", [])

        assert synthesis.synthesis_method == "empty"
        assert synthesis.overall_confidence == 0.0
        assert "No results" in synthesis.primary_content

    def test_synthesize_with_results(self, synthesizer, sample_results, sample_analysis):
        """Test synthesis with actual results."""
        synthesis = synthesizer.synthesize(
            "python password hashing", sample_results, sample_analysis
        )

        assert synthesis.primary_content != ""
        assert synthesis.synthesis_method in ["simple", "dedup"]
        assert len(synthesis.sources) > 0

    def test_code_snippet_extraction(self, synthesizer, sample_results):
        """Test that code snippets are extracted."""
        synthesis = synthesizer.synthesize("test", sample_results)

        assert len(synthesis.code_snippets) > 0
        # Check that python snippet was found
        python_snippets = [s for s in synthesis.code_snippets if s["language"] == "python"]
        assert len(python_snippets) > 0

    def test_file_change_extraction(self, synthesizer, sample_results):
        """Test that file changes are extracted."""
        synthesis = synthesizer.synthesize("test", sample_results)

        assert len(synthesis.file_changes) > 0
        assert synthesis.file_changes[0]["file_path"] == "/src/auth.py"
        assert synthesis.file_changes[0]["operation"] == "edit"

    def test_source_attribution(self, synthesizer, sample_results):
        """Test that sources are properly attributed."""
        synthesis = synthesizer.synthesize("test", sample_results)

        assert len(synthesis.sources) == len(sample_results)
        session_ids = [s.session_id for s in synthesis.sources]
        assert "session-1" in session_ids
        assert "session-2" in session_ids

    def test_deduplication(self, synthesizer):
        """Test that duplicate results are deduplicated."""
        duplicate_results = [
            {
                "id": "1",
                "content": "This is the same content about authentication.",
                "score": 0.2,
                "chunk_type": "turn",
                "session_id": "session-1",
            },
            {
                "id": "2",
                "content": "This is the same content about authentication.",  # Exact duplicate
                "score": 0.3,
                "chunk_type": "turn",
                "session_id": "session-2",
            },
        ]

        synthesis = synthesizer.synthesize("auth", duplicate_results)

        assert synthesis.deduplication_count > 0

    def test_similar_content_deduplication(self, synthesizer):
        """Test that highly similar content is deduplicated."""
        similar_results = [
            {
                "id": "1",
                "content": "Authentication is important for security. Always use strong passwords.",
                "score": 0.2,
                "chunk_type": "turn",
                "session_id": "session-1",
            },
            {
                "id": "2",
                "content": "Authentication is important for security. Always use strong passwords!",  # Very similar
                "score": 0.3,
                "chunk_type": "turn",
                "session_id": "session-2",
            },
        ]

        synthesis = synthesizer.synthesize("auth", similar_results)

        # Should detect high similarity and deduplicate
        assert synthesis.deduplication_count >= 1

    def test_key_points_extraction(self, synthesizer, sample_results, sample_analysis):
        """Test that key points are extracted."""
        synthesis = synthesizer.synthesize("password hashing", sample_results, sample_analysis)

        # Should extract some key points
        assert isinstance(synthesis.key_points, list)

    def test_max_key_points_limit(self, synthesizer):
        """Test that key points are limited to max."""
        # Create results with many potential key points
        verbose_results = [
            {
                "id": str(i),
                "content": f"Important point {i}. This is very important. Remember this. Critical information {i}.",
                "score": 0.2,
                "chunk_type": "turn",
                "session_id": f"session-{i}",
            }
            for i in range(20)
        ]

        synthesis = synthesizer.synthesize("important", verbose_results)

        assert len(synthesis.key_points) <= synthesizer.max_key_points

    def test_max_code_snippets_limit(self, synthesizer):
        """Test that code snippets are limited to max."""
        # Create results with many code blocks
        code_results = [
            {
                "id": str(i),
                "content": f"```python\nprint({i})\n```",
                "score": 0.2,
                "chunk_type": "turn",
                "session_id": f"session-{i}",
            }
            for i in range(10)
        ]

        synthesis = synthesizer.synthesize("python", code_results)

        assert len(synthesis.code_snippets) <= synthesizer.max_code_snippets

    def test_confidence_calculation(self, synthesizer, sample_results):
        """Test that confidence is calculated."""
        synthesis = synthesizer.synthesize("test", sample_results)

        assert 0 <= synthesis.overall_confidence <= 1

    def test_confidence_higher_with_more_results(self, synthesizer):
        """Test that confidence increases with more unique results."""
        few_results = [
            {
                "id": "1",
                "content": "Single result content.",
                "score": 0.2,
                "chunk_type": "turn",
                "session_id": "session-1",
            }
        ]

        many_results = [
            {
                "id": str(i),
                "content": f"Result {i} with unique content about topic {i}.",
                "score": 0.2,
                "chunk_type": "turn",
                "session_id": f"session-{i}",
            }
            for i in range(5)
        ]

        few_synthesis = synthesizer.synthesize("test", few_results)
        many_synthesis = synthesizer.synthesize("test", many_results)

        assert many_synthesis.overall_confidence >= few_synthesis.overall_confidence

    def test_invalid_similarity_threshold_raises(self):
        """Test that invalid similarity threshold raises ValueError."""
        with pytest.raises(ValueError):
            ResultSynthesizer(similarity_threshold=1.5)

        with pytest.raises(ValueError):
            ResultSynthesizer(similarity_threshold=-0.1)

    def test_invalid_max_key_points_raises(self):
        """Test that invalid max_key_points raises ValueError."""
        with pytest.raises(ValueError):
            ResultSynthesizer(max_key_points=0)

    def test_truncate_long_content(self, synthesizer):
        """Test that long content is truncated in file change previews."""
        long_content = "x" * 500  # Longer than truncate limit
        results = [
            {
                "id": "1",
                "content": long_content,
                "score": 0.2,
                "chunk_type": "file_change",
                "session_id": "session-1",
                "file_path": "/test.py",
                "operation": "edit",
            }
        ]

        synthesis = synthesizer.synthesize("test", results)

        # Content preview should be truncated
        if synthesis.file_changes:
            preview = synthesis.file_changes[0].get("content_preview", "")
            assert len(preview) <= 250  # 200 + buffer for "..."

    def test_sources_sorted_by_relevance(self, synthesizer, sample_results):
        """Test that sources are sorted by relevance score."""
        synthesis = synthesizer.synthesize("test", sample_results)

        # Sources should be sorted by relevance (highest first)
        relevance_scores = [s.relevance_score for s in synthesis.sources]
        assert relevance_scores == sorted(relevance_scores, reverse=True)
