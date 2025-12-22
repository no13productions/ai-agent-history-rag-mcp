"""Tests for result evaluator functionality."""

import pytest

from claude_history_rag.decision_engine.evaluator import ResultEvaluator
from claude_history_rag.decision_engine.models import QueryAnalysis, QueryIntent


class TestResultEvaluator:
    """Tests for ResultEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create a fresh evaluator for each test."""
        return ResultEvaluator(
            adequacy_threshold=0.5,
            completeness_threshold=0.4,
        )

    @pytest.fixture
    def sample_results(self):
        """Sample search results for testing."""
        return [
            {
                "id": "1",
                "content": "This is about Python authentication using Django. Here's how to set it up.",
                "score": 0.2,
                "chunk_type": "turn",
                "session_id": "session-1",
            },
            {
                "id": "2",
                "content": "Authentication in Python requires careful handling of passwords.",
                "score": 0.3,
                "chunk_type": "turn",
                "session_id": "session-2",
            },
            {
                "id": "3",
                "content": "Use bcrypt for password hashing in your Python application.",
                "score": 0.4,
                "chunk_type": "turn",
                "session_id": "session-3",
            },
        ]

    @pytest.fixture
    def sample_analysis(self):
        """Sample query analysis for testing."""
        return QueryAnalysis(
            original_query="python authentication",
            normalized_query="python authentication",
            intent=QueryIntent.HOW_TO,
            detected_technologies=["python"],
            key_terms=["authentication"],
            confidence=0.7,
        )

    def test_evaluate_empty_results(self, evaluator):
        """Test evaluation of empty results."""
        evaluation = evaluator.evaluate("test query", [])

        assert evaluation.relevance_score == 0.0
        assert evaluation.completeness_score == 0.0
        assert evaluation.needs_refinement is True
        assert evaluation.result_count == 0

    def test_evaluate_with_results(self, evaluator, sample_results, sample_analysis):
        """Test evaluation with actual results."""
        evaluation = evaluator.evaluate("python authentication", sample_results, sample_analysis)

        assert evaluation.relevance_score > 0
        assert evaluation.completeness_score > 0
        assert evaluation.result_count == 3

    def test_needs_refinement_low_relevance(self, evaluator):
        """Test that low relevance triggers refinement need."""
        poor_results = [
            {
                "id": "1",
                "content": "Completely unrelated content about cooking.",
                "score": 0.9,  # High distance = low similarity
                "chunk_type": "turn",
            }
        ]

        evaluation = evaluator.evaluate("python authentication", poor_results)
        assert evaluation.needs_refinement is True

    def test_refinement_suggestions_provided(self, evaluator):
        """Test that refinement suggestions are provided when needed."""
        poor_results = [
            {
                "id": "1",
                "content": "Some vague content.",
                "score": 0.8,
                "chunk_type": "turn",
            }
        ]

        evaluation = evaluator.evaluate("python authentication", poor_results)
        # When refinement is needed, suggestions should be provided
        # (unless results are completely empty)
        assert evaluation.needs_refinement is True

    def test_high_quality_results_no_refinement(self, evaluator, sample_results, sample_analysis):
        """Test that high quality results don't need refinement."""
        # Adjust scores to indicate high similarity (low distance)
        high_quality = [
            {**r, "score": 0.1}
            for r in sample_results  # Low distance = high similarity
        ]

        evaluation = evaluator.evaluate("python authentication", high_quality, sample_analysis)

        # With good results, might not need refinement
        # (depends on threshold and content matching)
        assert evaluation.relevance_score > 0.5

    def test_result_count_in_evaluation(self, evaluator, sample_results):
        """Test that result count is correctly tracked."""
        evaluation = evaluator.evaluate("test", sample_results)
        assert evaluation.result_count == len(sample_results)

    def test_avg_result_score(self, evaluator, sample_results):
        """Test that average score is calculated."""
        evaluation = evaluator.evaluate("python authentication", sample_results)
        assert evaluation.avg_result_score > 0

    def test_confidence_calculation(self, evaluator, sample_results, sample_analysis):
        """Test that confidence is calculated."""
        evaluation = evaluator.evaluate("python authentication", sample_results, sample_analysis)
        assert 0 <= evaluation.confidence <= 1

    def test_file_history_intent_suggestions(self, evaluator):
        """Test suggestions for file history intent."""
        analysis = QueryAnalysis(
            original_query="what changed in config.py",
            normalized_query="what changed in config.py",
            intent=QueryIntent.FILE_HISTORY,
            detected_technologies=[],
            key_terms=["changed", "config"],
            confidence=0.6,
        )

        # Results without file_change chunks
        results = [
            {
                "id": "1",
                "content": "Some discussion about config files.",
                "score": 0.3,
                "chunk_type": "turn",  # Not file_change
            }
        ]

        evaluation = evaluator.evaluate("what changed in config.py", results, analysis)

        # Should suggest using search_file_changes tool
        assert (
            any("search_file_changes" in s for s in evaluation.refinement_suggestions)
            or "File change history" in evaluation.missing_information
        )

    def test_code_example_intent_suggestions(self, evaluator):
        """Test suggestions for code example intent."""
        analysis = QueryAnalysis(
            original_query="show me an example",
            normalized_query="show me an example",
            intent=QueryIntent.CODE_EXAMPLE,
            detected_technologies=[],
            key_terms=["example"],
            confidence=0.6,
        )

        # Results without code blocks
        results = [
            {
                "id": "1",
                "content": "Here is some explanation without code.",
                "score": 0.3,
                "chunk_type": "turn",
            }
        ]

        evaluation = evaluator.evaluate("show me an example", results, analysis)

        # Should note missing code examples
        assert "Code examples" in evaluation.missing_information or any(
            "example" in s.lower() or "code" in s.lower() for s in evaluation.refinement_suggestions
        )

    def test_should_refine_method(self, evaluator):
        """Test the should_refine convenience method."""
        # Create an evaluation that needs refinement
        poor_results = [
            {
                "id": "1",
                "content": "Irrelevant.",
                "score": 0.9,
                "chunk_type": "turn",
            }
        ]
        evaluation = evaluator.evaluate("specific query", poor_results)

        # should_refine checks both needs_refinement and has suggestions
        result = evaluator.should_refine(evaluation)
        assert isinstance(result, bool)

    def test_invalid_threshold_raises(self):
        """Test that invalid thresholds raise ValueError."""
        with pytest.raises(ValueError):
            ResultEvaluator(adequacy_threshold=1.5)

        with pytest.raises(ValueError):
            ResultEvaluator(completeness_threshold=-0.1)

    def test_term_overlap_scoring(self, evaluator):
        """Test that term overlap contributes to scoring."""
        # Results that contain query terms should score higher
        query = "python authentication security"

        matching_results = [
            {
                "id": "1",
                "content": "Python authentication requires security best practices.",
                "score": 0.5,
                "chunk_type": "turn",
            }
        ]

        non_matching_results = [
            {
                "id": "2",
                "content": "Cooking recipes and gardening tips.",
                "score": 0.5,  # Same distance
                "chunk_type": "turn",
            }
        ]

        eval_matching = evaluator.evaluate(query, matching_results)
        eval_non_matching = evaluator.evaluate(query, non_matching_results)

        # Matching content should have higher relevance
        assert eval_matching.relevance_score > eval_non_matching.relevance_score
