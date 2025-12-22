"""Tests for query analyzer functionality."""

import pytest

from claude_history_rag.decision_engine.models import QueryIntent
from claude_history_rag.decision_engine.query_analyzer import QueryAnalyzer


class TestQueryAnalyzer:
    """Tests for QueryAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a fresh analyzer for each test."""
        return QueryAnalyzer()

    def test_analyze_empty_query(self, analyzer):
        """Test analyzing empty query."""
        analysis = analyzer.analyze("")
        assert analysis.original_query == ""
        assert analysis.normalized_query == ""
        assert analysis.confidence == 0.0

    def test_analyze_whitespace_query(self, analyzer):
        """Test analyzing whitespace-only query."""
        analysis = analyzer.analyze("   ")
        assert analysis.normalized_query == ""
        assert analysis.confidence == 0.0

    def test_detect_python_technology(self, analyzer):
        """Test detection of Python technology."""
        analysis = analyzer.analyze("How do I use Django models?")
        assert "python" in analysis.detected_technologies

    def test_detect_javascript_technology(self, analyzer):
        """Test detection of JavaScript technology."""
        analysis = analyzer.analyze("React component lifecycle")
        assert "javascript" in analysis.detected_technologies

    def test_detect_multiple_technologies(self, analyzer):
        """Test detection of multiple technologies."""
        analysis = analyzer.analyze("How to connect Python Flask to PostgreSQL database?")
        assert "python" in analysis.detected_technologies
        assert "postgresql" in analysis.detected_technologies

    def test_detect_docker_technology(self, analyzer):
        """Test detection of Docker technology."""
        analysis = analyzer.analyze("Create a Dockerfile for my app")
        assert "docker" in analysis.detected_technologies

    def test_how_to_intent(self, analyzer):
        """Test classification of how-to intent."""
        analysis = analyzer.analyze("How to implement authentication?")
        assert analysis.intent == QueryIntent.HOW_TO

    def test_problem_solving_intent(self, analyzer):
        """Test classification of problem-solving intent."""
        analysis = analyzer.analyze("Why is my app crashing on startup?")
        assert analysis.intent == QueryIntent.PROBLEM_SOLVING

        analysis2 = analyzer.analyze("Fix the error in the login form")
        assert analysis2.intent == QueryIntent.PROBLEM_SOLVING

    def test_code_example_intent(self, analyzer):
        """Test classification of code example intent."""
        analysis = analyzer.analyze("Show me an example of async await")
        assert analysis.intent == QueryIntent.CODE_EXAMPLE

    def test_file_history_intent(self, analyzer):
        """Test classification of file history intent."""
        analysis = analyzer.analyze("What did we change in the config file?")
        assert analysis.intent == QueryIntent.FILE_HISTORY

    def test_session_context_intent(self, analyzer):
        """Test classification of session context intent."""
        analysis = analyzer.analyze("What did we discuss earlier?")
        assert analysis.intent == QueryIntent.SESSION_CONTEXT

    def test_decision_recall_intent(self, analyzer):
        """Test classification of decision recall intent."""
        analysis = analyzer.analyze("Why did we decide to use Redis?")
        assert analysis.intent == QueryIntent.DECISION_RECALL

    def test_default_intent(self, analyzer):
        """Test default intent for ambiguous queries."""
        analysis = analyzer.analyze("database configuration")
        assert analysis.intent == QueryIntent.INFORMATION_SEEKING

    def test_extract_key_terms(self, analyzer):
        """Test extraction of key terms."""
        # Use query without words that trigger technology detection
        analysis = analyzer.analyze("authentication login implementation")
        # Should extract meaningful terms, excluding stop words
        assert "authentication" in analysis.key_terms
        assert "login" in analysis.key_terms
        assert "implementation" in analysis.key_terms

    def test_key_terms_exclude_stop_words(self, analyzer):
        """Test that stop words are excluded from key terms."""
        analysis = analyzer.analyze("how to implement the authentication")
        # Stop words like "how", "to", "the" should not be in key terms
        assert "how" not in analysis.key_terms
        assert "the" not in analysis.key_terms
        assert "to" not in analysis.key_terms

    def test_file_specific_detection(self, analyzer):
        """Test detection of file-specific queries."""
        analysis = analyzer.analyze("Changes to config.py")
        assert analysis.is_file_specific is True
        assert "config.py" in analysis.extracted_file_paths

    def test_file_path_extraction(self, analyzer):
        """Test extraction of file paths from query."""
        analysis = analyzer.analyze("What happened to src/auth/login.ts?")
        assert analysis.is_file_specific is True
        assert any("login.ts" in path for path in analysis.extracted_file_paths)

    def test_multiple_file_references(self, analyzer):
        """Test extraction of multiple file references."""
        analysis = analyzer.analyze("Changes to main.py and config.json")
        assert len(analysis.extracted_file_paths) >= 2

    def test_confidence_increases_with_specificity(self, analyzer):
        """Test that confidence is higher for more specific queries."""
        vague_analysis = analyzer.analyze("stuff")
        specific_analysis = analyzer.analyze(
            "How to fix the Python authentication error in Django?"
        )

        assert specific_analysis.confidence > vague_analysis.confidence

    def test_suggest_refinements_for_vague_query(self, analyzer):
        """Test refinement suggestions for vague queries."""
        analysis = analyzer.analyze("authentication")
        suggestions = analyzer.suggest_refinements("authentication", analysis)

        assert len(suggestions) > 0

    def test_suggest_refinements_for_how_to(self, analyzer):
        """Test refinement suggestions for how-to queries."""
        analysis = analyzer.analyze("How to use caching")
        suggestions = analyzer.suggest_refinements("How to use caching", analysis)

        # Should suggest adding "example" or "steps"
        assert any("example" in s or "steps" in s for s in suggestions)

    def test_normalized_query_lowercase(self, analyzer):
        """Test that normalized query is lowercase."""
        analysis = analyzer.analyze("PYTHON Django AUTHENTICATION")
        assert analysis.normalized_query == "python django authentication"

    def test_normalized_query_trimmed(self, analyzer):
        """Test that normalized query is trimmed."""
        analysis = analyzer.analyze("  test query  ")
        assert analysis.normalized_query == "test query"

    def test_technology_detection_case_insensitive(self, analyzer):
        """Test that technology detection is case insensitive."""
        analysis1 = analyzer.analyze("PYTHON")
        analysis2 = analyzer.analyze("python")
        analysis3 = analyzer.analyze("Python")

        assert analysis1.detected_technologies == analysis2.detected_technologies
        assert analysis2.detected_technologies == analysis3.detected_technologies

    def test_word_boundary_matching(self, analyzer):
        """Test that technology detection uses word boundaries."""
        # "go" should not match "google"
        analysis = analyzer.analyze("search using google")
        assert "go" not in analysis.detected_technologies

    def test_typescript_distinct_from_javascript(self, analyzer):
        """Test that TypeScript is detected separately from JavaScript."""
        ts_analysis = analyzer.analyze("TypeScript interfaces")
        js_analysis = analyzer.analyze("JavaScript functions")

        assert "typescript" in ts_analysis.detected_technologies
        assert "javascript" in js_analysis.detected_technologies
