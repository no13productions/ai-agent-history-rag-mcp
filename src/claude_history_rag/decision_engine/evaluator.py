"""Search result evaluation and relevance scoring.

Evaluates search results to determine if they adequately answer the query,
and identifies when query refinement might improve results.
"""

import logging
import math
import re
import threading
from typing import Any

from claude_history_rag.decision_engine.models import (
    EvaluationResult,
    QueryAnalysis,
    QueryIntent,
)

logger = logging.getLogger(__name__)

# Pre-compiled regex for word tokenization (P2 fix)
_WORD_PATTERN = re.compile(r"\b\w+\b")


class ResultEvaluator:
    """Evaluates search result quality and relevance.

    Uses heuristic scoring without requiring external LLM calls,
    making it fast and reliable for the MCP pipeline.
    """

    def __init__(
        self,
        adequacy_threshold: float = 0.5,
        completeness_threshold: float = 0.4,
        min_results_for_good_score: int = 3,
    ):
        """Initialize result evaluator.

        Args:
            adequacy_threshold: Minimum relevance score to consider results adequate
            completeness_threshold: Minimum completeness score before suggesting refinement
            min_results_for_good_score: Minimum results needed for high completeness
        """
        if not 0.0 <= adequacy_threshold <= 1.0:
            raise ValueError("adequacy_threshold must be between 0.0 and 1.0")
        if not 0.0 <= completeness_threshold <= 1.0:
            raise ValueError("completeness_threshold must be between 0.0 and 1.0")
        if min_results_for_good_score < 1:
            raise ValueError("min_results_for_good_score must be at least 1")

        self.adequacy_threshold = adequacy_threshold
        self.completeness_threshold = completeness_threshold
        self.min_results_for_good_score = min_results_for_good_score

        logger.debug(
            f"ResultEvaluator initialized: adequacy={adequacy_threshold}, "
            f"completeness={completeness_threshold}"
        )

    def evaluate(
        self,
        query: str,
        results: list[dict[str, Any]],
        analysis: QueryAnalysis | None = None,
    ) -> EvaluationResult:
        """Evaluate search results against the query.

        Args:
            query: Original search query
            results: Search results to evaluate
            analysis: Optional pre-computed query analysis

        Returns:
            EvaluationResult with scores and recommendations
        """
        logger.debug(f"evaluate() called: result_count={len(results)}")

        if not results:
            logger.debug("Early return: no results to evaluate")
            return EvaluationResult(
                relevance_score=0.0,
                completeness_score=0.0,
                confidence=0.9,  # High confidence that empty results are bad
                needs_refinement=True,
                refinement_suggestions=["Try broader search terms"],
                missing_information=["No results found"],
                result_count=0,
                avg_result_score=0.0,
            )

        # Calculate individual result scores
        result_scores = [self._score_single_result(query, result, analysis) for result in results]

        # Aggregate scores
        avg_score = sum(result_scores) / len(result_scores)

        # Calculate relevance (weighted by score position)
        # Top results matter more
        weighted_scores = []
        for i, score in enumerate(result_scores):
            weight = 1.0 / (i + 1)  # 1.0, 0.5, 0.33, 0.25, ...
            weighted_scores.append(score * weight)

        total_weight = sum(1.0 / (i + 1) for i in range(len(result_scores)))
        relevance_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0

        # Calculate completeness based on result count and score distribution
        completeness = self._calculate_completeness(results, result_scores, analysis)

        # Determine if refinement is needed
        needs_refinement = (
            relevance_score < self.adequacy_threshold or completeness < self.completeness_threshold
        )

        # Generate refinement suggestions if needed
        refinement_suggestions = []
        missing_info = []

        if needs_refinement:
            refinement_suggestions, missing_info = self._generate_suggestions(
                query, results, result_scores, analysis
            )

        # Calculate confidence in evaluation
        confidence = self._calculate_confidence(len(results), result_scores, analysis)

        evaluation = EvaluationResult(
            relevance_score=relevance_score,
            completeness_score=completeness,
            confidence=confidence,
            needs_refinement=needs_refinement,
            refinement_suggestions=refinement_suggestions,
            missing_information=missing_info,
            result_count=len(results),
            avg_result_score=avg_score,
        )

        logger.debug(
            f"Evaluation: relevance={relevance_score:.2f}, "
            f"completeness={completeness:.2f}, needs_refinement={needs_refinement}"
        )

        return evaluation

    def _score_single_result(
        self,
        query: str,
        result: dict[str, Any],
        analysis: QueryAnalysis | None,
    ) -> float:
        """Score a single search result against the query.

        Args:
            query: Original query
            result: Single search result
            analysis: Optional query analysis

        Returns:
            Score from 0.0 to 1.0
        """
        score = 0.0
        content = result.get("content", "").lower()
        query_lower = query.lower()

        # Base score from search distance (if available)
        # Lower distance = better match
        distance_score = result.get("score", 0.5)
        # Validate distance_score is numeric (D1 fix - log warning on type coercion)
        if not isinstance(distance_score, (int, float)):
            result_id = result.get("id", "unknown")
            result_keys = list(result.keys())
            score_type = type(distance_score).__name__
            score_val = str(distance_score)[:50]
            logger.error(
                f"Invalid score type in result {result_id}: "
                f"expected numeric, got {score_type}, value={score_val}. "
                f"Result keys: {result_keys}"
            )
            distance_score = 0.5
        elif math.isnan(distance_score) or math.isinf(distance_score):
            result_id = result.get("id", "unknown")
            result_keys = list(result.keys())
            logger.error(
                f"Invalid score value in result {result_id}: "
                f"score is {distance_score} (NaN or Inf). "
                f"Result keys: {result_keys}"
            )
            distance_score = 0.5
        # Clip distance to valid range [0, 1] before converting to similarity
        # LanceDB can return values slightly outside this range
        distance_score = max(0.0, min(1.0, distance_score))
        similarity = 1.0 - distance_score
        score += similarity * 0.4

        # Term overlap score (using pre-compiled pattern)
        query_terms = set(_WORD_PATTERN.findall(query_lower))
        content_terms = set(_WORD_PATTERN.findall(content))

        if query_terms:
            overlap = len(query_terms & content_terms)
            term_score = overlap / len(query_terms)
            score += term_score * 0.3

        # Technology match bonus (if analysis available)
        if analysis and analysis.detected_technologies:
            for tech in analysis.detected_technologies:
                if tech.lower() in content:
                    score += 0.1
                    break

        # Intent-specific scoring
        if analysis:
            score += self._score_by_intent(content, analysis.intent) * 0.2

        # Chunk type relevance
        chunk_type = result.get("chunk_type", "")
        if analysis and analysis.intent == QueryIntent.FILE_HISTORY:
            if chunk_type == "file_change":
                score += 0.1
        elif chunk_type == "turn":
            score += 0.05  # Turn chunks are generally useful

        # Note: Weights intentionally sum > 1.0 to allow bonuses to boost scores.
        # The final score is clamped to 1.0 at the end.
        return min(1.0, score)

    def _score_by_intent(self, content: str, intent: QueryIntent) -> float:
        """Score content based on query intent.

        Args:
            content: Result content
            intent: Query intent

        Returns:
            Intent-specific score component
        """
        content_lower = content.lower()

        if intent == QueryIntent.HOW_TO:
            # Look for instructional language
            how_to_patterns = ["step", "first", "then", "next", "finally", "to do this"]
            matches = sum(1 for p in how_to_patterns if p in content_lower)
            return min(1.0, matches / 3)

        elif intent == QueryIntent.PROBLEM_SOLVING:
            # Look for solution-oriented language
            solution_patterns = ["fix", "solution", "resolved", "workaround", "the issue"]
            matches = sum(1 for p in solution_patterns if p in content_lower)
            return min(1.0, matches / 2)

        elif intent == QueryIntent.CODE_EXAMPLE:
            # Look for code indicators
            code_patterns = ["```", "def ", "function ", "class ", "const ", "let ", "var "]
            matches = sum(1 for p in code_patterns if p in content_lower or p in content)
            return min(1.0, matches / 2)

        elif intent == QueryIntent.FILE_HISTORY:
            # Look for file change language
            file_patterns = ["edit", "change", "modify", "update", "write", "file_path"]
            matches = sum(1 for p in file_patterns if p in content_lower)
            return min(1.0, matches / 2)

        elif intent == QueryIntent.SESSION_CONTEXT:
            # Look for session/temporal references
            session_patterns = ["session", "earlier", "previous", "before", "we discussed"]
            matches = sum(1 for p in session_patterns if p in content_lower)
            return min(1.0, matches / 2)

        elif intent == QueryIntent.DECISION_RECALL:
            # Look for decision language
            decision_patterns = ["decided", "chose", "because", "reason", "approach"]
            matches = sum(1 for p in decision_patterns if p in content_lower)
            return min(1.0, matches / 2)

        # Default: no specific scoring
        return 0.5

    def _calculate_completeness(
        self,
        results: list[dict[str, Any]],
        scores: list[float],
        analysis: QueryAnalysis | None,
    ) -> float:
        """Calculate completeness score for result set.

        Args:
            results: Search results
            scores: Individual result scores
            analysis: Query analysis

        Returns:
            Completeness score from 0.0 to 1.0
        """
        if not results:
            return 0.0

        # Factor 1: Number of results
        count_factor = min(1.0, len(results) / self.min_results_for_good_score)

        # Factor 2: Score quality (what fraction of results are good?)
        good_results = sum(1 for s in scores if s >= 0.5)
        quality_factor = good_results / len(results) if results else 0

        # Factor 3: Score spread (low variance = consistent quality)
        if len(scores) > 1:
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            # Lower variance is better (more consistent)
            spread_factor = 1.0 - min(1.0, variance * 2)
        else:
            spread_factor = 0.5  # Neutral for single result

        # Weight the factors
        completeness = count_factor * 0.3 + quality_factor * 0.5 + spread_factor * 0.2

        return min(1.0, completeness)

    def _generate_suggestions(
        self,
        query: str,
        results: list[dict[str, Any]],
        scores: list[float],
        analysis: QueryAnalysis | None,
    ) -> tuple[list[str], list[str]]:
        """Generate refinement suggestions and identify missing information.

        Args:
            query: Original query
            results: Search results
            scores: Individual result scores
            analysis: Query analysis

        Returns:
            Tuple of (refinement suggestions, missing information)
        """
        suggestions = []
        missing = []

        # If no good results, suggest broadening
        if not results or max(scores) < 0.3:
            suggestions.append("Try more general search terms")
            missing.append("No closely matching content found")
            return suggestions, missing

        # If some results but low relevance, suggest specificity
        avg_score = sum(scores) / len(scores)
        if avg_score < 0.5:
            suggestions.append("Add more specific keywords")

        # Intent-specific suggestions
        if (
            analysis
            and analysis.intent == QueryIntent.CODE_EXAMPLE
            and not any("```" in r.get("content", "") for r in results)
        ):
            suggestions.append("Add 'example' or 'code' to query")
            missing.append("Code examples")

        if analysis and analysis.intent == QueryIntent.HOW_TO:
            suggestions.append("Try adding 'steps' or 'guide'")

        if (
            analysis
            and analysis.intent == QueryIntent.FILE_HISTORY
            and not any(r.get("chunk_type") == "file_change" for r in results)
        ):
            missing.append("File change history")
            suggestions.append("Use search_file_changes tool instead")

        # If detected technologies not in results, suggest adding them
        if analysis and analysis.detected_technologies:
            result_content = " ".join(r.get("content", "") for r in results).lower()
            for tech in analysis.detected_technologies:
                if tech not in result_content:
                    missing.append(f"Information about {tech}")

        return suggestions[:3], missing[:3]  # Limit to top 3

    def _calculate_confidence(
        self,
        result_count: int,
        scores: list[float],
        analysis: QueryAnalysis | None,
    ) -> float:
        """Calculate confidence in the evaluation.

        Args:
            result_count: Number of results
            scores: Individual result scores
            analysis: Query analysis

        Returns:
            Confidence score from 0.0 to 1.0
        """
        # More results = more confident in evaluation
        count_confidence = min(1.0, result_count / 5)

        # Higher average scores = more confident
        avg_score = sum(scores) / len(scores) if scores else 0
        score_confidence = avg_score

        # Query analysis confidence contributes
        analysis_confidence = analysis.confidence if analysis else 0.5

        confidence = count_confidence * 0.3 + score_confidence * 0.4 + analysis_confidence * 0.3

        return min(1.0, confidence)

    def should_refine(self, evaluation: EvaluationResult) -> bool:
        """Determine if query refinement is recommended.

        Args:
            evaluation: Evaluation result

        Returns:
            True if refinement is recommended
        """
        return evaluation.needs_refinement and len(evaluation.refinement_suggestions) > 0


# Global evaluator instance (lazy initialization with thread safety)
_global_evaluator: ResultEvaluator | None = None
_global_evaluator_lock = threading.Lock()


def get_result_evaluator(
    adequacy_threshold: float = 0.5,
    completeness_threshold: float = 0.4,
) -> ResultEvaluator:
    """Get or create the global result evaluator instance.

    Uses double-check locking pattern for thread-safe lazy initialization.

    Args:
        adequacy_threshold: Threshold for adequacy (only used on first call)
        completeness_threshold: Threshold for completeness (only used on first call)

    Returns:
        Global ResultEvaluator instance
    """
    global _global_evaluator
    if _global_evaluator is None:
        with _global_evaluator_lock:
            # Double-check after acquiring lock
            if _global_evaluator is None:
                _global_evaluator = ResultEvaluator(
                    adequacy_threshold=adequacy_threshold,
                    completeness_threshold=completeness_threshold,
                )
    return _global_evaluator


def reset_result_evaluator() -> None:
    """Reset global result evaluator for testing."""
    global _global_evaluator
    with _global_evaluator_lock:
        _global_evaluator = None
