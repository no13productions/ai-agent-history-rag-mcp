"""Decision Engine - Main orchestrator for intelligent search.

Coordinates query analysis, caching, evaluation, and synthesis
to provide enhanced search capabilities.
"""

import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from claude_history_rag.decision_engine.cache import SearchCache, get_search_cache
from claude_history_rag.decision_engine.evaluator import (
    ResultEvaluator,
    get_result_evaluator,
)
from claude_history_rag.decision_engine.models import (
    DecisionMetrics,
    EvaluationResult,
    QueryAnalysis,
    SynthesizedResult,
)
from claude_history_rag.decision_engine.query_analyzer import (
    QueryAnalyzer,
    get_query_analyzer,
)
from claude_history_rag.decision_engine.synthesizer import (
    ResultSynthesizer,
    get_result_synthesizer,
)

logger = logging.getLogger(__name__)


# Type alias for search function
SearchFunc = Callable[[str, list[float], int, str | None], Awaitable[list[dict[str, Any]]]]
EmbedFunc = Callable[[str], Awaitable[list[float]]]


class DecisionEngine:
    """Orchestrates the intelligent search pipeline.

    Implements a multi-stage decision process:
    1. Query Analysis - Understand intent and extract context
    2. Cache Check - Return cached results if available
    3. Search Execution - Execute vector/hybrid search
    4. Result Evaluation - Assess result quality
    5. Query Refinement - Retry with improved query if needed
    6. Result Synthesis - Combine and deduplicate results

    All stages are designed to be fast and work without external LLM calls.
    """

    def __init__(
        self,
        cache: SearchCache | None = None,
        analyzer: QueryAnalyzer | None = None,
        evaluator: ResultEvaluator | None = None,
        synthesizer: ResultSynthesizer | None = None,
        enable_cache: bool = True,
        enable_refinement: bool = True,
        enable_synthesis: bool = True,
        max_refinement_attempts: int = 1,
    ):
        """Initialize decision engine.

        Args:
            cache: Search cache (uses global instance if None)
            analyzer: Query analyzer (uses global instance if None)
            evaluator: Result evaluator (uses global instance if None)
            synthesizer: Result synthesizer (uses global instance if None)
            enable_cache: Whether to use caching
            enable_refinement: Whether to enable query refinement
            enable_synthesis: Whether to synthesize results
            max_refinement_attempts: Maximum refinement iterations
        """
        self.cache = cache if cache is not None else (get_search_cache() if enable_cache else None)
        self.analyzer = analyzer if analyzer is not None else get_query_analyzer()
        self.evaluator = evaluator if evaluator is not None else get_result_evaluator()
        self.synthesizer = synthesizer if synthesizer is not None else get_result_synthesizer()

        self.enable_cache = enable_cache and self.cache is not None
        self.enable_refinement = enable_refinement
        self.enable_synthesis = enable_synthesis
        self.max_refinement_attempts = max(0, max_refinement_attempts)

        logger.info(
            f"DecisionEngine initialized: cache={self.enable_cache}, "
            f"refinement={self.enable_refinement}, synthesis={self.enable_synthesis}"
        )

    async def search(
        self,
        query: str,
        search_func: SearchFunc,
        embed_func: EmbedFunc,
        limit: int = 5,
        project_filter: str | None = None,
        search_type: str = "hybrid",
        enable_synthesis: bool | None = None,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        """Execute intelligent search pipeline.

        Args:
            query: Search query
            search_func: Async function to execute search (query, vector, limit, filter) -> results
            embed_func: Async function to embed query -> vector
            limit: Maximum results
            project_filter: Optional project path filter
            search_type: Type of search to execute
            enable_synthesis: Override for synthesis setting (T2 fix - avoids global mutation).
                If None, uses instance default.
            include_debug: Include detailed metrics in response (U1/U4 fix - reduces noise
                in production). Default False for cleaner responses.

        Returns:
            Dict with results, metadata, and optional analysis/synthesis.
            When include_debug=True, also includes detailed metrics.
        """
        # E1 fix: Validate query is not empty
        if not query or not query.strip():
            return {
                "results": [],
                "count": 0,
                "query": query,
                "search_type": search_type,
                "cache_hit": False,
                "error": "Query cannot be empty",
            }

        # E2 fix: Validate query length (10,000 char limit)
        if len(query) > 10000:
            return {
                "results": [],
                "count": 0,
                "query": query[:100] + "...",
                "search_type": search_type,
                "cache_hit": False,
                "error": f"Query too long: {len(query)} chars (max 10,000)",
            }

        # Use parameter if provided, otherwise fall back to instance setting (T2 fix)
        do_synthesis = enable_synthesis if enable_synthesis is not None else self.enable_synthesis
        start_time = time.time()
        metrics = DecisionMetrics()

        # Stage 1: Query Analysis
        analysis_start = time.time()
        analysis = self.analyzer.analyze(query)
        metrics.query_analysis_ms = int((time.time() - analysis_start) * 1000)
        metrics.decisions_made.append(f"intent:{analysis.intent.value}")

        # Stage 2: Cache Check
        if self.enable_cache:
            cache_start = time.time()
            cached_results = await self.cache.get_cached_results(
                query=query,
                project_filter=project_filter,
                search_type=search_type,
                limit=limit,
            )
            metrics.cache_check_ms = int((time.time() - cache_start) * 1000)

            if cached_results is not None:
                metrics.cache_hit = True
                metrics.total_ms = int((time.time() - start_time) * 1000)
                metrics.result_count = len(cached_results)
                metrics.decisions_made.append("cache:hit")

                logger.debug(f"Cache hit for query: {query[:50]}...")

                # Re-evaluate cached results for schema consistency
                # Note: Evaluations are always fresh, even for cached results. If evaluator
                # thresholds change between caching and retrieval, the evaluation may differ
                # from when results were originally cached. This is intentional.
                evaluation = self.evaluator.evaluate(query, cached_results, analysis)

                response = {
                    "results": cached_results,
                    "count": len(cached_results),
                    "query": query,
                    "search_type": search_type,
                    "cache_hit": True,
                    "analysis": self._analysis_to_dict(analysis),
                    "evaluation": self._evaluation_to_dict(evaluation),
                }
                # U1/U4 fix: Only include metrics when debug mode requested
                if include_debug:
                    response["metrics"] = metrics.model_dump()
                return response

            metrics.decisions_made.append("cache:miss")

        # Stage 3: Search Execution
        search_start = time.time()
        query_vector = await embed_func(query)
        results = await search_func(query, query_vector, limit, project_filter)
        metrics.search_ms = int((time.time() - search_start) * 1000)
        metrics.result_count = len(results)

        # Stage 4: Result Evaluation
        eval_start = time.time()
        evaluation = self.evaluator.evaluate(query, results, analysis)
        metrics.evaluation_ms = int((time.time() - eval_start) * 1000)
        metrics.decisions_made.append(f"relevance:{evaluation.relevance_score:.2f}")

        # Stage 5: Query Refinement (if enabled and needed)
        refined_query = None
        if (
            self.enable_refinement
            and evaluation.needs_refinement
            and evaluation.refinement_suggestions
            and self.max_refinement_attempts > 0
        ):
            metrics.refinement_triggered = True
            metrics.decisions_made.append("refinement:triggered")

            # Try refinement
            refined_results, refined_query = await self._try_refinement(
                original_query=query,
                original_results=results,
                original_evaluation=evaluation,
                analysis=analysis,
                search_func=search_func,
                embed_func=embed_func,
                limit=limit,
                project_filter=project_filter,
            )

            if refined_results is not None:
                # Check if refinement improved results
                refined_eval = self.evaluator.evaluate(
                    refined_query or query, refined_results, analysis
                )
                if refined_eval.relevance_score > evaluation.relevance_score:
                    results = refined_results
                    evaluation = refined_eval
                    metrics.refinement_improved = True
                    metrics.result_count = len(results)
                    metrics.decisions_made.append("refinement:improved")
                else:
                    metrics.decisions_made.append("refinement:no_improvement")
            else:
                metrics.decisions_made.append("refinement:failed")

        # Stage 6: Result Synthesis (if enabled)
        synthesis = None
        if do_synthesis and results:
            synthesis_start = time.time()
            synthesis = self.synthesizer.synthesize(query, results, analysis)
            metrics.synthesis_ms = int((time.time() - synthesis_start) * 1000)
            metrics.decisions_made.append(f"synthesis:{synthesis.synthesis_method}")

        # Cache results (if caching enabled)
        if self.enable_cache and results:
            await self.cache.cache_results(
                query=query,
                results=results,
                project_filter=project_filter,
                search_type=search_type,
                limit=limit,
            )

        # Finalize metrics
        metrics.total_ms = int((time.time() - start_time) * 1000)

        # Build response (U1/U4 fix: cleaner response structure)
        response = {
            "results": results,
            "count": len(results),
            "query": query,
            "search_type": search_type,
            "cache_hit": False,
            "analysis": self._analysis_to_dict(analysis),
            "evaluation": self._evaluation_to_dict(evaluation),
        }

        # U2 note: When synthesis is enabled, primary_content may overlap with results.
        # This is intentional - synthesis provides a structured summary while results
        # preserve the original chunks for reference. Clients can choose which to use.
        if synthesis:
            response["synthesis"] = self._synthesis_to_dict(synthesis)

        if refined_query and refined_query != query:
            response["refined_query"] = refined_query

        # U1/U4 fix: Only include detailed metrics when debug mode requested
        if include_debug:
            response["metrics"] = metrics.model_dump()

        logger.debug(
            f"Search completed in {metrics.total_ms}ms: "
            f"{len(results)} results, cache_hit={metrics.cache_hit}, "
            f"refinement={metrics.refinement_triggered}, synthesis={do_synthesis}, "
            f"decisions={metrics.decisions_made}"
        )

        return response

    async def _try_refinement(
        self,
        original_query: str,
        original_results: list[dict[str, Any]],
        original_evaluation: EvaluationResult,
        analysis: QueryAnalysis,
        search_func: SearchFunc,
        embed_func: EmbedFunc,
        limit: int,
        project_filter: str | None,
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        """Attempt to refine the query for better results.

        Args:
            original_query: Original search query
            original_results: Original search results
            original_evaluation: Evaluation of original results
            analysis: Query analysis
            search_func: Search function
            embed_func: Embedding function
            limit: Result limit
            project_filter: Project filter

        Returns:
            Tuple of (refined results or None, refined query or None)
        """
        # Get refinement suggestions
        suggestions = self.analyzer.suggest_refinements(original_query, analysis)

        if not suggestions:
            return None, None

        # Try the first suggestion
        refined_query = suggestions[0]

        try:
            # Execute refined search
            refined_vector = await embed_func(refined_query)
            refined_results = await search_func(
                refined_query, refined_vector, limit, project_filter
            )

            if refined_results:
                logger.debug(f"Refinement produced {len(refined_results)} results")
                return refined_results, refined_query

        except Exception as e:
            # L2 fix: Add exc_info for better debugging
            logger.warning(f"Query refinement failed: {e}", exc_info=True)

        return None, None

    def _analysis_to_dict(self, analysis: QueryAnalysis) -> dict[str, Any]:
        """Convert QueryAnalysis to dict for response.

        Args:
            analysis: Query analysis object

        Returns:
            Dictionary representation
        """
        return {
            "intent": analysis.intent.value,
            "detected_technologies": analysis.detected_technologies,
            "key_terms": analysis.key_terms,
            "is_file_specific": analysis.is_file_specific,
            "confidence": analysis.confidence,
            "extracted_file_paths": analysis.extracted_file_paths,
        }

    def _evaluation_to_dict(self, evaluation: EvaluationResult) -> dict[str, Any]:
        """Convert EvaluationResult to dict for response.

        Args:
            evaluation: Evaluation result object

        Returns:
            Dictionary representation including refinement suggestions (U3 fix)
        """
        result = {
            "relevance_score": evaluation.relevance_score,
            "completeness_score": evaluation.completeness_score,
            "confidence": evaluation.confidence,
            "needs_refinement": evaluation.needs_refinement,
            "result_count": evaluation.result_count,
            "avg_result_score": evaluation.avg_result_score,
        }
        # Include actionable refinement info when refinement is needed (U3 fix)
        if evaluation.needs_refinement:
            if evaluation.refinement_suggestions:
                result["refinement_suggestions"] = evaluation.refinement_suggestions
            if evaluation.missing_information:
                result["missing_information"] = evaluation.missing_information
        return result

    def _synthesis_to_dict(self, synthesis: SynthesizedResult) -> dict[str, Any]:
        """Convert SynthesizedResult to dict for response.

        Args:
            synthesis: Synthesis result object

        Returns:
            Dictionary representation
        """
        return {
            "primary_content": synthesis.primary_content,
            "key_points": synthesis.key_points,
            "code_snippets": synthesis.code_snippets,
            "file_changes": synthesis.file_changes,
            "sources": [
                {
                    "session_id": s.session_id,
                    "chunk_id": s.chunk_id,
                    "chunk_type": s.chunk_type,
                    "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                    "relevance_score": s.relevance_score,
                }
                for s in synthesis.sources
            ],
            "source_count": len(synthesis.sources),
            "confidence": synthesis.overall_confidence,
            "synthesis_method": synthesis.synthesis_method,
            "deduplication_count": synthesis.deduplication_count,
        }

    async def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics if caching is enabled.

        Returns:
            Cache statistics or None if caching disabled
        """
        if not self.enable_cache or not self.cache:
            return None
        return self.cache.get_stats()

    async def clear_cache(self) -> int:
        """Clear all cached results.

        Returns:
            Number of entries cleared, or 0 if caching disabled
        """
        if not self.enable_cache or not self.cache:
            return 0
        return await self.cache.clear_all()


# Global engine instance (lazy initialization with thread safety - T1/I1 fix)
_global_engine: DecisionEngine | None = None
_global_engine_lock = threading.Lock()


def get_decision_engine(
    enable_cache: bool = True,
    enable_refinement: bool = True,
    enable_synthesis: bool = True,
) -> DecisionEngine:
    """Get or create the global decision engine instance.

    Uses double-check locking pattern for thread-safe lazy initialization.
    (T1/I1 fix: centralized instance with proper thread safety)

    Args:
        enable_cache: Enable caching (only used on first call)
        enable_refinement: Enable refinement (only used on first call)
        enable_synthesis: Enable synthesis (only used on first call)

    Returns:
        Global DecisionEngine instance
    """
    global _global_engine
    if _global_engine is None:
        with _global_engine_lock:
            # Double-check after acquiring lock
            if _global_engine is None:
                _global_engine = DecisionEngine(
                    enable_cache=enable_cache,
                    enable_refinement=enable_refinement,
                    enable_synthesis=enable_synthesis,
                )
    return _global_engine
