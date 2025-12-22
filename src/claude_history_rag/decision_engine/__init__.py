"""Decision engine for intelligent search orchestration.

This module provides AI-powered decision making capabilities inspired by
DocAIche's 10-point decision engine, adapted for the AI Agent History RAG MCP server.

Key components:
- QueryAnalyzer: Query understanding and technology detection
- ResultEvaluator: Relevance scoring and adequacy assessment
- ResultSynthesizer: Multi-result fusion and deduplication
- SearchCache: Multi-tier caching with LRU and TTL
- DecisionEngine: Main orchestrator coordinating all components

Response Schema (I2 documentation):
-------------------------------------
The search response structure varies based on flags. Base response always includes:
    - results: list[dict]       # Search results
    - count: int                # Number of results
    - query: str                # Original query
    - search_type: str          # "hybrid" or "vector"
    - cache_hit: bool           # Whether result came from cache

When enable_analysis=True (default), adds:
    - analysis: dict            # Query analysis
        - intent: str                    # Query intent classification
        - detected_technologies: list    # Technologies found in query
        - key_terms: list                # Important extracted terms
        - is_file_specific: bool         # Whether query is about specific files
        - confidence: float              # Analysis confidence (0.0-1.0)
        - extracted_file_paths: list     # File paths mentioned in query
    - evaluation: dict          # Result evaluation
        - relevance_score: float         # Overall relevance (0.0-1.0)
        - completeness_score: float      # Completeness (0.0-1.0)
        - confidence: float              # Evaluation confidence (0.0-1.0)
        - needs_refinement: bool         # Whether query needs refinement
        - result_count: int              # Number of results evaluated
        - avg_result_score: float        # Average score of results
        - refinement_suggestions: list   # (only if needs_refinement=True)
        - missing_information: list      # (only if needs_refinement=True)

When enable_synthesis=True, adds:
    - synthesis: dict           # Multi-result synthesis
        - primary_content: str           # Main synthesized content
        - key_points: list               # Key points extracted
        - code_snippets: list            # Code snippets with language/content
        - file_changes: list             # Relevant file changes
        - sources: list                  # Source attributions (session_id, chunk_id, etc.)
        - source_count: int              # Number of sources
        - confidence: float              # Synthesis confidence (0.0-1.0)
        - synthesis_method: str          # Method used (simple, dedup, ai)
        - deduplication_count: int       # Number of duplicates removed

When include_debug=True, adds:
    - metrics: dict             # Timing and decision tracking data

When query refinement occurs:
    - refined_query: str        # The refined query that was used

On error:
    - error: str                # Error message
"""

from claude_history_rag.decision_engine.cache import SearchCache, get_search_cache
from claude_history_rag.decision_engine.config import DecisionEngineConfig, get_config, reset_config
from claude_history_rag.decision_engine.engine import DecisionEngine, get_decision_engine
from claude_history_rag.decision_engine.evaluator import ResultEvaluator
from claude_history_rag.decision_engine.models import (
    CacheEntry,
    DecisionMetrics,
    EvaluationResult,
    QueryAnalysis,
    QueryIntent,
    SynthesizedResult,
)
from claude_history_rag.decision_engine.query_analyzer import QueryAnalyzer
from claude_history_rag.decision_engine.synthesizer import ResultSynthesizer

__all__ = [
    # Models
    "QueryAnalysis",
    "QueryIntent",
    "EvaluationResult",
    "SynthesizedResult",
    "CacheEntry",
    "DecisionMetrics",
    # Components
    "SearchCache",
    "QueryAnalyzer",
    "ResultEvaluator",
    "ResultSynthesizer",
    "DecisionEngine",
    # Configuration
    "DecisionEngineConfig",
    "get_config",
    "reset_config",
    # Factory functions
    "get_decision_engine",
    "get_search_cache",
]
