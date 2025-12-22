"""Data models for the decision engine.

These models represent the core data structures used throughout the decision
engine pipeline, from query analysis through result synthesis.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    """Classification of query intent."""

    INFORMATION_SEEKING = "information_seeking"  # Looking for general information
    PROBLEM_SOLVING = "problem_solving"  # Debugging or troubleshooting
    HOW_TO = "how_to"  # Step-by-step guidance
    REFERENCE = "reference"  # API/documentation lookup
    CODE_EXAMPLE = "code_example"  # Looking for code samples
    FILE_HISTORY = "file_history"  # Looking for file change history
    SESSION_CONTEXT = "session_context"  # Looking for session/conversation context
    DECISION_RECALL = "decision_recall"  # Recalling past decisions made


class QueryAnalysis(BaseModel):
    """Result of query analysis.

    Contains extracted information about the query including intent,
    technology context, and key terms.
    """

    original_query: str = Field(description="The original query string")
    normalized_query: str = Field(description="Lowercase, trimmed query")
    intent: QueryIntent = Field(
        default=QueryIntent.INFORMATION_SEEKING,
        description="Classified intent of the query",
    )
    detected_technologies: list[str] = Field(
        default_factory=list,
        description="Technologies detected in the query",
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description="Important terms extracted from query",
    )
    is_file_specific: bool = Field(
        default=False,
        description="Whether query is about specific files",
    )
    extracted_file_paths: list[str] = Field(
        default_factory=list,
        description="File paths mentioned in query",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the analysis",
    )


class EvaluationResult(BaseModel):
    """Result of search result evaluation.

    Assesses the quality and relevance of search results to determine
    if they adequately answer the query or need refinement.
    """

    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall relevance of results to query",
    )
    completeness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How completely results answer the query",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the evaluation",
    )
    needs_refinement: bool = Field(
        default=False,
        description="Whether query should be refined for better results",
    )
    refinement_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested terms to add to query",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Information gaps identified",
    )
    result_count: int = Field(
        default=0,
        description="Number of results evaluated",
    )
    avg_result_score: float = Field(
        default=0.0,
        description="Average score of individual results",
    )


class SourceAttribution(BaseModel):
    """Attribution for a piece of synthesized content."""

    session_id: str = Field(description="Source session ID")
    chunk_id: str = Field(description="Source chunk ID")
    chunk_type: str = Field(description="Type of source chunk")
    timestamp: datetime | None = Field(default=None, description="When source was created")
    relevance_score: float = Field(default=0.0, description="Relevance to query")


class SynthesizedResult(BaseModel):
    """Result of multi-document synthesis.

    Combines multiple search results into a coherent, deduplicated response
    with proper source attribution.
    """

    primary_content: str = Field(
        default="",
        description="Main synthesized content",
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Key points extracted from results",
    )
    code_snippets: list[dict[str, str]] = Field(
        default_factory=list,
        description="Code snippets with language and content",
    )
    file_changes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Relevant file changes found",
    )
    sources: list[SourceAttribution] = Field(
        default_factory=list,
        description="Source attributions",
    )
    overall_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in synthesis quality",
    )
    synthesis_method: str = Field(
        default="simple",
        description="Method used for synthesis (simple, dedup, ai)",
    )
    deduplication_count: int = Field(
        default=0,
        description="Number of duplicate results removed",
    )


class CacheEntry(BaseModel):
    """A cached search result entry.

    Stores search results with metadata for cache management including
    TTL and access tracking.
    """

    cache_key: str = Field(description="Unique cache key")
    query: str = Field(description="Original query")
    results: list[dict[str, Any]] = Field(description="Cached results")
    result_count: int = Field(description="Number of results")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When entry was created",
    )
    expires_at: datetime = Field(description="When entry expires")
    access_count: int = Field(default=0, description="Number of times accessed")
    last_accessed: datetime | None = Field(
        default=None,
        description="Last access timestamp",
    )
    search_type: str = Field(default="hybrid", description="Type of search used")
    project_filter: str | None = Field(default=None, description="Project filter if any")


class DecisionMetrics(BaseModel):
    """Metrics for decision engine performance.

    Tracks timing and decision outcomes for monitoring and optimization.
    """

    query_analysis_ms: int = Field(default=0, description="Time for query analysis")
    cache_check_ms: int = Field(default=0, description="Time for cache check")
    search_ms: int = Field(default=0, description="Time for search execution")
    evaluation_ms: int = Field(default=0, description="Time for result evaluation")
    synthesis_ms: int = Field(default=0, description="Time for result synthesis")
    total_ms: int = Field(default=0, description="Total processing time")
    cache_hit: bool = Field(default=False, description="Whether cache was hit")
    refinement_triggered: bool = Field(
        default=False,
        description="Whether query refinement was triggered",
    )
    refinement_improved: bool = Field(
        default=False,
        description="Whether refinement improved results",
    )
    result_count: int = Field(default=0, description="Final result count")
    decisions_made: list[str] = Field(
        default_factory=list,
        description="List of decisions made during pipeline",
    )
