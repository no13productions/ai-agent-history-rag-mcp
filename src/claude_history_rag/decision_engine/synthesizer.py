"""Multi-result synthesis and deduplication.

Combines multiple search results into coherent, deduplicated responses
with proper source attribution.
"""

import hashlib
import logging
import re
import threading
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from claude_history_rag.decision_engine.models import (
    QueryAnalysis,
    SourceAttribution,
    SynthesizedResult,
)

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for performance (P5 fix)
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
_NORMALIZE_PATTERN = re.compile(r"[^\w\s]")


class ResultSynthesizer:
    """Synthesizes multiple search results into coherent guidance.

    Handles deduplication, content extraction, and source attribution
    without requiring external LLM calls.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        max_key_points: int = 5,
        max_code_snippets: int = 3,
    ):
        """Initialize result synthesizer.

        Args:
            similarity_threshold: Threshold for considering content as duplicate (0-1)
            max_key_points: Maximum key points to extract
            max_code_snippets: Maximum code snippets to include
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if max_key_points < 1:
            raise ValueError("max_key_points must be at least 1")
        if max_code_snippets < 0:
            raise ValueError("max_code_snippets must be non-negative")

        self.similarity_threshold = similarity_threshold
        self.max_key_points = max_key_points
        self.max_code_snippets = max_code_snippets

        logger.debug(
            f"ResultSynthesizer initialized: similarity={similarity_threshold}, "
            f"max_points={max_key_points}"
        )

    def synthesize(
        self,
        query: str,
        results: list[dict[str, Any]],
        analysis: QueryAnalysis | None = None,
    ) -> SynthesizedResult:
        """Synthesize multiple results into a coherent response.

        Args:
            query: Original search query
            results: Search results to synthesize
            analysis: Optional query analysis for context

        Returns:
            SynthesizedResult with combined content
        """
        logger.debug(f"synthesize() called: result_count={len(results)}")

        if not results:
            return SynthesizedResult(
                primary_content="No results to synthesize.",
                synthesis_method="empty",
                overall_confidence=0.0,
            )

        # Defense-in-depth: limit results to prevent resource exhaustion
        # Callers should enforce limits (server.py enforces max 50), but
        # synthesizer should protect itself against misuse
        if len(results) > 100:
            logger.warning(
                f"Truncating {len(results)} results to 100 for synthesis "
                f"to prevent resource exhaustion"
            )
            results = results[:100]

        # Step 1: Deduplicate results
        unique_results, dedup_count = self._deduplicate(results)

        # Step 2: Extract source attributions
        sources = self._extract_sources(unique_results)

        # Step 3: Extract code snippets
        code_snippets = self._extract_code_snippets(unique_results)

        # Step 4: Extract file changes
        file_changes = self._extract_file_changes(unique_results)

        # Step 5: Extract key points
        key_points = self._extract_key_points(unique_results, analysis)

        # Step 6: Generate primary content
        primary_content = self._generate_primary_content(
            unique_results, key_points, code_snippets, analysis
        )

        # Step 7: Calculate confidence
        confidence = self._calculate_confidence(unique_results, dedup_count, key_points)

        synthesis = SynthesizedResult(
            primary_content=primary_content,
            key_points=key_points,
            code_snippets=code_snippets,
            file_changes=file_changes,
            sources=sources,
            overall_confidence=confidence,
            synthesis_method="dedup" if dedup_count > 0 else "simple",
            deduplication_count=dedup_count,
        )

        logger.debug(
            f"Synthesized {len(results)} results -> "
            f"{len(unique_results)} unique, {len(key_points)} points, "
            f"{len(code_snippets)} snippets"
        )

        return synthesis

    def _deduplicate(self, results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Remove duplicate or highly similar results.

        Uses a two-phase approach for O(n) average case:
        1. Hash-based check for exact/near-exact duplicates (fast)
        2. Similarity check only when hashes collide or for first few results

        Args:
            results: Results to deduplicate

        Returns:
            Tuple of (unique results, number of duplicates removed)
        """
        if not results:
            return [], 0

        unique = []
        seen_hashes: set[str] = set()
        seen_normalized_hashes: set[str] = set()
        duplicates = 0

        # Limit similarity comparisons to avoid O(n²) worst case
        max_similarity_checks = 10

        for result in results:
            content = result.get("content", "")

            # Phase 1: Quick hash check for exact duplicates
            # Use "replace" instead of "surrogateescape" for safe hashing
            content_hash = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()
            if content_hash in seen_hashes:
                duplicates += 1
                continue

            # Phase 2: Normalized hash check for near-duplicates
            normalized = self._normalize_for_comparison(content)
            normalized_hash = hashlib.md5(normalized.encode("utf-8", errors="replace")).hexdigest()
            if normalized_hash in seen_normalized_hashes:
                duplicates += 1
                continue

            # Phase 3: Similarity check only against recent results
            # Bounded similarity check: O(n * max_similarity_checks) where n=len(results)
            # and max_similarity_checks=10, preventing O(n²) worst case
            is_similar = False
            check_range = min(len(unique), max_similarity_checks)
            for existing in unique[-check_range:]:
                existing_content = existing.get("content", "")
                similarity = self._calculate_similarity(content, existing_content)
                if similarity >= self.similarity_threshold:
                    is_similar = True
                    duplicates += 1
                    break

            if not is_similar:
                unique.append(result)
            # Always track hashes to prevent future hash collisions
            seen_hashes.add(content_hash)
            seen_normalized_hashes.add(normalized_hash)

        return unique, duplicates

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts.

        Uses SequenceMatcher for efficient similarity computation.
        Normalizes texts before comparison.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity ratio from 0.0 to 1.0
        """
        # Normalize texts
        norm1 = self._normalize_for_comparison(text1)
        norm2 = self._normalize_for_comparison(text2)

        if not norm1 or not norm2:
            return 0.0

        # Use SequenceMatcher for efficient similarity
        return SequenceMatcher(None, norm1, norm2).ratio()

    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for similarity comparison.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove punctuation (using pre-compiled pattern)
        text = _NORMALIZE_PATTERN.sub("", text)
        return text

    def _extract_sources(self, results: list[dict[str, Any]]) -> list[SourceAttribution]:
        """Extract source attributions from results.

        Args:
            results: Results to extract sources from

        Returns:
            List of SourceAttribution objects
        """
        sources = []
        timestamp_success_count = 0
        timestamp_failure_count = 0

        for result in results:
            timestamp = result.get("timestamp")
            if timestamp and isinstance(timestamp, str):
                try:
                    parsed_ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    # Ensure timezone-aware datetime (add UTC if naive)
                    if parsed_ts.tzinfo is None:
                        from datetime import UTC

                        parsed_ts = parsed_ts.replace(tzinfo=UTC)
                    timestamp = parsed_ts
                    timestamp_success_count += 1
                except ValueError as e:
                    # E3 fix: Log warning when timestamp parsing fails
                    chunk_id = result.get("id", "unknown")
                    session_id = result.get("session_id", "unknown")
                    logger.warning(
                        f"Failed to parse timestamp '{timestamp}' "
                        f"(chunk_id={chunk_id}, session_id={session_id}): {e}"
                    )
                    timestamp = None
                    timestamp_failure_count += 1
                    # Note: timestamp=None is acceptable - SourceAttribution allows None,
                    # and engine.py safely handles None via `s.timestamp.isoformat() if s.timestamp else None`

            source = SourceAttribution(
                session_id=result.get("session_id", "unknown"),
                chunk_id=result.get("id", "unknown"),
                chunk_type=result.get("chunk_type", "unknown"),
                timestamp=timestamp,
                relevance_score=max(
                    0.0, min(1.0, 1.0 - result.get("score", 0.5))
                ),  # Convert distance to relevance
            )
            sources.append(source)

        # Log timestamp parsing statistics
        if timestamp_success_count > 0 or timestamp_failure_count > 0:
            logger.debug(
                f"Timestamp parsing: {timestamp_success_count} success, "
                f"{timestamp_failure_count} failure"
            )

        # Sort by relevance
        sources.sort(key=lambda s: s.relevance_score, reverse=True)

        return sources

    def _extract_code_snippets(self, results: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Extract code snippets from results.

        Args:
            results: Results to extract code from

        Returns:
            List of code snippets with language and content
        """
        snippets = []
        seen_code: set[str] = set()

        # Pattern for fenced code blocks
        code_pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

        for result in results:
            content = result.get("content", "")

            # Find all code blocks
            matches = code_pattern.findall(content)

            for lang, code in matches:
                code = code.strip()
                if not code:
                    continue

                # Skip if we've seen this code
                # Use "replace" instead of "surrogateescape" for safe hashing
                code_hash = hashlib.md5(code.encode("utf-8", errors="replace")).hexdigest()
                if code_hash in seen_code:
                    continue

                seen_code.add(code_hash)

                snippets.append(
                    {
                        "language": lang or "text",
                        "content": code,
                        "source_session": result.get("session_id", "unknown"),
                    }
                )

                if len(snippets) >= self.max_code_snippets:
                    return snippets

        return snippets

    def _extract_file_changes(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract file change information from results.

        Args:
            results: Results to extract file changes from

        Returns:
            List of file change records
        """
        changes = []
        seen_files: set[str] = set()

        for result in results:
            if result.get("chunk_type") != "file_change":
                continue

            file_path = result.get("file_path")
            if not file_path or file_path in seen_files:
                continue

            seen_files.add(file_path)

            changes.append(
                {
                    "file_path": file_path,
                    "operation": result.get("operation", "unknown"),
                    "session_id": result.get("session_id", "unknown"),
                    "timestamp": result.get("timestamp"),
                    "content_preview": self._truncate(result.get("content", ""), 200),
                }
            )

        return changes

    def _extract_key_points(
        self,
        results: list[dict[str, Any]],
        analysis: QueryAnalysis | None,
    ) -> list[str]:
        """Extract key points from results.

        Uses optimized approach to avoid O(n²) similarity comparisons:
        - Limits sentences processed per result
        - Uses hash-based dedup before expensive similarity checks
        - Limits similarity comparisons to recent points

        Args:
            results: Results to extract points from
            analysis: Query analysis for context

        Returns:
            List of key point strings
        """
        points = []
        seen_hashes: set[str] = set()
        seen_normalized: list[str] = []

        # Limit sentences per result to avoid processing huge texts
        max_sentences_per_result = 15
        # Limit similarity checks to avoid O(n²)
        max_similarity_checks = 5

        for result in results:
            content = result.get("content", "")

            # Extract sentences that might be key points
            sentences = self._extract_sentences(content)[:max_sentences_per_result]

            for sentence in sentences:
                # Skip if too short or too long
                if len(sentence) < 20 or len(sentence) > 300:
                    continue

                # Quick hash-based dedup
                normalized = self._normalize_for_comparison(sentence)
                norm_hash = hashlib.md5(normalized.encode("utf-8", errors="replace")).hexdigest()
                if norm_hash in seen_hashes:
                    continue

                # Only check similarity against most recent points (bounded)
                is_similar = False
                check_range = min(len(seen_normalized), max_similarity_checks)
                for existing in seen_normalized[-check_range:]:
                    if self._calculate_similarity(normalized, existing) > self.similarity_threshold:
                        is_similar = True
                        break
                if is_similar:
                    continue

                # Score the sentence for relevance
                score = self._score_sentence(sentence, analysis)
                if score < 0.3:
                    continue

                seen_hashes.add(norm_hash)
                seen_normalized.append(normalized)
                points.append(sentence)

                if len(points) >= self.max_key_points:
                    return points

        return points

    def _extract_sentences(self, text: str) -> list[str]:
        """Extract sentences from text.

        Args:
            text: Text to extract sentences from

        Returns:
            List of sentences
        """
        # Simple sentence splitting (using pre-compiled pattern)
        sentences = _SENTENCE_SPLIT_PATTERN.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _score_sentence(self, sentence: str, analysis: QueryAnalysis | None) -> float:
        """Score a sentence for relevance as a key point.

        Args:
            sentence: Sentence to score
            analysis: Query analysis

        Returns:
            Score from 0.0 to 1.0
        """
        score = 0.3  # Base score

        sentence_lower = sentence.lower()

        # Boost for action-oriented language
        action_words = ["should", "must", "need", "important", "note", "remember"]
        if any(word in sentence_lower for word in action_words):
            score += 0.2

        # Boost for explanatory language
        explain_words = ["because", "therefore", "this means", "in order to"]
        if any(word in sentence_lower for word in explain_words):
            score += 0.2

        # Boost if contains technology from analysis
        if analysis and analysis.detected_technologies:
            for tech in analysis.detected_technologies:
                if tech.lower() in sentence_lower:
                    score += 0.2
                    break

        # Boost for key terms from analysis
        if analysis and analysis.key_terms:
            matching_terms = sum(1 for term in analysis.key_terms if term.lower() in sentence_lower)
            score += min(0.3, matching_terms * 0.1)

        return min(1.0, score)

    def _generate_primary_content(
        self,
        results: list[dict[str, Any]],
        key_points: list[str],
        code_snippets: list[dict[str, str]],
        analysis: QueryAnalysis | None,
    ) -> str:
        """Generate primary synthesized content.

        Args:
            results: Deduplicated results
            key_points: Extracted key points
            code_snippets: Extracted code snippets
            analysis: Query analysis

        Returns:
            Primary content string
        """
        parts = []

        # Add key points summary
        if key_points:
            parts.append("**Key Points:**")
            for i, point in enumerate(key_points, 1):
                parts.append(f"{i}. {point}")
            parts.append("")

        # Add code snippets
        if code_snippets:
            parts.append("**Relevant Code:**")
            for snippet in code_snippets:
                lang = snippet.get("language", "")
                code = snippet.get("content", "")
                parts.append(f"```{lang}")
                parts.append(code)
                parts.append("```")
                parts.append("")

        # If no structured content, provide a simple summary
        if not parts:
            if results:
                # Take the most relevant result content
                # Use inf as default to ensure results WITH scores are preferred
                # (lower distance = better, so inf = worst possible)
                best_result = min(results, key=lambda r: r.get("score", float("inf")))
                # Check if ANY results lack scores for better logging
                missing_scores = sum(1 for r in results if r.get("score") is None)
                if missing_scores > 0:
                    logger.warning(f"{missing_scores}/{len(results)} results missing score field")
                # If best_result has no score, use first result
                if best_result.get("score") is None:
                    best_result = results[0]
                content = best_result.get("content", "")
                parts.append(self._truncate(content, 500))
            else:
                parts.append("No specific guidance available for this query.")

        return "\n".join(parts)

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length.

        Args:
            text: Text to truncate
            max_length: Maximum length

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        # Try to truncate at a word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")
        if last_space >= max_length * 0.7:
            truncated = truncated[:last_space]

        return truncated + "..."

    def _calculate_confidence(
        self,
        results: list[dict[str, Any]],
        dedup_count: int,
        key_points: list[str],
    ) -> float:
        """Calculate confidence in the synthesis.

        Args:
            results: Unique results
            dedup_count: Number of duplicates removed
            key_points: Extracted key points

        Returns:
            Confidence score from 0.0 to 1.0
        """
        if not results:
            return 0.0

        # More unique results = higher confidence
        result_confidence = min(1.0, len(results) / 5)

        # More key points = higher confidence
        point_confidence = min(1.0, len(key_points) / 3)

        # Some deduplication is good (shows consistency)
        # Too much might mean low diversity
        if dedup_count == 0:
            dedup_confidence = 0.5
        elif dedup_count <= len(results):
            dedup_confidence = 0.8
        else:
            dedup_confidence = 0.6

        confidence = result_confidence * 0.4 + point_confidence * 0.4 + dedup_confidence * 0.2

        return min(1.0, confidence)


# Global synthesizer instance (lazy initialization with thread safety)
_global_synthesizer: ResultSynthesizer | None = None
_global_synthesizer_lock = threading.Lock()


def get_result_synthesizer(
    similarity_threshold: float = 0.7,
) -> ResultSynthesizer:
    """Get or create the global result synthesizer instance.

    Uses double-check locking pattern for thread-safe lazy initialization.

    Args:
        similarity_threshold: Threshold for deduplication (only used on first call)

    Returns:
        Global ResultSynthesizer instance
    """
    global _global_synthesizer
    if _global_synthesizer is None:
        with _global_synthesizer_lock:
            # Double-check after acquiring lock
            if _global_synthesizer is None:
                _global_synthesizer = ResultSynthesizer(
                    similarity_threshold=similarity_threshold,
                )
    return _global_synthesizer


def reset_result_synthesizer() -> None:
    """Reset global result synthesizer for testing."""
    global _global_synthesizer
    with _global_synthesizer_lock:
        _global_synthesizer = None
