"""Query analysis and understanding.

Analyzes search queries to extract intent, technology context, and key terms.
This enables smarter search strategies and result evaluation.
"""

import logging
import os
import re
import threading

from claude_history_rag.decision_engine.models import (
    QueryAnalysis,
    QueryIntent,
)
from claude_history_rag.decision_engine.technology_patterns import (
    TECHNOLOGY_PATTERNS,
)

logger = logging.getLogger(__name__)

# Pre-compiled pattern for problem keyword detection
# (Bug 1 fix - move to module level)
_PROBLEM_KEYWORDS_PATTERN = re.compile(
    r"\b(bug|error|fix|issue|fail|crash|broken)\b", re.IGNORECASE
)

# Intent detection patterns
INTENT_PATTERNS: dict[QueryIntent, list[str]] = {
    QueryIntent.HOW_TO: [
        r"\bhow\s+(to|do|can|should)\b",
        r"\bsteps?\s+to\b",
        r"\bguide\b",
        r"\btutorial\b",
        r"\bwalkthrough\b",
        r"\bimplement\b",
        r"\bcreate\b",
        r"\bset\s*up\b",
        r"\bconfigure\b",
    ],
    QueryIntent.PROBLEM_SOLVING: [
        r"\berror\b",
        r"\bfail(ed|ing|ure)?\b",
        r"\bbug\b",
        r"\bfix\b",
        r"\bissue\b",
        r"\bproblem\b",
        r"\bcrash(ed|ing)?\b",
        r"\bdebug\b",
        r"\btroubleshoot\b",
        r"\bwhy\s+(is|does|doesn't|isn't|won't)\b",
        r"\bnot\s+work(ing)?\b",
        r"\bbroken\b",
        r"\bdoesn't\s+work\b",
        r"\bexception\b",
        r"\btraceback\b",
        r"\btimeout\b",
        r"\bundefined\b",
    ],
    QueryIntent.CODE_EXAMPLE: [
        r"\bexample\b",
        r"\bsample\b",
        r"\bsnippet\b",
        r"\bcode\s+for\b",
        r"\bshow\s+(me\s+)?how\b",
        r"\btemplate\b",
        r"\bboilerplate\b",
    ],
    QueryIntent.REFERENCE: [
        r"\bapi\b",
        r"\bdocumentation\b",
        r"\bdocs?\b",
        r"\breference\b",
        r"\bsyntax\b",
        r"\bsignature\b",
        r"\bparameters?\b",
        r"\breturn\s+type\b",
    ],
    QueryIntent.FILE_HISTORY: [
        r"\bwhat\s+(did|have)\s+(we|i)\s+(change|modify|edit|update)\b",
        r"\bfile\s+(changes?|modifications?|edits?)\b",
        r"\brecent\s+(edits?|changes?)\b",
        r"\bshow\s+(me\s+)?changes?\b",
        r"\bhistory\s+of\b",
        r"\bwhat\s+happened\s+to\b",
    ],
    QueryIntent.SESSION_CONTEXT: [
        r"\blast\s+session\b",
        r"\bprevious(ly)?\b",
        r"\bearlier\b",
        r"\bwhat\s+(did|have)\s+we\s+(discuss|talk|work)\b",
        r"\bresume\b",
        r"\bcontinue\b",
        r"\bwhere\s+were\s+we\b",
    ],
    QueryIntent.DECISION_RECALL: [
        r"\bwhy\s+did\s+(we|i)\b",
        r"\bdecision\b",
        r"\bdecided\b",
        r"\breason(ing)?\b",
        r"\bwhat\s+(was|were)\s+the\s+reason\b",
        r"\bremember\s+when\b",
    ],
}


class QueryAnalyzer:
    """Analyzes queries to extract intent, technologies, and key terms.

    This analyzer runs without any external dependencies (no LLM required)
    using pattern matching for fast, deterministic analysis.
    """

    def __init__(
        self,
        technology_patterns: dict[str, list[str]] | None = None,
        intent_patterns: dict[QueryIntent, list[str]] | None = None,
    ):
        """Initialize query analyzer.

        Args:
            technology_patterns: Custom technology patterns (uses defaults if None)
            intent_patterns: Custom intent patterns (uses defaults if None)
        """
        self._tech_patterns = technology_patterns or TECHNOLOGY_PATTERNS
        self._intent_patterns = intent_patterns or INTENT_PATTERNS

        # Pre-compile intent patterns for efficiency
        self._compiled_intent_patterns: dict[QueryIntent, list[re.Pattern]] = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self._intent_patterns.items()
        }

        # Pre-compile technology patterns for efficiency (P4 fix)
        # Maps technology name -> list of compiled word-boundary patterns
        self._compiled_tech_patterns: dict[str, list[re.Pattern]] = {
            tech: [re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE) for keyword in keywords]
            for tech, keywords in self._tech_patterns.items()
        }

        logger.debug(f"QueryAnalyzer initialized with {len(self._tech_patterns)} tech patterns")

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a search query.

        Args:
            query: The search query to analyze

        Returns:
            QueryAnalysis with extracted information
        """
        logger.debug(f"analyze() called: query_length={len(query)}")

        if not query or not query.strip():
            logger.debug("Early return: empty or whitespace-only query")
            return QueryAnalysis(
                original_query=query or "",
                normalized_query="",
                confidence=0.0,
            )

        normalized = query.lower().strip()

        # Detect technologies
        technologies = self._detect_technologies(normalized)

        # Classify intent
        intent, intent_confidence = self._classify_intent(normalized)

        # Extract key terms
        key_terms = self._extract_key_terms(normalized, technologies)

        # Check for file-specific queries
        is_file_specific, file_paths = self._detect_file_references(query)

        # Calculate overall confidence
        confidence = self._calculate_confidence(technologies, intent, intent_confidence, key_terms)

        analysis = QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            intent=intent,
            detected_technologies=technologies,
            key_terms=key_terms,
            is_file_specific=is_file_specific,
            extracted_file_paths=file_paths,
            confidence=confidence,
        )

        logger.debug(
            f"Query analysis: intent={intent.value}, "
            f"techs={technologies}, confidence={confidence:.2f}"
        )

        return analysis

    def _detect_technologies(self, query: str) -> list[str]:
        """Detect technologies mentioned in query.

        Uses pre-compiled patterns for efficiency (P4 fix).

        Args:
            query: Normalized query string

        Returns:
            List of detected technology names
        """
        detected = []

        for tech, patterns in self._compiled_tech_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    if tech not in detected:
                        detected.append(tech)
                    break

        return detected

    def _classify_intent(self, query: str) -> tuple[QueryIntent, float]:
        """Classify query intent.

        Args:
            query: Normalized query string

        Returns:
            Tuple of (QueryIntent, confidence score)
        """
        intent_scores: dict[QueryIntent, int] = {}

        for intent, patterns in self._compiled_intent_patterns.items():
            match_count = sum(1 for p in patterns if p.search(query))
            if match_count > 0:
                intent_scores[intent] = match_count

        if not intent_scores:
            return QueryIntent.INFORMATION_SEEKING, 0.3

        # Get intent with highest match count
        max_score = max(intent_scores.values())

        # M1 fix: If HOW_TO and PROBLEM_SOLVING both match with same score,
        # prefer PROBLEM_SOLVING when bug/error/fix words are present
        if (
            QueryIntent.HOW_TO in intent_scores
            and QueryIntent.PROBLEM_SOLVING in intent_scores
            and intent_scores[QueryIntent.HOW_TO] == intent_scores[QueryIntent.PROBLEM_SOLVING]
            and intent_scores[QueryIntent.HOW_TO] == max_score
        ):
            # Check for problem-solving keywords using pre-compiled pattern
            if _PROBLEM_KEYWORDS_PATTERN.search(query):
                best_intent = QueryIntent.PROBLEM_SOLVING
            else:
                best_intent = QueryIntent.HOW_TO
        else:
            best_intent = max(intent_scores, key=lambda k: intent_scores[k])

        # Calculate confidence based on number of patterns matched
        total_patterns = len(self._compiled_intent_patterns[best_intent])
        confidence = min(1.0, 0.4 + (max_score / total_patterns) * 0.6)

        return best_intent, confidence

    def _extract_key_terms(self, query: str, detected_technologies: list[str]) -> list[str]:
        """Extract key terms from query.

        Filters out common stop words and technology names to get
        the core search terms.

        Args:
            query: Normalized query string
            detected_technologies: Already detected technologies

        Returns:
            List of key terms
        """
        # Important 2-letter tech terms to preserve (C3 fix)
        two_letter_tech_whitelist = {
            "go",  # Go language
            "js",  # JavaScript
            "ts",  # TypeScript
            "py",  # Python
            "db",  # Database
            "ui",  # User Interface
            "ai",  # Artificial Intelligence
            "ml",  # Machine Learning
            "ci",  # Continuous Integration
            "cd",  # Continuous Deployment
        }

        # Common stop words to filter out
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "but",
            "and",
            "or",
            "if",
            "what",
            "which",
            "who",
            "this",
            "that",
            "these",
            "those",
            "am",
            "i",
            "we",
            "you",
            "he",
            "she",
            "it",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "show",
            "get",
            "find",
            "look",
        }

        # Technology keywords to filter (already captured)
        tech_keywords = set()
        for tech in detected_technologies:
            if tech in self._tech_patterns:
                tech_keywords.update(kw.lower() for kw in self._tech_patterns[tech])

        # Tokenize and filter
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", query)
        key_terms = []

        for word in words:
            word_lower = word.lower()
            # Whitelist important 2-letter tech terms FIRST
            if word_lower in two_letter_tech_whitelist:
                if word_lower not in tech_keywords and word_lower not in key_terms:
                    key_terms.append(word_lower)
                continue
            # Then check stop words for non-whitelisted terms
            if word_lower in stop_words:
                continue
            # Include longer terms
            if len(word) > 2 and word_lower not in tech_keywords and word_lower not in key_terms:
                key_terms.append(word_lower)

        return key_terms[:10]  # Limit to top 10 terms

    def _detect_file_references(self, query: str) -> tuple[bool, list[str]]:
        """Detect file path references in query.

        Args:
            query: Original query string (not normalized, to preserve paths)

        Returns:
            Tuple of (is_file_specific, list of file paths)
        """
        # Look for explicit file patterns
        file_extensions = [
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".java",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".cs",
            ".swift",
            ".kt",
            ".dart",
            ".cpp",
            ".c",
            ".h",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".xml",
            ".html",
            ".css",
            ".md",
            ".txt",
            ".sh",
            ".bash",
            ".sql",
            ".env",
            ".config",
        ]

        # Find potential file references
        files = []
        words = query.split()

        for word in words:
            # Clean up word (remove trailing punctuation)
            clean_word = word.rstrip(".,;:!?")

            # Check if it looks like a file path
            if any(clean_word.endswith(ext) for ext in file_extensions):
                files.append(clean_word)
            elif (os.sep in clean_word or "/" in clean_word or "\\" in clean_word) and any(
                ext in clean_word for ext in file_extensions
            ):
                # Might be a path with extension (normalize separators)
                # os.sep handles platform-specific separators
                files.append(clean_word)

        # Check for file-specific language patterns
        file_patterns = [
            r"\bfile\s+\w+\.\w+\b",
            r"\bin\s+\w+\.\w+\b",
            r"\bto\s+\w+\.\w+\b",
        ]

        try:
            for pattern in file_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                for match in matches:
                    # Extract file name from match
                    parts = match.split()
                    if len(parts) >= 2:
                        potential_file = parts[-1]
                        if (
                            any(potential_file.endswith(ext) for ext in file_extensions)
                            and potential_file not in files
                        ):
                            files.append(potential_file)

            is_file_specific = (
                len(files) > 0
                or re.search(r"\bfile\s*(change|edit|modif)", query, re.IGNORECASE) is not None
            )
        except re.error as e:
            logger.warning(
                f"Regex error in _detect_file_references: {e}. Query preview: {query[:100]}"
            )
            is_file_specific = False

        return is_file_specific, files

    def _calculate_confidence(
        self,
        technologies: list[str],
        intent: QueryIntent,
        intent_confidence: float,
        key_terms: list[str],
    ) -> float:
        """Calculate overall confidence in the analysis.

        Args:
            technologies: Detected technologies
            intent: Classified intent
            intent_confidence: Confidence in intent classification
            key_terms: Extracted key terms

        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        # Base confidence from intent classification
        confidence = intent_confidence * 0.4

        # Bonus for detected technologies
        if technologies:
            confidence += 0.2

        # Bonus for key terms
        if len(key_terms) >= 2:
            confidence += 0.2
        elif len(key_terms) >= 1:
            confidence += 0.1

        # Bonus for specific intent (not default)
        if intent != QueryIntent.INFORMATION_SEEKING:
            confidence += 0.2

        return min(1.0, confidence)

    def suggest_refinements(self, query: str, analysis: QueryAnalysis | None = None) -> list[str]:
        """Suggest query refinements to improve search results.

        Args:
            query: Original query
            analysis: Pre-computed analysis (will compute if None)

        Returns:
            List of suggested refined queries
        """
        if analysis is None:
            analysis = self.analyze(query)

        suggestions = []
        base_query = analysis.normalized_query

        # Add technology context if not present
        if not analysis.detected_technologies:
            # Suggest common technology additions
            suggestions.append(f"{base_query} python")
            suggestions.append(f"{base_query} implementation")

        # Add specificity based on intent
        if analysis.intent == QueryIntent.HOW_TO:
            if "example" not in base_query:
                suggestions.append(f"{base_query} example")
            if "step" not in base_query:
                suggestions.append(f"{base_query} steps")

        if analysis.intent == QueryIntent.PROBLEM_SOLVING:
            if "fix" not in base_query:
                suggestions.append(f"fix {base_query}")
            if "solution" not in base_query:
                suggestions.append(f"{base_query} solution")

        if analysis.intent == QueryIntent.CODE_EXAMPLE and "code" not in base_query:
            suggestions.append(f"{base_query} code")

        # Limit suggestions
        return suggestions[:3]


# Global analyzer instance (lazy initialization with thread safety)
_global_analyzer: QueryAnalyzer | None = None
_global_analyzer_lock = threading.Lock()


def get_query_analyzer() -> QueryAnalyzer:
    """Get or create the global query analyzer instance.

    Uses double-check locking pattern for thread-safe lazy initialization.

    Returns:
        Global QueryAnalyzer instance
    """
    global _global_analyzer
    if _global_analyzer is None:
        with _global_analyzer_lock:
            # Double-check after acquiring lock
            if _global_analyzer is None:
                _global_analyzer = QueryAnalyzer()
    return _global_analyzer


def reset_query_analyzer() -> None:
    """Reset global query analyzer for testing."""
    global _global_analyzer
    with _global_analyzer_lock:
        _global_analyzer = None
