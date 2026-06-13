"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def sample_session_path() -> Path:
    """Path to sample session fixture (legacy pre-2.1 format)."""
    return Path(__file__).parent / "fixtures" / "sample_session.jsonl"


@pytest.fixture
def sample_session_compact_path() -> Path:
    """Path to sample session fixture in current Claude Code (>=2.1) format.

    Uses isCompactSummary entries and tool_result user messages instead of
    the legacy dedicated "summary" entry type.
    """
    return Path(__file__).parent / "fixtures" / "sample_session_compact.jsonl"


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Temporary database path for testing."""
    return tmp_path / "test_lancedb"
