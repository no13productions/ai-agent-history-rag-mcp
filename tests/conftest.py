"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def sample_session_path() -> Path:
    """Path to sample session fixture."""
    return Path(__file__).parent / "fixtures" / "sample_session.jsonl"


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Temporary database path for testing."""
    return tmp_path / "test_lancedb"
