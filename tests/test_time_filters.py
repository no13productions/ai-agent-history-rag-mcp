"""Tests for search timeframe parsing."""

from datetime import UTC, datetime

import pytest

from claude_history_rag.time_filters import format_time_filter, parse_timeframe


def test_parse_timeframe_normalizes_iso_bounds_to_utc():
    """ISO bounds are parsed as inclusive UTC datetimes."""
    date_from, date_to = parse_timeframe(
        "2026-06-13T00:12:25-04:00",
        "2026-06-15T04:12:25Z",
    )

    assert date_from == datetime(2026, 6, 13, 4, 12, 25, tzinfo=UTC)
    assert date_to == datetime(2026, 6, 15, 4, 12, 25, tzinfo=UTC)
    assert format_time_filter(date_from) == "2026-06-13T04:12:25Z"


def test_parse_timeframe_expands_date_only_upper_bound():
    """Date-only upper bounds include the whole UTC day."""
    date_from, date_to = parse_timeframe("2026-06-13", "2026-06-15")

    assert date_from == datetime(2026, 6, 13, 0, 0, tzinfo=UTC)
    assert date_to is not None
    assert date_to.date().isoformat() == "2026-06-15"
    assert date_to.hour == 23
    assert date_to.minute == 59


def test_parse_timeframe_rejects_inverted_bounds():
    """Search timeframes must be ordered."""
    with pytest.raises(ValueError, match="date_from must be before date_to"):
        parse_timeframe("2026-06-15", "2026-06-13")
