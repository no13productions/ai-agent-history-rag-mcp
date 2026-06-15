"""Shared parsing for search timeframe filters."""

from __future__ import annotations

from datetime import UTC, date, datetime, time


def parse_time_filter(
    value: str | datetime | None,
    field_name: str,
    *,
    end_of_day: bool = False,
) -> datetime | None:
    """Parse an ISO-8601 search time bound as a UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = value.strip()
        if not raw:
            return None
        try:
            if len(raw) == 10:
                parsed_date = date.fromisoformat(raw)
                parsed = datetime.combine(
                    parsed_date,
                    time.max if end_of_day else time.min,
                    tzinfo=UTC,
                )
            else:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(f"Invalid {field_name}: expected ISO-8601 date or datetime") from e

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_timeframe(
    date_from: str | datetime | None = None,
    date_to: str | datetime | None = None,
) -> tuple[datetime | None, datetime | None]:
    """Parse and validate inclusive search timeframe bounds."""
    parsed_from = parse_time_filter(date_from, "date_from")
    parsed_to = parse_time_filter(date_to, "date_to", end_of_day=True)
    if parsed_from and parsed_to and parsed_from > parsed_to:
        raise ValueError("Invalid timeframe: date_from must be before date_to")
    return parsed_from, parsed_to


def format_time_filter(value: datetime | None) -> str | None:
    """Return the normalized ISO string used across API boundaries."""
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
