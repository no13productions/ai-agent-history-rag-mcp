"""Error recording module for status tracking.

This module is intentionally minimal to avoid circular imports.
The StatusCollector registers itself here so errors can be recorded from anywhere.
"""

from datetime import datetime, timezone
from typing import Any

# The actual error storage - StatusCollector will register its list here
_error_list: list[dict[str, Any]] | None = None
_error_counts: dict[str, int] | None = None


def register_error_storage(errors: list[dict[str, Any]], counts: dict[str, int]) -> None:
    """Register the error storage from StatusCollector.

    Called by StatusCollector during initialization to connect the
    error recording system to the actual storage.
    """
    global _error_list, _error_counts
    _error_list = errors
    _error_counts = counts


def record_error(error_type: str, message: str, details: dict[str, Any] | None = None) -> None:
    """Record an error to the status collector.

    This is safe to call from anywhere, even before the status collector
    is initialized. Errors recorded before initialization are silently dropped.

    Args:
        error_type: Category of error (e.g., "embedding", "indexing", "database")
        message: Human-readable error message
        details: Optional dict with additional context
    """
    if _error_list is None or _error_counts is None:
        # Status collector not initialized yet - silently drop
        return

    error_entry = {
        "type": error_type,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    }
    _error_list.append(error_entry)

    # Keep only last 50 errors
    if len(_error_list) > 50:
        _error_list.pop(0)

    # Update error counts by type
    _error_counts[error_type] = _error_counts.get(error_type, 0) + 1
