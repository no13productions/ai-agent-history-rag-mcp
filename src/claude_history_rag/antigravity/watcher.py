"""Antigravity watcher adapter using the generic HistoryWatcher."""

import threading
from pathlib import Path

from claude_history_rag.config import settings
from claude_history_rag.antigravity.chunker import chunk_antigravity_file
from claude_history_rag.watcher import HistoryWatcher

antigravity_watcher: HistoryWatcher | None = None
_antigravity_watcher_lock = threading.Lock()


def _is_antigravity_file(path: Path) -> bool:
    return path.suffix == ".pb" and path.is_file()


def get_antigravity_watcher() -> HistoryWatcher:
    """Get or create the global Antigravity watcher instance (thread-safe)."""
    global antigravity_watcher
    if antigravity_watcher is None:
        with _antigravity_watcher_lock:
            if antigravity_watcher is None:
                antigravity_watcher = HistoryWatcher(
                    projects_path=settings.antigravity_sessions_path,
                    debounce_ms=settings.debounce_delay,
                    state_path=settings.antigravity_state_path,
                    chunker=chunk_antigravity_file,
                    source_name="Antigravity",
                    path_filter=_is_antigravity_file,
                )
    return antigravity_watcher
