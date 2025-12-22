"""Codex watcher adapter using the generic HistoryWatcher."""

import threading

from claude_history_rag.codex.chunker import chunk_codex_session_file
from claude_history_rag.config import settings
from claude_history_rag.watcher import HistoryWatcher

codex_watcher: HistoryWatcher | None = None
_codex_watcher_lock = threading.Lock()


def get_codex_watcher() -> HistoryWatcher:
    """Get or create the global Codex watcher instance (thread-safe)."""
    global codex_watcher
    if codex_watcher is None:
        with _codex_watcher_lock:
            if codex_watcher is None:
                codex_watcher = HistoryWatcher(
                    projects_path=settings.codex_sessions_path,
                    debounce_ms=settings.debounce_delay,
                    state_path=settings.codex_state_path,
                    chunker=chunk_codex_session_file,
                    source_name="Codex",
                )
    return codex_watcher
