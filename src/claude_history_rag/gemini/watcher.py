"""Gemini watcher adapter using the generic HistoryWatcher."""

import threading
from pathlib import Path

from claude_history_rag.config import settings
from claude_history_rag.gemini.chunker import chunk_gemini_session_file
from claude_history_rag.watcher import HistoryWatcher

gemini_watcher: HistoryWatcher | None = None
_gemini_watcher_lock = threading.Lock()


def _is_gemini_history_file(path: Path) -> bool:
    if path.name == "logs.json":
        return True
    if path.suffix != ".json":
        return False
    return path.parent.name == "chats"


def get_gemini_watcher() -> HistoryWatcher:
    """Get or create the global Gemini watcher instance (thread-safe)."""
    global gemini_watcher
    if gemini_watcher is None:
        with _gemini_watcher_lock:
            if gemini_watcher is None:
                gemini_watcher = HistoryWatcher(
                    projects_path=settings.gemini_sessions_path,
                    debounce_ms=settings.debounce_delay,
                    state_path=settings.gemini_state_path,
                    chunker=chunk_gemini_session_file,
                    source_name="Gemini",
                    path_filter=_is_gemini_history_file,
                )
    return gemini_watcher
