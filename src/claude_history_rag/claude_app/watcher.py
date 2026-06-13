"""Claude app export watcher adapter."""

import threading
from pathlib import Path

from claude_history_rag.claude_app.chunker import chunk_claude_app_export_file
from claude_history_rag.config import settings
from claude_history_rag.watcher import HistoryWatcher

claude_app_watcher: HistoryWatcher | None = None
_claude_app_watcher_lock = threading.Lock()


def _is_claude_app_export_file(path: Path) -> bool:
    return path.is_file() and path.name == "conversations.json"


def get_claude_app_watcher() -> HistoryWatcher:
    """Get or create the global Claude app export watcher instance."""
    global claude_app_watcher
    if claude_app_watcher is None:
        with _claude_app_watcher_lock:
            if claude_app_watcher is None:
                settings.claude_app_exports_path.mkdir(parents=True, exist_ok=True)
                claude_app_watcher = HistoryWatcher(
                    projects_path=settings.claude_app_exports_path,
                    debounce_ms=settings.debounce_delay,
                    state_path=settings.claude_app_state_path,
                    chunker=chunk_claude_app_export_file,
                    source_name="Claude App",
                    path_filter=_is_claude_app_export_file,
                )
    return claude_app_watcher
