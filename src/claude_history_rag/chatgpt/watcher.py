"""ChatGPT export watcher adapter."""

import threading
from pathlib import Path

from claude_history_rag.chatgpt.chunker import chunk_chatgpt_export_file
from claude_history_rag.config import settings
from claude_history_rag.watcher import HistoryWatcher

chatgpt_watcher: HistoryWatcher | None = None
_chatgpt_watcher_lock = threading.Lock()


def _is_chatgpt_export_file(path: Path) -> bool:
    return path.is_file() and path.name == "conversations.json"


def get_chatgpt_watcher() -> HistoryWatcher:
    """Get or create the global ChatGPT export watcher instance."""
    global chatgpt_watcher
    if chatgpt_watcher is None:
        with _chatgpt_watcher_lock:
            if chatgpt_watcher is None:
                settings.chatgpt_exports_path.mkdir(parents=True, exist_ok=True)
                chatgpt_watcher = HistoryWatcher(
                    projects_path=settings.chatgpt_exports_path,
                    debounce_ms=settings.debounce_delay,
                    state_path=settings.chatgpt_state_path,
                    chunker=chunk_chatgpt_export_file,
                    source_name="ChatGPT",
                    path_filter=_is_chatgpt_export_file,
                )
    return chatgpt_watcher
