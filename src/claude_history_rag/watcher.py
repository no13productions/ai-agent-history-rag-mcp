"""File watcher for Claude Code history files using watchfiles.

Supports two modes:
- Server mode: Embed and store chunks locally
- Client mode: Batch chunks and upload to central server
"""

import asyncio
import contextlib
import gc
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil
from watchfiles import Change, awatch

from claude_history_rag import __version__ as package_version
from claude_history_rag.chunker import chunk_session_file
from claude_history_rag.config import settings
from claude_history_rag.errors import record_error
from claude_history_rag.models import Chunk

if TYPE_CHECKING:
    from claude_history_rag.api_client import APIClient
    from claude_history_rag.client_state import ClientStateManager
    from claude_history_rag.embedder import AsyncEmbedder

logger = logging.getLogger(__name__)


def _is_safe_path(path: Path, base_path: Path) -> bool:
    """Check if path is safely within base_path (no symlink escape)."""
    try:
        resolved = path.resolve()
        base_resolved = base_path.resolve()
        resolved.relative_to(base_resolved)  # Raises ValueError if not under base
        return True
    except (OSError, ValueError):
        return False


def _count_file_lines(file_path: Path) -> int:
    """Count total lines in a file."""
    try:
        with open(file_path) as f:
            return sum(1 for _ in f)
    except (OSError, UnicodeDecodeError):
        return 0


async def _queue_all_watchers_for_reindex() -> int:
    """Queue all files for indexing across all client watchers."""
    total_queued = 0
    for watcher in get_all_watchers():
        watcher.clear_failed_files()
        total_queued += await watcher.queue_all_files_for_indexing()
    return total_queued


async def _handle_server_reindex(
    api_client: "APIClient",
    state_manager: "ClientStateManager",
    reindex_requested_at: str | None,
    reason: str | None = None,
) -> None:
    """Handle a server-triggered reindex in client mode."""
    if not reindex_requested_at:
        return
    if not await state_manager.should_handle_reindex(reindex_requested_at):
        return

    parsed_at: datetime | None = None
    try:
        parsed_at = datetime.fromisoformat(reindex_requested_at)
        if parsed_at.tzinfo is None:
            parsed_at = parsed_at.replace(tzinfo=timezone.utc)
    except Exception:
        parsed_at = datetime.now(timezone.utc)

    await state_manager.set_reindex_required(parsed_at)
    await state_manager.reset_for_reindex()

    queued = await _queue_all_watchers_for_reindex()
    logger.warning(f"Server requested reindex at {reindex_requested_at}; queued {queued} files")

    try:
        await api_client.ack_reindex(
            reindex_requested_at=reindex_requested_at,
            status="queued",
            reason=reason,
        )
        await state_manager.set_reindex_ack(status="queued")
    except Exception as e:
        logger.warning(f"Failed to acknowledge reindex request: {type(e).__name__}: {e}")


async def _maybe_ack_reindex_completed(
    api_client: "APIClient",
    state_manager: "ClientStateManager",
) -> None:
    """Send a completed ack once a reindex has finished uploading."""
    state = await state_manager.get_state()
    if not state.reindex_required_at:
        return
    if state.reindex_status == "completed":
        return
    if state.pending_uploads:
        return

    if any(watcher.queue.qsize() > 0 for watcher in get_all_watchers()):
        return

    if state.last_server_sync and state.last_server_sync < state.reindex_required_at:
        return
    if not state.last_server_sync and state.local_positions:
        return

    try:
        await api_client.ack_reindex(
            reindex_requested_at=state.reindex_required_at.isoformat(),
            status="completed",
            reason="uploads_finished",
        )
        await state_manager.set_reindex_ack(status="completed")
    except Exception as e:
        logger.warning(f"Failed to send completed reindex ack: {type(e).__name__}: {e}")


class FilePositionState:
    """Track file positions for incremental reading."""

    def __init__(self, state_path: Path | None = None):
        self.state_path = state_path or settings.state_path
        self._positions: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        """Load state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                    self._positions = data.get("file_positions", {})
                    # LOG LOW #1: Log state file size and sample paths for better debugging
                    file_size = self.state_path.stat().st_size
                    sample_paths = list(self._positions.keys())[:3]
                    logger.info(
                        f"Loaded state for {len(self._positions)} files "
                        f"({file_size} bytes, sample: {sample_paths})"
                    )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load state file: {type(e).__name__}")
                self._positions = {}

    def save(self) -> None:
        """Save state to disk atomically."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.state_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump({"file_positions": self._positions}, f)
            temp_path.replace(self.state_path)  # Atomic on POSIX
        except OSError as e:
            logger.error(f"Failed to save state: {type(e).__name__}")

    def get_position(self, file_path: str) -> int:
        """Get last processed line number for a file."""
        return self._positions.get(file_path, 0)

    def set_position(self, file_path: str, line_number: int) -> None:
        """Set last processed line number for a file."""
        self._positions[file_path] = line_number

    def get_all_files(self) -> list[str]:
        """Get all tracked file paths."""
        return list(self._positions.keys())

    def reset_all_positions(self) -> int:
        """Reset all file positions to 0, forcing a full re-index.

        Returns:
            Number of file positions reset.
        """
        count = len(self._positions)
        self._positions.clear()
        self.save()
        logger.info(f"Reset positions for {count} files")
        return count


class HistoryWatcher:
    """Watch Claude Code history files and index changes."""

    def __init__(
        self,
        projects_path: Path | None = None,
        debounce_ms: int = 5000,
        state_path: Path | None = None,
        chunker: Callable[[Path, int], Iterator[Chunk]] | None = None,
        source_name: str = "Claude Code",
        path_filter: Callable[[Path], bool] | None = None,
    ):
        self.projects_path = projects_path or settings.projects_path
        if debounce_ms < 0:
            raise ValueError(f"debounce_ms must be non-negative, got {debounce_ms}")
        self.debounce_ms = debounce_ms
        self.state = FilePositionState(state_path=state_path)
        self._chunker = chunker or chunk_session_file
        self._source_name = source_name
        self._path_filter = path_filter or (lambda p: p.suffix == ".jsonl")
        # Limit queue size to prevent unbounded memory growth
        # 10000 allows for large initial indexing while preventing memory issues
        self.queue: asyncio.Queue[Path] = asyncio.Queue(maxsize=10000)
        self._running = False
        self._watch_task: asyncio.Task | None = None
        self._process_task: asyncio.Task | None = None
        self._client_sync_task: asyncio.Task | None = None
        # MEDIUM #3: Track failed files separately to avoid infinite retry loops
        self._failed_files: set[str] = set()
        self._shutdown_event = asyncio.Event()
        self._last_indexed_file: str | None = None
        self._last_indexed_at: datetime | None = None
        self._last_upload_at: datetime | None = None
        self._last_heartbeat_at: float = 0.0

    @property
    def is_running(self) -> bool:
        """Check if the watcher is currently running."""
        return self._running

    @property
    def source_name(self) -> str:
        """Human-readable source name for logs and status output."""
        return self._source_name

    @property
    def failed_files_count(self) -> int:
        """Number of files that failed indexing for this watcher."""
        return len(self._failed_files)

    def failed_files(self) -> list[str]:
        """Return failed file paths for status reporting."""
        return list(self._failed_files)

    def clear_failed_files(self) -> None:
        """Clear failed file tracking before a forced reindex."""
        self._failed_files.clear()

    def discover_files(self) -> list[Path]:
        """Return currently discoverable history files for this watcher."""
        if not self.projects_path.exists():
            return []
        return [
            file_path
            for file_path in self.projects_path.glob("**/*")
            if _is_safe_path(file_path, self.projects_path) and self._path_filter(file_path)
        ]

    def is_allowed_history_path(self, path: Path) -> bool:
        """Return whether a path is inside this watch root and matches this source filter."""
        return _is_safe_path(path, self.projects_path) and self._path_filter(path)

    async def _watch_files(self) -> None:
        """Producer: watch for file changes."""
        logger.info(f"Starting file watcher on {self.projects_path}")

        if not self.projects_path.exists():
            logger.warning(f"Projects path does not exist: {self.projects_path}")
            return

        try:
            async for changes in awatch(
                self.projects_path,
                watch_filter=lambda _, p: self._path_filter(Path(p)),
                debounce=self.debounce_ms,
                recursive=True,
            ):
                if not self._running:
                    break

                for change_type, path_str in changes:
                    if change_type in (Change.added, Change.modified):
                        path = Path(path_str)
                        # Security: validate path is within projects_path
                        if not _is_safe_path(path, self.projects_path):
                            logger.warning(f"Ignoring file outside projects path: {path.name}")
                            continue
                        logger.debug(f"File changed: {path.name}")
                        await self.queue.put(path)

        except Exception as e:
            logger.exception(f"Watcher error: {type(e).__name__}")
            record_error("watcher", f"File watcher error: {type(e).__name__}", {"error": str(e)})

    async def _process_files(self) -> None:
        """Consumer: process changed files.

        Handles both server mode (local embedding) and client mode (upload to server).
        """
        # Import conditionally based on mode
        if settings.is_client_mode:
            from claude_history_rag.api_client import get_api_client
            from claude_history_rag.client_state import get_client_state_manager

            api_client = get_api_client()
            state_manager = get_client_state_manager()
            embedder = None  # Not used in client mode
        else:
            from claude_history_rag.store import store as chunk_store

            if (
                settings.storage_backend == "spanner"
                and settings.spanner_embedding_mode == "spanner"
            ):
                embedder = None
            else:
                from claude_history_rag.embedder import get_embedder

                embedder = get_embedder()
            api_client = None
            state_manager = None

        while self._running:
            try:
                # MEDIUM #1: Use asyncio.wait instead of timeout polling
                queue_get_task = asyncio.create_task(self.queue.get())
                shutdown_task = asyncio.create_task(self._shutdown_event.wait())

                done, pending = await asyncio.wait(
                    {queue_get_task, shutdown_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel pending task
                for task in pending:
                    task.cancel()

                # Check if shutdown was signaled
                if shutdown_task in done:
                    break

                # Process the file if we got one
                if queue_get_task in done:
                    try:
                        path = await queue_get_task
                        if settings.is_client_mode:
                            await self._index_file_client_mode(path, api_client, state_manager)
                        else:
                            await self._index_file(path, embedder, chunk_store)
                    finally:
                        # Always mark task done, even on error
                        self.queue.task_done()

            except asyncio.CancelledError:
                logger.debug("Process task cancelled")
                raise  # Re-raise to allow proper cancellation
            except Exception as e:
                logger.exception(f"Processing error: {type(e).__name__}")
                record_error(
                    "processing", f"File processing error: {type(e).__name__}", {"error": str(e)}
                )

    async def _embed_and_store_batch(
        self, chunk_batch: list[dict], file_path: Path, embedder: "AsyncEmbedder", store
    ) -> int | None:
        """Embed and store a batch of chunks. Returns chunk count or None on failure."""
        if settings.storage_backend == "spanner" and settings.spanner_embedding_mode == "spanner":
            try:
                await store.add_chunks_async(chunk_batch)
                return len(chunk_batch)
            except Exception as e:
                logger.error(
                    f"Failed to store Spanner-native embedding batch from {file_path.name}: "
                    f"{type(e).__name__}",
                    exc_info=True,
                )
                record_error(
                    "database",
                    f"Failed to store Spanner-native embedding batch: {type(e).__name__}",
                    {"file": file_path.name, "error_type": type(e).__name__},
                )
                return None

        try:
            embedded_chunks = await embedder.embed_chunks(chunk_batch)
        except Exception as e:
            logger.error(
                f"Failed to embed batch from {file_path.name}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            record_error(
                "embedding",
                f"Failed to embed batch: {type(e).__name__}",
                {"file": file_path.name, "error": str(e)},
            )
            return None

        if not embedded_chunks:
            logger.warning(f"No chunks embedded from batch in {file_path.name}")
            return 0

        try:
            await store.add_chunks_async(embedded_chunks)
            stored_count = len(embedded_chunks)

            # Explicitly clear embedded chunks to free memory immediately
            embedded_chunks.clear()
            del embedded_chunks
            gc.collect()

            return stored_count
        except Exception as e:
            logger.error(
                f"Failed to store batch from {file_path.name}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            record_error(
                "database",
                f"Failed to store batch: {type(e).__name__}",
                {"file": file_path.name, "error": str(e)},
            )
            return None

    # ============================================================
    # Client Mode Methods
    # ============================================================

    def _hash_value(self, value: str) -> str:
        """Return a short stable hash for sensitive values."""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]

    def _get_config_snapshot(self) -> dict[str, Any]:
        """Return a minimal, non-sensitive config snapshot for diagnostics."""
        snapshot = {
            "batch_size": settings.batch_size,
            "max_chunks_per_file": settings.max_chunks_per_file,
            "debounce_ms": self.debounce_ms,
            "source_name": self._source_name,
        }
        snapshot_hash = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "hash": snapshot_hash,
            "projects_path_hash": self._hash_value(str(self.projects_path)),
            "snapshot": snapshot,
        }

    async def _collect_client_heartbeat(
        self,
        state_manager: "ClientStateManager",
    ) -> dict[str, Any]:
        """Collect client status details for heartbeat payload."""
        state = await state_manager.get_state()
        pending = list(state.pending_uploads)
        pending_age_sec = None
        if pending:
            oldest = min(p.created_at for p in pending)
            pending_age_sec = int((datetime.now(timezone.utc) - oldest).total_seconds())

        status = "ok"
        if not state.connected:
            status = "degraded"

        resources = {}
        try:
            process = psutil.Process(os.getpid())
            resources = {
                "memory_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                "cpu_percent": psutil.cpu_percent(interval=None),
            }
        except Exception:
            resources = {}

        return {
            "client_version": package_version,
            "os": platform.platform(),
            "arch": platform.machine(),
            "python_version": sys.version.split()[0],
            "hostname": socket.gethostname(),
            "timezone": time.tzname[0] if time.tzname else None,
            "heartbeat_interval_s": settings.client_heartbeat_interval_seconds,
            "status": status,
            "last_upload_at": self._last_upload_at,
            "last_indexed_at": self._last_indexed_at,
            "queue": {
                "pending_uploads": len(pending),
                "pending_uploads_oldest_age_sec": pending_age_sec,
                "queue_size": self.queue.qsize(),
                "queue_max_size": self.queue.maxsize,
            },
            "watcher": {
                "failed_files_count": len(self._failed_files),
                "debounce_ms": self.debounce_ms,
                "last_indexed_file": self._last_indexed_file,
            },
            "reindex": {
                "required_at": state.reindex_required_at.isoformat()
                if state.reindex_required_at
                else None,
                "ack_at": state.reindex_ack_at.isoformat() if state.reindex_ack_at else None,
                "status": state.reindex_status,
            },
            "errors": {"count_10m": 0},
            "config": self._get_config_snapshot(),
            "doctor": {"client_state": await state_manager.get_summary()},
            "resources": resources,
            "sent_at": datetime.now(timezone.utc),
        }

    async def _send_client_heartbeat(
        self,
        api_client: "APIClient",
        state_manager: "ClientStateManager",
    ) -> None:
        """Send client heartbeat to central server."""
        from claude_history_rag.api_client import ServerConnectionError

        try:
            payload = await self._collect_client_heartbeat(state_manager)
            await api_client.send_heartbeat(payload)
            await state_manager.set_connected(True)
        except ServerConnectionError:
            await state_manager.set_connected(False)
        except Exception as e:
            logger.warning(f"Heartbeat failed: {type(e).__name__}: {e}")

    async def _upload_chunks_to_server(
        self,
        chunk_batch: list[dict],
        file_path: Path,
        file_position: int,
        api_client: "APIClient",
        state_manager: "ClientStateManager",
    ) -> bool:
        """Upload chunks to central server. Returns True on success."""
        from claude_history_rag.api_client import ServerConnectionError

        try:
            logger.debug(
                "Uploading %d chunks for %s at position %d",
                len(chunk_batch),
                file_path.name,
                file_position,
            )
            response = await api_client.upload_chunks(
                chunks=chunk_batch,
                source_file=str(file_path),
                file_position=file_position,
            )

            if response.status == "ok":
                if response.reindex_required:
                    await _handle_server_reindex(
                        api_client,
                        state_manager,
                        response.reindex_requested_at,
                        reason="flagged_on_upload",
                    )
                logger.info(
                    f"Uploaded {response.chunks_stored} chunks from {file_path.name} "
                    f"to server (received={response.chunks_received}, "
                    f"embedded={response.chunks_embedded})"
                )
                self._last_upload_at = datetime.now(timezone.utc)
                # Update server-confirmed position
                await state_manager.update_server_position(str(file_path), file_position)
                await state_manager.set_connected(True)
                return True
            else:
                logger.error(f"Server rejected upload: {response.error}")
                # Add to pending uploads for retry
                await state_manager.add_pending_upload(str(file_path), chunk_batch, file_position)
                return False

        except ServerConnectionError as e:
            logger.warning(f"Failed to upload to server: {e}")
            await state_manager.set_connected(False)
            # Add to pending uploads for later retry
            await state_manager.add_pending_upload(str(file_path), chunk_batch, file_position)
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading chunks: {type(e).__name__}: {e}")
            record_error(
                "upload",
                f"Failed to upload chunks: {type(e).__name__}",
                {"file": file_path.name, "error": str(e)},
            )
            await state_manager.add_pending_upload(str(file_path), chunk_batch, file_position)
            return False

    async def _process_pending_uploads(
        self,
        api_client: "APIClient",
        state_manager: "ClientStateManager",
    ) -> int:
        """Process pending uploads. Returns number successfully uploaded."""
        from claude_history_rag.api_client import ServerConnectionError

        pending = await state_manager.get_pending_uploads()
        if not pending:
            return 0

        logger.info(f"Processing {len(pending)} pending uploads")
        successful = 0

        for upload in pending:
            try:
                response = await api_client.upload_chunks(
                    chunks=upload.chunks,
                    source_file=upload.file_path,
                    file_position=upload.file_position,
                )

                if response.status == "ok":
                    if response.reindex_required:
                        await _handle_server_reindex(
                            api_client,
                            state_manager,
                            response.reindex_requested_at,
                            reason="flagged_on_pending_upload",
                        )
                    logger.info(f"Successfully uploaded pending chunks for {upload.file_path}")
                    self._last_upload_at = datetime.now(timezone.utc)
                    await state_manager.remove_pending_upload(upload.file_path)
                    await state_manager.update_server_position(
                        upload.file_path, upload.file_position
                    )
                    successful += 1
                else:
                    logger.warning(f"Server rejected pending upload: {response.error}")
                    await state_manager.increment_retry_count(upload.file_path)

            except ServerConnectionError:
                logger.warning("Server unavailable, will retry pending uploads later")
                await state_manager.set_connected(False)
                break  # Don't try more if server is down
            except Exception as e:
                logger.error(f"Failed to upload pending: {type(e).__name__}: {e}")
                await state_manager.increment_retry_count(upload.file_path)

        return successful

    async def _sync_positions_with_server(
        self,
        api_client: "APIClient",
        state_manager: "ClientStateManager",
    ) -> None:
        """Sync positions with server and identify files needing catch-up."""
        from claude_history_rag.api_client import ServerConnectionError

        try:
            response = await api_client.get_positions()
            if response.error:
                logger.warning(f"Failed to get server positions: {response.error}")
                return
            if response.reindex_required:
                await _handle_server_reindex(
                    api_client,
                    state_manager,
                    response.reindex_requested_at,
                    reason="flagged_on_position_sync",
                )
                return

            server_positions = response.positions
            catchup_files = await state_manager.get_files_needing_catchup(server_positions)

            if catchup_files:
                logger.info(f"Found {len(catchup_files)} files needing catch-up")
                # Re-queue these files for indexing
                for file_path, server_pos, _local_pos in catchup_files:
                    path = Path(file_path)
                    if path.exists() and self.is_allowed_history_path(path):
                        await self.queue.put(path)
                    elif path.exists():
                        logger.warning(
                            "Ignoring catch-up path outside %s watcher root: %s",
                            self.source_name,
                            path.name,
                        )
                    else:
                        # File was deleted locally - handle gracefully
                        await state_manager.handle_missing_history(file_path, server_pos)

            await state_manager.set_connected(True)

        except ServerConnectionError:
            logger.warning("Server unavailable for position sync")
            await state_manager.set_connected(False)

    async def _client_sync_loop(
        self,
        api_client: "APIClient",
        state_manager: "ClientStateManager",
    ) -> None:
        """Periodically sync positions and retry pending uploads in client mode."""
        sync_interval = max(30, int(settings.upload_interval_seconds))
        heartbeat_interval = max(30, int(settings.client_heartbeat_interval_seconds))
        next_sync = time.monotonic()
        next_heartbeat = time.monotonic()
        while self._running and not self._shutdown_event.is_set():
            try:
                now = time.monotonic()

                if now >= next_sync:
                    await self._sync_positions_with_server(api_client, state_manager)
                    await self._process_pending_uploads(api_client, state_manager)
                    await state_manager.clear_stale_pending_uploads()
                    await _maybe_ack_reindex_completed(api_client, state_manager)
                    next_sync = now + sync_interval

                if now >= next_heartbeat:
                    await self._send_client_heartbeat(api_client, state_manager)
                    self._last_heartbeat_at = time.monotonic()
                    next_heartbeat = now + heartbeat_interval
            except Exception as e:
                logger.warning(f"Client sync loop error: {type(e).__name__}: {e}")

            try:
                now = time.monotonic()
                timeout = min(next_sync, next_heartbeat) - now
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=max(timeout, 0.5))
            except asyncio.TimeoutError:
                continue

    async def _index_file_client_mode(
        self,
        file_path: Path,
        api_client: "APIClient",
        state_manager: "ClientStateManager",
    ) -> None:
        """Index a file in client mode - chunk and prepare for upload."""
        path_str = str(file_path)
        if not self.is_allowed_history_path(file_path):
            logger.warning(
                "[CLIENT] Ignoring path outside %s watcher root: %s",
                self.source_name,
                file_path.name,
            )
            return

        # Skip files that have failed recently
        if path_str in self._failed_files:
            logger.debug(f"Skipping previously failed file: {file_path.name}")
            return

        # Get local position (what we've chunked)
        state = await state_manager.get_state()
        start_line = state.local_positions.get(path_str, 0)

        start_time = time.time()
        total_lines = _count_file_lines(file_path)
        logger.info(
            f"[CLIENT] Chunking {file_path.name} from line {start_line} (total_lines={total_lines})"
        )

        max_chunks = settings.max_chunks_per_file
        chunk_batch: list[dict] = []
        max_line = start_line
        total_chunks_processed = 0

        try:
            for chunk in self._chunker(file_path, start_line):
                chunk_dict = chunk.model_dump()
                # Add machine_id to chunk
                chunk_dict["machine_id"] = settings.machine_id
                chunk_batch.append(chunk_dict)
                if chunk.source_line > max_line:
                    max_line = chunk.source_line

                # Process batch when it reaches max size
                if len(chunk_batch) >= max_chunks:
                    success = await self._upload_chunks_to_server(
                        chunk_batch, file_path, max_line, api_client, state_manager
                    )
                    if success:
                        total_chunks_processed += len(chunk_batch)
                    chunk_batch.clear()
                    gc.collect()

            # Process final partial batch
            if chunk_batch:
                success = await self._upload_chunks_to_server(
                    chunk_batch, file_path, max_line, api_client, state_manager
                )
                if success:
                    total_chunks_processed += len(chunk_batch)
                chunk_batch.clear()

        except Exception as e:
            logger.error(f"Failed to chunk {file_path.name}: {type(e).__name__}")
            record_error(
                "chunking",
                f"Failed to chunk file: {type(e).__name__}",
                {"file": file_path.name, "error": str(e)},
            )
            self._failed_files.add(path_str)
            return

        # Update local position regardless of upload success
        # (we've chunked the data, it will be uploaded when connection returns)
        final_line = max(max_line, total_lines)
        if final_line > start_line:
            await state_manager.update_local_position(path_str, final_line)

        elapsed = time.time() - start_time
        logger.info(
            f"[CLIENT] Processed {total_chunks_processed} chunks from {file_path.name} "
            f"in {elapsed:.2f}s"
        )

        self._last_indexed_file = file_path.name
        self._last_indexed_at = datetime.now(timezone.utc)
        self._failed_files.discard(path_str)

    async def _index_file(self, file_path: Path, embedder: "AsyncEmbedder", store) -> None:
        """Index a single file from its last known position."""
        # LOW #2: Use Path consistently instead of str for dict keys
        path_str = str(file_path)
        if not self.is_allowed_history_path(file_path):
            logger.warning(
                "Ignoring path outside %s watcher root: %s",
                self.source_name,
                file_path.name,
            )
            return

        # MEDIUM #3: Skip files that have failed recently
        if path_str in self._failed_files:
            logger.debug(f"Skipping previously failed file: {file_path.name}")
            return

        start_line = self.state.get_position(path_str)

        # LOW #4: Track time taken for indexing
        start_time = time.time()

        # MEDIUM #2: Eliminate double file read - track max line from chunker
        # Get total lines for logging only
        total_lines = _count_file_lines(file_path)

        # LOW #3: Include total_lines in log message
        logger.info(f"Indexing {file_path.name} from line {start_line} (total_lines={total_lines})")

        # Stream chunks in batches to avoid loading huge files into memory
        max_chunks = settings.max_chunks_per_file
        chunk_batch = []
        max_line = start_line
        total_chunks_stored = 0
        batch_num = 0

        try:
            for chunk in self._chunker(file_path, start_line):
                chunk_dict = chunk.model_dump()
                chunk_dict["machine_id"] = settings.machine_id
                chunk_batch.append(chunk_dict)
                if chunk.source_line > max_line:
                    max_line = chunk.source_line

                # Process batch when it reaches max size
                if len(chunk_batch) >= max_chunks:
                    batch_num += 1
                    logger.info(
                        f"Processing chunk batch {batch_num} ({len(chunk_batch)} chunks) from {file_path.name}"
                    )

                    # Embed and store this batch
                    stored_count = await self._embed_and_store_batch(
                        chunk_batch, file_path, embedder, store
                    )
                    if stored_count is None:
                        # Embedding or storage failed
                        self._failed_files.add(path_str)
                        return
                    total_chunks_stored += stored_count

                    # Clear batch to free memory
                    chunk_batch.clear()
                    gc.collect()

            # Process final partial batch
            if chunk_batch:
                batch_num += 1
                logger.info(
                    f"Processing final chunk batch {batch_num} ({len(chunk_batch)} chunks) from {file_path.name}"
                )
                stored_count = await self._embed_and_store_batch(
                    chunk_batch, file_path, embedder, store
                )
                if stored_count is None:
                    self._failed_files.add(path_str)
                    return
                total_chunks_stored += stored_count
                chunk_batch.clear()

        except Exception as e:
            logger.error(f"Failed to chunk {file_path.name}: {type(e).__name__}")
            record_error(
                "chunking",
                f"Failed to chunk file: {type(e).__name__}",
                {"file": file_path.name, "error": str(e)},
            )
            self._failed_files.add(path_str)
            return

        if total_chunks_stored == 0:
            logger.debug(f"No new chunks in {file_path.name}")
            # Update state to mark file as processed (prevents re-scanning)
            # Use max of max_line, total_lines, or at least 1 for empty files
            final_line = max(max_line, total_lines, 1 if total_lines == 0 else 0)
            if final_line > start_line or start_line == 0:
                self.state.set_position(path_str, final_line)
                self.state.save()
            self._last_indexed_file = file_path.name
            self._last_indexed_at = datetime.now(timezone.utc)
            return

        # Log completion
        elapsed = time.time() - start_time
        logger.info(f"Indexed {total_chunks_stored} chunks from {file_path.name} in {elapsed:.2f}s")

        # Update state ONLY after successful storage (prevents data loss)
        # MEDIUM #2: Use max_line from chunker instead of total_lines
        final_line = max(max_line, total_lines)
        if final_line > start_line:
            self.state.set_position(path_str, final_line)
            self.state.save()

        # MEDIUM #3: Clear from failed files on success
        self._failed_files.discard(path_str)
        self._last_indexed_file = file_path.name
        self._last_indexed_at = datetime.now(timezone.utc)

    async def startup_sync(self) -> None:
        """Scan all files and index any new content.

        Handles both server mode and client mode.
        In client mode, also syncs positions with server and processes pending uploads.
        """
        mode_str = "CLIENT" if settings.is_client_mode else "SERVER"
        logger.info(f"[{mode_str}] Starting sync in: {self.projects_path}")

        if not self.projects_path.exists():
            logger.error(
                f"CRITICAL: Projects path does not exist: {self.projects_path}. "
                f"No conversation history will be indexed. Please ensure {self._source_name} "
                f"has created conversation history files."
            )
            return

        # Log directory contents for debugging
        try:
            all_files = list(self.projects_path.glob("**/*"))
            logger.info(
                f"Directory scan: found {len(all_files)} total files/dirs in {self.projects_path}"
            )
            logger.debug(f"Sample files: {[f.name for f in all_files[:10]]}")
        except Exception as e:
            logger.error(f"Failed to scan directory {self.projects_path}: {e}")

        # Initialize mode-specific resources
        if settings.is_client_mode:
            from claude_history_rag.api_client import get_api_client
            from claude_history_rag.client_state import get_client_state_manager

            api_client = get_api_client()
            state_manager = get_client_state_manager()
            embedder = None

            # In client mode, sync positions with server first
            logger.info("[CLIENT] Syncing positions with server...")
            await self._sync_positions_with_server(api_client, state_manager)

            # Process any pending uploads from previous sessions
            logger.info("[CLIENT] Processing pending uploads...")
            await self._process_pending_uploads(api_client, state_manager)

            # Clear stale pending uploads
            await state_manager.clear_stale_pending_uploads()
        else:
            from claude_history_rag.store import store as chunk_store

            if (
                settings.storage_backend == "spanner"
                and settings.spanner_embedding_mode == "spanner"
            ):
                embedder = None
            else:
                from claude_history_rag.embedder import get_embedder

                embedder = get_embedder()
            api_client = None
            state_manager = None

        # Find all history files, filtering out symlinks outside projects_path
        jsonl_files = []
        for file_path in self.projects_path.glob("**/*"):
            if _is_safe_path(file_path, self.projects_path):
                if self._path_filter(file_path):
                    jsonl_files.append(file_path)
                logger.debug(
                    f"Found JSONL file: {file_path.name} ({file_path.stat().st_size} bytes)"
                )
            else:
                logger.info(f"Ignoring symlink outside projects path: {file_path.name}")

        if len(jsonl_files) == 0:
            logger.warning(
                f"No JSONL files found in {self.projects_path}. "
                f"This means no conversation history exists yet. "
                f"The watcher will monitor for new files."
            )
        else:
            logger.info(f"Found {len(jsonl_files)} JSONL files to index")

        indexed_count = 0
        failed_count = 0
        batch_size = settings.max_file_batch_size

        for idx, file_path in enumerate(jsonl_files, 1):
            try:
                if settings.is_client_mode:
                    await self._index_file_client_mode(file_path, api_client, state_manager)
                else:
                    await self._index_file(file_path, embedder, chunk_store)
                indexed_count += 1
            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to index {file_path.name}: {e}", exc_info=True)
                record_error(
                    "indexing",
                    f"Failed to index file: {type(e).__name__}",
                    {"file": file_path.name, "error": str(e)},
                )

            # Throttle indexing if configured (helps prevent system overload)
            if settings.startup_indexing_delay_ms > 0:
                await asyncio.sleep(settings.startup_indexing_delay_ms / 1000.0)
            else:
                # Yield control to event loop
                await asyncio.sleep(0)

            # Run garbage collection after processing a batch of files
            if settings.gc_after_files and idx % batch_size == 0:
                logger.info(f"Processed {idx}/{len(jsonl_files)} files, running garbage collection")
                gc.collect()

        # Final GC after all files processed
        if settings.gc_after_files:
            logger.info("All files processed, running final garbage collection")
            gc.collect()

        logger.info(
            f"[{mode_str}] Startup sync complete: indexed {indexed_count} files, "
            f"failed {failed_count} files"
        )

    async def start(self) -> None:
        """Start the file watcher."""
        if self._running:
            logger.warning("Watcher already running")
            return

        self._running = True
        # Reset shutdown event for new start
        self._shutdown_event.clear()

        # Run startup sync first (unless deferred)
        if not settings.defer_startup_indexing:
            await self.startup_sync()
        else:
            logger.info(
                "Startup indexing deferred (defer_startup_indexing=True). "
                "Files will be indexed as they are modified."
            )

        # Start watch and process tasks
        if settings.is_client_mode:
            from claude_history_rag.api_client import get_api_client
            from claude_history_rag.client_state import get_client_state_manager

            api_client = get_api_client()
            state_manager = get_client_state_manager()
            self._client_sync_task = asyncio.create_task(
                self._client_sync_loop(api_client, state_manager)
            )

        self._watch_task = asyncio.create_task(self._watch_files())
        self._process_task = asyncio.create_task(self._process_files())

        logger.info("File watcher started")

    async def queue_all_files_for_indexing(self) -> int:
        """Queue all JSONL files for indexing.

        This is useful for manually triggering a full re-index of all files,
        especially when DEFER_STARTUP_INDEXING is enabled.

        Returns:
            Number of files queued for indexing.
        """
        if not self.projects_path.exists():
            logger.warning(f"Projects path does not exist: {self.projects_path}")
            return 0

        queued_count = 0
        for file_path in self.projects_path.glob("**/*"):
            if _is_safe_path(file_path, self.projects_path):
                if not self._path_filter(file_path):
                    continue
                try:
                    # Use put_nowait to avoid blocking if queue is full
                    self.queue.put_nowait(file_path)
                    queued_count += 1
                except asyncio.QueueFull:
                    logger.warning(
                        f"Queue is full ({self.queue.maxsize} items), "
                        f"could not queue {file_path.name}"
                    )
                    break

        logger.info(f"Queued {queued_count} files for indexing")
        return queued_count

    async def force_full_reindex(self) -> tuple[int, int]:
        """Reset all file positions and queue all files for re-indexing.

        This is a destructive operation that will re-process all files from scratch.

        Returns:
            Tuple of (files_reset, files_queued).
        """
        # Reset all positions first
        files_reset = self.state.reset_all_positions()

        # Clear failed files set so they get retried
        self._failed_files.clear()

        # Queue all files
        files_queued = await self.queue_all_files_for_indexing()

        logger.info(f"Force re-index: reset {files_reset} positions, queued {files_queued} files")
        return files_reset, files_queued

    async def stop(self) -> None:
        """Stop the file watcher."""
        if not self._running:
            return

        logger.info("Stopping file watcher...")
        self._running = False

        # Signal shutdown event to wake up the process task
        self._shutdown_event.set()

        # Cancel watcher first (stops producer)
        if self._watch_task:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task

        # Drain queue with timeout before stopping consumer
        try:
            await asyncio.wait_for(self.queue.join(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning(f"Queue drain timeout, {self.queue.qsize()} items remaining")

        # Then cancel consumer
        if self._process_task:
            self._process_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._process_task

        # Cancel client sync loop
        if self._client_sync_task:
            self._client_sync_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._client_sync_task

        # Save final state
        self.state.save()
        logger.info("File watcher stopped")


# Global watcher instance
watcher: HistoryWatcher | None = None
# LOW #6: Add threading.Lock for global watcher singleton like embedder
_watcher_lock = threading.Lock()


def get_watcher() -> HistoryWatcher:
    """Get or create the global watcher instance (thread-safe)."""
    global watcher
    if watcher is None:
        with _watcher_lock:
            # Double-check pattern for thread safety
            if watcher is None:
                watcher = HistoryWatcher()
    return watcher


def get_all_watchers() -> list[HistoryWatcher]:
    """Return all configured local history watchers."""
    from claude_history_rag.antigravity.watcher import get_antigravity_watcher
    from claude_history_rag.chatgpt.watcher import get_chatgpt_watcher
    from claude_history_rag.claude_app.watcher import get_claude_app_watcher
    from claude_history_rag.codex.watcher import get_codex_watcher
    from claude_history_rag.gemini.watcher import get_gemini_watcher

    return [
        get_watcher(),
        get_codex_watcher(),
        get_gemini_watcher(),
        get_antigravity_watcher(),
        get_chatgpt_watcher(),
        get_claude_app_watcher(),
    ]
