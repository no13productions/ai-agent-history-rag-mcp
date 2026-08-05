"""File watcher for Claude Code history files using watchfiles.

Supports two modes:
- Server mode: Embed and store chunks locally
- Client mode: Batch chunks and upload to central server
"""

import asyncio
import contextlib
import gc
import hashlib
import inspect
import json
import logging
import os
import platform
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import psutil
from watchfiles import Change, awatch

from claude_history_rag import __version__ as package_version
from claude_history_rag import durable_io
from claude_history_rag.chunker import chunk_session_file, resolve_safe_session_interval
from claude_history_rag.client_state import CatchupInterval
from claude_history_rag.config import settings
from claude_history_rag.errors import record_error
from claude_history_rag.models import (
    MAX_CHUNK_UPLOAD_REQUEST_BYTES,
    Chunk,
    chunk_upload_request_bytes,
    chunk_upload_request_sha256,
)

if TYPE_CHECKING:
    from claude_history_rag.api_client import APIClient
    from claude_history_rag.client_state import ClientStateManager
    from claude_history_rag.embedder import AsyncEmbedder

logger = logging.getLogger(__name__)

CursorSemantics = Literal["bounded_physical_lines", "unsupported_snapshot"]
MAX_WATCHER_STATE_BYTES = 8 * 1024 * 1024
# Upper bound on one snapshotted history source. Chosen far above any observed
# history file so ordinary indexing is never converted into a failure, while an
# unbounded source still fails closed instead of filling the snapshot volume.
MAX_SOURCE_SNAPSHOT_BYTES = 512 * 1024 * 1024
PATH_FILTER_REJECTED = "path_filter_rejected"
# Exactly the conditions that mean "this source cannot be safely served". A
# transient durable-write failure is deliberately NOT one of them: that must
# stay retryable rather than permanently condemning the source.
#
# UnicodeDecodeError belongs here because a parse that aborts partway through a
# source must never look like a completed read: the bounds were derived from the
# whole source, so committing them would advance a cursor over content that was
# never parsed.
UNSAFE_SOURCE_ERRORS = (
    durable_io.UnsafeDurablePathError,
    durable_io.DurableRootUnavailableError,
    durable_io.DurableSizeLimitExceeded,
    UnicodeDecodeError,
)


def _log_safe_failure(
    event: str,
    error: BaseException,
    *,
    source_hash: str | None = None,
    phase: str | None = None,
) -> None:
    """Emit one stable failure event without exception text or traceback."""
    logger.error(
        "%s: reason=operation_failed error_type=%s source=%s phase=%s",
        event,
        type(error).__name__,
        source_hash or "none",
        phase or "none",
    )


def _is_safe_path(path: Path, base_path: Path) -> bool:
    """Check whether path is lexically inside base_path.

    ``Path.resolve()`` is deliberately not used here. It answers where a name
    points right now, so a base directory replaced by a link makes it report
    containment for content outside the watched subtree. Containment is decided
    lexically; whether the base is still the same object is proved separately by
    the pinned root's held handle.
    """
    return durable_io.lexical_relative_parts(path, base_path) is not None


def _accepts_source_path(chunker: Callable[..., Any]) -> bool:
    """Whether a chunker can be told the provenance path behind a snapshot.

    Only an explicitly declared parameter counts. A callable that merely absorbs
    ``**kwargs`` would accept the provenance and silently ignore it, deriving
    project and session identity from the snapshot location instead, which is
    exactly the failure this parameter exists to prevent.
    """
    try:
        parameters = inspect.signature(chunker).parameters
    except (TypeError, ValueError):
        return False
    return "source_path" in parameters


def _durable_failure_reason(refusal_reason: str) -> str:
    """Map a refusal code to the durable catch-up failure vocabulary.

    The top-severity distinction is preserved deliberately: an operator must be
    able to tell "this file is too big to snapshot" from "the watched root was
    substituted", because detecting the latter is what this authority exists
    for. Collapsing every refusal into one code destroys exactly that signal.
    """
    if refusal_reason == durable_io.SOURCE_NOT_DECODABLE:
        return "source_not_decodable"
    if refusal_reason == durable_io.SOURCE_TOO_LARGE:
        return "source_too_large"
    if refusal_reason in {
        durable_io.ROOT_IDENTITY_CHANGED,
        durable_io.ROOT_IS_LINK_OR_REPARSE,
        durable_io.ROOT_NOT_A_DIRECTORY,
        durable_io.ROOT_UNAVAILABLE,
        durable_io.ROOT_UNBOUND,
    }:
        return "watch_root_unusable"
    return "source_outside_authority"


class _SourceSnapshot:
    """One immutable copy of a history source plus its original provenance.

    Counting, digesting, interval resolution and parsing all read this one copy,
    so they cannot disagree with each other the way separate opens of a mutable
    pathname can. Provenance stays the original path: the snapshot location is
    an implementation detail and must never be stored anywhere.
    """

    def __init__(self, path: Path, source_path: Path):
        self.path = path
        self.source_path = source_path
        self._line_count: int | None = None
        self._digests: dict[int, str] = {}

    def line_count(self) -> int:
        """Total physical lines in this snapshot."""
        if self._line_count is None:
            self._line_count = _count_file_lines(self.path)
        return self._line_count

    def prefix_digest(self, end_inclusive: int) -> str:
        """Digest of this snapshot's physical-line prefix."""
        digest = self._digests.get(end_inclusive)
        if digest is None:
            digest = _prefix_digest(self.path, end_inclusive)
            self._digests[end_inclusive] = digest
        return digest


def _count_file_lines(file_path: Path) -> int:
    """Count total physical lines, exactly as the prefix digest counts them.

    Read as bytes so the count never depends on decodability. Returning 0 for an
    undecodable source would be a silent lie to a cursor authority: the caller
    would conclude there is nothing past the current position and skip the
    source with no diagnostic at all. Counting bytes also keeps this function
    and _prefix_digest on one definition of a line.
    """
    with open(file_path, "rb") as source:
        return sum(1 for _ in source)


def _prefix_digest(file_path: Path, end_inclusive: int) -> str:
    """Hash the exact physical-line prefix bound to queued catch-up work."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as source:
        for line_number, line in enumerate(source, start=1):
            if line_number > end_inclusive:
                break
            digest.update(line)
    return digest.hexdigest()


def _machine_scoped_chunk_id(machine_id: str, chunk_id: Any) -> str:
    """Return a deterministic machine-scoped storage id for shared backends."""
    return hashlib.sha256(f"{machine_id}\x00{chunk_id or ''}".encode()).hexdigest()


def _scope_chunk_for_direct_store(chunk: dict[str, Any], machine_id: str) -> dict[str, Any]:
    """Scope direct-write chunk ids so multiple machines can share one database."""
    scoped = dict(chunk)
    scoped["id"] = _machine_scoped_chunk_id(machine_id, scoped.get("id"))
    scoped["machine_id"] = machine_id

    parent_chunk_id = scoped.get("parent_chunk_id")
    if parent_chunk_id is not None:
        scoped["parent_chunk_id"] = _machine_scoped_chunk_id(machine_id, parent_chunk_id)

    child_chunk_ids = scoped.get("child_chunk_ids")
    if isinstance(child_chunk_ids, list):
        scoped["child_chunk_ids"] = [
            _machine_scoped_chunk_id(machine_id, child_id) for child_id in child_chunk_ids
        ]

    return scoped


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
) -> bool:
    """Handle a server-triggered reindex in client mode."""
    if not reindex_requested_at:
        raise ValueError("reindex flag missing reindex_requested_at")
    try:
        parsed_at = datetime.fromisoformat(reindex_requested_at)
        if parsed_at.tzinfo is None:
            raise ValueError("reindex_requested_at must include a timezone")
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid reindex_requested_at") from exc

    phase = await state_manager.prepare_reindex(parsed_at)
    if phase == "completed":
        return False
    if phase == "in_progress" and (await state_manager.get_state()).reindex_ack_at:
        return False

    invalidated_current_work = phase in {"reset", "resume"}
    if invalidated_current_work:
        queued = await _queue_all_watchers_for_reindex()
        await state_manager.mark_reindex_queued()
        reindex_state = await state_manager.get_state()
        logger.warning(
            "Committed reindex queue phase: request=%s phase=%s generation=%d queued=%d",
            reindex_requested_at,
            phase,
            reindex_state.reindex_generation,
            queued,
        )

    try:
        await api_client.ack_reindex(
            reindex_requested_at=reindex_requested_at,
            status="queued",
            reason=reason,
        )
    except Exception as e:
        logger.warning(
            "External reindex queued acknowledgment failed: request=%s phase=%s error_type=%s",
            reindex_requested_at,
            phase,
            type(e).__name__,
        )
        return invalidated_current_work
    try:
        await state_manager.set_reindex_ack(status="queued")
        state = await state_manager.get_state()
        logger.info(
            "Persisted reindex queued acknowledgment: request=%s generation=%d",
            reindex_requested_at,
            state.reindex_generation,
        )
    except Exception as e:
        logger.error(
            "Local reindex queued acknowledgment commit failed after external success: request=%s phase=%s error_type=%s",
            reindex_requested_at,
            phase,
            type(e).__name__,
        )
    return invalidated_current_work


async def _maybe_ack_reindex_completed(
    api_client: "APIClient",
    state_manager: "ClientStateManager",
) -> None:
    """Send a completed ack once a reindex has finished uploading."""
    state = await state_manager.get_state()
    if not state.reindex_required_at:
        return
    if state.reindex_status != "queued":
        return
    if state.pending_uploads:
        return
    # A TERMINAL failure is not outstanding work: it will never resolve by
    # waiting, so blocking on it means the server never learns this client
    # finished, for every source on the machine, forever. Completion is
    # acknowledged with a reason that names the shortfall instead, and the
    # failures stay visible in the heartbeat.
    terminal_failures = len(state.catchup_failures)

    if any(
        watcher.queue.qsize() > 0 or watcher._active_queue_claims > 0
        for watcher in get_all_watchers()
    ):
        return

    if state.last_server_sync and state.last_server_sync < state.reindex_required_at:
        return
    if not state.last_server_sync and state.local_positions:
        return

    try:
        await api_client.ack_reindex(
            reindex_requested_at=state.reindex_required_at.isoformat(),
            status="completed",
            reason=(
                f"uploads_finished_with_{terminal_failures}_unreadable_sources"
                if terminal_failures
                else "uploads_finished"
            ),
        )
    except Exception as e:
        logger.warning(
            "External reindex completion acknowledgment failed: request=%s generation=%d error_type=%s",
            state.reindex_required_at.isoformat(),
            state.reindex_generation,
            type(e).__name__,
        )
        return
    try:
        await state_manager.set_reindex_ack(status="completed")
        logger.info(
            "Persisted reindex completion acknowledgment: request=%s generation=%d",
            state.reindex_required_at.isoformat(),
            state.reindex_generation,
        )
    except Exception as e:
        logger.error(
            "Local reindex completion commit failed after external success: request=%s generation=%d error_type=%s",
            state.reindex_required_at.isoformat(),
            state.reindex_generation,
            type(e).__name__,
        )


class FilePositionState:
    """Track file positions for incremental reading."""

    def __init__(self, state_path: Path | None = None):
        self.state_path = state_path or settings.state_path
        self._positions: dict[str, int] = {}
        self._durable_positions: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        """Load state from disk."""
        try:
            if durable_io.durable_file_exists(self.state_path, durable_root=self.state_path.parent):
                raw = durable_io.read_text(
                    self.state_path,
                    durable_root=self.state_path.parent,
                    max_bytes=MAX_WATCHER_STATE_BYTES,
                )
                data = json.loads(raw)
                positions = data.get("file_positions", {})
                if not isinstance(positions, dict) or any(
                    not isinstance(path, str) or not isinstance(position, int) or position < 0
                    for path, position in positions.items()
                ):
                    raise ValueError("watcher state has invalid file positions")
                self._positions = positions
                self._durable_positions = dict(positions)
                logger.info(
                    "Loaded watcher state: files=%d bytes=%d",
                    len(self._positions),
                    len(raw.encode("utf-8")),
                )
        except (json.JSONDecodeError, OSError, ValueError) as e:
            _log_safe_failure("Failed to load watcher state", e)
            raise RuntimeError("durable watcher state could not be loaded") from e

    def save(self) -> None:
        """Save state to disk atomically."""
        try:
            durable_io.atomic_write_text(
                self.state_path,
                json.dumps({"file_positions": self._positions}),
                durable_root=self.state_path.parent,
            )
        except durable_io.DurableCommitUncertainError as e:
            if e.committed:
                self._durable_positions = dict(self._positions)
            _log_safe_failure("Failed to save watcher state", e)
            raise RuntimeError("durable watcher state could not be saved") from e
        except OSError as e:
            self._positions = dict(self._durable_positions)
            _log_safe_failure("Failed to save watcher state", e)
            raise RuntimeError("durable watcher state could not be saved") from e
        self._durable_positions = dict(self._positions)

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
        cursor_semantics: CursorSemantics = "bounded_physical_lines",
    ):
        self.projects_path = projects_path or settings.projects_path
        if debounce_ms < 0:
            raise ValueError(f"debounce_ms must be non-negative, got {debounce_ms}")
        self.debounce_ms = debounce_ms
        self.state = FilePositionState(state_path=state_path)
        self._chunker = chunker or chunk_session_file
        self._chunker_reports_provenance = _accepts_source_path(self._chunker)
        # The configured root is a mutable name; this binds the object behind it
        # once it first exists and refuses every other object thereafter.
        self._root = durable_io.PinnedRoot(self.projects_path)
        self._root_hash = hashlib.sha256(str(self.projects_path).encode("utf-8")).hexdigest()[:12]
        self._source_name = source_name
        self._path_filter = path_filter or (lambda p: p.suffix == ".jsonl")
        self._cursor_semantics = cursor_semantics
        # Limit queue size to prevent unbounded memory growth
        # 10000 allows for large initial indexing while preventing memory issues
        self.queue: asyncio.Queue[Path | CatchupInterval] = asyncio.Queue(maxsize=10000)
        self._queued_catchups: set[tuple[str, int, int, int]] = set()
        self._queued_paths: set[str] = set()
        self._active_queue_claims = 0
        self._running = False
        self._watch_task: asyncio.Task | None = None
        self._process_task: asyncio.Task | None = None
        self._client_sync_task: asyncio.Task | None = None
        # MEDIUM #3: Track failed files separately to avoid infinite retry loops
        self._failed_files: set[str] = set()
        self._shutdown_event = asyncio.Event()
        self._last_indexed_file_hash: str | None = None
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
        """Return stable source hashes for status reporting."""
        return [self._hash_value(path) for path in self._failed_files]

    def clear_failed_files(self) -> None:
        """Clear failed file tracking before a forced reindex."""
        self._failed_files.clear()

    def _verify_root(self) -> bool:
        """Bind or re-verify the watch root object, never rebinding to a new one.

        A root that does not exist yet is bound the first time a valid ordinary
        directory appears under that name. Once bound, a missing, replaced or
        substituted root fails closed rather than silently following the name to
        whatever object now answers to it.
        """
        try:
            if self._root.bind():
                return True
            reason = durable_io.ROOT_UNBOUND
        except OSError as error:
            reason = getattr(error, "reason", durable_io.ROOT_UNAVAILABLE)
        logger.warning(
            "Watch root unavailable: reason=%s root=%s source_type=%s",
            reason,
            self._root_hash,
            self._source_name,
        )
        return False

    def _classify_history_path(self, path: Path) -> str:
        """Return "ok" or the fixed reason this path cannot be served."""
        if not self._path_filter(path):
            return PATH_FILTER_REJECTED
        return self._root.classify(path)

    @contextlib.contextmanager
    def _source_snapshot(self, file_path: Path) -> Iterator[_SourceSnapshot]:
        """Yield one immutable snapshot of a history source under the pinned root."""
        with self._root.snapshot(file_path, max_bytes=MAX_SOURCE_SNAPSHOT_BYTES) as snapshot_path:
            yield _SourceSnapshot(snapshot_path, file_path)

    def _refuse_unsafe_source(
        self,
        path_str: str,
        error: BaseException,
        *,
        phase: str,
    ) -> str:
        """Record one fail-closed refusal without advancing any durable state.

        The source is also marked failed so a permanently unsafe path is not
        retried forever by the queue. Returns the fixed reason code so callers
        can record the same terminal observation against a bounded interval.
        """
        if isinstance(error, durable_io.DurableSizeLimitExceeded):
            reason = durable_io.SOURCE_TOO_LARGE
        elif isinstance(error, UnicodeDecodeError):
            reason = durable_io.SOURCE_NOT_DECODABLE
        else:
            reason = getattr(error, "reason", durable_io.DESCENDANT_NOT_TRAVERSABLE)
        source_hash = self._hash_value(path_str)
        logger.error(
            "Refusing unsafe source: reason=%s phase=%s source_type=%s source=%s",
            reason,
            phase,
            self._source_name,
            source_hash,
        )
        record_error(
            "filesystem",
            f"Refused unsafe source: {reason}",
            {"source_hash": source_hash, "reason": reason},
        )
        self._failed_files.add(path_str)
        return reason

    def _chunk_snapshot(
        self,
        snapshot: _SourceSnapshot,
        start_line: int,
        end_line: int | None = None,
    ) -> Iterator[Chunk]:
        """Chunk a snapshot while reporting the original source as provenance."""
        arguments: list[Any] = [snapshot.path, start_line]
        if end_line is not None:
            arguments.append(end_line)
        keywords = {"source_path": snapshot.source_path} if self._chunker_reports_provenance else {}
        provenance = str(snapshot.source_path)
        for chunk in self._chunker(*arguments, **keywords):
            # A snapshot location must never survive into stored provenance.
            if chunk.source_file != provenance:
                chunk = chunk.model_copy(update={"source_file": provenance})
            yield chunk

    def discover_files(self) -> list[Path]:
        """Return currently discoverable history files for this watcher."""
        if not self._verify_root():
            return []
        return [
            file_path
            for file_path in self._root.path.glob("**/*")
            if _is_safe_path(file_path, self._root.path) and self._path_filter(file_path)
        ]

    def is_allowed_history_path(self, path: Path) -> bool:
        """Return whether a path is served by this watcher's pinned root and filter."""
        return self._classify_history_path(path) == "ok"

    async def _watch_files(self) -> None:
        """Producer: watch for file changes."""
        logger.info("Starting file watcher: root=%s", self._root_hash)

        if not self._verify_root():
            return

        try:
            async for changes in awatch(
                self._root.path,
                watch_filter=lambda _, p: self._path_filter(Path(p)),
                debounce=self.debounce_ms,
                recursive=True,
            ):
                if not self._running:
                    break

                for change_type, path_str in changes:
                    if change_type in (Change.added, Change.modified):
                        path = Path(path_str)
                        # Containment is decided lexically here; the pinned root
                        # proves object identity when the source is opened.
                        if not _is_safe_path(path, self._root.path):
                            logger.warning(
                                "Ignoring change outside watch root: reason=%s source=%s",
                                durable_io.PATH_OUTSIDE_ROOT,
                                self._hash_value(str(path)),
                            )
                            continue
                        logger.debug("File changed: source=%s", self._hash_value(str(path)))
                        await self.queue.put(path)

        except Exception as e:
            logger.error("Watcher error: error_type=%s", type(e).__name__)
            record_error(
                "watcher",
                f"File watcher error: {type(e).__name__}",
                {"error_type": type(e).__name__},
            )

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
            queue_get_task: asyncio.Task | None = None
            shutdown_task: asyncio.Task | None = None
            work: Path | CatchupInterval | None = None
            queue_claim_balanced = False
            claimed_raw_path: Path | None = None
            raw_claim_durable = False
            raw_claim_requeued = False
            try:
                # MEDIUM #1: Use asyncio.wait instead of timeout polling
                queue_get_task = asyncio.create_task(self.queue.get())
                shutdown_task = asyncio.create_task(self._shutdown_event.wait())

                done, pending = await asyncio.wait(
                    {queue_get_task, shutdown_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel and reap only tasks that did not win the race. A queue
                # get that completed alongside shutdown has claimed real work
                # and must be balanced before the loop exits.
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

                if queue_get_task not in done:
                    break

                try:
                    work = await queue_get_task
                    self._active_queue_claims += 1
                    if isinstance(work, Path):
                        claimed_raw_path = work
                    path = Path(work.file_path) if isinstance(work, CatchupInterval) else work
                    if settings.is_client_mode:
                        if (
                            isinstance(work, Path)
                            and self._cursor_semantics == "bounded_physical_lines"
                        ):
                            work = await self._durabilize_client_path(work, state_manager)
                            raw_claim_durable = True
                        if work is not None:
                            await self._index_file_client_mode(work, api_client, state_manager)
                    else:
                        await self._index_file(path, embedder, chunk_store)
                except asyncio.CancelledError:
                    if claimed_raw_path is not None and not raw_claim_durable:
                        self._guaranteed_put_nowait(claimed_raw_path)
                        raw_claim_requeued = True
                        logger.info(
                            "Restored cancelled raw claim: source=%s outcome=requeued queue_depth=%d",
                            self._hash_value(str(claimed_raw_path)),
                            self.queue.qsize(),
                        )
                    elif isinstance(work, CatchupInterval):
                        logger.info(
                            "Retained durable claim after cancellation: record=%s generation=%d source=%s interval=(%d,%d] outcome=durable_owner_retained",
                            work.outbox_record_id or "unknown",
                            work.generation,
                            self._hash_value(work.file_path),
                            work.start_exclusive,
                            work.end_inclusive,
                        )
                    raise
                except Exception as error:
                    if claimed_raw_path is not None and not raw_claim_durable:
                        self._guaranteed_put_nowait(claimed_raw_path)
                        raw_claim_requeued = True
                        logger.warning(
                            "Restored failed raw claim: source=%s outcome=requeued error_type=%s queue_depth=%d",
                            self._hash_value(str(claimed_raw_path)),
                            type(error).__name__,
                            self.queue.qsize(),
                        )
                    raise
                finally:
                    if isinstance(work, CatchupInterval):
                        self._queued_catchups.discard(
                            (
                                work.file_path,
                                work.start_exclusive,
                                work.end_inclusive,
                                work.generation,
                            )
                        )
                    self.queue.task_done()
                    self._active_queue_claims -= 1
                    if claimed_raw_path is not None and not raw_claim_requeued:
                        self._queued_paths.discard(str(claimed_raw_path))
                    queue_claim_balanced = True

                if shutdown_task in done:
                    break

            except asyncio.CancelledError:
                logger.debug("Process task cancelled after queue ownership reconciliation")
                raise  # Re-raise to allow proper cancellation
            except Exception as e:
                logger.error("Processing error: error_type=%s", type(e).__name__)
                record_error(
                    "processing",
                    f"File processing error: {type(e).__name__}",
                    {"error_type": type(e).__name__},
                )
            finally:
                children = [task for task in (queue_get_task, shutdown_task) if task is not None]
                for child in children:
                    if not child.done():
                        child.cancel()
                if children:
                    await asyncio.gather(*children, return_exceptions=True)
                if (
                    not queue_claim_balanced
                    and queue_get_task is not None
                    and queue_get_task.done()
                    and not queue_get_task.cancelled()
                ):
                    claimed = queue_get_task.result()
                    self.queue.task_done()
                    self._guaranteed_put_nowait(claimed)

    def _guaranteed_put_nowait(self, work: Path | CatchupInterval) -> None:
        """Restore an owned claim even when producers refilled the bounded queue."""
        original_maxsize = self.queue._maxsize
        try:
            self.queue._maxsize = 0
            self.queue.put_nowait(work)
        finally:
            self.queue._maxsize = original_maxsize

    def _queue_raw_path(self, file_path: Path) -> bool:
        """Queue one raw source path exactly once with guaranteed ownership."""
        key = str(file_path)
        if key in self._queued_paths:
            return False
        self._queued_paths.add(key)
        self._guaranteed_put_nowait(file_path)
        return True

    async def _embed_and_store_batch(
        self, chunk_batch: list[dict], file_path: Path, embedder: "AsyncEmbedder", store
    ) -> int | None:
        """Embed and store a batch of chunks. Returns chunk count or None on failure."""
        if settings.storage_backend == "spanner" and settings.spanner_embedding_mode == "spanner":
            try:
                await store.add_chunks_async(chunk_batch)
                return len(chunk_batch)
            except Exception as e:
                source_hash = self._hash_value(str(file_path))
                _log_safe_failure(
                    "Failed to store Spanner-native embedding batch",
                    e,
                    source_hash=source_hash,
                )
                record_error(
                    "database",
                    f"Failed to store Spanner-native embedding batch: {type(e).__name__}",
                    {"source_hash": source_hash, "error_type": type(e).__name__},
                )
                return None

        try:
            embedded_chunks = await embedder.embed_chunks(chunk_batch)
        except Exception as e:
            source_hash = self._hash_value(str(file_path))
            _log_safe_failure("Failed to embed batch", e, source_hash=source_hash)
            record_error(
                "embedding",
                f"Failed to embed batch: {type(e).__name__}",
                {"source_hash": source_hash, "error_type": type(e).__name__},
            )
            return None

        if not embedded_chunks:
            logger.warning(
                "No chunks embedded from batch: source=%s",
                self._hash_value(str(file_path)),
            )
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
            source_hash = self._hash_value(str(file_path))
            _log_safe_failure("Failed to store batch", e, source_hash=source_hash)
            record_error(
                "database",
                f"Failed to store batch: {type(e).__name__}",
                {"source_hash": source_hash, "error_type": type(e).__name__},
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
        retry_total = sum(record.retry_count for record in pending)
        blocked_records = sum(record.retry_count > 0 for record in pending)
        failure_reason_counts: dict[str, int] = {}
        failure_oldest_age_sec = None
        for failure in state.catchup_failures.values():
            failure_reason_counts[failure.reason] = failure_reason_counts.get(failure.reason, 0) + 1
            age = int((datetime.now(timezone.utc) - failure.observed_at).total_seconds())
            failure_oldest_age_sec = (
                age if failure_oldest_age_sec is None else max(failure_oldest_age_sec, age)
            )

        status = "ok"
        if (
            not state.connected
            or blocked_records
            or state.catchup_failures
            or self._active_queue_claims
        ):
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
                "active_claims": self._active_queue_claims,
                "blocked_records": blocked_records,
                "retry_total": retry_total,
            },
            "watcher": {
                "failed_files_count": len(self._failed_files),
                "debounce_ms": self.debounce_ms,
                "last_indexed_file_hash": self._last_indexed_file_hash,
            },
            "reindex": {
                "required_at": state.reindex_required_at.isoformat()
                if state.reindex_required_at
                else None,
                "ack_at": state.reindex_ack_at.isoformat() if state.reindex_ack_at else None,
                "status": state.reindex_status,
                "generation": state.reindex_generation,
            },
            "errors": {
                "count": retry_total + len(state.catchup_failures) + len(self._failed_files),
                "catchup_failure_count": len(state.catchup_failures),
                "catchup_failure_reasons": failure_reason_counts,
                "catchup_failure_oldest_age_sec": failure_oldest_age_sec,
            },
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
            logger.warning("Heartbeat failed: error_type=%s", type(e).__name__)

    async def _process_pending_uploads(
        self,
        api_client: "APIClient",
        state_manager: "ClientStateManager",
    ) -> int:
        """Give exactly one coroutine authority to drain this watcher's outbox."""
        async with state_manager.drain_lock:
            return await self._process_pending_uploads_unlocked(api_client, state_manager)

    async def _process_pending_uploads_unlocked(
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
        blocked_files: set[str] = set()
        current_generation = (await state_manager.get_state()).reindex_generation

        for upload in pending:
            if upload.file_path in blocked_files:
                continue
            if upload.generation != current_generation:
                logger.error(
                    "Refusing stale-generation outbox record %s: record=%d current=%d",
                    upload.record_id,
                    upload.generation,
                    current_generation,
                )
                blocked_files.add(upload.file_path)
                continue
            if upload.is_continuation:
                if upload.followup_interval is None:
                    retry_count = await state_manager.increment_retry_count(upload.record_id)
                    logger.error(
                        "Continuation record missing interval: record=%s generation=%d sequence=%d retry=%d",
                        upload.record_id,
                        upload.generation,
                        upload.sequence,
                        retry_count,
                    )
                    blocked_files.add(upload.file_path)
                    continue
                interval = upload.followup_interval.model_copy(
                    update={"outbox_record_id": upload.record_id}
                )
                await self._queue_catchup_interval(interval)
                blocked_files.add(upload.file_path)
                continue
            try:
                chunks = await state_manager.load_pending_chunks(upload)
                request_machine_id = getattr(api_client, "machine_id", settings.machine_id)
                request_client_name = getattr(
                    api_client,
                    "client_name",
                    settings.client_name or settings.machine_id,
                )
                request_bytes = chunk_upload_request_bytes(
                    chunks,
                    machine_id=request_machine_id,
                    client_name=request_client_name,
                    source_file=upload.file_path,
                    file_position=upload.file_position,
                )
                request_sha256 = chunk_upload_request_sha256(
                    chunks,
                    machine_id=request_machine_id,
                    client_name=request_client_name,
                    source_file=upload.file_path,
                    file_position=upload.file_position,
                )
                max_chunks = min(settings.max_chunks_per_file, 500)
                if len(chunks) > max_chunks or request_bytes > MAX_CHUNK_UPLOAD_REQUEST_BYTES:
                    retry_count = await state_manager.increment_retry_count(upload.record_id)
                    logger.error(
                        "Refusing unbounded outbox record: record=%s generation=%d sequence=%d source=%s chunks=%d bytes=%d retry=%d",
                        upload.record_id,
                        upload.generation,
                        upload.sequence,
                        self._hash_value(upload.file_path),
                        len(chunks),
                        request_bytes,
                        retry_count,
                    )
                    blocked_files.add(upload.file_path)
                    continue
                if (
                    request_machine_id != upload.request_machine_id
                    or request_client_name != upload.request_client_name
                    or request_sha256 != upload.request_sha256
                ):
                    retry_count = await state_manager.increment_retry_count(upload.record_id)
                    logger.error(
                        "Refusing request-binding mismatch: record=%s generation=%d sequence=%d source=%s retry=%d",
                        upload.record_id,
                        upload.generation,
                        upload.sequence,
                        self._hash_value(upload.file_path),
                        retry_count,
                    )
                    blocked_files.add(upload.file_path)
                    continue
                if request_bytes != upload.request_bytes:
                    retry_count = await state_manager.increment_retry_count(upload.record_id)
                    logger.error(
                        "Refusing request-size mismatch: record=%s generation=%d sequence=%d source=%s index=%d actual=%d retry=%d",
                        upload.record_id,
                        upload.generation,
                        upload.sequence,
                        self._hash_value(upload.file_path),
                        upload.request_bytes,
                        request_bytes,
                        retry_count,
                    )
                    blocked_files.add(upload.file_path)
                    continue
                response = await api_client.upload_chunks(
                    chunks=chunks,
                    source_file=upload.file_path,
                    file_position=upload.file_position,
                )

                complete_response = (
                    response.status == "ok"
                    and response.chunks_received == len(chunks)
                    and response.chunks_embedded == len(chunks)
                    and response.chunks_stored == len(chunks)
                )
                if complete_response:
                    if response.reindex_required:
                        invalidated = await _handle_server_reindex(
                            api_client,
                            state_manager,
                            response.reindex_requested_at,
                            reason="flagged_on_pending_upload",
                        )
                        if invalidated:
                            break
                    completed, durable_followup = await state_manager.complete_pending_upload(
                        upload.record_id
                    )
                    if completed and durable_followup is not None:
                        await self._queue_catchup_interval(durable_followup)
                    if completed:
                        logger.info(
                            "Durably acknowledged outbox record: record=%s generation=%d sequence=%d source=%s position=%d",
                            upload.record_id,
                            upload.generation,
                            upload.sequence,
                            self._hash_value(upload.file_path),
                            upload.file_position,
                        )
                        self._last_upload_at = datetime.now(timezone.utc)
                        successful += 1
                    else:
                        logger.info(
                            "Ignored stale upload completion after durable state reset: %s",
                            upload.record_id,
                        )
                else:
                    retry_count = await state_manager.increment_retry_count(upload.record_id)
                    logger.warning(
                        "Server rejected outbox record: record=%s generation=%d sequence=%d source=%s position=%d expected=%d received=%d embedded=%d stored=%d status=%s retry=%d",
                        upload.record_id,
                        upload.generation,
                        upload.sequence,
                        self._hash_value(upload.file_path),
                        upload.file_position,
                        len(chunks),
                        response.chunks_received,
                        response.chunks_embedded,
                        response.chunks_stored,
                        response.status,
                        retry_count,
                    )
                    blocked_files.add(upload.file_path)

            except ServerConnectionError:
                retry_count = await state_manager.increment_retry_count(upload.record_id)
                logger.warning(
                    "Server unavailable for outbox record: record=%s generation=%d sequence=%d source=%s retry=%d",
                    upload.record_id,
                    upload.generation,
                    upload.sequence,
                    self._hash_value(upload.file_path),
                    retry_count,
                )
                await state_manager.set_connected(False)
                break  # Don't try more if server is down
            except Exception as e:
                retry_count = await state_manager.increment_retry_count(upload.record_id)
                logger.error(
                    "Failed outbox record before acknowledgment: record=%s generation=%d sequence=%d source=%s position=%d error_type=%s retry=%d",
                    upload.record_id,
                    upload.generation,
                    upload.sequence,
                    self._hash_value(upload.file_path),
                    upload.file_position,
                    type(e).__name__,
                    retry_count,
                )
                blocked_files.add(upload.file_path)

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
                logger.warning("Failed to get server positions: reason=server_error_response")
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
            await state_manager.reconcile_server_positions(server_positions)
            catchup_files = await state_manager.get_files_needing_catchup(server_positions)

            if catchup_files:
                logger.info(f"Found {len(catchup_files)} files needing catch-up")
                # Re-queue these files for indexing
                for interval in catchup_files:
                    path = Path(interval.file_path)
                    state = await state_manager.get_state()
                    prior_failure = state.catchup_failures.get(interval.file_path)
                    reason = self._classify_history_path(path)
                    if reason == "ok" and path.exists():
                        if self._cursor_semantics != "bounded_physical_lines":
                            if not prior_failure or (
                                prior_failure.start_exclusive != interval.start_exclusive
                                or prior_failure.end_inclusive != interval.end_inclusive
                                or prior_failure.reason != "unsupported_cursor_contract"
                            ):
                                await state_manager.record_catchup_failure(
                                    interval, "unsupported_cursor_contract"
                                )
                            continue
                        try:
                            # One snapshot decides both the truncation verdict and
                            # the digest that seals the queued interval.
                            with self._source_snapshot(path) as snapshot:
                                total_lines = snapshot.line_count()
                                if total_lines < interval.end_inclusive:
                                    truncated = True
                                    sealed_digest = None
                                else:
                                    truncated = False
                                    sealed_digest = snapshot.prefix_digest(interval.end_inclusive)
                        except FileNotFoundError:
                            if not prior_failure or (
                                prior_failure.start_exclusive != interval.start_exclusive
                                or prior_failure.end_inclusive != interval.end_inclusive
                                or prior_failure.reason != "history_missing"
                            ):
                                await state_manager.handle_missing_history(interval)
                            continue
                        except UNSAFE_SOURCE_ERRORS as error:
                            self._refuse_unsafe_source(
                                interval.file_path,
                                error,
                                phase="position_sync",
                            )
                            continue
                        if truncated:
                            if not prior_failure or (
                                prior_failure.start_exclusive != interval.start_exclusive
                                or prior_failure.end_inclusive != interval.end_inclusive
                                or prior_failure.reason != "history_truncated"
                            ):
                                await state_manager.record_catchup_failure(
                                    interval, "history_truncated"
                                )
                            continue
                        await state_manager.clear_catchup_failure(interval.file_path)
                        work = interval.model_copy(update={"snapshot_digest": sealed_digest})
                        durable_work = await state_manager.add_continuation(work)
                        await self._queue_catchup_interval(durable_work)
                    elif reason != "ok" and path.exists():
                        logger.warning(
                            "Refusing catch-up source outside watcher authority: reason=%s source_type=%s source=%s",
                            reason,
                            self.source_name,
                            self._hash_value(str(path)),
                        )
                    else:
                        if not prior_failure or (
                            prior_failure.start_exclusive != interval.start_exclusive
                            or prior_failure.end_inclusive != interval.end_inclusive
                            or prior_failure.reason != "history_missing"
                        ):
                            await state_manager.handle_missing_history(interval)

            await state_manager.set_connected(True)

        except ServerConnectionError:
            logger.warning("Server unavailable for position sync")
            await state_manager.set_connected(False)

    async def _queue_catchup_interval(self, interval: CatchupInterval) -> bool:
        """Queue one immutable interval exactly once until it is consumed."""
        key = (
            interval.file_path,
            interval.start_exclusive,
            interval.end_inclusive,
            interval.generation,
        )
        if key in self._queued_catchups:
            return False
        self._queued_catchups.add(key)
        try:
            await self.queue.put(interval)
        except BaseException:
            self._queued_catchups.discard(key)
            raise
        return True

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
                    await self._process_pending_uploads(api_client, state_manager)
                    await self._sync_positions_with_server(api_client, state_manager)
                    await state_manager.clear_stale_pending_uploads()
                    await _maybe_ack_reindex_completed(api_client, state_manager)
                    next_sync = now + sync_interval

                if now >= next_heartbeat:
                    await self._send_client_heartbeat(api_client, state_manager)
                    self._last_heartbeat_at = time.monotonic()
                    next_heartbeat = now + heartbeat_interval
            except Exception as e:
                logger.warning("Client sync loop error: error_type=%s", type(e).__name__)

            try:
                now = time.monotonic()
                timeout = min(next_sync, next_heartbeat) - now
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=max(timeout, 0.5))
            except asyncio.TimeoutError:
                continue

    async def _durabilize_client_path(
        self,
        file_path: Path,
        state_manager: "ClientStateManager",
    ) -> CatchupInterval | None:
        """Convert one raw queue claim to a durable bounded continuation."""
        path_str = str(file_path)
        reason = self._classify_history_path(file_path)
        if reason != "ok":
            logger.warning(
                "[CLIENT] Refusing source outside watcher authority: reason=%s source_type=%s source=%s",
                reason,
                self.source_name,
                self._hash_value(path_str),
            )
            return None
        if path_str in self._failed_files:
            logger.debug("Skipping previously failed source: source=%s", self._hash_value(path_str))
            return None
        try:
            with self._source_snapshot(file_path) as snapshot:
                # Bounds and the digest that seals them come from one snapshot,
                # so the interval can never describe a file state that was never
                # observed as a whole.
                total_lines = snapshot.line_count()
                state = await state_manager.get_state()
                start_line = state.local_positions.get(path_str, 0)
                if total_lines <= start_line:
                    return None
                had_prior_work = any(
                    record.file_path == path_str for record in state.pending_uploads
                )
                durable_interval = await state_manager.add_continuation(
                    CatchupInterval(
                        file_path=path_str,
                        start_exclusive=start_line,
                        end_inclusive=total_lines,
                        snapshot_digest=snapshot.prefix_digest(total_lines),
                        generation=state.reindex_generation,
                    )
                )
        except FileNotFoundError:
            logger.info(
                "Source path disappeared before durable claim: source=%s",
                self._hash_value(path_str),
            )
            return None
        except UNSAFE_SOURCE_ERRORS as error:
            self._refuse_unsafe_source(path_str, error, phase="durable_claim")
            return None
        return None if had_prior_work else durable_interval

    async def _index_file_client_mode(
        self,
        work: Path | CatchupInterval,
        api_client: "APIClient",
        state_manager: "ClientStateManager",
    ) -> None:
        """Expand one durable source interval into bounded ordered outbox records."""
        interval = work if isinstance(work, CatchupInterval) else None
        file_path = Path(interval.file_path) if interval else work
        path_str = str(file_path)
        reason = self._classify_history_path(file_path)
        if reason != "ok":
            logger.warning(
                "[CLIENT] Refusing source outside watcher authority: reason=%s source_type=%s source=%s",
                reason,
                self.source_name,
                self._hash_value(path_str),
            )
            return

        # Skip files that have failed recently
        if path_str in self._failed_files:
            logger.debug("Skipping previously failed source: source=%s", self._hash_value(path_str))
            return

        if interval is None and self._cursor_semantics == "bounded_physical_lines":
            durable_interval = await self._durabilize_client_path(file_path, state_manager)
            if durable_interval is None:
                return
            await self._index_file_client_mode(durable_interval, api_client, state_manager)
            return

        start_time = time.time()
        try:
            with self._source_snapshot(file_path) as snapshot:
                await self._expand_snapshot_into_outbox(
                    snapshot,
                    interval,
                    api_client,
                    state_manager,
                    start_time,
                )
        except FileNotFoundError:
            if interval is not None:
                await state_manager.handle_missing_history(interval)
                return
            logger.info(
                "Source path disappeared before indexing: source=%s",
                self._hash_value(path_str),
            )
        except UNSAFE_SOURCE_ERRORS as error:
            reason = self._refuse_unsafe_source(path_str, error, phase="client_index")
            if interval is not None:
                # Terminal for this interval. Record the observation, then RETIRE
                # the continuation: a record that can never be expanded would
                # otherwise sit in the outbox forever. Completion acking is kept
                # unblocked separately, by treating terminal failures as reported
                # shortfalls rather than outstanding work.
                await state_manager.record_catchup_failure(
                    interval,
                    _durable_failure_reason(reason),
                )
                await state_manager.complete_continuation(interval.outbox_record_id)
        except OSError as error:
            # A read that aborted for any other reason is still an aborted read:
            # the bounds came from the whole source. Handled here rather than
            # left to escape, because an escaping error skips the failed-file
            # marking and leaves the continuation outstanding forever.
            self._refuse_unsafe_source(path_str, error, phase="client_index")
            if interval is not None:
                await state_manager.record_catchup_failure(interval, "source_outside_authority")
                await state_manager.complete_continuation(interval.outbox_record_id)

    async def _expand_snapshot_into_outbox(
        self,
        snapshot: _SourceSnapshot,
        interval: CatchupInterval | None,
        api_client: "APIClient",
        state_manager: "ClientStateManager",
        start_time: float,
    ) -> None:
        """Expand exactly one immutable snapshot into bounded outbox records."""
        path_str = str(snapshot.source_path)
        total_lines = snapshot.line_count()
        state = await state_manager.get_state()

        if interval:
            if interval.generation != state.reindex_generation:
                logger.info(
                    "Discarding stale catch-up generation: source=%s interval=%d current=%d",
                    self._hash_value(path_str),
                    interval.generation,
                    state.reindex_generation,
                )
                await state_manager.complete_continuation(interval.outbox_record_id)
                return
            if self._cursor_semantics != "bounded_physical_lines":
                await state_manager.record_catchup_failure(interval, "unsupported_cursor_contract")
                return
            if total_lines < interval.end_inclusive:
                await state_manager.record_catchup_failure(interval, "history_truncated")
                return
            # Validated once against the snapshot that will actually be parsed.
            # A second check after parsing would be tautological here, because
            # the snapshot cannot change underneath the parse.
            if interval.snapshot_digest and (
                snapshot.prefix_digest(interval.end_inclusive) != interval.snapshot_digest
            ):
                await state_manager.record_catchup_failure(interval, "history_replaced")
                return
            start_line = max(
                interval.start_exclusive,
                state.server_positions.get(path_str, interval.start_exclusive),
            )
            end_line = interval.end_inclusive
            if start_line >= end_line:
                committed_digest = state.committed_snapshot_digests.get(path_str)
                if interval.snapshot_digest and (
                    committed_digest is None or committed_digest != interval.snapshot_digest
                ):
                    await state_manager.record_catchup_failure(
                        interval,
                        "history_replaced"
                        if committed_digest is not None
                        else "history_continuity_unproven",
                    )
                    return
                await state_manager.clear_catchup_failure(path_str)
                followup_interval = None
                if start_line < total_lines:
                    followup_interval = CatchupInterval(
                        file_path=path_str,
                        start_exclusive=start_line,
                        end_inclusive=total_lines,
                        snapshot_digest=snapshot.prefix_digest(total_lines),
                        generation=interval.generation,
                    )
                durable_followup = await state_manager.replace_continuation(
                    interval.outbox_record_id, followup_interval
                )
                if durable_followup is not None:
                    await self._queue_catchup_interval(durable_followup)
                return
        else:
            # Snapshot adapters do not expose physical-line incremental semantics.
            # Preserve their existing full-snapshot behavior for ordinary file
            # events, while rejecting server catch-up intervals above.
            start_line = (
                state.local_positions.get(path_str, 0)
                if self._cursor_semantics == "bounded_physical_lines"
                else 0
            )
            end_line = total_lines

        if self._cursor_semantics == "bounded_physical_lines":
            parser_start, safe_end = resolve_safe_session_interval(
                snapshot.path, start_line, end_line
            )
        else:
            parser_start, safe_end = start_line, end_line

        # Old clients advanced local state to EOF even when EOF was an unpaired
        # user. Correct that unsafe legacy cursor to the last replayable boundary.
        if interval and safe_end < end_line and state.local_positions.get(path_str, 0) >= end_line:
            await state_manager.update_local_position(path_str, safe_end)

        followup_interval: CatchupInterval | None = None
        if interval and total_lines > end_line:
            followup_start = max(start_line, safe_end)
            if followup_start < total_lines:
                followup_interval = CatchupInterval(
                    file_path=path_str,
                    start_exclusive=followup_start,
                    end_inclusive=total_lines,
                    # Derived from the same snapshot that determined the bounds.
                    snapshot_digest=snapshot.prefix_digest(total_lines),
                    generation=interval.generation,
                )

        if safe_end <= start_line and self._cursor_semantics == "bounded_physical_lines":
            await state_manager.clear_catchup_failure(path_str)
            durable_followup = await state_manager.replace_continuation(
                interval.outbox_record_id if interval else None,
                followup_interval,
            )
            if durable_followup is not None:
                await self._queue_catchup_interval(durable_followup)
            return

        logger.info(
            "[CLIENT] Chunking source=%s over consumed interval (%d, %d] with parser context from %d",
            self._hash_value(path_str),
            start_line,
            safe_end,
            parser_start,
        )

        max_chunks = min(settings.max_chunks_per_file, 500)
        machine_id = getattr(api_client, "machine_id", settings.machine_id)
        client_name = getattr(
            api_client,
            "client_name",
            settings.client_name or settings.machine_id,
        )
        semantic_spool = tempfile.SpooledTemporaryFile(  # noqa: SIM115 - spans async uploads
            max_size=8 * 1024 * 1024,
            mode="w+t",
            encoding="utf-8",
            newline="\n",
        )

        try:
            if self._cursor_semantics == "bounded_physical_lines":
                chunks = self._chunk_snapshot(snapshot, parser_start, safe_end)
            else:
                chunks = self._chunk_snapshot(snapshot, parser_start)
            previous_consumed_line = start_line

            for chunk in chunks:
                if self._cursor_semantics == "bounded_physical_lines":
                    consumed_line = chunk.consumed_line or chunk.source_line
                    if consumed_line <= start_line or consumed_line > safe_end:
                        continue
                else:
                    # A snapshot is one semantic unit: no cursor may advance until
                    # every chunk emitted by that snapshot has been accepted.
                    consumed_line = safe_end
                if consumed_line < previous_consumed_line:
                    raise ValueError("chunker emitted a regressive consumed-line boundary")
                previous_consumed_line = consumed_line
                chunk_dict = chunk.model_dump(mode="json")
                chunk_dict["machine_id"] = settings.machine_id
                semantic_spool.write(json.dumps([consumed_line, chunk_dict], separators=(",", ":")))
                semantic_spool.write("\n")
        except Exception as e:
            semantic_spool.close()
            logger.error(
                "Failed to chunk source=%s error_type=%s",
                self._hash_value(path_str),
                type(e).__name__,
            )
            record_error(
                "chunking",
                f"Failed to chunk file: {type(e).__name__}",
                {"source_hash": self._hash_value(path_str), "error_type": type(e).__name__},
            )
            self._failed_files.add(path_str)
            if interval is not None:
                # The source is now marked failed, so this interval will never be
                # retried. Leaving its continuation in the outbox would keep
                # outstanding work that nothing can ever drain.
                await state_manager.record_catchup_failure(
                    interval,
                    "source_not_decodable"
                    if isinstance(e, UnicodeDecodeError)
                    else "source_outside_authority",
                )
                await state_manager.complete_continuation(interval.outbox_record_id)
            return

        # The source digest was proved against this snapshot before parsing and
        # the snapshot is immutable, so only mutable durable state is rechecked
        # here before any result is committed.
        if interval and (await state_manager.get_state()).reindex_generation != interval.generation:
            semantic_spool.close()
            return

        semantic_spool.seek(0)

        def read_spooled_chunks() -> Iterator[tuple[int, dict]]:
            """Read one semantic chunk at a time from the disk-spillable spool."""
            for record in semantic_spool:
                consumed_line, chunk = json.loads(record)
                yield consumed_line, chunk

        current_cursor = start_line
        batch: list[dict] = []
        batch_line: int | None = None

        def request_size(chunks: list[dict], file_position: int) -> int:
            return chunk_upload_request_bytes(
                chunks,
                machine_id=machine_id,
                client_name=client_name,
                source_file=path_str,
                file_position=file_position,
            )

        async def persist_batch(file_position: int) -> None:
            nonlocal batch, current_cursor
            if not batch:
                return
            payload_bytes = request_size(batch, file_position)
            record_start = current_cursor
            if len(batch) > max_chunks or payload_bytes > MAX_CHUNK_UPLOAD_REQUEST_BYTES:
                raise ValueError("single chunk exceeds the bounded upload request contract")
            identity = hashlib.sha256(
                json.dumps(
                    {
                        "file_path": path_str,
                        "snapshot": interval.snapshot_digest if interval else None,
                        "generation": state.reindex_generation,
                        "from": current_cursor,
                        "through": file_position,
                        "chunk_ids": [chunk.get("id") for chunk in batch],
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            await state_manager.replace_pending_upload(
                path_str,
                list(batch),
                file_position,
                record_id=identity,
                request_bytes=payload_bytes,
                start_position=record_start,
                source_snapshot_digest=interval.snapshot_digest if interval else None,
                final_fragment=file_position > record_start,
                generation=state.reindex_generation,
                machine_id=machine_id,
                client_name=client_name,
            )
            current_cursor = max(current_cursor, file_position)
            batch = []

        try:
            for consumed_line, chunk in read_spooled_chunks():
                if batch_line is not None and consumed_line != batch_line:
                    await persist_batch(batch_line)
                batch_line = consumed_line
                candidate = [*batch, chunk]
                if (
                    len(candidate) > max_chunks
                    or request_size(candidate, consumed_line) > MAX_CHUNK_UPLOAD_REQUEST_BYTES
                ):
                    await persist_batch(current_cursor)
                    candidate = [chunk]
                    if request_size(candidate, consumed_line) > MAX_CHUNK_UPLOAD_REQUEST_BYTES:
                        raise ValueError("single chunk exceeds the bounded upload request contract")
                batch = candidate
            if batch and batch_line is not None:
                await persist_batch(batch_line)

            if current_cursor < safe_end:
                batch = []
                await state_manager.replace_pending_upload(
                    path_str,
                    [],
                    safe_end,
                    record_id=hashlib.sha256(
                        f"{path_str}\x00{interval.snapshot_digest if interval else ''}\x00{state.reindex_generation}\x00{safe_end}\x00empty".encode()
                    ).hexdigest(),
                    request_bytes=request_size([], safe_end),
                    start_position=current_cursor,
                    source_snapshot_digest=interval.snapshot_digest if interval else None,
                    final_fragment=True,
                    generation=state.reindex_generation,
                    machine_id=machine_id,
                    client_name=client_name,
                )
        finally:
            semantic_spool.close()

        await state_manager.replace_continuation(
            interval.outbox_record_id if interval else None,
            followup_interval,
        )
        await state_manager.clear_catchup_failure(path_str)
        await self._process_pending_uploads(api_client, state_manager)

        elapsed = time.time() - start_time
        logger.info(
            "[CLIENT] Expanded bounded outbox records: source=%s elapsed=%.2fs",
            self._hash_value(path_str),
            elapsed,
        )

        self._last_indexed_file_hash = self._hash_value(path_str)
        self._last_indexed_at = datetime.now(timezone.utc)
        self._failed_files.discard(path_str)

    async def _index_file(self, file_path: Path, embedder: "AsyncEmbedder", store) -> None:
        """Index a single file from its last known position."""
        # LOW #2: Use Path consistently instead of str for dict keys
        path_str = str(file_path)
        reason = self._classify_history_path(file_path)
        if reason != "ok":
            logger.warning(
                "Refusing source outside watcher authority: reason=%s source_type=%s source=%s",
                reason,
                self.source_name,
                self._hash_value(path_str),
            )
            return

        # MEDIUM #3: Skip files that have failed recently
        if path_str in self._failed_files:
            logger.debug("Skipping previously failed source=%s", self._hash_value(path_str))
            return

        try:
            with self._source_snapshot(file_path) as snapshot:
                await self._index_snapshot(snapshot, embedder, store)
        except FileNotFoundError:
            logger.info(
                "Source path disappeared before indexing: source=%s",
                self._hash_value(path_str),
            )
        except UNSAFE_SOURCE_ERRORS as error:
            self._refuse_unsafe_source(path_str, error, phase="server_index")

    async def _index_snapshot(
        self,
        snapshot: _SourceSnapshot,
        embedder: "AsyncEmbedder",
        store,
    ) -> None:
        """Index exactly one immutable snapshot, keyed by its original path."""
        file_path = snapshot.source_path
        path_str = str(file_path)
        start_line = self.state.get_position(path_str)

        # LOW #4: Track time taken for indexing
        start_time = time.time()

        # MEDIUM #2: Eliminate double file read - track max line from chunker
        # Counted from the snapshot the chunker will read, so the logged total
        # and the parsed content can never describe different file states.
        total_lines = snapshot.line_count()

        # LOW #3: Include total_lines in log message
        logger.info(
            "Indexing source=%s from_line=%d total_lines=%d",
            self._hash_value(path_str),
            start_line,
            total_lines,
        )

        # Stream chunks in batches to avoid loading huge files into memory
        max_chunks = settings.max_chunks_per_file
        chunk_batch = []
        max_line = start_line
        total_chunks_stored = 0
        batch_num = 0

        try:
            for chunk in self._chunk_snapshot(snapshot, start_line):
                chunk_dict = chunk.model_dump()
                if settings.storage_backend == "spanner":
                    chunk_dict = _scope_chunk_for_direct_store(chunk_dict, settings.machine_id)
                else:
                    chunk_dict["machine_id"] = settings.machine_id
                chunk_batch.append(chunk_dict)
                if chunk.source_line > max_line:
                    max_line = chunk.source_line

                # Process batch when it reaches max size
                if len(chunk_batch) >= max_chunks:
                    batch_num += 1
                    logger.info(
                        "Processing chunk batch: source=%s batch=%d chunks=%d",
                        self._hash_value(path_str),
                        batch_num,
                        len(chunk_batch),
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
                    "Processing final chunk batch: source=%s batch=%d chunks=%d",
                    self._hash_value(path_str),
                    batch_num,
                    len(chunk_batch),
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
            source_hash = self._hash_value(path_str)
            _log_safe_failure("Failed to chunk", e, source_hash=source_hash)
            record_error(
                "chunking",
                f"Failed to chunk file: {type(e).__name__}",
                {"source_hash": source_hash, "error_type": type(e).__name__},
            )
            self._failed_files.add(path_str)
            return

        if total_chunks_stored == 0:
            logger.debug("No new chunks: source=%s", self._hash_value(path_str))
            # Update state to mark file as processed (prevents re-scanning)
            # Use max of max_line, total_lines, or at least 1 for empty files
            final_line = max(max_line, total_lines, 1 if total_lines == 0 else 0)
            if final_line > start_line or start_line == 0:
                self.state.set_position(path_str, final_line)
                self.state.save()
            self._last_indexed_file_hash = self._hash_value(path_str)
            self._last_indexed_at = datetime.now(timezone.utc)
            return

        # Log completion
        elapsed = time.time() - start_time
        logger.info(
            "Indexed source=%s chunks=%d elapsed=%.2fs",
            self._hash_value(path_str),
            total_chunks_stored,
            elapsed,
        )

        # Update state ONLY after successful storage (prevents data loss)
        # MEDIUM #2: Use max_line from chunker instead of total_lines
        final_line = max(max_line, total_lines)
        if final_line > start_line:
            self.state.set_position(path_str, final_line)
            self.state.save()

        # MEDIUM #3: Clear from failed files on success
        self._failed_files.discard(path_str)
        self._last_indexed_file_hash = self._hash_value(path_str)
        self._last_indexed_at = datetime.now(timezone.utc)

    async def startup_sync(self) -> None:
        """Scan all files and index any new content.

        Handles both server mode and client mode.
        In client mode, also syncs positions with server and processes pending uploads.
        """
        mode_str = "CLIENT" if settings.is_client_mode else "SERVER"
        root_hash = self._root_hash
        logger.info("[%s] Starting sync: root=%s", mode_str, root_hash)

        if not self._verify_root():
            return

        # Log directory contents for debugging
        try:
            all_files = list(self._root.path.glob("**/*"))
            logger.info("Directory scan: root=%s entries=%d", root_hash, len(all_files))
            logger.debug("Directory scan sample_count=%d", min(len(all_files), 10))
        except Exception as e:
            logger.error(
                "Failed to scan directory: root=%s error_type=%s",
                root_hash,
                type(e).__name__,
            )

        # Initialize mode-specific resources
        if settings.is_client_mode:
            from claude_history_rag.api_client import get_api_client
            from claude_history_rag.client_state import get_client_state_manager

            api_client = get_api_client()
            state_manager = get_client_state_manager()
            embedder = None

            # Process any pending uploads from previous sessions
            logger.info("[CLIENT] Processing pending uploads...")
            await self._process_pending_uploads(api_client, state_manager)

            # Reconcile only after the outbox has had the first chance to advance
            # the remote cursor, preventing overlapping catch-up work.
            logger.info("[CLIENT] Syncing positions with server...")
            await self._sync_positions_with_server(api_client, state_manager)

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

        # Find all history files. Containment is lexical here; the pinned root
        # proves object identity when each source is actually opened.
        jsonl_files = []
        for file_path in self._root.path.glob("**/*"):
            if _is_safe_path(file_path, self._root.path):
                if self._path_filter(file_path):
                    jsonl_files.append(file_path)
                if logger.isEnabledFor(logging.DEBUG):
                    # Sized only when the record will actually be emitted, and
                    # never by following a link. A vanished entry or a dangling
                    # reparse point used to raise here and abort the whole scan,
                    # skipping every remaining source.
                    size = -1
                    with contextlib.suppress(OSError):
                        size = os.lstat(file_path).st_size
                    logger.debug(
                        "Found source: source=%s bytes=%d",
                        self._hash_value(str(file_path)),
                        size,
                    )
            else:
                logger.info(
                    "Ignoring entry outside watch root: reason=%s source=%s",
                    durable_io.PATH_OUTSIDE_ROOT,
                    self._hash_value(str(file_path)),
                )

        if len(jsonl_files) == 0:
            logger.warning("No history files found: root=%s; watcher remains active", root_hash)
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
                logger.error(
                    "Failed to index source=%s error_type=%s",
                    self._hash_value(str(file_path)),
                    type(e).__name__,
                )
                record_error(
                    "indexing",
                    f"Failed to index file: {type(e).__name__}",
                    {
                        "source_hash": self._hash_value(str(file_path)),
                        "error_type": type(e).__name__,
                    },
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
        if not self._verify_root():
            return 0

        queued_count = 0
        for file_path in self._root.path.glob("**/*"):
            if _is_safe_path(file_path, self._root.path):
                if not self._path_filter(file_path):
                    continue
                if self._queue_raw_path(file_path):
                    queued_count += 1

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
