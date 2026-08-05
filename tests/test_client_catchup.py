"""Deterministic regression proofs for client catch-up cursor semantics."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from aiohttp.test_utils import TestClient, TestServer
from pydantic import ValidationError

from claude_history_rag import api_client as api_client_module
from claude_history_rag import client_registry as client_registry_module
from claude_history_rag import client_state as client_state_module
from claude_history_rag import durable_io
from claude_history_rag import embedder as embedder_module
from claude_history_rag import status as status_module
from claude_history_rag import status_server as status_server_module
from claude_history_rag import store as store_module
from claude_history_rag import watcher as watcher_module
from claude_history_rag.antigravity import watcher as antigravity_watcher_module
from claude_history_rag.auth import AuthCheckResult
from claude_history_rag.chatgpt import watcher as chatgpt_watcher_module
from claude_history_rag.chunker import chunk_session_file, missing_source_timestamp
from claude_history_rag.claude_app import watcher as claude_app_watcher_module
from claude_history_rag.client_registry import ClientRegistry
from claude_history_rag.client_state import CatchupInterval, ClientStateManager
from claude_history_rag.codex import watcher as codex_watcher_module
from claude_history_rag.config import settings
from claude_history_rag.gemini import watcher as gemini_watcher_module
from claude_history_rag.models import (
    MAX_CHUNK_UPLOAD_REQUEST_BYTES,
    Chunk,
    ChunkUploadResponse,
    GetPositionsResponse,
    ReindexAckRequest,
    chunk_upload_request_body,
    chunk_upload_request_bytes,
)
from claude_history_rag.parser import parse_jsonl_file
from claude_history_rag.watcher import FilePositionState, HistoryWatcher


def inject_durable_replace_failure(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    message: str,
) -> None:
    """Fail the confined pre-commit replacement on either supported platform."""
    if os.name == "nt":
        original = durable_io._windows_rename_relative

        def reject(held, source, destination, *, replace, directory=False):
            if held.path == target.parent and destination == target.name:
                raise OSError(message)
            return original(
                held,
                source,
                destination,
                replace=replace,
                directory=directory,
            )

        monkeypatch.setattr(durable_io, "_windows_rename_relative", reject)
    else:
        original = durable_io.os.replace

        def reject(source, destination, **kwargs):
            if destination == target.name:
                raise OSError(message)
            return original(source, destination, **kwargs)

        monkeypatch.setattr(durable_io.os, "replace", reject)


OUTSIDE_SENTINEL = "OUTSIDE_HISTORY_CONTENT_SENTINEL"


def json_fragment(value: str) -> str:
    """Return how a string appears once embedded in serialized JSON.

    Windows paths carry backslashes, which JSON escapes. Searching serialized
    state for a raw path would therefore never match and the assertion would
    pass whether or not the path leaked.
    """
    return json.dumps(value)[1:-1]


def build_watch_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Create a watch root holding one real history file."""
    root = tmp_path / "history"
    project = root / "-Users-test-myproject"
    project.mkdir(parents=True)
    target = project / "session.jsonl"
    shutil.copyfile(Path(__file__).parent / "fixtures" / "sample_session.jsonl", target)
    return root, target


def build_outside_tree(tmp_path: Path, name: str = "outside") -> Path:
    """Create content outside the watch root with the same relative layout."""
    outside = tmp_path / name
    project = outside / "-Users-test-myproject"
    project.mkdir(parents=True)
    (project / "session.jsonl").write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": "outside-session",
                "uuid": "outside-uuid",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": OUTSIDE_SENTINEL},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "sessionId": "outside-session",
                "uuid": "outside-uuid-2",
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {
                    "role": "assistant",
                    "model": "m",
                    "content": [{"type": "text", "text": OUTSIDE_SENTINEL}],
                },
            }
        )
        + "\n"
    )
    return outside


def link_directories_supported() -> bool:
    """Whether this platform can publish a directory link without privilege."""
    return os.name == "nt" or hasattr(os, "symlink")


def substitute_directory_with_link(target: Path, parked: Path, destination: Path) -> None:
    """Move a directory aside and publish a link of the same name in its place."""
    os.replace(target, parked)
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(destination)],
            check=True,
            capture_output=True,
        )
    else:
        target.symlink_to(destination, target_is_directory=True)


def restore_substituted_directory(target: Path, parked: Path) -> None:
    """Remove a published link and restore the original directory."""
    if target.is_symlink():
        target.unlink()
    elif os.name == "nt" and os.path.isjunction(target):
        os.rmdir(target)
    elif target.exists():
        shutil.rmtree(target)
    if parked.exists():
        os.replace(parked, target)


async def assert_no_work_was_committed(manager: ClientStateManager, api: FakeAPI) -> None:
    """No upload, no stored chunk, and no cursor advancement of any kind."""
    state = await manager.get_state()
    assert api.uploads == []
    assert state.pending_uploads == []
    assert state.local_positions == {}
    assert state.server_positions == {}
    assert OUTSIDE_SENTINEL not in json.dumps(state.model_dump(mode="json"))


class FakeAPI:
    """In-memory API double that mirrors acknowledged upload positions."""

    def __init__(self, file_path: Path, server_position: int = 0):
        self.file_path = file_path
        self.server_position = server_position
        self.uploads: list[dict] = []
        self.accepted_ids: list[str] = []

    async def get_positions(self):
        return SimpleNamespace(
            error=None,
            reindex_required=False,
            reindex_requested_at=None,
            positions={str(self.file_path): self.server_position},
        )

    async def upload_chunks(self, *, chunks, source_file, file_position):
        self.uploads.append(
            {
                "source_file": source_file,
                "file_position": file_position,
                "ids": [chunk["id"] for chunk in chunks],
                "source_lines": [chunk["source_line"] for chunk in chunks],
            }
        )
        self.server_position = file_position
        self.accepted_ids.extend(chunk["id"] for chunk in chunks)
        return SimpleNamespace(
            status="ok",
            reindex_required=False,
            reindex_requested_at=None,
            chunks_stored=len(chunks),
            chunks_received=len(chunks),
            chunks_embedded=len(chunks),
            error=None,
        )


class FailOnCallAPI(FakeAPI):
    def __init__(self, file_path: Path, fail_on_call: int):
        super().__init__(file_path)
        self.fail_on_call = fail_on_call
        self.calls = 0

    async def upload_chunks(self, *, chunks, source_file, file_position):
        self.calls += 1
        if self.calls == self.fail_on_call:
            self.uploads.append(
                {
                    "source_file": source_file,
                    "file_position": file_position,
                    "ids": [chunk["id"] for chunk in chunks],
                    "source_lines": [chunk["source_line"] for chunk in chunks],
                }
            )
            return SimpleNamespace(
                status="error",
                reindex_required=False,
                reindex_requested_at=None,
                chunks_stored=0,
                chunks_received=len(chunks),
                chunks_embedded=0,
                error="injected",
            )
        return await super().upload_chunks(
            chunks=chunks,
            source_file=source_file,
            file_position=file_position,
        )


@pytest.fixture
def history_file(tmp_path: Path) -> Path:
    project = tmp_path / "-Users-test-myproject"
    project.mkdir()
    target = project / "session.jsonl"
    fixture = Path(__file__).parent / "fixtures" / "sample_session.jsonl"
    shutil.copyfile(fixture, target)
    return target


def make_watcher(history_file: Path, tmp_path: Path) -> HistoryWatcher:
    return HistoryWatcher(
        projects_path=tmp_path,
        state_path=tmp_path / "watcher-state.json",
        chunker=chunk_session_file,
    )


async def manager_at(tmp_path: Path, history_file: Path, local_position: int):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.update_local_position(str(history_file), local_position)
    return manager


async def pending_payloads(manager: ClientStateManager, records):
    return [await manager.load_pending_chunks(record) for record in records]


async def consume_queued_client_work(
    watcher: HistoryWatcher,
    api: FakeAPI,
    manager: ClientStateManager,
) -> None:
    """Consume all currently and transitively queued client work like the watcher loop."""
    while not watcher.queue.empty():
        work = watcher.queue.get_nowait()
        try:
            await watcher._index_file_client_mode(work, api, manager)
        finally:
            if isinstance(work, CatchupInterval):
                watcher._queued_catchups.discard(
                    (
                        work.file_path,
                        work.start_exclusive,
                        work.end_inclusive,
                        work.generation,
                    )
                )
            watcher.queue.task_done()


@pytest.mark.parametrize(
    ("server_position", "expected_source_lines"),
    [
        (0, [2, 4, 5, 6]),
        (2, [2, 4, 5, 6]),  # Legacy cursor falls inside the first semantic pair.
        (3, [4, 5, 6]),  # Exact consumed boundary after the first assistant.
    ],
)
async def test_server_behind_reconstructs_bounded_interval_without_pending(
    tmp_path: Path,
    history_file: Path,
    server_position: int,
    expected_source_lines: list[int],
):
    manager = await manager_at(tmp_path, history_file, local_position=6)
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file, server_position)

    await watcher._sync_positions_with_server(api, manager)

    work = watcher.queue.get_nowait()
    assert isinstance(work, CatchupInterval)
    assert work.file_path == str(history_file)
    assert work.start_exclusive == server_position
    assert work.end_inclusive == 6
    assert work.snapshot_digest

    await watcher._index_file_client_mode(work, api, manager)

    assert [line for upload in api.uploads for line in upload["source_lines"]] == (
        expected_source_lines
    )
    assert api.server_position == 6


@pytest.mark.parametrize("server_position", [0, 1])
async def test_legacy_interval_tail_schedules_bounded_followup_without_path_event(
    tmp_path: Path,
    history_file: Path,
    server_position: int,
):
    """A stale cursor inside an already-complete pair cannot strand the assistant tail."""
    history_file.write_text("\n".join(history_file.read_text().splitlines()[:3]) + "\n")
    manager = await manager_at(tmp_path, history_file, local_position=2)
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file, server_position=server_position)

    await watcher._sync_positions_with_server(api, manager)
    initial = watcher.queue.get_nowait()
    assert isinstance(initial, CatchupInterval)
    assert initial.start_exclusive == server_position
    assert initial.end_inclusive == 2
    try:
        await watcher._index_file_client_mode(initial, api, manager)
    finally:
        watcher._queued_catchups.discard(
            (
                initial.file_path,
                initial.start_exclusive,
                initial.end_inclusive,
                initial.generation,
            )
        )
        watcher.queue.task_done()

    intermediate = await manager.get_state()
    assert intermediate.local_positions[str(history_file)] == 1
    assert intermediate.server_positions[str(history_file)] == 1
    followup = watcher.queue.get_nowait()
    assert isinstance(followup, CatchupInterval)
    assert followup.start_exclusive == 1
    assert followup.end_inclusive == 3
    try:
        await watcher._index_file_client_mode(followup, api, manager)
    finally:
        watcher._queued_catchups.discard(
            (
                followup.file_path,
                followup.start_exclusive,
                followup.end_inclusive,
                followup.generation,
            )
        )
        watcher.queue.task_done()
    await watcher._sync_positions_with_server(api, manager)

    state = await manager.get_state()
    assert watcher.queue.empty()
    assert api.server_position == 3
    assert state.server_positions[str(history_file)] == 3
    assert state.local_positions[str(history_file)] == 3
    assert api.accepted_ids


async def test_legacy_followup_survives_failed_cursor_only_commit(
    tmp_path: Path,
    history_file: Path,
):
    history_file.write_text("\n".join(history_file.read_text().splitlines()[:3]) + "\n")
    manager = await manager_at(tmp_path, history_file, local_position=2)
    watcher = make_watcher(history_file, tmp_path)
    api = FailOnCallAPI(history_file, fail_on_call=1)

    await watcher._sync_positions_with_server(api, manager)
    await consume_queued_client_work(watcher, api, manager)
    pending = (await manager.get_state()).pending_uploads
    upload_records = [record for record in pending if not record.is_continuation]
    continuation_records = [record for record in pending if record.is_continuation]
    assert len(upload_records) == 1
    assert upload_records[0].file_position == 1
    assert len(continuation_records) == 1
    assert continuation_records[0].followup_interval is not None
    assert continuation_records[0].followup_interval.start_exclusive == 1
    assert continuation_records[0].followup_interval.end_inclusive == 3

    assert await watcher._process_pending_uploads(api, manager) == 1
    await consume_queued_client_work(watcher, api, manager)
    await watcher._sync_positions_with_server(api, manager)

    state = await manager.get_state()
    assert api.server_position == 3
    assert state.server_positions[str(history_file)] == 3
    assert state.local_positions[str(history_file)] == 3


async def test_repeat_reconnect_after_catchup_has_no_duplicate_work(
    tmp_path: Path, history_file: Path
):
    manager = await manager_at(tmp_path, history_file, local_position=6)
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file, server_position=0)

    await watcher._sync_positions_with_server(api, manager)
    await watcher._index_file_client_mode(watcher.queue.get_nowait(), api, manager)
    first_upload_count = len(api.uploads)

    await watcher._sync_positions_with_server(api, manager)

    assert watcher.queue.empty()
    assert len(api.uploads) == first_upload_count
    assert api.server_position == 6


async def test_turn_source_line_remains_provenance_while_cursor_consumes_assistant(
    tmp_path: Path, history_file: Path
):
    three_lines = "\n".join(history_file.read_text().splitlines()[:3]) + "\n"
    history_file.write_text(three_lines)
    manager = ClientStateManager(tmp_path / "client-state.json")
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file)

    await watcher._index_file_client_mode(history_file, api, manager)

    assert api.uploads == [
        {
            "source_file": str(history_file),
            "file_position": 3,
            "ids": api.uploads[0]["ids"],
            "source_lines": [2],
        }
    ]
    state = await manager.get_state()
    assert state.local_positions[str(history_file)] == 3
    assert state.server_positions[str(history_file)] == 3


async def test_zero_chunk_interval_advances_consumed_cursor(tmp_path: Path, history_file: Path):
    history_file.write_text(history_file.read_text().splitlines()[0] + "\n")
    manager = ClientStateManager(tmp_path / "client-state.json")
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file)

    await watcher._index_file_client_mode(history_file, api, manager)

    assert api.uploads == [
        {
            "source_file": str(history_file),
            "file_position": 1,
            "ids": [],
            "source_lines": [],
        }
    ]
    assert api.server_position == 1


async def test_catchup_respects_snapshotted_end_if_file_grows_while_queued(
    tmp_path: Path, history_file: Path
):
    all_lines = history_file.read_text().splitlines()
    history_file.write_text("\n".join(all_lines[:3]) + "\n")
    manager = await manager_at(tmp_path, history_file, local_position=3)
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file)

    await watcher._sync_positions_with_server(api, manager)
    work = watcher.queue.get_nowait()
    history_file.write_text("\n".join(all_lines) + "\n")

    await watcher._index_file_client_mode(work, api, manager)

    assert [line for upload in api.uploads for line in upload["source_lines"]] == [2]
    assert api.server_position == 3


async def test_pending_upload_suppresses_overlapping_catchup(tmp_path: Path, history_file: Path):
    manager = await manager_at(tmp_path, history_file, local_position=6)
    await manager.add_pending_upload(
        str(history_file),
        [{"id": "pending", "source_line": 2}],
        file_position=6,
    )
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file)

    await watcher._sync_positions_with_server(api, manager)

    assert watcher.queue.empty()
    assert await watcher._process_pending_uploads(api, manager) == 1
    await watcher._sync_positions_with_server(api, manager)
    assert watcher.queue.empty()
    assert (await manager.get_state()).pending_uploads == []
    assert api.server_position == 6


async def test_deleted_history_records_unrecoverable_gap_without_claiming_success(
    tmp_path: Path, history_file: Path
):
    manager = await manager_at(tmp_path, history_file, local_position=6)
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file)
    history_file.unlink()

    await watcher._sync_positions_with_server(api, manager)

    state = await manager.get_state()
    failure = state.catchup_failures[str(history_file)]
    assert failure.reason == "history_missing"
    assert failure.start_exclusive == 0
    assert failure.end_inclusive == 6
    assert state.server_positions[str(history_file)] == 0
    assert watcher.queue.empty()


async def test_truncated_history_records_unrecoverable_gap_without_upload(
    tmp_path: Path, history_file: Path
):
    manager = await manager_at(tmp_path, history_file, local_position=6)
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file)
    await watcher._sync_positions_with_server(api, manager)
    work = watcher.queue.get_nowait()
    history_file.write_text("\n".join(history_file.read_text().splitlines()[:3]) + "\n")

    await watcher._index_file_client_mode(work, api, manager)

    state = await manager.get_state()
    failure = state.catchup_failures[str(history_file)]
    assert failure.reason == "history_truncated"
    assert failure.start_exclusive == 0
    assert failure.end_inclusive == 6
    assert api.uploads == []


async def test_start_inside_pair_expands_context_and_preserves_stable_id(
    tmp_path: Path, history_file: Path
):
    expected_first_turn = next(chunk_session_file(history_file, 0))
    assert expected_first_turn.source_line == 2
    manager = await manager_at(tmp_path, history_file, local_position=6)
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file, server_position=2)

    await watcher._sync_positions_with_server(api, manager)
    await watcher._index_file_client_mode(watcher.queue.get_nowait(), api, manager)

    uploaded_ids = [chunk_id for upload in api.uploads for chunk_id in upload["ids"]]
    assert expected_first_turn.id in uploaded_ids


async def test_pair_context_spans_metadata_and_preserves_full_replay_identity(
    tmp_path: Path, history_file: Path
):
    source = history_file.read_text().splitlines()
    metadata = json.dumps(
        {
            "type": "system",
            "subtype": "metadata",
            "sessionId": "test-session-123",
            "timestamp": "2025-12-14T10:01:03.000Z",
        }
    )
    history_file.write_text("\n".join([source[0], source[1], metadata, source[2]]) + "\n")
    expected = [chunk.id for chunk in chunk_session_file(history_file, 0)]
    manager = await manager_at(tmp_path, history_file, local_position=4)
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file, server_position=3)

    await watcher._sync_positions_with_server(api, manager)
    await watcher._index_file_client_mode(watcher.queue.get_nowait(), api, manager)

    assert [chunk_id for upload in api.uploads for chunk_id in upload["ids"]] == expected
    assert api.server_position == 4


async def test_file_change_parent_child_identity_survives_pair_boundary_expansion(
    tmp_path: Path, history_file: Path
):
    source = history_file.read_text().splitlines()
    history_file.write_text("\n".join([source[0], source[3], source[4]]) + "\n")
    expected = list(chunk_session_file(history_file, 0))
    assert len(expected) == 2
    assert expected[0].child_chunk_ids == [expected[1].id]
    assert expected[1].parent_chunk_id == expected[0].id
    manager = await manager_at(tmp_path, history_file, local_position=3)
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file, server_position=2)

    await watcher._sync_positions_with_server(api, manager)
    await watcher._index_file_client_mode(watcher.queue.get_nowait(), api, manager)

    assert [chunk_id for upload in api.uploads for chunk_id in upload["ids"]] == [
        chunk.id for chunk in expected
    ]


async def test_legacy_cursor_past_trailing_user_is_corrected_then_pair_completes(
    tmp_path: Path, history_file: Path
):
    source = history_file.read_text().splitlines()
    history_file.write_text("\n".join(source[:4]) + "\n")
    manager = await manager_at(tmp_path, history_file, local_position=4)
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file, server_position=3)

    await watcher._sync_positions_with_server(api, manager)
    await watcher._index_file_client_mode(watcher.queue.get_nowait(), api, manager)

    assert (await manager.get_state()).local_positions[str(history_file)] == 3
    assert api.uploads == []

    history_file.write_text("\n".join(source[:5]) + "\n")
    expected_tail_ids = [chunk.id for chunk in chunk_session_file(history_file, 3)]
    await watcher._index_file_client_mode(history_file, api, manager)

    assert [chunk_id for upload in api.uploads for chunk_id in upload["ids"]] == expected_tail_ids
    assert api.server_position == 5
    assert (await manager.get_state()).local_positions[str(history_file)] == 5


def test_missing_timestamp_replay_identity_is_deterministic(history_file: Path):
    rows = [json.loads(line) for line in history_file.read_text().splitlines()]
    for row in rows:
        row.pop("timestamp", None)
    history_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    first = list(chunk_session_file(history_file, 0))
    second = list(chunk_session_file(history_file, 0))

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert [chunk.timestamp for chunk in first] == [
        missing_source_timestamp(3),
        missing_source_timestamp(5),
        missing_source_timestamp(5),
        missing_source_timestamp(6),
    ]


async def test_semantic_group_is_not_acknowledged_until_every_chunk_is_accepted(
    tmp_path: Path, history_file: Path, monkeypatch: pytest.MonkeyPatch
):
    source = history_file.read_text().splitlines()
    history_file.write_text("\n".join([source[0], source[3], source[4]]) + "\n")
    monkeypatch.setattr("claude_history_rag.watcher.settings.max_chunks_per_file", 1)
    manager = ClientStateManager(tmp_path / "client-state.json")
    watcher = make_watcher(history_file, tmp_path)
    api = FailOnCallAPI(history_file, fail_on_call=2)

    await watcher._index_file_client_mode(history_file, api, manager)

    assert api.server_position == 0
    state = await manager.get_state()
    assert state.local_positions.get(str(history_file), 0) == 0
    assert len(state.pending_uploads) == 1
    assert state.pending_uploads[0].file_position == 3


async def test_failed_oversized_group_retains_complete_tail_and_final_boundary(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def grouped_chunker(path: Path, start_line: int, end_line: int | None = None):
        for chunk_id, consumed_line in [
            ("same-1", 3),
            ("same-2", 3),
            ("same-3", 3),
            ("later-1", 4),
        ]:
            yield Chunk(
                id=chunk_id,
                content=chunk_id,
                chunk_type="turn",
                session_id="grouped-session",
                project_path=str(tmp_path),
                project_name="grouped",
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                source_file=str(path),
                source_line=consumed_line,
                consumed_line=consumed_line,
            )

    monkeypatch.setattr(settings, "max_chunks_per_file", 1)
    manager = ClientStateManager(tmp_path / "client-state.json")
    watcher = HistoryWatcher(
        projects_path=tmp_path,
        state_path=tmp_path / "watcher-state.json",
        chunker=grouped_chunker,
    )
    api = FailOnCallAPI(history_file, fail_on_call=2)

    await watcher._index_file_client_mode(history_file, api, manager)

    pending = (await manager.get_state()).pending_uploads
    payloads = await pending_payloads(manager, pending)
    assert [chunk["id"] for payload in payloads for chunk in payload] == [
        "same-2",
        "same-3",
        "later-1",
    ]
    assert pending[-1].file_position == 6
    assert api.accepted_ids == ["same-1"]
    assert api.server_position == 0

    assert await watcher._process_pending_uploads(api, manager) == len(pending)
    state = await manager.get_state()
    assert api.accepted_ids == ["same-1", "same-2", "same-3", "later-1"]
    assert api.server_position == 6
    assert state.server_positions[str(history_file)] == 6
    assert state.local_positions[str(history_file)] == 6


async def test_successful_pending_retry_advances_local_and_prevents_unchanged_reupload(
    tmp_path: Path,
    history_file: Path,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    watcher = make_watcher(history_file, tmp_path)
    api = FailOnCallAPI(history_file, fail_on_call=1)

    await watcher._index_file_client_mode(history_file, api, manager)
    failed_state = await manager.get_state()
    assert failed_state.pending_uploads
    assert failed_state.pending_uploads[-1].file_position == 6
    assert failed_state.local_positions.get(str(history_file), 0) == 0
    assert failed_state.server_positions.get(str(history_file), 0) == 0
    pending_count = len(failed_state.pending_uploads)
    assert await watcher._process_pending_uploads(api, manager) == pending_count
    uploads_after_retry = len(api.uploads)
    accepted_after_retry = list(api.accepted_ids)

    await watcher._index_file_client_mode(history_file, api, manager)

    state = await manager.get_state()
    assert state.local_positions[str(history_file)] == 6
    assert state.server_positions[str(history_file)] == 6
    assert len(api.uploads) == uploads_after_retry
    assert api.accepted_ids == accepted_after_retry


async def test_partial_success_response_does_not_advance_cursor(tmp_path: Path, history_file: Path):
    class PartialAPI(FakeAPI):
        async def upload_chunks(self, *, chunks, source_file, file_position):
            return SimpleNamespace(
                status="ok",
                reindex_required=False,
                reindex_requested_at=None,
                chunks_stored=max(0, len(chunks) - 1),
                chunks_received=len(chunks),
                chunks_embedded=max(0, len(chunks) - 1),
                error=None,
            )

    manager = ClientStateManager(tmp_path / "client-state.json")
    watcher = make_watcher(history_file, tmp_path)
    api = PartialAPI(history_file)

    await watcher._index_file_client_mode(history_file, api, manager)

    state = await manager.get_state()
    assert str(history_file) not in state.server_positions
    assert state.local_positions.get(str(history_file), 0) == 0
    assert state.pending_uploads


async def test_duplicate_sync_before_consumption_queues_one_interval(
    tmp_path: Path, history_file: Path
):
    manager = await manager_at(tmp_path, history_file, local_position=6)
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file)

    await watcher._sync_positions_with_server(api, manager)
    await watcher._sync_positions_with_server(api, manager)

    assert watcher.queue.qsize() == 1


async def test_repeated_missing_sync_is_terminal_but_recovery_requeues(
    tmp_path: Path, history_file: Path
):
    original = history_file.read_text()
    manager = await manager_at(tmp_path, history_file, local_position=6)
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file)
    history_file.unlink()

    await watcher._sync_positions_with_server(api, manager)
    first = (await manager.get_state()).catchup_failures[str(history_file)].observed_at
    await watcher._sync_positions_with_server(api, manager)
    second = (await manager.get_state()).catchup_failures[str(history_file)].observed_at
    assert second == first

    history_file.write_text(original)
    await watcher._sync_positions_with_server(api, manager)

    assert watcher.queue.qsize() == 1
    assert str(history_file) not in (await manager.get_state()).catchup_failures


async def test_same_length_replacement_is_rejected_by_snapshot_identity(
    tmp_path: Path, history_file: Path
):
    manager = await manager_at(tmp_path, history_file, local_position=6)
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file)
    await watcher._sync_positions_with_server(api, manager)
    work = watcher.queue.get_nowait()
    original = history_file.read_text()
    history_file.write_text(original.replace("authentication", "authenticatioN", 1))

    await watcher._index_file_client_mode(work, api, manager)

    failure = (await manager.get_state()).catchup_failures[str(history_file)]
    assert failure.reason == "history_replaced"
    assert api.uploads == []


async def test_incompatible_snapshot_adapter_fails_closed(tmp_path: Path, history_file: Path):
    manager = await manager_at(tmp_path, history_file, local_position=6)
    watcher = HistoryWatcher(
        projects_path=tmp_path,
        state_path=tmp_path / "snapshot-state.json",
        chunker=lambda path, start: iter(()),
        cursor_semantics="unsupported_snapshot",
    )
    api = FakeAPI(history_file)

    await watcher._sync_positions_with_server(api, manager)

    assert watcher.queue.empty()
    failure = (await manager.get_state()).catchup_failures[str(history_file)]
    assert failure.reason == "unsupported_cursor_contract"


async def test_incompatible_snapshot_failure_refreshes_for_new_interval(
    tmp_path: Path, history_file: Path
):
    manager = await manager_at(tmp_path, history_file, local_position=6)
    watcher = HistoryWatcher(
        projects_path=tmp_path,
        state_path=tmp_path / "snapshot-state.json",
        chunker=lambda path, start: iter(()),
        cursor_semantics="unsupported_snapshot",
    )
    api = FakeAPI(history_file)

    await watcher._sync_positions_with_server(api, manager)
    history_file.write_text(history_file.read_text() + "{}\n")
    await manager.update_local_position(str(history_file), 7)
    await watcher._sync_positions_with_server(api, manager)

    failure = (await manager.get_state()).catchup_failures[str(history_file)]
    assert failure.start_exclusive == 0
    assert failure.end_inclusive == 7
    assert failure.reason == "unsupported_cursor_contract"


async def test_snapshot_adapter_normal_event_remains_full_snapshot(
    tmp_path: Path, history_file: Path
):
    def snapshot_chunker(path: Path, start_line: int):
        assert start_line == 0
        yield Chunk(
            id="snapshot-chunk",
            content="snapshot",
            chunk_type="turn",
            session_id="snapshot-session",
            project_path=str(tmp_path),
            project_name="snapshot",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            source_file=str(path),
            source_line=0,
        )

    manager = await manager_at(tmp_path, history_file, local_position=6)
    watcher = HistoryWatcher(
        projects_path=tmp_path,
        state_path=tmp_path / "snapshot-state.json",
        chunker=snapshot_chunker,
        cursor_semantics="unsupported_snapshot",
    )
    api = FakeAPI(history_file)

    await watcher._index_file_client_mode(history_file, api, manager)

    assert len(api.uploads) == 1
    assert api.uploads[0]["ids"] == ["snapshot-chunk"]
    assert api.uploads[0]["file_position"] == 6


@pytest.mark.parametrize(
    ("module", "global_name", "getter", "source_path_name", "state_path_name"),
    [
        (
            codex_watcher_module,
            "codex_watcher",
            "get_codex_watcher",
            "codex_sessions_path",
            "codex_state_path",
        ),
        (
            gemini_watcher_module,
            "gemini_watcher",
            "get_gemini_watcher",
            "gemini_sessions_path",
            "gemini_state_path",
        ),
        (
            chatgpt_watcher_module,
            "chatgpt_watcher",
            "get_chatgpt_watcher",
            "chatgpt_exports_path",
            "chatgpt_state_path",
        ),
        (
            claude_app_watcher_module,
            "claude_app_watcher",
            "get_claude_app_watcher",
            "claude_app_exports_path",
            "claude_app_state_path",
        ),
        (
            antigravity_watcher_module,
            "antigravity_watcher",
            "get_antigravity_watcher",
            "antigravity_sessions_path",
            "antigravity_state_path",
        ),
    ],
)
def test_non_claude_adapters_declare_fail_closed_cursor_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    global_name: str,
    getter: str,
    source_path_name: str,
    state_path_name: str,
):
    monkeypatch.setattr(module, global_name, None)
    monkeypatch.setattr(settings, source_path_name, tmp_path / source_path_name)
    monkeypatch.setattr(settings, state_path_name, tmp_path / f"{state_path_name}.json")

    watcher = getattr(module, getter)()

    assert watcher._cursor_semantics == "unsupported_snapshot"


def test_catchup_interval_rejects_empty_or_reversed_work(history_file: Path):
    with pytest.raises(ValueError):
        CatchupInterval(file_path=str(history_file), start_exclusive=3, end_inclusive=3)
    with pytest.raises(ValueError):
        CatchupInterval(file_path=str(history_file), start_exclusive=4, end_inclusive=3)


def test_catchup_interval_is_immutable(history_file: Path):
    interval = CatchupInterval(file_path=str(history_file), start_exclusive=1, end_inclusive=2)

    with pytest.raises(ValueError):
        interval.end_inclusive = 3


class FakeRegistry:
    def register_client(self, machine_id: str, client_name: str | None = None) -> None:
        return None

    def get_reindex_status(self, machine_id: str):
        return False, None

    def record_upload(self, machine_id: str, client_name: str | None = None) -> None:
        return None


def fake_request(payload: dict):
    async def request_json():
        return payload

    return SimpleNamespace(json=request_json, headers={})


async def configure_status_server(monkeypatch: pytest.MonkeyPatch):
    server = status_server_module.StatusServer()

    async def allow(*args, **kwargs):
        return AuthCheckResult(ok=True, key_type="active")

    monkeypatch.setattr(server, "_require_auth", allow)
    monkeypatch.setattr(status_server_module, "get_client_registry", lambda: FakeRegistry())
    status_server_module._clear_machine_positions()
    return server


async def test_real_empty_upload_endpoint_commits_cursor(monkeypatch: pytest.MonkeyPatch):
    server = await configure_status_server(monkeypatch)
    registry = FakeRegistry()
    registry.uploads = []
    registry.record_upload = lambda machine_id, client_name=None: registry.uploads.append(
        (machine_id, client_name)
    )
    monkeypatch.setattr(status_server_module, "get_client_registry", lambda: registry)
    monkeypatch.setattr(status_server_module.settings, "storage_backend", "spanner")
    monkeypatch.setattr(status_server_module.settings, "spanner_embedding_mode", "spanner")
    payload = {
        "machine_id": "machine-a",
        "chunks": [],
        "source_file": "/history/session.jsonl",
        "file_position": 7,
    }

    response = await server.handle_api_chunks(fake_request(payload))

    body = json.loads(response.text)
    assert response.status == 200
    assert body["status"] == "ok"
    assert body["chunks_stored"] == 0
    assert status_server_module._machine_positions["machine-a"][payload["source_file"]] == 7
    assert registry.uploads == [("machine-a", "machine-a")]

    payload["file_position"] = 3
    response = await server.handle_api_chunks(fake_request(payload))

    assert response.status == 200
    assert status_server_module._machine_positions["machine-a"][payload["source_file"]] == 7


async def test_real_partial_embedding_endpoint_rejects_before_store_or_cursor(
    monkeypatch: pytest.MonkeyPatch,
):
    server = await configure_status_server(monkeypatch)
    monkeypatch.setattr(status_server_module.settings, "storage_backend", "lancedb")

    class PartialEmbedder:
        async def embed_chunks(self, chunks):
            return []

    class RejectStore:
        async def add_chunks_async(self, chunks):
            raise AssertionError("partial batch must not be stored")

    monkeypatch.setattr(embedder_module, "get_embedder", lambda: PartialEmbedder())
    monkeypatch.setattr(store_module, "store", RejectStore())
    payload = {
        "machine_id": "machine-a",
        "chunks": [
            {
                "id": "chunk-a",
                "content": "content",
                "chunk_type": "turn",
                "session_id": "session",
                "project_path": "/project",
                "project_name": "project",
                "source_file": "/history/session.jsonl",
                "source_line": 2,
            }
        ],
        "source_file": "/history/session.jsonl",
        "file_position": 3,
    }

    response = await server.handle_api_chunks(fake_request(payload))

    body = json.loads(response.text)
    assert response.status == 503
    assert body["error"] == "incomplete_embedding_batch"
    assert "machine-a" not in status_server_module._machine_positions


async def test_queue_consumer_preserves_client_interval_work(
    tmp_path: Path, history_file: Path, monkeypatch: pytest.MonkeyPatch
):
    manager = await manager_at(tmp_path, history_file, local_position=6)
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file)
    monkeypatch.setattr(settings, "server_url", "https://history.example.test")
    monkeypatch.setattr(api_client_module, "get_api_client", lambda: api)
    monkeypatch.setattr(client_state_module, "get_client_state_manager", lambda: manager)
    await watcher._sync_positions_with_server(api, manager)
    watcher._running = True

    task = asyncio.create_task(watcher._process_files())
    await asyncio.wait_for(watcher.queue.join(), timeout=2)
    watcher._shutdown_event.set()
    await asyncio.wait_for(task, timeout=2)

    assert api.server_position == 6
    assert watcher._queued_catchups == set()


async def test_queue_consumer_keeps_server_mode_path_compatibility(
    tmp_path: Path, history_file: Path, monkeypatch: pytest.MonkeyPatch
):
    watcher = make_watcher(history_file, tmp_path)
    observed: list[Path] = []

    async def record_path(file_path, embedder, store):
        observed.append(file_path)

    monkeypatch.setattr(settings, "server_url", None)
    monkeypatch.setattr(settings, "storage_backend", "spanner")
    monkeypatch.setattr(settings, "spanner_embedding_mode", "spanner")
    monkeypatch.setattr(watcher, "_index_file", record_path)
    await watcher.queue.put(history_file)
    watcher._running = True

    task = asyncio.create_task(watcher._process_files())
    await asyncio.wait_for(watcher.queue.join(), timeout=2)
    watcher._shutdown_event.set()
    await asyncio.wait_for(task, timeout=2)

    assert observed == [history_file]


async def test_shutdown_queue_race_balances_claimed_interval_and_dedupe(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    api = FakeAPI(history_file)
    monkeypatch.setattr(settings, "server_url", "https://history.example.test")
    monkeypatch.setattr(api_client_module, "get_api_client", lambda: api)
    monkeypatch.setattr(client_state_module, "get_client_state_manager", lambda: manager)

    for run in range(200):
        watcher = make_watcher(history_file, tmp_path / f"run-{run}")
        interval = CatchupInterval(
            file_path=str(history_file),
            start_exclusive=0,
            end_inclusive=1,
        )
        key = (
            interval.file_path,
            interval.start_exclusive,
            interval.end_inclusive,
            interval.generation,
        )
        watcher._queued_catchups.add(key)
        await watcher.queue.put(interval)
        watcher._shutdown_event.set()
        watcher._running = True

        await asyncio.wait_for(watcher._process_files(), timeout=2)

        assert watcher.queue.empty()
        assert watcher.queue._unfinished_tasks == 0
        assert watcher._queued_catchups == set()


def grouped_chunks(tmp_path: Path, count: int, *, content_size: int = 1):
    """Build one same-boundary semantic group of deterministic chunks."""

    def chunker(path: Path, start_line: int, end_line: int | None = None):
        for index in range(count):
            yield Chunk(
                id=f"group-{index}",
                content="x" * content_size,
                chunk_type="turn",
                session_id="grouped-session",
                project_path=str(tmp_path),
                project_name="grouped",
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                source_file=str(path),
                source_line=3,
                consumed_line=3,
            )

    return chunker


async def test_503_chunk_group_failure_persists_bounded_ordered_fragments(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    history_file.write_text("\n".join(history_file.read_text().splitlines()[:3]) + "\n")
    monkeypatch.setattr(settings, "max_chunks_per_file", 1)
    manager = ClientStateManager(tmp_path / "client-state.json")
    watcher = HistoryWatcher(
        projects_path=tmp_path,
        state_path=tmp_path / "watcher-state.json",
        chunker=grouped_chunks(tmp_path, 503),
    )
    api = FailOnCallAPI(history_file, fail_on_call=2)

    await watcher._index_file_client_mode(history_file, api, manager)

    pending = (await manager.get_state()).pending_uploads
    assert len(pending) == 502
    assert all(record.chunks == [] and record.payload_externalized for record in pending)
    assert "group-502" not in manager.state_path.read_text()
    assert all(len(payload) == 1 for payload in await pending_payloads(manager, pending))
    assert [record.sequence for record in pending] == sorted(record.sequence for record in pending)
    assert all(record.start_position == 0 for record in pending)
    assert all(record.source_snapshot_digest for record in pending)
    assert all(not record.final_fragment for record in pending[:-1])
    assert pending[-1].final_fragment
    assert all(record.file_position == 0 for record in pending[:-1])
    assert pending[-1].file_position == 3
    assert api.server_position == 0

    assert await watcher._process_pending_uploads(api, manager) == 502
    state = await manager.get_state()
    assert state.pending_uploads == []
    assert state.server_positions[str(history_file)] == 3
    assert state.local_positions[str(history_file)] == 3


def test_upload_body_limit_has_exact_fit_and_one_byte_over_boundary():
    base = {
        "id": "boundary",
        "content": "",
        "chunk_type": "turn",
        "session_id": "session",
        "project_path": "/project",
        "project_name": "project",
        "timestamp": "2026-01-01T00:00:00Z",
        "source_file": "/history/session.jsonl",
        "source_line": 3,
        "machine_id": "machine-a",
    }
    overhead = chunk_upload_request_bytes(
        [base],
        machine_id="machine-a",
        client_name="machine-a",
        source_file=base["source_file"],
        file_position=3,
    )
    exact = dict(base, content="x" * (MAX_CHUNK_UPLOAD_REQUEST_BYTES - overhead))
    one_over = dict(exact, content=exact["content"] + "x")

    assert (
        chunk_upload_request_bytes(
            [exact],
            machine_id="machine-a",
            client_name="machine-a",
            source_file=base["source_file"],
            file_position=3,
        )
        == MAX_CHUNK_UPLOAD_REQUEST_BYTES
    )
    assert (
        chunk_upload_request_bytes(
            [one_over],
            machine_id="machine-a",
            client_name="machine-a",
            source_file=base["source_file"],
            file_position=3,
        )
        == MAX_CHUNK_UPLOAD_REQUEST_BYTES + 1
    )


def test_upload_byte_measurement_matches_httpx_wire_encoding():
    from claude_history_rag.models import ChunkUploadRequest

    request = ChunkUploadRequest(
        machine_id="machine-a",
        client_name="client-a",
        chunks=[{"content": "Unicode snowman: ☃", "id": "chunk-a"}],
        source_file="/history/session.jsonl",
        file_position=17,
    )
    wire = httpx.Request(
        "POST",
        "https://history.example.test/api/chunks",
        json=request.model_dump(mode="json"),
    ).content

    assert len(wire) == chunk_upload_request_bytes(
        request.chunks,
        machine_id=request.machine_id,
        client_name=request.client_name,
        source_file=request.source_file,
        file_position=request.file_position,
    )


async def test_byte_oversized_group_is_split_before_persistence_and_transmission(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    history_file.write_text("\n".join(history_file.read_text().splitlines()[:3]) + "\n")
    monkeypatch.setattr(settings, "max_chunks_per_file", 500)
    manager = ClientStateManager(tmp_path / "client-state.json")
    watcher = HistoryWatcher(
        projects_path=tmp_path,
        state_path=tmp_path / "watcher-state.json",
        chunker=grouped_chunks(tmp_path, 2, content_size=600_000),
    )
    api = FailOnCallAPI(history_file, fail_on_call=1)

    await watcher._index_file_client_mode(history_file, api, manager)

    pending = (await manager.get_state()).pending_uploads
    assert len(pending) == 2
    assert all(record.request_bytes <= MAX_CHUNK_UPLOAD_REQUEST_BYTES for record in pending)
    assert all(len(upload["ids"]) == 1 for upload in api.uploads)


async def test_unbounded_record_is_rejected_by_persistence_authority(tmp_path: Path):
    manager = ClientStateManager(tmp_path / "client-state.json")

    with pytest.raises(ValueError, match="unbounded upload"):
        await manager.replace_pending_upload(
            "/history/session.jsonl",
            [{"id": "too-large", "content": "x" * MAX_CHUNK_UPLOAD_REQUEST_BYTES}],
            3,
        )

    assert (await manager.get_state()).pending_uploads == []
    assert not manager._outbox_dir.exists()


def test_client_catchup_has_no_all_tail_materialization_helper():
    source = inspect.getsource(HistoryWatcher._index_file_client_mode)

    assert "read_remaining_chunks" not in source
    assert "remaining.extend" not in source


async def test_same_path_replacement_cannot_overwrite_earlier_pending_history(tmp_path: Path):
    manager = ClientStateManager(tmp_path / "client-state.json")

    await manager.replace_pending_upload("/history/same.jsonl", [{"id": "A"}], 3)
    await manager.replace_pending_upload("/history/same.jsonl", [{"id": "B"}], 3)

    pending = (await manager.get_state()).pending_uploads
    assert [payload[0]["id"] for payload in await pending_payloads(manager, pending)] == [
        "A",
        "B",
    ]
    assert pending[0].sequence < pending[1].sequence


async def test_same_length_replacement_waits_behind_predecessor_and_fails_continuity(
    tmp_path: Path,
    history_file: Path,
):
    history_file.write_text("\n".join(history_file.read_text().splitlines()[:3]) + "\n")
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        str(history_file),
        [{"id": "history-A", "source_line": 3}],
        3,
        record_id="history-A-record",
        source_snapshot_digest="snapshot-A",
        start_position=0,
        final_fragment=True,
    )
    replacement = await manager.add_continuation(
        CatchupInterval(
            file_path=str(history_file),
            start_exclusive=0,
            end_inclusive=3,
            snapshot_digest="snapshot-B",
        )
    )
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file)

    assert await watcher._process_pending_uploads(api, manager) == 1
    queued = watcher.queue.get_nowait()
    assert queued.outbox_record_id == replacement.outbox_record_id
    await watcher._index_file_client_mode(queued, api, manager)

    state = await manager.get_state()
    assert api.accepted_ids == ["history-A"]
    assert state.catchup_failures[str(history_file)].reason == "history_replaced"
    assert any(record.record_id == replacement.outbox_record_id for record in state.pending_uploads)


async def test_old_pending_shape_migrates_to_bounded_records_without_data_loss(tmp_path: Path):
    state_path = tmp_path / "client-state.json"
    legacy_chunks = [{"id": f"legacy-{index}", "content": "x" * 3_000} for index in range(503)]
    state_path.write_text(
        json.dumps(
            {
                "local_positions": {"/history/legacy.jsonl": 9},
                "server_positions": {"/history/legacy.jsonl": 0},
                "pending_uploads": [
                    {
                        "file_path": "/history/legacy.jsonl",
                        "chunks": legacy_chunks,
                        "file_position": 9,
                        "created_at": "2026-01-01T00:00:00Z",
                        "retry_count": 2,
                    }
                ],
            }
        )
    )

    manager = ClientStateManager(state_path)
    state = await manager.get_state()
    payloads = await pending_payloads(manager, state.pending_uploads)

    assert len(state.pending_uploads) > 1
    assert [chunk["id"] for payload in payloads for chunk in payload] == [
        chunk["id"] for chunk in legacy_chunks
    ]
    assert all(len(payload) <= min(settings.max_chunks_per_file, 500) for payload in payloads)
    assert all(
        record.request_bytes <= MAX_CHUNK_UPLOAD_REQUEST_BYTES for record in state.pending_uploads
    )
    assert all(record.file_position == 0 for record in state.pending_uploads[:-1])
    assert state.pending_uploads[-1].file_position == 9


async def test_corrupt_durable_outbox_fails_closed_instead_of_discarding(tmp_path: Path):
    state_path = tmp_path / "client-state.json"
    state_path.write_text('{"pending_uploads": [')

    with pytest.raises(RuntimeError, match="durable client outbox state"):
        await ClientStateManager(state_path).get_state()


async def test_outbox_record_identity_cannot_escape_payload_directory(tmp_path: Path):
    state_path = tmp_path / "client-state.json"
    state_path.write_text(
        json.dumps(
            {
                "pending_uploads": [
                    {
                        "record_id": "../escape",
                        "file_path": "/history/session.jsonl",
                        "chunks": [],
                        "file_position": 1,
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                ]
            }
        )
    )

    with pytest.raises(RuntimeError, match="durable client outbox state"):
        await ClientStateManager(state_path).get_state()

    assert not (tmp_path / "escape.json").exists()


async def test_outbox_persistence_failure_prevents_network_transmission(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file)

    def fail_save(*args):
        raise OSError("injected durable-write failure")

    monkeypatch.setattr(manager, "_write_state_snapshot", fail_save)

    with pytest.raises(OSError, match="durable-write failure"):
        await watcher._index_file_client_mode(history_file, api, manager)

    assert api.uploads == []


async def test_pending_success_persists_followup_before_queue_cancellation(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    history_file.write_text("\n".join(history_file.read_text().splitlines()[:3]) + "\n")
    manager = await manager_at(tmp_path, history_file, local_position=2)
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file)
    followup = CatchupInterval(
        file_path=str(history_file),
        start_exclusive=1,
        end_inclusive=3,
    )
    await manager.replace_pending_upload(
        str(history_file),
        [],
        1,
        followup_interval=followup,
    )

    async def cancel_queue(interval):
        raise asyncio.CancelledError

    monkeypatch.setattr(watcher, "_queue_catchup_interval", cancel_queue)
    with pytest.raises(asyncio.CancelledError):
        await watcher._process_pending_uploads(api, manager)

    reloaded = ClientStateManager(manager.state_path)
    state = await reloaded.get_state()
    assert state.server_positions[str(history_file)] == 1
    assert state.local_positions[str(history_file)] == 1
    assert any(record.followup_interval == followup for record in state.pending_uploads)

    restarted = make_watcher(history_file, tmp_path)
    assert await restarted._process_pending_uploads(api, reloaded) == 0
    await consume_queued_client_work(restarted, api, reloaded)
    await restarted._sync_positions_with_server(api, reloaded)
    state = await reloaded.get_state()
    assert state.server_positions[str(history_file)] == 3
    assert state.local_positions[str(history_file)] == 3
    assert state.pending_uploads == []


async def test_safe_end_noop_followup_is_durable_before_queue(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = history_file.read_text().splitlines()
    history_file.write_text("\n".join(source[:3]) + "\n")
    manager = await manager_at(tmp_path, history_file, local_position=2)
    watcher = make_watcher(history_file, tmp_path)
    interval = CatchupInterval(
        file_path=str(history_file),
        start_exclusive=1,
        end_inclusive=2,
    )

    async def cancel_queue(followup):
        raise asyncio.CancelledError

    monkeypatch.setattr(watcher, "_queue_catchup_interval", cancel_queue)
    with pytest.raises(asyncio.CancelledError):
        await watcher._index_file_client_mode(interval, FakeAPI(history_file), manager)

    reloaded = ClientStateManager(manager.state_path)
    state = await reloaded.get_state()
    assert any(record.followup_interval is not None for record in state.pending_uploads)

    restarted = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file, server_position=1)
    assert await restarted._process_pending_uploads(api, reloaded) == 0
    await consume_queued_client_work(restarted, api, reloaded)
    state = await reloaded.get_state()
    assert state.server_positions[str(history_file)] == 3
    assert state.local_positions[str(history_file)] == 3


async def test_direct_process_files_cancellation_reaps_wait_children_and_preserves_future_work(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings, "server_url", None)
    monkeypatch.setattr(settings, "storage_backend", "spanner")
    monkeypatch.setattr(settings, "spanner_embedding_mode", "spanner")
    for run in range(100):
        watcher = make_watcher(history_file, tmp_path / f"cancel-{run}")
        watcher._running = True
        before = set(asyncio.all_tasks())
        task = asyncio.create_task(watcher._process_files())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)

        live_children = [child for child in asyncio.all_tasks() - before if not child.done()]
        await watcher.queue.put(history_file)
        await asyncio.sleep(0)

        assert live_children == []
        assert watcher.queue.qsize() == 1
        assert watcher.queue._unfinished_tasks == 1


async def test_concurrent_pending_drains_have_one_network_owner(
    tmp_path: Path,
    history_file: Path,
):
    class BarrierAPI(FakeAPI):
        def __init__(self, file_path: Path):
            super().__init__(file_path)
            self.calls = 0
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def upload_chunks(self, *, chunks, source_file, file_position):
            self.calls += 1
            self.entered.set()
            await self.release.wait()
            return await super().upload_chunks(
                chunks=chunks,
                source_file=source_file,
                file_position=file_position,
            )

    for run in range(100):
        run_path = tmp_path / f"drain-{run}"
        manager = ClientStateManager(run_path / "client-state.json")
        await manager.replace_pending_upload(
            str(history_file),
            [{"id": f"one-{run}", "source_line": 1}],
            1,
        )
        watcher = make_watcher(history_file, run_path)
        competing_watcher = make_watcher(history_file, run_path)
        api = BarrierAPI(history_file)
        first = asyncio.create_task(watcher._process_pending_uploads(api, manager))
        await asyncio.wait_for(api.entered.wait(), timeout=1)
        second = asyncio.create_task(competing_watcher._process_pending_uploads(api, manager))
        await asyncio.sleep(0)
        observed_calls = api.calls
        api.release.set()
        results = await asyncio.gather(first, second)

        assert observed_calls == 1
        assert api.calls == 1
        assert sorted(results) == [0, 1]


async def test_record_creation_snapshot_failure_rolls_back_memory_disk_and_payload(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state_path = tmp_path / "client-state.json"
    manager = ClientStateManager(state_path)
    await manager.save()
    original_write = manager._write_state_snapshot
    record_id = "stable-record-id"

    def fail_once(*args):
        raise OSError("injected create snapshot failure")

    monkeypatch.setattr(manager, "_write_state_snapshot", fail_once)
    with pytest.raises(OSError, match="create snapshot failure"):
        await manager.replace_pending_upload(
            str(history_file),
            [{"id": "one", "source_line": 1}],
            1,
            record_id=record_id,
        )

    assert (await manager.get_state()).pending_uploads == []
    assert (await ClientStateManager(state_path).get_state()).pending_uploads == []
    assert not manager._payload_path(record_id).exists()

    monkeypatch.setattr(manager, "_write_state_snapshot", original_write)
    await manager.replace_pending_upload(
        str(history_file),
        [{"id": "one", "source_line": 1}],
        1,
        record_id=record_id,
    )
    restarted = await ClientStateManager(state_path).get_state()
    assert [record.record_id for record in restarted.pending_uploads] == [record_id]


async def test_ack_snapshot_failure_retains_retry_in_memory_disk_and_restart(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        str(history_file),
        [{"id": "one", "source_line": 1}],
        1,
        record_id="ack-record",
    )
    watcher = make_watcher(history_file, tmp_path)
    api = FakeAPI(history_file)
    original_write = manager._write_state_snapshot
    writes = 0

    def fail_first(*args):
        nonlocal writes
        writes += 1
        if writes == 1:
            raise OSError("injected ack snapshot failure")
        return original_write(*args)

    monkeypatch.setattr(manager, "_write_state_snapshot", fail_first)
    assert await watcher._process_pending_uploads(api, manager) == 0

    assert [record.record_id for record in (await manager.get_state()).pending_uploads] == [
        "ack-record"
    ]
    restarted_after_failure = await ClientStateManager(manager.state_path).get_state()
    assert [record.record_id for record in restarted_after_failure.pending_uploads] == [
        "ack-record"
    ]

    assert await watcher._process_pending_uploads(api, manager) == 1
    assert len(api.uploads) == 2
    assert (await manager.get_state()).pending_uploads == []
    assert (await ClientStateManager(manager.state_path).get_state()).pending_uploads == []


async def test_raw_path_cancellation_before_durable_boundary_requeues_claim(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    api = FakeAPI(history_file)
    watcher = make_watcher(history_file, tmp_path)
    entered = asyncio.Event()

    async def block_before_durable_boundary(interval):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(settings, "server_url", "https://history.example.test")
    monkeypatch.setattr(api_client_module, "get_api_client", lambda: api)
    monkeypatch.setattr(client_state_module, "get_client_state_manager", lambda: manager)
    monkeypatch.setattr(manager, "add_continuation", block_before_durable_boundary)
    await watcher.queue.put(history_file)
    watcher._running = True

    task = asyncio.create_task(watcher._process_files())
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert watcher.queue.get_nowait() == history_file
    assert watcher.queue._unfinished_tasks == 1


async def test_interval_cancellation_after_durable_boundary_keeps_durable_owner(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    interval = await manager.add_continuation(
        CatchupInterval(file_path=str(history_file), start_exclusive=0, end_inclusive=1)
    )
    api = FakeAPI(history_file)
    watcher = make_watcher(history_file, tmp_path)
    entered = asyncio.Event()

    async def block_after_durable_boundary(work, api_client, state_manager):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(settings, "server_url", "https://history.example.test")
    monkeypatch.setattr(api_client_module, "get_api_client", lambda: api)
    monkeypatch.setattr(client_state_module, "get_client_state_manager", lambda: manager)
    monkeypatch.setattr(watcher, "_index_file_client_mode", block_after_durable_boundary)
    await watcher.queue.put(interval)
    watcher._queued_catchups.add(
        (
            interval.file_path,
            interval.start_exclusive,
            interval.end_inclusive,
            interval.generation,
        )
    )
    watcher._running = True

    with caplog.at_level("INFO"):
        task = asyncio.create_task(watcher._process_files())
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert watcher.queue.empty()
    assert watcher.queue._unfinished_tasks == 0
    assert watcher._queued_catchups == set()
    assert len((await ClientStateManager(manager.state_path).get_state()).pending_uploads) == 1
    assert "outcome=durable_owner_retained" in caplog.text
    assert f"record={interval.outbox_record_id}" in caplog.text
    assert "generation=0" in caplog.text
    assert "interval=(0,1]" in caplog.text
    assert str(history_file) not in caplog.text


async def test_oversize_legacy_migration_fails_without_filesystem_mutation(tmp_path: Path):
    state_path = tmp_path / "client-state.json"
    legacy = {
        "pending_uploads": [
            {
                "file_path": "/history/oversize.jsonl",
                "chunks": [
                    {
                        "id": "oversize",
                        "content": "x" * MAX_CHUNK_UPLOAD_REQUEST_BYTES,
                    }
                ],
                "file_position": 1,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]
    }
    original = json.dumps(legacy)
    state_path.write_text(original)
    manager = ClientStateManager(state_path)

    with pytest.raises(RuntimeError, match="durable client outbox state"):
        await manager.get_state()

    assert state_path.read_text() == original
    assert not manager._outbox_dir.exists()
    assert not state_path.with_suffix(f"{state_path.suffix}.tmp").exists()


async def test_reindex_response_cannot_restore_retired_upload_cursor(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        str(history_file),
        [{"id": "one", "source_line": 1}],
        7,
        record_id="stale-before-reindex",
    )
    watcher = make_watcher(history_file, tmp_path)

    class ReindexAPI(FakeAPI):
        async def upload_chunks(self, *, chunks, source_file, file_position):
            response = await super().upload_chunks(
                chunks=chunks,
                source_file=source_file,
                file_position=file_position,
            )
            response.reindex_required = True
            response.reindex_requested_at = "2026-08-04T12:00:00+00:00"
            return response

        async def ack_reindex(self, **kwargs):
            return SimpleNamespace(status="ok")

    async def queue_reindex():
        return 1

    monkeypatch.setattr(
        "claude_history_rag.watcher._queue_all_watchers_for_reindex",
        queue_reindex,
    )
    assert await watcher._process_pending_uploads(ReindexAPI(history_file), manager) == 0

    state = await manager.get_state()
    assert state.pending_uploads == []
    assert state.local_positions == {}
    assert state.server_positions == {}
    assert state.reindex_status == "queued"
    restarted = await ClientStateManager(manager.state_path).get_state()
    assert restarted.local_positions == {}
    assert restarted.server_positions == {}


async def test_exact_one_mib_body_passes_real_aiohttp_transport(
    monkeypatch: pytest.MonkeyPatch,
):
    server = await configure_status_server(monkeypatch)
    monkeypatch.setattr(status_server_module.settings, "storage_backend", "spanner")
    monkeypatch.setattr(status_server_module.settings, "spanner_embedding_mode", "spanner")

    class AcceptStore:
        async def add_chunks_async(self, chunks):
            return None

    monkeypatch.setattr(store_module, "store", AcceptStore())
    base = {
        "machine_id": "machine-a",
        "client_name": "machine-a",
        "chunks": [
            {
                "id": "boundary",
                "content": "",
                "chunk_type": "turn",
                "session_id": "session",
                "project_path": "/project",
                "project_name": "project",
                "timestamp": "2026-01-01T00:00:00Z",
                "source_file": "/history/session.jsonl",
                "source_line": 3,
                "machine_id": "machine-a",
            }
        ],
        "source_file": "/history/session.jsonl",
        "file_position": 3,
    }

    def compact(payload):
        return json.dumps(payload, separators=(",", ":")).encode()

    base_size = len(compact(base))
    base["chunks"][0]["content"] = "x" * (MAX_CHUNK_UPLOAD_REQUEST_BYTES - base_size)
    body = compact(base)
    assert len(body) == MAX_CHUNK_UPLOAD_REQUEST_BYTES

    client = TestClient(TestServer(server.app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/chunks",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        assert response.status == 200
    finally:
        await client.close()


async def test_tampered_payload_fails_before_network_or_cursor(
    tmp_path: Path,
    history_file: Path,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        str(history_file),
        [{"id": "original", "source_line": 1}],
        1,
        record_id="integrity-record",
    )
    manager._payload_path("integrity-record").write_text(
        json.dumps([{"id": "tampered", "source_line": 1}])
    )
    api = FakeAPI(history_file)

    assert await make_watcher(history_file, tmp_path)._process_pending_uploads(api, manager) == 0
    state = await manager.get_state()
    assert api.uploads == []
    assert state.local_positions == {}
    assert state.server_positions == {}
    assert [record.record_id for record in state.pending_uploads] == ["integrity-record"]


async def test_oversized_payload_file_is_bounded_before_network(
    tmp_path: Path,
    history_file: Path,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        str(history_file),
        [{"id": "original", "source_line": 1}],
        1,
        record_id="bounded-read-record",
    )
    manager._payload_path("bounded-read-record").write_bytes(
        b"[" + b"x" * MAX_CHUNK_UPLOAD_REQUEST_BYTES + b"]"
    )
    api = FakeAPI(history_file)

    assert await make_watcher(history_file, tmp_path)._process_pending_uploads(api, manager) == 0
    assert api.uploads == []
    assert len((await manager.get_state()).pending_uploads) == 1


async def test_duplicate_record_ids_fail_closed_before_legacy_migration(tmp_path: Path):
    state_path = tmp_path / "client-state.json"
    original = json.dumps(
        {
            "pending_uploads": [
                {
                    "record_id": "duplicate",
                    "file_path": "/history/a.jsonl",
                    "chunks": [{"id": "a"}],
                    "file_position": 1,
                    "created_at": "2026-01-01T00:00:00Z",
                },
                {
                    "record_id": "duplicate",
                    "file_path": "/history/b.jsonl",
                    "chunks": [{"id": "b"}],
                    "file_position": 2,
                    "created_at": "2026-01-01T00:00:00Z",
                },
            ]
        }
    )
    state_path.write_text(original)
    manager = ClientStateManager(state_path)

    with pytest.raises(RuntimeError, match="durable client outbox state"):
        await manager.get_state()
    assert state_path.read_text() == original
    assert not manager._outbox_dir.exists()


async def test_ordinary_pre_durable_failure_requeues_raw_claim(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    watcher = make_watcher(history_file, tmp_path)

    async def fail_once(interval):
        watcher._running = False
        raise OSError("injected pre-durable failure")

    monkeypatch.setattr(settings, "server_url", "https://history.example.test")
    monkeypatch.setattr(api_client_module, "get_api_client", lambda: FakeAPI(history_file))
    monkeypatch.setattr(client_state_module, "get_client_state_manager", lambda: manager)
    monkeypatch.setattr(manager, "add_continuation", fail_once)
    await watcher.queue.put(history_file)
    watcher._running = True

    await watcher._process_files()
    assert watcher.queue.get_nowait() == history_file
    assert watcher.queue._unfinished_tasks == 1


async def test_queue_full_cancellation_still_restores_raw_claim(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    watcher = make_watcher(history_file, tmp_path)
    watcher.queue._maxsize = 1
    entered = asyncio.Event()

    async def block(interval):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(settings, "server_url", "https://history.example.test")
    monkeypatch.setattr(api_client_module, "get_api_client", lambda: FakeAPI(history_file))
    monkeypatch.setattr(client_state_module, "get_client_state_manager", lambda: manager)
    monkeypatch.setattr(manager, "add_continuation", block)
    await watcher.queue.put(history_file)
    watcher._running = True
    task = asyncio.create_task(watcher._process_files())
    await asyncio.wait_for(entered.wait(), timeout=1)
    competing = tmp_path / "competing.jsonl"
    competing.write_text("{}\n")
    watcher.queue.put_nowait(competing)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert watcher.queue.qsize() == 2
    assert watcher.queue._unfinished_tasks == 2
    assert {watcher.queue.get_nowait(), watcher.queue.get_nowait()} == {
        history_file,
        competing,
    }


async def test_reindex_begin_snapshot_failure_is_atomic_and_same_request_resumes(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(str(history_file), [{"id": "old", "source_line": 1}], 7)
    original_write = manager._write_state_snapshot

    def fail(*args):
        raise OSError("injected reindex begin failure")

    monkeypatch.setattr(manager, "_write_state_snapshot", fail)
    requested_at = datetime(2026, 8, 4, 12, tzinfo=timezone.utc)
    with pytest.raises(OSError, match="reindex begin failure"):
        await manager.prepare_reindex(requested_at)
    assert len((await manager.get_state()).pending_uploads) == 1
    assert (await ClientStateManager(manager.state_path).get_state()).reindex_required_at is None

    monkeypatch.setattr(manager, "_write_state_snapshot", original_write)
    assert await manager.prepare_reindex(requested_at) == "reset"
    assert await manager.prepare_reindex(requested_at) == "resume"
    state = await manager.get_state()
    assert state.reindex_status == "queue_pending"
    assert state.pending_uploads == []


async def test_stale_interval_cannot_repopulate_reset_generation(
    tmp_path: Path,
    history_file: Path,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    stale = await manager.add_continuation(
        CatchupInterval(
            file_path=str(history_file),
            start_exclusive=0,
            end_inclusive=3,
            generation=0,
        )
    )
    await manager.prepare_reindex(datetime(2026, 8, 4, 12, tzinfo=timezone.utc))

    await make_watcher(history_file, tmp_path)._index_file_client_mode(
        stale, FakeAPI(history_file), manager
    )
    state = await manager.get_state()
    assert state.reindex_generation == 1
    assert state.pending_uploads == []
    assert state.local_positions == {}
    assert state.server_positions == {}


async def test_same_reindex_request_resumes_after_queue_phase_failure(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(str(history_file), [{"id": "old", "source_line": 1}], 7)

    class AckAPI(FakeAPI):
        def __init__(self, file_path):
            super().__init__(file_path)
            self.acks = []

        async def ack_reindex(self, **kwargs):
            self.acks.append(kwargs)

    attempts = 0

    async def queue_with_first_failure():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected queue phase failure")
        return 3

    monkeypatch.setattr(watcher_module, "_queue_all_watchers_for_reindex", queue_with_first_failure)
    api = AckAPI(history_file)
    requested_at = "2026-08-04T12:00:00+00:00"
    with pytest.raises(OSError, match="queue phase failure"):
        await watcher_module._handle_server_reindex(api, manager, requested_at)

    failed = await manager.get_state()
    assert failed.reindex_generation == 1
    assert failed.reindex_status == "queue_pending"
    assert failed.pending_uploads == []

    await watcher_module._handle_server_reindex(api, manager, requested_at)
    resumed = await manager.get_state()
    assert resumed.reindex_generation == 1
    assert resumed.reindex_status == "queued"
    assert attempts == 2
    assert len(api.acks) == 1


async def test_reindex_completion_waits_for_claimed_queue_work(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.prepare_reindex(datetime(2026, 8, 4, 12, tzinfo=timezone.utc))
    await manager.mark_reindex_queued()
    watcher = make_watcher(history_file, tmp_path)
    watcher._active_queue_claims = 1
    monkeypatch.setattr(watcher_module, "get_all_watchers", lambda: [watcher])

    class AckAPI(FakeAPI):
        def __init__(self, file_path):
            super().__init__(file_path)
            self.acks = []

        async def ack_reindex(self, **kwargs):
            self.acks.append(kwargs)

    api = AckAPI(history_file)
    await watcher_module._maybe_ack_reindex_completed(api, manager)
    assert api.acks == []
    assert (await manager.get_state()).reindex_status == "queued"


async def test_full_reindex_enumeration_does_not_drop_at_queue_capacity(tmp_path: Path):
    for index in range(5):
        (tmp_path / f"history-{index}.jsonl").write_text("{}\n")
    watcher = HistoryWatcher(projects_path=tmp_path)
    watcher.queue._maxsize = 2

    assert await watcher.queue_all_files_for_indexing() == 5
    assert watcher.queue.qsize() == 5
    assert watcher.queue._unfinished_tasks == 5


async def test_reindex_response_aborts_stale_multi_record_drain(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(str(history_file), [{"id": "first", "source_line": 1}], 1)
    await manager.replace_pending_upload(str(history_file), [{"id": "second", "source_line": 2}], 2)
    watcher = make_watcher(history_file, tmp_path)

    class ReindexFirstAPI(FakeAPI):
        def __init__(self, file_path):
            super().__init__(file_path)
            self.calls = 0

        async def upload_chunks(self, **kwargs):
            self.calls += 1
            response = await super().upload_chunks(**kwargs)
            response.reindex_required = True
            response.reindex_requested_at = "2026-08-04T12:00:00+00:00"
            return response

        async def ack_reindex(self, **kwargs):
            return None

    async def queue_reindex():
        return 0

    monkeypatch.setattr(watcher_module, "_queue_all_watchers_for_reindex", queue_reindex)
    api = ReindexFirstAPI(history_file)
    assert await watcher._process_pending_uploads(api, manager) == 0
    assert api.calls == 1
    assert (await manager.get_state()).pending_uploads == []


async def test_same_in_progress_reindex_flag_does_not_poison_current_generation(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    requested_at = datetime(2026, 8, 4, 12, tzinfo=timezone.utc)
    await manager.prepare_reindex(requested_at)
    await manager.mark_reindex_queued()
    await manager.replace_pending_upload(
        str(history_file), [{"id": "current", "source_line": 1}], 1
    )

    class StillFlaggedAPI(FakeAPI):
        async def upload_chunks(self, **kwargs):
            response = await super().upload_chunks(**kwargs)
            response.reindex_required = True
            response.reindex_requested_at = requested_at.isoformat()
            return response

        async def ack_reindex(self, **kwargs):
            return None

    async def must_not_requeue():
        raise AssertionError("same live generation must not restart queue phase")

    monkeypatch.setattr(watcher_module, "_queue_all_watchers_for_reindex", must_not_requeue)
    watcher = make_watcher(history_file, tmp_path)
    assert await watcher._process_pending_uploads(StillFlaggedAPI(history_file), manager) == 1
    state = await manager.get_state()
    assert state.pending_uploads == []
    assert state.local_positions[str(history_file)] == 1
    assert state.reindex_generation == 1


async def test_live_source_replacement_during_spool_never_reaches_the_outbox(
    tmp_path: Path,
    history_file: Path,
):
    """Replacing the live source mid-parse cannot contaminate the bounded work.

    The parse reads an immutable snapshot, so a replacement written to the live
    path during chunking is simply not part of this operation: the record that
    persists carries only content that was digest-verified, and the replacement
    bytes never appear in the outbox.
    """
    outside_sentinel = "OUTSIDE_SUBSTITUTED_HISTORY_CONTENT"
    original_lines = history_file.read_text().splitlines()
    history_file.write_text("\n".join(original_lines[:3]) + "\n")
    replaced = False
    snapshot_paths: list[str] = []

    def replacing_chunker(path: Path, start_line: int, end_line: int | None = None):
        nonlocal replaced
        snapshot_paths.append(str(path))
        yield Chunk(
            id="verified-content",
            content="content",
            chunk_type="turn",
            session_id="session",
            project_path=str(tmp_path),
            project_name="project",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            source_file=str(path),
            source_line=2,
            consumed_line=3,
        )
        # Replace the LIVE source, not the snapshot handed to this chunker.
        history_file.write_text(outside_sentinel + "\n")
        replaced = True

    manager = ClientStateManager(tmp_path / "client-state.json")
    watcher = HistoryWatcher(projects_path=tmp_path, chunker=replacing_chunker)
    api = FakeAPI(history_file)
    await watcher._index_file_client_mode(history_file, api, manager)

    assert replaced is True
    # The chunker was never handed the live path.
    assert snapshot_paths and str(history_file) not in snapshot_paths

    state = await manager.get_state()
    persisted = json.dumps(await pending_payloads(manager, state.pending_uploads))
    assert outside_sentinel not in json.dumps(api.uploads)
    assert outside_sentinel not in persisted
    # Provenance is the original source path, never the snapshot location.
    for upload in api.uploads:
        assert upload["source_file"] == str(history_file)
    for record in state.pending_uploads:
        assert record.file_path == str(history_file)
    rendered_state = json.dumps(state.model_dump(mode="json"))
    for snapshot_path in snapshot_paths:
        assert json_fragment(snapshot_path) not in rendered_state
    assert json_fragment(str(history_file)) in rendered_state


async def test_replaced_source_is_refused_on_the_next_catch_up_pass(
    tmp_path: Path,
    history_file: Path,
):
    """A source replaced between snapshots is refused, not silently consumed."""
    original_lines = history_file.read_text().splitlines()
    history_file.write_text("\n".join(original_lines[:3]) + "\n")

    manager = ClientStateManager(tmp_path / "client-state.json")
    watcher = make_watcher(history_file, tmp_path)
    interval = await manager.add_continuation(
        CatchupInterval(
            file_path=str(history_file),
            start_exclusive=0,
            end_inclusive=3,
            snapshot_digest="0" * 64,
            generation=0,
        )
    )

    api = FakeAPI(history_file)
    await watcher._index_file_client_mode(interval, api, manager)

    state = await manager.get_state()
    failure = state.catchup_failures[str(history_file)]
    assert failure.reason == "history_replaced"
    assert api.uploads == []
    assert state.local_positions == {}
    assert state.server_positions == {}


async def test_get_state_returns_isolated_read_snapshot(tmp_path: Path):
    manager = ClientStateManager(tmp_path / "client-state.json")
    snapshot = await manager.get_state()
    snapshot.local_positions["/mutated"] = 9

    assert (await manager.get_state()).local_positions == {}
    assert (await ClientStateManager(manager.state_path).get_state()).local_positions == {}


async def test_heartbeat_degrades_for_durable_failures_retries_and_active_claims(
    tmp_path: Path,
    history_file: Path,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        str(history_file), [{"id": "blocked", "source_line": 1}], 1
    )
    record_id = (await manager.get_state()).pending_uploads[0].record_id
    for _ in range(4):
        await manager.increment_retry_count(record_id)
    await manager.record_catchup_failure(
        CatchupInterval(file_path=str(history_file), start_exclusive=1, end_inclusive=2),
        "history_truncated",
    )
    await manager.set_connected(True)
    watcher = make_watcher(history_file, tmp_path)
    watcher._active_queue_claims = 1

    heartbeat = await watcher._collect_client_heartbeat(manager)
    assert heartbeat["status"] == "degraded"
    assert heartbeat["queue"]["active_claims"] == 1
    assert heartbeat["queue"]["blocked_records"] == 1
    assert heartbeat["queue"]["retry_total"] == 4
    assert heartbeat["errors"]["catchup_failure_reasons"] == {"history_truncated": 1}
    assert heartbeat["reindex"]["generation"] == 0


async def test_corrupt_state_log_redacts_rejected_input(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    secret = "TOP_SECRET_CONVERSATION_TEXT"
    state_path = tmp_path / "client-state.json"
    state_path.write_text(
        json.dumps(
            {
                "pending_uploads": [
                    {
                        "record_id": "redaction-test",
                        "file_path": "/history/session.jsonl",
                        "chunks": [],
                        "file_position": secret,
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                ]
            }
        )
    )

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError):
        await ClientStateManager(state_path).get_state()
    assert secret not in caplog.text
    assert "ValidationError" in caplog.text
    assert "reason=state_schema_invalid" in caplog.text
    assert "record=redaction-test" in caplog.text


async def test_partial_response_log_has_record_context_without_source_path(
    tmp_path: Path,
    history_file: Path,
    caplog: pytest.LogCaptureFixture,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        str(history_file),
        [{"id": "partial", "source_line": 1}],
        1,
        record_id="partial-record",
    )
    api = FailOnCallAPI(history_file, fail_on_call=1)

    with caplog.at_level("WARNING"):
        assert (
            await make_watcher(history_file, tmp_path)._process_pending_uploads(api, manager) == 0
        )
    assert "record=partial-record" in caplog.text
    assert "retry=1" in caplog.text
    assert str(history_file) not in caplog.text


async def test_unicode_payload_uses_wire_equivalent_utf8_size_and_drains(
    tmp_path: Path,
    history_file: Path,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    content = "é" * 180_000
    chunks = [{"id": "unicode", "content": content, "source_line": 1}]

    await manager.replace_pending_upload(str(history_file), chunks, 1)
    pending = (await manager.get_state()).pending_uploads[0]
    payload = manager._payload_path(pending.record_id).read_bytes()

    assert len(payload) < MAX_CHUNK_UPLOAD_REQUEST_BYTES
    assert pending.payload_bytes == len(payload)
    assert b"\\u00e9" not in payload
    assert (
        await make_watcher(history_file, tmp_path)._process_pending_uploads(
            FakeAPI(history_file), manager
        )
        == 1
    )


@pytest.mark.parametrize("remove_schema_version", [False, True])
async def test_external_payload_cannot_drop_current_integrity_contract(
    tmp_path: Path,
    history_file: Path,
    remove_schema_version: bool,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        str(history_file),
        [{"id": "integrity", "content": "original", "source_line": 1}],
        1,
        record_id="integrity-contract",
    )
    state_data = json.loads(manager.state_path.read_text())
    record = state_data["pending_uploads"][0]
    record.pop("payload_sha256")
    record.pop("payload_bytes")
    record.pop("request_bytes")
    if remove_schema_version:
        state_data.pop("schema_version")
    manager.state_path.write_text(json.dumps(state_data))
    manager._payload_path("integrity-contract").write_text(
        json.dumps([{"id": "integrity", "content": "tampered", "source_line": 1}])
    )

    with pytest.raises(RuntimeError, match="durable client outbox state"):
        await ClientStateManager(manager.state_path).get_state()


async def test_schema_v1_pending_reindex_migrates_to_resumable_phase(tmp_path: Path):
    state_path = tmp_path / "client-state.json"
    state_path.write_text(
        json.dumps(
            {
                "reindex_required_at": "2026-08-04T12:00:00+00:00",
                "reindex_status": "pending",
            }
        )
    )

    state = await ClientStateManager(state_path).get_state()

    assert state.schema_version == 3
    assert state.reindex_status == "queue_pending"
    assert state.reindex_queue_session is None
    assert json.loads(state_path.read_text())["schema_version"] == 3


async def test_current_state_rejects_unknown_fields(tmp_path: Path):
    state_path = tmp_path / "client-state.json"
    state_path.write_text(json.dumps({"schema_version": 3, "unknown_contract": True}))

    with pytest.raises(RuntimeError, match="durable client outbox state"):
        await ClientStateManager(state_path).get_state()


async def test_schema_v2_pending_upload_fails_closed_without_identity_upgrade(tmp_path: Path):
    state_path = tmp_path / "client-state.json"
    original = json.dumps(
        {
            "schema_version": 2,
            "pending_uploads": [
                {
                    "record_id": "v2-unbound",
                    "file_path": "/history/a.jsonl",
                    "chunks": [],
                    "file_position": 1,
                    "created_at": "2026-01-01T00:00:00Z",
                    "payload_externalized": True,
                    "payload_sha256": "0" * 64,
                    "payload_bytes": 2,
                    "request_bytes": 128,
                }
            ],
        }
    )
    state_path.write_text(original)

    with pytest.raises(RuntimeError, match="durable client outbox state"):
        await ClientStateManager(state_path).get_state()

    assert state_path.read_text() == original


async def test_schema_v2_without_upload_work_upgrades_to_current(tmp_path: Path):
    state_path = tmp_path / "client-state.json"
    state_path.write_text(json.dumps({"schema_version": 2, "pending_uploads": []}))

    state = await ClientStateManager(state_path).get_state()

    assert state.schema_version == 3
    assert json.loads(state_path.read_text())["schema_version"] == 3


async def test_catchup_interval_inherits_current_reindex_generation(tmp_path: Path):
    manager = ClientStateManager(tmp_path / "client-state.json")
    requested_at = datetime(2026, 8, 4, 12, tzinfo=timezone.utc)
    assert await manager.prepare_reindex(requested_at) == "reset"
    await manager.update_local_position("/history/current.jsonl", 7)

    intervals = await manager.get_files_needing_catchup({"/history/current.jsonl": 3})

    assert len(intervals) == 1
    assert intervals[0].generation == 1


@pytest.mark.parametrize(
    "response_type,kwargs",
    [
        (
            ChunkUploadResponse,
            {
                "status": "ok",
                "chunks_received": 0,
                "chunks_embedded": 0,
                "chunks_stored": 0,
            },
        ),
        (
            GetPositionsResponse,
            {"machine_id": "machine", "positions": {}},
        ),
    ],
)
def test_reindex_response_contract_rejects_unidentified_request_or_naive_timestamp(
    response_type,
    kwargs,
):
    # A demanded reindex must say which request it is.
    with pytest.raises(ValueError, match="must carry reindex_requested_at"):
        response_type(**kwargs, reindex_required=True)
    # Whenever a request identity is reported, it must be unambiguous in time.
    with pytest.raises(ValueError, match="timezone"):
        response_type(
            **kwargs,
            reindex_required=True,
            reindex_requested_at="2026-08-04T12:00:00",
        )
    with pytest.raises(ValueError, match="timezone"):
        response_type(
            **kwargs,
            reindex_required=False,
            reindex_requested_at="2026-08-04T12:00:00",
        )
    # The converse is NOT an error. Once a client acknowledges a request the
    # registry keeps reporting that request's identity with required=False so
    # the client can recognize it as already handled. Rejecting this pair broke
    # every upload and position sync for an acknowledged client.
    accepted = response_type(
        **kwargs,
        reindex_required=False,
        reindex_requested_at="2026-08-04T12:00:00+00:00",
    )
    assert accepted.reindex_required is False
    assert accepted.reindex_requested_at == "2026-08-04T12:00:00+00:00"


def test_response_contract_rejects_coerced_or_negative_progress():
    with pytest.raises(ValueError):
        ChunkUploadResponse(
            status="ok",
            chunks_received="1",
            chunks_embedded=1,
            chunks_stored=1,
        )
    with pytest.raises(ValueError):
        ChunkUploadResponse(
            status="ok",
            chunks_received=-1,
            chunks_embedded=0,
            chunks_stored=0,
        )
    with pytest.raises(ValueError):
        GetPositionsResponse(machine_id="machine", positions={"/history": "1"})
    with pytest.raises(ValueError):
        GetPositionsResponse(machine_id="machine", positions={"/history": -1})


async def test_malformed_reindex_flag_does_not_invalidate_current_state(tmp_path: Path):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.update_local_position("/history/current.jsonl", 7)
    api = SimpleNamespace()

    with pytest.raises(ValueError, match="missing"):
        await watcher_module._handle_server_reindex(api, manager, None)
    with pytest.raises(ValueError, match="invalid"):
        await watcher_module._handle_server_reindex(api, manager, "not-a-timestamp")

    assert (await manager.get_state()).local_positions == {"/history/current.jsonl": 7}


async def test_upload_failure_log_redacts_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    secret = "TOP_SECRET_CONVERSATION_TEXT"
    server = await configure_status_server(monkeypatch)
    monkeypatch.setattr(status_server_module.settings, "storage_backend", "spanner")
    monkeypatch.setattr(status_server_module.settings, "spanner_embedding_mode", "spanner")

    class RejectStore:
        async def add_chunks_async(self, chunks):
            raise RuntimeError(secret)

    monkeypatch.setattr(store_module, "store", RejectStore())
    payload = {
        "machine_id": "machine-a",
        "chunks": [
            {
                "id": "chunk-a",
                "content": "content",
                "chunk_type": "turn",
                "session_id": "session",
                "project_path": "/project",
                "project_name": "project",
                "source_file": "/history/session.jsonl",
                "source_line": 2,
            }
        ],
        "source_file": "/history/session.jsonl",
        "file_position": 3,
    }

    with caplog.at_level("ERROR"):
        response = await server.handle_api_chunks(fake_request(payload))

    assert response.status == 500
    assert secret not in caplog.text
    assert "error_type=RuntimeError" in caplog.text


async def test_client_control_loop_logs_redact_exception_text(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    secret = "TOP_SECRET_CONVERSATION_TEXT"
    manager = ClientStateManager(tmp_path / "client-state.json")
    watcher = make_watcher(history_file, tmp_path)

    async def collect(_manager):
        return {}

    class RejectHeartbeat:
        async def send_heartbeat(self, payload):
            raise RuntimeError(secret)

    monkeypatch.setattr(watcher, "_collect_client_heartbeat", collect)
    with caplog.at_level("WARNING"):
        await watcher._send_client_heartbeat(RejectHeartbeat(), manager)
    assert secret not in caplog.text
    assert "Heartbeat failed: error_type=RuntimeError" in caplog.text

    async def fail_pending(_api, _manager):
        watcher._shutdown_event.set()
        watcher._running = False
        raise RuntimeError(secret)

    monkeypatch.setattr(watcher, "_process_pending_uploads", fail_pending)
    watcher._running = True
    watcher._shutdown_event.clear()
    caplog.clear()
    with caplog.at_level("WARNING"):
        await watcher._client_sync_loop(SimpleNamespace(), manager)
    assert secret not in caplog.text
    assert "Client sync loop error: error_type=RuntimeError" in caplog.text


@pytest.mark.parametrize(
    "corruption,expected_reason",
    [("missing", "payload_missing"), ("digest", "payload_digest_mismatch")],
)
async def test_corrupt_payload_logs_safe_actionable_reason(
    tmp_path: Path,
    history_file: Path,
    caplog: pytest.LogCaptureFixture,
    corruption: str,
    expected_reason: str,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        str(history_file),
        [{"id": "payload", "content": "original", "source_line": 1}],
        1,
        record_id="corrupt-payload-record",
    )
    payload_path = manager._payload_path("corrupt-payload-record")
    if corruption == "missing":
        payload_path.unlink()
    else:
        payload_path.write_text('[{"id":"payload","content":"changed"}]')

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError):
        await ClientStateManager(manager.state_path).get_state()

    assert f"reason={expected_reason}" in caplog.text
    assert "record=corrupt-payload-record" in caplog.text
    assert str(history_file) not in caplog.text
    assert "original" not in caplog.text
    assert "changed" not in caplog.text


async def test_client_chunk_failure_redacts_source_and_exception_content(
    tmp_path: Path,
    history_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    secret_name = "TOP_SECRET_FILENAME.jsonl"
    secret_error = "TOP_SECRET_CONVERSATION_TEXT"
    secret_path = history_file.with_name(secret_name)
    shutil.copyfile(history_file, secret_path)
    diagnostics: list[tuple[str, str, dict]] = []

    def fail_chunking(path, start_line, end_line=None):
        raise RuntimeError(secret_error)
        yield  # pragma: no cover - preserve generator failure timing

    monkeypatch.setattr(
        watcher_module,
        "record_error",
        lambda kind, message, details: diagnostics.append((kind, message, details)),
    )
    watcher = HistoryWatcher(projects_path=tmp_path, chunker=fail_chunking)
    manager = ClientStateManager(tmp_path / "client-state.json")

    with caplog.at_level("DEBUG"):
        await watcher._index_file_client_mode(secret_path, FakeAPI(secret_path), manager)

    rendered_diagnostics = json.dumps(diagnostics)
    assert secret_name not in caplog.text
    assert secret_error not in caplog.text
    assert secret_name not in rendered_diagnostics
    assert secret_error not in rendered_diagnostics
    assert "error_type=RuntimeError" in caplog.text
    assert diagnostics[0][2]["error_type"] == "RuntimeError"
    assert diagnostics[0][2]["source_hash"]


async def test_server_position_error_text_is_not_logged_or_applied(
    tmp_path: Path,
    history_file: Path,
    caplog: pytest.LogCaptureFixture,
):
    secret = "TOP_SECRET_SERVER_TEXT\nFORGED_LOG_LINE"
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.update_local_position(str(history_file), 3)
    watcher = make_watcher(history_file, tmp_path)

    class ErrorAPI:
        async def get_positions(self):
            return SimpleNamespace(
                error=secret,
                reindex_required=False,
                reindex_requested_at=None,
                positions={str(history_file): 99},
            )

    with caplog.at_level("WARNING"):
        await watcher._sync_positions_with_server(ErrorAPI(), manager)

    state = await manager.get_state()
    assert state.server_positions == {}
    assert secret not in caplog.text
    assert "FORGED_LOG_LINE" not in caplog.text
    assert "reason=server_error_response" in caplog.text


async def test_real_chunk_endpoint_returns_stable_invalid_json_error(
    caplog: pytest.LogCaptureFixture,
):
    server = status_server_module.StatusServer()
    client = TestClient(TestServer(server.app))
    await client.start_server()
    try:
        with caplog.at_level("ERROR"):
            response = await client.post(
                "/api/chunks",
                data=b'{"machine_id": "secret-machine",',
                headers={"Content-Type": "application/json"},
            )
            body = await response.json()
    finally:
        await client.close()

    assert response.status == 400
    assert body["error"] == "invalid_json"
    assert "Expecting" not in caplog.text
    assert "secret-machine" not in caplog.text
    assert "reason=json_invalid" in caplog.text


async def test_authenticated_client_fields_cannot_forge_upload_or_reindex_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    forged = "TOP_SECRET_CLIENT\nFORGED_LOG_LINE"
    server = await configure_status_server(monkeypatch)
    registry = FakeRegistry()
    registry.acks = []
    registry.ack_reindex = lambda machine_id, **kwargs: registry.acks.append((machine_id, kwargs))
    monkeypatch.setattr(status_server_module, "get_client_registry", lambda: registry)
    monkeypatch.setattr(status_server_module.settings, "storage_backend", "spanner")
    monkeypatch.setattr(status_server_module.settings, "spanner_embedding_mode", "spanner")

    with caplog.at_level("DEBUG"):
        upload_response = await server.handle_api_chunks(
            fake_request(
                {
                    "machine_id": "machine-a",
                    "client_name": forged,
                    "chunks": [],
                    "source_file": "/history/session.jsonl",
                    "file_position": 1,
                }
            )
        )
        ack_response = await server.handle_api_reindex_ack(
            fake_request(
                {
                    "machine_id": "machine-a",
                    "client_name": forged,
                    "reindex_requested_at": "2026-08-04T12:00:00+00:00",
                    "status": "queued",
                    "reason": forged,
                }
            )
        )

    assert upload_response.status == 200
    assert ack_response.status == 200
    assert forged not in caplog.text
    assert "FORGED_LOG_LINE" not in caplog.text
    assert "client_hash=" in caplog.text
    assert registry.acks


async def test_status_endpoints_log_machine_hash_never_raw_identifiers(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Upload, cursor-only upload, stored upload, ack and purge all hash identity."""
    machine_id = "MACHINE_IDENTIFIER_SENTINEL"
    source_file = "/history/PRIVATE_PROJECT/PRIVATE_SESSION.jsonl"
    machine_hash = hashlib.sha256(machine_id.encode("utf-8")).hexdigest()[:12]
    source_digest = hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:12]

    server = await configure_status_server(monkeypatch)
    registry = FakeRegistry()
    registry.acks = []
    registry.ack_reindex = lambda mid, **kwargs: registry.acks.append((mid, kwargs))
    registry.mark_purged = lambda mid, client_name=None: None
    monkeypatch.setattr(status_server_module, "get_client_registry", lambda: registry)
    monkeypatch.setattr(status_server_module.settings, "storage_backend", "spanner")
    monkeypatch.setattr(status_server_module.settings, "spanner_embedding_mode", "spanner")

    stored: list[list[dict]] = []

    class FakeStore:
        async def add_chunks_async(self, chunks):
            stored.append(chunks)

        async def delete_by_machine_id_async(self, mid):
            return 3

    monkeypatch.setattr(store_module, "store", FakeStore())

    chunk = {
        "id": "chunk-1",
        "content": "content",
        "chunk_type": "turn",
        "session_id": "session",
        "project_path": "/project",
        "project_name": "project",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "source_file": source_file,
        "source_line": 1,
    }

    with caplog.at_level("DEBUG"):
        # Cursor-only upload (no chunks) and a stored upload (one chunk).
        cursor_only = await server.handle_api_chunks(
            fake_request(
                {
                    "machine_id": machine_id,
                    "client_name": machine_id,
                    "chunks": [],
                    "source_file": source_file,
                    "file_position": 1,
                }
            )
        )
        stored_upload = await server.handle_api_chunks(
            fake_request(
                {
                    "machine_id": machine_id,
                    "client_name": machine_id,
                    "chunks": [chunk],
                    "source_file": source_file,
                    "file_position": 2,
                }
            )
        )
        ack = await server.handle_api_reindex_ack(
            fake_request(
                {
                    "machine_id": machine_id,
                    "client_name": machine_id,
                    "reindex_requested_at": "2026-08-04T12:00:00+00:00",
                    "status": "queued",
                    "reason": "PURGE_REASON_SENTINEL",
                }
            )
        )
        purge = await server.handle_api_purge_client(
            fake_request({"machine_id": machine_id, "reason": "PURGE_REASON_SENTINEL"})
        )

    assert cursor_only.status == 200
    assert stored_upload.status == 200
    assert ack.status == 200
    assert purge.status == 200
    assert stored

    # No raw machine identifier, source path or free-text reason in diagnostics.
    assert machine_id not in caplog.text
    assert source_file not in caplog.text
    assert "PRIVATE_SESSION.jsonl" not in caplog.text
    assert "PURGE_REASON_SENTINEL" not in caplog.text
    assert "machine_id=" not in caplog.text

    # The stable hash is what joins these records together.
    assert caplog.text.count(f"machine_hash={machine_hash}") >= 5
    assert f"source={source_digest}" in caplog.text
    assert "Received chunk upload:" in caplog.text
    assert "Committed cursor-only upload:" in caplog.text
    assert "Stored uploaded chunks:" in caplog.text
    assert "Reindex ack:" in caplog.text
    assert "Purging client data:" in caplog.text

    # Response bodies still carry the real identifier: only logs are hashed.
    assert json.loads(purge.body.decode())["machine_id"] == machine_id


async def test_api_retry_never_logs_transport_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """A transport exception can carry request content and forged newlines."""
    secret = "TOP_SECRET_REQUEST_BODY\nFORGED_LOG_LINE"

    client = api_client_module.APIClient(
        server_url="http://localhost:9",
        machine_id="machine-a",
        client_name="machine-a",
        retry_count=2,
        retry_delay_seconds=0,
    )

    class ExplodingTransport:
        async def post(self, *args, **kwargs):
            raise RuntimeError(secret)

        async def get(self, *args, **kwargs):
            raise RuntimeError(secret)

    async def fake_ensure_client():
        return ExplodingTransport()

    monkeypatch.setattr(client, "_ensure_client", fake_ensure_client)
    monkeypatch.setattr(client, "_auth_headers", lambda: {})

    with caplog.at_level("DEBUG"), pytest.raises(api_client_module.ServerConnectionError):
        await client._request_with_retry("GET", "/health")

    assert secret not in caplog.text
    assert "FORGED_LOG_LINE" not in caplog.text
    assert "TOP_SECRET_REQUEST_BODY" not in caplog.text
    assert "reason=unexpected_transport_error" in caplog.text
    assert "error_type=RuntimeError" in caplog.text


async def test_acknowledged_reindex_state_does_not_break_upload_or_position_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A client that has acknowledged a reindex must keep uploading normally.

    The real registry reports ``(False, requested_at)`` once a client acks: the
    request identity is still returned so the client can recognize it as handled.
    Treating that legitimate pair as a contract violation rejected the response
    AFTER the chunks were stored and the position advanced, so the client never
    completed its outbox record and re-uploaded the same chunks forever.

    This uses the REAL ClientRegistry deliberately. A registry double is what
    hid this: it never reproduces the post-acknowledgement state.
    """
    server = status_server_module.StatusServer()

    async def allow(*args, **kwargs):
        return AuthCheckResult(ok=True, key_type="active")

    monkeypatch.setattr(server, "_require_auth", allow)
    status_server_module._clear_machine_positions()

    registry = ClientRegistry(tmp_path / "client_registry.json")
    monkeypatch.setattr(status_server_module, "get_client_registry", lambda: registry)
    monkeypatch.setattr(status_server_module.settings, "storage_backend", "spanner")
    monkeypatch.setattr(status_server_module.settings, "spanner_embedding_mode", "spanner")

    stored: list[list[dict]] = []

    class FakeStore:
        async def add_chunks_async(self, chunks):
            stored.append(list(chunks))

    monkeypatch.setattr(store_module, "store", FakeStore())

    machine_id = "machine-a"
    requested_at = registry.mark_reindex_requested()
    registry.ack_reindex(machine_id, reindex_requested_at=requested_at, status="completed")

    # This is the exact pair the response contract must tolerate.
    assert registry.get_reindex_status(machine_id) == (False, requested_at)

    chunk = {
        "id": "chunk-1",
        "content": "content",
        "chunk_type": "turn",
        "session_id": "session",
        "project_path": "/project",
        "project_name": "project",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "source_file": "/history/session.jsonl",
        "source_line": 1,
    }

    upload = await server.handle_api_chunks(
        fake_request(
            {
                "machine_id": machine_id,
                "client_name": machine_id,
                "chunks": [chunk],
                "source_file": "/history/session.jsonl",
                "file_position": 9,
            }
        )
    )
    assert upload.status == 200
    body = json.loads(upload.body.decode())
    assert body["status"] == "ok"
    assert body["chunks_stored"] == 1
    assert body["reindex_required"] is False
    assert body["reindex_requested_at"] == requested_at
    assert len(stored) == 1

    positions = await server.handle_api_get_positions(
        SimpleNamespace(
            match_info={"machine_id": machine_id},
            query={"client_name": machine_id},
            headers={},
        )
    )
    assert positions.status == 200
    positions_body = json.loads(positions.body.decode())
    assert positions_body.get("error") is None
    assert positions_body["positions"] == {"/history/session.jsonl": 9}
    assert positions_body["reindex_required"] is False


def test_reindex_required_response_still_requires_its_request_identity():
    """The guard that matters is preserved: a demanded reindex must be identified."""
    with pytest.raises(ValueError, match="must carry reindex_requested_at"):
        ChunkUploadResponse(
            status="ok",
            chunks_received=0,
            chunks_embedded=0,
            chunks_stored=0,
            reindex_required=True,
            reindex_requested_at=None,
        )
    with pytest.raises(ValueError, match="must carry reindex_requested_at"):
        GetPositionsResponse(
            machine_id="machine-a",
            positions={},
            reindex_required=True,
            reindex_requested_at=None,
        )


def test_model_level_validation_error_yields_a_label_instead_of_crashing():
    """A model-level rejection reports an EMPTY location.

    The handlers formatted a field label by indexing that location directly,
    which raised IndexError inside their own `except ValidationError` block.
    Python does not route that to sibling handlers, so the failure escaped the
    handler entirely rather than becoming a bounded 400 response.
    """
    model_level = ValidationError.from_exception_data(
        "ChunkUploadResponse",
        [{"type": "value_error", "loc": (), "input": {}, "ctx": {"error": "model level"}}],
    )
    field_level = ValidationError.from_exception_data(
        "ChunkUploadResponse",
        [{"type": "value_error", "loc": ("chunks",), "input": {}, "ctx": {"error": "field"}}],
    )

    # The precise shape that used to detonate.
    assert model_level.errors()[0]["loc"] == ()
    with pytest.raises(IndexError):
        model_level.errors()[0].get("loc", ["request"])[-1]

    assert status_server_module._validation_error_field(model_level) == "request"
    assert status_server_module._validation_error_field(field_level) == "chunks"


async def test_invalid_request_still_reports_its_rejected_field(
    monkeypatch: pytest.MonkeyPatch,
):
    """The bounded 400 path keeps naming the field it rejected."""
    server = await configure_status_server(monkeypatch)

    response = await server.handle_api_chunks(
        fake_request(
            {
                "machine_id": "machine-a",
                "client_name": "machine-a",
                "chunks": [],
                "source_file": "/history/session.jsonl",
                "file_position": -1,
            }
        )
    )

    assert response.status == 400
    assert json.loads(response.body.decode())["error"] == "Invalid request: file_position"


def test_undecodable_source_is_counted_not_silently_reported_as_empty(tmp_path: Path):
    """A physical-line count must not depend on the source decoding as UTF-8.

    Returning 0 here is a silent lie to a cursor authority: the caller concludes
    there is nothing past the current position and skips the source with no
    diagnostic and no failed-file marking at all.
    """
    source = tmp_path / "session.jsonl"
    source.write_bytes("first caf\xe9\nsecond\n".encode("latin-1"))

    assert watcher_module._count_file_lines(source) == 2


@pytest.mark.parametrize("terminator,expected", [(b"\n", 3), (b"\r\n", 3), (b"\r", 1)])
def test_count_digest_and_parser_share_one_definition_of_a_line(
    terminator: bytes,
    expected: int,
    tmp_path: Path,
):
    """The count, the digest and the parse must number lines identically.

    The count and digest bound a consumed interval that the parse then fills. If
    the parse splits on a separator the count does not recognise, it numbers
    lines beyond the counted bound and the interval stops describing what was
    read. A bare carriage return is the case that used to diverge.
    """
    record = json.dumps(
        {
            "type": "user",
            "sessionId": "s",
            "uuid": "u",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"role": "user", "content": "hello"},
        }
    ).encode()
    source = tmp_path / "session.jsonl"
    source.write_bytes(terminator.join([record] * 3) + terminator)

    counted = watcher_module._count_file_lines(source)
    parsed = [line for _, line in parse_jsonl_file(source)]

    assert counted == expected
    # The parse must never number a line beyond the counted bound.
    assert not parsed or max(parsed) <= counted


def test_prefix_digest_of_a_partial_bound_is_not_the_whole_file(tmp_path: Path):
    """Digesting a prefix must actually stop at the bound it was given.

    Asserting only that the full-count digest equals the whole file is true for
    every byte string and proves nothing about the bound.
    """
    # CRLF plus a byte that is not valid UTF-8: a text-mode implementation would
    # translate the line endings and could not decode the source at all, so this
    # fixture distinguishes the byte-exact digest from a decoded one. An ASCII-LF
    # fixture cannot - both implementations agree on it.
    source = tmp_path / "session.jsonl"
    source.write_bytes(b"one\r\ntwo\xff\r\nthree\r\n")

    whole = hashlib.sha256(source.read_bytes()).hexdigest()

    assert watcher_module._prefix_digest(source, 3) == whole
    assert watcher_module._prefix_digest(source, 2) != whole
    assert (
        watcher_module._prefix_digest(source, 2)
        == hashlib.sha256(b"one\r\ntwo\xff\r\n").hexdigest()
    )


async def test_committed_cursor_never_regresses_on_an_older_upload(
    monkeypatch: pytest.MonkeyPatch,
):
    """A late or replayed record must not walk the server cursor backwards."""
    server = await configure_status_server(monkeypatch)
    monkeypatch.setattr(status_server_module.settings, "storage_backend", "spanner")
    monkeypatch.setattr(status_server_module.settings, "spanner_embedding_mode", "spanner")

    source_file = "/history/session.jsonl"

    async def upload_at(position: int):
        return await server.handle_api_chunks(
            fake_request(
                {
                    "machine_id": "machine-a",
                    "client_name": "machine-a",
                    "chunks": [],
                    "source_file": source_file,
                    "file_position": position,
                }
            )
        )

    assert (await upload_at(9)).status == 200
    assert status_server_module._machine_positions["machine-a"][source_file] == 9

    # An older position arriving afterwards must not lower the committed cursor.
    assert (await upload_at(4)).status == 200
    assert status_server_module._machine_positions["machine-a"][source_file] == 9


async def test_committed_cursor_never_regresses_on_a_chunk_bearing_upload(
    monkeypatch: pytest.MonkeyPatch,
):
    """The chunk-bearing commit path has its own cursor write and its own guard.

    Testing only the empty-chunk path leaves half the monotonic guard uncovered.
    """
    server = await configure_status_server(monkeypatch)
    monkeypatch.setattr(status_server_module.settings, "storage_backend", "spanner")
    monkeypatch.setattr(status_server_module.settings, "spanner_embedding_mode", "spanner")

    class FakeStore:
        async def add_chunks_async(self, chunks):
            return None

    monkeypatch.setattr(store_module, "store", FakeStore())
    source_file = "/history/session.jsonl"

    def chunk_at(line: int) -> dict:
        return {
            "id": f"chunk-{line}",
            "content": "content",
            "chunk_type": "turn",
            "session_id": "session",
            "project_path": "/project",
            "project_name": "project",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "source_file": source_file,
            "source_line": line,
        }

    async def upload_at(position: int):
        return await server.handle_api_chunks(
            fake_request(
                {
                    "machine_id": "machine-b",
                    "client_name": "machine-b",
                    "chunks": [chunk_at(position)],
                    "source_file": source_file,
                    "file_position": position,
                }
            )
        )

    assert (await upload_at(12)).status == 200
    assert status_server_module._machine_positions["machine-b"][source_file] == 12

    assert (await upload_at(5)).status == 200
    assert status_server_module._machine_positions["machine-b"][source_file] == 12


@pytest.mark.parametrize("value", [123, ["a"], {"a": 1}, True, None, "ordinary"])
def test_client_name_sanitizer_never_raises_on_request_supplied_types(value):
    """A sanitizer that raises turns a rejectable request into a server error.

    It also splits the identity write it sits inside: the writes above it have
    already landed while the writes below it never run.
    """
    result = client_registry_module._safe_client_name(value)

    assert isinstance(result, str)
    assert "\n" not in result and "\x00" not in result


def test_unusable_content_length_is_reported_as_unknown_not_raised():
    """The size is read for diagnostics only, before the handler's try block."""

    class ExplodingLength:
        @property
        def content_length(self):
            raise ValueError("malformed Content-Length")

    class MissingLength:
        pass

    assert status_server_module._safe_content_length(ExplodingLength()) is None
    assert status_server_module._safe_content_length(MissingLength()) is None
    assert status_server_module._safe_content_length(SimpleNamespace(content_length=7)) == 7


def test_reindex_ack_request_rejects_unbounded_status_or_naive_timestamp():
    with pytest.raises(ValueError):
        ReindexAckRequest(
            machine_id="machine-a",
            reindex_requested_at="2026-08-04T12:00:00+00:00",
            status="attacker-controlled",
        )
    with pytest.raises(ValueError, match="timezone"):
        ReindexAckRequest(
            machine_id="machine-a",
            reindex_requested_at="2026-08-04T12:00:00",
            status="queued",
        )


def test_payload_cleanup_failure_log_redacts_path_and_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    secret = "TOP_SECRET_PATH_AND_PROVIDER_TEXT"
    manager = ClientStateManager(tmp_path / "client-state.json")

    def reject_delete(*args, **kwargs):
        raise OSError(secret)

    monkeypatch.setattr(durable_io, "delete_file", reject_delete)
    with caplog.at_level("WARNING"):
        assert manager._delete_payload("cleanup-record", "0" * 64) is False

    assert secret not in caplog.text
    assert "record=cleanup-record" in caplog.text
    assert "reason=cleanup_failed" in caplog.text
    assert "error_type=OSError" in caplog.text


async def test_authenticated_position_sync_route_cannot_advance_cursor(
    monkeypatch: pytest.MonkeyPatch,
):
    server = await configure_status_server(monkeypatch)
    status_server_module._clear_machine_positions()

    response = await server.handle_api_sync_position(
        fake_request(
            {
                "machine_id": "machine-a",
                "file_path": "/history/session.jsonl",
                "position": 99,
            }
        )
    )

    body = json.loads(response.text)
    assert response.status == 409
    assert body["error"] == "cursor_sync_forbidden"
    assert status_server_module._machine_positions == {}


async def test_real_heartbeat_round_trip_preserves_durable_health_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    server = await configure_status_server(monkeypatch)
    registry = ClientRegistry(tmp_path / "client-registry.json")
    monkeypatch.setattr(status_server_module, "get_client_registry", lambda: registry)
    heartbeat = {
        "machine_id": "machine-a",
        "client_name": "client-a",
        "status": "degraded",
        "queue": {
            "pending_uploads": 3,
            "pending_uploads_oldest_age_sec": 41,
            "queue_size": 2,
            "queue_max_size": 1000,
            "active_claims": 1,
            "blocked_records": 2,
            "retry_total": 7,
        },
        "reindex": {
            "required_at": "2026-08-04T12:00:00+00:00",
            "ack_at": "2026-08-04T12:01:00+00:00",
            "status": "queued",
            "generation": 4,
        },
        "errors": {
            "count": 3,
            "catchup_failure_count": 1,
            "catchup_failure_reasons": {"history_truncated": 1},
            "catchup_failure_oldest_age_sec": 22,
        },
    }

    heartbeat_response = await server.handle_api_heartbeat(fake_request(heartbeat))
    status_response = await server.handle_api_auth_state(fake_request({}))
    client = json.loads(status_response.text)["clients"][0]

    assert heartbeat_response.status == 200
    assert client["status_label"] == "Degraded"
    assert client["queue"] == heartbeat["queue"]
    assert client["reindex"] == heartbeat["reindex"]
    assert client["errors"] == heartbeat["errors"]

    invalid = await server.handle_api_heartbeat(
        fake_request({**heartbeat, "status": "attacker-defined-healthy"})
    )
    assert invalid.status == 400


async def test_client_startup_and_reindex_enumeration_redact_source_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    secret_root = tmp_path / "TOP_SECRET_ROOT"
    secret_root.mkdir()
    secret_file = secret_root / "TOP_SECRET_FILENAME.jsonl"
    secret_file.write_text("{}\n")
    secret_error = "TOP_SECRET_EXCEPTION_TEXT"
    manager = ClientStateManager(tmp_path / "client-state.json")
    watcher = HistoryWatcher(projects_path=secret_root)
    diagnostics: list[tuple[str, str, dict]] = []

    async def no_pending(api, state):
        return 0

    async def no_sync(api, state):
        return None

    async def fail_index(path, api, state):
        raise RuntimeError(secret_error)

    monkeypatch.setattr(settings, "server_url", "https://history.example.test")
    monkeypatch.setattr(api_client_module, "get_api_client", lambda: FakeAPI(secret_file))
    monkeypatch.setattr(client_state_module, "get_client_state_manager", lambda: manager)
    monkeypatch.setattr(watcher, "_process_pending_uploads", no_pending)
    monkeypatch.setattr(watcher, "_sync_positions_with_server", no_sync)
    monkeypatch.setattr(watcher, "_index_file_client_mode", fail_index)
    monkeypatch.setattr(
        watcher_module,
        "record_error",
        lambda kind, message, details: diagnostics.append((kind, message, details)),
    )

    with caplog.at_level("DEBUG"):
        await watcher.startup_sync()

    rendered = json.dumps(diagnostics)
    assert str(secret_root) not in caplog.text
    assert secret_file.name not in caplog.text
    assert secret_error not in caplog.text
    assert secret_file.name not in rendered
    assert secret_error not in rendered
    assert "error_type=RuntimeError" in caplog.text
    assert diagnostics[0][2]["source_hash"]

    missing = HistoryWatcher(projects_path=tmp_path / "TOP_SECRET_MISSING_ROOT")
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert await missing.queue_all_files_for_indexing() == 0
    assert "TOP_SECRET_MISSING_ROOT" not in caplog.text
    assert "root=" in caplog.text


async def test_reindex_trigger_failure_log_redacts_exception_details(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    secret = "TOP_SECRET_DATABASE_OR_PATH_TEXT"
    server = await configure_status_server(monkeypatch)

    class RejectStore:
        async def clear_all_async(self):
            raise RuntimeError(secret)

    monkeypatch.setattr(store_module, "store", RejectStore())
    with caplog.at_level("ERROR"):
        response = await server.handle_trigger_reindex(fake_request({}))

    body = json.loads(response.text)
    assert response.status == 500
    assert body["error"] == "Reindex failed: RuntimeError"
    assert secret not in caplog.text
    assert "reason=operation_failed" in caplog.text
    assert "phase=clear_store" in caplog.text
    assert "error_type=RuntimeError" in caplog.text


@pytest.mark.parametrize("endpoint", ["status", "search", "purge", "auth_state"])
async def test_endpoint_failure_diagnostics_never_emit_exception_content(
    endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    secret = f"TOP_SECRET_{endpoint}\nFORGED_LOG_LINE"
    server = await configure_status_server(monkeypatch)

    class RejectCollector:
        async def collect_status(self, detail_level):
            raise RuntimeError(secret)

    class RejectStore:
        async def embed_query_text_async(self, query):
            raise RuntimeError(secret)

        async def delete_by_machine_id_async(self, machine_id):
            raise RuntimeError(secret)

    class RejectAuthManager:
        def get_rotation_state(self):
            raise RuntimeError(secret)

    request = fake_request({})
    request.query = {}
    if endpoint == "status":

        async def reject_collector():
            return RejectCollector()

        monkeypatch.setattr(status_server_module, "get_status_collector", reject_collector)
        call = server.handle_status(request)
    elif endpoint == "search":
        monkeypatch.setattr(status_server_module.settings, "storage_backend", "spanner")
        monkeypatch.setattr(status_server_module.settings, "spanner_embedding_mode", "spanner")
        monkeypatch.setattr(store_module, "store", RejectStore())
        call = server.handle_api_search(fake_request({"query": "safe-query"}))
    elif endpoint == "purge":
        monkeypatch.setattr(store_module, "store", RejectStore())
        call = server.handle_api_purge_client(
            fake_request({"machine_id": "machine-a", "reason": secret})
        )
    else:
        monkeypatch.setattr(status_server_module, "get_auth_manager", lambda: RejectAuthManager())
        call = server.handle_api_auth_state(request)

    with caplog.at_level("ERROR"):
        response = await call

    assert response.status >= 500
    assert secret not in caplog.text
    assert "FORGED_LOG_LINE" not in caplog.text
    assert "reason=operation_failed" in caplog.text
    assert "error_type=RuntimeError" in caplog.text


def test_corrupt_client_registry_fails_closed_without_overwrite(tmp_path: Path):
    registry_path = tmp_path / "client-registry.json"
    original = '{"clients": '
    registry_path.write_text(original)

    with pytest.raises(RuntimeError, match="durable client registry"):
        ClientRegistry(registry_path).get_client_status()

    assert registry_path.read_text() == original


def test_client_registry_save_failure_rolls_back_memory_and_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    secret = "TOP_SECRET_REGISTRY_PATH_OR_PROVIDER"
    registry_path = tmp_path / "client-registry.json"
    registry = ClientRegistry(registry_path)
    registry.register_client("machine-a", client_name="client-a")
    inject_durable_replace_failure(monkeypatch, registry_path, secret)
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="could not be saved"):
        registry.register_client("machine-b", client_name="client-b")

    assert secret not in caplog.text
    assert "reason=save_failed" in caplog.text
    assert [client["machine_id"] for client in registry.get_client_status()["clients"]] == [
        "machine-a"
    ]
    restarted = ClientRegistry(registry_path).get_client_status()
    assert [client["machine_id"] for client in restarted["clients"]] == ["machine-a"]


async def test_heartbeat_endpoint_does_not_ack_failed_registry_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    server = await configure_status_server(monkeypatch)
    registry_path = tmp_path / "client-registry.json"
    registry = ClientRegistry(registry_path)
    registry.register_client("machine-a", client_name="client-a")
    monkeypatch.setattr(status_server_module, "get_client_registry", lambda: registry)
    inject_durable_replace_failure(
        monkeypatch,
        registry_path,
        "TOP_SECRET_REGISTRY_PROVIDER",
    )
    response = await server.handle_api_heartbeat(
        fake_request(
            {
                "machine_id": "machine-b",
                "client_name": "client-b",
                "status": "ok",
            }
        )
    )

    body = json.loads(response.text)
    assert response.status == 500
    assert body["status"] == "error"
    assert body["error"] == "Heartbeat failed: RuntimeError"
    restarted = ClientRegistry(registry_path).get_client_status()
    assert [client["machine_id"] for client in restarted["clients"]] == ["machine-a"]


def test_legacy_cursor_corruption_and_save_failure_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    corrupt_path = tmp_path / "corrupt-watcher.json"
    corrupt_path.write_text('{"file_positions": ')
    with pytest.raises(RuntimeError, match="durable watcher state"):
        FilePositionState(corrupt_path)

    state_path = tmp_path / "watcher.json"
    state = FilePositionState(state_path)
    state.set_position("/history/session.jsonl", 3)
    state.save()
    secret = "TOP_SECRET_CURSOR_PATH_OR_PROVIDER"
    inject_durable_replace_failure(monkeypatch, state_path, secret)
    state.set_position("/history/session.jsonl", 9)
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="could not be saved"):
        state.save()

    assert secret not in caplog.text
    assert state.get_position("/history/session.jsonl") == 3
    assert FilePositionState(state_path).get_position("/history/session.jsonl") == 3


async def test_reindex_handler_does_not_report_failed_cursor_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    server = await configure_status_server(monkeypatch)
    watcher = HistoryWatcher(
        projects_path=tmp_path,
        state_path=tmp_path / "watcher-state.json",
    )
    watcher.state.set_position("/history/session.jsonl", 3)
    watcher.state.save()

    class ClearStore:
        async def clear_all_async(self):
            return 0

    def reject_save():
        raise RuntimeError("durable watcher state could not be saved")

    monkeypatch.setattr(store_module, "store", ClearStore())
    monkeypatch.setattr(watcher_module, "get_all_watchers", lambda: [watcher])
    monkeypatch.setattr(watcher.state, "save", reject_save)

    response = await server.handle_trigger_reindex(fake_request({}))

    assert response.status == 500
    assert json.loads(response.text)["status"] == "error"


async def test_status_payload_hashes_all_filesystem_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    secret_root = tmp_path / "TOP_SECRET_ROOT"
    watcher = HistoryWatcher(projects_path=secret_root)
    secret_file = str(secret_root / "TOP_SECRET_HISTORY.jsonl")
    watcher._failed_files.add(secret_file)
    monkeypatch.setattr(status_module, "get_all_watchers", lambda: [watcher])
    monkeypatch.setattr(settings, "projects_path", secret_root)
    collector = status_module.StatusCollector()

    payload = {
        "indexing": await collector._get_indexing_status(),
        "watcher": collector._get_watcher_stats(),
        "configuration": collector._get_configuration(),
    }
    rendered = json.dumps(payload)

    # Compared as a JSON fragment: a raw Windows path never appears verbatim in
    # serialized JSON, so searching for it would pass whether or not it leaked.
    assert json_fragment(str(secret_root)) not in rendered
    assert "TOP_SECRET_HISTORY" not in rendered
    assert '"watch_path":' not in rendered
    assert '"projects_path":' not in rendered
    assert "watch_path_hash" in rendered
    assert "projects_path_hash" in rendered


def test_status_collector_log_redacts_exception_and_newline(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    secret = "TOP_SECRET_STATUS_EXCEPTION\nFORGED_LOG_LINE"

    def reject_watchers():
        raise RuntimeError(secret)

    monkeypatch.setattr(status_module, "get_all_watchers", reject_watchers)
    with caplog.at_level("ERROR"):
        result = status_module.StatusCollector()._get_watcher_stats()

    assert "error" in result
    assert secret not in caplog.text
    assert "FORGED_LOG_LINE" not in caplog.text
    assert "reason=operation_failed" in caplog.text
    assert "error_type=RuntimeError" in caplog.text


@pytest.mark.parametrize("endpoint", ["positions", "search", "reindex", "purge"])
async def test_validation_errors_do_not_reflect_attacker_fields(
    endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
):
    server = await configure_status_server(monkeypatch)
    secret = "TOP_SECRET_INPUT\nFORGED_RESPONSE_LINE"
    if endpoint == "positions":
        response = await server.handle_api_sync_position(
            fake_request({"machine_id": secret, "file_path": secret, "position": 1})
        )
    elif endpoint == "search":
        response = await server.handle_api_search(fake_request({"query": "x" * 513 + secret}))
    elif endpoint == "reindex":
        response = await server.handle_api_reindex_ack(
            fake_request(
                {
                    "machine_id": secret,
                    "reindex_requested_at": "not-a-timestamp",
                    "status": "forged",
                }
            )
        )
    else:
        response = await server.handle_api_purge_client(
            fake_request({"machine_id": secret, "reason": secret})
        )

    rendered = response.text
    assert response.status == 400
    assert secret not in rendered
    assert "FORGED_RESPONSE_LINE" not in rendered


async def test_actual_drain_rejects_same_length_api_identity_substitution(
    tmp_path: Path,
    history_file: Path,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    chunks = [{"id": "bound", "content": "payload", "source_line": 1}]
    await manager.replace_pending_upload(
        "/history/a.jsonl",
        chunks,
        7,
        record_id="full-request-binding",
        machine_id="machine-A",
        client_name="client-X",
    )
    api = FakeAPI(history_file)
    api.machine_id = "machine-B"
    api.client_name = "client-X"

    assert await make_watcher(history_file, tmp_path)._process_pending_uploads(api, manager) == 0
    assert api.uploads == []
    state = await manager.get_state()
    assert [record.record_id for record in state.pending_uploads] == ["full-request-binding"]
    assert state.server_positions.get("/history/a.jsonl", 0) == 0


async def test_preoccupied_deterministic_payload_temp_hardlink_cannot_escape_root(
    tmp_path: Path,
):
    manager = ClientStateManager(tmp_path / "state.json")
    manager._outbox_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside-sentinel")
    deterministic_temp = manager._outbox_dir / "hardlink-record.tmp"
    os.link(outside, deterministic_temp)

    await manager.replace_pending_upload(
        "/history/a.jsonl",
        [{"id": "payload", "source_line": 1}],
        1,
        record_id="hardlink-record",
        machine_id="machine-A",
        client_name="client-X",
    )

    assert outside.read_bytes() == b"outside-sentinel"
    assert not os.path.samefile(manager._payload_path("hardlink-record"), outside)


@pytest.mark.parametrize(
    "api_machine_id,api_client_name",
    [("machine-B", "client-X"), ("machine-A", "client-Y")],
)
async def test_actual_drain_rejects_same_length_identity_field_substitution(
    tmp_path: Path,
    history_file: Path,
    api_machine_id: str,
    api_client_name: str,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        "/history/a.jsonl",
        [{"id": "bound", "content": "payload-A", "source_line": 1}],
        7,
        record_id="identity-substitution",
        machine_id="machine-A",
        client_name="client-X",
    )
    api = FakeAPI(history_file)
    api.machine_id = api_machine_id
    api.client_name = api_client_name
    pending = (await manager.get_state()).pending_uploads[0]
    assert (
        chunk_upload_request_bytes(
            [{"id": "bound", "content": "payload-A", "source_line": 1}],
            machine_id=api_machine_id,
            client_name=api_client_name,
            source_file="/history/a.jsonl",
            file_position=7,
        )
        == pending.request_bytes
    )

    assert await make_watcher(history_file, tmp_path)._process_pending_uploads(api, manager) == 0
    assert api.uploads == []
    assert (await manager.get_state()).pending_uploads[0].retry_count == 1


@pytest.mark.parametrize("substitution", ["source_file", "file_position", "chunks"])
async def test_restart_rejects_same_length_semantic_request_substitution(
    tmp_path: Path,
    substitution: str,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        "/history/a.jsonl",
        [{"id": "bound", "content": "payload-A", "source_line": 1}],
        7,
        record_id="semantic-substitution",
        machine_id="machine-A",
        client_name="client-X",
    )
    state_data = json.loads(manager.state_path.read_text())
    record = state_data["pending_uploads"][0]
    effective_chunks = [{"id": "bound", "content": "payload-A", "source_line": 1}]
    if substitution == "source_file":
        record["file_path"] = "/history/b.jsonl"
    elif substitution == "file_position":
        record["file_position"] = 8
    else:
        changed = [{"id": "bound", "content": "payload-B", "source_line": 1}]
        changed_payload = manager._payload_bytes(changed)
        manager._payload_path(record["record_id"]).write_bytes(changed_payload)
        record["payload_sha256"] = hashlib.sha256(changed_payload).hexdigest()
        record["payload_bytes"] = len(changed_payload)
        effective_chunks = changed
    assert (
        chunk_upload_request_bytes(
            effective_chunks,
            machine_id=record["request_machine_id"],
            client_name=record["request_client_name"],
            source_file=record["file_path"],
            file_position=record["file_position"],
        )
        == record["request_bytes"]
    )
    manager.state_path.write_text(json.dumps(state_data))

    with pytest.raises(RuntimeError, match="durable client outbox state"):
        await ClientStateManager(manager.state_path).get_state()


async def test_restart_preserves_full_request_binding_and_matching_drain(
    tmp_path: Path,
    history_file: Path,
):
    state_path = tmp_path / "client-state.json"
    manager = ClientStateManager(state_path)
    await manager.replace_pending_upload(
        "/history/a.jsonl",
        [{"id": "bound", "content": "payload", "source_line": 1}],
        7,
        record_id="restart-binding",
        machine_id="machine-A",
        client_name="client-X",
    )
    restarted = ClientStateManager(state_path)
    pending = (await restarted.get_state()).pending_uploads[0]
    api = FakeAPI(history_file)
    api.machine_id = "machine-A"
    api.client_name = "client-X"

    assert pending.request_sha256 is not None
    assert pending.request_machine_id == "machine-A"
    assert pending.request_client_name == "client-X"
    assert await make_watcher(history_file, tmp_path)._process_pending_uploads(api, restarted) == 1
    assert len(api.uploads) == 1
    assert (await restarted.get_state()).server_positions["/history/a.jsonl"] == 7


async def test_current_schema_missing_request_digest_fails_closed(tmp_path: Path):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        "/history/a.jsonl",
        [{"id": "bound", "source_line": 1}],
        1,
        record_id="missing-request-digest",
        machine_id="machine-A",
        client_name="client-X",
    )
    state_data = json.loads(manager.state_path.read_text())
    state_data["pending_uploads"][0].pop("request_sha256")
    manager.state_path.write_text(json.dumps(state_data))

    with pytest.raises(RuntimeError, match="durable client outbox state"):
        await ClientStateManager(manager.state_path).get_state()


async def test_schema_v1_inline_upload_gains_full_request_binding(tmp_path: Path):
    state_path = tmp_path / "client-state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pending_uploads": [
                    {
                        "record_id": "legacy-inline",
                        "file_path": "/history/a.jsonl",
                        "chunks": [{"id": "legacy", "source_line": 1}],
                        "file_position": 1,
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                ],
            }
        )
    )

    state = await ClientStateManager(state_path).get_state()
    pending = state.pending_uploads[0]

    assert pending.payload_externalized is True
    assert pending.request_sha256 is not None
    assert pending.request_machine_id == settings.machine_id
    assert pending.request_client_name == (settings.client_name or settings.machine_id)


async def test_final_payload_hardlink_is_rejected_before_read(tmp_path: Path):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        "/history/a.jsonl",
        [{"id": "bound", "source_line": 1}],
        1,
        record_id="hardlinked-final",
    )
    pending = (await manager.get_state()).pending_uploads[0]
    payload_path = manager._payload_path(pending.record_id)
    outside_link = tmp_path / "outside-link.json"
    os.link(payload_path, outside_link)

    with pytest.raises(durable_io.UnsafeDurablePathError, match="multiple hard links"):
        await manager.load_pending_chunks(pending)


async def test_final_payload_symlink_is_rejected_before_read(tmp_path: Path):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        "/history/a.jsonl",
        [{"id": "bound", "source_line": 1}],
        1,
        record_id="symlinked-final",
    )
    pending = (await manager.get_state()).pending_uploads[0]
    payload_path = manager._payload_path(pending.record_id)
    outside = tmp_path / "outside-payload.json"
    outside.write_bytes(payload_path.read_bytes())
    payload_path.unlink()
    try:
        payload_path.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"file symlinks unavailable: {type(error).__name__}")

    with pytest.raises(durable_io.UnsafeDurablePathError, match="link or reparse"):
        await manager.load_pending_chunks(pending)


async def test_api_client_transmits_the_canonical_bound_request_body():
    chunks = [{"id": "wire", "content": "é", "source_line": 1}]
    expected = chunk_upload_request_body(
        chunks,
        machine_id="machine-A",
        client_name="client-X",
        source_file="/history/a.jsonl",
        file_position=7,
    )
    captured: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.content)
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "chunks_received": 1,
                "chunks_embedded": 1,
                "chunks_stored": 1,
                "reindex_required": False,
            },
        )

    api = api_client_module.APIClient(
        server_url="https://history.example.test",
        machine_id="machine-A",
        client_name="client-X",
        retry_count=1,
    )
    api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        response = await api.upload_chunks(
            chunks=chunks,
            source_file="/history/a.jsonl",
            file_position=7,
        )
    finally:
        await api.close()

    assert response.status == "ok"
    assert captured == [expected]
    assert hashlib.sha256(captured[0]).hexdigest() == hashlib.sha256(expected).hexdigest()


async def test_all_state_owners_reject_multilink_durable_files(tmp_path: Path):
    client_path = tmp_path / "client-state.json"
    client = ClientStateManager(client_path)
    await client.update_local_position("/history/a.jsonl", 1)
    os.link(client_path, tmp_path / "client-state-outside-link.json")
    with pytest.raises(RuntimeError, match="durable client outbox state"):
        await ClientStateManager(client_path).get_state()

    registry_path = tmp_path / "registry.json"
    registry = ClientRegistry(registry_path)
    registry.register_client("machine-a")
    os.link(registry_path, tmp_path / "registry-outside-link.json")
    with pytest.raises(RuntimeError, match="durable client registry"):
        ClientRegistry(registry_path).get_client_status()

    cursor_path = tmp_path / "cursor.json"
    cursor = FilePositionState(cursor_path)
    cursor.set_position("/history/a.jsonl", 1)
    cursor.save()
    os.link(cursor_path, tmp_path / "cursor-outside-link.json")
    with pytest.raises(RuntimeError, match="durable watcher state"):
        FilePositionState(cursor_path)


async def test_outbox_directory_symlink_is_rejected_without_outside_write(tmp_path: Path):
    manager = ClientStateManager(tmp_path / "client-state.json")
    outside = tmp_path / "outside-directory"
    outside.mkdir()
    try:
        manager._outbox_dir.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {type(error).__name__}")

    with pytest.raises(durable_io.UnsafeDurablePathError, match="link or reparse"):
        await manager.replace_pending_upload(
            "/history/a.jsonl",
            [{"id": "must-not-write", "source_line": 1}],
            1,
            record_id="linked-directory",
        )

    assert list(outside.iterdir()) == []


async def test_existing_payload_is_restored_when_index_replacement_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        "/history/a.jsonl",
        [{"id": "old", "content": "payload-A", "source_line": 1}],
        1,
        record_id="replace-rollback",
    )
    payload_path = manager._payload_path("replace-rollback")
    old_payload = payload_path.read_bytes()
    disk_state = json.loads(manager.state_path.read_text())
    disk_state["pending_uploads"][0]["retry_count"] = 1
    manager.state_path.write_text(json.dumps(disk_state))

    def fail_snapshot(*args):
        raise OSError("injected index failure")

    monkeypatch.setattr(manager, "_write_state_snapshot", fail_snapshot)
    with pytest.raises(OSError, match="index failure"):
        await manager.replace_pending_upload(
            "/history/a.jsonl",
            [{"id": "new", "content": "payload-B", "source_line": 1}],
            1,
            record_id="replace-rollback",
        )

    assert payload_path.read_bytes() == old_payload
    restarted = await ClientStateManager(manager.state_path).get_state()
    assert restarted.pending_uploads[0].retry_count == 1
    assert (
        await ClientStateManager(manager.state_path).load_pending_chunks(
            restarted.pending_uploads[0]
        )
    )[0]["id"] == "old"


async def test_committed_unconfirmed_index_write_reconciles_on_idempotent_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.save()
    original_write = manager._write_state_snapshot

    def commit_then_lose_confirmation(state=None):
        original_write(state)
        raise durable_io.DurableCommitUncertainError(
            "injected lost index confirmation",
            committed=True,
        )

    monkeypatch.setattr(manager, "_write_state_snapshot", commit_then_lose_confirmation)
    with pytest.raises(durable_io.DurableCommitUncertainError):
        await manager.replace_pending_upload(
            "/history/a.jsonl",
            [{"id": "bound", "source_line": 1}],
            1,
            record_id="uncertain-index",
        )

    assert [record.record_id for record in (await manager.get_state()).pending_uploads] == [
        "uncertain-index"
    ]
    monkeypatch.setattr(manager, "_write_state_snapshot", original_write)
    await manager.replace_pending_upload(
        "/history/a.jsonl",
        [{"id": "bound", "source_line": 1}],
        1,
        record_id="uncertain-index",
    )
    restarted = await ClientStateManager(manager.state_path).get_state()
    assert [record.record_id for record in restarted.pending_uploads] == ["uncertain-index"]


async def test_ack_cleanup_failure_remains_durable_and_retries_without_resend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        "/history/a.jsonl",
        [{"id": "bound", "source_line": 1}],
        1,
        record_id="cleanup-retry",
    )
    expected_digest = (await manager.get_pending_uploads())[0].payload_sha256
    assert expected_digest is not None
    original_delete = durable_io.delete_file

    def reject_delete(*args, **kwargs):
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(durable_io, "delete_file", reject_delete)
    with pytest.raises(OSError, match="cleanup remains pending"):
        await manager.complete_pending_upload("cleanup-retry")

    state = await manager.get_state()
    assert state.pending_uploads == []
    assert state.payload_cleanup_pending == {"cleanup-retry": expected_digest}

    monkeypatch.setattr(durable_io, "delete_file", original_delete)
    completed, _ = await manager.complete_pending_upload("cleanup-retry")
    assert completed is True
    assert (await manager.get_state()).payload_cleanup_pending == {}
    assert not manager._payload_path("cleanup-retry").exists()


async def test_ack_cleanup_substitution_never_removes_unrelated_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.replace_pending_upload(
        "/history/a.jsonl",
        [{"id": "bound", "source_line": 1}],
        1,
        record_id="cleanup-substitution",
    )
    payload_path = manager._payload_path("cleanup-substitution")
    expected_payload = payload_path.read_bytes()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside-sentinel")
    original_delete = durable_io.delete_file
    substituted = False

    def substitute_then_delete(path, **kwargs):
        nonlocal substituted
        if not substituted:
            payload_path.unlink()
            os.link(outside, payload_path)
            substituted = True
        return original_delete(path, **kwargs)

    monkeypatch.setattr(durable_io, "delete_file", substitute_then_delete)
    with pytest.raises(OSError, match="cleanup remains pending"):
        await manager.complete_pending_upload("cleanup-substitution")

    assert outside.read_bytes() == b"outside-sentinel"
    assert payload_path.read_bytes() == b"outside-sentinel"
    assert "cleanup-substitution" in (await manager.get_state()).payload_cleanup_pending

    payload_path.unlink()
    durable_io.atomic_write_bytes(
        payload_path,
        expected_payload,
        durable_root=manager._outbox_dir,
    )
    monkeypatch.setattr(durable_io, "delete_file", original_delete)
    completed, _ = await manager.complete_pending_upload("cleanup-substitution")
    assert completed is True
    assert outside.read_bytes() == b"outside-sentinel"


async def test_creation_rollback_substitution_never_removes_unrelated_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = ClientStateManager(tmp_path / "client-state.json")
    await manager.save()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside-sentinel")
    payload_path = manager._payload_path("rollback-substitution")

    def substitute_then_fail(*args):
        payload_path.unlink()
        os.link(outside, payload_path)
        raise OSError("injected index failure")

    monkeypatch.setattr(manager, "_write_state_snapshot", substitute_then_fail)
    with pytest.raises(
        durable_io.UnsafeDurablePathError,
        match="content identity mismatch|multiple hard links",
    ):
        await manager.replace_pending_upload(
            "/history/a.jsonl",
            [{"id": "new", "source_line": 1}],
            1,
            record_id="rollback-substitution",
        )

    assert (await manager.get_state()).pending_uploads == []
    assert outside.read_bytes() == b"outside-sentinel"
    assert payload_path.read_bytes() == b"outside-sentinel"


# ============================================================
# Watch-root object identity: a configured root is a mutable
# name, so containment alone can never authorize its content.
# ============================================================


def make_pinned_watcher(root: Path, tmp_path: Path) -> HistoryWatcher:
    """Build a watcher whose durable state lives outside the watch root."""
    return HistoryWatcher(
        projects_path=root,
        state_path=tmp_path / "watcher-state.json",
        chunker=chunk_session_file,
    )


def test_provenance_is_only_offered_to_chunkers_that_declare_it():
    """A callable that merely absorbs **kwargs must not be treated as capable.

    It would accept the provenance and ignore it, deriving project and session
    identity from the snapshot location instead - the exact failure the
    parameter exists to prevent.
    """

    def declares(path, start_line=0, *, source_path=None):
        return iter(())

    def absorbs(path, start_line=0, **kwargs):
        return iter(())

    def neither(path, start_line=0):
        return iter(())

    assert watcher_module._accepts_source_path(declares) is True
    assert watcher_module._accepts_source_path(absorbs) is False
    assert watcher_module._accepts_source_path(neither) is False

    # Every real chunker must declare it, or provenance silently degrades.
    for source_watcher in watcher_module.get_all_watchers():
        assert source_watcher._chunker_reports_provenance is True


async def test_stable_root_still_indexes_and_catches_up(tmp_path: Path):
    """The pinned root must not obstruct ordinary work on an unchanged root."""
    root, history = build_watch_tree(tmp_path)
    watcher = make_pinned_watcher(root, tmp_path)
    manager = ClientStateManager(tmp_path / "client-state.json")
    api = FakeAPI(history)

    assert watcher._verify_root() is True
    assert watcher.is_allowed_history_path(history) is True
    assert history in watcher.discover_files()

    await watcher._index_file_client_mode(history, api, manager)
    await watcher._process_pending_uploads(api, manager)

    state = await manager.get_state()
    assert api.uploads
    assert state.local_positions[str(history)] > 0
    # Provenance stays the original source path, never a snapshot location.
    for upload in api.uploads:
        assert upload["source_file"] == str(history)

    # A server-driven catch-up over the same stable root still queues work.
    await manager.update_local_position(str(history), 0)
    api.server_position = 0
    await watcher._sync_positions_with_server(api, manager)


@pytest.mark.skipif(not link_directories_supported(), reason="directory links unavailable")
async def test_watch_root_replaced_by_link_serves_no_outside_content(tmp_path: Path):
    """A root replaced by a junction or symlink must serve nothing."""
    root, history = build_watch_tree(tmp_path)
    outside = build_outside_tree(tmp_path)
    parked = tmp_path / "parked"
    watcher = make_pinned_watcher(root, tmp_path)
    manager = ClientStateManager(tmp_path / "client-state.json")
    api = FakeAPI(history)

    assert watcher._verify_root() is True

    try:
        substitute_directory_with_link(root, parked, outside)
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"directory links unavailable: {type(error).__name__}")
    try:
        # The replacement really is in place and really does hold other content.
        assert OUTSIDE_SENTINEL in history.read_text()
        assert watcher._classify_history_path(history) == durable_io.ROOT_IS_LINK_OR_REPARSE
        assert watcher.is_allowed_history_path(history) is False
        assert watcher.discover_files() == []

        await watcher._index_file_client_mode(history, api, manager)
        await watcher._index_file(history, None, None)
        await assert_no_work_was_committed(manager, api)
    finally:
        restore_substituted_directory(root, parked)


async def test_watch_root_replaced_by_same_named_directory_serves_no_outside_content(
    tmp_path: Path,
):
    """Only the identity pin catches this: the replacement is an ordinary directory."""
    root, history = build_watch_tree(tmp_path)
    source_of_outside = build_outside_tree(tmp_path, "outside-source")
    parked = tmp_path / "parked"
    watcher = make_pinned_watcher(root, tmp_path)
    manager = ClientStateManager(tmp_path / "client-state.json")
    api = FakeAPI(history)

    assert watcher._verify_root() is True

    os.replace(root, parked)
    substitute = root / "-Users-test-myproject"
    substitute.mkdir(parents=True)
    (substitute / "session.jsonl").write_text(
        (source_of_outside / "-Users-test-myproject" / "session.jsonl").read_text()
    )

    # The substitute is a perfectly ordinary directory, so a reparse-point check
    # cannot see it. Only the bound object identity can.
    assert not os.path.islink(root)
    assert root.is_dir()
    assert OUTSIDE_SENTINEL in history.read_text()

    assert watcher._classify_history_path(history) == durable_io.ROOT_IDENTITY_CHANGED
    assert watcher.is_allowed_history_path(history) is False
    assert watcher.discover_files() == []

    await watcher._index_file_client_mode(history, api, manager)
    await watcher._index_file(history, None, None)
    await assert_no_work_was_committed(manager, api)


@pytest.mark.skipif(not link_directories_supported(), reason="directory links unavailable")
async def test_nested_descendant_replaced_by_link_serves_no_outside_content(tmp_path: Path):
    """The root is untouched here. The substituted object is one level down."""
    root, history = build_watch_tree(tmp_path)
    outside_project = build_outside_tree(tmp_path) / "-Users-test-myproject"
    parked = tmp_path / "parked-project"
    watcher = make_pinned_watcher(root, tmp_path)
    manager = ClientStateManager(tmp_path / "client-state.json")
    api = FakeAPI(history)

    assert watcher._verify_root() is True

    try:
        substitute_directory_with_link(history.parent, parked, outside_project)
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"directory links unavailable: {type(error).__name__}")
    try:
        assert OUTSIDE_SENTINEL in history.read_text()
        # The root itself is still the bound object, so containment succeeds and
        # the refusal has to come from the handle-relative descent instead.
        assert watcher._classify_history_path(history) == "ok"

        await watcher._index_file_client_mode(history, api, manager)
        await watcher._index_file(history, None, None)
        await assert_no_work_was_committed(manager, api)
        assert str(history) in watcher._failed_files
    finally:
        restore_substituted_directory(history.parent, parked)


async def test_missing_root_binds_only_when_a_valid_root_appears(tmp_path: Path):
    """Late binding is allowed. Silent rebinding to a different object is not."""
    root = tmp_path / "history"
    watcher = make_pinned_watcher(root, tmp_path)

    # Nothing to bind yet.
    assert watcher._verify_root() is False
    assert watcher._root.is_bound is False
    assert watcher.discover_files() == []

    project = root / "-Users-test-myproject"
    project.mkdir(parents=True)
    history = project / "session.jsonl"
    shutil.copyfile(Path(__file__).parent / "fixtures" / "sample_session.jsonl", history)

    assert watcher._verify_root() is True
    assert watcher._root.is_bound is True
    bound_identity = watcher._root.identity

    # A replaced root of the same name is refused rather than silently rebound.
    parked = tmp_path / "parked"
    os.replace(root, parked)
    (root / "-Users-test-myproject").mkdir(parents=True)
    assert watcher._verify_root() is False
    assert watcher._root.identity == bound_identity

    # Restoring the original object makes it usable again without rebinding.
    shutil.rmtree(root)
    os.replace(parked, root)
    assert watcher._verify_root() is True
    assert watcher._root.identity == bound_identity


async def test_source_growth_after_snapshot_does_not_alter_the_bounded_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Appending to the live source mid-parse cannot widen or corrupt the interval."""
    root, history = build_watch_tree(tmp_path)
    watcher = make_pinned_watcher(root, tmp_path)
    manager = ClientStateManager(tmp_path / "client-state.json")
    api = FakeAPI(history)

    original_copy = durable_io.copy_held_file
    grown = False

    def grow_after_copy(held, name, destination, *, max_bytes):
        nonlocal grown
        written = original_copy(held, name, destination, max_bytes=max_bytes)
        if not grown:
            with open(history, "a", encoding="utf-8") as appended:
                appended.write(
                    json.dumps(
                        {
                            "type": "user",
                            "sessionId": "grown",
                            "uuid": "grown-uuid",
                            "timestamp": "2026-02-01T00:00:00Z",
                            "message": {"role": "user", "content": OUTSIDE_SENTINEL},
                        }
                    )
                    + "\n"
                )
            grown = True
        return written

    monkeypatch.setattr(durable_io, "copy_held_file", grow_after_copy)
    await watcher._index_file_client_mode(history, api, manager)
    await watcher._process_pending_uploads(api, manager)

    assert grown is True
    state = await manager.get_state()
    persisted = json.dumps(await pending_payloads(manager, state.pending_uploads))
    # Content appended after the snapshot is simply not part of this operation.
    assert OUTSIDE_SENTINEL not in json.dumps(api.uploads)
    assert OUTSIDE_SENTINEL not in persisted


async def test_snapshot_location_never_becomes_stored_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Chunks, outbox records and upload requests all carry the original path."""
    root, history = build_watch_tree(tmp_path)
    watcher = make_pinned_watcher(root, tmp_path)
    manager = ClientStateManager(tmp_path / "client-state.json")
    api = FakeAPI(history)

    observed_snapshots: list[str] = []
    original_copy = durable_io.copy_held_file

    def observe(held, name, destination, *, max_bytes):
        observed_snapshots.append(str(destination))
        return original_copy(held, name, destination, max_bytes=max_bytes)

    uploaded_chunks: list[dict] = []

    class RecordingAPI(FakeAPI):
        async def upload_chunks(self, *, chunks, source_file, file_position):
            uploaded_chunks.extend(chunks)
            return await super().upload_chunks(
                chunks=chunks,
                source_file=source_file,
                file_position=file_position,
            )

    api = RecordingAPI(history)

    monkeypatch.setattr(durable_io, "copy_held_file", observe)
    await watcher._index_file_client_mode(history, api, manager)
    await watcher._process_pending_uploads(api, manager)

    assert observed_snapshots
    # The indexing call drains the outbox itself, so the surviving pending
    # records are empty. Chunk-level provenance has to be inspected on what was
    # actually transmitted, or this assertion inspects nothing at all.
    assert uploaded_chunks
    state = await manager.get_state()
    rendered = json.dumps(
        {
            "uploads": api.uploads,
            "state": state.model_dump(mode="json"),
            "chunks": uploaded_chunks,
        }
    )
    for snapshot_path in observed_snapshots:
        assert json_fragment(snapshot_path) not in rendered
        assert json_fragment(str(Path(snapshot_path).parent)) not in rendered
    assert json_fragment(str(history)) in rendered
    assert all(chunk["source_file"] == str(history) for chunk in uploaded_chunks)


async def test_provenance_is_rewritten_for_a_chunker_that_ignores_it(
    tmp_path: Path,
):
    """A chunker that does not declare source_path must still not leak the snapshot.

    This fallback is not dead code: `chunker` is a public constructor parameter,
    and a double that reports the path it was handed transmits the snapshot
    location as chunk provenance unless the watcher rewrites it.
    """
    root, history = build_watch_tree(tmp_path)
    manager = ClientStateManager(tmp_path / "client-state.json")

    def reports_the_path_it_was_given(path: Path, start_line: int, end_line: int | None = None):
        yield Chunk(
            id="from-double",
            content="content",
            chunk_type="turn",
            session_id="session",
            project_path="/project",
            project_name="project",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            source_file=str(path),
            source_line=1,
            consumed_line=1,
        )

    watcher = HistoryWatcher(
        projects_path=root,
        state_path=tmp_path / "watcher-state.json",
        chunker=reports_the_path_it_was_given,
    )
    assert watcher._chunker_reports_provenance is False

    uploaded_chunks: list[dict] = []

    class RecordingAPI(FakeAPI):
        async def upload_chunks(self, *, chunks, source_file, file_position):
            uploaded_chunks.extend(chunks)
            return await super().upload_chunks(
                chunks=chunks,
                source_file=source_file,
                file_position=file_position,
            )

    api = RecordingAPI(history)
    await watcher._index_file_client_mode(history, api, manager)
    await watcher._process_pending_uploads(api, manager)

    assert uploaded_chunks
    for chunk in uploaded_chunks:
        assert chunk["source_file"] == str(history)
        assert "history-snapshot-" not in chunk["source_file"]


def build_partially_undecodable_source(tmp_path: Path) -> tuple[Path, Path, int]:
    """A watch tree whose history file aborts UTF-8 decoding partway through."""
    root = tmp_path / "history"
    project = root / "-Users-test-myproject"
    project.mkdir(parents=True)
    history = project / "session.jsonl"
    record = json.dumps(
        {
            "type": "user",
            "sessionId": "s",
            "uuid": "u",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"role": "user", "content": "hello"},
        }
    ).encode()
    history.write_bytes((record + b"\n") * 4 + b"caf\xe9 undecodable\n" + (record + b"\n") * 3)
    return root, history, 8


async def test_undecodable_source_advances_no_cursor_in_client_mode(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    """An aborted parse must never commit the whole-source bound.

    The bounds were derived from the entire source while the parse consumed only
    part of it, so committing them marks never-read content as consumed.
    """
    root, history, physical_lines = build_partially_undecodable_source(tmp_path)
    assert watcher_module._count_file_lines(history) == physical_lines

    watcher = make_pinned_watcher(root, tmp_path)
    manager = ClientStateManager(tmp_path / "client-state.json")
    api = FakeAPI(history)

    with caplog.at_level("ERROR"):
        await watcher._index_file_client_mode(history, api, manager)

    state = await manager.get_state()
    assert api.uploads == []
    assert state.local_positions == {}
    assert state.server_positions == {}
    assert str(history) in watcher._failed_files
    assert f"reason={durable_io.SOURCE_NOT_DECODABLE}" in caplog.text


async def test_undecodable_source_advances_no_cursor_in_server_mode(tmp_path: Path):
    """Server mode has no safe-interval clamp, so an over-advance is unbounded."""
    root, history, physical_lines = build_partially_undecodable_source(tmp_path)

    watcher = make_pinned_watcher(root, tmp_path)
    stored: list[dict] = []

    class Store:
        async def add_chunks_async(self, chunks):
            stored.extend(chunks)

    class Embedder:
        async def embed_chunks(self, chunks):
            return chunks

    await watcher._index_file(history, Embedder(), Store())

    assert watcher.state.get_position(str(history)) == 0
    assert watcher.state.get_position(str(history)) != physical_lines
    assert str(history) in watcher._failed_files


async def test_undecodable_source_retires_its_continuation(tmp_path: Path):
    """A record that can never be expanded must not sit in the outbox forever."""
    root, history, _ = build_partially_undecodable_source(tmp_path)
    watcher = make_pinned_watcher(root, tmp_path)
    manager = ClientStateManager(tmp_path / "client-state.json")
    api = FakeAPI(history)

    await watcher._index_file_client_mode(history, api, manager)

    state = await manager.get_state()
    assert state.pending_uploads == []
    assert state.catchup_failures[str(history)].reason == durable_io.SOURCE_NOT_DECODABLE


async def test_terminal_failure_does_not_block_reindex_acking_machine_wide(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A terminal failure is a reported shortfall, not outstanding work.

    Blocking completion on it means the server never learns this client
    finished, for every source on the machine, forever. This drives the real
    acking path rather than only observing the outbox, because the previous
    version of this test asserted the outbox and never once called the function
    whose behaviour it was named for.
    """
    acks: list[tuple[str, str | None]] = []

    class AckAPI(FakeAPI):
        async def ack_reindex(self, *, reindex_requested_at, status, reason=None):
            acks.append((status, reason))

    manager = ClientStateManager(tmp_path / "client-state.json")
    requested_at = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    await manager.prepare_reindex(requested_at)
    await manager.mark_reindex_queued()
    await manager.set_reindex_ack(status="queued")
    await manager.record_catchup_failure(
        CatchupInterval(
            file_path="/history/unreadable.jsonl",
            start_exclusive=0,
            end_inclusive=8,
            snapshot_digest="0" * 64,
            generation=(await manager.get_state()).reindex_generation,
        ),
        "source_not_decodable",
    )

    state = await manager.get_state()
    assert state.catchup_failures
    assert state.pending_uploads == []

    monkeypatch.setattr(watcher_module, "get_all_watchers", list)
    await watcher_module._maybe_ack_reindex_completed(AckAPI(tmp_path / "x.jsonl"), manager)

    assert acks, "a terminal failure must not suppress the completion ack"
    status, reason = acks[-1]
    assert status == "completed"
    # The shortfall is reported rather than hidden.
    assert reason is not None and "unreadable" in reason


def test_outbox_positions_for_one_file_must_be_nondecreasing():
    """Ordered outbox records define the consumed interval for a file.

    A record that walks backwards would re-commit an already-consumed range and
    make the ordered sequence stop describing one monotonic advance.
    """
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def record(record_id: str, sequence: int, position: int) -> dict:
        return {
            "record_id": record_id,
            "sequence": sequence,
            "file_path": "/history/session.jsonl",
            "chunks": [],
            "file_position": position,
            "created_at": now,
            "request_bytes": 1,
            "request_sha256": "a" * 64,
            "request_machine_id": "machine-a",
            "request_client_name": "machine-a",
            "payload_externalized": True,
            "payload_sha256": "b" * 64,
            "payload_bytes": 1,
        }

    ascending = client_state_module.ClientState(
        pending_uploads=[record("r1", 0, 3), record("r2", 1, 7)]
    )
    assert [item.file_position for item in ascending.pending_uploads] == [3, 7]

    with pytest.raises(ValueError, match="nondecreasing"):
        client_state_module.ClientState(pending_uploads=[record("r1", 0, 7), record("r2", 1, 3)])
