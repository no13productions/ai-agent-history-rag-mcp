"""Client-side state management for tracking uploads and catch-up logic.

Handles:
- Tracking which files need to be uploaded
- Local position tracking when offline
- Catch-up logic when reconnecting to server
- Pending upload queue
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from claude_history_rag import durable_io
from claude_history_rag.config import settings
from claude_history_rag.models import (
    MAX_CHUNK_UPLOAD_REQUEST_BYTES,
    chunk_upload_request_bytes,
    chunk_upload_request_sha256,
)

logger = logging.getLogger(__name__)

ReindexStatus = Literal["queue_pending", "queued", "completed"]
ReindexPhase = Literal["reset", "resume", "in_progress", "completed"]
CatchupFailureReason = Literal[
    "history_missing",
    "history_truncated",
    "history_replaced",
    "history_continuity_unproven",
    "unsupported_cursor_contract",
    # The source exists and is bounded, but could not be read as text at all.
    "source_not_decodable",
    # The source could not be served safely for any other reason. The precise
    # refusal code is logged; this vocabulary stays small deliberately so the
    # durable record does not drift every time a new refusal reason is added.
    "source_unreadable",
]
CURRENT_CLIENT_STATE_SCHEMA_VERSION = 3
MAX_CLIENT_STATE_BYTES = 16 * 1024 * 1024


class CatchupInterval(BaseModel):
    """A bounded physical-line interval requested by the server cursor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: str
    start_exclusive: int = Field(ge=0)
    end_inclusive: int = Field(ge=0)
    snapshot_digest: str | None = None
    outbox_record_id: str | None = None
    generation: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_non_empty(self) -> "CatchupInterval":
        """Reject empty or reversed work; those are not catch-up intervals."""
        if self.start_exclusive >= self.end_inclusive:
            raise ValueError("catch-up start must be before end")
        return self


class PendingUpload(BaseModel):
    """One immutable, ordered, bounded upload or durable continuation record."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    sequence: int = Field(default=0, ge=0)
    file_path: str
    chunks: list[dict[str, Any]]
    file_position: int
    created_at: datetime
    retry_count: int = 0
    followup_interval: CatchupInterval | None = None
    request_bytes: int = Field(default=0, ge=0)
    is_continuation: bool = False
    payload_externalized: bool = False
    start_position: int = Field(default=0, ge=0)
    source_snapshot_digest: str | None = None
    final_fragment: bool = True
    payload_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    payload_bytes: int = Field(default=0, ge=0)
    request_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    request_machine_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )
    request_client_name: str | None = Field(default=None, max_length=256)
    generation: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_record_shape(self) -> "PendingUpload":
        """Keep continuation, inline-legacy, and external payload shapes disjoint."""
        if self.is_continuation:
            if self.followup_interval is None:
                raise ValueError("continuation record requires an interval")
            if self.payload_externalized or self.chunks or self.request_bytes:
                raise ValueError("continuation record cannot carry an upload payload")
            if self.payload_sha256 is not None or self.payload_bytes:
                raise ValueError("continuation record cannot carry payload integrity fields")
            if self.request_sha256 is not None or self.request_machine_id is not None:
                raise ValueError("continuation record cannot carry request integrity fields")
            if self.request_client_name is not None:
                raise ValueError("continuation record cannot carry request identity")
            if self.followup_interval.generation != self.generation:
                raise ValueError("continuation generation must match its interval")
            if self.file_position != self.followup_interval.start_exclusive:
                raise ValueError("continuation position must match its interval start")
        elif self.payload_externalized and self.chunks:
            raise ValueError("externalized upload record cannot retain inline chunks")
        elif not self.payload_externalized and (
            self.payload_sha256 is not None
            or self.payload_bytes
            or self.request_sha256 is not None
            or self.request_machine_id is not None
            or self.request_client_name is not None
        ):
            raise ValueError("inline legacy record cannot claim external payload integrity")
        if not self.is_continuation and self.start_position > self.file_position:
            raise ValueError("upload start cannot exceed its acknowledged position")
        return self


class CatchupFailure(BaseModel):
    """A durable statement that a requested history interval cannot be reconstructed."""

    start_exclusive: int = Field(ge=0)
    end_inclusive: int = Field(ge=0)
    model_config = ConfigDict(extra="forbid")

    reason: CatchupFailureReason
    observed_at: datetime


class ClientState(BaseModel):
    """Client-side state for tracking uploads and positions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[3] = CURRENT_CLIENT_STATE_SCHEMA_VERSION

    # Local file positions (what we've chunked locally)
    local_positions: dict[str, int] = {}

    # Server-confirmed positions (last successful upload position per file)
    server_positions: dict[str, int] = {}

    # Pending uploads that failed and need retry
    pending_uploads: list[PendingUpload] = []

    # Content-bound cleanup work retained until the payload name is safely retired.
    payload_cleanup_pending: dict[str, str] = {}

    # Monotonic sequence authority for the durable ordered outbox.
    next_outbox_sequence: int = Field(default=1, ge=1)

    # Snapshot identity at the last fully acknowledged source boundary.
    committed_snapshot_digests: dict[str, str] = {}

    # Explicit terminal observations for catch-up intervals whose source history vanished.
    catchup_failures: dict[str, CatchupFailure] = {}

    # Last time we synced with the server
    last_server_sync: datetime | None = None

    # Connection status
    connected: bool = False

    # Reindex tracking
    reindex_required_at: datetime | None = None
    reindex_ack_at: datetime | None = None
    reindex_status: ReindexStatus | None = None
    reindex_generation: int = Field(default=0, ge=0)
    reindex_queue_session: str | None = None

    @model_validator(mode="after")
    def validate_outbox_identity(self) -> "ClientState":
        """Reject ambiguous durable identities before any migration can mutate disk."""
        record_ids = [record.record_id for record in self.pending_uploads]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("durable outbox record ids must be unique")
        if any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", record_id) is None
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            for record_id, digest in self.payload_cleanup_pending.items()
        ):
            raise ValueError("payload cleanup bindings must contain safe identities")
        positive_sequences = [record.sequence for record in self.pending_uploads if record.sequence]
        if len(positive_sequences) != len(set(positive_sequences)):
            raise ValueError("positive durable outbox sequences must be unique")
        file_positions: dict[str, int] = {}
        for record in sorted(self.pending_uploads, key=lambda item: item.sequence):
            if record.generation != self.reindex_generation:
                raise ValueError("pending record generation must match current state")
            if record.payload_externalized and (
                record.payload_sha256 is None
                or record.payload_bytes <= 0
                or record.request_bytes <= 0
                or record.request_sha256 is None
                or record.request_machine_id is None
            ):
                raise ValueError("current external payload requires complete integrity bindings")
            if record.is_continuation:
                continue
            previous = file_positions.get(record.file_path, 0)
            if record.file_position < previous:
                raise ValueError("per-file outbox positions must be nondecreasing")
            file_positions[record.file_path] = record.file_position
        if any(position < 0 for position in self.local_positions.values()):
            raise ValueError("local positions must be nonnegative")
        if any(position < 0 for position in self.server_positions.values()):
            raise ValueError("server positions must be nonnegative")
        if self.reindex_status is None:
            if self.reindex_required_at or self.reindex_queue_session:
                raise ValueError("inactive reindex cannot retain request or queue session")
        else:
            if self.reindex_required_at is None:
                raise ValueError("active reindex requires a request timestamp")
            if self.reindex_status == "queue_pending" and self.reindex_queue_session:
                raise ValueError("pending reindex queue cannot have a session owner")
            if self.reindex_status in {"queued", "completed"} and not self.reindex_queue_session:
                raise ValueError("queued reindex requires a session owner")
        return self


class ClientStateManager:
    """Manages client-side state persistence and catch-up logic."""

    def __init__(self, state_path: Path | None = None):
        self.state_path = state_path or (settings.state_path.parent / "client_state.json")
        self._state = ClientState()
        self._lock = asyncio.Lock()
        self._drain_lock = asyncio.Lock()
        self._session_id = uuid.uuid4().hex
        self._loaded = False

    @property
    def drain_lock(self) -> asyncio.Lock:
        """Process-wide network ownership gate for this durable outbox."""
        return self._drain_lock

    @property
    def _outbox_dir(self) -> Path:
        return self.state_path.parent / f"{self.state_path.name}.outbox"

    def _payload_path(self, record_id: str) -> Path:
        return self._outbox_dir / f"{record_id}.json"

    @staticmethod
    def _payload_bytes(chunks: list[dict[str, Any]]) -> bytes:
        return json.dumps(chunks, separators=(",", ":"), ensure_ascii=False).encode()

    @staticmethod
    def _payload_sha256(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def _read_payload_bytes(self, record_id: str) -> bytes:
        return durable_io.read_bytes(
            self._payload_path(record_id),
            durable_root=self._outbox_dir,
            max_bytes=MAX_CHUNK_UPLOAD_REQUEST_BYTES,
        )

    def _write_payload(self, record_id: str, chunks: list[dict[str, Any]]) -> None:
        """Atomically persist one bounded record payload outside the state index."""
        durable_io.atomic_write_bytes(
            self._payload_path(record_id),
            self._payload_bytes(chunks),
            durable_root=self._outbox_dir,
        )

    def _delete_payload(self, record_id: str, expected_sha256: str) -> bool:
        try:
            durable_io.delete_file(
                self._payload_path(record_id),
                durable_root=self._outbox_dir,
                expected_sha256=expected_sha256,
            )
            return True
        except OSError as error:
            logger.warning(
                "Could not remove acknowledged outbox payload: record=%s reason=cleanup_failed error_type=%s",
                record_id,
                type(error).__name__,
            )
            return False

    async def _finish_payload_cleanup(self, record_id: str, expected_sha256: str) -> None:
        if not self._delete_payload(record_id, expected_sha256):
            raise OSError("durable outbox payload cleanup remains pending")

        def clear_cleanup(state: ClientState) -> None:
            if state.payload_cleanup_pending.get(record_id) == expected_sha256:
                state.payload_cleanup_pending.pop(record_id)

        await self._mutate_state(clear_cleanup)
        self._remove_empty_outbox_dir()

    def _remove_empty_outbox_dir(self) -> None:
        # A non-empty or unconfirmed directory remains a safe retry target.
        with contextlib.suppress(OSError):
            durable_io.remove_empty_directory(
                self._outbox_dir,
                durable_parent=self.state_path.parent,
            )

    def _write_state_snapshot(self, state: ClientState | None = None) -> None:
        """Atomically replace the durable outbox index."""
        data = (state or self._state).model_dump(mode="json")
        durable_io.atomic_write_text(
            self.state_path,
            json.dumps(data, indent=2, default=str),
            durable_root=self.state_path.parent,
        )

    def _record_is_durable(self, record: PendingUpload) -> bool:
        """Prove a duplicate identity exists in both the disk index and payload store."""
        try:
            data = json.loads(
                durable_io.read_text(
                    self.state_path,
                    durable_root=self.state_path.parent,
                    max_bytes=MAX_CLIENT_STATE_BYTES,
                )
            )
            if not isinstance(data, dict):
                return False
            disk_record = next(
                (
                    PendingUpload(**item)
                    for item in data.get("pending_uploads", [])
                    if isinstance(item, dict) and item.get("record_id") == record.record_id
                ),
                None,
            )
            if disk_record != record:
                return False
            if not record.payload_externalized:
                return True
            payload_bytes = self._read_payload_bytes(record.record_id)
            payload = json.loads(payload_bytes)
            return (
                isinstance(payload, list)
                and record.payload_sha256 is not None
                and record.payload_bytes == len(payload_bytes)
                and self._payload_sha256(payload_bytes) == record.payload_sha256
            )
        except (OSError, TypeError, ValueError):
            return False

    @staticmethod
    def _load_failure_code(error: Exception) -> str:
        """Map detailed validation failures to stable, non-sensitive operator codes."""
        if isinstance(error, json.JSONDecodeError):
            return "state_json_invalid"
        if isinstance(error, FileNotFoundError):
            return "payload_missing"
        message = str(error)
        classified = (
            ("unsupported durable client state schema version", "schema_version_unsupported"),
            ("schema-v2 pending uploads lack", "legacy_request_binding_missing"),
            ("record ids must be unique", "record_identity_duplicate"),
            ("sequences must be unique", "record_sequence_duplicate"),
            ("digest mismatch", "payload_digest_mismatch"),
            ("byte count mismatch", "payload_size_mismatch"),
            ("payload file exceeds", "payload_oversize"),
            ("cannot securely authorize an external payload", "legacy_external_rejected"),
            ("request byte count mismatch", "request_size_mismatch"),
            ("request digest mismatch", "request_digest_mismatch"),
            ("unbound inline upload", "request_binding_missing"),
            ("exceeds upload limit", "request_oversize"),
        )
        for marker, code in classified:
            if marker in message:
                return code
        if isinstance(error, ValidationError):
            return "state_schema_invalid"
        return "state_load_failed"

    def _upgrade_legacy_state_data(self, data: Any) -> tuple[dict[str, Any], bool]:
        """Normalize schema-v1 data in memory before strict current-model parsing."""
        if not isinstance(data, dict):
            raise ValueError("durable client state must be an object")
        version = data.get("schema_version", 1)
        if version == CURRENT_CLIENT_STATE_SCHEMA_VERSION:
            return data, False
        if version == 2:
            pending = data.get("pending_uploads", [])
            if not isinstance(pending, list):
                raise ValueError("schema-v2 pending uploads must be a list")
            if any(
                not isinstance(record, dict) or not record.get("is_continuation", False)
                for record in pending
            ):
                raise ValueError(
                    "schema-v2 pending uploads lack a trustworthy full request binding"
                )
            return {**data, "schema_version": CURRENT_CLIENT_STATE_SCHEMA_VERSION}, True
        if version != 1:
            raise ValueError("unsupported durable client state schema version")

        upgraded = dict(data)
        upgraded["schema_version"] = CURRENT_CLIENT_STATE_SCHEMA_VERSION
        generation = upgraded.get("reindex_generation", 0)
        status = upgraded.get("reindex_status")
        required_at = upgraded.get("reindex_required_at")
        if status in {"pending", "queued"} or (required_at and status is None):
            upgraded["reindex_status"] = "queue_pending"
            upgraded["reindex_queue_session"] = None
        elif status == "completed":
            upgraded["reindex_queue_session"] = "legacy-completed"

        upgraded_pending: list[dict[str, Any]] = []
        for raw_record in upgraded.get("pending_uploads", []):
            if not isinstance(raw_record, dict):
                raise ValueError("legacy pending upload must be an object")
            record = dict(raw_record)
            record["generation"] = generation
            followup = record.get("followup_interval")
            if isinstance(followup, dict):
                record["followup_interval"] = {**followup, "generation": generation}
            if record.get("payload_externalized"):
                raise ValueError("schema-v1 state cannot securely authorize an external payload")
            upgraded_pending.append(record)
        upgraded["pending_uploads"] = upgraded_pending
        return upgraded, True

    async def _mutate_state(self, mutate: Callable[[ClientState], Any]) -> Any:
        """Persist a copied successor before publishing it as current memory state."""
        await self.get_state()
        async with self._lock:
            candidate = self._state.model_copy(deep=True)
            result = mutate(candidate)
            try:
                self._write_state_snapshot(candidate)
            except durable_io.DurableCommitUncertainError as error:
                if error.committed:
                    self._state = candidate
                raise
            self._state = candidate
            return result

    async def load_pending_chunks(self, pending: PendingUpload) -> list[dict[str, Any]]:
        """Load exactly one bounded payload record for transmission or inspection."""
        if not pending.payload_externalized:
            return list(pending.chunks)
        payload = self._read_payload_bytes(pending.record_id)
        if pending.payload_sha256 is None:
            raise ValueError("durable outbox payload is missing its integrity digest")
        if pending.payload_bytes != len(payload):
            raise ValueError("durable outbox payload byte count mismatch")
        if self._payload_sha256(payload) != pending.payload_sha256:
            raise ValueError("durable outbox payload digest mismatch")
        chunks = json.loads(payload)
        if not isinstance(chunks, list) or any(not isinstance(chunk, dict) for chunk in chunks):
            raise ValueError("durable outbox payload must be a list of chunk objects")
        return chunks

    async def load(self) -> ClientState:
        """Load state from disk."""
        async with self._lock:
            if self._loaded:
                return self._state.model_copy(deep=True)

            try:
                state_exists = durable_io.durable_file_exists(
                    self.state_path, durable_root=self.state_path.parent
                )
            except Exception as e:
                logger.error(
                    "Failed to load durable client outbox state: reason=%s error_type=%s record=none",
                    self._load_failure_code(e),
                    type(e).__name__,
                )
                raise RuntimeError("durable client outbox state could not be loaded") from e
            if state_exists:
                active_record_id: str | None = None
                try:
                    data, legacy_schema = self._upgrade_legacy_state_data(
                        json.loads(
                            durable_io.read_text(
                                self.state_path,
                                durable_root=self.state_path.parent,
                                max_bytes=MAX_CLIENT_STATE_BYTES,
                            )
                        )
                    )
                    raw_pending = data.get("pending_uploads", [])
                    if not legacy_schema and any(
                        isinstance(raw, dict)
                        and not raw.get("is_continuation", False)
                        and not raw.get("payload_externalized", False)
                        for raw in raw_pending
                    ):
                        raise ValueError(
                            "current durable client state cannot contain an unbound inline upload"
                        )
                    if raw_pending and isinstance(raw_pending[0], dict):
                        raw_record_id = raw_pending[0].get("record_id")
                        if isinstance(raw_record_id, str) and re.fullmatch(
                            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", raw_record_id
                        ):
                            active_record_id = raw_record_id
                    candidate_state = ClientState(**data)
                    changed = legacy_schema
                    migrated: list[PendingUpload] = []
                    payload_writes: list[tuple[str, list[dict[str, Any]]]] = []
                    max_chunks = min(settings.max_chunks_per_file, 500)
                    legacy_machine_id = settings.machine_id
                    legacy_client_name = settings.client_name or settings.machine_id
                    for pending in candidate_state.pending_uploads:
                        active_record_id = pending.record_id
                        if pending.is_continuation:
                            migrated.append(pending)
                            continue
                        if pending.payload_externalized:
                            payload = self._read_payload_bytes(pending.record_id)
                            chunks = json.loads(payload)
                            if not isinstance(chunks, list) or any(
                                not isinstance(chunk, dict) for chunk in chunks
                            ):
                                raise ValueError(
                                    "durable outbox payload must be a list of chunk objects"
                                )
                            canonical_payload = self._payload_bytes(chunks)
                            payload_sha256 = self._payload_sha256(canonical_payload)
                            if pending.payload_sha256 != payload_sha256:
                                raise ValueError("durable outbox payload digest mismatch")
                            if pending.payload_bytes != len(canonical_payload):
                                raise ValueError("durable outbox payload byte count mismatch")
                            request_bytes = chunk_upload_request_bytes(
                                chunks,
                                machine_id=pending.request_machine_id or "",
                                client_name=pending.request_client_name,
                                source_file=pending.file_path,
                                file_position=pending.file_position,
                            )
                            request_sha256 = chunk_upload_request_sha256(
                                chunks,
                                machine_id=pending.request_machine_id or "",
                                client_name=pending.request_client_name,
                                source_file=pending.file_path,
                                file_position=pending.file_position,
                            )
                            if (
                                len(chunks) > max_chunks
                                or request_bytes > MAX_CHUNK_UPLOAD_REQUEST_BYTES
                            ):
                                raise ValueError("externalized outbox record exceeds upload limit")
                            if pending.request_bytes != request_bytes:
                                raise ValueError("durable outbox request byte count mismatch")
                            if pending.request_sha256 != request_sha256:
                                raise ValueError("durable outbox request digest mismatch")
                            updated = pending.model_copy(
                                update={
                                    "payload_sha256": payload_sha256,
                                    "request_bytes": request_bytes,
                                    "request_sha256": request_sha256,
                                    "payload_bytes": len(canonical_payload),
                                }
                            )
                            migrated.append(updated)
                            changed = changed or updated != pending
                            if legacy_schema:
                                payload_writes.append((pending.record_id, chunks))
                            continue
                        base_position = candidate_state.server_positions.get(pending.file_path, 0)
                        chunks = list(pending.chunks)
                        fragments: list[list[dict[str, Any]]] = []
                        batch: list[dict[str, Any]] = []
                        for chunk in chunks:
                            candidate = [*batch, chunk]
                            candidate_too_large = len(candidate) > max_chunks or (
                                chunk_upload_request_bytes(
                                    candidate,
                                    machine_id=legacy_machine_id,
                                    client_name=legacy_client_name,
                                    source_file=pending.file_path,
                                    file_position=pending.file_position,
                                )
                                > MAX_CHUNK_UPLOAD_REQUEST_BYTES
                            )
                            if candidate_too_large and batch:
                                fragments.append(batch)
                                batch = [chunk]
                                if (
                                    chunk_upload_request_bytes(
                                        batch,
                                        machine_id=legacy_machine_id,
                                        client_name=legacy_client_name,
                                        source_file=pending.file_path,
                                        file_position=pending.file_position,
                                    )
                                    > MAX_CHUNK_UPLOAD_REQUEST_BYTES
                                ):
                                    raise ValueError(
                                        "indivisible legacy chunk exceeds upload limit"
                                    )
                            elif candidate_too_large:
                                raise ValueError("indivisible legacy chunk exceeds upload limit")
                            else:
                                batch = candidate
                        if batch or not chunks:
                            fragments.append(batch)
                        if len(fragments) == 1:
                            request_bytes = chunk_upload_request_bytes(
                                pending.chunks,
                                machine_id=legacy_machine_id,
                                client_name=legacy_client_name,
                                source_file=pending.file_path,
                                file_position=pending.file_position,
                            )
                            if request_bytes > MAX_CHUNK_UPLOAD_REQUEST_BYTES:
                                raise ValueError("legacy outbox record exceeds upload limit")
                            payload_writes.append((pending.record_id, pending.chunks))
                            migrated.append(
                                pending.model_copy(
                                    update={
                                        "request_bytes": request_bytes,
                                        "request_sha256": chunk_upload_request_sha256(
                                            pending.chunks,
                                            machine_id=legacy_machine_id,
                                            client_name=legacy_client_name,
                                            source_file=pending.file_path,
                                            file_position=pending.file_position,
                                        ),
                                        "request_machine_id": legacy_machine_id,
                                        "request_client_name": legacy_client_name,
                                        "chunks": [],
                                        "payload_externalized": True,
                                        "payload_sha256": self._payload_sha256(
                                            self._payload_bytes(pending.chunks)
                                        ),
                                        "payload_bytes": len(self._payload_bytes(pending.chunks)),
                                    }
                                )
                            )
                            changed = True
                            continue
                        changed = True
                        for index, fragment in enumerate(fragments):
                            is_final = index == len(fragments) - 1
                            position = pending.file_position if is_final else base_position
                            fragment_id = hashlib.sha256(
                                f"{pending.record_id}\x00{index}".encode()
                            ).hexdigest()
                            fragment_bytes = chunk_upload_request_bytes(
                                fragment,
                                machine_id=legacy_machine_id,
                                client_name=legacy_client_name,
                                source_file=pending.file_path,
                                file_position=position,
                            )
                            if fragment_bytes > MAX_CHUNK_UPLOAD_REQUEST_BYTES:
                                raise ValueError("legacy outbox fragment exceeds upload limit")
                            payload_writes.append((fragment_id, fragment))
                            migrated.append(
                                pending.model_copy(
                                    update={
                                        "record_id": fragment_id,
                                        "sequence": 0,
                                        "chunks": [],
                                        "file_position": position,
                                        "start_position": base_position,
                                        "final_fragment": is_final,
                                        "followup_interval": pending.followup_interval
                                        if is_final
                                        else None,
                                        "request_bytes": fragment_bytes,
                                        "request_sha256": chunk_upload_request_sha256(
                                            fragment,
                                            machine_id=legacy_machine_id,
                                            client_name=legacy_client_name,
                                            source_file=pending.file_path,
                                            file_position=position,
                                        ),
                                        "request_machine_id": legacy_machine_id,
                                        "request_client_name": legacy_client_name,
                                        "payload_externalized": True,
                                        "payload_sha256": self._payload_sha256(
                                            self._payload_bytes(fragment)
                                        ),
                                        "payload_bytes": len(self._payload_bytes(fragment)),
                                    }
                                )
                            )
                    candidate_state.pending_uploads = migrated
                    next_sequence = 1
                    for pending in candidate_state.pending_uploads:
                        if pending.sequence < next_sequence:
                            pending.sequence = next_sequence
                            changed = True
                        next_sequence = pending.sequence + 1
                    if candidate_state.next_outbox_sequence < next_sequence:
                        candidate_state.next_outbox_sequence = next_sequence
                        changed = True
                    if changed:
                        created_payloads: dict[str, str] = {}
                        replaced_payloads: dict[str, tuple[bytes, str]] = {}
                        try:
                            for record_id, chunks in payload_writes:
                                payload_digest = self._payload_sha256(self._payload_bytes(chunks))
                                existed = durable_io.durable_file_exists(
                                    self._payload_path(record_id),
                                    durable_root=self._outbox_dir,
                                )
                                if existed:
                                    replaced_payloads[record_id] = (
                                        self._read_payload_bytes(record_id),
                                        payload_digest,
                                    )
                                self._write_payload(record_id, chunks)
                                if not existed:
                                    created_payloads[record_id] = payload_digest
                            try:
                                self._write_state_snapshot(candidate_state)
                            except durable_io.DurableCommitUncertainError as error:
                                if error.committed:
                                    self._state = candidate_state
                                raise
                        except durable_io.DurableCommitUncertainError:
                            # A committed payload or index is reconciled by the next load.
                            raise
                        except BaseException:
                            for record_id, digest in created_payloads.items():
                                durable_io.delete_file(
                                    self._payload_path(record_id),
                                    durable_root=self._outbox_dir,
                                    expected_sha256=digest,
                                )
                            for record_id, (payload, staged_digest) in replaced_payloads.items():
                                durable_io.delete_file(
                                    self._payload_path(record_id),
                                    durable_root=self._outbox_dir,
                                    expected_sha256=staged_digest,
                                )
                                durable_io.atomic_write_bytes(
                                    self._payload_path(record_id),
                                    payload,
                                    durable_root=self._outbox_dir,
                                )
                            self._remove_empty_outbox_dir()
                            raise
                    self._state = candidate_state
                    logger.info(
                        f"Loaded client state: {len(self._state.local_positions)} files, "
                        f"{len(self._state.pending_uploads)} pending uploads"
                    )
                except Exception as e:
                    logger.error(
                        "Failed to load durable client outbox state: reason=%s error_type=%s record=%s",
                        self._load_failure_code(e),
                        type(e).__name__,
                        active_record_id or "none",
                    )
                    raise RuntimeError("durable client outbox state could not be loaded") from e
            else:
                logger.info("No existing client state, starting fresh")

            self._loaded = True
            return self._state.model_copy(deep=True)

    async def save(self) -> None:
        """Save state to disk."""
        async with self._lock:
            try:
                self._write_state_snapshot()
                logger.debug("Saved durable client state snapshot")
            except Exception as e:
                logger.error("Failed to save durable client state: error_type=%s", type(e).__name__)
                raise

    async def get_state(self) -> ClientState:
        """Get current state, loading from disk if needed."""
        if not self._loaded:
            await self.load()
        return self._state.model_copy(deep=True)

    async def get_summary(self) -> dict[str, Any]:
        """Get a lightweight summary of client state for diagnostics."""
        state = await self.get_state()
        summary: dict[str, Any] = {
            "pending_uploads": len(state.pending_uploads),
            "last_server_sync": state.last_server_sync.isoformat()
            if state.last_server_sync
            else None,
            "connected": state.connected,
            "retry_total": sum(record.retry_count for record in state.pending_uploads),
            "catchup_failures": len(state.catchup_failures),
            "reindex_generation": state.reindex_generation,
            "reindex_status": state.reindex_status,
        }
        if state.last_server_sync:
            summary["last_server_sync_age_min"] = int(
                (datetime.now(timezone.utc) - state.last_server_sync).total_seconds() / 60
            )
        return summary

    async def update_local_position(self, file_path: str, position: int) -> None:
        """Update the local position for a file."""
        await self._mutate_state(
            lambda state: state.local_positions.__setitem__(file_path, position)
        )

    async def advance_local_position(self, file_path: str, position: int) -> None:
        """Advance local progress monotonically after a complete server acknowledgment."""

        def mutate(state: ClientState) -> None:
            state.local_positions[file_path] = max(
                state.local_positions.get(file_path, 0), position
            )

        await self._mutate_state(mutate)

    async def update_server_position(self, file_path: str, position: int) -> None:
        """Update the server-confirmed position for a file."""

        def mutate(state: ClientState) -> None:
            state.server_positions[file_path] = max(
                state.server_positions.get(file_path, 0), position
            )
            state.last_server_sync = datetime.now(timezone.utc)

        await self._mutate_state(mutate)

    async def reconcile_server_positions(self, positions: dict[str, int]) -> None:
        """Replace the local mirror with the latest server-authoritative cursor snapshot."""

        def mutate(state: ClientState) -> None:
            state.server_positions = dict(positions)
            state.last_server_sync = datetime.now(timezone.utc)

        await self._mutate_state(mutate)

    async def set_connected(self, connected: bool) -> None:
        """Update connection status."""
        await self._mutate_state(lambda state: setattr(state, "connected", connected))

    async def prepare_reindex(self, requested_at: datetime) -> ReindexPhase:
        """Atomically begin/reset a new generation or resume its queue phase."""
        removed_payloads: list[tuple[str, str]] = []

        def mutate(state: ClientState) -> ReindexPhase:
            same_request = state.reindex_required_at == requested_at
            if same_request and state.reindex_status == "completed":
                return "completed"
            if (
                same_request
                and state.reindex_status == "queued"
                and state.reindex_queue_session == self._session_id
            ):
                return "in_progress"
            if same_request and state.reindex_status in {"queue_pending", "queued"}:
                state.reindex_status = "queue_pending"
                state.reindex_queue_session = None
                state.reindex_ack_at = None
                return "resume"
            removed_payloads.extend(
                (pending.record_id, pending.payload_sha256)
                for pending in state.pending_uploads
                if pending.payload_externalized and pending.payload_sha256 is not None
            )
            state.payload_cleanup_pending.update(removed_payloads)
            state.reindex_generation += 1
            state.reindex_required_at = requested_at
            state.reindex_ack_at = None
            state.reindex_status = "queue_pending"
            state.reindex_queue_session = None
            state.local_positions = {}
            state.server_positions = {}
            state.pending_uploads = []
            state.committed_snapshot_digests = {}
            state.catchup_failures = {}
            state.last_server_sync = None
            return "reset"

        await self.get_state()
        async with self._lock:
            candidate = self._state.model_copy(deep=True)
            phase = mutate(candidate)
            if phase not in {"completed", "in_progress"}:
                try:
                    self._write_state_snapshot(candidate)
                except durable_io.DurableCommitUncertainError as error:
                    if error.committed:
                        self._state = candidate
                    raise
                self._state = candidate
        for record_id, digest in removed_payloads:
            await self._finish_payload_cleanup(record_id, digest)
        return phase

    async def mark_reindex_queued(self) -> None:
        """Persist successful full-source scheduling before acknowledging externally."""

        def mutate(state: ClientState) -> None:
            if state.reindex_required_at and state.reindex_status == "queue_pending":
                state.reindex_status = "queued"
                state.reindex_queue_session = self._session_id

        await self._mutate_state(mutate)

    async def set_reindex_ack(self, status: Literal["queued", "completed"] | None = None) -> None:
        """Record that we acknowledged a reindex request."""

        def mutate(state: ClientState) -> None:
            state.reindex_ack_at = datetime.now(timezone.utc)
            if status:
                state.reindex_status = status

        await self._mutate_state(mutate)

    async def add_pending_upload(
        self,
        file_path: str,
        chunks: list[dict[str, Any]],
        file_position: int,
    ) -> None:
        """Append one immutable upload record to the durable ordered outbox."""
        await self.replace_pending_upload(
            file_path,
            chunks,
            file_position,
            record_id=uuid.uuid4().hex,
        )

    async def replace_pending_upload(
        self,
        file_path: str,
        chunks: list[dict[str, Any]],
        file_position: int,
        followup_interval: CatchupInterval | None = None,
        *,
        record_id: str | None = None,
        request_bytes: int = 0,
        is_continuation: bool = False,
        start_position: int = 0,
        source_snapshot_digest: str | None = None,
        final_fragment: bool = True,
        generation: int | None = None,
        machine_id: str | None = None,
        client_name: str | None = None,
    ) -> None:
        """Append one immutable record; retained name preserves old callers."""
        await self.get_state()
        durable_record_id = record_id or uuid.uuid4().hex
        durable_generation = self._state.reindex_generation if generation is None else generation
        measured_bytes = 0
        request_sha256: str | None = None
        request_machine_id: str | None = None
        request_client_name: str | None = None
        if not is_continuation:
            request_machine_id = machine_id if machine_id is not None else settings.machine_id
            request_client_name = (
                client_name
                if client_name is not None
                else settings.client_name or settings.machine_id
            )
            measured_bytes = chunk_upload_request_bytes(
                chunks,
                machine_id=request_machine_id,
                client_name=request_client_name,
                source_file=file_path,
                file_position=file_position,
            )
            request_sha256 = chunk_upload_request_sha256(
                chunks,
                machine_id=request_machine_id,
                client_name=request_client_name,
                source_file=file_path,
                file_position=file_position,
            )
            if (
                len(chunks) > min(settings.max_chunks_per_file, 500)
                or measured_bytes > MAX_CHUNK_UPLOAD_REQUEST_BYTES
            ):
                raise ValueError("unbounded upload cannot enter the durable outbox")
            if request_bytes and request_bytes != measured_bytes:
                raise ValueError("outbox request byte measurement mismatch")
            request_bytes = measured_bytes
        async with self._lock:
            if durable_generation != self._state.reindex_generation:
                raise ValueError("stale reindex generation cannot enter durable outbox")
            existing = next(
                (p for p in self._state.pending_uploads if p.record_id == durable_record_id),
                None,
            )
            if existing is not None and self._record_is_durable(existing):
                return

            candidate = self._state.model_copy(deep=True)
            candidate.pending_uploads = [
                pending
                for pending in candidate.pending_uploads
                if pending.record_id != durable_record_id
            ]
            candidate.pending_uploads.append(
                PendingUpload(
                    record_id=durable_record_id,
                    sequence=candidate.next_outbox_sequence,
                    file_path=file_path,
                    chunks=[] if not is_continuation else chunks,
                    file_position=file_position,
                    created_at=datetime.now(timezone.utc),
                    followup_interval=followup_interval,
                    request_bytes=request_bytes,
                    is_continuation=is_continuation,
                    payload_externalized=not is_continuation,
                    start_position=start_position,
                    source_snapshot_digest=source_snapshot_digest,
                    final_fragment=final_fragment,
                    payload_sha256=(
                        None
                        if is_continuation
                        else self._payload_sha256(self._payload_bytes(chunks))
                    ),
                    payload_bytes=(0 if is_continuation else len(self._payload_bytes(chunks))),
                    request_sha256=request_sha256,
                    request_machine_id=request_machine_id,
                    request_client_name=request_client_name,
                    generation=durable_generation,
                )
            )
            candidate.next_outbox_sequence += 1
            staged_payload = False
            previous_payload: bytes | None = None
            staged_digest = self._payload_sha256(self._payload_bytes(chunks))
            if not is_continuation:
                if durable_io.durable_file_exists(
                    self._payload_path(durable_record_id), durable_root=self._outbox_dir
                ):
                    previous_payload = self._read_payload_bytes(durable_record_id)
                self._write_payload(durable_record_id, chunks)
                staged_payload = True
            try:
                self._write_state_snapshot(candidate)
            except durable_io.DurableCommitUncertainError as error:
                if error.committed:
                    self._state = candidate
                raise
            except BaseException:
                if staged_payload:
                    durable_io.delete_file(
                        self._payload_path(durable_record_id),
                        durable_root=self._outbox_dir,
                        expected_sha256=staged_digest,
                    )
                    if previous_payload is None:
                        pass
                    else:
                        durable_io.atomic_write_bytes(
                            self._payload_path(durable_record_id),
                            previous_payload,
                            durable_root=self._outbox_dir,
                        )
                    self._remove_empty_outbox_dir()
                raise
            self._state = candidate
        logger.info(
            "Appended durable outbox record: record=%s generation=%d source=%s chunks=%d through=%d",
            durable_record_id,
            durable_generation,
            hashlib.sha256(file_path.encode()).hexdigest()[:12],
            len(chunks),
            file_position,
        )

    async def add_continuation(self, interval: CatchupInterval) -> CatchupInterval:
        """Persist a continuation marker before exposing it to the in-memory queue."""
        identity = hashlib.sha256(
            json.dumps(
                {
                    "file_path": interval.file_path,
                    "start": interval.start_exclusive,
                    "end": interval.end_inclusive,
                    "snapshot": interval.snapshot_digest,
                    "generation": interval.generation,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        await self.replace_pending_upload(
            interval.file_path,
            [],
            interval.start_exclusive,
            followup_interval=interval,
            record_id=identity,
            is_continuation=True,
            generation=interval.generation,
        )
        return interval.model_copy(update={"outbox_record_id": identity})

    async def complete_continuation(self, record_id: str | None) -> None:
        """Retire an expanded continuation by stable identity."""
        if record_id is None:
            return

        def mutate(state: ClientState) -> None:
            state.pending_uploads = [
                pending for pending in state.pending_uploads if pending.record_id != record_id
            ]

        await self._mutate_state(mutate)

    async def replace_continuation(
        self,
        record_id: str | None,
        followup: CatchupInterval | None,
    ) -> CatchupInterval | None:
        """Atomically retire one continuation and persist its successor, if any."""
        durable_followup: CatchupInterval | None = None

        def mutate(state: ClientState) -> None:
            nonlocal durable_followup
            if record_id is not None:
                state.pending_uploads = [
                    pending for pending in state.pending_uploads if pending.record_id != record_id
                ]
            if followup is None:
                return
            if followup.generation != state.reindex_generation:
                return
            identity = hashlib.sha256(
                json.dumps(
                    {
                        "file_path": followup.file_path,
                        "start": followup.start_exclusive,
                        "end": followup.end_inclusive,
                        "snapshot": followup.snapshot_digest,
                        "generation": followup.generation,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if not any(p.record_id == identity for p in state.pending_uploads):
                state.pending_uploads.append(
                    PendingUpload(
                        record_id=identity,
                        sequence=state.next_outbox_sequence,
                        file_path=followup.file_path,
                        chunks=[],
                        file_position=followup.start_exclusive,
                        created_at=datetime.now(timezone.utc),
                        followup_interval=followup,
                        is_continuation=True,
                        generation=followup.generation,
                    )
                )
                state.next_outbox_sequence += 1
            durable_followup = followup.model_copy(update={"outbox_record_id": identity})

        await self._mutate_state(mutate)
        return durable_followup

    async def complete_pending_upload(
        self,
        record_id: str,
    ) -> tuple[bool, CatchupInterval | None]:
        """Retire one acknowledged record and durably append its continuation atomically."""
        removed: PendingUpload | None = None
        durable_followup: CatchupInterval | None = None
        cleanup_digest: str | None = None

        def mutate(state: ClientState) -> tuple[bool, CatchupInterval | None]:
            nonlocal cleanup_digest, durable_followup, removed
            removed = next((p for p in state.pending_uploads if p.record_id == record_id), None)
            if removed is None:
                cleanup_digest = state.payload_cleanup_pending.get(record_id)
                return cleanup_digest is not None, None
            if removed.generation != state.reindex_generation:
                return False, None
            file_head = min(
                (
                    pending.sequence
                    for pending in state.pending_uploads
                    if pending.file_path == removed.file_path
                ),
                default=removed.sequence,
            )
            if removed.sequence != file_head:
                raise ValueError("outbox record cannot complete ahead of its file predecessor")
            state.pending_uploads = [p for p in state.pending_uploads if p.record_id != record_id]
            if removed.payload_externalized and removed.payload_sha256 is not None:
                cleanup_digest = removed.payload_sha256
                state.payload_cleanup_pending[record_id] = cleanup_digest
            state.server_positions[removed.file_path] = removed.file_position
            state.local_positions[removed.file_path] = removed.file_position
            if removed.final_fragment and removed.source_snapshot_digest:
                state.committed_snapshot_digests[removed.file_path] = removed.source_snapshot_digest
            state.last_server_sync = datetime.now(timezone.utc)
            followup_interval = removed.followup_interval
            if followup_interval is not None:
                identity = hashlib.sha256(
                    json.dumps(
                        {
                            "file_path": followup_interval.file_path,
                            "start": followup_interval.start_exclusive,
                            "end": followup_interval.end_inclusive,
                            "snapshot": followup_interval.snapshot_digest,
                            "generation": followup_interval.generation,
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                if not any(p.record_id == identity for p in state.pending_uploads):
                    state.pending_uploads.append(
                        PendingUpload(
                            record_id=identity,
                            sequence=state.next_outbox_sequence,
                            file_path=removed.file_path,
                            chunks=[],
                            file_position=removed.file_position,
                            created_at=datetime.now(timezone.utc),
                            followup_interval=followup_interval,
                            is_continuation=True,
                            generation=followup_interval.generation,
                        )
                    )
                    state.next_outbox_sequence += 1
                durable_followup = followup_interval.model_copy(
                    update={"outbox_record_id": identity}
                )
            return True, durable_followup

        await self.get_state()
        async with self._lock:
            candidate = self._state.model_copy(deep=True)
            completed, durable_followup = mutate(candidate)
            if not completed:
                return False, None
            try:
                self._write_state_snapshot(candidate)
            except durable_io.DurableCommitUncertainError as error:
                if error.committed:
                    self._state = candidate
                raise
            self._state = candidate
        if completed and cleanup_digest is not None:
            await self._finish_payload_cleanup(record_id, cleanup_digest)
        return completed, durable_followup

    async def get_pending_uploads(self) -> list[PendingUpload]:
        """Get all pending uploads."""
        state = await self.get_state()
        return sorted(state.pending_uploads, key=lambda record: record.sequence)

    async def remove_pending_upload(self, record_id: str) -> None:
        """Remove exactly one outbox record by stable identity."""
        removed: PendingUpload | None = None

        def mutate(state: ClientState) -> None:
            nonlocal removed
            removed = next((p for p in state.pending_uploads if p.record_id == record_id), None)
            state.pending_uploads = [
                pending for pending in state.pending_uploads if pending.record_id != record_id
            ]
            if removed and removed.payload_externalized and removed.payload_sha256 is not None:
                state.payload_cleanup_pending[record_id] = removed.payload_sha256

        await self._mutate_state(mutate)
        if removed and removed.payload_externalized and removed.payload_sha256 is not None:
            await self._finish_payload_cleanup(record_id, removed.payload_sha256)

    async def increment_retry_count(self, record_id: str) -> int:
        """Increment retry count for a pending upload. Returns new count."""

        def mutate(state: ClientState) -> int:
            for pending in state.pending_uploads:
                if pending.record_id == record_id:
                    pending.retry_count += 1
                    return pending.retry_count
            return 0

        return await self._mutate_state(mutate)

    async def get_files_needing_catchup(
        self,
        server_positions: dict[str, int],
    ) -> list[CatchupInterval]:
        """Get list of files that need catch-up after reconnecting.

        Compares local positions with server positions to find gaps.

        Args:
            server_positions: Positions reported by the server for our machine

        Returns:
            Bounded intervals for files where local > server (we have more than
            the server). Files with pending uploads are withheld until that
            outbox work succeeds or fails, preventing overlapping reconstruction.
        """
        state = await self.get_state()
        catchup_files: list[CatchupInterval] = []
        pending_files = {pending.file_path for pending in state.pending_uploads}

        for file_path, local_pos in state.local_positions.items():
            server_pos = server_positions.get(file_path, 0)
            if local_pos > server_pos and file_path not in pending_files:
                catchup_files.append(
                    CatchupInterval(
                        file_path=file_path,
                        start_exclusive=server_pos,
                        end_inclusive=local_pos,
                        generation=state.reindex_generation,
                    )
                )
                logger.info(
                    "File needs catch-up: source=%s server=%d local=%d",
                    hashlib.sha256(file_path.encode()).hexdigest()[:12],
                    server_pos,
                    local_pos,
                )

        return catchup_files

    async def record_catchup_failure(self, interval: CatchupInterval, reason: str) -> None:
        """Record a bounded interval that cannot be reconstructed from local history."""

        def mutate(state: ClientState) -> bool:
            if interval.generation != state.reindex_generation:
                return False
            state.catchup_failures[interval.file_path] = CatchupFailure(
                start_exclusive=interval.start_exclusive,
                end_inclusive=interval.end_inclusive,
                reason=reason,
                observed_at=datetime.now(timezone.utc),
            )
            return True

        if await self._mutate_state(mutate):
            logger.warning(
                "Persisted catch-up failure: reason=%s source=%s interval=(%d,%d] generation=%d",
                reason,
                hashlib.sha256(interval.file_path.encode()).hexdigest()[:12],
                interval.start_exclusive,
                interval.end_inclusive,
                interval.generation,
            )

    async def clear_catchup_failure(self, file_path: str) -> None:
        """Clear a prior terminal observation after the interval becomes reconstructable."""
        await self._mutate_state(lambda state: state.catchup_failures.pop(file_path, None))

    async def handle_missing_history(self, interval: CatchupInterval) -> None:
        """Record deleted source history without claiming the gap was recovered."""
        logger.warning(
            "Cannot reconstruct catch-up interval (%d, %d] because local history is missing: source=%s generation=%d",
            interval.start_exclusive,
            interval.end_inclusive,
            hashlib.sha256(interval.file_path.encode()).hexdigest()[:12],
            interval.generation,
        )
        await self.record_catchup_failure(interval, "history_missing")

    async def clear_stale_pending_uploads(self, max_age_hours: int = 72) -> int:
        """Report stale durable work without deleting unacknowledged history."""
        state = await self.get_state()
        now = datetime.now(timezone.utc)
        stale = sum(
            (now - pending.created_at).total_seconds() >= max_age_hours * 3600
            for pending in state.pending_uploads
        )
        if stale:
            logger.warning(
                "Retaining %d unacknowledged outbox records older than %dh",
                stale,
                max_age_hours,
            )
        return 0


# Global state manager instance
_state_manager: ClientStateManager | None = None


def get_client_state_manager() -> ClientStateManager:
    """Get or create the global client state manager."""
    global _state_manager
    if _state_manager is None:
        _state_manager = ClientStateManager()
    return _state_manager
