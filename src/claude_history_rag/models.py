"""Pydantic models for parsing and chunking."""

import hashlib
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MachineId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
ShortText = Annotated[str, Field(min_length=1, max_length=512)]
PathText = Annotated[str, Field(min_length=1, max_length=4096)]
DiagnosticMap = Annotated[dict[str, Any], Field(max_length=64)]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
StrictBool = Annotated[bool, Field(strict=True)]


class UserMessage(BaseModel):
    """User message content."""

    role: str = "user"
    content: str | list[dict[str, Any]]


class AssistantMessage(BaseModel):
    """Assistant message with content blocks.

    Note: The usage field format has evolved over time to support new billing features.
    We use dict[str, Any] to flexibly support both legacy and future formats.

    Legacy format (pre-Dec 2025):
        {"input_tokens": 100, "output_tokens": 50}

    Current format (Dec 2025):
        {
            "input_tokens": 3,
            "cache_creation_input_tokens": 5871,
            "cache_read_input_tokens": 14747,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 5871,
                "ephemeral_1h_input_tokens": 0
            },
            "output_tokens": 3,
            "service_tier": "standard",
            "server_tool_use": {
                "web_search_requests": 0,
                "web_fetch_requests": 0
            }
        }
    """

    id: str | None = None
    role: str = "assistant"
    model: str | None = None
    content: list[dict[str, Any]]
    usage: dict[str, Any] | None = None  # Flexible to handle format evolution


class HistoryEntry(BaseModel):
    """A single entry from the JSONL history file."""

    type: str  # "user", "assistant", "summary", "system", and newer no-op types
    message: UserMessage | AssistantMessage | None = None
    summary: str | None = None  # For legacy "summary" type entries (pre-2.1)
    # Current Claude Code (>=2.1) emits compaction summaries as ordinary
    # "user"/"assistant" entries flagged with isCompactSummary instead of a
    # dedicated "summary" type. The summary text lives in message.content.
    isCompactSummary: bool = Field(default=False, alias="isCompactSummary")
    isVisibleInTranscriptOnly: bool = Field(default=False, alias="isVisibleInTranscriptOnly")
    isMeta: bool = Field(default=False, alias="isMeta")
    subtype: str | None = None  # For system type (e.g., "init")
    uuid: str | None = None
    parentUuid: str | None = Field(default=None, alias="parentUuid")
    leafUuid: str | None = Field(default=None, alias="leafUuid")
    timestamp: datetime | None = None
    sessionId: str | None = Field(default=None, alias="sessionId")
    cwd: str | None = None
    version: str | None = None
    costUSD: float | None = Field(default=None, alias="costUSD")
    durationMs: int | None = Field(default=None, alias="durationMs")

    model_config = {"populate_by_name": True}


class Chunk(BaseModel):
    """A chunk ready for embedding and storage."""

    id: str
    content: str  # Text for embedding (with context prefix)
    chunk_type: str  # "turn", "file_change", "summary"
    session_id: str
    project_path: str
    project_name: str
    timestamp: datetime
    user_uuid: str | None = None
    assistant_uuid: str | None = None
    file_path: str | None = None  # For file_change chunks
    operation: str | None = None  # For file_change chunks
    model: str | None = None
    source_file: str
    source_line: int
    # Internal parser progress metadata. This is deliberately excluded from the
    # stored chunk contract: source_line remains provenance, while consumed_line
    # is the inclusive physical input boundary completed by this semantic unit.
    consumed_line: int | None = Field(default=None, ge=0, exclude=True)
    parent_chunk_id: str | None = None
    child_chunk_ids: list[str] | None = None
    machine_id: str | None = None  # For multi-machine support


# ============================================================
# API Request/Response Models (Client/Server Communication)
# ============================================================

# aiohttp's request-body ceiling is 1 MiB by default. The client and server
# share this explicit authority so an upload is bounded by serialized bytes,
# including its envelope, before it is persisted or transmitted.
MAX_CHUNK_UPLOAD_REQUEST_BYTES = 1024 * 1024


class ChunkUploadRequest(BaseModel):
    """Request to upload chunks from a client to the server."""

    machine_id: MachineId  # Identifies source machine
    client_name: str | None = Field(default=None, max_length=256)
    chunks: list[dict[str, Any]] = Field(max_length=500)  # Chunks without vectors
    source_file: PathText  # Source file being indexed
    file_position: int = Field(ge=0)  # Line number reached in file


def chunk_upload_request_body(
    chunks: list[dict[str, Any]],
    *,
    machine_id: str,
    client_name: str | None,
    source_file: str,
    file_position: int,
) -> bytes:
    """Serialize the one canonical UTF-8 JSON body for a chunk upload."""
    request = ChunkUploadRequest(
        machine_id=machine_id,
        client_name=client_name,
        chunks=chunks,
        source_file=source_file,
        file_position=file_position,
    )
    return request.model_dump_json().encode("utf-8")


def chunk_upload_request_bytes(
    chunks: list[dict[str, Any]],
    *,
    machine_id: str,
    client_name: str | None,
    source_file: str,
    file_position: int,
) -> int:
    """Return the exact byte size of the canonical chunk upload body."""
    return len(
        chunk_upload_request_body(
            chunks,
            machine_id=machine_id,
            client_name=client_name,
            source_file=source_file,
            file_position=file_position,
        )
    )


def chunk_upload_request_sha256(
    chunks: list[dict[str, Any]],
    *,
    machine_id: str,
    client_name: str | None,
    source_file: str,
    file_position: int,
) -> str:
    """Bind every transmitted semantic field to the canonical request body."""
    return hashlib.sha256(
        chunk_upload_request_body(
            chunks,
            machine_id=machine_id,
            client_name=client_name,
            source_file=source_file,
            file_position=file_position,
        )
    ).hexdigest()


class ChunkUploadResponse(BaseModel):
    """Response after uploading chunks."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error"]
    chunks_received: NonNegativeInt
    chunks_embedded: NonNegativeInt
    chunks_stored: NonNegativeInt
    reindex_required: StrictBool = False
    reindex_requested_at: str | None = None
    auth: dict[str, Any] | None = None
    message: str | None = None
    error: str | None = None

    @field_validator("reindex_requested_at")
    @classmethod
    def validate_reindex_requested_at(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("reindex_requested_at must include a timezone")
        return parsed.isoformat()

    @model_validator(mode="after")
    def validate_response_contract(self) -> "ChunkUploadResponse":
        # One-directional: a demanded reindex must say which request it is.
        # The converse is a legitimate registry state, not an error - once a
        # client acknowledges a request the server still reports that request's
        # identity so the client can recognize it as already handled.
        if self.reindex_required and self.reindex_requested_at is None:
            raise ValueError("reindex_required responses must carry reindex_requested_at")
        if self.status == "ok" and self.error is not None:
            raise ValueError("successful responses cannot include an error")
        if self.status == "error" and not self.error:
            raise ValueError("error responses must include an error")
        return self


class SearchRequest(BaseModel):
    """Request for semantic search."""

    query: ShortText
    limit: int = Field(default=5, ge=1, le=100)
    project_filter: str | None = Field(default=None, max_length=4096)
    date_from: str | None = Field(default=None, max_length=64)
    date_to: str | None = Field(default=None, max_length=64)
    use_hybrid: bool = True
    enable_analysis: bool = True
    enable_synthesis: bool = False
    include_debug: bool = False


class SearchResponse(BaseModel):
    """Response from semantic search."""

    results: list[dict[str, Any]]
    count: int
    query: str
    search_type: str
    cache_hit: bool = False
    error: str | None = None
    analysis: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    synthesis: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None


class FileSearchRequest(BaseModel):
    """Request for file change search."""

    file_path: str | None = Field(default=None, max_length=4096)
    query: str | None = Field(default=None, max_length=512)
    project_filter: str | None = Field(default=None, max_length=4096)
    operation_filter: str | None = Field(default=None, max_length=128)
    date_from: str | None = Field(default=None, max_length=64)
    date_to: str | None = Field(default=None, max_length=64)
    limit: int = Field(default=10, ge=1, le=100)


class FileSearchResponse(BaseModel):
    """Response from file change search."""

    results: list[dict[str, Any]]
    count: int
    file_path_filter: str | None = None
    operation_filter: str | None = None
    error: str | None = None


class SessionSummaryRequest(BaseModel):
    """Request for session summary."""

    session_id: str | None = Field(default=None, max_length=256)
    project_filter: str | None = Field(default=None, max_length=4096)
    count: int = Field(default=1, ge=1, le=50)


class SessionSummaryResponse(BaseModel):
    """Response with session summaries."""

    summaries: list[dict[str, Any]]
    count: int
    error: str | None = None


class PositionSyncRequest(BaseModel):
    """Request to sync file positions for a machine."""

    machine_id: MachineId
    client_name: str | None = Field(default=None, max_length=256)
    file_path: PathText
    position: int = Field(ge=0)


class PositionSyncResponse(BaseModel):
    """Response after syncing positions."""

    status: str
    machine_id: str
    file_path: str
    position: int
    auth: dict[str, Any] | None = None
    error: str | None = None


class ClientHeartbeatRequest(BaseModel):
    """Client heartbeat payload for status and diagnostics."""

    machine_id: MachineId
    client_name: str | None = Field(default=None, max_length=256)
    client_version: str | None = Field(default=None, max_length=128)
    os: str | None = Field(default=None, max_length=128)
    arch: str | None = Field(default=None, max_length=64)
    python_version: str | None = Field(default=None, max_length=64)
    hostname: str | None = Field(default=None, max_length=256)
    timezone: str | None = Field(default=None, max_length=128)
    heartbeat_interval_s: int | None = None
    status: Literal["ok", "degraded"] | None = None
    last_upload_at: datetime | None = None
    last_indexed_at: datetime | None = None
    queue: DiagnosticMap | None = None
    watcher: DiagnosticMap | None = None
    reindex: DiagnosticMap | None = None
    errors: DiagnosticMap | None = None
    config: DiagnosticMap | None = None
    doctor: DiagnosticMap | None = None
    resources: DiagnosticMap | None = None
    sent_at: datetime | None = None


class ClientHeartbeatResponse(BaseModel):
    """Response after recording a client heartbeat."""

    status: str
    message: str | None = None
    auth: dict[str, Any] | None = None
    error: str | None = None


class GetPositionsRequest(BaseModel):
    """Request to get all positions for a machine."""

    machine_id: MachineId


class GetPositionsResponse(BaseModel):
    """Response with all positions for a machine."""

    model_config = ConfigDict(extra="forbid")

    machine_id: MachineId
    positions: dict[str, NonNegativeInt]  # file_path -> line_number
    reindex_required: StrictBool = False
    reindex_requested_at: str | None = None
    auth: dict[str, Any] | None = None
    error: str | None = None

    @field_validator("reindex_requested_at")
    @classmethod
    def validate_reindex_requested_at(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("reindex_requested_at must include a timezone")
        return parsed.isoformat()

    @model_validator(mode="after")
    def validate_response_contract(self) -> "GetPositionsResponse":
        # One-directional for the same reason as the upload response: an
        # acknowledged request keeps reporting its identity with required=False.
        if self.reindex_required and self.reindex_requested_at is None:
            raise ValueError("reindex_required responses must carry reindex_requested_at")
        return self


class ReindexAckRequest(BaseModel):
    """Client acknowledgement for a server reindex request."""

    machine_id: MachineId
    client_name: str | None = Field(default=None, max_length=256)
    reindex_requested_at: str
    status: Literal["queued", "completed"] = "queued"
    reason: str | None = Field(default=None, max_length=512)

    @field_validator("reindex_requested_at")
    @classmethod
    def validate_reindex_requested_at(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("reindex_requested_at must include a timezone")
        return parsed.isoformat()


class AuthRotateAckRequest(BaseModel):
    """Client acknowledgement for key rotation."""

    machine_id: MachineId
    client_name: str | None = Field(default=None, max_length=256)
    rotate_id: str | None = Field(default=None, max_length=128)


class ReindexAckResponse(BaseModel):
    """Response after recording a reindex acknowledgement."""

    status: str
    machine_id: str
    reindex_requested_at: str | None = None
    auth: dict[str, Any] | None = None
    message: str | None = None
    error: str | None = None


class PurgeClientRequest(BaseModel):
    """Request to purge a single client's data."""

    machine_id: MachineId
    reason: str | None = Field(default=None, max_length=512)


class PurgeClientResponse(BaseModel):
    """Response after purging a single client."""

    status: str
    machine_id: str
    chunks_deleted: int = 0
    auth: dict[str, Any] | None = None
    message: str | None = None
    error: str | None = None
