"""Pydantic models for parsing and chunking."""

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, Field

MachineId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
ShortText = Annotated[str, Field(min_length=1, max_length=512)]
PathText = Annotated[str, Field(min_length=1, max_length=4096)]
DiagnosticMap = Annotated[dict[str, Any], Field(max_length=64)]


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
    parent_chunk_id: str | None = None
    child_chunk_ids: list[str] | None = None
    machine_id: str | None = None  # For multi-machine support


# ============================================================
# API Request/Response Models (Client/Server Communication)
# ============================================================


class ChunkUploadRequest(BaseModel):
    """Request to upload chunks from a client to the server."""

    machine_id: MachineId  # Identifies source machine
    client_name: str | None = Field(default=None, max_length=256)
    chunks: list[dict[str, Any]] = Field(max_length=500)  # Chunks without vectors
    source_file: PathText  # Source file being indexed
    file_position: int = Field(ge=0)  # Line number reached in file


class ChunkUploadResponse(BaseModel):
    """Response after uploading chunks."""

    status: str  # "ok" or "error"
    chunks_received: int
    chunks_embedded: int
    chunks_stored: int
    reindex_required: bool | None = None
    reindex_requested_at: str | None = None
    auth: dict[str, Any] | None = None
    message: str | None = None
    error: str | None = None


class SearchRequest(BaseModel):
    """Request for semantic search."""

    query: ShortText
    limit: int = Field(default=5, ge=1, le=100)
    project_filter: str | None = Field(default=None, max_length=4096)
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
    status: str | None = Field(default=None, max_length=64)
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

    machine_id: str
    positions: dict[str, int]  # file_path -> line_number
    reindex_required: bool | None = None
    reindex_requested_at: str | None = None
    auth: dict[str, Any] | None = None
    error: str | None = None


class ReindexAckRequest(BaseModel):
    """Client acknowledgement for a server reindex request."""

    machine_id: MachineId
    client_name: str | None = Field(default=None, max_length=256)
    reindex_requested_at: str | None = None
    status: str = Field(default="queued", max_length=64)
    reason: str | None = Field(default=None, max_length=512)


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
