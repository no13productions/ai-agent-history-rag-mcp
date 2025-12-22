"""Pydantic models for parsing and chunking."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


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

    type: str  # "user", "assistant", "summary", "system"
    message: UserMessage | AssistantMessage | None = None
    summary: str | None = None  # For summary type
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

    machine_id: str  # Identifies source machine
    client_name: str | None = None
    chunks: list[dict[str, Any]]  # Chunks without vectors (server will embed)
    source_file: str  # Source file being indexed
    file_position: int  # Line number reached in file


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

    query: str
    limit: int = 5
    project_filter: str | None = None
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

    file_path: str | None = None
    query: str | None = None
    project_filter: str | None = None
    operation_filter: str | None = None
    limit: int = 10


class FileSearchResponse(BaseModel):
    """Response from file change search."""

    results: list[dict[str, Any]]
    count: int
    file_path_filter: str | None = None
    operation_filter: str | None = None
    error: str | None = None


class SessionSummaryRequest(BaseModel):
    """Request for session summary."""

    session_id: str | None = None
    project_filter: str | None = None
    count: int = 1


class SessionSummaryResponse(BaseModel):
    """Response with session summaries."""

    summaries: list[dict[str, Any]]
    count: int
    error: str | None = None


class PositionSyncRequest(BaseModel):
    """Request to sync file positions for a machine."""

    machine_id: str
    client_name: str | None = None
    file_path: str
    position: int


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

    machine_id: str
    client_name: str | None = None
    client_version: str | None = None
    os: str | None = None
    arch: str | None = None
    python_version: str | None = None
    hostname: str | None = None
    timezone: str | None = None
    heartbeat_interval_s: int | None = None
    status: str | None = None
    last_upload_at: datetime | None = None
    last_indexed_at: datetime | None = None
    queue: dict[str, Any] | None = None
    watcher: dict[str, Any] | None = None
    reindex: dict[str, Any] | None = None
    errors: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    doctor: dict[str, Any] | None = None
    resources: dict[str, Any] | None = None
    sent_at: datetime | None = None


class ClientHeartbeatResponse(BaseModel):
    """Response after recording a client heartbeat."""

    status: str
    message: str | None = None
    auth: dict[str, Any] | None = None
    error: str | None = None


class GetPositionsRequest(BaseModel):
    """Request to get all positions for a machine."""

    machine_id: str


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

    machine_id: str
    client_name: str | None = None
    reindex_requested_at: str | None = None
    status: str = "queued"
    reason: str | None = None


class AuthRotateAckRequest(BaseModel):
    """Client acknowledgement for key rotation."""

    machine_id: str
    client_name: str | None = None
    rotate_id: str | None = None


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

    machine_id: str
    reason: str | None = None


class PurgeClientResponse(BaseModel):
    """Response after purging a single client."""

    status: str
    machine_id: str
    chunks_deleted: int = 0
    auth: dict[str, Any] | None = None
    message: str | None = None
    error: str | None = None
