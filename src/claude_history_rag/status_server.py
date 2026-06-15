"""HTTP status server for monitoring, dashboard, and API endpoints.

In server mode, this provides:
- Dashboard and status endpoints for monitoring
- API endpoints for client mode MCP instances to upload chunks and query

In client mode, MCP instances connect to this server's API endpoints.
"""

import hashlib
import hmac
import json
import logging
import re
import time
import traceback
from pathlib import Path
from typing import Any

from aiohttp import web
from pydantic import ValidationError

from claude_history_rag.auth import AuthCheckResult, get_auth_manager
from claude_history_rag.client_registry import get_client_registry
from claude_history_rag.config import settings
from claude_history_rag.models import (
    AuthRotateAckRequest,
    ChunkUploadRequest,
    ChunkUploadResponse,
    ClientHeartbeatRequest,
    ClientHeartbeatResponse,
    FileSearchRequest,
    FileSearchResponse,
    GetPositionsResponse,
    PositionSyncRequest,
    PositionSyncResponse,
    PurgeClientRequest,
    PurgeClientResponse,
    ReindexAckRequest,
    ReindexAckResponse,
    SearchRequest,
    SearchResponse,
    SessionSummaryRequest,
    SessionSummaryResponse,
)
from claude_history_rag.status import get_status_collector
from claude_history_rag.time_filters import parse_timeframe

logger = logging.getLogger(__name__)

# Server-side state for tracking per-machine file positions
# Structure: {machine_id: {file_path: line_number}}
_machine_positions: dict[str, dict[str, int]] = {}
_MACHINE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def _clear_machine_positions(machine_id: str | None = None) -> None:
    """Clear tracked upload positions globally or for one machine."""
    if machine_id is None:
        _machine_positions.clear()
    else:
        _machine_positions.pop(machine_id, None)


def _server_chunk_id(machine_id: str, client_chunk_id: Any) -> str:
    """Derive a server-owned chunk id scoped to one machine."""
    return hashlib.sha256(f"{machine_id}\x00{client_chunk_id or ''}".encode()).hexdigest()


def _client_chunk_identity(chunk: dict[str, Any]) -> str:
    """Return a stable client chunk identity, even for older clients missing id."""
    chunk_id = chunk.get("id")
    if chunk_id:
        return str(chunk_id)
    fallback = {
        "source_file": chunk.get("source_file") or "",
        "source_line": chunk.get("source_line") or 0,
        "chunk_type": chunk.get("chunk_type") or "",
        "session_id": chunk.get("session_id") or "",
        "content": chunk.get("content") or "",
    }
    return hashlib.sha256(json.dumps(fallback, sort_keys=True).encode()).hexdigest()


def _consume_search_type_marker(
    results: list[dict[str, Any]], requested_search_type: str
) -> tuple[str, list[dict[str, Any]]]:
    """Remove internal fallback markers and return the actual search type."""
    actual_search_type = requested_search_type
    cleaned_results: list[dict[str, Any]] = []
    for result in results:
        marker = result.pop("_search_type", None)
        if marker:
            actual_search_type = str(marker)
        cleaned_results.append(result)
    return actual_search_type, cleaned_results


def _validate_upload_chunks(chunks: list[dict[str, Any]]) -> str | None:
    """Return an error string if uploaded chunks miss required storage fields."""
    required_fields = (
        "content",
        "chunk_type",
        "session_id",
        "project_path",
        "project_name",
        "source_file",
    )
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            return f"chunks[{index}] must be an object"
        for field in required_fields:
            value = chunk.get(field)
            if not isinstance(value, str) or not value:
                return f"chunks[{index}].{field} is required"
        source_line = chunk.get("source_line", 0)
        if not isinstance(source_line, int) or source_line < 0:
            return f"chunks[{index}].source_line must be a non-negative integer"
        child_chunk_ids = chunk.get("child_chunk_ids", [])
        if child_chunk_ids is not None and not isinstance(child_chunk_ids, list):
            return f"chunks[{index}].child_chunk_ids must be a list"
    return None


def _scope_uploaded_chunk_ids(machine_id: str, chunks: list[dict[str, Any]]) -> None:
    """Rewrite chunk ids and intra-batch references into the server id namespace."""
    id_map: dict[str, str] = {}
    for chunk in chunks:
        client_identity = _client_chunk_identity(chunk)
        id_map[client_identity] = _server_chunk_id(machine_id, client_identity)

    for chunk in chunks:
        client_identity = _client_chunk_identity(chunk)
        chunk["id"] = id_map[client_identity]
        chunk["machine_id"] = machine_id

        parent_chunk_id = chunk.get("parent_chunk_id")
        if parent_chunk_id is not None:
            chunk["parent_chunk_id"] = id_map.get(
                str(parent_chunk_id), _server_chunk_id(machine_id, str(parent_chunk_id))
            )

        child_chunk_ids = chunk.get("child_chunk_ids")
        if isinstance(child_chunk_ids, list):
            chunk["child_chunk_ids"] = [
                id_map.get(str(child_id), _server_chunk_id(machine_id, str(child_id)))
                for child_id in child_chunk_ids
            ]


class StatusServer:
    """HTTP server for status monitoring and dashboard."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None
        self._setup_routes()

    def _setup_routes(self):
        """Set up HTTP routes."""
        # Dashboard and monitoring routes
        self.app.router.add_get("/", self.handle_dashboard)
        self.app.router.add_get("/dashboard", self.handle_dashboard)
        self.app.router.add_get("/status", self.handle_status)
        self.app.router.add_get("/health", self.handle_health)
        self.app.router.add_get("/metrics", self.handle_metrics)
        self.app.router.add_post("/trigger-index", self.handle_trigger_index)
        self.app.router.add_post("/trigger-reindex", self.handle_trigger_reindex)

        # API routes for client/server mode
        self.app.router.add_post("/api/chunks", self.handle_api_chunks)
        self.app.router.add_post("/api/search", self.handle_api_search)
        self.app.router.add_post("/api/search/files", self.handle_api_search_files)
        self.app.router.add_post("/api/sessions", self.handle_api_sessions)
        self.app.router.add_get("/api/positions/{machine_id}", self.handle_api_get_positions)
        self.app.router.add_post("/api/positions", self.handle_api_sync_position)
        self.app.router.add_post("/api/reindex-ack", self.handle_api_reindex_ack)
        self.app.router.add_post("/api/purge-client", self.handle_api_purge_client)
        self.app.router.add_post("/api/heartbeat", self.handle_api_heartbeat)
        self.app.router.add_get("/api/auth/state", self.handle_api_auth_state)
        self.app.router.add_post("/api/auth/rotate", self.handle_api_auth_rotate)
        self.app.router.add_post("/api/auth/allowlist-keep", self.handle_api_auth_allowlist_keep)
        self.app.router.add_post("/api/auth/rotation-error", self.handle_api_auth_rotation_error)
        self.app.router.add_post("/api/auth/rotation-ack", self.handle_api_auth_rotation_ack)
        self.app.router.add_get("/api/auth/dashboard-hash", self.handle_api_auth_dashboard_hash)
        self.app.router.add_post("/api/auth/key", self.handle_api_auth_key)

    def _get_bearer_token(self, request: web.Request) -> str | None:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            return None
        if not auth_header.lower().startswith("bearer "):
            return None
        return auth_header.split(" ", 1)[1].strip()

    def _get_client_identity(self, request: web.Request) -> str | None:
        identity = request.headers.get("X-Client-Identity", "").strip()
        if not identity:
            return None
        if len(identity) > 128 or not all(c in "0123456789abcdefABCDEF" for c in identity):
            return None
        return identity.lower()

    async def _require_auth(
        self,
        request: web.Request,
        machine_id: str | None = None,
        client_name: str | None = None,
        require_client_identity: bool = False,
    ) -> AuthCheckResult | web.Response:
        if machine_id and not _MACHINE_ID_RE.fullmatch(machine_id):
            return web.json_response({"error": "invalid_machine_id"}, status=400)
        auth_manager = get_auth_manager()
        registry = get_client_registry()

        removed = auth_manager.cleanup_allowlist()
        for removed_id in removed:
            registry.set_key_old(removed_id)

        if not auth_manager.auth_enabled():
            return AuthCheckResult(ok=True, key_type="active")

        raw_key = self._get_bearer_token(request)
        if not raw_key:
            return web.json_response({"error": "missing_auth"}, status=401)

        result = auth_manager.validate_key(raw_key, machine_id, registry)
        if not result.ok:
            return web.json_response({"error": result.error or "unauthorized"}, status=403)

        if machine_id:
            client_identity = self._get_client_identity(request)
            if require_client_identity and not client_identity:
                return web.json_response({"error": "missing_client_identity"}, status=403)
            stored_identity = registry.get_client_identity_hash(machine_id)
            if stored_identity:
                if not client_identity or not hmac.compare_digest(client_identity, stored_identity):
                    return web.json_response({"error": "invalid_client_identity"}, status=403)
            elif client_identity:
                registry.set_client_identity_hash(machine_id, client_identity)
            registry.register_client(machine_id, client_name=client_name)
            if result.key_type == "pending":
                registry.mark_key_rotated(machine_id, result.rotate_id)
            elif result.rotation_required:
                registry.set_rotation_awaiting(machine_id)

        return result

    def _auth_payload(self, result: AuthCheckResult | None) -> dict[str, Any] | None:
        if not result:
            return None
        if result.rotation_required and result.rotate_to:
            return {"rotate_to": result.rotate_to, "rotate_id": result.rotate_id}
        return None

    async def handle_dashboard(self, request: web.Request) -> web.Response:
        """Serve HTML dashboard."""
        try:
            # Read dashboard HTML template
            template_path = Path(__file__).parent / "templates" / "dashboard.html"
            if not template_path.exists():
                return web.Response(
                    text="Dashboard template not found",
                    status=404,
                    content_type="text/html",
                )

            html_content = template_path.read_text()

            # Replace template variables
            html_content = html_content.replace(
                "{{REFRESH_INTERVAL}}", str(settings.status_refresh_interval * 1000)
            )
            html_content = html_content.replace("{{PORT}}", str(self.port))

            return web.Response(text=html_content, content_type="text/html")
        except Exception as e:
            logger.error(f"Failed to serve dashboard: {e}", exc_info=True)
            return web.Response(
                text=f"Error loading dashboard: {type(e).__name__}",
                status=500,
                content_type="text/html",
            )

    async def handle_status(self, request: web.Request) -> web.Response:
        """Return JSON status."""
        start = time.monotonic()
        try:
            auth_result = await self._require_auth(request)
            if isinstance(auth_result, web.Response):
                return auth_result

            # Check for detail level
            detail = request.query.get("detail", "full")
            if detail not in ["basic", "full"]:
                detail = "full"

            # Check for prometheus format
            format_type = request.query.get("format", "json")

            collector = await get_status_collector()
            status = await collector.collect_status(detail_level=detail)

            if format_type == "prometheus":
                # Convert to Prometheus format
                metrics = self._convert_to_prometheus(status)
                logger.debug(
                    "Status collected (prometheus) in %.2fms", (time.monotonic() - start) * 1000
                )
                return web.Response(text=metrics, content_type="text/plain")

            logger.debug(
                "Status collected (detail=%s) in %.2fms",
                detail,
                (time.monotonic() - start) * 1000,
            )
            return web.json_response(status)
        except Exception as e:
            logger.error(f"Failed to collect status: {e}", exc_info=True)
            return web.json_response(
                {"error": f"Status collection failed: {type(e).__name__}"}, status=500
            )

    async def handle_health(self, request: web.Request) -> web.Response:
        """Simple health check endpoint."""
        try:
            auth_result = await self._require_auth(request)
            if isinstance(auth_result, web.Response):
                return auth_result

            collector = await get_status_collector()
            status = await collector.collect_status(detail_level="basic")

            health_status = status.get("health", {}).get("status", "unknown")

            if health_status == "healthy":
                return web.json_response({"status": "healthy"}, status=200)
            elif health_status == "degraded":
                return web.json_response({"status": "degraded"}, status=200)
            else:
                return web.json_response({"status": "unhealthy"}, status=503)
        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return web.json_response(
                {"status": "error", "error": f"Health check failed: {type(e).__name__}"},
                status=503,
            )

    async def handle_metrics(self, request: web.Request) -> web.Response:
        """Prometheus metrics endpoint."""
        try:
            auth_result = await self._require_auth(request)
            if isinstance(auth_result, web.Response):
                return auth_result

            collector = await get_status_collector()
            status = await collector.collect_status(detail_level="full")

            metrics = self._convert_to_prometheus(status)
            return web.Response(text=metrics, content_type="text/plain; version=0.0.4")
        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}", exc_info=True)
            return web.Response(text=f"# Error: metrics_failed:{type(e).__name__}\n", status=500)

    async def handle_trigger_index(self, request: web.Request) -> web.Response:
        """Trigger full indexing of all unindexed files."""
        try:
            auth_result = await self._require_auth(request)
            if isinstance(auth_result, web.Response):
                return auth_result

            from claude_history_rag.watcher import get_all_watchers

            queued_count = 0
            for watcher in get_all_watchers():
                queued_count += await watcher.queue_all_files_for_indexing()

            logger.info(f"Triggered full indexing: {queued_count} files queued")

            return web.json_response(
                {
                    "status": "ok",
                    "message": f"Queued {queued_count} files for indexing",
                    "queued_files": queued_count,
                }
            )
        except Exception as e:
            logger.error(f"Failed to trigger indexing: {e}", exc_info=True)
            return web.json_response(
                {
                    "status": "error",
                    "error": f"Index trigger failed: {type(e).__name__}",
                },
                status=500,
            )

    async def handle_trigger_reindex(self, request: web.Request) -> web.Response:
        """Force full re-index: clear positions, clear database, and re-index all files."""
        try:
            auth_result = await self._require_auth(request)
            if isinstance(auth_result, web.Response):
                return auth_result

            from claude_history_rag.store import store
            from claude_history_rag.watcher import get_all_watchers

            watchers = get_all_watchers()
            registry = get_client_registry()

            # Clear the database first (removes all embeddings)
            chunks_deleted = await store.clear_all_async()
            logger.info(f"Cleared {chunks_deleted} chunks from database")
            _clear_machine_positions()

            # Force full re-index (resets positions and queues files)
            files_reset = 0
            files_queued = 0
            for watcher in watchers:
                reset_count, queued_count = await watcher.force_full_reindex()
                files_reset += reset_count
                files_queued += queued_count
            reindex_requested_at = registry.mark_reindex_requested()

            logger.info(
                f"Triggered force re-index: cleared {chunks_deleted} chunks, "
                f"reset {files_reset} positions, queued {files_queued} files"
            )

            return web.json_response(
                {
                    "status": "ok",
                    "message": f"Cleared {chunks_deleted} chunks, queued {files_queued} files for re-indexing",
                    "chunks_deleted": chunks_deleted,
                    "files_reset": files_reset,
                    "files_queued": files_queued,
                    "reindex_requested_at": reindex_requested_at,
                }
            )
        except Exception as e:
            logger.error(f"Failed to trigger re-index: {e}", exc_info=True)
            return web.json_response(
                {
                    "status": "error",
                    "error": f"Reindex failed: {type(e).__name__}",
                },
                status=500,
            )

    # ============================================================
    # API Endpoints for Client/Server Mode
    # ============================================================

    async def handle_api_chunks(self, request: web.Request) -> web.Response:
        """Handle chunk upload from remote clients.

        Receives chunks without vectors, embeds them, and stores in LanceDB.
        """
        start = time.monotonic()
        content_length = request.headers.get("Content-Length")
        try:
            data = await request.json()
            upload_request = ChunkUploadRequest(**data)
            auth_result = await self._require_auth(
                request,
                machine_id=upload_request.machine_id,
                client_name=upload_request.client_name or upload_request.machine_id,
                require_client_identity=True,
            )
            if isinstance(auth_result, web.Response):
                return auth_result
            registry = get_client_registry()
            registry.register_client(
                upload_request.machine_id,
                client_name=upload_request.client_name or upload_request.machine_id,
            )
            validation_error = _validate_upload_chunks(upload_request.chunks)
            if validation_error:
                response = ChunkUploadResponse(
                    status="error",
                    chunks_received=len(upload_request.chunks),
                    chunks_embedded=0,
                    chunks_stored=0,
                    auth=self._auth_payload(auth_result),
                    error=validation_error,
                )
                return web.json_response(response.model_dump(), status=400)

            logger.info(
                f"Received {len(upload_request.chunks)} chunks from machine "
                f"'{upload_request.machine_id}' for file '{upload_request.source_file}'"
            )
            logger.debug(
                "Chunk upload meta: machine_id=%s client_name=%s file=%s position=%s content_length=%s",
                upload_request.machine_id,
                upload_request.client_name or upload_request.machine_id,
                upload_request.source_file,
                upload_request.file_position,
                content_length,
            )

            # Import here to avoid circular imports and allow client-only installs
            from claude_history_rag.store import store

            _scope_uploaded_chunk_ids(upload_request.machine_id, upload_request.chunks)

            if (
                settings.storage_backend == "spanner"
                and settings.spanner_embedding_mode == "spanner"
            ):
                embedded_chunks = upload_request.chunks
            else:
                from claude_history_rag.embedder import get_embedder

                embedder = get_embedder()
                embedded_chunks = await embedder.embed_chunks(upload_request.chunks)

            if not embedded_chunks:
                reindex_required, reindex_requested_at = registry.get_reindex_status(
                    upload_request.machine_id
                )
                response = ChunkUploadResponse(
                    status="ok",
                    chunks_received=len(upload_request.chunks),
                    chunks_embedded=0,
                    chunks_stored=0,
                    reindex_required=reindex_required,
                    reindex_requested_at=reindex_requested_at,
                    auth=self._auth_payload(auth_result),
                    message="No chunks were successfully embedded",
                )
                return web.json_response(response.model_dump())

            # Store the embedded chunks
            await store.add_chunks_async(embedded_chunks)
            registry.record_upload(
                upload_request.machine_id,
                client_name=upload_request.client_name or upload_request.machine_id,
            )
            logger.info(
                "Stored %d chunks for machine %s (embed_time=%.2fms)",
                len(embedded_chunks),
                upload_request.machine_id,
                (time.monotonic() - start) * 1000,
            )

            # Update the machine's file position
            global _machine_positions
            if upload_request.machine_id not in _machine_positions:
                _machine_positions[upload_request.machine_id] = {}
            _machine_positions[upload_request.machine_id][upload_request.source_file] = (
                upload_request.file_position
            )

            reindex_required, reindex_requested_at = registry.get_reindex_status(
                upload_request.machine_id
            )

            response = ChunkUploadResponse(
                status="ok",
                chunks_received=len(upload_request.chunks),
                chunks_embedded=len(embedded_chunks),
                chunks_stored=len(embedded_chunks),
                reindex_required=reindex_required,
                reindex_requested_at=reindex_requested_at,
                auth=self._auth_payload(auth_result),
                message=f"Successfully stored {len(embedded_chunks)} chunks",
            )
            return web.json_response(response.model_dump())

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in chunk upload: {e}")
            response = ChunkUploadResponse(
                status="error",
                chunks_received=0,
                chunks_embedded=0,
                chunks_stored=0,
                auth=None,
                error=f"Invalid JSON: {e}",
            )
            return web.json_response(response.model_dump(), status=400)
        except ValidationError as e:
            response = ChunkUploadResponse(
                status="error",
                chunks_received=0,
                chunks_embedded=0,
                chunks_stored=0,
                auth=None,
                error=f"Invalid request: {e.errors()[0].get('loc', ['request'])[-1]}",
            )
            return web.json_response(response.model_dump(), status=400)
        except web.HTTPRequestEntityTooLarge as e:
            logger.error(
                "Chunk upload too large: content_length=%s error=%s",
                content_length,
                e,
            )
            response = ChunkUploadResponse(
                status="error",
                chunks_received=0,
                chunks_embedded=0,
                chunks_stored=0,
                auth=None,
                error="Request too large",
            )
            return web.json_response(response.model_dump(), status=413)
        except Exception as e:
            logger.error(f"Chunk upload failed: {type(e).__name__}: {e}", exc_info=True)
            response = ChunkUploadResponse(
                status="error",
                chunks_received=0,
                chunks_embedded=0,
                chunks_stored=0,
                auth=None,
                error=f"Chunk upload failed: {type(e).__name__}",
            )
            return web.json_response(response.model_dump(), status=500)

    async def handle_api_search(self, request: web.Request) -> web.Response:
        """Handle semantic search from remote clients."""
        try:
            auth_result = await self._require_auth(request)
            if isinstance(auth_result, web.Response):
                return auth_result
            data = await request.json()
            search_request = SearchRequest(**data)
            date_from, date_to = parse_timeframe(search_request.date_from, search_request.date_to)

            # Import here to avoid circular imports
            from claude_history_rag.store import store

            if (
                settings.storage_backend == "spanner"
                and settings.spanner_embedding_mode == "spanner"
                and hasattr(store, "embed_query_text_async")
            ):
                query_vector = await store.embed_query_text_async(search_request.query)
            else:
                from claude_history_rag.embedder import get_embedder

                embedder = get_embedder()
                query_vector = await embedder.embed_query(search_request.query)

            if search_request.use_hybrid:
                results = await store.hybrid_search_async(
                    query=search_request.query,
                    query_vector=query_vector,
                    limit=search_request.limit,
                    project_filter=search_request.project_filter,
                    date_from=date_from,
                    date_to=date_to,
                )
                search_type = "hybrid"
            else:
                results = await store.search_async(
                    query_vector=query_vector,
                    limit=search_request.limit,
                    project_filter=search_request.project_filter,
                    date_from=date_from,
                    date_to=date_to,
                )
                search_type = "vector"

            search_type, results = _consume_search_type_marker(results, search_type)
            response = SearchResponse(
                results=results,
                count=len(results),
                query=search_request.query,
                search_type=search_type,
            )
            return web.json_response(response.model_dump())

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in search request: {e}")
            response = SearchResponse(
                results=[],
                count=0,
                query="",
                search_type="error",
                error=f"Invalid JSON: {e}",
            )
            return web.json_response(response.model_dump(), status=400)
        except ValidationError as e:
            response = SearchResponse(
                results=[],
                count=0,
                query=data.get("query", "") if "data" in dir() else "",
                search_type="error",
                error=f"Invalid request: {e.errors()[0].get('loc', ['request'])[-1]}",
            )
            return web.json_response(response.model_dump(), status=400)
        except ValueError as e:
            response = SearchResponse(
                results=[],
                count=0,
                query=data.get("query", "") if "data" in dir() else "",
                search_type="error",
                error=str(e),
            )
            return web.json_response(response.model_dump(), status=400)
        except Exception as e:
            logger.error(f"Search failed: {type(e).__name__}: {e}", exc_info=True)
            response = SearchResponse(
                results=[],
                count=0,
                query=data.get("query", "") if "data" in dir() else "",
                search_type="error",
                error=f"Search failed: {type(e).__name__}",
            )
            return web.json_response(response.model_dump(), status=500)

    async def handle_api_search_files(self, request: web.Request) -> web.Response:
        """Handle file change search from remote clients."""
        try:
            auth_result = await self._require_auth(request)
            if isinstance(auth_result, web.Response):
                return auth_result
            data = await request.json()
            search_request = FileSearchRequest(**data)
            date_from, date_to = parse_timeframe(search_request.date_from, search_request.date_to)

            # Import here to avoid circular imports
            from claude_history_rag.store import store

            # Use query if provided, otherwise search by file path pattern
            if search_request.query:
                query_text = search_request.query
            elif search_request.file_path:
                query_text = f"file changes to {search_request.file_path}"
            else:
                response = FileSearchResponse(
                    results=[],
                    count=0,
                    error="Either query or file_path must be provided",
                )
                return web.json_response(response.model_dump(), status=400)

            if (
                settings.storage_backend == "spanner"
                and settings.spanner_embedding_mode == "spanner"
                and hasattr(store, "embed_query_text_async")
            ):
                query_vector = await store.embed_query_text_async(query_text)
            else:
                from claude_history_rag.embedder import get_embedder

                embedder = get_embedder()
                query_vector = await embedder.embed_query(query_text)

            results = await store.search_async(
                query_vector=query_vector,
                limit=search_request.limit,
                project_filter=search_request.project_filter,
                chunk_type_filter="file_change",
                file_path_filter=search_request.file_path,
                operation_filter=search_request.operation_filter,
                date_from=date_from,
                date_to=date_to,
            )

            response = FileSearchResponse(
                results=results,
                count=len(results),
                file_path_filter=search_request.file_path,
                operation_filter=search_request.operation_filter,
            )
            return web.json_response(response.model_dump())

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file search request: {e}")
            response = FileSearchResponse(
                results=[],
                count=0,
                error=f"Invalid JSON: {e}",
            )
            return web.json_response(response.model_dump(), status=400)
        except ValidationError as e:
            response = FileSearchResponse(
                results=[],
                count=0,
                error=f"Invalid request: {e.errors()[0].get('loc', ['request'])[-1]}",
            )
            return web.json_response(response.model_dump(), status=400)
        except ValueError as e:
            response = FileSearchResponse(
                results=[],
                count=0,
                error=str(e),
            )
            return web.json_response(response.model_dump(), status=400)
        except Exception as e:
            logger.error(f"File search failed: {type(e).__name__}: {e}", exc_info=True)
            response = FileSearchResponse(
                results=[],
                count=0,
                error=f"File search failed: {type(e).__name__}",
            )
            return web.json_response(response.model_dump(), status=500)

    async def handle_api_sessions(self, request: web.Request) -> web.Response:
        """Handle session summary request from remote clients."""
        try:
            auth_result = await self._require_auth(request)
            if isinstance(auth_result, web.Response):
                return auth_result
            data = await request.json()
            summary_request = SessionSummaryRequest(**data)

            # Import here to avoid circular imports
            from claude_history_rag.store import store

            # Search for summary chunks
            query = "session summary conversation overview"
            if (
                settings.storage_backend == "spanner"
                and settings.spanner_embedding_mode == "spanner"
                and hasattr(store, "embed_query_text_async")
            ):
                query_vector = await store.embed_query_text_async(query)
            else:
                from claude_history_rag.embedder import get_embedder

                embedder = get_embedder()
                query_vector = await embedder.embed_query(query)

            results = await store.search_async(
                query_vector=query_vector,
                limit=summary_request.count * 3,  # Get more, filter later
                project_filter=summary_request.project_filter,
                chunk_type_filter="summary",
            )

            # Filter by session_id if provided
            if summary_request.session_id:
                results = [r for r in results if r.get("session_id") == summary_request.session_id]

            # Deduplicate by session_id, keeping most recent per session
            seen_sessions = set()
            summaries = []
            for result in results:
                session_id = result.get("session_id")
                if session_id not in seen_sessions:
                    seen_sessions.add(session_id)
                    summaries.append(result)
                    if len(summaries) >= summary_request.count:
                        break

            response = SessionSummaryResponse(
                summaries=summaries,
                count=len(summaries),
            )
            return web.json_response(response.model_dump())

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in session summary request: {e}")
            response = SessionSummaryResponse(
                summaries=[],
                count=0,
                error=f"Invalid JSON: {e}",
            )
            return web.json_response(response.model_dump(), status=400)
        except ValidationError as e:
            response = SessionSummaryResponse(
                summaries=[],
                count=0,
                error=f"Invalid request: {e.errors()[0].get('loc', ['request'])[-1]}",
            )
            return web.json_response(response.model_dump(), status=400)
        except Exception as e:
            logger.error(f"Session summary failed: {type(e).__name__}: {e}", exc_info=True)
            response = SessionSummaryResponse(
                summaries=[],
                count=0,
                error=f"Session summary failed: {type(e).__name__}",
            )
            return web.json_response(response.model_dump(), status=500)

    async def handle_api_get_positions(self, request: web.Request) -> web.Response:
        """Get all file positions for a machine."""
        try:
            machine_id = request.match_info["machine_id"]
            registry = get_client_registry()
            client_name = request.query.get("client_name") or machine_id
            auth_result = await self._require_auth(
                request,
                machine_id=machine_id,
                client_name=client_name,
                require_client_identity=True,
            )
            if isinstance(auth_result, web.Response):
                return auth_result
            registry.register_client(machine_id, client_name=client_name)

            global _machine_positions
            positions = _machine_positions.get(machine_id, {})
            reindex_required, reindex_requested_at = registry.get_reindex_status(machine_id)

            response = GetPositionsResponse(
                machine_id=machine_id,
                positions=positions,
                reindex_required=reindex_required,
                reindex_requested_at=reindex_requested_at,
                auth=self._auth_payload(auth_result),
            )
            return web.json_response(response.model_dump())

        except Exception as e:
            logger.error(f"Get positions failed: {type(e).__name__}: {e}", exc_info=True)
            response = GetPositionsResponse(
                machine_id=request.match_info.get("machine_id", "unknown"),
                positions={},
                error=f"Get positions failed: {type(e).__name__}",
            )
            return web.json_response(response.model_dump(), status=500)

    async def handle_api_sync_position(self, request: web.Request) -> web.Response:
        """Update file position for a machine."""
        try:
            data = await request.json()
            sync_request = PositionSyncRequest(**data)
            registry = get_client_registry()
            auth_result = await self._require_auth(
                request,
                machine_id=sync_request.machine_id,
                client_name=sync_request.client_name or sync_request.machine_id,
                require_client_identity=True,
            )
            if isinstance(auth_result, web.Response):
                return auth_result
            registry.register_client(
                sync_request.machine_id,
                client_name=sync_request.client_name or sync_request.machine_id,
            )

            global _machine_positions
            if sync_request.machine_id not in _machine_positions:
                _machine_positions[sync_request.machine_id] = {}
            _machine_positions[sync_request.machine_id][sync_request.file_path] = (
                sync_request.position
            )

            response = PositionSyncResponse(
                status="ok",
                machine_id=sync_request.machine_id,
                file_path=sync_request.file_path,
                position=sync_request.position,
                auth=self._auth_payload(auth_result),
            )
            return web.json_response(response.model_dump())

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in position sync: {e}")
            response = PositionSyncResponse(
                status="error",
                machine_id="",
                file_path="",
                position=0,
                error=f"Invalid JSON: {e}",
            )
            return web.json_response(response.model_dump(), status=400)
        except ValidationError as e:
            response = PositionSyncResponse(
                status="error",
                machine_id=data.get("machine_id", "") if "data" in dir() else "",
                file_path=data.get("file_path", "") if "data" in dir() else "",
                position=0,
                error=f"Invalid request: {e.errors()[0].get('loc', ['request'])[-1]}",
            )
            return web.json_response(response.model_dump(), status=400)
        except Exception as e:
            logger.error(f"Position sync failed: {type(e).__name__}: {e}", exc_info=True)
            response = PositionSyncResponse(
                status="error",
                machine_id=data.get("machine_id", "") if "data" in dir() else "",
                file_path=data.get("file_path", "") if "data" in dir() else "",
                position=0,
                error=f"Position sync failed: {type(e).__name__}",
            )
            return web.json_response(response.model_dump(), status=500)

    async def handle_api_reindex_ack(self, request: web.Request) -> web.Response:
        """Record a client acknowledgement for a reindex request."""
        try:
            data = await request.json()
            ack_request = ReindexAckRequest(**data)
            registry = get_client_registry()
            auth_result = await self._require_auth(
                request,
                machine_id=ack_request.machine_id,
                client_name=ack_request.client_name or ack_request.machine_id,
                require_client_identity=True,
            )
            if isinstance(auth_result, web.Response):
                return auth_result
            registry.ack_reindex(
                ack_request.machine_id,
                reindex_requested_at=ack_request.reindex_requested_at,
                status=ack_request.status,
                reason=ack_request.reason,
            )
            registry.register_client(
                ack_request.machine_id,
                client_name=ack_request.client_name or ack_request.machine_id,
            )
            logger.info(
                "Reindex ack: machine_id=%s client_name=%s status=%s requested_at=%s reason=%s",
                ack_request.machine_id,
                ack_request.client_name or ack_request.machine_id,
                ack_request.status,
                ack_request.reindex_requested_at,
                ack_request.reason,
            )

            response = ReindexAckResponse(
                status="ok",
                machine_id=ack_request.machine_id,
                reindex_requested_at=ack_request.reindex_requested_at,
                auth=self._auth_payload(auth_result),
                message="Reindex acknowledgement recorded",
            )
            return web.json_response(response.model_dump())
        except json.JSONDecodeError as e:
            response = ReindexAckResponse(
                status="error",
                machine_id="",
                error=f"Invalid JSON: {e}",
            )
            return web.json_response(response.model_dump(), status=400)
        except ValidationError as e:
            response = ReindexAckResponse(
                status="error",
                machine_id=data.get("machine_id", "") if "data" in dir() else "",
                error=f"Invalid request: {e.errors()[0].get('loc', ['request'])[-1]}",
            )
            return web.json_response(response.model_dump(), status=400)
        except Exception as e:
            logger.error(f"Reindex ack failed: {type(e).__name__}: {e}", exc_info=True)
            response = ReindexAckResponse(
                status="error",
                machine_id=data.get("machine_id", "") if "data" in dir() else "",
                error=f"Reindex acknowledgement failed: {type(e).__name__}",
            )
            return web.json_response(response.model_dump(), status=500)

    async def handle_api_purge_client(self, request: web.Request) -> web.Response:
        """Purge all chunks for a given client machine_id."""
        try:
            data = await request.json()
            purge_request = PurgeClientRequest(**data)
            from claude_history_rag.store import store

            registry = get_client_registry()
            auth_result = await self._require_auth(
                request,
                machine_id=purge_request.machine_id,
                client_name=purge_request.machine_id,
                require_client_identity=True,
            )
            if isinstance(auth_result, web.Response):
                return auth_result

            logger.warning(
                "Purging client data: machine_id=%s reason=%s",
                purge_request.machine_id,
                purge_request.reason,
            )
            chunks_deleted = await store.delete_by_machine_id_async(purge_request.machine_id)
            _clear_machine_positions(purge_request.machine_id)
            registry.mark_purged(purge_request.machine_id, client_name=purge_request.machine_id)

            response = PurgeClientResponse(
                status="ok",
                machine_id=purge_request.machine_id,
                chunks_deleted=chunks_deleted,
                auth=self._auth_payload(auth_result),
                message="Client data purged",
            )
            return web.json_response(response.model_dump())
        except json.JSONDecodeError as e:
            response = PurgeClientResponse(
                status="error",
                machine_id="",
                error=f"Invalid JSON: {e}",
            )
            return web.json_response(response.model_dump(), status=400)
        except ValidationError as e:
            response = PurgeClientResponse(
                status="error",
                machine_id=data.get("machine_id", "") if "data" in dir() else "",
                error=f"Invalid request: {e.errors()[0].get('loc', ['request'])[-1]}",
            )
            return web.json_response(response.model_dump(), status=400)
        except Exception as e:
            logger.error(f"Purge client failed: {type(e).__name__}: {e}", exc_info=True)
            response = PurgeClientResponse(
                status="error",
                machine_id=data.get("machine_id", "") if "data" in dir() else "",
                error=f"Purge failed: {type(e).__name__}",
            )
            return web.json_response(response.model_dump(), status=500)

    async def handle_api_heartbeat(self, request: web.Request) -> web.Response:
        """Record a client heartbeat with status details."""
        try:
            data = await request.json()
            heartbeat_request = ClientHeartbeatRequest(**data)
            registry = get_client_registry()
            auth_result = await self._require_auth(
                request,
                machine_id=heartbeat_request.machine_id,
                client_name=heartbeat_request.client_name or heartbeat_request.machine_id,
                require_client_identity=True,
            )
            if isinstance(auth_result, web.Response):
                return auth_result

            payload = heartbeat_request.model_dump(mode="json")
            payload.pop("machine_id", None)
            payload.pop("client_name", None)

            registry.record_heartbeat(
                heartbeat_request.machine_id,
                client_name=heartbeat_request.client_name or heartbeat_request.machine_id,
                heartbeat=payload,
            )

            response = ClientHeartbeatResponse(
                status="ok",
                auth=self._auth_payload(auth_result),
                message="Heartbeat recorded",
            )
            return web.json_response(response.model_dump())
        except json.JSONDecodeError as e:
            response = ClientHeartbeatResponse(status="error", error=f"Invalid JSON: {e}")
            return web.json_response(response.model_dump(), status=400)
        except ValidationError as e:
            response = ClientHeartbeatResponse(
                status="error",
                error=f"Invalid request: {e.errors()[0].get('loc', ['request'])[-1]}",
            )
            return web.json_response(response.model_dump(), status=400)
        except Exception as e:
            logger.error(f"Heartbeat failed: {type(e).__name__}: {e}", exc_info=True)
            response = ClientHeartbeatResponse(
                status="error",
                error=f"Heartbeat failed: {type(e).__name__}",
            )
            return web.json_response(response.model_dump(), status=500)

    async def handle_api_auth_state(self, request: web.Request) -> web.Response:
        """Return auth state for dashboard."""
        try:
            auth_result = await self._require_auth(request)
            if isinstance(auth_result, web.Response):
                return auth_result
            auth_manager = get_auth_manager()
            registry = get_client_registry()
            state = auth_manager.get_rotation_state()

            def scrub(entry: dict[str, Any] | None) -> dict[str, Any] | None:
                if not entry:
                    return None
                return {
                    "key_id": entry.get("key_id"),
                    "created_at": entry.get("created_at"),
                    "allowlist": entry.get("allowlist"),
                    "allowlist_days": entry.get("allowlist_days"),
                    "allowlist_expires_at": entry.get("allowlist_expires_at"),
                }

            response = {
                "auth_enabled": state.get("auth_enabled", True),
                "env_override": state.get("env_override", False),
                "active": scrub(state.get("active")),
                "pending": scrub(state.get("pending")),
                "rotation": state.get("rotation"),
                "clients": registry.get_client_status().get("clients", []),
            }
            return web.json_response(response)
        except Exception as e:
            logger.error("Auth state failed: %s", type(e).__name__, exc_info=True)
            return web.json_response({"error": f"auth_state_failed:{type(e).__name__}"}, status=500)

    async def handle_api_auth_rotate(self, request: web.Request) -> web.Response:
        """Rotate PSK with optional allowlist and expiry."""
        try:
            auth_result = await self._require_auth(request)
            if isinstance(auth_result, web.Response):
                return auth_result
            auth_manager = get_auth_manager()
            if auth_manager.is_env_override():
                return web.json_response({"error": "env_override"}, status=400)
            data = await request.json()
            allowlist = data.get("allowlist") or []
            allow_days = int(data.get("allow_days", 0))
            revoke_old = bool(data.get("revoke_old", False))
            result = auth_manager.rotate_key(allowlist, allow_days, revoke_old)
            registry = get_client_registry()
            for machine_id in allowlist:
                registry.set_rotation_awaiting(machine_id)
            return web.json_response({"status": "ok", **result})
        except json.JSONDecodeError as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)
        except ValidationError as e:
            return web.json_response(
                {"error": f"Invalid request: {e.errors()[0].get('loc', ['request'])[-1]}"},
                status=400,
            )
        except Exception as e:
            logger.error("Auth rotate failed: %s", type(e).__name__, exc_info=True)
            return web.json_response(
                {"error": f"auth_rotate_failed:{type(e).__name__}"}, status=500
            )

    async def handle_api_auth_allowlist_keep(self, request: web.Request) -> web.Response:
        """Keep a client on the allowlist (temporary)."""
        try:
            auth_result = await self._require_auth(request)
            if isinstance(auth_result, web.Response):
                return auth_result
            auth_manager = get_auth_manager()
            data = await request.json()
            machine_id = data.get("machine_id")
            if not machine_id:
                return web.json_response({"error": "missing_machine_id"}, status=400)
            if not auth_manager.keep_on_allowlist(machine_id):
                return web.json_response({"error": "allowlist_expired"}, status=400)
            return web.json_response({"status": "ok"})
        except json.JSONDecodeError as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)
        except Exception as e:
            logger.error("Auth allowlist keep failed: %s", type(e).__name__, exc_info=True)
            return web.json_response(
                {"error": f"auth_allowlist_keep_failed:{type(e).__name__}"}, status=500
            )

    async def handle_api_auth_rotation_error(self, request: web.Request) -> web.Response:
        """Record a client rotation failure."""
        try:
            data = await request.json()
            machine_id = data.get("machine_id")
            client_name = data.get("client_name")
            if not machine_id:
                return web.json_response({"error": "missing_machine_id"}, status=400)
            auth_result = await self._require_auth(
                request,
                machine_id=machine_id,
                client_name=client_name or machine_id,
                require_client_identity=True,
            )
            if isinstance(auth_result, web.Response):
                return auth_result
            registry = get_client_registry()
            registry.record_key_rotation_error(machine_id, data.get("error", "rotation_failed"))
            return web.json_response({"status": "ok"})
        except json.JSONDecodeError as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)
        except Exception as e:
            logger.error("Auth rotation error record failed: %s", type(e).__name__, exc_info=True)
            return web.json_response(
                {"error": f"auth_rotation_error_failed:{type(e).__name__}"}, status=500
            )

    async def handle_api_auth_rotation_ack(self, request: web.Request) -> web.Response:
        """Record a client rotation acknowledgement without replaying payload."""
        try:
            data = await request.json()
            ack_request = AuthRotateAckRequest(**data)
            auth_result = await self._require_auth(
                request,
                machine_id=ack_request.machine_id,
                client_name=ack_request.client_name or ack_request.machine_id,
                require_client_identity=True,
            )
            if isinstance(auth_result, web.Response):
                return auth_result
            registry = get_client_registry()
            registry.mark_key_rotated(ack_request.machine_id, ack_request.rotate_id)
            return web.json_response({"status": "ok"})
        except json.JSONDecodeError as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)
        except ValidationError as e:
            return web.json_response(
                {"error": f"Invalid request: {e.errors()[0].get('loc', ['request'])[-1]}"},
                status=400,
            )
        except Exception as e:
            logger.error("Auth rotation ack failed: %s", type(e).__name__, exc_info=True)
            return web.json_response(
                {"error": f"auth_rotation_ack_failed:{type(e).__name__}"}, status=500
            )

    async def handle_api_auth_dashboard_hash(self, request: web.Request) -> web.Response:
        """Return dashboard hash for local storage (requires auth)."""
        try:
            auth_result = await self._require_auth(request)
            if isinstance(auth_result, web.Response):
                return auth_result
            auth_manager = get_auth_manager()
            return web.json_response({"dashboard_hash": auth_manager.get_dashboard_hash()})
        except Exception as e:
            logger.error("Dashboard hash failed: %s", type(e).__name__, exc_info=True)
            return web.json_response(
                {"error": f"dashboard_hash_failed:{type(e).__name__}"}, status=500
            )

    async def handle_api_auth_key(self, request: web.Request) -> web.Response:
        """Reveal current PSK for dashboard after hash check."""
        try:
            auth_result = await self._require_auth(request)
            if isinstance(auth_result, web.Response):
                return auth_result
            auth_manager = get_auth_manager()
            expected = auth_manager.get_dashboard_hash()
            provided = request.headers.get("X-Auth-Hash")
            if not expected or not provided or not hmac.compare_digest(expected, provided):
                return web.json_response({"error": "invalid_hash"}, status=403)
            key = auth_manager.get_active_key_plain()
            if not key:
                return web.json_response({"error": "key_unavailable"}, status=404)
            return web.json_response({"key": key})
        except Exception as e:
            logger.error("Auth key reveal failed: %s", type(e).__name__, exc_info=True)
            return web.json_response({"error": f"auth_key_failed:{type(e).__name__}"}, status=500)

    def _convert_to_prometheus(self, status: dict[str, Any]) -> str:
        """Convert status JSON to Prometheus text format."""
        lines = []

        # Server uptime
        uptime = status.get("server", {}).get("uptime_seconds", 0)
        lines.append("# HELP mcp_server_uptime_seconds Server uptime in seconds")
        lines.append("# TYPE mcp_server_uptime_seconds gauge")
        lines.append(f"mcp_server_uptime_seconds {uptime}")
        lines.append("")

        # Database chunks
        chunks = status.get("database", {}).get("total_chunks", 0)
        lines.append("# HELP mcp_chunks_total Total chunks indexed")
        lines.append("# TYPE mcp_chunks_total counter")
        lines.append(f"mcp_chunks_total {chunks}")
        lines.append("")

        # Indexing progress
        files_pending = status.get("indexing", {}).get("files_pending", 0)
        lines.append("# HELP mcp_indexing_files_pending Number of files pending indexing")
        lines.append("# TYPE mcp_indexing_files_pending gauge")
        lines.append(f"mcp_indexing_files_pending {files_pending}")
        lines.append("")

        files_failed = status.get("indexing", {}).get("files_failed", 0)
        lines.append("# HELP mcp_indexing_files_failed Number of failed file indexing attempts")
        lines.append("# TYPE mcp_indexing_files_failed gauge")
        lines.append(f"mcp_indexing_files_failed {files_failed}")
        lines.append("")

        # Memory usage
        memory_mb = status.get("performance", {}).get("memory_usage_mb", 0)
        memory_bytes = int(memory_mb * 1024 * 1024)
        lines.append("# HELP mcp_memory_usage_bytes Memory usage in bytes")
        lines.append("# TYPE mcp_memory_usage_bytes gauge")
        lines.append(f"mcp_memory_usage_bytes {memory_bytes}")
        lines.append("")

        # CPU usage
        cpu_percent = status.get("performance", {}).get("cpu_percent", 0)
        lines.append("# HELP mcp_cpu_percent CPU usage percentage")
        lines.append("# TYPE mcp_cpu_percent gauge")
        lines.append(f"mcp_cpu_percent {cpu_percent}")
        lines.append("")

        # Query metrics
        queries_total = status.get("performance", {}).get("queries_total", 0)
        lines.append("# HELP mcp_queries_total Total number of queries processed")
        lines.append("# TYPE mcp_queries_total counter")
        lines.append(f"mcp_queries_total {queries_total}")
        lines.append("")

        avg_latency = status.get("performance", {}).get("avg_query_latency_ms", 0)
        avg_latency_seconds = avg_latency / 1000
        lines.append("# HELP mcp_query_duration_seconds_avg Average query duration in seconds")
        lines.append("# TYPE mcp_query_duration_seconds_avg gauge")
        lines.append(f"mcp_query_duration_seconds_avg {avg_latency_seconds:.6f}")
        lines.append("")

        # Cache metrics
        cache_hits = status.get("cache", {}).get("hits", 0)
        lines.append("# HELP mcp_cache_hits_total Cache hit count")
        lines.append("# TYPE mcp_cache_hits_total counter")
        lines.append(f"mcp_cache_hits_total {cache_hits}")
        lines.append("")

        cache_misses = status.get("cache", {}).get("misses", 0)
        lines.append("# HELP mcp_cache_misses_total Cache miss count")
        lines.append("# TYPE mcp_cache_misses_total counter")
        lines.append(f"mcp_cache_misses_total {cache_misses}")
        lines.append("")

        cache_size = status.get("cache", {}).get("size", 0)
        lines.append("# HELP mcp_cache_size Current cache size")
        lines.append("# TYPE mcp_cache_size gauge")
        lines.append(f"mcp_cache_size {cache_size}")
        lines.append("")

        # Health status (1 = healthy, 0.5 = degraded, 0 = unhealthy)
        health_status = status.get("health", {}).get("status", "unknown")
        health_value = (
            1.0 if health_status == "healthy" else (0.5 if health_status == "degraded" else 0.0)
        )
        lines.append(
            "# HELP mcp_health_status Health status (1=healthy, 0.5=degraded, 0=unhealthy)"
        )
        lines.append("# TYPE mcp_health_status gauge")
        lines.append(f"mcp_health_status {health_value}")
        lines.append("")

        return "\n".join(lines)

    async def start(self):
        """Start the HTTP server on the configured static port."""
        logger.info(f"Starting status server on http://{self.host}:{self.port}...")

        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()

            # Enable SO_REUSEADDR to handle rapid restarts
            self.site = web.TCPSite(self.runner, self.host, self.port, reuse_address=True)
            await self.site.start()

            logger.info(f"Status server started on http://{self.host}:{self.port}")
            logger.info(f"Dashboard: http://{self.host}:{self.port}/dashboard")

        except OSError as e:
            if e.errno == 48:  # Address already in use
                logger.error(
                    f"Port {self.port} is already in use. "
                    f"Set CLAUDE_HISTORY_RAG_STATUS_SERVER_PORT to use a different port."
                )
            raise
        except Exception as e:
            logger.error(f"Failed to start status server: {e}", exc_info=True)
            raise

    async def stop(self):
        """Stop the HTTP server."""
        logger.info("[STATUS_SERVER] Stop called, beginning shutdown...")
        try:
            if self.site:
                logger.info("[STATUS_SERVER] Stopping TCPSite...")
                await self.site.stop()
                logger.info("[STATUS_SERVER] TCPSite stopped")
            if self.runner:
                logger.info("[STATUS_SERVER] Cleaning up AppRunner...")
                await self.runner.cleanup()
                logger.info("[STATUS_SERVER] AppRunner cleaned up")
            logger.info("Status server stopped")
        except Exception as e:
            logger.error(f"[STATUS_SERVER] Error stopping status server: {e}", exc_info=True)


# Global server instance
_status_server: StatusServer | None = None


def get_status_server() -> StatusServer | None:
    """Get the global status server instance if it exists.

    Returns None if the server hasn't been created yet. Use create_status_server() to create it.
    """
    global _status_server
    return _status_server


def create_status_server() -> StatusServer:
    """Create and return the global status server instance."""
    global _status_server
    logger.info(
        f"[STATUS_SERVER] create_status_server() called. Current instance: {_status_server}"
    )
    logger.info(f"[STATUS_SERVER] Call stack:\n{''.join(traceback.format_stack()[-5:-1])}")
    if _status_server is None:
        logger.info("[STATUS_SERVER] Creating new StatusServer instance")
        _status_server = StatusServer(
            host=settings.status_server_host,
            port=settings.status_server_port,
        )
    else:
        logger.info("[STATUS_SERVER] Returning existing StatusServer instance")
    return _status_server
