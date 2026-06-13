"""HTTP client for client mode communication with central server.

Handles chunk uploads, search queries, and position synchronization
with retry logic and offline resilience.
"""

import asyncio
import contextlib
import hashlib
import logging
import secrets
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from claude_history_rag.auth import _read_json, _write_secure_json
from claude_history_rag.config import settings
from claude_history_rag.models import (
    ChunkUploadRequest,
    ChunkUploadResponse,
    ClientHeartbeatRequest,
    ClientHeartbeatResponse,
    FileSearchRequest,
    FileSearchResponse,
    GetPositionsResponse,
    PositionSyncRequest,
    PositionSyncResponse,
    SearchRequest,
    SearchResponse,
    SessionSummaryRequest,
    SessionSummaryResponse,
)

logger = logging.getLogger(__name__)

# HTTP client configuration
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=5.0)


def _redact_url(url: str) -> str:
    """Remove userinfo from URLs before logging."""
    try:
        parts = urlsplit(url)
        hostname = parts.hostname or ""
        netloc = hostname
        if parts.port is not None:
            netloc = f"{netloc}:{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        return "<invalid-url>"


class ServerConnectionError(Exception):
    """Raised when unable to connect to the central server."""

    pass


class APIClient:
    """HTTP client for communicating with central server in client mode."""

    def __init__(
        self,
        server_url: str | None = None,
        machine_id: str | None = None,
        client_name: str | None = None,
        retry_count: int | None = None,
        retry_delay_seconds: int | None = None,
    ):
        self.server_url = (server_url or settings.server_url or "").rstrip("/")
        self.machine_id = machine_id or settings.machine_id
        self.client_name = client_name or settings.client_name or settings.machine_id
        self.retry_count = retry_count or settings.upload_retry_count
        self.retry_delay_seconds = retry_delay_seconds or settings.upload_retry_delay_seconds

        if not self.server_url:
            raise ValueError("server_url is required for client mode")

        self._client: httpx.AsyncClient | None = None
        self._connected = False
        self._last_connection_attempt: datetime | None = None
        self._connection_failures = 0
        self._auth_cache: dict[str, Any] | None = None
        self._redacted_server_url = _redact_url(self.server_url)

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure async HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=HTTP_TIMEOUT,
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                    keepalive_expiry=30.0,
                ),
            )
        return self._client

    def _load_client_auth(self) -> dict[str, Any]:
        if self._auth_cache is not None:
            return self._auth_cache
        data = _read_json(settings.client_auth_path)
        if data is None:
            data = {}
        self._auth_cache = data
        return data

    def _save_client_auth(self, data: dict[str, Any]) -> None:
        self._auth_cache = data
        _write_secure_json(settings.client_auth_path, data)

    def _get_psk(self) -> str | None:
        if settings.client_psk:
            return settings.client_psk
        data = self._load_client_auth()
        return data.get("psk")

    def _set_psk(self, psk: str) -> None:
        data = self._load_client_auth()
        data["psk"] = psk
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_client_auth(data)

    def _get_or_create_client_secret(self) -> str:
        data = self._load_client_auth()
        secrets_by_machine = data.setdefault("client_secrets", {})
        secret = secrets_by_machine.get(self.machine_id)
        if not secret:
            secret = secrets.token_urlsafe(32)
            secrets_by_machine[self.machine_id] = secret
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._save_client_auth(data)
        return secret

    def _client_identity_hash(self) -> str:
        secret = self._get_or_create_client_secret()
        return hashlib.sha256(f"{self.machine_id}\x00{secret}".encode()).hexdigest()

    def _auth_headers(self) -> dict[str, str]:
        psk = self._get_psk()
        if not psk:
            return {}
        return {
            "Authorization": f"Bearer {psk}",
            "X-Client-Identity": self._client_identity_hash(),
        }

    async def _maybe_rotate_psk(
        self,
        auth_payload: dict[str, Any] | None,
    ) -> None:
        if not auth_payload or "rotate_to" not in auth_payload:
            return
        rotate_to = auth_payload.get("rotate_to")
        rotate_id = auth_payload.get("rotate_id")
        if not rotate_to:
            return
        old_psk = self._get_psk()
        if not old_psk:
            return
        # Immediate retry using new key (ack only)
        self._set_psk(rotate_to)
        try:
            await self._request_with_retry(
                "POST",
                "/api/auth/rotation-ack",
                json_data={
                    "machine_id": self.machine_id,
                    "client_name": self.client_name,
                    "rotate_id": rotate_id,
                },
                allow_rotate=False,
            )
        except Exception as e:
            # Rotation failed, fall back and report error
            self._set_psk(old_psk)
            with contextlib.suppress(Exception):
                await self._request_with_retry(
                    "POST",
                    "/api/auth/rotation-error",
                    json_data={
                        "machine_id": self.machine_id,
                        "client_name": self.client_name,
                        "error": f"rotation_failed: {type(e).__name__}",
                    },
                    allow_rotate=False,
                )

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        allow_rotate: bool = True,
    ) -> dict[str, Any]:
        """Make HTTP request with retry logic.

        Retry strategy per requirements:
        - Try up to retry_count times
        - Wait retry_delay_seconds between retries
        - On failure, raise ServerConnectionError for caller to handle

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/api/chunks")
            json_data: JSON body for POST requests

        Returns:
            Response JSON as dict

        Raises:
            ServerConnectionError: If all retries fail
        """
        client = await self._ensure_client()
        url = f"{self.server_url}{endpoint}"

        last_error: Exception | None = None

        for attempt in range(self.retry_count):
            try:
                headers = self._auth_headers()
                if method.upper() == "GET":
                    response = await client.get(url, params=params, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(url, json=json_data, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()

                # Success - reset failure tracking
                self._connected = True
                self._connection_failures = 0

                payload = response.json()
                if allow_rotate:
                    await self._maybe_rotate_psk(payload.get("auth"))
                return payload

            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                last_error = e
                self._connected = False
                self._connection_failures += 1
                self._last_connection_attempt = datetime.now(timezone.utc)

                if attempt < self.retry_count - 1:
                    logger.warning(
                        f"Connection to {self._redacted_server_url} failed (attempt {attempt + 1}/{self.retry_count}), "
                        f"retrying in {self.retry_delay_seconds}s: {type(e).__name__}"
                    )
                    await asyncio.sleep(self.retry_delay_seconds)
                else:
                    logger.error(
                        f"Connection to {self._redacted_server_url} failed after {self.retry_count} attempts"
                    )

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.error(f"HTTP error from server: {e.response.status_code}")
                # Don't retry on HTTP errors (server is reachable but request failed)
                self._connected = True
                raise ServerConnectionError(
                    f"Server returned error: {e.response.status_code}"
                ) from e

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {type(e).__name__}: {e}")
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(self.retry_delay_seconds)

        raise ServerConnectionError(
            f"Failed to connect to server after {self.retry_count} attempts"
        ) from last_error

    @property
    def is_connected(self) -> bool:
        """Check if we're currently connected to the server."""
        return self._connected

    async def health_check(self) -> bool:
        """Check if server is reachable."""
        try:
            await self._request_with_retry("GET", "/health")
            return True
        except ServerConnectionError:
            return False

    async def upload_chunks(
        self,
        chunks: list[dict[str, Any]],
        source_file: str,
        file_position: int,
    ) -> ChunkUploadResponse:
        """Upload chunks to the central server.

        Args:
            chunks: List of chunk dicts (without vectors)
            source_file: Source file path being indexed
            file_position: Line number reached in file

        Returns:
            ChunkUploadResponse with status

        Raises:
            ServerConnectionError: If unable to connect
        """
        request = ChunkUploadRequest(
            machine_id=self.machine_id,
            client_name=self.client_name,
            chunks=chunks,
            source_file=source_file,
            file_position=file_position,
        )
        logger.debug(
            "POST /api/chunks machine_id=%s client_name=%s file=%s position=%s chunks=%d",
            self.machine_id,
            self.client_name,
            source_file,
            file_position,
            len(chunks),
        )

        response_data = await self._request_with_retry(
            "POST", "/api/chunks", request.model_dump(mode="json")
        )

        return ChunkUploadResponse(**response_data)

    async def search(
        self,
        query: str,
        limit: int = 5,
        project_filter: str | None = None,
        use_hybrid: bool = True,
        enable_analysis: bool = True,
        enable_synthesis: bool = False,
        include_debug: bool = False,
    ) -> SearchResponse:
        """Search conversations on the central server.

        Args:
            query: Search query
            limit: Maximum results
            project_filter: Filter by project path
            use_hybrid: Use hybrid search
            enable_analysis: Enable query analysis
            enable_synthesis: Enable result synthesis
            include_debug: Include debug metrics

        Returns:
            SearchResponse with results

        Raises:
            ServerConnectionError: If unable to connect
        """
        request = SearchRequest(
            query=query,
            limit=limit,
            project_filter=project_filter,
            use_hybrid=use_hybrid,
            enable_analysis=enable_analysis,
            enable_synthesis=enable_synthesis,
            include_debug=include_debug,
        )

        response_data = await self._request_with_retry(
            "POST", "/api/search", request.model_dump(mode="json")
        )

        return SearchResponse(**response_data)

    async def search_files(
        self,
        file_path: str | None = None,
        query: str | None = None,
        project_filter: str | None = None,
        operation_filter: str | None = None,
        limit: int = 10,
    ) -> FileSearchResponse:
        """Search file changes on the central server.

        Args:
            file_path: Filter by file path
            query: Semantic query about changes
            project_filter: Filter by project
            operation_filter: Filter by operation (edit/write)
            limit: Maximum results

        Returns:
            FileSearchResponse with results

        Raises:
            ServerConnectionError: If unable to connect
        """
        request = FileSearchRequest(
            file_path=file_path,
            query=query,
            project_filter=project_filter,
            operation_filter=operation_filter,
            limit=limit,
        )

        response_data = await self._request_with_retry(
            "POST", "/api/search/files", request.model_dump(mode="json")
        )

        return FileSearchResponse(**response_data)

    async def get_session_summary(
        self,
        session_id: str | None = None,
        project_filter: str | None = None,
        count: int = 1,
    ) -> SessionSummaryResponse:
        """Get session summaries from the central server.

        Args:
            session_id: Specific session to retrieve
            project_filter: Filter by project
            count: Number of sessions to summarize

        Returns:
            SessionSummaryResponse with summaries

        Raises:
            ServerConnectionError: If unable to connect
        """
        request = SessionSummaryRequest(
            session_id=session_id,
            project_filter=project_filter,
            count=count,
        )

        response_data = await self._request_with_retry(
            "POST", "/api/sessions", request.model_dump(mode="json")
        )

        return SessionSummaryResponse(**response_data)

    async def get_positions(self) -> GetPositionsResponse:
        """Get all file positions for this machine from server.

        Returns:
            GetPositionsResponse with positions

        Raises:
            ServerConnectionError: If unable to connect
        """
        response_data = await self._request_with_retry(
            "GET",
            f"/api/positions/{self.machine_id}",
            params={"client_name": self.client_name},
        )

        return GetPositionsResponse(**response_data)

    async def ack_reindex(
        self,
        reindex_requested_at: str | None,
        status: str = "queued",
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Acknowledge a server reindex request."""
        from claude_history_rag.models import ReindexAckRequest

        request = ReindexAckRequest(
            machine_id=self.machine_id,
            client_name=self.client_name,
            reindex_requested_at=reindex_requested_at,
            status=status,
            reason=reason,
        )
        return await self._request_with_retry(
            "POST", "/api/reindex-ack", request.model_dump(mode="json")
        )

    async def send_heartbeat(
        self,
        payload: dict[str, Any],
    ) -> ClientHeartbeatResponse:
        """Send a client heartbeat to the central server."""
        request = ClientHeartbeatRequest(
            machine_id=self.machine_id,
            client_name=self.client_name,
            **payload,
        )
        response_data = await self._request_with_retry(
            "POST", "/api/heartbeat", request.model_dump(mode="json")
        )
        return ClientHeartbeatResponse(**response_data)

    async def sync_position(
        self,
        file_path: str,
        position: int,
    ) -> PositionSyncResponse:
        """Update file position on the central server.

        Args:
            file_path: File path being tracked
            position: Line number reached

        Returns:
            PositionSyncResponse with status

        Raises:
            ServerConnectionError: If unable to connect
        """
        request = PositionSyncRequest(
            machine_id=self.machine_id,
            client_name=self.client_name,
            file_path=file_path,
            position=position,
        )

        response_data = await self._request_with_retry(
            "POST", "/api/positions", request.model_dump(mode="json")
        )

        return PositionSyncResponse(**response_data)

    async def get_index_status(self) -> dict[str, Any]:
        """Get index status from the central server.

        Returns:
            Status dict

        Raises:
            ServerConnectionError: If unable to connect
        """
        return await self._request_with_retry("GET", "/status")

    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# Global client instance (lazy initialization)
_api_client: APIClient | None = None


def get_api_client() -> APIClient | None:
    """Get the global API client instance if in client mode.

    Returns None if not in client mode (no server_url configured).
    """
    global _api_client
    if not settings.is_client_mode:
        return None
    if _api_client is None:
        _api_client = APIClient()
    return _api_client
