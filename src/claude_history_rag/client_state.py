"""Client-side state management for tracking uploads and catch-up logic.

Handles:
- Tracking which files need to be uploaded
- Local position tracking when offline
- Catch-up logic when reconnecting to server
- Pending upload queue
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from claude_history_rag.config import settings

logger = logging.getLogger(__name__)


class PendingUpload(BaseModel):
    """A pending chunk upload waiting to be sent to the server."""

    file_path: str
    chunks: list[dict[str, Any]]
    file_position: int
    created_at: datetime
    retry_count: int = 0


class ClientState(BaseModel):
    """Client-side state for tracking uploads and positions."""

    # Local file positions (what we've chunked locally)
    local_positions: dict[str, int] = {}

    # Server-confirmed positions (last successful upload position per file)
    server_positions: dict[str, int] = {}

    # Pending uploads that failed and need retry
    pending_uploads: list[PendingUpload] = []

    # Last time we synced with the server
    last_server_sync: datetime | None = None

    # Connection status
    connected: bool = False

    # Reindex tracking
    reindex_required_at: datetime | None = None
    reindex_ack_at: datetime | None = None
    reindex_status: str | None = None


class ClientStateManager:
    """Manages client-side state persistence and catch-up logic."""

    def __init__(self, state_path: Path | None = None):
        self.state_path = state_path or (settings.state_path.parent / "client_state.json")
        self._state = ClientState()
        self._lock = asyncio.Lock()
        self._loaded = False

    async def load(self) -> ClientState:
        """Load state from disk."""
        async with self._lock:
            if self._loaded:
                return self._state

            if self.state_path.exists():
                try:
                    data = json.loads(self.state_path.read_text())
                    self._state = ClientState(**data)
                    logger.info(
                        f"Loaded client state: {len(self._state.local_positions)} files, "
                        f"{len(self._state.pending_uploads)} pending uploads"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load client state: {e}, starting fresh")
                    self._state = ClientState()
            else:
                logger.info("No existing client state, starting fresh")

            self._loaded = True
            return self._state

    async def save(self) -> None:
        """Save state to disk."""
        async with self._lock:
            try:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                data = self._state.model_dump(mode="json")
                self.state_path.write_text(json.dumps(data, indent=2, default=str))
                logger.debug(f"Saved client state to {self.state_path}")
            except Exception as e:
                logger.error(f"Failed to save client state: {e}")

    async def get_state(self) -> ClientState:
        """Get current state, loading from disk if needed."""
        if not self._loaded:
            await self.load()
        return self._state

    async def get_summary(self) -> dict[str, Any]:
        """Get a lightweight summary of client state for diagnostics."""
        state = await self.get_state()
        summary: dict[str, Any] = {
            "pending_uploads": len(state.pending_uploads),
            "last_server_sync": state.last_server_sync.isoformat()
            if state.last_server_sync
            else None,
            "connected": state.connected,
        }
        if state.last_server_sync:
            summary["last_server_sync_age_min"] = int(
                (datetime.now(timezone.utc) - state.last_server_sync).total_seconds() / 60
            )
        return summary

    async def update_local_position(self, file_path: str, position: int) -> None:
        """Update the local position for a file."""
        state = await self.get_state()
        state.local_positions[file_path] = position
        await self.save()

    async def update_server_position(self, file_path: str, position: int) -> None:
        """Update the server-confirmed position for a file."""
        state = await self.get_state()
        state.server_positions[file_path] = position
        state.last_server_sync = datetime.now(timezone.utc)
        await self.save()

    async def set_connected(self, connected: bool) -> None:
        """Update connection status."""
        state = await self.get_state()
        state.connected = connected
        await self.save()

    async def set_reindex_required(self, requested_at: datetime | None) -> None:
        """Mark that the server requested a reindex."""
        state = await self.get_state()
        state.reindex_required_at = requested_at
        state.reindex_ack_at = None
        state.reindex_status = "pending"
        await self.save()

    async def should_handle_reindex(self, requested_at: str | None) -> bool:
        """Check if a reindex request is new."""
        if not requested_at:
            return False
        state = await self.get_state()
        return not (
            state.reindex_required_at and state.reindex_required_at.isoformat() == requested_at
        )

    async def set_reindex_ack(self, status: str | None = None) -> None:
        """Record that we acknowledged a reindex request."""
        state = await self.get_state()
        state.reindex_ack_at = datetime.now(timezone.utc)
        if status:
            state.reindex_status = status
        await self.save()

    async def clear_reindex(self) -> None:
        """Clear reindex tracking fields."""
        state = await self.get_state()
        state.reindex_required_at = None
        state.reindex_ack_at = None
        state.reindex_status = None
        await self.save()

    async def reset_for_reindex(self) -> None:
        """Clear positions and pending uploads for a full reindex."""
        state = await self.get_state()
        state.local_positions = {}
        state.server_positions = {}
        state.pending_uploads = []
        state.last_server_sync = None
        await self.save()

    async def add_pending_upload(
        self,
        file_path: str,
        chunks: list[dict[str, Any]],
        file_position: int,
    ) -> None:
        """Add a pending upload to the queue."""
        state = await self.get_state()

        # Check if we already have a pending upload for this file
        # If so, update it instead of adding a new one
        for pending in state.pending_uploads:
            if pending.file_path == file_path:
                # Merge chunks and update position
                pending.chunks.extend(chunks)
                pending.file_position = max(pending.file_position, file_position)
                logger.debug(
                    f"Merged {len(chunks)} chunks into existing pending upload for {file_path}"
                )
                await self.save()
                return

        # Add new pending upload
        pending = PendingUpload(
            file_path=file_path,
            chunks=chunks,
            file_position=file_position,
            created_at=datetime.now(timezone.utc),
        )
        state.pending_uploads.append(pending)
        logger.info(f"Added pending upload: {file_path} with {len(chunks)} chunks")
        await self.save()

    async def get_pending_uploads(self) -> list[PendingUpload]:
        """Get all pending uploads."""
        state = await self.get_state()
        return list(state.pending_uploads)

    async def remove_pending_upload(self, file_path: str) -> None:
        """Remove a pending upload after successful upload."""
        state = await self.get_state()
        state.pending_uploads = [p for p in state.pending_uploads if p.file_path != file_path]
        await self.save()

    async def increment_retry_count(self, file_path: str) -> int:
        """Increment retry count for a pending upload. Returns new count."""
        state = await self.get_state()
        for pending in state.pending_uploads:
            if pending.file_path == file_path:
                pending.retry_count += 1
                await self.save()
                return pending.retry_count
        return 0

    async def get_files_needing_catchup(
        self,
        server_positions: dict[str, int],
    ) -> list[tuple[str, int, int]]:
        """Get list of files that need catch-up after reconnecting.

        Compares local positions with server positions to find gaps.

        Args:
            server_positions: Positions reported by the server for our machine

        Returns:
            List of (file_path, server_position, local_position) tuples
            for files where local > server (we have more than the server)
        """
        state = await self.get_state()
        catchup_files = []

        for file_path, local_pos in state.local_positions.items():
            server_pos = server_positions.get(file_path, 0)
            if local_pos > server_pos:
                catchup_files.append((file_path, server_pos, local_pos))
                logger.info(
                    f"File needs catch-up: {file_path} (server: {server_pos}, local: {local_pos})"
                )

        return catchup_files

    async def handle_missing_history(self, file_path: str, server_position: int) -> None:
        """Handle case where server position references deleted local content.

        Per requirements: "this should not be an error it should just continue
        with what it has."

        If the server thinks we're at position X but that content no longer exists,
        we log a warning and continue from wherever we are now.
        """
        state = await self.get_state()
        local_pos = state.local_positions.get(file_path, 0)

        logger.warning(
            f"Server position ({server_position}) references content that may have been "
            f"cleaned up for {file_path}. Continuing from local position ({local_pos}). "
            f"Some history may be lost but this is expected after Claude history cleanup."
        )

        # Update server position to match what we have locally
        # This prevents infinite retry loops
        state.server_positions[file_path] = local_pos
        await self.save()

    async def clear_stale_pending_uploads(self, max_age_hours: int = 72) -> int:
        """Clear pending uploads older than max_age_hours.

        Returns number of cleared uploads.
        """
        state = await self.get_state()
        now = datetime.now(timezone.utc)
        initial_count = len(state.pending_uploads)

        state.pending_uploads = [
            p
            for p in state.pending_uploads
            if (now - p.created_at).total_seconds() < max_age_hours * 3600
        ]

        cleared = initial_count - len(state.pending_uploads)
        if cleared > 0:
            logger.info(f"Cleared {cleared} stale pending uploads older than {max_age_hours}h")
            await self.save()

        return cleared


# Global state manager instance
_state_manager: ClientStateManager | None = None


def get_client_state_manager() -> ClientStateManager:
    """Get or create the global client state manager."""
    global _state_manager
    if _state_manager is None:
        _state_manager = ClientStateManager()
    return _state_manager
