"""Chunking for Claude web/Desktop official export files."""

import json
import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claude_history_rag.chunker import generate_chunk_id, split_content
from claude_history_rag.models import Chunk

logger = logging.getLogger(__name__)

MAX_CHUNK_CONTENT_LENGTH = 8000


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str) and value:
        try:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _text_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for part in (_text_from_value(item) for item in value) if part)
    if isinstance(value, dict):
        for key in ("text", "content", "message", "value"):
            text = _text_from_value(value.get(key))
            if text:
                return text
    return ""


def _messages(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("chat_messages", "messages"):
        value = conversation.get(key)
        if isinstance(value, list):
            return [m for m in value if isinstance(m, dict)]
    return []


def _role(message: dict[str, Any]) -> str:
    sender = message.get("sender") or message.get("role") or message.get("author")
    if isinstance(sender, dict):
        sender = sender.get("role") or sender.get("name")
    sender_text = str(sender or "").lower()
    if sender_text in {"human", "user"}:
        return "user"
    if sender_text in {"assistant", "claude"}:
        return "assistant"
    return sender_text


def _message_text(message: dict[str, Any]) -> str:
    for key in ("text", "content", "message"):
        text = _text_from_value(message.get(key))
        if text:
            return text
    return ""


def _create_turn_chunks(
    user_content: str,
    assistant_content: str,
    conversation_id: str,
    title: str,
    timestamp: datetime,
    source_file: str,
    source_line: int,
) -> list[Chunk]:
    content = f"In Claude app conversation {title}, user asked:\n{user_content}\n\nClaude responded:\n{assistant_content}"
    parts = split_content(content, MAX_CHUNK_CONTENT_LENGTH)
    base_id = generate_chunk_id(content, conversation_id, str(timestamp))
    chunks: list[Chunk] = []
    for index, part in enumerate(parts, start=1):
        chunk_id = (
            generate_chunk_id(part, conversation_id, f"{timestamp}_part{index}")
            if len(parts) > 1
            else base_id
        )
        chunks.append(
            Chunk(
                id=chunk_id,
                content=f"[Part {index}/{len(parts)}] {part}" if len(parts) > 1 else part,
                chunk_type="turn",
                session_id=conversation_id,
                project_path="/claude-app/export",
                project_name="Claude App",
                timestamp=timestamp,
                source_file=source_file,
                source_line=source_line,
                parent_chunk_id=base_id if len(parts) > 1 and index > 1 else None,
            )
        )
    return chunks


def chunk_claude_app_export_file(file_path: Path, start_line: int = 0) -> Iterator[Chunk]:
    """Yield chunks from a Claude web/Desktop export `conversations.json` file."""
    del start_line
    try:
        data = json.loads(file_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to parse Claude app export %s: %s", file_path.name, type(e).__name__)
        return

    conversations = data if isinstance(data, list) else data.get("conversations", [])
    if not isinstance(conversations, list):
        return

    source_file = str(file_path)
    for conv_index, conversation in enumerate(conversations, start=1):
        if not isinstance(conversation, dict):
            continue
        conversation_id = str(
            conversation.get("uuid")
            or conversation.get("id")
            or conversation.get("conversation_id")
            or conv_index
        )
        title = str(conversation.get("name") or conversation.get("title") or "Untitled")
        pending_user: tuple[str, datetime, int] | None = None
        for msg_index, message in enumerate(_messages(conversation), start=1):
            text = _message_text(message).strip()
            if not text:
                continue
            timestamp = _parse_timestamp(
                message.get("created_at") or message.get("createdAt") or message.get("timestamp")
            )
            line = conv_index * 100000 + msg_index
            role = _role(message)
            if role == "user":
                pending_user = (text, timestamp, line)
            elif role == "assistant" and pending_user:
                user_text, user_timestamp, user_line = pending_user
                yield from _create_turn_chunks(
                    user_text,
                    text,
                    conversation_id,
                    title,
                    timestamp or user_timestamp,
                    source_file,
                    user_line,
                )
                pending_user = None
