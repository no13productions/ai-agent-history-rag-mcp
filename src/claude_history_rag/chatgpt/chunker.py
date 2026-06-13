"""Chunking for official ChatGPT data export files."""

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


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_text_from_content(item) for item in content]
        return "\n".join(part for part in parts if part)
    if not isinstance(content, dict):
        return ""

    parts = content.get("parts")
    if isinstance(parts, list):
        return "\n".join(_text_from_content(part) for part in parts if _text_from_content(part))

    text = content.get("text") or content.get("value") or content.get("content")
    if isinstance(text, str):
        return text
    if isinstance(text, list | dict):
        return _text_from_content(text)
    return ""


def _message_role(message: dict[str, Any]) -> str:
    author = message.get("author")
    if isinstance(author, dict):
        role = author.get("role")
        if isinstance(role, str):
            return role
    role = message.get("role") or message.get("sender")
    return str(role).lower() if role else ""


def _message_text(message: dict[str, Any]) -> str:
    for key in ("content", "text", "message"):
        text = _text_from_content(message.get(key))
        if text:
            return text
    return ""


def _conversation_messages(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(conversation.get("messages"), list):
        return [m for m in conversation["messages"] if isinstance(m, dict)]

    mapping = conversation.get("mapping")
    if isinstance(mapping, dict):
        messages: list[dict[str, Any]] = []
        for node in mapping.values():
            if not isinstance(node, dict):
                continue
            message = node.get("message")
            if isinstance(message, dict):
                messages.append(message)
        return sorted(messages, key=lambda m: m.get("create_time") or m.get("update_time") or 0)

    return []


def _create_turn_chunks(
    user_content: str,
    assistant_content: str,
    conversation_id: str,
    title: str,
    timestamp: datetime,
    source_file: str,
    source_line: int,
) -> list[Chunk]:
    content = f"In ChatGPT conversation {title}, user asked:\n{user_content}\n\nChatGPT responded:\n{assistant_content}"
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
                project_path="/chatgpt/export",
                project_name="ChatGPT",
                timestamp=timestamp,
                source_file=source_file,
                source_line=source_line,
                parent_chunk_id=base_id if len(parts) > 1 and index > 1 else None,
            )
        )
    return chunks


def chunk_chatgpt_export_file(file_path: Path, start_line: int = 0) -> Iterator[Chunk]:
    """Yield chunks from a ChatGPT `conversations.json` export file.

    ChatGPT exports are full snapshots, not append-only logs, so start_line is
    intentionally ignored. Stable chunk IDs make reprocessing idempotent.
    """
    del start_line
    try:
        data = json.loads(file_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to parse ChatGPT export %s: %s", file_path.name, type(e).__name__)
        return

    conversations = data if isinstance(data, list) else data.get("conversations", [])
    if not isinstance(conversations, list):
        return

    source_file = str(file_path)
    for conv_index, conversation in enumerate(conversations, start=1):
        if not isinstance(conversation, dict):
            continue
        conversation_id = str(
            conversation.get("id") or conversation.get("conversation_id") or conv_index
        )
        title = str(conversation.get("title") or "Untitled")
        pending_user: tuple[str, datetime, int] | None = None
        for msg_index, message in enumerate(_conversation_messages(conversation), start=1):
            role = _message_role(message)
            text = _message_text(message).strip()
            if not text:
                continue
            timestamp = _parse_timestamp(message.get("create_time") or message.get("created_at"))
            line = conv_index * 100000 + msg_index
            if role == "user":
                pending_user = (text, timestamp, line)
            elif role in {"assistant", "tool"} and pending_user:
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
