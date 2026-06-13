"""Chunking for Google Antigravity history files.

Modern Antigravity writes JSONL transcripts under
``~/.gemini/antigravity/brain/<conversation-id>/.system_generated/logs/``.
Older builds used protobuf-like binary files under ``conversations/``; those
remain supported with a best-effort string extractor.
"""

import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claude_history_rag.chunker import generate_chunk_id, split_content
from claude_history_rag.models import Chunk

logger = logging.getLogger(__name__)

MAX_CHUNK_CONTENT_LENGTH = 8000


def _extract_strings(data: bytes, min_length: int = 4) -> list[str]:
    """Extract printable strings from binary data."""
    # Find sequences of printable characters
    # This regex looks for 4 or more printable characters
    # excluding some common binary noise
    text = data.decode("utf-8", errors="ignore")
    # Just return the whole decoded string but cleaned up a bit?
    # No, protobuf mixes binary tags.
    # Let's try to just clean up non-printable chars
    clean_text = "".join(c if c.isprintable() or c in "\n\t\r" else " " for c in text)
    # Collapse multiple spaces
    clean_text = re.sub(r"\s+", " ", clean_text)

    return [clean_text]


def _parse_timestamp(value: Any, fallback: datetime) -> datetime:
    if not isinstance(value, str) or not value:
        return fallback
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return fallback


def _session_id_from_path(file_path: Path) -> str:
    parts = file_path.parts
    if "brain" in parts:
        brain_idx = parts.index("brain")
        if brain_idx + 1 < len(parts):
            return parts[brain_idx + 1]
    return file_path.stem


def _content_from_event(event: dict[str, Any]) -> str:
    sections = [
        f"Source: {event.get('source', 'unknown')}",
        f"Type: {event.get('type', 'unknown')}",
        f"Status: {event.get('status', 'unknown')}",
    ]
    if event.get("content"):
        sections.append(f"Content:\n{event['content']}")
    if event.get("thinking"):
        sections.append(f"Thinking:\n{event['thinking']}")
    tool_calls = event.get("tool_calls")
    if tool_calls:
        sections.append(f"Tool calls:\n{json.dumps(tool_calls, ensure_ascii=False)}")
    return "\n\n".join(sections)


def _extract_apply_patch_ops_from_text(text: str) -> list[dict[str, str]]:
    ops: list[dict[str, str]] = []
    for line in text.splitlines():
        if line.startswith("*** Add File: "):
            ops.append(
                {
                    "file_path": line.replace("*** Add File: ", "", 1).strip(),
                    "operation": "write",
                    "summary": "Added file",
                }
            )
        elif line.startswith("*** Update File: "):
            ops.append(
                {
                    "file_path": line.replace("*** Update File: ", "", 1).strip(),
                    "operation": "edit",
                    "summary": "Updated file",
                }
            )
        elif line.startswith("*** Delete File: "):
            ops.append(
                {
                    "file_path": line.replace("*** Delete File: ", "", 1).strip(),
                    "operation": "delete",
                    "summary": "Deleted file",
                }
            )
    return ops


def _extract_shell_ops_from_command(command: str) -> list[dict[str, str]]:
    patterns = [
        (r"\brm\s+(-[^\s]+\s+)?(?P<path>[^\s]+)", "delete", "Deleted file"),
        (r"\bmv\s+([^\s]+)\s+(?P<path>[^\s]+)", "move", "Moved file"),
        (r"\bcp\s+([^\s]+)\s+(?P<path>[^\s]+)", "write", "Copied file"),
        (r"\btouch\s+(?P<path>[^\s]+)", "write", "Touched file"),
        (r">\s*(?P<path>[^\s]+)", "write", "Wrote file"),
        (r"\btee\s+(-a\s+)?(?P<path>[^\s]+)", "write", "Wrote file"),
        (r"\bsed\s+-i.*\s+(?P<path>[^\s]+)$", "edit", "Edited file"),
    ]
    for pattern, operation, summary in patterns:
        match = re.search(pattern, command)
        if not match:
            continue
        file_path = match.groupdict().get("path")
        if file_path:
            return [{"file_path": file_path, "operation": operation, "summary": summary}]
    return []


def _extract_tool_ops(event: dict[str, Any]) -> list[dict[str, str]]:
    tool_calls = event.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    ops: list[dict[str, str]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "")
        args = call.get("args")
        if isinstance(args, dict):
            for key in ("file_path", "path"):
                file_path = args.get(key)
                if isinstance(file_path, str) and file_path:
                    operation = "edit" if "edit" in name.lower() else "write"
                    ops.append(
                        {
                            "file_path": file_path,
                            "operation": operation,
                            "summary": f"{name} tool call",
                        }
                    )
            patch_text = args.get("patch")
            if isinstance(patch_text, str):
                ops.extend(_extract_apply_patch_ops_from_text(patch_text))
            command = args.get("command") or args.get("CommandLine")
            if isinstance(command, str):
                ops.extend(_extract_shell_ops_from_command(command))
        elif isinstance(args, str):
            ops.extend(_extract_apply_patch_ops_from_text(args))
            ops.extend(_extract_shell_ops_from_command(args))
    return ops


def _create_file_change_chunks(
    ops: list[dict[str, str]],
    project_path: str,
    project_name: str,
    session_id: str,
    timestamp: datetime,
    source_file: str,
    source_line: int,
    parent_chunk_id: str,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[tuple[str, str]] = set()
    for op in ops:
        file_path = op.get("file_path", "unknown")
        operation = op.get("operation", "edit")
        key = (file_path, operation)
        if key in seen:
            continue
        seen.add(key)
        summary = op.get("summary", "File change")
        content = (
            f"In Antigravity session {session_id}, file {file_path} was {operation}.\n"
            f"Summary: {summary}"
        )
        chunk_id = generate_chunk_id(content, session_id, f"{source_line}:{file_path}:{operation}")
        chunks.append(
            Chunk(
                id=chunk_id,
                content=content,
                chunk_type="file_change",
                session_id=session_id,
                project_path=project_path,
                project_name=project_name,
                timestamp=timestamp,
                file_path=file_path,
                operation=operation,
                source_file=source_file,
                source_line=source_line,
                parent_chunk_id=parent_chunk_id,
            )
        )
    return chunks


def _chunk_antigravity_jsonl_file(file_path: Path, start_line: int = 0) -> Iterator[Chunk]:
    """Process a modern Antigravity JSONL transcript and yield chunks."""
    fallback_timestamp = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
    session_id = _session_id_from_path(file_path)
    project_path = f"/antigravity/{session_id}"
    project_name = "Antigravity Session"
    chunk_counts: dict[str, int] = defaultdict(int)

    try:
        with open(file_path) as f:
            for line_num, line in enumerate(f, start=1):
                if line_num <= start_line or not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Skipping malformed Antigravity JSONL line %s in session=%s file=%s: %s",
                        line_num,
                        session_id,
                        file_path.name,
                        e,
                    )
                    continue
                if not isinstance(event, dict):
                    continue

                content = _content_from_event(event)
                if not content.strip():
                    continue

                timestamp = _parse_timestamp(event.get("created_at"), fallback_timestamp)
                step_index = event.get("step_index", line_num)
                chunk_type = "turn"
                content_parts = split_content(content, MAX_CHUNK_CONTENT_LENGTH)
                base_id = generate_chunk_id(content, session_id, str(step_index))

                for part_num, part_content in enumerate(content_parts, start=1):
                    if len(content_parts) > 1:
                        part_content = f"[Part {part_num}/{len(content_parts)}] {part_content}"
                    chunk_id = (
                        generate_chunk_id(part_content, session_id, f"{step_index}_part{part_num}")
                        if len(content_parts) > 1
                        else base_id
                    )
                    yield Chunk(
                        id=chunk_id,
                        content=part_content,
                        chunk_type=chunk_type,
                        session_id=session_id,
                        project_path=project_path,
                        project_name=project_name,
                        timestamp=timestamp,
                        model="gemini-unknown",
                        source_file=str(file_path),
                        source_line=line_num,
                        parent_chunk_id=base_id if part_num > 1 else None,
                    )
                    chunk_counts[chunk_type] += 1
                for file_chunk in _create_file_change_chunks(
                    _extract_tool_ops(event),
                    project_path=project_path,
                    project_name=project_name,
                    session_id=session_id,
                    timestamp=timestamp,
                    source_file=str(file_path),
                    source_line=line_num,
                    parent_chunk_id=base_id,
                ):
                    chunk_counts["file_change"] += 1
                    yield file_chunk
    except OSError as e:
        logger.error("Error reading %s: %s", file_path, e)
        return

    total = sum(chunk_counts.values())
    logger.info("Completed Antigravity JSONL chunking %s: %s chunks", file_path.name, total)


def chunk_antigravity_file(file_path: Path, start_line: int = 0) -> Iterator[Chunk]:
    """Process an Antigravity history file and yield chunks."""
    logger.debug(f"Starting Antigravity chunking: {file_path}")

    if file_path.suffix == ".jsonl":
        yield from _chunk_antigravity_jsonl_file(file_path, start_line=start_line)
        return

    # We can't really support start_line efficiently on binary files without a proper parser
    # But since we treat it as one blob or simple string extraction, we'll just process it all
    # if it changed. The watcher handles debounce.

    if not file_path.exists():
        return iter(())

    try:
        data = file_path.read_bytes()
    except OSError as e:
        logger.error(f"Error reading {file_path}: {e}")
        return iter(())

    # Extract text (best effort)
    extracted_strings = _extract_strings(data)
    content = "\n".join(extracted_strings)

    if not content.strip():
        return iter(())

    session_id = file_path.stem  # e.g. "uuid"
    timestamp = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)

    # We treat it as one "assistant" chunk for now, or just generic info
    # differentiating user/assistant is hard without proto definition.
    # We label it as "history" with the filename.

    header = f"Google Antigravity History from {file_path.name}\n\n"
    full_content = header + content

    content_parts = split_content(full_content, MAX_CHUNK_CONTENT_LENGTH)
    total_parts = len(content_parts)
    base_id = generate_chunk_id(full_content, session_id, str(timestamp))

    chunk_counts: dict[str, int] = defaultdict(int)

    for part_num, part_content in enumerate(content_parts, start=1):
        if total_parts > 1:
            part_prefix = f"[Part {part_num}/{total_parts}] "
            part_content = part_prefix + part_content

        chunk_id = (
            generate_chunk_id(part_content, session_id, str(timestamp) + f"_part{part_num}")
            if total_parts > 1
            else base_id
        )

        chunk = Chunk(
            id=chunk_id,
            content=part_content,
            chunk_type="antigravity_history",  # Special type? Or just "turn"
            session_id=session_id,
            project_path="/antigravity/unknown",  # We don't know the project
            project_name="Antigravity Session",
            timestamp=timestamp,
            model="gemini-unknown",
            source_file=str(file_path),
            source_line=0,  # Binary file
            parent_chunk_id=base_id if total_parts > 1 and part_num > 1 else None,
        )

        chunk_counts["history"] += 1
        yield chunk

    total = sum(chunk_counts.values())
    logger.info(f"Completed Antigravity chunking {file_path.name}: {total} chunks")
