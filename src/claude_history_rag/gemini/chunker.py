"""Chunking for Gemini CLI session JSON files."""

import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from claude_history_rag.chunker import generate_chunk_id, split_content
from claude_history_rag.gemini.parser import load_gemini_json_file
from claude_history_rag.models import Chunk

logger = logging.getLogger(__name__)

MAX_CHUNK_CONTENT_LENGTH = 8000


@dataclass
class _PendingUser:
    content: str
    timestamp: datetime | None
    line_number: int


def _parse_timestamp(ts: str | None) -> datetime | None:
    if not ts or not isinstance(ts, str):
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _project_from_path(file_path: Path) -> tuple[str, str]:
    """Return (project_path, project_name) from Gemini tmp structure."""
    parts = list(file_path.parts)
    if "tmp" in parts:
        idx = parts.index("tmp")
        if idx + 1 < len(parts):
            project_hash = parts[idx + 1]
            return f"/gemini/{project_hash}", project_hash
    return "/gemini/unknown", "unknown"


def _create_turn_chunks(
    user_content: str,
    assistant_content: str,
    project_path: str,
    project_name: str,
    session_id: str,
    timestamp: datetime,
    source_file: str,
    source_line: int,
    model: str | None = None,
) -> list[Chunk]:
    content = (
        f"In project {project_name}, user asked:\n{user_content}\n\n"
        f"Assistant responded:\n{assistant_content}"
    )

    content_parts = split_content(content, MAX_CHUNK_CONTENT_LENGTH)
    total_parts = len(content_parts)
    base_id = generate_chunk_id(content, session_id, str(timestamp))
    chunks: list[Chunk] = []

    for part_num, part_content in enumerate(content_parts, start=1):
        if total_parts > 1:
            part_prefix = f"[Part {part_num}/{total_parts}] "
            part_content = part_prefix + part_content

        chunk_id = (
            generate_chunk_id(part_content, session_id, str(timestamp) + f"_part{part_num}")
            if total_parts > 1
            else base_id
        )

        chunks.append(
            Chunk(
                id=chunk_id,
                content=part_content,
                chunk_type="turn",
                session_id=session_id,
                project_path=project_path,
                project_name=project_name,
                timestamp=timestamp,
                model=model,
                source_file=source_file,
                source_line=source_line,
                parent_chunk_id=base_id if total_parts > 1 and part_num > 1 else None,
            )
        )

    return chunks


def _create_assistant_only_chunks(
    assistant_content: str,
    project_path: str,
    project_name: str,
    session_id: str,
    timestamp: datetime,
    source_file: str,
    source_line: int,
    model: str | None = None,
) -> list[Chunk]:
    if not assistant_content:
        return []

    content = f"In project {project_name}, assistant responded:\n{assistant_content}"

    content_parts = split_content(content, MAX_CHUNK_CONTENT_LENGTH)
    total_parts = len(content_parts)
    base_id = generate_chunk_id(content, session_id, str(timestamp))
    chunks: list[Chunk] = []

    for part_num, part_content in enumerate(content_parts, start=1):
        if total_parts > 1:
            part_prefix = f"[Part {part_num}/{total_parts}] "
            part_content = part_prefix + part_content

        chunk_id = (
            generate_chunk_id(part_content, session_id, str(timestamp) + f"_part{part_num}")
            if total_parts > 1
            else base_id
        )

        chunks.append(
            Chunk(
                id=chunk_id,
                content=part_content,
                chunk_type="turn",
                session_id=session_id,
                project_path=project_path,
                project_name=project_name,
                timestamp=timestamp,
                model=model,
                source_file=source_file,
                source_line=source_line,
                parent_chunk_id=base_id if total_parts > 1 and part_num > 1 else None,
            )
        )

    return chunks


def _create_user_only_chunks(
    user_content: str,
    project_path: str,
    project_name: str,
    session_id: str,
    timestamp: datetime,
    source_file: str,
    source_line: int,
    model: str | None = None,
) -> list[Chunk]:
    if not user_content:
        return []

    content = f"In project {project_name}, user said:\n{user_content}"

    content_parts = split_content(content, MAX_CHUNK_CONTENT_LENGTH)
    total_parts = len(content_parts)
    base_id = generate_chunk_id(content, session_id, str(timestamp))
    chunks: list[Chunk] = []

    for part_num, part_content in enumerate(content_parts, start=1):
        if total_parts > 1:
            part_prefix = f"[Part {part_num}/{total_parts}] "
            part_content = part_prefix + part_content

        chunk_id = (
            generate_chunk_id(part_content, session_id, str(timestamp) + f"_part{part_num}")
            if total_parts > 1
            else base_id
        )

        chunks.append(
            Chunk(
                id=chunk_id,
                content=part_content,
                chunk_type="turn",
                session_id=session_id,
                project_path=project_path,
                project_name=project_name,
                timestamp=timestamp,
                model=model,
                source_file=source_file,
                source_line=source_line,
                parent_chunk_id=base_id if total_parts > 1 and part_num > 1 else None,
            )
        )

    return chunks


def _create_file_change_chunks(
    ops: list[dict],
    project_path: str,
    project_name: str,
    session_id: str,
    timestamp: datetime,
    source_file: str,
    source_line: int,
    parent_chunk_id: str | None,
) -> list[Chunk]:
    if not ops:
        return []

    chunks: list[Chunk] = []
    for op in ops:
        file_path = op.get("file_path", "unknown")
        operation = op.get("operation", "edit")
        summary = op.get("summary", "File change")
        content = (
            f"In project {project_name}, file {file_path} was {operation}.\nSummary: {summary}"
        )
        chunk_id = generate_chunk_id(content, session_id, str(timestamp) + file_path)
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


def _gemini_tool_calls_text(tool_calls: list) -> str:
    if not tool_calls:
        return ""
    rendered = []
    for call in tool_calls:
        if isinstance(call, dict):
            rendered.append(json.dumps(call))
        else:
            rendered.append(str(call))
    return "\n".join(rendered)


def _extract_apply_patch_ops_from_text(text: str) -> list[dict]:
    ops: list[dict] = []
    if not text:
        return ops
    for line in text.splitlines():
        if line.startswith("*** Add File: "):
            file_path = line.replace("*** Add File: ", "", 1).strip()
            ops.append({"file_path": file_path, "operation": "write", "summary": "Added file"})
        elif line.startswith("*** Update File: "):
            file_path = line.replace("*** Update File: ", "", 1).strip()
            ops.append({"file_path": file_path, "operation": "edit", "summary": "Updated file"})
        elif line.startswith("*** Delete File: "):
            file_path = line.replace("*** Delete File: ", "", 1).strip()
            ops.append({"file_path": file_path, "operation": "delete", "summary": "Deleted file"})
        elif line.startswith("*** Move to: "):
            file_path = line.replace("*** Move to: ", "", 1).strip()
            ops.append({"file_path": file_path, "operation": "move", "summary": "Moved file"})
    return ops


def _extract_shell_ops_from_command(command: str) -> list[dict]:
    ops: list[dict] = []
    if not command:
        return ops
    patterns = [
        (r"\brm\s+(-[^\s]+\s+)?(?P<path>[^\s]+)", "delete", "Deleted file"),
        (r"\bmv\s+([^\s]+)\s+(?P<path>[^\s]+)", "move", "Moved file"),
        (r"\bcp\s+([^\s]+)\s+(?P<path>[^\s]+)", "write", "Copied file"),
        (r"\btouch\s+(?P<path>[^\s]+)", "write", "Touched file"),
        (r"\bmkdir\s+(-p\s+)?(?P<path>[^\s]+)", "write", "Created directory"),
        (r">\s*(?P<path>[^\s]+)", "write", "Wrote file"),
        (r"\btee\s+(-a\s+)?(?P<path>[^\s]+)", "write", "Wrote file"),
        (r"\bsed\s+-i.*\s+(?P<path>[^\s]+)$", "edit", "Edited file"),
    ]
    for pattern, op, summary in patterns:
        m = re.search(pattern, command)
        if not m:
            continue
        path = m.groupdict().get("path")
        if path:
            ops.append({"file_path": path, "operation": op, "summary": summary})
            break
    return ops


def _extract_tool_ops_from_call(tool_call: dict) -> list[dict]:
    if not isinstance(tool_call, dict):
        return []
    name = str(tool_call.get("name", ""))
    args = tool_call.get("args")
    result_display = tool_call.get("resultDisplay")
    ops: list[dict] = []

    if name in ("apply_patch", "applyPatch", "patch"):
        if isinstance(args, dict):
            patch_text = args.get("patch")
            if isinstance(patch_text, str):
                ops.extend(_extract_apply_patch_ops_from_text(patch_text))
        if isinstance(result_display, str):
            ops.extend(_extract_apply_patch_ops_from_text(result_display))

    if name in ("shell", "shell_command", "bash") and isinstance(args, dict):
        cmd = args.get("command")
        if isinstance(cmd, str):
            ops.extend(_extract_shell_ops_from_command(cmd))

    return ops


def chunk_gemini_session_file(file_path: Path, start_line: int = 0) -> Iterator[Chunk]:
    """Process a Gemini CLI session/log JSON file and yield chunks."""
    logger.debug(f"Starting Gemini chunking: {file_path} from line {start_line}")

    source_file = str(file_path)
    chunk_counts: dict[str, int] = defaultdict(int)

    data = load_gemini_json_file(file_path)
    if data is None:
        return iter(())

    project_path, project_name = _project_from_path(file_path)

    # logs.json (list of events)
    if isinstance(data, list):
        for idx, event in enumerate(data, start=1):
            content = str(event) if not isinstance(event, dict) else json.dumps(event)
            timestamp = (
                _parse_timestamp(event.get("timestamp")) if isinstance(event, dict) else None
            )
            if not timestamp:
                timestamp = datetime.now(timezone.utc)
            session_id = (
                event.get("sessionId")
                if isinstance(event, dict) and event.get("sessionId")
                else "gemini-logs"
            )
            chunks = _create_assistant_only_chunks(
                assistant_content=content,
                project_path=project_path,
                project_name=project_name,
                session_id=session_id,
                timestamp=timestamp,
                source_file=source_file,
                source_line=idx,
                model=None,
            )
            for chunk in chunks:
                chunk_counts[chunk.chunk_type] += 1
                yield chunk

        total = sum(chunk_counts.values())
        logger.info(
            f"Completed Gemini chunking {file_path.name}: {total} chunks "
            f"(turns={chunk_counts['turn']})"
        )
        return

    if not isinstance(data, dict):
        return iter(())

    session_id = data.get("sessionId", "gemini-session")
    model_default = None
    messages = data.get("messages", [])

    pending_user: _PendingUser | None = None
    assistant_fragments: list[str] = []
    pending_ops: list[dict] = []

    def finalize_turn() -> Iterator[Chunk]:
        nonlocal pending_user, assistant_fragments, pending_ops
        if pending_user is None:
            return iter(())

        user_content = pending_user.content
        assistant_content = "\n".join([f for f in assistant_fragments if f]).strip()
        timestamp = pending_user.timestamp or datetime.now(timezone.utc)

        chunks = _create_turn_chunks(
            user_content=user_content,
            assistant_content=assistant_content,
            project_path=project_path,
            project_name=project_name,
            session_id=session_id,
            timestamp=timestamp,
            source_file=source_file,
            source_line=pending_user.line_number,
            model=model_default,
        )

        first_turn_chunk = chunks[0] if chunks else None
        file_chunks = _create_file_change_chunks(
            pending_ops,
            project_path=project_path,
            project_name=project_name,
            session_id=session_id,
            timestamp=timestamp,
            source_file=source_file,
            source_line=pending_user.line_number,
            parent_chunk_id=first_turn_chunk.id if first_turn_chunk else None,
        )

        if first_turn_chunk and file_chunks:
            chunks[0] = Chunk(
                **first_turn_chunk.model_dump(exclude={"child_chunk_ids"}),
                child_chunk_ids=[c.id for c in file_chunks],
            )

        pending_user = None
        assistant_fragments = []
        pending_ops = []

        def _yield_all() -> Iterator[Chunk]:
            yield from chunks
            if file_chunks:
                yield from file_chunks

        return _yield_all()

    if not isinstance(messages, list):
        messages = []

    for idx, msg in enumerate(messages, start=1):
        if not isinstance(msg, dict):
            content = str(msg)
            timestamp = datetime.now(timezone.utc)
            chunks = _create_assistant_only_chunks(
                assistant_content=content,
                project_path=project_path,
                project_name=project_name,
                session_id=session_id,
                timestamp=timestamp,
                source_file=source_file,
                source_line=idx,
                model=model_default,
            )
            for chunk in chunks:
                chunk_counts[chunk.chunk_type] += 1
                yield chunk
            continue

        mtype = msg.get("type")
        content = msg.get("content", "")
        timestamp = _parse_timestamp(msg.get("timestamp")) or datetime.now(timezone.utc)

        if mtype == "user":
            for chunk in finalize_turn():
                chunk_counts[chunk.chunk_type] += 1
                yield chunk
            pending_user = _PendingUser(
                content=str(content),
                timestamp=timestamp,
                line_number=idx,
            )
            continue

        if mtype == "gemini":
            tool_calls = msg.get("toolCalls", [])
            thoughts = msg.get("thoughts", [])
            model_default = msg.get("model") or model_default

            parts = [str(content)]
            if thoughts:
                parts.append("Thoughts:\n" + "\n".join(str(t) for t in thoughts))
            if tool_calls:
                parts.append("Tool calls:\n" + _gemini_tool_calls_text(tool_calls))
            combined = "\n\n".join(p for p in parts if p)

            tool_ops: list[dict] = []
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    tool_ops.extend(_extract_tool_ops_from_call(call))

            if pending_user is None:
                chunks = _create_assistant_only_chunks(
                    assistant_content=combined,
                    project_path=project_path,
                    project_name=project_name,
                    session_id=session_id,
                    timestamp=timestamp,
                    source_file=source_file,
                    source_line=idx,
                    model=model_default,
                )
                for chunk in chunks:
                    chunk_counts[chunk.chunk_type] += 1
                    yield chunk
                if chunks and tool_ops:
                    file_chunks = _create_file_change_chunks(
                        tool_ops,
                        project_path=project_path,
                        project_name=project_name,
                        session_id=session_id,
                        timestamp=timestamp,
                        source_file=source_file,
                        source_line=idx,
                        parent_chunk_id=chunks[0].id,
                    )
                    for chunk in file_chunks:
                        chunk_counts[chunk.chunk_type] += 1
                        yield chunk
            else:
                assistant_fragments.append(combined)
                if tool_ops:
                    pending_ops.extend(tool_ops)
            continue

        # info or any other type
        if mtype == "info":
            chunks = _create_assistant_only_chunks(
                assistant_content=str(content),
                project_path=project_path,
                project_name=project_name,
                session_id=session_id,
                timestamp=timestamp,
                source_file=source_file,
                source_line=idx,
                model=model_default,
            )
            for chunk in chunks:
                chunk_counts[chunk.chunk_type] += 1
                yield chunk
            continue

        # Unknown message types still ingested
        chunks = _create_assistant_only_chunks(
            assistant_content=json.dumps(msg),
            project_path=project_path,
            project_name=project_name,
            session_id=session_id,
            timestamp=timestamp,
            source_file=source_file,
            source_line=idx,
            model=model_default,
        )
        for chunk in chunks:
            chunk_counts[chunk.chunk_type] += 1
            yield chunk

    for chunk in finalize_turn():
        chunk_counts[chunk.chunk_type] += 1
        yield chunk

    total = sum(chunk_counts.values())
    logger.info(
        f"Completed Gemini chunking {file_path.name}: {total} chunks (turns={chunk_counts['turn']})"
    )
