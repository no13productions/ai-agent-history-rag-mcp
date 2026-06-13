"""Chunking for Codex session JSONL files."""

import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from claude_history_rag.chunker import generate_chunk_id, split_content
from claude_history_rag.codex.parser import parse_codex_jsonl_file
from claude_history_rag.models import Chunk

logger = logging.getLogger(__name__)

# Reuse core chunk sizing
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
        # Codex timestamps end with 'Z'
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _session_id_from_filename(file_path: Path) -> str | None:
    stem = file_path.stem
    if len(stem) >= 36:
        candidate = stem[-36:]
        # UUID-like pattern with hyphens
        if re.fullmatch(r"[0-9a-fA-F-]{36}", candidate):
            return candidate.lower()
    return None


def _project_path_from_cwd(cwd: str | None) -> str:
    if not cwd or not isinstance(cwd, str):
        return "/unknown"
    return cwd


def _extract_message_text(payload: dict) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        text = item.get("text")
        if isinstance(text, str) and item_type in ("input_text", "output_text", "text"):
            parts.append(text)
    return "\n".join(p for p in parts if p)


def _summarize_tool_call(payload: dict) -> tuple[str, list[dict]]:
    """Return (text_summary, file_ops)."""
    ptype = payload.get("type")
    if not isinstance(ptype, str):
        return "", []

    if ptype == "function_call":
        name = str(payload.get("name", "unknown"))
        args = payload.get("arguments")
        args_str = args if isinstance(args, str) else json.dumps(args) if args else ""
        ops = _extract_tool_ops(name, args)
        if name == "apply_patch" and isinstance(args, str):
            ops = _extract_apply_patch_ops(args)
        return f"[Tool call] {name} {args_str}".strip(), ops

    if ptype == "function_call_output":
        output = payload.get("output")
        out_str = output if isinstance(output, str) else json.dumps(output) if output else ""
        return f"[Tool output] {out_str}".strip(), []

    if ptype == "web_search_call":
        action = payload.get("action", {})
        query = ""
        if isinstance(action, dict):
            query = str(action.get("query", ""))
        return f"[Web search] {query}".strip(), []

    if ptype == "reasoning":
        summary = payload.get("summary")
        if isinstance(summary, list):
            summary_text = " ".join(str(s) for s in summary if s)
            return f"[Reasoning summary] {summary_text}".strip(), []
        return "[Reasoning summary]", []

    if ptype == "ghost_snapshot":
        return "[Ghost snapshot captured]", []

    return "", []


def _extract_tool_ops(tool_name: str, args: object) -> list[dict]:
    """Heuristic file operation extraction from tool calls."""
    ops: list[dict] = []
    if tool_name not in ("shell", "shell_command"):
        return ops

    command = ""
    if isinstance(args, dict):
        cmd = args.get("command")
        if isinstance(cmd, str):
            command = cmd
    elif isinstance(args, str):
        # Try to parse JSON string arguments
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            cmd = parsed.get("command")
            if isinstance(cmd, str):
                command = cmd

    if not command:
        return ops

    # Simple heuristics for common file operations
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


def _extract_apply_patch_ops(arguments: str) -> list[dict]:
    """Extract file operations from apply_patch arguments JSON."""
    ops: list[dict] = []
    try:
        data = json.loads(arguments)
    except json.JSONDecodeError:
        return ops

    patch = data.get("patch")
    if not isinstance(patch, str):
        return ops

    for line in patch.splitlines():
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


def _create_turn_chunks(
    user_content: str,
    assistant_content: str,
    project_path: str,
    session_id: str,
    timestamp: datetime,
    source_file: str,
    source_line: int,
    model: str | None = None,
) -> list[Chunk]:
    project_name = Path(project_path).name or "unknown"
    if not assistant_content:
        assistant_content = ""

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
    session_id: str,
    timestamp: datetime,
    source_file: str,
    source_line: int,
    model: str | None = None,
) -> list[Chunk]:
    project_name = Path(project_path).name or "unknown"
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
    session_id: str,
    timestamp: datetime,
    source_file: str,
    source_line: int,
    model: str | None = None,
) -> list[Chunk]:
    project_name = Path(project_path).name or "unknown"
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
    session_id: str,
    timestamp: datetime,
    source_file: str,
    source_line: int,
    parent_chunk_id: str | None,
) -> list[Chunk]:
    if not ops:
        return []

    project_name = Path(project_path).name or "unknown"
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


def chunk_codex_session_file(file_path: Path, start_line: int = 0) -> Iterator[Chunk]:
    """Process a Codex session JSONL file and yield chunks."""
    logger.debug(f"Starting Codex chunking: {file_path} from line {start_line}")

    source_file = str(file_path)
    chunk_counts: dict[str, int] = defaultdict(int)

    session_id = _session_id_from_filename(file_path) or "unknown"
    session_cwd: str | None = None
    last_cwd: str | None = None
    model: str | None = None

    pending_user: _PendingUser | None = None
    assistant_fragments: list[str] = []
    pending_ops: list[dict] = []

    def finalize_turn() -> Iterator[Chunk]:
        nonlocal pending_user, assistant_fragments, pending_ops
        if pending_user is None:
            return iter(())

        user_content = pending_user.content
        assistant_content = "\n".join([f for f in assistant_fragments if f]).strip()

        project_path = _project_path_from_cwd(last_cwd or session_cwd)
        timestamp = pending_user.timestamp or datetime.now(timezone.utc)

        turn_chunks = _create_turn_chunks(
            user_content=user_content,
            assistant_content=assistant_content,
            project_path=project_path,
            session_id=session_id,
            timestamp=timestamp,
            source_file=source_file,
            source_line=pending_user.line_number,
            model=model,
        )

        if not turn_chunks:
            pending_user = None
            assistant_fragments = []
            pending_ops = []
            return iter(())

        first_turn_chunk = turn_chunks[0]
        file_chunks = _create_file_change_chunks(
            pending_ops,
            project_path=project_path,
            session_id=session_id,
            timestamp=timestamp,
            source_file=source_file,
            source_line=pending_user.line_number,
            parent_chunk_id=first_turn_chunk.id,
        )

        if file_chunks:
            turn_chunks[0] = Chunk(
                **first_turn_chunk.model_dump(exclude={"child_chunk_ids"}),
                child_chunk_ids=[c.id for c in file_chunks],
            )

        pending_user = None
        assistant_fragments = []
        pending_ops = []

        def _yield_all() -> Iterator[Chunk]:
            yield from turn_chunks
            if file_chunks:
                yield from file_chunks

        return _yield_all()

    for event, line_number in parse_codex_jsonl_file(file_path, start_line):
        etype = event.get("type")
        payload = event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}

        if etype == "session_meta":
            session_id = payload.get("id") or session_id
            session_cwd = payload.get("cwd") or session_cwd
            continue

        if etype == "turn_context":
            last_cwd = payload.get("cwd") or last_cwd
            model = payload.get("model") or model
            continue

        if etype == "event_msg":
            msg_type = payload.get("type")
            msg = payload.get("message")
            text = msg if isinstance(msg, str) else json.dumps(payload) if payload else ""
            if text:
                project_path = _project_path_from_cwd(last_cwd or session_cwd)
                timestamp = _parse_timestamp(event.get("timestamp")) or datetime.now(timezone.utc)
                if msg_type == "user_message":
                    chunks = _create_user_only_chunks(
                        user_content=text,
                        project_path=project_path,
                        session_id=session_id,
                        timestamp=timestamp,
                        source_file=source_file,
                        source_line=line_number,
                        model=model,
                    )
                else:
                    chunks = _create_assistant_only_chunks(
                        assistant_content=text,
                        project_path=project_path,
                        session_id=session_id,
                        timestamp=timestamp,
                        source_file=source_file,
                        source_line=line_number,
                        model=model,
                    )
                for chunk in chunks:
                    chunk_counts[chunk.chunk_type] += 1
                    yield chunk
            continue

        if etype != "response_item":
            text = json.dumps(event)
            project_path = _project_path_from_cwd(last_cwd or session_cwd)
            timestamp = _parse_timestamp(event.get("timestamp")) or datetime.now(timezone.utc)
            chunks = _create_assistant_only_chunks(
                assistant_content=text,
                project_path=project_path,
                session_id=session_id,
                timestamp=timestamp,
                source_file=source_file,
                source_line=line_number,
                model=model,
            )
            for chunk in chunks:
                chunk_counts[chunk.chunk_type] += 1
                yield chunk
            continue

        ptype = payload.get("type")
        ts = _parse_timestamp(event.get("timestamp"))

        if ptype == "message":
            role = payload.get("role")
            text = _extract_message_text(payload).strip()
            if not text:
                continue

            if role == "user":
                # finalize previous turn before starting a new one
                for chunk in finalize_turn():
                    chunk_counts[chunk.chunk_type] += 1
                    yield chunk
                pending_user = _PendingUser(content=text, timestamp=ts, line_number=line_number)
            elif role == "assistant":
                # If no user is pending, emit assistant-only chunk
                if pending_user is None:
                    project_path = _project_path_from_cwd(last_cwd or session_cwd)
                    timestamp = ts or datetime.now(timezone.utc)
                    chunks = _create_assistant_only_chunks(
                        assistant_content=text,
                        project_path=project_path,
                        session_id=session_id,
                        timestamp=timestamp,
                        source_file=source_file,
                        source_line=line_number,
                        model=model,
                    )
                    for chunk in chunks:
                        chunk_counts[chunk.chunk_type] += 1
                        yield chunk
                else:
                    assistant_fragments.append(text)
            else:
                # Unknown role: still capture content
                project_path = _project_path_from_cwd(last_cwd or session_cwd)
                timestamp = ts or datetime.now(timezone.utc)
                chunks = _create_assistant_only_chunks(
                    assistant_content=text,
                    project_path=project_path,
                    session_id=session_id,
                    timestamp=timestamp,
                    source_file=source_file,
                    source_line=line_number,
                    model=model,
                )
                for chunk in chunks:
                    chunk_counts[chunk.chunk_type] += 1
                    yield chunk
            continue

        summary, ops = _summarize_tool_call(payload)
        if pending_user is not None:
            if summary:
                assistant_fragments.append(summary)
            if ops:
                pending_ops.extend(ops)
        else:
            if summary:
                project_path = _project_path_from_cwd(last_cwd or session_cwd)
                timestamp = ts or datetime.now(timezone.utc)
                chunks = _create_assistant_only_chunks(
                    assistant_content=summary,
                    project_path=project_path,
                    session_id=session_id,
                    timestamp=timestamp,
                    source_file=source_file,
                    source_line=line_number,
                    model=model,
                )
                if chunks:
                    first_chunk = chunks[0]
                    for chunk in chunks:
                        chunk_counts[chunk.chunk_type] += 1
                        yield chunk
                    if ops:
                        file_chunks = _create_file_change_chunks(
                            ops,
                            project_path=project_path,
                            session_id=session_id,
                            timestamp=timestamp,
                            source_file=source_file,
                            source_line=line_number,
                            parent_chunk_id=first_chunk.id,
                        )
                        for chunk in file_chunks:
                            chunk_counts[chunk.chunk_type] += 1
                            yield chunk

    # Finalize any pending user turn
    for chunk in finalize_turn():
        chunk_counts[chunk.chunk_type] += 1
        yield chunk

    total = sum(chunk_counts.values())
    logger.info(
        f"Completed Codex chunking {file_path.name}: {total} chunks "
        f"(turns={chunk_counts['turn']}, file_changes={chunk_counts['file_change']})"
    )
