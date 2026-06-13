"""Chunking engine for creating embeddable chunks from history entries."""

import hashlib
import logging
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from claude_history_rag.models import AssistantMessage, Chunk, HistoryEntry, UserMessage
from claude_history_rag.parser import (
    decode_project_path,
    extract_file_operations,
    extract_text_content,
    get_project_name,
    parse_jsonl_file,
)

logger = logging.getLogger(__name__)

# Maximum length for truncated user content in file change chunks
USER_CONTENT_TRUNCATE_LENGTH = 200

# Chunk splitting configuration
# bge-m3 has 8192 token limit. Token ratios vary widely:
# - English text: ~4 chars/token (32K theoretical max)
# - Code/special chars: ~1.3 chars/token (10.6K theoretical max)
# Use very conservative 8K to handle worst-case (dense code, unicode, etc.)
MAX_CHUNK_CONTENT_LENGTH = 8000
CHUNK_OVERLAP = 1500  # ~19% overlap to preserve context at boundaries


def split_content(
    content: str, max_length: int = MAX_CHUNK_CONTENT_LENGTH, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split content into overlapping chunks that fit within embedding model limits.

    Tries to split at sentence/paragraph boundaries for better semantic coherence.
    Returns a list of content strings, each within max_length.
    """
    if len(content) <= max_length:
        return [content]

    chunks = []
    start = 0

    while start < len(content):
        # Determine end position for this chunk
        end = start + max_length

        if end >= len(content):
            # Last chunk - take everything remaining
            chunks.append(content[start:])
            break

        # Try to find a good break point (paragraph or sentence boundary)
        # Look in the last 1000 chars of the chunk for a break point
        search_start = max(start, end - 1000)
        search_region = content[search_start:end]

        # Prefer paragraph breaks (double newline)
        para_break = search_region.rfind("\n\n")
        if para_break > 0:
            break_pos = search_start + para_break + 2  # Include the newlines
        else:
            # Fall back to sentence boundaries
            last_period = search_region.rfind(". ")
            last_exclaim = search_region.rfind("! ")
            last_question = search_region.rfind("? ")
            last_newline = search_region.rfind("\n")

            best_break = max(last_period, last_exclaim, last_question, last_newline)
            break_pos = search_start + best_break + 1 if best_break > 0 else end

        chunks.append(content[start:break_pos])

        # Next chunk starts with overlap from the previous chunk
        # But don't go backwards past where we just cut
        start = max(start + 1, break_pos - overlap)

    return chunks


def generate_chunk_id(content: str, session_id: str, timestamp: str) -> str:
    """Generate a unique chunk ID from content hash.

    Creates a 16-character hex identifier by hashing the content, session ID,
    and timestamp together. The truncation to 16 characters provides sufficient
    uniqueness (2^64 possibilities) while keeping IDs compact for storage.

    Args:
        content: The chunk content to hash
        session_id: Session identifier for uniqueness
        timestamp: Timestamp string for temporal uniqueness

    Returns:
        16-character hex string identifier
    """
    data = f"{content}{session_id}{timestamp}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def create_turn_chunks(
    user_entry: HistoryEntry,
    assistant_entry: HistoryEntry,
    project_path: str,
    source_file: str,
    source_line: int,
) -> list[Chunk]:
    """Create turn chunk(s) from user + assistant message pair.

    If the content exceeds MAX_CHUNK_CONTENT_LENGTH, it will be split into
    multiple overlapping chunks with part numbers (e.g., "Part 1/3").

    Returns:
        List of Chunks (empty if no content, multiple if content was split).
    """
    user_content = ""
    if user_entry.message and isinstance(user_entry.message, UserMessage):
        user_content = extract_text_content(user_entry.message).strip()

    assistant_content = ""
    model = None
    if assistant_entry.message and isinstance(assistant_entry.message, AssistantMessage):
        assistant_content = extract_text_content(assistant_entry.message).strip()
        model = assistant_entry.message.model

    # Validate that we have actual content
    if not user_content and not assistant_content:
        logger.warning(
            f"Skipping empty turn chunk at line {source_line} in {source_file} - both user and assistant content are empty"
        )
        return []

    # Build chunk content with context
    project_name = get_project_name(project_path)
    timestamp = assistant_entry.timestamp or user_entry.timestamp
    if not timestamp:
        logger.warning(f"Missing timestamp for turn at line {source_line}, using current UTC time")
        timestamp = datetime.now(timezone.utc)

    # Build content based on what's available
    if user_content and assistant_content:
        content = f"In project {project_name}, user asked:\n{user_content}\n\nAssistant responded:\n{assistant_content}"
    elif user_content:
        content = (
            f"In project {project_name}, user asked:\n{user_content}\n\n[No assistant response]"
        )
    else:
        content = f"In project {project_name}, assistant response:\n{assistant_content}"

    session_id = user_entry.sessionId or assistant_entry.sessionId or "unknown"

    # Split content if it exceeds the limit
    content_parts = split_content(content)
    total_parts = len(content_parts)

    if total_parts > 1:
        logger.info(
            f"Split large turn chunk into {total_parts} parts at line {source_line} "
            f"(original size: {len(content)} chars)"
        )

    chunks = []
    # Generate a base ID for linking related chunks
    base_id = generate_chunk_id(content, session_id, str(timestamp))

    for part_num, part_content in enumerate(content_parts, start=1):
        # Add part indicator if content was split
        if total_parts > 1:
            part_prefix = f"[Part {part_num}/{total_parts}] "
            part_content = part_prefix + part_content

        # Generate unique ID for each part
        if total_parts > 1:
            chunk_id = generate_chunk_id(
                part_content, session_id, str(timestamp) + f"_part{part_num}"
            )
        else:
            chunk_id = base_id

        chunks.append(
            Chunk(
                id=chunk_id,
                content=part_content,
                chunk_type="turn",
                session_id=session_id,
                project_path=project_path,
                project_name=project_name,
                timestamp=timestamp,
                user_uuid=user_entry.uuid,
                assistant_uuid=assistant_entry.uuid,
                model=model,
                source_file=source_file,
                source_line=source_line,
                # Link split chunks together
                parent_chunk_id=base_id if total_parts > 1 and part_num > 1 else None,
            )
        )

    return chunks


def create_summary_chunk(
    entry: HistoryEntry,
    project_path: str,
    source_file: str,
    source_line: int,
) -> Chunk | None:
    """Create a summary chunk from a compaction event.

    Handles both formats:
    - Legacy (Claude Code < 2.1): a dedicated ``type: "summary"`` entry whose
      text lives in the top-level ``summary`` field.
    - Current (Claude Code >= 2.1): an ordinary ``user``/``assistant`` entry
      flagged with ``isCompactSummary`` whose text lives in ``message.content``.

    Returns:
        Chunk if summary content exists, None otherwise.
    """
    # Resolve summary text from whichever format produced this entry.
    summary_text = entry.summary
    if (not summary_text or not summary_text.strip()) and entry.message is not None:
        summary_text = extract_text_content(entry.message)

    # Validate summary content
    if not summary_text or not summary_text.strip():
        logger.warning(
            f"Skipping empty summary chunk at line {source_line} in {source_file} - summary content is empty"
        )
        return None
    summary_text = summary_text.strip()

    project_name = get_project_name(project_path)
    timestamp = entry.timestamp
    if not timestamp:
        logger.warning(
            f"Missing timestamp for summary at line {source_line}, using current UTC time"
        )
        timestamp = datetime.now(timezone.utc)
    session_id = entry.sessionId or "unknown"

    content = f"Session summary for {project_name}:\n{summary_text}"
    chunk_id = generate_chunk_id(content, session_id, str(timestamp))

    return Chunk(
        id=chunk_id,
        content=content,
        chunk_type="summary",
        session_id=session_id,
        project_path=project_path,
        project_name=project_name,
        timestamp=timestamp,
        source_file=source_file,
        source_line=source_line,
    )


def create_file_change_chunks(
    assistant_entry: HistoryEntry,
    user_content: str,
    project_path: str,
    source_file: str,
    source_line: int,
    parent_chunk_id: str | None,
) -> list[Chunk]:
    """Create file change chunks from assistant message tool uses."""
    chunks = []

    if not assistant_entry.message or not isinstance(assistant_entry.message, AssistantMessage):
        return chunks

    operations = extract_file_operations(assistant_entry.message)
    project_name = get_project_name(project_path)
    timestamp = assistant_entry.timestamp
    if not timestamp:
        logger.warning(
            f"Missing timestamp for file change at line {source_line}, using current UTC time"
        )
        timestamp = datetime.now(timezone.utc)
    session_id = assistant_entry.sessionId or "unknown"

    for idx, op in enumerate(operations):
        # Build contextual content
        truncated = user_content[:USER_CONTENT_TRUNCATE_LENGTH] + (
            "..." if len(user_content) > USER_CONTENT_TRUNCATE_LENGTH else ""
        )
        content = (
            f"In project {project_name}, file {op['file_path']} was {op['operation']}. "
            f"{op['summary']}. "
            f"This was in response to: {truncated}"
        )

        # Include tool_id to ensure uniqueness when same file is modified multiple times
        # Fall back to index if tool_id is missing to guarantee uniqueness
        tool_id = op.get("tool_id") or f"idx{idx}"
        chunk_id = generate_chunk_id(
            content, session_id, str(timestamp) + op["file_path"] + tool_id
        )

        chunks.append(
            Chunk(
                id=chunk_id,
                content=content,
                chunk_type="file_change",
                session_id=session_id,
                project_path=project_path,
                project_name=project_name,
                timestamp=timestamp,
                assistant_uuid=assistant_entry.uuid,
                file_path=op["file_path"],
                operation=op["operation"],
                model=assistant_entry.message.model,
                source_file=source_file,
                source_line=source_line,
                parent_chunk_id=parent_chunk_id,
            )
        )

    return chunks


def chunk_session_file(
    file_path: Path,
    start_line: int = 0,
) -> Iterator[Chunk]:
    """Process a session JSONL file and yield chunks.

    Args:
        file_path: Path to the JSONL file
        start_line: Line number to start from (for incremental processing)

    Yields:
        Chunk objects ready for embedding
    """
    logger.debug(f"Starting chunking: {file_path} from line {start_line}")

    # Decode project path from directory name
    project_dir = file_path.parent.name
    project_path = decode_project_path(project_dir)
    source_file = str(file_path)

    chunk_counts: dict[str, int] = defaultdict(int)

    # Track pending user message for pairing
    pending_user: tuple[HistoryEntry, int] | None = None

    for entry, line_number in parse_jsonl_file(file_path, start_line):
        # Current Claude Code (>=2.1) marks compaction summaries as ordinary
        # user/assistant entries with isCompactSummary=True rather than a
        # dedicated "summary" type. Route these to a summary chunk and skip
        # the turn-pairing logic so the summary text isn't mislabeled as a turn.
        if entry.isCompactSummary:
            summary_chunk = create_summary_chunk(
                entry=entry,
                project_path=project_path,
                source_file=source_file,
                source_line=line_number,
            )
            if summary_chunk is not None:
                yield summary_chunk
                chunk_counts["summary"] += 1
            continue

        if entry.type == "user":
            pending_user = (entry, line_number)

        elif entry.type == "assistant" and pending_user is not None:
            user_entry, user_line = pending_user

            # Extract user content for file change context
            user_content = ""
            if user_entry.message and isinstance(user_entry.message, UserMessage):
                user_content = extract_text_content(user_entry.message)

            # Create turn chunk(s) - may be split if content is too large
            turn_chunks = create_turn_chunks(
                user_entry=user_entry,
                assistant_entry=entry,
                project_path=project_path,
                source_file=source_file,
                source_line=user_line,
            )

            # Only process if we have valid turn chunks
            if turn_chunks:
                # Use first chunk's ID as parent for file change chunks
                first_turn_chunk = turn_chunks[0]

                # Create file change chunks with parent reference
                file_chunks = create_file_change_chunks(
                    assistant_entry=entry,
                    user_content=user_content,
                    project_path=project_path,
                    source_file=source_file,
                    source_line=line_number,
                    parent_chunk_id=first_turn_chunk.id,
                )

                # Update first turn chunk with child IDs
                if file_chunks:
                    turn_chunks[0] = Chunk(
                        **first_turn_chunk.model_dump(exclude={"child_chunk_ids"}),
                        child_chunk_ids=[c.id for c in file_chunks],
                    )

                # Yield all turn chunks
                yield from turn_chunks
                chunk_counts["turn"] += len(turn_chunks)

                # Then yield file change chunks
                if file_chunks:
                    yield from file_chunks
                    chunk_counts["file_change"] += len(file_chunks)

            pending_user = None

        elif entry.type == "assistant" and pending_user is None:
            # Log warning for unpaired assistant messages
            session_id = entry.sessionId or "unknown"
            logger.warning(
                f"Unpaired assistant message at line {line_number} in {source_file} "
                f"(session_id={session_id}, uuid={entry.uuid})"
            )
            # Still extract file operations from unpaired assistant messages
            if entry.message and isinstance(entry.message, AssistantMessage):
                file_chunks = create_file_change_chunks(
                    assistant_entry=entry,
                    user_content="[unpaired assistant message]",
                    project_path=project_path,
                    source_file=source_file,
                    source_line=line_number,
                    parent_chunk_id=None,
                )
                if file_chunks:
                    yield from file_chunks
                    chunk_counts["file_change"] += len(file_chunks)

        elif entry.type == "summary":
            summary_chunk = create_summary_chunk(
                entry=entry,
                project_path=project_path,
                source_file=source_file,
                source_line=line_number,
            )
            if summary_chunk is not None:
                yield summary_chunk
                chunk_counts["summary"] += 1

        elif entry.type == "system":
            # System entries (init) don't need chunking
            pass

    # Log warning if file ends with unpaired user message
    if pending_user is not None:
        user_entry, user_line = pending_user
        session_id = user_entry.sessionId or "unknown"

        # Extract and truncate user content for debugging
        user_content = ""
        if user_entry.message and isinstance(user_entry.message, UserMessage):
            user_content = extract_text_content(user_entry.message)
        truncated_content = user_content[:USER_CONTENT_TRUNCATE_LENGTH] + (
            "..." if len(user_content) > USER_CONTENT_TRUNCATE_LENGTH else ""
        )

        logger.warning(
            f"File {source_file} ends with unpaired user message at line {user_line} "
            f"(session_id={session_id}, uuid={user_entry.uuid}). "
            f"Message: {truncated_content}"
        )

    total = sum(chunk_counts.values())
    logger.info(
        f"Completed chunking {file_path.name}: {total} chunks "
        f"(turns={chunk_counts['turn']}, file_changes={chunk_counts['file_change']}, "
        f"summaries={chunk_counts['summary']})"
    )
