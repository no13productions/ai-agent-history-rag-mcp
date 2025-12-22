"""Chunking for Google Antigravity (.pb) history files.

Since we don't have the .proto definition, we attempt to extract text strings from the binary protobuf.
This is a "best effort" approach.
"""

import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from claude_history_rag.models import Chunk
from claude_history_rag.chunker import generate_chunk_id, split_content

logger = logging.getLogger(__name__)

MAX_CHUNK_CONTENT_LENGTH = 8000


def _extract_strings(data: bytes, min_length: int = 4) -> list[str]:
    """Extract printable strings from binary data."""
    # Find sequences of printable characters
    # This regex looks for 4 or more printable characters
    # excluding some common binary noise
    text = data.decode("utf-8", errors="ignore")
    # We filter for a reasonable set of characters
    # This is a heuristic to separate content from binary markers
    # We look for typical text patterns
    
    # Just return the whole decoded string but cleaned up a bit?
    # No, protobuf mixes binary tags. 
    # Let's try to just clean up non-printable chars
    clean_text = "".join(c if c.isprintable() or c in "\n\t\r" else " " for c in text)
    # Collapse multiple spaces
    clean_text = re.sub(r"\s+", " ", clean_text)
    
    return [clean_text]


def chunk_antigravity_file(file_path: Path, start_line: int = 0) -> Iterator[Chunk]:
    """Process an Antigravity .pb file and yield chunks."""
    logger.debug(f"Starting Antigravity chunking: {file_path}")
    
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
        
    session_id = file_path.stem # e.g. "uuid"
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
            project_path="/antigravity/unknown", # We don't know the project
            project_name="Antigravity Session",
            timestamp=timestamp,
            model="gemini-unknown",
            source_file=str(file_path),
            source_line=0, # Binary file
            parent_chunk_id=base_id if total_parts > 1 and part_num > 1 else None,
        )
        
        chunk_counts["history"] += 1
        yield chunk

    total = sum(chunk_counts.values())
    logger.info(f"Completed Antigravity chunking {file_path.name}: {total} chunks")
