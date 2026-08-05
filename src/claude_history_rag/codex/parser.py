"""Parser for Codex session JSONL files."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from claude_history_rag.parser import source_hash

logger = logging.getLogger(__name__)

# Resource limits to prevent exhaustion attacks
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB
MAX_LINE_LENGTH = 512 * 1024  # 512KB
MAX_JSON_DEPTH = 50


def _check_json_depth(obj: dict | list, depth: int = 0) -> bool:
    """Check if JSON object exceeds max depth."""
    if depth > MAX_JSON_DEPTH:
        return False
    if isinstance(obj, dict):
        return all(
            _check_json_depth(v, depth + 1) for v in obj.values() if isinstance(v, (dict, list))
        )
    if isinstance(obj, list):
        return all(
            _check_json_depth(item, depth + 1) for item in obj if isinstance(item, (dict, list))
        )
    return True


def parse_codex_jsonl_file(
    file_path: Path,
    start_line: int = 0,
) -> Iterator[tuple[dict, int]]:
    """Parse a Codex session JSONL file and yield (event, line_number) tuples."""
    try:
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds maximum allowed "
                f"({MAX_FILE_SIZE} bytes): {file_path}"
            )

        # newline="\n" so a line here means what byte counting means.
        with open(file_path, encoding="utf-8", errors="strict", newline="\n") as f:
            for line_number, line in enumerate(f, start=1):
                if line_number <= start_line:
                    continue

                if len(line) > MAX_LINE_LENGTH:
                    logger.warning(
                        "Skipped line: reason=line_exceeds_max_length line=%d bytes=%d max_bytes=%d",
                        line_number,
                        len(line),
                        MAX_LINE_LENGTH,
                    )
                    continue

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    # A JSONDecodeError message quotes the offending document.
                    logger.warning(
                        "Skipped line: reason=json_invalid line=%d position=%d",
                        line_number,
                        e.pos,
                    )
                    continue

                if not _check_json_depth(data):
                    logger.warning(
                        "Skipped line: reason=json_exceeds_max_depth line=%d max_depth=%d",
                        line_number,
                        MAX_JSON_DEPTH,
                    )
                    continue

                # A JSONL line may decode to any JSON value, not just an object.
                # Testing membership against a bare scalar raises TypeError out
                # of this generator and loses every remaining line in the file,
                # and a non-mapping that happens to support `in` reaches the
                # chunker and fails there instead.
                if not isinstance(data, dict) or "type" not in data:
                    logger.warning("Skipped line: reason=missing_type_field line=%d", line_number)
                    continue

                yield data, line_number
    # Diagnosed and re-raised, never swallowed: a generator that returns
    # normally after a partial read is indistinguishable from one that consumed
    # the whole source, and the cursor commits the whole-source bound.
    except UnicodeDecodeError:
        logger.error(
            "Failed reading source: reason=utf8_decode_error source=%s",
            source_hash(str(file_path)),
        )
        raise
    except FileNotFoundError:
        logger.error(
            "Failed reading source: reason=file_not_found source=%s",
            source_hash(str(file_path)),
        )
        raise
    except PermissionError:
        logger.error(
            "Failed reading source: reason=permission_denied source=%s",
            source_hash(str(file_path)),
        )
        raise
    except (OSError, ValueError) as e:
        logger.error(
            "Failed reading source: reason=read_failed source=%s error_type=%s",
            source_hash(str(file_path)),
            type(e).__name__,
        )
        raise
