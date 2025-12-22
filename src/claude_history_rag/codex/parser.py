"""Parser for Codex session JSONL files."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

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

        with open(file_path, encoding="utf-8", errors="strict") as f:
            for line_number, line in enumerate(f, start=1):
                if line_number <= start_line:
                    continue

                if len(line) > MAX_LINE_LENGTH:
                    logger.warning(
                        f"Line {line_number} exceeds maximum length "
                        f"({len(line)} > {MAX_LINE_LENGTH}), skipping"
                    )
                    continue

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON at line {line_number}: {e}")
                    continue

                if not _check_json_depth(data):
                    logger.warning(
                        f"JSON at line {line_number} exceeds maximum depth of {MAX_JSON_DEPTH}"
                    )
                    continue

                if "type" not in data:
                    logger.warning(f"Missing 'type' field at line {line_number}")
                    continue

                yield data, line_number
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except PermissionError:
        logger.error(f"Permission denied: {file_path}")
    except (OSError, ValueError) as e:
        logger.exception(f"Error reading file {file_path}: {e}")
