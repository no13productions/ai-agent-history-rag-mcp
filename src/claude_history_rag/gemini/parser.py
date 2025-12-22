"""Parser for Gemini CLI session JSON files."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Resource limits to prevent exhaustion attacks
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
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


def load_gemini_json_file(file_path: Path) -> dict | list | None:
    """Load a Gemini CLI JSON file (session or logs)."""
    try:
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds maximum allowed "
                f"({MAX_FILE_SIZE} bytes): {file_path}"
            )

        with open(file_path, encoding="utf-8", errors="strict") as f:
            data = json.load(f)

        if not _check_json_depth(data):
            logger.warning(f"JSON in {file_path} exceeds maximum depth of {MAX_JSON_DEPTH}")
            return None

        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except PermissionError:
        logger.error(f"Permission denied: {file_path}")
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logger.exception(f"Error reading file {file_path}: {e}")
    return None
