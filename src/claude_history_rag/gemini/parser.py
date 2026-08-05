"""Parser for Gemini CLI session JSON files."""

import json
import logging
from pathlib import Path

from claude_history_rag.parser import source_hash

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
            logger.warning(
                "Rejected source: reason=json_exceeds_max_depth source=%s max_depth=%d",
                source_hash(str(file_path)),
                MAX_JSON_DEPTH,
            )
            return None

        return data
    except FileNotFoundError:
        logger.error(
            "Failed reading source: reason=file_not_found source=%s",
            source_hash(str(file_path)),
        )
    except PermissionError:
        logger.error(
            "Failed reading source: reason=permission_denied source=%s",
            source_hash(str(file_path)),
        )
    except (OSError, ValueError, json.JSONDecodeError) as e:
        # Neither the exception text nor a traceback may be emitted: a decode
        # error quotes the document and a traceback can republish frame locals.
        logger.error(
            "Failed reading source: reason=read_failed source=%s error_type=%s",
            source_hash(str(file_path)),
            type(e).__name__,
        )
    return None
