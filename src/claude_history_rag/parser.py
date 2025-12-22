"""JSONL parser for Claude Code history files."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from claude_history_rag.models import AssistantMessage, HistoryEntry, UserMessage

logger = logging.getLogger(__name__)

# Resource limits to prevent exhaustion attacks
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB (for large conversation files)
MAX_LINE_LENGTH = 512 * 1024  # 512KB
MAX_JSON_DEPTH = 50  # Prevent stack overflow from deeply nested JSON


def decode_project_path(encoded: str) -> str:
    """Decode Claude Code's project path encoding.

    Example: "-Users-brandon-projects-aidition" -> "/Users/brandon/projects/aidition"

    Security: Validates against path traversal attempts in the encoded string.
    Note: This returns the decoded path as-is without filesystem resolution,
    since it's used for metadata purposes. Actual file access is validated
    separately through config.py's path validators.
    """
    # Check for traversal sequences in encoded form
    if ".." in encoded:
        logger.warning(f"Path traversal detected in encoded path: {repr(encoded[:50])}")
        return "/invalid/path"

    if encoded.startswith("-"):
        encoded = encoded[1:]
    decoded = "/" + encoded.replace("-", "/")

    # Additional validation: ensure no traversal after decoding
    # and path is absolute
    if ".." in decoded:
        logger.warning(f"Path traversal detected after decoding: {repr(decoded[:50])}")
        return "/invalid/path"

    if not decoded.startswith("/"):
        logger.warning(f"Non-absolute path after decoding: {repr(decoded[:50])}")
        return "/invalid/path"

    return decoded


def get_project_name(project_path: str) -> str:
    """Extract human-readable project name from path."""
    return Path(project_path).name


def parse_message(msg_data: dict, msg_type: str) -> UserMessage | AssistantMessage | None:
    """Parse message data into appropriate model."""
    try:
        if msg_type == "user":
            return UserMessage(**msg_data)
        elif msg_type == "assistant":
            return AssistantMessage(**msg_data)
        return None
    except ValidationError as e:
        logger.warning(f"Failed to validate {msg_type} message: {e}")
        return None


def _check_json_depth(obj: dict | list, depth: int = 0) -> bool:
    """Check if JSON object exceeds max depth."""
    if depth > MAX_JSON_DEPTH:
        return False
    if isinstance(obj, dict):
        return all(
            _check_json_depth(v, depth + 1) for v in obj.values() if isinstance(v, (dict, list))
        )
    elif isinstance(obj, list):
        return all(
            _check_json_depth(item, depth + 1) for item in obj if isinstance(item, (dict, list))
        )
    return True


def parse_entry(line: str, line_number: int) -> HistoryEntry | None:
    """Parse a single JSONL line into a HistoryEntry."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON at line {line_number}: {e}")
        return None

    # Check for excessively nested JSON
    if not _check_json_depth(data):
        logger.warning(f"JSON at line {line_number} exceeds maximum depth of {MAX_JSON_DEPTH}")
        return None

    entry_type = data.get("type")
    if not entry_type:
        logger.warning(f"Missing 'type' field at line {line_number}")
        return None

    # Parse message if present
    if "message" in data:
        if data["message"] is not None:
            data["message"] = parse_message(data["message"], entry_type)
        else:
            data["message"] = None

    try:
        return HistoryEntry(**data)
    except (ValidationError, ValueError, TypeError, KeyError) as e:
        logger.warning(
            f"Failed to create HistoryEntry at line {line_number}: {e}, "
            f"type={entry_type}, keys={list(data.keys())}"
        )
        return None


def parse_jsonl_file(
    file_path: Path,
    start_line: int = 0,
) -> Iterator[tuple[HistoryEntry, int]]:
    """Parse a JSONL file and yield (entry, line_number) tuples.

    Args:
        file_path: Path to the JSONL file
        start_line: Line number to start from (for incremental reads)

    Yields:
        Tuples of (HistoryEntry, line_number)

    Raises:
        ValueError: If file exceeds MAX_FILE_SIZE or a line exceeds MAX_LINE_LENGTH
    """
    try:
        # Check file size before reading
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds maximum allowed "
                f"({MAX_FILE_SIZE} bytes): {file_path}"
            )

        try:
            with open(file_path, encoding="utf-8", errors="strict") as f:
                for line_number, line in enumerate(f, start=1):
                    if line_number <= start_line:
                        continue

                    # Check line length before processing
                    if len(line) > MAX_LINE_LENGTH:
                        logger.warning(
                            f"Line {line_number} exceeds maximum length "
                            f"({len(line)} > {MAX_LINE_LENGTH}), skipping"
                        )
                        continue

                    line = line.strip()
                    if not line:
                        continue

                    entry = parse_entry(line, line_number)
                    if entry:
                        yield entry, line_number
        except UnicodeDecodeError as e:
            logger.error(f"UTF-8 decode error in {file_path} at position {e.start}: {e.reason}")
            raise
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except PermissionError:
        logger.error(f"Permission denied: {file_path}")
    except (OSError, ValueError) as e:
        logger.exception(f"Error reading file {file_path}: {e}")


def extract_text_content(message: UserMessage | AssistantMessage) -> str:
    """Extract text content from a message."""
    if not isinstance(message, (UserMessage, AssistantMessage)):
        logger.warning(f"Invalid message type: {type(message)}")
        return ""

    if isinstance(message.content, str):
        return message.content

    # Handle list of content blocks
    if not isinstance(message.content, list):
        logger.warning(f"Unexpected content type: {type(message.content)}")
        return ""

    texts = []
    for block in message.content:
        # Validate block is a dict with a 'type' field
        if not isinstance(block, dict):
            logger.warning(f"Non-dict block encountered in message content: {type(block)}")
            continue

        block_type = block.get("type")
        if not isinstance(block_type, str):
            logger.warning(f"Invalid or missing 'type' field in content block: {type(block_type)}")
            continue

        if block_type == "text":
            text_val = block.get("text", "")
            texts.append(str(text_val) if text_val else "")
        elif block.get("type") == "tool_use":
            tool_name = str(block.get("name", "unknown"))
            tool_input = block.get("input")
            if not isinstance(tool_input, dict):
                tool_input = {}
            # Summarize tool use
            if tool_name in ("Read", "Edit", "Write"):
                file_path = str(tool_input.get("file_path", "unknown"))
                texts.append(f"[Used {tool_name} on {file_path}]")
            elif tool_name == "Bash":
                cmd_val = tool_input.get("command", "")
                # Sanitize command to prevent log injection
                cmd = str(cmd_val)[:100].replace("\n", " ").replace("\r", " ") if cmd_val else ""
                texts.append(f"[Ran command: {cmd}]")
            else:
                texts.append(f"[Used {tool_name}]")
        elif block.get("type") == "tool_result":
            # Include short tool results, summarize long ones
            content = block.get("content", "")
            if isinstance(content, str) and len(content) < 500:
                texts.append(f"[Result: {content}]")
            else:
                texts.append("[Tool result truncated]")

    return "\n".join(texts)


def extract_file_operations(message: AssistantMessage) -> list[dict]:
    """Extract file operations from tool_use blocks.

    Returns list of dicts with: file_path, operation, summary
    """
    operations = []

    if not isinstance(message.content, list):
        return operations

    for block in message.content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue

        tool_name = str(block.get("name", ""))
        tool_input = block.get("input")
        if not isinstance(tool_input, dict):
            tool_input = {}

        if tool_name == "Write":
            raw_path = tool_input.get("file_path", "unknown")
            file_path = str(raw_path) if isinstance(raw_path, str) else "unknown"
            operations.append(
                {
                    "file_path": file_path,
                    "operation": "write",
                    "summary": "Created/overwrote file",
                    "tool_id": block.get("id"),
                }
            )
        elif tool_name == "Edit":
            raw_path = tool_input.get("file_path", "unknown")
            file_path = str(raw_path) if isinstance(raw_path, str) else "unknown"

            # Validate that old_string and new_string exist
            if "old_string" not in tool_input or "new_string" not in tool_input:
                logger.warning(
                    f"Edit operation missing required parameters: {list(tool_input.keys())}"
                )
                continue

            old_val = tool_input.get("old_string", "")
            new_val = tool_input.get("new_string", "")
            # Only convert to string if it's actually a string type
            old_str = old_val[:100] if isinstance(old_val, str) else ""
            new_str = new_val[:100] if isinstance(new_val, str) else ""
            # Only add ellipsis if truncated
            old_suffix = "..." if isinstance(old_val, str) and len(old_val) > 100 else ""
            new_suffix = "..." if isinstance(new_val, str) and len(new_val) > 100 else ""
            operations.append(
                {
                    "file_path": file_path,
                    "operation": "edit",
                    "summary": f"Replaced '{old_str}{old_suffix}' with '{new_str}{new_suffix}'",
                    "tool_id": block.get("id"),
                }
            )

    return operations
