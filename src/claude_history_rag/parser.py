"""JSONL parser for Claude Code history files."""

import hashlib
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

# Entry types the diagnostics may name verbatim. Anything else is caller
# controlled text that could carry conversation data or forge a log record.
_KNOWN_ENTRY_TYPES = frozenset(
    {"user", "assistant", "summary", "system"},
)


def _safe_entry_type(entry_type: object) -> str:
    """Return an entry type only when it is one of the known fixed values.

    The value comes from arbitrary JSON, so it may be unhashable (an object or
    an array). Testing membership without the isinstance guard raises TypeError
    from inside the caller's own except handler, which escapes it and aborts the
    whole file instead of skipping the one malformed entry.
    """
    if isinstance(entry_type, str) and entry_type in _KNOWN_ENTRY_TYPES:
        return entry_type
    return "other"


def source_hash(value: str) -> str:
    """Return the stable short digest used for every logged identifier.

    Defined here rather than in the chunker because the parser is the lowest
    layer that needs it, and the chunker already depends on this module.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


_source_hash = source_hash


def decode_project_path(encoded: str) -> str:
    """Decode Claude Code's project path encoding.

    Example: "-Users-youruser-projects-myapp" -> "/Users/youruser/projects/myapp"

    Security: Validates against path traversal attempts in the encoded string.
    Note: This returns the decoded path as-is without filesystem resolution,
    since it's used for metadata purposes. Actual file access is validated
    separately through config.py's path validators.
    """
    # Check for traversal sequences in encoded form. The encoded value is a
    # project path, so it is logged only as a digest.
    if ".." in encoded:
        logger.warning(
            "Rejected project path: reason=traversal_in_encoded_path source=%s",
            source_hash(encoded),
        )
        return "/invalid/path"

    if encoded.startswith("-"):
        encoded = encoded[1:]
    decoded = "/" + encoded.replace("-", "/")

    # Additional validation: ensure no traversal after decoding
    # and path is absolute
    if ".." in decoded:
        logger.warning(
            "Rejected project path: reason=traversal_after_decoding source=%s",
            source_hash(decoded),
        )
        return "/invalid/path"

    if not decoded.startswith("/"):
        logger.warning(
            "Rejected project path: reason=not_absolute_after_decoding source=%s",
            source_hash(decoded),
        )
        return "/invalid/path"

    return decoded


def get_project_name(project_path: str) -> str:
    """Extract human-readable project name from path."""
    return Path(project_path).name


def parse_message(msg_data: dict, msg_type: str) -> UserMessage | AssistantMessage | None:
    """Parse message data into appropriate model."""
    # A message field may decode to any JSON value. Unpacking a non-mapping with
    # ** raises TypeError, which the handler below does not catch, so it would
    # escape the parser and cost every remaining entry in the file rather than
    # just this one message.
    if not isinstance(msg_data, dict):
        # msg_type is arbitrary source text. Being a str is not enough: it can
        # carry newlines and forge whole log records, so it goes through the
        # same fixed-value allowlist every other site uses.
        logger.warning(
            "Skipped message: reason=message_not_an_object type=%s",
            _safe_entry_type(msg_type),
        )
        return None
    try:
        if msg_type == "user":
            return UserMessage(**msg_data)
        elif msg_type == "assistant":
            return AssistantMessage(**msg_data)
        return None
    except ValidationError as e:
        # A pydantic ValidationError renders the rejected input inside its own
        # message (`input_value=...`), so formatting the exception here would
        # publish the conversation content the entry carries.
        logger.warning(
            "Skipped message: reason=message_validation_failed type=%s error_count=%d",
            _safe_entry_type(msg_type),
            e.error_count(),
        )
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
        # A JSONDecodeError message quotes the offending document text.
        logger.warning(
            "Skipped line: reason=json_invalid line=%d position=%d",
            line_number,
            e.pos,
        )
        return None

    # Check for excessively nested JSON
    if not _check_json_depth(data):
        logger.warning(
            "Skipped line: reason=json_exceeds_max_depth line=%d max_depth=%d",
            line_number,
            MAX_JSON_DEPTH,
        )
        return None

    # A line may decode to any JSON value. Every access below assumes a mapping,
    # so a bare scalar or array would raise out of this generator and cost every
    # remaining line rather than just this one.
    if not isinstance(data, dict):
        logger.warning("Skipped line: reason=entry_not_an_object line=%d", line_number)
        return None

    entry_type = data.get("type")
    if not entry_type:
        logger.warning("Skipped line: reason=missing_type_field line=%d", line_number)
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
        # Neither the exception text nor the entry's key set may be emitted:
        # the first embeds rejected input, the second is caller-controlled.
        logger.warning(
            "Skipped entry: reason=entry_validation_failed line=%d type=%s error_type=%s keys=%d",
            line_number,
            _safe_entry_type(entry_type),
            type(e).__name__,
            len(data),
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
            # newline="\n" so a line here means exactly what it means to
            # _count_file_lines and _prefix_digest, which read bytes. Universal
            # newline translation would split on a bare \r that byte counting
            # does not, leaving the counted bound short of the parsed content.
            with open(file_path, encoding="utf-8", errors="strict", newline="\n") as f:
                for line_number, line in enumerate(f, start=1):
                    if line_number <= start_line:
                        continue

                    # Check line length before processing
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

                    entry = parse_entry(line, line_number)
                    if entry:
                        yield entry, line_number
        except UnicodeDecodeError as e:
            logger.error(
                "Failed reading source: reason=utf8_decode_error source=%s position=%d",
                _source_hash(str(file_path)),
                e.start,
            )
            raise
    # Every failure below is DIAGNOSED here and then re-raised. Swallowing any
    # of them lets this generator return normally after reading nothing or only
    # part of the source, which is indistinguishable to every caller from having
    # consumed the whole thing - and the bounds those callers commit were
    # derived from the whole thing. The defect this forecloses is a shape, not
    # one exception type, so no clause here may end without `raise`.
    except UnicodeDecodeError:
        # Already diagnosed at the inner site.
        raise
    except FileNotFoundError:
        logger.error(
            "Failed reading source: reason=file_not_found source=%s",
            _source_hash(str(file_path)),
        )
        raise
    except PermissionError:
        logger.error(
            "Failed reading source: reason=permission_denied source=%s",
            _source_hash(str(file_path)),
        )
        raise
    except (OSError, ValueError) as e:
        # No logger.exception here: a traceback republishes the frames' locals
        # in some handlers and the message text can carry file content.
        logger.error(
            "Failed reading source: reason=read_failed source=%s error_type=%s",
            _source_hash(str(file_path)),
            type(e).__name__,
        )
        raise


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

            # Validate that old_string and new_string exist. The key set is
            # model-generated conversation content, so only its size is logged.
            if "old_string" not in tool_input or "new_string" not in tool_input:
                logger.warning(
                    "Skipped tool use: reason=edit_missing_parameters keys=%d",
                    len(tool_input),
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
