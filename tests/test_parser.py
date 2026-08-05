"""Tests for JSONL parser."""

import json
import logging
from pathlib import Path

import pytest

from claude_history_rag.parser import (
    decode_project_path,
    extract_text_content,
    get_project_name,
    parse_entry,
    parse_jsonl_file,
    parse_message,
)


@pytest.mark.parametrize(
    "entry_type",
    [{"nested": "object"}, ["nested", "array"], 5, True, "unrecognized"],
)
def test_unusable_entry_type_skips_only_that_entry(entry_type):
    """A malformed `type` must cost one entry, never the whole file.

    The value comes from arbitrary JSON, so it can be unhashable. Testing it
    against a set of known types without a type guard raises TypeError from
    inside the parser's own except handler, which escapes it, propagates out of
    the generator, and loses every remaining entry in the file.
    """
    record = json.dumps({"type": entry_type, "message": {"role": "user", "content": "x"}})

    # Must not raise, whatever the value is.
    parse_entry(record, 1)


def test_malformed_entry_does_not_abandon_the_rest_of_the_file(tmp_path: Path):
    """One unusable line must not cost the surviving lines around it."""
    good = {
        "type": "user",
        "sessionId": "s",
        "uuid": "u",
        "timestamp": "2026-01-01T00:00:00Z",
        "message": {"role": "user", "content": "hello"},
    }
    source = tmp_path / "session.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(good),
                json.dumps({"type": {"unhashable": "object"}, "message": None}),
                json.dumps({"type": ["unhashable", "array"], "message": None}),
                json.dumps(good),
            ]
        )
        + "\n"
    )

    entries = list(parse_jsonl_file(source))

    assert [line for _, line in entries] == [1, 4]


@pytest.mark.parametrize("value", [42, None, True, "text", [1, 2], 3.5])
def test_non_object_line_skips_only_that_line(value, tmp_path: Path):
    """A JSONL line may decode to any JSON value, not only an object.

    Every field access assumes a mapping, so an unguarded line raises out of the
    generator and costs every remaining line in the file.
    """
    good = {
        "type": "user",
        "sessionId": "s",
        "uuid": "u",
        "timestamp": "2026-01-01T00:00:00Z",
        "message": {"role": "user", "content": "hello"},
    }
    source = tmp_path / "session.jsonl"
    source.write_text("\n".join([json.dumps(good), json.dumps(value), json.dumps(good)]) + "\n")

    assert [line for _, line in parse_jsonl_file(source)] == [1, 3]


@pytest.mark.parametrize("message", [5, "text", [1, 2], True])
def test_non_object_message_skips_only_that_entry(message, tmp_path: Path):
    """Unpacking a non-mapping message with ** raises past the handler."""
    good = {
        "type": "user",
        "sessionId": "s",
        "uuid": "u",
        "timestamp": "2026-01-01T00:00:00Z",
        "message": {"role": "user", "content": "hello"},
    }
    bad = {**good, "message": message}
    source = tmp_path / "session.jsonl"
    source.write_text("\n".join([json.dumps(good), json.dumps(bad), json.dumps(good)]) + "\n")

    lines = [line for _, line in parse_jsonl_file(source)]
    # The malformed entry may still yield with message=None, but it must never
    # cost the surrounding lines.
    assert 1 in lines
    assert 3 in lines


def test_aborted_read_raises_instead_of_returning_an_empty_iteration(tmp_path: Path):
    """A partial read must be distinguishable from a complete one.

    A generator that returns normally after aborting mid-stream is identical,
    from the caller's side, to one that consumed the whole source. The bounds
    the caller then commits were derived from the whole source, so the cursor
    advances over content that was never parsed.
    """
    good = json.dumps(
        {
            "type": "user",
            "sessionId": "s",
            "uuid": "u",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"role": "user", "content": "hello"},
        }
    ).encode()
    source = tmp_path / "session.jsonl"
    source.write_bytes((good + b"\n") * 4 + b"caf\xe9 undecodable\n" + (good + b"\n") * 3)

    with open(source, "rb") as raw:
        assert sum(1 for _ in raw) == 8

    with pytest.raises(UnicodeDecodeError):
        list(parse_jsonl_file(source))


def test_missing_and_unreadable_sources_also_raise(tmp_path: Path):
    """The swallow shape is foreclosed for every failure, not one exception type."""
    with pytest.raises(FileNotFoundError):
        list(parse_jsonl_file(tmp_path / "absent.jsonl"))


@pytest.mark.parametrize(
    "entry_type",
    ["evil\nCRITICAL forged administrative record", "x" * 500, "user\r\nFORGED"],
)
def test_entry_type_can_never_forge_a_log_record(
    entry_type: str,
    caplog: pytest.LogCaptureFixture,
):
    """`type` is arbitrary source text; being a str is not enough to emit it."""
    with caplog.at_level(logging.DEBUG):
        assert parse_message("not-a-mapping", entry_type) is None
        assert parse_message({"role": "user", "content": {"k": "v"}}, entry_type) is None

    assert caplog.records
    for record in caplog.records:
        assert "\n" not in record.getMessage()
        assert "\r" not in record.getMessage()
    assert "CRITICAL forged administrative record" not in caplog.text
    assert "FORGED" not in caplog.text
    assert "type=other" in caplog.text


def test_project_path_rejection_does_not_disclose_the_path(caplog: pytest.LogCaptureFixture):
    """A rejected project path is conversation-identifying and must be hashed."""
    secret = "-Users-bob-..-SECRET_PROJECT_DIRECTORY"

    with caplog.at_level(logging.DEBUG):
        assert decode_project_path(secret) == "/invalid/path"

    assert "SECRET_PROJECT_DIRECTORY" not in caplog.text
    assert secret not in caplog.text
    assert "reason=traversal_in_encoded_path" in caplog.text


def test_decode_project_path():
    """Test project path decoding."""
    assert decode_project_path("-Users-youruser-projects-myapp") == "/Users/youruser/projects/myapp"
    # On macOS, /home resolves to /System/Volumes/Data/home due to firmlinks
    # So we check that the path ends correctly rather than exact match
    result = decode_project_path("-home-user-code")
    assert result.endswith("/home/user/code")


def test_get_project_name():
    """Test project name extraction."""
    assert get_project_name("/Users/youruser/projects/myapp") == "myapp"
    assert get_project_name("/home/user/code") == "code"


def test_parse_jsonl_file(sample_session_path: Path):
    """Test parsing sample JSONL file."""
    entries = list(parse_jsonl_file(sample_session_path))

    assert len(entries) == 6  # system, user, assistant, user, assistant, summary

    # Check entry types
    types = [e[0].type for e in entries]
    assert types == ["system", "user", "assistant", "user", "assistant", "summary"]

    # Check system entry
    system_entry = entries[0][0]
    assert system_entry.subtype == "init"
    assert system_entry.sessionId == "test-session-123"

    # Check user entry
    user_entry = entries[1][0]
    assert user_entry.message is not None
    assert "authentication" in extract_text_content(user_entry.message)

    # Check summary entry
    summary_entry = entries[5][0]
    assert summary_entry.summary is not None
    assert "logout" in summary_entry.summary


def test_parse_jsonl_file_incremental(sample_session_path: Path):
    """Test incremental parsing with start_line."""
    # Start from line 3 (skip first 2 entries)
    entries = list(parse_jsonl_file(sample_session_path, start_line=2))

    assert len(entries) == 4  # assistant, user, assistant, summary
    assert entries[0][0].type == "assistant"
