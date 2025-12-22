"""Tests for JSONL parser."""

from pathlib import Path

from claude_history_rag.parser import (
    decode_project_path,
    extract_text_content,
    get_project_name,
    parse_jsonl_file,
)


def test_decode_project_path():
    """Test project path decoding."""
    assert decode_project_path("-Users-brandon-projects-myapp") == "/Users/brandon/projects/myapp"
    # On macOS, /home resolves to /System/Volumes/Data/home due to firmlinks
    # So we check that the path ends correctly rather than exact match
    result = decode_project_path("-home-user-code")
    assert result.endswith("/home/user/code")


def test_get_project_name():
    """Test project name extraction."""
    assert get_project_name("/Users/brandon/projects/myapp") == "myapp"
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
