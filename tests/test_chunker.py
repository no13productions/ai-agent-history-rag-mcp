"""Tests for chunking engine."""

from pathlib import Path

from claude_history_rag.chunker import chunk_session_file


def test_chunk_session_file(sample_session_path: Path, tmp_path: Path):
    """Test chunking a session file."""
    # Create a mock project structure
    project_dir = tmp_path / "-Users-test-myproject"
    project_dir.mkdir()

    # Copy fixture to project dir
    session_file = project_dir / "test-session.jsonl"
    session_file.write_text(sample_session_path.read_text())

    chunks = list(chunk_session_file(session_file))

    # Should have: 2 turn chunks + 1 file_change chunk + 1 summary chunk
    chunk_types = [c.chunk_type for c in chunks]
    assert "turn" in chunk_types
    assert "summary" in chunk_types

    # Check turn chunk content
    turn_chunks = [c for c in chunks if c.chunk_type == "turn"]
    assert len(turn_chunks) >= 1
    assert "authentication" in turn_chunks[0].content.lower()

    # Check project path decoding
    assert all(c.project_path == "/Users/test/myproject" for c in chunks)
    assert all(c.project_name == "myproject" for c in chunks)


def test_chunk_file_change_extraction(sample_session_path: Path, tmp_path: Path):
    """Test that file change chunks are extracted."""
    project_dir = tmp_path / "-Users-test-myproject"
    project_dir.mkdir()
    session_file = project_dir / "test-session.jsonl"
    session_file.write_text(sample_session_path.read_text())

    chunks = list(chunk_session_file(session_file))

    file_changes = [c for c in chunks if c.chunk_type == "file_change"]

    # Should have at least one file change (the Edit to auth.py)
    assert len(file_changes) >= 1

    auth_change = next((c for c in file_changes if "auth.py" in (c.file_path or "")), None)
    assert auth_change is not None
    assert auth_change.operation == "edit"


def test_multiple_edits_same_file_unique_ids(tmp_path: Path):
    """Test that multiple edits to the same file in one turn generate unique chunk IDs."""
    project_dir = tmp_path / "-Users-test-myproject"
    project_dir.mkdir()
    session_file = project_dir / "test-session.jsonl"

    # Create a session with multiple edits to the same file in one turn
    session_content = """{"type":"system","subtype":"init","cwd":"/Users/test/myproject","sessionId":"test-session-456","timestamp":"2025-12-14T12:00:00.000Z"}
{"type":"user","message":{"role":"user","content":"Refactor the config file"},"uuid":"user-msg-001","timestamp":"2025-12-14T12:01:00.000Z","sessionId":"test-session-456"}
{"type":"assistant","message":{"id":"asst-001","role":"assistant","model":"claude-sonnet-4-20250514","content":[{"type":"text","text":"I'll make multiple changes to config.py"},{"type":"tool_use","id":"tool-edit-001","name":"Edit","input":{"file_path":"/Users/test/myproject/config.py","old_string":"DEBUG = False","new_string":"DEBUG = True"}},{"type":"tool_use","id":"tool-edit-002","name":"Edit","input":{"file_path":"/Users/test/myproject/config.py","old_string":"PORT = 8000","new_string":"PORT = 3000"}},{"type":"tool_use","id":"tool-edit-003","name":"Edit","input":{"file_path":"/Users/test/myproject/config.py","old_string":"TIMEOUT = 30","new_string":"TIMEOUT = 60"}}]},"uuid":"asst-msg-001","parentUuid":"user-msg-001","timestamp":"2025-12-14T12:01:30.000Z","sessionId":"test-session-456"}
"""
    session_file.write_text(session_content)

    chunks = list(chunk_session_file(session_file))
    file_changes = [c for c in chunks if c.chunk_type == "file_change"]

    # Should have 3 file change chunks (all for config.py)
    assert len(file_changes) == 3

    # All should be for the same file
    assert all(c.file_path == "/Users/test/myproject/config.py" for c in file_changes)
    assert all(c.operation == "edit" for c in file_changes)

    # Critical: All chunk IDs must be unique despite being for the same file
    chunk_ids = [c.id for c in file_changes]
    assert len(chunk_ids) == len(set(chunk_ids)), f"Duplicate chunk IDs found: {chunk_ids}"
