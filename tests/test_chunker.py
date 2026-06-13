"""Tests for chunking engine."""

from pathlib import Path

from claude_history_rag.antigravity.chunker import chunk_antigravity_file
from claude_history_rag.antigravity.watcher import _is_antigravity_file
from claude_history_rag.chatgpt.chunker import chunk_chatgpt_export_file
from claude_history_rag.chatgpt.watcher import _is_chatgpt_export_file
from claude_history_rag.chunker import chunk_session_file
from claude_history_rag.claude_app.chunker import chunk_claude_app_export_file
from claude_history_rag.claude_app.watcher import _is_claude_app_export_file


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


def test_chunk_compact_summary(sample_session_compact_path: Path, tmp_path: Path):
    """Current Claude Code (>=2.1) compaction summaries are emitted as
    user entries flagged isCompactSummary, not a dedicated "summary" type.

    Verify they still produce a summary chunk (so get_session_summary works)
    and that the summary text is captured from message.content.
    """
    project_dir = tmp_path / "-Users-test-myproject"
    project_dir.mkdir()
    session_file = project_dir / "compact-session.jsonl"
    session_file.write_text(sample_session_compact_path.read_text())

    chunks = list(chunk_session_file(session_file))
    chunk_types = [c.chunk_type for c in chunks]

    # A summary chunk must be produced from the isCompactSummary entry.
    summary_chunks = [c for c in chunks if c.chunk_type == "summary"]
    assert len(summary_chunks) == 1, f"expected 1 summary chunk, got {chunk_types}"
    assert "logout" in summary_chunks[0].content.lower()
    assert summary_chunks[0].session_id == "compact-session-123"

    # The summary entry must NOT be double-counted as a turn chunk.
    summary_text_in_turns = [
        c for c in chunks if c.chunk_type == "turn" and "ran out of context" in c.content.lower()
    ]
    assert not summary_text_in_turns, "compact summary leaked into a turn chunk"

    # Regular turns and file changes from the same file still work.
    assert "turn" in chunk_types
    file_changes = [c for c in chunks if c.chunk_type == "file_change"]
    assert any("auth.py" in (c.file_path or "") for c in file_changes)


def test_chunk_antigravity_jsonl_transcript(tmp_path: Path):
    """Modern Antigravity stores JSONL transcripts under brain/<id>/logs."""
    transcript_dir = tmp_path / "brain" / "session-123" / ".system_generated" / "logs"
    transcript_dir.mkdir(parents=True)
    transcript = transcript_dir / "transcript_full.jsonl"
    transcript.write_text(
        '{"step_index":0,"source":"USER_EXPLICIT","type":"USER_INPUT",'
        '"status":"DONE","created_at":"2026-06-12T20:39:17Z",'
        '"content":"Please inspect auth.py"}\n'
        '{"step_index":1,"source":"MODEL","type":"PLANNER_RESPONSE",'
        '"status":"DONE","created_at":"2026-06-12T20:39:18Z",'
        '"thinking":"I will inspect the auth flow.",'
        '"tool_calls":[{"name":"run_command","args":{"CommandLine":"cat auth.py"}},'
        '{"name":"run_command","args":{"CommandLine":"cat <<EOF > auth.py\\npass\\nEOF"}}]}\n'
    )

    chunks = list(chunk_antigravity_file(transcript))

    assert len(chunks) == 3
    assert {chunk.session_id for chunk in chunks} == {"session-123"}
    assert all(chunk.project_path == "/antigravity/session-123" for chunk in chunks)
    assert "Please inspect auth.py" in chunks[0].content
    assert "run_command" in chunks[1].content
    file_changes = [chunk for chunk in chunks if chunk.chunk_type == "file_change"]
    assert len(file_changes) == 1
    assert file_changes[0].file_path == "auth.py"
    assert file_changes[0].operation == "write"


def test_chunk_chatgpt_export_conversations_json(tmp_path: Path):
    """Official ChatGPT exports include a conversations.json snapshot."""
    export = tmp_path / "conversations.json"
    export.write_text(
        """
[
  {
    "id": "chatgpt-1",
    "title": "Auth Debugging",
    "mapping": {
      "u1": {"message": {"author": {"role": "user"}, "create_time": 1760000000, "content": {"parts": ["Why is auth failing?"]}}},
      "a1": {"message": {"author": {"role": "assistant"}, "create_time": 1760000001, "content": {"parts": ["Check the token refresh path."]}}}
    }
  }
]
"""
    )

    chunks = list(chunk_chatgpt_export_file(export))

    assert len(chunks) == 1
    assert chunks[0].session_id == "chatgpt-1"
    assert chunks[0].project_name == "ChatGPT"
    assert "Why is auth failing?" in chunks[0].content
    assert "Check the token refresh path." in chunks[0].content
    assert _is_chatgpt_export_file(export)
    assert not _is_chatgpt_export_file(tmp_path / "other.json")


def test_chunk_claude_app_export_conversations_json(tmp_path: Path):
    """Claude web/Desktop exports include conversation JSON snapshots."""
    export = tmp_path / "conversations.json"
    export.write_text(
        """
[
  {
    "uuid": "claude-app-1",
    "name": "Planning",
    "chat_messages": [
      {"sender": "human", "created_at": "2026-06-13T10:00:00Z", "text": "Plan the migration."},
      {"sender": "assistant", "created_at": "2026-06-13T10:00:01Z", "text": "Start with storage interfaces."}
    ]
  }
]
"""
    )

    chunks = list(chunk_claude_app_export_file(export))

    assert len(chunks) == 1
    assert chunks[0].session_id == "claude-app-1"
    assert chunks[0].project_name == "Claude App"
    assert "Plan the migration." in chunks[0].content
    assert "Start with storage interfaces." in chunks[0].content
    assert _is_claude_app_export_file(export)
    assert not _is_claude_app_export_file(tmp_path / "other.json")


def test_antigravity_watcher_prefers_full_transcript(tmp_path: Path):
    logs_dir = tmp_path / "brain" / "session-123" / ".system_generated" / "logs"
    logs_dir.mkdir(parents=True)
    transcript = logs_dir / "transcript.jsonl"
    full_transcript = logs_dir / "transcript_full.jsonl"
    transcript.write_text("{}\n")
    full_transcript.write_text("{}\n")

    legacy_dir = tmp_path / "conversations"
    legacy_dir.mkdir()
    legacy_pb = legacy_dir / "session-456.pb"
    legacy_pb.write_bytes(b"hello")

    assert _is_antigravity_file(full_transcript)
    assert not _is_antigravity_file(transcript)
    assert _is_antigravity_file(legacy_pb)
