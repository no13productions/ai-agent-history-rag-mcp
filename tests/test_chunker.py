"""Tests for chunking engine."""

import hashlib
import json
import logging
from pathlib import Path

import pytest

from claude_history_rag.antigravity.chunker import chunk_antigravity_file
from claude_history_rag.antigravity.watcher import _is_antigravity_file
from claude_history_rag.chatgpt.chunker import chunk_chatgpt_export_file
from claude_history_rag.chatgpt.watcher import _is_chatgpt_export_file
from claude_history_rag.chunker import chunk_session_file
from claude_history_rag.claude_app.chunker import chunk_claude_app_export_file
from claude_history_rag.claude_app.watcher import _is_claude_app_export_file
from claude_history_rag.codex.parser import parse_codex_jsonl_file


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


@pytest.mark.parametrize("payload", [42, "text", None, True, 3.5])
def test_export_chunkers_survive_a_bare_scalar_document(payload, tmp_path: Path):
    """A top-level bare scalar has neither .get nor list semantics.

    An unguarded lookup raises AttributeError and costs the entire export.
    """
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps(payload))

    assert list(chunk_chatgpt_export_file(export)) == []
    assert list(chunk_claude_app_export_file(export)) == []


@pytest.mark.parametrize("terminator,expected", [(b"\n", 3), (b"\r\n", 3), (b"\r", 1)])
def test_codex_parser_numbers_lines_the_way_bytes_are_counted(
    terminator: bytes,
    expected: int,
    tmp_path: Path,
):
    """The Codex parser shares the byte-counted definition of a line.

    Universal newline translation would split on a bare carriage return that
    byte counting does not, numbering lines past the counted bound.
    """
    record = json.dumps({"type": "message", "role": "user", "content": "hello"}).encode()
    source = tmp_path / "session.jsonl"
    source.write_bytes(terminator.join([record] * 3) + terminator)

    with open(source, "rb") as raw:
        counted = sum(1 for _ in raw)
    parsed = [line for _, line in parse_codex_jsonl_file(source)]

    assert counted == expected
    assert not parsed or max(parsed) <= counted


def test_codex_parser_aborted_read_raises(tmp_path: Path):
    """A partial Codex read must not look like a completed one."""
    record = json.dumps({"type": "message", "role": "user", "content": "hello"}).encode()
    source = tmp_path / "session.jsonl"
    source.write_bytes((record + b"\n") * 3 + b"caf\xe9 undecodable\n" + (record + b"\n") * 2)

    with pytest.raises(UnicodeDecodeError):
        list(parse_codex_jsonl_file(source))


@pytest.mark.parametrize("payload", [42, "text", None, True, 3.5, [1, 2], {"a": 1}])
def test_codex_parser_survives_a_bare_scalar_line(payload, tmp_path: Path):
    """One unusable Codex line must not cost every remaining line in the file."""
    good = {"type": "message", "role": "user", "content": "hello"}
    source = tmp_path / "session.jsonl"
    source.write_text("\n".join([json.dumps(good), json.dumps(payload), json.dumps(good)]) + "\n")

    assert [line for _, line in parse_codex_jsonl_file(source)] == [1, 3]


def test_chunker_diagnostics_disclose_no_source_session_or_conversation_text(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    """Every diagnostic in the chunking pipeline is a code plus digests.

    The file below deliberately triggers each warning path: an empty turn, an
    unpaired assistant, a trailing unpaired user, a MALFORMED entry that fails
    model validation, and completion. None of them may disclose the path,
    filename, session id, entry uuid, or any conversation text - not even
    truncated.

    The malformed entry matters: a pydantic ValidationError renders the rejected
    input inside its own message, so a parser that formats the exception
    republishes the conversation content the entry carried.
    """
    project_dir = tmp_path / "-Users-test-PRIVATEPROJECT"
    project_dir.mkdir()
    session_file = project_dir / "PRIVATE_SESSION_FILENAME.jsonl"
    session_id = "SESSION_IDENTIFIER_SENTINEL"
    unpaired_uuid = "UUID_IDENTIFIER_SENTINEL"
    conversation_text = "CONVERSATION_CONTENT_SENTINEL"

    session_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "sessionId": session_id,
                        "uuid": unpaired_uuid,
                        "timestamp": "2026-01-01T00:00:00Z",
                        "message": {
                            "role": "assistant",
                            "model": "m",
                            "content": [{"type": "text", "text": conversation_text}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "sessionId": session_id,
                        "uuid": unpaired_uuid,
                        "timestamp": "2026-01-01T00:00:01Z",
                        "message": {"role": "user", "content": ""},
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "sessionId": session_id,
                        "uuid": unpaired_uuid,
                        "timestamp": "2026-01-01T00:00:02Z",
                        "message": {"role": "assistant", "model": "m", "content": []},
                    }
                ),
                # Malformed: content must be a string or a list, never a dict.
                # Model validation rejects it and the rejected input carries
                # the conversation text.
                json.dumps(
                    {
                        "type": "user",
                        "sessionId": session_id,
                        "uuid": unpaired_uuid,
                        "timestamp": "2026-01-01T00:00:03Z",
                        "message": {"role": "user", "content": {"text": conversation_text}},
                    }
                ),
                # Not valid JSON at all: the decoder quotes the document.
                '{"type": "user", "message": {"role": "user", "content": "'
                + conversation_text
                + '"',
                json.dumps(
                    {
                        "type": "user",
                        "sessionId": session_id,
                        "uuid": unpaired_uuid,
                        "timestamp": "2026-01-01T00:00:04Z",
                        "message": {"role": "user", "content": conversation_text},
                    }
                ),
            ]
        )
        + "\n"
    )

    expected_source = hashlib.sha256(str(session_file).encode("utf-8")).hexdigest()[:12]

    with caplog.at_level(logging.DEBUG):
        list(chunk_session_file(session_file))

    assert str(session_file) not in caplog.text
    assert "PRIVATE_SESSION_FILENAME" not in caplog.text
    assert "PRIVATEPROJECT" not in caplog.text
    assert session_id not in caplog.text
    assert unpaired_uuid not in caplog.text
    assert conversation_text not in caplog.text

    assert f"source={expected_source}" in caplog.text
    assert "reason=empty_turn" in caplog.text
    assert "reason=assistant_without_user" in caplog.text
    assert "reason=trailing_user_without_assistant" in caplog.text
    assert "Completed chunking:" in caplog.text
    # The malformed-entry paths really did run, so the absence assertions above
    # are proving sanitization rather than the absence of a diagnostic.
    assert "reason=message_validation_failed" in caplog.text
    assert "reason=json_invalid" in caplog.text


def test_chunker_records_declared_provenance_for_a_relocated_snapshot(
    sample_session_path: Path,
    tmp_path: Path,
):
    """Reading a relocated snapshot must not change provenance or project identity."""
    project_dir = tmp_path / "-Users-test-myproject"
    project_dir.mkdir()
    original = project_dir / "session.jsonl"
    original.write_text(sample_session_path.read_text())

    snapshot_dir = tmp_path / "snapshot-elsewhere"
    snapshot_dir.mkdir()
    snapshot = snapshot_dir / "session.jsonl"
    snapshot.write_text(original.read_text())

    direct = list(chunk_session_file(original))
    relocated = list(chunk_session_file(snapshot, source_path=original))

    assert relocated
    assert [chunk.id for chunk in relocated] == [chunk.id for chunk in direct]
    assert all(chunk.source_file == str(original) for chunk in relocated)
    assert all(chunk.project_path == "/Users/test/myproject" for chunk in relocated)
    assert all(str(snapshot) not in chunk.source_file for chunk in relocated)
