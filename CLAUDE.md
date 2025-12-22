# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Knowledge Source

**Primary reference**: [claude-history-rag-mcp-requirements.md](claude-history-rag-mcp-requirements.md) - Contains complete specifications including Section 13 with December 2025 knowledge updates for MCP SDK, LanceDB, fastembed, and watchfiles patterns.

**API Reference**: [docs/api-knowledge.md](docs/api-knowledge.md) - Latest version numbers and code patterns researched from the web (Dec 2025). Agents should read this before implementation work.

## Project Overview

This is an MCP server that provides RAG (Retrieval-Augmented Generation) over AI coding agent history (Claude Code, Codex, Gemini CLI). It solves the compaction problem where long sessions lose context by providing persistent, searchable memory across all sessions.

## Tech Stack

- **Python 3.10+** with async/await patterns
- **FastMCP** (from `mcp.server.fastmcp`) - official MCP SDK, STDIO transport
- **LanceDB 0.26+** - embedded vector database with hybrid search
- **fastembed** - ONNX-based embeddings (nomic-embed-text-v1.5 or bge-small-en-v1.5)
- **watchfiles** - Rust-based async file watching
- **pydantic** - data validation and settings

## Build and Run Commands

```bash
# Install dependencies
uv sync

# Run the MCP server
uv run ai-agent-history-rag

# Run tests
uv run pytest

# Run single test
uv run pytest tests/test_parser.py::test_specific_function

# Lint and format
uv run ruff check .
uv run ruff format .

# Test with MCP Inspector
npx @modelcontextprotocol/inspector uv run ai-agent-history-rag
```

## Architecture

### Data Flow

```
~/.claude/projects/*.jsonl, ~/.codex/sessions/**/*.jsonl, ~/.gemini/tmp/**/chats/*.json → File Watcher → Chunking Engine → Embedding → LanceDB → MCP Tools
```

### Key Components

1. **Parser** ([src/claude_history_rag/parser.py](src/claude_history_rag/parser.py)) - Parses JSONL entries (user, assistant, summary, system types)
2. **Chunker** ([src/claude_history_rag/chunker.py](src/claude_history_rag/chunker.py)) - Creates turn, file_change, and summary chunks
3. **Embedder** ([src/claude_history_rag/embedder.py](src/claude_history_rag/embedder.py)) - Async wrapper around fastembed with ThreadPoolExecutor
4. **Store** ([src/claude_history_rag/store.py](src/claude_history_rag/store.py)) - LanceDB operations, hybrid search with RRF reranking
5. **Watcher** ([src/claude_history_rag/watcher.py](src/claude_history_rag/watcher.py)) - watchfiles async file monitoring with debouncing
6. **Server** ([src/claude_history_rag/server.py](src/claude_history_rag/server.py)) - FastMCP server with async tools

### MCP Tools Exposed

- `search_conversations` - Semantic search across all conversation history
- `search_file_changes` - Find specific file modifications
- `get_session_summary` - Get overview of session(s)
- `get_index_status` - Check indexing status

## Critical Implementation Notes

### STDIO Transport Rules
- **NEVER use print() or write to stdout** - corrupts JSON-RPC messages
- All logging must go to stderr via `logging.basicConfig(stream=sys.stderr)`
- All MCP tools must be `async` functions

### Claude Code History Format
- History files are at `~/.claude/projects/` in JSONL format
- Project paths are encoded: `/Users/brandon/project` → `-Users-brandon-project`
- Entry types: `user`, `assistant`, `summary`, `system`

### Chunk Types
1. **Turn chunks** - User message paired with assistant response
2. **File change chunks** - Extracted from Edit/Write tool_use blocks
3. **Summary chunks** - From compaction events

### Embedding Prefixes (Nomic)
- Queries: `"search_query: " + query`
- Documents: `"search_document: " + content`

### LanceDB Configuration
- Database location: `~/.ai-agent-history-rag/lancedb/`
- Index type: IVF_HNSW_SQ for collections > 10,000 chunks
- Use RRFReranker for hybrid search fusion
- Vector weight 0.6 / BM25 weight 0.4

## Performance Targets

- Query latency: <500ms
- Indexing: <30s for 1000 chunks
- Memory idle: <200MB
- Update latency: <60s after file change


## Post-Completion Verification Loop

After completing substantial work (features, refactors, multi-file changes), run the verification loop:

### Verification Agents (5 non-overlapping domains)

These are **code review agents**, not CLI runners. Each agent must READ the actual code, understand the logic, and report substantive issues. Running build or lint is NOT verification — that's just tooling. You **MUST** spawn all 5 agents for every round, even if 4 come back with zero errors you still spawn all 5 for the next round.

1. **Functionality & Logic Review**
   - Read each modified function line-by-line
   - Trace call paths to ensure nothing is broken
   - Check edge cases: null handling, empty arrays, missing fields
   - Verify error paths are handled, not just happy paths
   - Confirm existing behavior is preserved (no silent regressions)

2. **Logging & Observability Review**
   - Read the code to verify `writeTraceEvent` or structured logging exists at entry/exit/error points
   - Check that `traceId` is propagated through async chains
   - Verify log events have meaningful names and include relevant context
   - Confirm errors log `reason` fields, not just stack traces

3. **Type Safety & Contract Review**
   - Read function signatures and verify types are correct (not just that it compiles)
   - Check for unsafe casts, `as any`, or type assertions that hide bugs
   - Verify Firestore document shapes match TypeScript interfaces
   - Review API request/response contracts for consistency

4. **Architecture & Standards Review**
   - Read code structure for proper separation of concerns
   - Verify naming conventions match project standards
   - Check for code duplication that should be extracted
   - Confirm imports and dependencies are appropriate
   - For Flutter: verify design token usage (AppSpacing, AppRadius, AppElevation)

5. **Security & Data Integrity Review**
   - Read auth checks and verify they cover all entry points
   - Check for data leakage in logs or error messages
   - Verify user input is validated before use
   - Confirm destructive operations have proper guards
   - Check that ownership/permission checks exist where needed

### Scope

Verify only the **blast radius of current changes**:
- Files modified in this work session
- Functions/components directly changed
- Direct callers and dependencies of modified code
- Anything the changes could have broken

Do NOT verify the entire codebase — only what this PR touches and affects.

**Agents must actually READ and REVIEW code.** Running lint is NOT verification — those are build checks that should already pass. Verification means reading the code, understanding the logic, and identifying substantive issues that automated tools miss.

**Agents must work in the correct worktree.** When spawning agents, explicitly provide the full worktree path. Agents default to the directory where Claude Code was originally launched, NOT the current worktree. Always include the worktree path in the agent task, e.g.:
```
"Working directory: /path/to/repo-feature-branch
Review the changes in this worktree..."
```
