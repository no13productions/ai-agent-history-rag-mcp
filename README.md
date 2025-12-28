# AI Agent History RAG MCP Server

An MCP (Model Context Protocol) server that provides RAG (Retrieval-Augmented Generation) over **AI coding agent history** (Claude Code, Codex, Gemini CLI). It solves the compaction problem where long sessions lose context by providing persistent, searchable memory across all sessions and tools.

## Features

- **Multi-Agent History**: Ingests Claude Code, Codex, Gemini CLI, and **Google Antigravity** histories
- **Semantic Search**: Find relevant context from past conversations using natural language queries
- **Hybrid Search**: Combines vector similarity and BM25 full-text search with RRF reranking
- **File Change Tracking**: Search for specific file modifications across all sessions
- **Session Summaries**: Retrieve summaries of past sessions
- **Real-time Indexing**: Automatically watches and indexes new conversation data
- **Incremental Updates**: Only processes new content, not entire files
- **Multi-Machine Support**: Centralize history from multiple machines to a single server
- **Offline Resilience**: Client mode queues uploads when server is unavailable
- **Client Registry**: Track connected clients, last uploads, and reindex status
- **Server-Triggered Reindex**: One click to reindex server + notify clients
- **Diagnostic Tool**: Built-in `doctor` command for troubleshooting (cross-platform)
- **Installation Wizard**: Interactive setup with automatic verification

## Supported Sources

- **Claude Code**: `~/.claude/projects/**/*.jsonl`
- **Codex**: `~/.codex/sessions/**/*.jsonl`
- **Gemini CLI**: `~/.gemini/tmp/**/chats/*.json` and `~/.gemini/tmp/**/logs.json`
- **Google Antigravity**: `~/.gemini/antigravity/conversations/*.pb`

All sources are ingested **fully** (user, assistant, tool calls, and tool outputs). The only difference between sources is how we parse their on-disk formats and where we watch for files.

### About diffs and file changes

Diffs are ingested **when the tool provides them**:
- **Codex**: `apply_patch` tool calls include the patch diff in arguments.
- **Gemini CLI**: tool calls may include diffs in `args.patch` or `resultDisplay`.
- **Claude Code**: tool logs include file operations and edit snippets, but full diffs are not guaranteed unless the tool output contains them.

We always store full tool outputs; no truncation.

## Architecture Overview

The system supports two deployment modes:

### Single-Machine Mode (Default)

Everything runs locally - embeddings, storage, and search all happen on one machine.

```
┌─────────────────────────────────────────────────────────────┐
│                     Local Machine                            │
│                                                              │
│  Claude Code ──► MCP Server ──► Daemon ──► Storage (SQLite/Qdrant) │
│                                    │                               │
│                              Embeddings (Ollama/OpenAI API)        │
└────────────────────────────────────────────────────────────────────┘
```

### Multi-Machine Mode (Client/Server)

Consolidate conversation history from multiple machines to a central server:

```
┌─────────────────────────┐     ┌─────────────────────────┐
│      Machine 1          │     │      Machine 2          │
│                         │     │                         │
│  Claude Code            │     │  Claude Code            │
│       │                 │     │       │                 │
│       ▼                 │     │       ▼                 │
│  MCP Client ────────────┼─────┼─► MCP Client            │
│  (chunks only)          │     │  (chunks only)          │
└─────────────────────────┘     └─────────────────────────┘
              │                           │
              │      HTTP POST            │
              ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Central Server                            │
│                                                              │
│  API Endpoints ◄── Status Server (port 4680)                │
│       │                                                      │
│       ▼                                                      │
│  Embedder ──► Storage ──► Search API                        │
│  (Ollama/vLLM)           (Qdrant/SQLite)                    │
└─────────────────────────────────────────────────────────────┘
```

**Benefits of multi-machine mode:**
- Search across all your machines' conversation history from any machine
- Centralized embeddings - only one machine needs GPU/compute resources
- Offline resilience - clients queue uploads when server is unavailable
- Catch-up sync - reconnecting clients automatically upload missed content

## Installation

### Prerequisites

The server uses an **OpenAI-compatible embeddings API** for generating vectors. This works with:
- **Ollama** (recommended for local use)
- **vLLM**
- **text-embeddings-inference**
- **OpenAI API**
- **LiteLLM**
- Any other service implementing the `/v1/embeddings` endpoint

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/bmeyer99/claude-history-rag-mcp.git
cd claude-history-rag-mcp

# Install all dependencies (both server and client)
uv sync --all-extras

# Or install only what you need:
uv sync --extra server   # Server mode (embeddings + storage)
uv sync --extra client   # Client mode (lightweight, uploads only)
```

### Using pip

```bash
# Full installation
pip install -e ".[all]"

# Server only
pip install -e ".[server]"

# Client only (lightweight)
pip install -e ".[client]"
```

## Quick Start

### Install Wizard (Recommended)

The install wizard configures everything for you - MCP servers, daemon service, and all settings:

```bash
uv run ai-agent-history-rag install
```

The wizard will:
1. Ask whether to install MCP server, daemon, or both
2. Configure server mode (local) or client mode (multi-machine)
2.5. **Update mode** (new): reuses existing daemon config (from the service) to reinstall without prompts
3. Detect installed AI tools (Claude Desktop, Claude Code, Cursor, VS Code, Gemini CLI, OpenAI Codex)
4. Add MCP configuration to selected applications
5. Install daemon as a system service (launchd/systemd/Windows Task) — removing any existing service first to ensure updates apply
6. Prompt for **PSK authentication** settings (optional PSK overrides + auth paths)
7. **Verify installation** - waits for daemon startup and runs health checks

Note: ChatGPT connectors are configured in-app (Developer mode) and are not managed by this installer.

### Docker (Server Only)

1. **Start Ollama** on your host machine:
   ```bash
   ollama serve
   ollama pull bge-m3
   ```

2. **Start the container**:
   ```bash
   docker compose up -d
   ```

Access the dashboard at http://localhost:4680/dashboard

The container connects to Ollama on your host via `host.docker.internal`.
On Linux with custom Docker networks, `host.docker.internal` may not resolve—either keep the default bridge network or point the embedding URL to your host’s IP address.

**Configuration**: Create a `.env` file to customize the embedding server:
```bash
# Use a different embedding server (default: host.docker.internal:11434)
CLAUDE_HISTORY_RAG_EMBEDDING_BASE_URL=http://192.168.1.100:11434/v1
```

**PSK Authentication** (recommended behind TLS):
```bash
# Enable PSK auth and set a server key override
CLAUDE_HISTORY_RAG_AUTH_ENABLED=true
CLAUDE_HISTORY_RAG_SERVER_PSK=change-me
```

See [.env.docker.example](.env.docker.example) for all options.

**Client machines** can connect to this Docker server:
```bash
export CLAUDE_HISTORY_RAG_SERVER_URL=http://docker-host:4680
uv run ai-agent-history-rag-daemon start
```

### Single-Machine Setup (Default)

1. **Start Ollama** (or another embeddings server):
   ```bash
   ollama serve
   ollama pull nomic-embed-text
   ```

2. **Start the daemon**:
   ```bash
   uv run ai-agent-history-rag-daemon start
   ```

3. **Configure Claude Code** (see Configuration section below)

### Multi-Machine Setup

#### On the Central Server

1. **Start the embeddings server** (Ollama example):
   ```bash
   ollama serve
   ollama pull nomic-embed-text
   ```

2. **Start the daemon in server mode** (no `SERVER_URL` set):
   ```bash
   # Bind to all interfaces to accept remote connections
   CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST=0.0.0.0 \
   uv run ai-agent-history-rag-daemon start
   ```

3. The server exposes:
   - Dashboard: `http://server-ip:4680/dashboard`
   - API: `http://server-ip:4680/api/`

#### On Each Client Machine

1. **Configure to point to the server**:
   ```bash
   export CLAUDE_HISTORY_RAG_SERVER_URL=http://192.168.1.100:4680
   export CLAUDE_HISTORY_RAG_MACHINE_ID=my-laptop  # Optional, defaults to hostname
   export CLAUDE_HISTORY_RAG_CLIENT_NAME="Brandon MacBook"  # Optional label
   ```

2. **Start the daemon in client mode**:
   ```bash
   uv run ai-agent-history-rag-daemon start
   ```

3. **Configure Claude Code** to use the MCP server (see Configuration section)

## Configuration

### Claude Code MCP Settings

#### Option 1: Using `claude mcp add-json` (Easiest)

**Server Mode (default)**:
```bash
claude mcp add-json ai-agent-history-rag '{
  "command": "uv",
  "args": ["--directory", "/path/to/claude-history-rag-mcp", "run", "ai-agent-history-rag"],
  "env": {
    "CLAUDE_HISTORY_RAG_DEFER_STARTUP_INDEXING": "true"
  }
}'
```

**Client Mode (multi-machine)**:
```bash
claude mcp add-json ai-agent-history-rag '{
  "command": "uv",
  "args": ["--directory", "/path/to/claude-history-rag-mcp", "run", "ai-agent-history-rag"],
  "env": {
    "CLAUDE_HISTORY_RAG_SERVER_URL": "http://192.168.1.100:4680",
    "CLAUDE_HISTORY_RAG_MACHINE_ID": "my-laptop",
    "CLAUDE_HISTORY_RAG_CLIENT_NAME": "Brandon MacBook"
  }
}'
```

Replace `/path/to/claude-history-rag-mcp` with your actual project path.

#### Option 2: Manual Configuration

Add to `~/.config/Claude/claude_desktop_config.json`:

**Server Mode**:
```json
{
  "mcpServers": {
    "ai-agent-history-rag": {
      "command": "uv",
      "args": ["--directory", "/path/to/claude-history-rag-mcp", "run", "ai-agent-history-rag"],
      "env": {
        "CLAUDE_HISTORY_RAG_EMBEDDING_BASE_URL": "http://localhost:11434/v1",
        "CLAUDE_HISTORY_RAG_EMBEDDING_MODEL": "nomic-embed-text"
      }
    }
  }
}
```

**Client Mode**:
```json
{
  "mcpServers": {
    "ai-agent-history-rag": {
      "command": "uv",
      "args": ["--directory", "/path/to/claude-history-rag-mcp", "run", "ai-agent-history-rag"],
      "env": {
        "CLAUDE_HISTORY_RAG_SERVER_URL": "http://192.168.1.100:4680",
        "CLAUDE_HISTORY_RAG_CLIENT_NAME": "Brandon MacBook"
      }
    }
  }
}
```

### Environment Variables

#### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_HISTORY_RAG_STORAGE_BACKEND` | `sqlite` | Storage backend: `sqlite` or `qdrant` |
| `CLAUDE_HISTORY_RAG_SQLITE_DB_PATH` | `~/.claude-history-rag/history.db` | SQLite database location |
| `CLAUDE_HISTORY_RAG_QDRANT_URL` | `None` | Qdrant server URL (e.g. `http://localhost:6333`) |
| `CLAUDE_HISTORY_RAG_QDRANT_API_KEY` | `None` | Qdrant API key |
| `CLAUDE_HISTORY_RAG_QDRANT_COLLECTION` | `history_rag` | Qdrant collection name |
| `CLAUDE_HISTORY_RAG_STATE_PATH` | `~/.claude-history-rag/state.json` | File position state |
| `CLAUDE_HISTORY_RAG_PROJECTS_PATH` | `~/.claude/projects` | Claude Code projects directory |
| `CLAUDE_HISTORY_RAG_CODEX_SESSIONS_PATH` | `~/.codex/sessions` | Codex session history directory |
| `CLAUDE_HISTORY_RAG_CODEX_STATE_PATH` | `~/.claude-history-rag/codex_state.json` | Codex file position state |
| `CLAUDE_HISTORY_RAG_GEMINI_SESSIONS_PATH` | `~/.gemini/tmp` | Gemini CLI session history directory |
| `CLAUDE_HISTORY_RAG_GEMINI_STATE_PATH` | `~/.claude-history-rag/gemini_state.json` | Gemini file position state |
| `CLAUDE_HISTORY_RAG_ANTIGRAVITY_SESSIONS_PATH` | `~/.gemini/antigravity/conversations` | Google Antigravity sessions directory |
| `CLAUDE_HISTORY_RAG_ANTIGRAVITY_STATE_PATH` | `~/.claude-history-rag/antigravity_state.json` | Google Antigravity file position state |
| `CLAUDE_HISTORY_RAG_LOG_LEVEL` | `INFO` | Logging level |

#### Client/Server Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_HISTORY_RAG_SERVER_URL` | `None` | Central server URL. If set, runs in **client mode** |
| `CLAUDE_HISTORY_RAG_MACHINE_ID` | hostname | Unique identifier for this machine |
| `CLAUDE_HISTORY_RAG_CLIENT_NAME` | `""` | Optional human-friendly label for this client |
| `CLAUDE_HISTORY_RAG_UPLOAD_INTERVAL_SECONDS` | `300` | Batch upload interval (5 min) |
| `CLAUDE_HISTORY_RAG_UPLOAD_RETRY_COUNT` | `3` | Retries before queuing for later |
| `CLAUDE_HISTORY_RAG_UPLOAD_RETRY_DELAY_SECONDS` | `30` | Delay between retries |

#### Embedding Settings (OpenAI-compatible API)

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_HISTORY_RAG_EMBEDDING_BASE_URL` | `http://localhost:11434/v1` | Embeddings API base URL |
| `CLAUDE_HISTORY_RAG_EMBEDDING_MODEL` | `nomic-embed-text` | Model name |
| `CLAUDE_HISTORY_RAG_EMBEDDING_API_KEY` | `""` | API key (for OpenAI, etc.) |

**Example URLs:**
- Ollama: `http://localhost:11434/v1`
- vLLM: `http://localhost:8000/v1`
- OpenAI: `https://api.openai.com/v1`
- text-embeddings-inference: `http://localhost:8080/v1`

#### Status Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_HISTORY_RAG_STATUS_SERVER_ENABLED` | `true` | Enable HTTP status server |
| `CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST` | `127.0.0.1` | Status server host |
| `CLAUDE_HISTORY_RAG_STATUS_SERVER_PORT` | `4680` | Status server port |

#### Auth (PSK) Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_HISTORY_RAG_AUTH_ENABLED` | `true` | Require PSK on status + API endpoints |
| `CLAUDE_HISTORY_RAG_SERVER_PSK` | `""` | Optional server PSK override (disables rotation UI) |
| `CLAUDE_HISTORY_RAG_CLIENT_PSK` | `""` | Optional client PSK override (if unset, uses local JSON) |
| `CLAUDE_HISTORY_RAG_AUTH_STATE_PATH` | `~/.claude-history-rag/auth.json` | Server auth state (rotation, allowlist, hashes) |
| `CLAUDE_HISTORY_RAG_CLIENT_AUTH_PATH` | `~/.claude-history-rag/client_auth.json` | Client PSK storage |

#### Performance Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_HISTORY_RAG_DEBOUNCE_DELAY` | `5000` | File watcher debounce (ms) |
| `CLAUDE_HISTORY_RAG_BATCH_SIZE` | `32` | Embedding batch size |
| `CLAUDE_HISTORY_RAG_MAX_CHUNKS_PER_FILE` | `100` | Max chunks per batch |
| `CLAUDE_HISTORY_RAG_MAX_FILE_BATCH_SIZE` | `50` | Files to process before GC |
| `CLAUDE_HISTORY_RAG_GC_AFTER_FILES` | `true` | Enable garbage collection |
| `CLAUDE_HISTORY_RAG_DEFER_STARTUP_INDEXING` | `false` | Skip initial indexing on startup |

### Embedding Model Selection

The server supports multiple embedding models. Choose based on your priorities:

| Model | MTEB | Retrieval | Dims | Size | Best For |
|-------|------|-----------|------|------|----------|
| `mxbai-embed-large` | 64.68 | 54.39 | 1024 | 670MB | Maximum quality |
| `bge-m3` | ~63 | ~53 | 1024 | 1.2GB | Long context, multilingual |
| `nomic-embed-text` | 62.28 | ~50 | 768 | 274MB | Balanced (default) |
| `snowflake-arctic-embed` | ~60 | ~48 | var | 46-669MB | Memory-constrained |

**Switching models** requires re-indexing:
```bash
# Delete existing index (if switching models abruptly)
rm ~/.claude-history-rag/history.db

# Set new model
export CLAUDE_HISTORY_RAG_EMBEDDING_MODEL=mxbai-embed-large

# Pull the model (if using Ollama)
ollama pull mxbai-embed-large

# Restart daemon
uv run ai-agent-history-rag-daemon restart
```

## CLI Commands

The package provides seven command-line tools:

| Command | Description |
|---------|-------------|
| `ai-agent-history-rag` | Main entry point (use `doctor`, `settings`, `install`, `daemon` subcommands) |
| `ai-agent-history-rag-daemon` | (Deprecated) Background daemon for indexing |
| `ai-agent-history-rag-install` | (Deprecated) Interactive installation wizard |
| `ai-agent-history-rag-doctor` | (Deprecated) Diagnostic tool |
| `ai-agent-history-rag-settings` | (Deprecated) Interactive settings wizard |
| `ai-agent-history-rag-docker` | (Deprecated) Docker deployment wizard |

Run with `uv run <command>` or directly if installed globally.

## Running Modes

### Daemon Mode (Recommended)

Run the indexer and status server as a standalone background daemon:

```bash
# Start the daemon
uv run ai-agent-history-rag daemon start

# Check daemon status
uv run ai-agent-history-rag daemon status

# Stop the daemon
uv run ai-agent-history-rag daemon stop

# Restart the daemon
uv run ai-agent-history-rag daemon restart
```

The daemon:
- Runs in the foreground (use `&` or a process manager for background)
- Writes PID to `~/.claude-history-rag/daemon.pid`
- Logs to `~/.claude-history-rag/daemon.log`
- Provides the dashboard at http://127.0.0.1:4680/dashboard

**Server mode log output**:
```
Starting daemon [SERVER] | backend=sqlite | embedding_url=http://localhost:11434/v1 | embedding_model=nomic-embed-text
```

**Client mode log output**:
```
Starting daemon [CLIENT] | server_url=http://192.168.1.100:4680 | machine_id=my-laptop
```

### Standalone Mode

Run everything in a single process:

```bash
uv run ai-agent-history-rag --standalone
```

### Auto-start on Boot

#### macOS (launchd)

```bash
./scripts/install-launchd.sh
```

To configure for client mode, edit `~/Library/LaunchAgents/com.ai-agent-history-rag.daemon.plist` after installation.

#### Linux (systemd)

```bash
./scripts/install-systemd.sh
```

To configure environment variables:
```bash
# Edit the service file
nano ~/.config/systemd/user/ai-agent-history-rag.service

# Reload and restart
systemctl --user daemon-reload
systemctl --user restart ai-agent-history-rag
```

#### Windows (Scheduled Task)

```powershell
.\scripts\install-windows.ps1
```

To configure for client mode, set user environment variables (`CLAUDE_HISTORY_RAG_SERVER_URL`) and restart the task.

## Status Monitoring

The status server provides monitoring endpoints:

- **Dashboard**: http://127.0.0.1:4680/dashboard - Auto-refreshing web UI
- **Health Check**: http://127.0.0.1:4680/health - Simple health status
- **Status API**: http://127.0.0.1:4680/status - JSON status
- **Prometheus Metrics**: http://127.0.0.1:4680/metrics - Prometheus format
- **Client Registry**: Included in `/status?detail=full` under `clients`

### PSK Authentication & Rotation

All status server endpoints (dashboard + API + health/metrics) require a **pre-shared key (PSK)** by default. Clients send:

```
Authorization: Bearer <psk>
```

**TLS required**: Run the status server behind HTTPS (e.g., Traefik). The PSK is sent raw over the wire and is only protected by TLS.

**Server storage (auth.json)**:
- The active key is stored **hashed** for validation.
- The active key is also stored **in plaintext** to support dashboard reveal and rotation flows.
- If you set `CLAUDE_HISTORY_RAG_SERVER_PSK`, the dashboard disables rotation (tooltip: “PSK assigned in .env — rotate in your .env and rebuild”).

**Client storage (client_auth.json)**:
- Clients store the raw PSK locally for requests.
- The client auth file is written with **0600 permissions** on macOS/Linux (best-effort on Windows).

**Rotation flow**:
- “Rotate PSK” lets you select existing clients to temporarily keep using the old key for *X days*.
- New/unknown clients must use the **new key**.
- Clients receive a rotation hint, retry immediately with the new key, and ack success.
- If rotation fails, the client falls back to the old key and reports an error; the dashboard shows a red **Error** key status with an “Allow stay” button (temporary allowlist, expires after X days).

**Dashboard key reveal**:
- You must unlock the dashboard with the current PSK to access protected endpoints.
- The dashboard stores a **hash** in `localStorage` to authorize key reveal; the PSK itself is only held in-memory while the reveal modal is open.
- Auto-refresh is paused while the key modal is open.

**Key status column**:
- **Current** (green): using the active key
- **Awaiting Rotation** (yellow): allowlisted to use old key
- **Old** (orange): old key expired or removed
- **Error** (red): failed rotation

**Security limitations**:
- The PSK is **plaintext in server auth.json** to support dashboard reveal/rotation.
- Protect your host and `auth.json` file; restrict filesystem access.
- Do not expose the status server without TLS.

### Re-index Behavior (Server Mode)

Using the dashboard **Re-index** button will:
1. Clear the server database and reset server-side file positions
2. Set a reindex request flag for all clients
3. Clients acknowledge the request, clear their local positions, and re-upload
4. Clients send a **completed** ack after uploads finish

You can see client ack status in the dashboard Clients panel.

Client registry data is stored under the configured state directory (e.g., `~/.claude-history-rag/client_registry.json` or `/data/state` in Docker) so it survives upgrades/reinstalls.

### API Endpoints (Server Mode)

When running in server mode, additional API endpoints are available for client machines:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chunks` | POST | Upload chunks from clients |
| `/api/search` | POST | Semantic search |
| `/api/search/files` | POST | File change search |
| `/api/sessions` | POST | Session summaries |
| `/api/positions/{machine_id}` | GET | Get file positions for a machine |
| `/api/positions` | POST | Update file position |
| `/api/reindex-ack` | POST | Client acknowledgement for server reindex |
| `/api/purge-client` | POST | Purge all chunks for a single client |

## MCP Tools

### search_conversations

Search conversation history for relevant context.

```
Arguments:
  query: str           - Natural language query
  project_filter: str  - Limit to specific project (optional)
  limit: int           - Maximum results (default: 5)
  use_hybrid: bool     - Use hybrid search (default: True)
```

### search_file_changes

Find file modifications in conversation history.

```
Arguments:
  file_path: str       - Filter by file path (optional, supports partial match)
  query: str           - Semantic query about changes (optional)
  project_filter: str  - Limit to specific project (optional)
  operation_filter: str - Filter by "edit" or "write" (optional)
  limit: int           - Maximum results (default: 10)
```

### get_session_summary

Get summary of conversation session(s).

```
Arguments:
  session_id: str      - Specific session ID (optional)
  project_filter: str  - Limit to specific project (optional)
  count: int           - Number of sessions (default: 1)
```

### get_index_status

Get status of the RAG index.

```
Returns:
  mode: str                    - "server" or "client"
  total_chunks: int            - Number of indexed chunks (server mode)
  watched_files: int           - Number of files being tracked
  pending_files: int           - Files in queue for processing
  pending_uploads: int         - Uploads waiting to send (client mode)
  connected: bool              - Server connection status (client mode)
  server_status: dict          - Remote server status (client mode)
  status: str                  - Overall health status
```

### get_server_status

Get comprehensive server status and health information.

```
Arguments:
  detail_level: str  - "basic" for summary, "full" for detailed metrics (default: "basic")

Returns:
  server: dict      - Version, uptime, PID, platform info
  health: dict      - Overall status and component health checks
  database: dict    - Chunk counts, database size (full detail only)
  indexing: dict    - File processing progress (full detail only)
  performance: dict - Memory, CPU, query metrics (full detail only)
  cache: dict       - Hit rates, cache size (full detail only)
```

## Development

### Running Tests

```bash
uv run pytest
```

### Linting

```bash
uv run ruff check .
uv run ruff format .
```

### Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector uv run ai-agent-history-rag
```

## Detailed Architecture

### Single-Machine Mode

```
┌─────────────────────────────────────────────────────────────┐
│                     Daemon Process                          │
│  (ai-agent-history-rag-daemon)                                │
│                                                             │
│  ~/.claude/projects/*.jsonl                                 │
│           │                                                 │
│           ▼                                                 │
│     File Watcher ──► Chunker ──► Embedder ──► LanceDB       │
│                                               (shared)      │
│                                                   │         │
│     Status Server (dashboard, health, metrics)    │         │
└───────────────────────────────────────────────────│─────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   MCP Server Process                        │
│  (ai-agent-history-rag - lightweight mode)                    │
│                                                             │
│     Claude Code ◄──► STDIO Transport ◄──► MCP Tools         │
│                                               │             │
│                                               ▼             │
│                                               ▼             │
│                                           Storage           │
│                                           (queries)         │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Machine Mode

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Machine                           │
│                                                             │
│  ~/.claude/projects/*.jsonl                                 │
│           │                                                 │
│           ▼                                                 │
│     File Watcher ──► Chunker ──► HTTP Client                │
│                                      │                      │
│                            ┌─────────┴─────────┐            │
│                            │  Pending Queue    │            │
│                            │  (offline mode)   │            │
│                            └───────────────────┘            │
│                                      │                      │
│     MCP Tools ◄── proxy to server ◄──┘                      │
└──────────────────────────────│──────────────────────────────┘
                               │ HTTP POST /api/chunks
                               │ HTTP POST /api/search
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Central Server                           │
│                                                             │
│     API Endpoints ◄── Status Server (port 4680)             │
│           │                                                 │
│           ▼                                                 │
│     Embedder ──► Storage ◄── Search API                    │
│     (Ollama/vLLM)                                          │
│                                                             │
│     Position Tracking (per machine)                        │
└─────────────────────────────────────────────────────────────┘
```

### Offline Resilience (Client Mode)

When the server is unavailable:

1. **Chunking continues locally** - Files are still processed into chunks
2. **Uploads are queued** - Chunks are stored in `~/.claude-history-rag/client_state.json`
3. **Retry logic** - 3 retries with 30s delay, then waits for next sync interval
4. **Catch-up on reconnect** - Compares local vs server positions, re-uploads gaps
5. **Search degrades gracefully** - Returns "server unavailable" error

### Chunk Types

1. **Turn chunks**: User message paired with assistant response
2. **File change chunks**: Extracted from Edit/Write tool_use blocks with parent-child linking
3. **Summary chunks**: From compaction events

Each chunk includes `machine_id` in multi-machine mode for tracking origin.

### Tech Stack

- **Python 3.10+** with async/await patterns
- **FastMCP** (official MCP SDK) - STDIO transport
- **SQLite + sqlite-vec** - Embedded vector search (default)
- **Qdrant** - Optional scalable vector store
- **httpx** - Async HTTP client for embeddings API and client/server communication
- **watchfiles** - Rust-based async file watching
- **pydantic** - Data validation and settings
- **aiohttp** - Status server and API endpoints

## Performance

| Metric | Target | Implementation |
|--------|--------|----------------|
| Query latency | <500ms | SQLite/Qdrant + RRF reranking |
| Indexing | <30s/1000 chunks | Batch embedding, async I/O |
| Memory idle | <200MB | Lazy model loading |
| Update latency | <60s | 5s debounce + incremental indexing |


## Storage Backends

The server supports two storage backends:

### SQLite (Default)
Best for single-machine setups and lightweight deployments.
- Uses `sqlite-vec` for high-performance vector search
- Zero-dependency deployment (embedded)
- Database stored in a single file (`history.db`)

### Qdrant
Best for power users, large datasets, and multi-machine setups.
- Dedicated vector database server
- Higher scalability and advanced filtering
- Run via Docker: `docker run -p 6333:6333 qdrant/qdrant`

To switch, set `CLAUDE_HISTORY_RAG_STORAGE_BACKEND=qdrant` and configure `CLAUDE_HISTORY_RAG_QDRANT_URL`.

## Troubleshooting

### Diagnostic Tool

Run the doctor command for comprehensive system diagnostics:

```bash
uv run ai-agent-history-rag doctor
```

The doctor checks:
- **Configuration** - validates settings and detects client/server mode
- **Daemon Status** - verifies the daemon is running (checks PID file)
- **Port Availability** - checks if port 4680 is in use and by what process
- **Service Connectivity** - tests connection to embedding server or central server
- **File System** - validates database and projects directories exist
- **Recent Logs** - displays last 10 log entries with error highlighting
- **Environment Variables** - shows configured env vars
- **Service Installation** - checks launchd/systemd/Windows task status

Cross-platform support: macOS (launchd), Linux (systemd), Windows (scheduled tasks).

Example output:
```
============================================================
                 AI Agent History RAG Doctor
============================================================

Configuration
  ✓ Configuration loaded successfully
  → Mode: CLIENT (connecting to http://192.168.1.100:4680)
  → Machine ID: my-laptop

Daemon Status
  ✓ Daemon is running (PID 12345)

Service Connectivity
  ✓ Central server is reachable (HTTP 200)

...

============================================================
                          Summary
============================================================

All checks passed!
```

### Client can't connect to server

1. Check server is running: `curl http://server-ip:4680/health`
2. Verify firewall allows port 4680
3. Check `STATUS_SERVER_HOST` is set to `0.0.0.0` on server (not `127.0.0.1`)

### Embeddings failing

1. Verify embedding server is running: `curl http://localhost:11434/v1/models`
2. Check model is pulled: `ollama list`
3. Verify `EMBEDDING_BASE_URL` and `EMBEDDING_MODEL` are correct

### Pending uploads not syncing

1. Check server connectivity: `curl http://server-ip:4680/health`
2. View pending uploads: `cat ~/.claude-history-rag/client_state.json`
3. Stale uploads (>72h) are automatically cleared

## License

MIT
