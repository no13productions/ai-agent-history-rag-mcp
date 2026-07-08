#!/bin/bash
# Status script for the ai-agent-history-rag daemon.

set -e

PORT="${CLAUDE_HISTORY_RAG_STATUS_SERVER_PORT:-4680}"
AUTH_STATE_PATH="${CLAUDE_HISTORY_RAG_AUTH_STATE_PATH:-$HOME/.claude-history-rag/auth.json}"
STATUS_URL="http://127.0.0.1:${PORT}/status?detail=full"

echo "=== AI Agent History RAG Daemon Status ==="
echo ""

if [ -n "${CLAUDE_HISTORY_RAG_SERVER_PSK:-}" ]; then
  AUTH_HEADER="Authorization: Bearer ${CLAUDE_HISTORY_RAG_SERVER_PSK}"
elif [ -f "$AUTH_STATE_PATH" ]; then
  PSK="$(jq -r '.active.key_plain // empty' "$AUTH_STATE_PATH")"
  AUTH_HEADER="Authorization: Bearer ${PSK}"
else
  AUTH_HEADER=""
fi

if [ -n "$AUTH_HEADER" ]; then
  STATUS_JSON="$(curl -fsS -H "$AUTH_HEADER" "$STATUS_URL")"
else
  STATUS_JSON="$(curl -fsS "$STATUS_URL")"
fi

echo "$STATUS_JSON" | jq '{
  server: .server,
  health: .health.status,
  database: .database,
  watcher: .file_watcher,
  configuration: .configuration
}'
