#!/bin/bash
# Start script for ai-agent-history-rag MCP server
# Starts the server in the background and monitors logs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$HOME/.claude-history-rag/claude-history-rag.log"
PID_FILE="/tmp/mcp-server.pid"
OUTPUT_FILE="/tmp/mcp-server-out.log"

echo "=== Starting AI Agent History RAG MCP Server ==="
echo ""

# Check if already running
if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "Server is already running (PID: $PID)"
    echo "Log file: $LOG_FILE"
    exit 0
  else
    echo "Removing stale PID file..."
    rm "$PID_FILE"
  fi
fi

# Start server
echo "Starting server in background..."
cd "$PROJECT_DIR"
nohup ~/.local/bin/uv run ai-agent-history-rag > "$OUTPUT_FILE" 2>&1 &
PID=$!
echo $PID > "$PID_FILE"

echo "✓ Server started with PID: $PID"
echo "  Log file: $LOG_FILE"
echo "  Output file: $OUTPUT_FILE"
echo ""

# Wait for server to initialize
echo "Waiting for server to initialize..."
sleep 5

if kill -0 "$PID" 2>/dev/null; then
  echo "✓ Server is running"
  echo ""
  echo "Monitor logs with:"
  echo "  tail -f $LOG_FILE"
  echo ""
  echo "Stop server with:"
  echo "  $SCRIPT_DIR/stop.sh"
else
  echo "✗ Server failed to start"
  echo "Check output file: $OUTPUT_FILE"
  rm "$PID_FILE"
  exit 1
fi
