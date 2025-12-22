#!/bin/bash
# Stop script for ai-agent-history-rag MCP server

set -e

PID_FILE="/tmp/mcp-server.pid"

echo "=== Stopping AI Agent History RAG MCP Server ==="
echo ""

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping server (PID: $PID)..."
    kill "$PID" 2>/dev/null || true

    # Wait up to 10 seconds for graceful shutdown
    for i in {1..10}; do
      if ! kill -0 "$PID" 2>/dev/null; then
        echo "✓ Server stopped gracefully"
        rm "$PID_FILE"
        exit 0
      fi
      sleep 1
    done

    # Force kill if still running
    echo "Force killing server..."
    kill -9 "$PID" 2>/dev/null || true
    echo "✓ Server force stopped"
  else
    echo "Server not running (PID $PID not found)"
  fi
  rm "$PID_FILE"
else
  echo "No PID file found, server may not be running"
fi

echo ""
echo "=== Stop complete ==="
