#!/bin/bash
# Cleanup script for ai-agent-history-rag MCP server
# This script stops the running server and optionally cleans up data

set -e

echo "=== AI Agent History RAG Cleanup Script ==="
echo ""

# Stop running server
if [ -f /tmp/mcp-server.pid ]; then
  PID=$(cat /tmp/mcp-server.pid)
  echo "Stopping server (PID: $PID)..."
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
      echo "Force killing server..."
      kill -9 "$PID" 2>/dev/null || true
    fi
    echo "✓ Server stopped"
  else
    echo "Server not running"
  fi
  rm /tmp/mcp-server.pid
else
  echo "No PID file found, checking for running processes..."
  PIDS=$(pgrep -f "ai-agent-history-rag" || true)
  if [ -n "$PIDS" ]; then
    echo "Found running processes: $PIDS"
    echo "Kill them? (y/N)"
    read -r response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
      echo "$PIDS" | xargs kill 2>/dev/null || true
      sleep 1
      echo "$PIDS" | xargs kill -9 2>/dev/null || true
      echo "✓ Processes killed"
    fi
  else
    echo "No running server found"
  fi
fi

echo ""
echo "Clean up data? This will delete:"
echo "  - LanceDB database (~/.claude-history-rag/lancedb/)"
echo "  - State file (~/.claude-history-rag/state.json)"
echo "  - Log file (~/.claude-history-rag/claude-history-rag.log)"
echo ""
echo "Clean data? (y/N)"
read -r response

if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
  echo "Cleaning data..."
  rm -rf ~/.claude-history-rag/lancedb/
  rm -f ~/.claude-history-rag/state.json
  > ~/.claude-history-rag/claude-history-rag.log
  echo "✓ Data cleaned"
else
  echo "Keeping data"
fi

echo ""
echo "=== Cleanup complete ==="
