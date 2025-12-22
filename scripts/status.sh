#!/bin/bash
# Status script for ai-agent-history-rag MCP server

set -e

PID_FILE="/tmp/mcp-server.pid"
LOG_FILE="$HOME/.claude-history-rag/claude-history-rag.log"
DB_PATH="$HOME/.claude-history-rag/lancedb/"

echo "=== AI Agent History RAG MCP Server Status ==="
echo ""

# Check if running
if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "✓ Server is running (PID: $PID)"

    # Get uptime
    if ps -p "$PID" -o etime= > /dev/null 2>&1; then
      UPTIME=$(ps -p "$PID" -o etime= | tr -d ' ')
      echo "  Uptime: $UPTIME"
    fi

    # Get memory usage
    if ps -p "$PID" -o rss= > /dev/null 2>&1; then
      RSS_KB=$(ps -p "$PID" -o rss= | tr -d ' ')
      RSS_MB=$((RSS_KB / 1024))
      echo "  Memory: ${RSS_MB}MB"
    fi
  else
    echo "✗ Server not running (stale PID file)"
  fi
else
  echo "✗ Server not running (no PID file)"
fi

echo ""
echo "=== Database Status ==="

# Check if database exists
if [ -d "$DB_PATH" ]; then
  echo "✓ Database exists: $DB_PATH"
  DB_SIZE=$(du -sh "$DB_PATH" | cut -f1)
  echo "  Size: $DB_SIZE"

  # Count chunks using Python
  if command -v ~/.local/bin/uv &> /dev/null; then
    CHUNK_COUNT=$(~/.local/bin/uv run python -c "
from claude_history_rag.store import store
try:
    stats = store.get_stats()
    print(stats.get('total_chunks', 0))
except Exception as e:
    print('error')
" 2>/dev/null || echo "error")

    if [ "$CHUNK_COUNT" != "error" ]; then
      echo "  Chunks indexed: $CHUNK_COUNT"
    fi
  fi
else
  echo "✗ Database not initialized"
fi

echo ""
echo "=== Log Status ==="

if [ -f "$LOG_FILE" ]; then
  LOG_SIZE=$(wc -l < "$LOG_FILE" | tr -d ' ')
  echo "✓ Log file: $LOG_FILE"
  echo "  Lines: $LOG_SIZE"

  # Check for recent errors
  RECENT_ERRORS=$(tail -100 "$LOG_FILE" | grep -c "ERROR" || echo "0")
  if [ "$RECENT_ERRORS" -gt 0 ]; then
    echo "  ⚠️  Recent errors: $RECENT_ERRORS (last 100 lines)"
  else
    echo "  No recent errors"
  fi
else
  echo "✗ Log file not found"
fi

echo ""
