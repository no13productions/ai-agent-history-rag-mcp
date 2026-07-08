#!/bin/bash
# Install the ai-agent-history-rag daemon as a Linux systemd user service

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="ai-agent-history-rag.service"
SERVICE_SRC="$SCRIPT_DIR/$SERVICE_NAME"
SERVICE_DEST="$HOME/.config/systemd/user/$SERVICE_NAME"

# Check if service file exists
if [ ! -f "$SERVICE_SRC" ]; then
    echo "Error: $SERVICE_SRC not found"
    exit 1
fi

# Find uv binary
UV_PATH=$(which uv 2>/dev/null || echo "$HOME/.local/bin/uv")
if [ ! -x "$UV_PATH" ]; then
    echo "Error: uv not found. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create systemd user directory
mkdir -p "$HOME/.config/systemd/user"

# Create log directory
mkdir -p "$HOME/.claude-history-rag"

# Generate service file with correct paths
echo "Generating service file..."
cat > "$SERVICE_DEST" << EOF
[Unit]
Description=AI Agent History RAG Daemon
After=network.target

[Service]
Type=simple
ExecStart=$UV_PATH --directory $PROJECT_DIR run ai-agent-history-rag-daemon supervise
Restart=on-failure
RestartSec=10
WorkingDirectory=$PROJECT_DIR

# Environment variables - edit as needed
# Client mode (connects to central server):
EOF

# Add environment variables if set
if [ -n "$CLAUDE_HISTORY_RAG_SERVER_URL" ]; then
    echo "Environment=\"CLAUDE_HISTORY_RAG_SERVER_URL=$CLAUDE_HISTORY_RAG_SERVER_URL\"" >> "$SERVICE_DEST"
fi
if [ -n "$CLAUDE_HISTORY_RAG_MACHINE_ID" ]; then
    echo "Environment=\"CLAUDE_HISTORY_RAG_MACHINE_ID=$CLAUDE_HISTORY_RAG_MACHINE_ID\"" >> "$SERVICE_DEST"
fi
if [ -n "$CLAUDE_RAG_EMBEDDING_BASE_URL" ]; then
    echo "Environment=\"CLAUDE_RAG_EMBEDDING_BASE_URL=$CLAUDE_RAG_EMBEDDING_BASE_URL\"" >> "$SERVICE_DEST"
fi
if [ -n "$CLAUDE_RAG_EMBEDDING_MODEL" ]; then
    echo "Environment=\"CLAUDE_RAG_EMBEDDING_MODEL=$CLAUDE_RAG_EMBEDDING_MODEL\"" >> "$SERVICE_DEST"
fi

cat >> "$SERVICE_DEST" << EOF

[Install]
WantedBy=default.target
EOF

# Reload systemd
echo "Reloading systemd..."
systemctl --user daemon-reload

# Enable and start service
echo "Enabling and starting service..."
systemctl --user enable --now ai-agent-history-rag

echo ""
echo "✓ Daemon installed and started!"
echo ""
echo "Commands:"
echo "  Status:   systemctl --user status ai-agent-history-rag"
echo "  Logs:     journalctl --user -u ai-agent-history-rag -f"
echo "  Stop:     systemctl --user stop ai-agent-history-rag"
echo "  Start:    systemctl --user start ai-agent-history-rag"
echo "  Restart:  systemctl --user restart ai-agent-history-rag"
echo "  Disable:  systemctl --user disable ai-agent-history-rag"
echo ""
echo "To configure environment variables, edit:"
echo "  $SERVICE_DEST"
echo "Then run: systemctl --user daemon-reload && systemctl --user restart ai-agent-history-rag"
echo ""
