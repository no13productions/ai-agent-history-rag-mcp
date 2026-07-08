#!/bin/bash
# Install the ai-agent-history-rag daemon as a macOS launchd service

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.ai-agent-history-rag.daemon.plist"
PLIST_TEMPLATE="$SCRIPT_DIR/$PLIST_NAME.template"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_NAME"

# Find uv binary
UV_PATH=$(which uv 2>/dev/null || echo "$HOME/.local/bin/uv")
if [ ! -x "$UV_PATH" ]; then
    echo "Error: uv not found. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Using uv at: $UV_PATH"
echo "Project directory: $PROJECT_DIR"

# Create LaunchAgents directory if needed
mkdir -p "$HOME/Library/LaunchAgents"

# Create log directory
mkdir -p "$HOME/.claude-history-rag"

# Stop existing service if running
if launchctl list 2>/dev/null | grep -q "com.ai-agent-history-rag.daemon"; then
    echo "Stopping existing service..."
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
fi

# Generate plist from template with correct paths
echo "Generating launch agent..."
if [ -f "$PLIST_TEMPLATE" ]; then
    sed -e "s|__UV_PATH__|$UV_PATH|g" \
        -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
        -e "s|__HOME__|$HOME|g" \
        "$PLIST_TEMPLATE" > "$PLIST_DEST"
else
    # Fallback: generate plist directly
    cat > "$PLIST_DEST" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ai-agent-history-rag.daemon</string>

    <key>ProgramArguments</key>
    <array>
        <string>$UV_PATH</string>
        <string>--directory</string>
        <string>$PROJECT_DIR</string>
        <string>run</string>
        <string>ai-agent-history-rag-daemon</string>
        <string>supervise</string>
    </array>

    <key>EnvironmentVariables</key>
    <dict>
        <key>CLAUDE_HISTORY_RAG_RUNTIME_CONTRACT</key>
        <string>production</string>
        <key>CLAUDE_HISTORY_RAG_STORAGE_BACKEND</key>
        <string>spanner</string>
        <key>CLAUDE_HISTORY_RAG_SPANNER_PROJECT</key>
        <string>jeeves-486102</string>
        <key>CLAUDE_HISTORY_RAG_SPANNER_INSTANCE</key>
        <string>jeeves-rg-spanner-prod-4d0e4c43</string>
        <key>CLAUDE_HISTORY_RAG_SPANNER_DATABASE</key>
        <string>ai-agent-history-rag</string>
        <key>CLAUDE_HISTORY_RAG_SPANNER_EMBEDDING_MODE</key>
        <string>spanner</string>
        <key>CLAUDE_HISTORY_RAG_SPANNER_EMBEDDING_MODEL_ID</key>
        <string>ConversationEmbeddingModel</string>
        <key>CLAUDE_HISTORY_RAG_EMBEDDING_PROVIDER</key>
        <string>vertex</string>
        <key>CLAUDE_HISTORY_RAG_EMBEDDING_MODEL</key>
        <string>gemini-embedding-001</string>
        <key>CLAUDE_HISTORY_RAG_EMBEDDING_DIMENSION</key>
        <string>3072</string>
        <key>CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST</key>
        <string>127.0.0.1</string>
        <key>CLAUDE_HISTORY_RAG_STATUS_SERVER_PORT</key>
        <string>4680</string>
        <key>GOOGLE_APPLICATION_CREDENTIALS</key>
        <string>$HOME/Meridian/alfred-sa-key.json</string>
        <key>GOOGLE_CLOUD_PROJECT</key>
        <string>jeeves-486102</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>$HOME/.claude-history-rag/launchd-stdout.log</string>

    <key>StandardErrorPath</key>
    <string>$HOME/.claude-history-rag/launchd-stderr.log</string>

    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
</dict>
</plist>
EOF
fi

# Load the service
echo "Starting service..."
launchctl load "$PLIST_DEST"

echo ""
echo "✓ Daemon installed and started!"
echo ""
echo "Commands:"
echo "  Status:    launchctl list | grep ai-agent-history-rag"
echo "  Logs:      tail -f ~/.claude-history-rag/daemon.log"
echo "  Stop:      launchctl unload ~/Library/LaunchAgents/$PLIST_NAME"
echo "  Start:     launchctl load ~/Library/LaunchAgents/$PLIST_NAME"
echo "  Uninstall: launchctl unload ~/Library/LaunchAgents/$PLIST_NAME && rm ~/Library/LaunchAgents/$PLIST_NAME"
echo ""
echo "To configure environment variables (e.g., for client mode), edit:"
echo "  $PLIST_DEST"
echo "Then run: launchctl unload $PLIST_DEST && launchctl load $PLIST_DEST"
echo ""
