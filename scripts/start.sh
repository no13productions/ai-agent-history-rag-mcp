#!/usr/bin/env bash
# Foreground STDIO entrypoint for the production-gated native MCP proxy.
# MCP clients own this process lifecycle; the indexing daemon remains separate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/history-rag-mcp-native.sh" "$@"
