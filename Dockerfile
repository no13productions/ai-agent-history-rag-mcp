# Claude History RAG MCP Server
# Multi-stage build for minimal image size

FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock README.md ./

# Copy source code (needed for building the package)
COPY src/ src/

# Install dependencies + package (server extras)
RUN uv sync --frozen --extra server --no-dev


FROM python:3.12-slim as runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source and package info
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Create data directories
RUN mkdir -p /data/db /data/state /data/logs

# Default environment variables for server mode
ENV CLAUDE_HISTORY_RAG_DB_PATH=/data/db/lancedb \
    CLAUDE_HISTORY_RAG_STATE_PATH=/data/state/state.json \
    CLAUDE_HISTORY_RAG_STATUS_SERVER_HOST=0.0.0.0 \
    CLAUDE_HISTORY_RAG_STATUS_SERVER_PORT=4680 \
    CLAUDE_HISTORY_RAG_LOG_LEVEL=INFO

# Expose the API/dashboard port
EXPOSE 4680

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:4680/health || exit 1

# Run the daemon in server mode
CMD ["python", "-m", "claude_history_rag.daemon", "start"]
