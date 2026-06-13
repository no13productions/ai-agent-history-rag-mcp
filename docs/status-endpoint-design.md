# Status Endpoint Design

## Overview

Design for a comprehensive status/monitoring system for the Claude History RAG MCP server with:
- JSON API endpoint for programmatic access
- HTML dashboard for human visualization
- Prometheus-compatible metrics export
- MCP tool integration for Claude to query status

## Access Methods Analysis

### Option 1: Separate HTTP Server (RECOMMENDED)
**Pros:**
- Clean separation of concerns (STDIO for MCP, HTTP for monitoring)
- No interference with MCP protocol
- Standard HTTP status codes and content negotiation
- Can use standard Python web frameworks
- Easy to secure (firewall rules, localhost-only)
- Prometheus scraping works out of the box

**Cons:**
- Additional port to manage
- Slightly more complexity

**Implementation:** asyncio HTTP server (aiohttp) on localhost:8765

### Option 2: MCP Tool for Status
**Pros:**
- No additional HTTP server needed
- Claude can directly query status via MCP
- Follows MCP patterns

**Cons:**
- Only accessible through MCP client
- No web dashboard without separate solution
- Can't be scraped by Prometheus
- Limited to text output

**Implementation:** Add `get_server_status` MCP tool

### Option 3: File-Based Status
**Pros:**
- Simplest implementation
- No network ports
- Can be read by any process

**Cons:**
- No real-time queries
- Manual refresh required
- No web dashboard
- File I/O overhead

### RECOMMENDED HYBRID APPROACH:
**Implement both Option 1 (HTTP) AND Option 2 (MCP tool)**
- HTTP server for monitoring, dashboards, and Prometheus
- MCP tool for Claude to check status programmatically
- Both read from the same status collector module

## Status JSON Schema

### Core Status Data Structure

```json
{
  "server": {
    "version": "0.1.0",
    "uptime_seconds": 3600,
    "started_at": "2025-12-16T09:22:15Z",
    "pid": 72157,
    "python_version": "3.13.1",
    "platform": "Darwin-25.1.0-arm64"
  },
  "health": {
    "status": "healthy",  // "healthy", "degraded", "unhealthy"
    "checks": {
      "database": {"status": "ok", "latency_ms": 2.3},
      "embedder": {"status": "ok", "model_loaded": true},
      "file_watcher": {"status": "ok", "is_running": true},
      "cache": {"status": "ok", "size": 45}
    }
  },
  "database": {
    "total_chunks": 22,
    "total_conversations": 1,
    "total_projects": 1,
    "index_created": false,
    "index_type": null,
    "database_size_bytes": 102400,
    "oldest_chunk": "2024-12-14T17:42:00Z",
    "newest_chunk": "2025-12-15T14:09:00Z"
  },
  "indexing": {
    "status": "idle",  // "idle", "active", "paused", "error"
    "files_discovered": 558,
    "files_indexed": 1,
    "files_pending": 557,
    "files_failed": 0,
    "chunks_processed": 22,
    "last_indexed_file": "agent-a24ba2c.jsonl",
    "last_indexed_at": "2025-12-16T09:22:30Z",
    "current_file": null,
    "current_progress": null
  },
  "performance": {
    "memory_usage_mb": 150.5,
    "memory_percent": 0.9,
    "cpu_percent": 0.5,
    "queries_total": 0,
    "queries_per_minute": 0.0,
    "avg_query_latency_ms": 0.0,
    "embedding_time_total_seconds": 8.95,
    "chunks_per_second": 2.46
  },
  "cache": {
    "size": 45,
    "max_size": 100,
    "hit_rate": 0.0,
    "hits": 0,
    "misses": 0,
    "evictions": 0,
    "ttl_seconds": 300
  },
  "embedder": {
    "model": "nomic-ai/nomic-embed-text-v1.5",
    "dimension": 768,
    "loaded": true,
    "batches_processed": 1,
    "batches_failed": 0,
    "total_chunks_embedded": 22
  },
  "file_watcher": {
    "is_running": true,
    "projects_path": "/Users/youruser/.claude/projects",
    "debounce_ms": 5000,
    "queue_size": 0,
    "queue_max_size": 1000,
    "failed_files": []
  },
  "errors": {
    "total": 0,
    "recent": [],  // Last 10 errors with timestamps
    "by_type": {}
  },
  "configuration": {
    "db_path": "/Users/youruser/.claude-history-rag/lancedb",
    "projects_path": "/Users/youruser/.claude/projects",
    "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
    "log_level": "INFO",
    "batch_size": 32
  }
}
```

## Implementation Plan

### Phase 1: Status Collector Module
**File:** `src/claude_history_rag/status.py`

Create a central status collector that gathers metrics from:
- Store (database stats)
- Watcher (indexing progress)
- Embedder (model status)
- Cache (hit rates)
- System (CPU, memory, uptime)

### Phase 2: HTTP Status Server
**File:** `src/claude_history_rag/status_server.py`

Using `aiohttp` (async HTTP framework):
- GET `/status` - Returns JSON
- GET `/status?format=prometheus` - Prometheus metrics format
- GET `/` or `/dashboard` - HTML dashboard
- GET `/health` - Simple health check (200 OK or 503 Service Unavailable)
- GET `/metrics` - Prometheus-compatible metrics endpoint

**Configuration:**
- Default port: 8765
- Bind to localhost only (security)
- Optional: environment variable to enable/disable

### Phase 3: HTML Dashboard
**File:** `src/claude_history_rag/templates/dashboard.html`

Single-page dashboard with:
- **Header:** Server version, uptime, health status badge
- **Database Stats:** Total chunks, conversations, size
- **Indexing Progress:** Progress bar, files pending, current file
- **Performance Metrics:** Memory, CPU, query latency charts
- **Recent Activity:** Last indexed files, recent errors
- **System Info:** Configuration, paths, model info

**Technology:**
- Pure HTML + vanilla JavaScript (no framework dependencies)
- Auto-refresh every 5 seconds
- Chart.js for visualizations (CDN)

### Phase 4: MCP Tool Integration
**File:** `src/claude_history_rag/server.py`

Add new MCP tool:
```python
@mcp.tool()
async def get_server_status(
    detail_level: Literal["basic", "full"] = "basic"
) -> dict:
    """Get MCP server status and health information.

    Args:
        detail_level: "basic" for summary, "full" for detailed metrics
    """
```

This allows Claude to check server health programmatically.

### Phase 5: Prometheus Metrics
**Format:** Standard Prometheus text format

Key metrics to expose:
```
# HELP mcp_server_uptime_seconds Server uptime in seconds
# TYPE mcp_server_uptime_seconds gauge
mcp_server_uptime_seconds 3600

# HELP mcp_chunks_total Total chunks indexed
# TYPE mcp_chunks_total counter
mcp_chunks_total 22

# HELP mcp_indexing_files_pending Number of files pending indexing
# TYPE mcp_indexing_files_pending gauge
mcp_indexing_files_pending 557

# HELP mcp_query_duration_seconds Query execution time
# TYPE mcp_query_duration_seconds histogram
mcp_query_duration_seconds_bucket{le="0.1"} 10
mcp_query_duration_seconds_bucket{le="0.5"} 25
mcp_query_duration_seconds_bucket{le="1.0"} 30
mcp_query_duration_seconds_sum 15.5
mcp_query_duration_seconds_count 30

# HELP mcp_memory_usage_bytes Memory usage in bytes
# TYPE mcp_memory_usage_bytes gauge
mcp_memory_usage_bytes 157810688

# HELP mcp_cache_hits_total Cache hit count
# TYPE mcp_cache_hits_total counter
mcp_cache_hits_total 45

# HELP mcp_cache_misses_total Cache miss count
# TYPE mcp_cache_misses_total counter
mcp_cache_misses_total 5
```

## Security Considerations

1. **Localhost Only:** Bind HTTP server to 127.0.0.1 by default
2. **Optional Authentication:** Environment variable for API key
3. **Rate Limiting:** Prevent status endpoint abuse
4. **Read-Only:** Status endpoints never modify state
5. **Sanitize Paths:** Don't expose full file system paths in public endpoints

## Configuration

Add to `config.py`:
```python
status_server_enabled: bool = True
status_server_host: str = "127.0.0.1"
status_server_port: int = 8765
status_refresh_interval: int = 5  # seconds
```

## Benefits

### For Developers:
- Quick health checks during development
- Debug performance issues
- Monitor indexing progress
- Understand system behavior

### For Production:
- Integration with monitoring systems (Prometheus/Grafana)
- Alerting on failures (Alertmanager)
- Capacity planning (memory/storage trends)
- SLA monitoring

### For Claude:
- Self-check server status before queries
- Provide context about available data
- Detect and report issues
- Estimate query readiness

## Example Usage

### CLI Check
```bash
curl http://localhost:8765/health
# {"status": "healthy"}

curl http://localhost:8765/status | jq .
# Full JSON status

open http://localhost:8765/dashboard
# Opens HTML dashboard in browser
```

### Claude MCP Tool
```
User: Is the history RAG server working?
Claude: *calls get_server_status tool*
The server is healthy and has indexed 22 chunks across 1 file,
with 557 files pending. Memory usage is 150MB.
```

### Prometheus Scraping
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'mcp-server'
    static_configs:
      - targets: ['localhost:8765']
    metrics_path: '/metrics'
```

## Implementation Priority

1. ✅ **High:** Status collector module (core functionality)
2. ✅ **High:** Basic HTTP server with JSON endpoint
3. ✅ **High:** MCP tool integration
4. **Medium:** HTML dashboard
5. **Medium:** Prometheus metrics endpoint
6. **Low:** Advanced features (historical data, alerts)

## Dependencies

New dependencies to add:
```toml
aiohttp>=3.9.0      # Async HTTP server
psutil>=5.9.0       # System metrics (CPU, memory)
```

Optional (for dashboard):
- No additional Python deps (uses CDN for Chart.js)

## Resources

Based on research:
- [MCP Best Practices: Architecture & Implementation Guide](https://modelcontextprotocol.info/docs/best-practices/)
- [7 MCP Server Best Practices for Scalable AI Integrations in 2025](https://www.marktechpost.com/2025/07/23/7-mcp-server-best-practices-for-scalable-ai-integrations-in-2025/)
- [Python Monitoring with Prometheus (Beginner's Guide)](https://betterstack.com/community/guides/monitoring/prometheus-python-metrics/)
- [Prometheus HTTP API](https://prometheus.io/docs/prometheus/latest/querying/api/)
