"""Status collector module for monitoring MCP server health and metrics."""

import asyncio
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any

import psutil

from claude_history_rag.config import settings
from claude_history_rag.decision_engine.cache import get_search_cache
from claude_history_rag.embedder import get_embedder
from claude_history_rag.store import store
from claude_history_rag.watcher import get_watcher

logger = logging.getLogger(__name__)

# Global start time for uptime calculation
_start_time = time.time()
_start_datetime = datetime.now(timezone.utc)


def get_version() -> str:
    """Get server version from package."""
    return "0.1.0"  # Could be read from pyproject.toml dynamically


class StatusCollector:
    """Collects comprehensive server status and metrics."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.query_count = 0
        self.query_latencies: list[float] = []
        self.errors: list[dict[str, Any]] = []
        self.errors_by_type: dict[str, int] = {}

        # Register error storage with the errors module so record_error works
        from claude_history_rag.errors import register_error_storage

        register_error_storage(self.errors, self.errors_by_type)

    def record_query(self, latency_ms: float):
        """Record a query for metrics tracking."""
        self.query_count += 1
        self.query_latencies.append(latency_ms)
        # Keep only last 100 latencies to prevent memory growth
        if len(self.query_latencies) > 100:
            self.query_latencies.pop(0)

    def record_error(self, error_type: str, message: str, details: dict[str, Any] | None = None):
        """Record an error for tracking."""
        error_entry = {
            "type": error_type,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        }
        self.errors.append(error_entry)
        # Keep only last 50 errors
        if len(self.errors) > 50:
            self.errors.pop(0)

        # Update error counts by type
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

    async def collect_status(self, detail_level: str = "full") -> dict[str, Any]:
        """Collect comprehensive server status.

        Args:
            detail_level: "basic" for summary, "full" for detailed metrics
        """
        try:
            status = {
                "server": self._get_server_info(),
                "health": await self._get_health_status(),
            }

            if detail_level == "full":
                status.update(
                    {
                        "database": await self._get_database_stats(),
                        "indexing": await self._get_indexing_status(),
                        "performance": self._get_performance_metrics(),
                        "cache": self._get_cache_stats(),
                        "embedder": self._get_embedder_stats(),
                        "file_watcher": self._get_watcher_stats(),
                        "clients": self._get_client_registry_stats(),
                        "errors": self._get_error_stats(),
                        "configuration": self._get_configuration(),
                    }
                )

            return status

        except Exception as e:
            logger.error(f"Failed to collect status: {e}", exc_info=True)
            return {
                "error": "Failed to collect status",
                "message": str(e),
                "server": self._get_server_info(),
            }

    def _get_server_info(self) -> dict[str, Any]:
        """Get basic server information."""
        uptime = time.time() - _start_time

        # Get actual status server port if running
        status_server_port = None
        status_server_url = None
        try:
            from claude_history_rag.status_server import get_status_server

            server = get_status_server()
            if server and hasattr(server, "port"):
                status_server_port = server.port
                status_server_url = f"http://{server.host}:{server.port}"
        except Exception:
            pass  # Status server not running or not available

        result = {
            "version": get_version(),
            "uptime_seconds": round(uptime, 2),
            "started_at": _start_datetime.isoformat(),
            "pid": os.getpid(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        }

        # Add status server info if available
        if status_server_port:
            result["status_server_port"] = status_server_port
            result["status_server_url"] = status_server_url
            result["dashboard_url"] = f"{status_server_url}/dashboard"

        return result

    async def _get_health_status(self) -> dict[str, Any]:
        """Get health status with component checks."""
        checks = {}

        # Database check
        try:
            stats = await store.get_stats_async()
            checks["database"] = {
                "status": "ok",
                "chunks": stats.get("total_chunks", 0),
            }
            if stats.get("error"):
                checks["database"] = {
                    "status": "error",
                    "error": stats.get("error"),
                }
        except Exception as e:
            checks["database"] = {"status": "error", "error": str(e)}
            db_error = str(e)
        else:
            db_error = stats.get("error")

        # Data integrity check
        if db_error:
            checks["database_integrity"] = {
                "status": "error",
                "error": db_error,
            }

        # Embedder check
        try:
            embedder = get_embedder()
            checks["embedder"] = {
                "status": "ok" if embedder else "error",
                "model_loaded": embedder is not None,
            }
        except Exception as e:
            checks["embedder"] = {"status": "error", "error": str(e)}

        # File watcher check
        try:
            watcher = get_watcher()
            checks["file_watcher"] = {
                "status": "ok" if watcher.is_running else "stopped",
                "is_running": watcher.is_running,
            }
        except Exception as e:
            checks["file_watcher"] = {"status": "error", "error": str(e)}

        # Cache check
        try:
            cache = get_search_cache()
            checks["cache"] = {
                "status": "ok",
                "size": cache._lru.size,
            }
        except Exception as e:
            checks["cache"] = {"status": "error", "error": str(e)}

        # FTS index check (degraded if not available, not error)
        try:
            fts_available = store.has_fts_index()
            checks["fts_index"] = {
                "status": "ok" if fts_available else "degraded",
                "available": fts_available,
                "message": "Hybrid search enabled"
                if fts_available
                else "Falling back to vector-only search (install tantivy for hybrid search)",
            }
        except Exception as e:
            checks["fts_index"] = {"status": "error", "error": str(e)}

        # Overall status
        all_ok = all(check.get("status") == "ok" for check in checks.values())
        any_error = any(check.get("status") == "error" for check in checks.values())

        overall_status = "healthy" if all_ok else ("unhealthy" if any_error else "degraded")

        return {"status": overall_status, "checks": checks}

    async def _get_database_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        try:
            stats = await store.get_stats_async()
            
            # Add file size if SQLite
            if settings.storage_backend == "sqlite":
                if settings.sqlite_db_path.exists():
                    stats["database_size_bytes"] = settings.sqlite_db_path.stat().st_size

            return stats
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}

    async def _get_indexing_status(self) -> dict[str, Any]:
        """Get indexing progress and status."""
        try:
            watcher = get_watcher()
            from claude_history_rag.codex.watcher import get_codex_watcher
            from claude_history_rag.gemini.watcher import get_gemini_watcher

            # Count JSONL files
            files_discovered = 0
            if settings.projects_path.exists():
                files_discovered = len(list(settings.projects_path.glob("**/*.jsonl")))

            # Get indexed files from state
            files_indexed = len(watcher.state.get_all_files())
            files_pending = max(0, files_discovered - files_indexed)

            codex_watcher = get_codex_watcher()
            gemini_watcher = get_gemini_watcher()
            codex_files_discovered = 0
            if settings.codex_sessions_path.exists():
                codex_files_discovered = len(
                    list(settings.codex_sessions_path.glob("**/*.jsonl"))
                )
            codex_files_indexed = len(codex_watcher.state.get_all_files())
            codex_files_pending = max(0, codex_files_discovered - codex_files_indexed)
            gemini_files_discovered = 0
            if settings.gemini_sessions_path.exists():
                gemini_files_discovered = len(
                    list(settings.gemini_sessions_path.glob("**/*.json"))
                )
            gemini_files_indexed = len(gemini_watcher.state.get_all_files())
            gemini_files_pending = max(0, gemini_files_discovered - gemini_files_indexed)

            return {
                "status": "active" if watcher.is_running else "stopped",
                "files_discovered": files_discovered,
                "files_indexed": files_indexed,
                "files_pending": files_pending,
                "files_failed": len(watcher._failed_files),
                "failed_files": list(watcher._failed_files),
                "codex_files_discovered": codex_files_discovered,
                "codex_files_indexed": codex_files_indexed,
                "codex_files_pending": codex_files_pending,
                "gemini_files_discovered": gemini_files_discovered,
                "gemini_files_indexed": gemini_files_indexed,
                "gemini_files_pending": gemini_files_pending,
            }
        except Exception as e:
            logger.error(f"Failed to get indexing status: {e}")
            return {"error": str(e)}

    def _get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        try:
            # Memory usage
            mem_info = self.process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024

            # CPU usage (non-blocking, returns cached value)
            cpu_percent = self.process.cpu_percent()

            # Query metrics
            avg_latency = (
                sum(self.query_latencies) / len(self.query_latencies)
                if self.query_latencies
                else 0.0
            )

            # Calculate queries per minute (approximate)
            uptime_minutes = max(1, (time.time() - _start_time) / 60)
            qpm = self.query_count / uptime_minutes

            return {
                "memory_usage_mb": round(mem_mb, 2),
                "cpu_percent": round(cpu_percent, 1),
                "queries_total": self.query_count,
                "queries_per_minute": round(qpm, 2),
                "avg_query_latency_ms": round(avg_latency, 2),
            }
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}

    def _get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            cache = get_search_cache()

            # Access the LRU cache stats through the SearchCache wrapper
            return {
                "size": cache._lru.size,
                "max_size": cache._lru.maxsize,
                "hit_rate": round(cache._lru.hit_rate, 3),
                "hits": cache._lru._hits,
                "misses": cache._lru._misses,
                "ttl_seconds": cache.default_ttl,
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}

    def _get_embedder_stats(self) -> dict[str, Any]:
        """Get embedder statistics."""
        try:
            embedder = get_embedder()

            # Determine dimension based on model (use store's mapping)
            from claude_history_rag.store import get_vector_dim

            dimension = get_vector_dim()

            return {
                "model": settings.embedding_model,
                "embedding_url": settings.embedding_base_url,
                "dimension": dimension,
                "loaded": embedder is not None,
            }
        except Exception as e:
            logger.error(f"Failed to get embedder stats: {e}")
            return {"error": str(e)}

    def _get_watcher_stats(self) -> dict[str, Any]:
        """Get file watcher statistics."""
        try:
            watcher = get_watcher()
            from claude_history_rag.codex.watcher import get_codex_watcher
            from claude_history_rag.gemini.watcher import get_gemini_watcher
            codex_watcher = get_codex_watcher()
            gemini_watcher = get_gemini_watcher()

            return {
                "is_running": watcher.is_running,
                "projects_path": str(settings.projects_path),
                "codex_sessions_path": str(settings.codex_sessions_path),
                "gemini_sessions_path": str(settings.gemini_sessions_path),
                "debounce_ms": watcher.debounce_ms,
                "queue_size": watcher.queue.qsize(),
                "queue_max_size": watcher.queue.maxsize,
                "failed_files_count": len(watcher._failed_files),
                "codex_is_running": codex_watcher.is_running,
                "codex_queue_size": codex_watcher.queue.qsize(),
                "codex_queue_max_size": codex_watcher.queue.maxsize,
                "codex_failed_files_count": len(codex_watcher._failed_files),
                "gemini_is_running": gemini_watcher.is_running,
                "gemini_queue_size": gemini_watcher.queue.qsize(),
                "gemini_queue_max_size": gemini_watcher.queue.maxsize,
                "gemini_failed_files_count": len(gemini_watcher._failed_files),
            }
        except Exception as e:
            logger.error(f"Failed to get watcher stats: {e}")
            return {"error": str(e)}

    def _get_error_stats(self) -> dict[str, Any]:
        """Get error statistics."""
        return {
            "total": len(self.errors),
            "recent": self.errors[-10:],  # Last 10 errors
            "by_type": self.errors_by_type.copy(),
        }

    def _get_client_registry_stats(self) -> dict[str, Any]:
        """Get client registry status."""
        try:
            from claude_history_rag.client_registry import get_client_registry

            registry = get_client_registry()
            return registry.get_client_status()
        except Exception as e:
            logger.error(f"Failed to get client registry stats: {e}")
            return {"error": str(e)}

    def _get_configuration(self) -> dict[str, Any]:
        """Get current configuration."""
        return {
            "storage_backend": settings.storage_backend,
            "sqlite_db_path": str(settings.sqlite_db_path),
            "qdrant_url": settings.qdrant_url,
            "projects_path": str(settings.projects_path),
            "codex_sessions_path": str(settings.codex_sessions_path),
            "gemini_sessions_path": str(settings.gemini_sessions_path),
            "embedding_model": settings.embedding_model,
            "embedding_url": settings.embedding_base_url,
            "log_level": settings.log_level,
            "batch_size": settings.batch_size,
            "status_server_enabled": settings.status_server_enabled,
            "status_server_port_configured": settings.status_server_port,
            "auth_enabled": settings.auth_enabled,
        }


# Global status collector instance
_status_collector: StatusCollector | None = None
_collector_lock = asyncio.Lock()


async def get_status_collector() -> StatusCollector:
    """Get or create the global status collector instance."""
    global _status_collector
    if _status_collector is None:
        async with _collector_lock:
            if _status_collector is None:
                _status_collector = StatusCollector()
    return _status_collector


def get_status_collector_sync() -> StatusCollector | None:
    """Get the status collector instance synchronously (returns None if not initialized)."""
    return _status_collector


# Re-export record_error from errors module for backwards compatibility
from claude_history_rag.errors import record_error  # noqa: E402, F401
