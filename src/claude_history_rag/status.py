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
from claude_history_rag.embedder import get_embedder, redact_url
from claude_history_rag.store import store
from claude_history_rag.watcher import get_all_watchers

logger = logging.getLogger(__name__)

# Global start time for uptime calculation
_start_time = time.time()
_start_datetime = datetime.now(timezone.utc)
HEALTH_STATS_TIMEOUT_SECONDS = 1.0
FULL_STATS_TIMEOUT_SECONDS = 1.0
FTS_CHECK_TIMEOUT_SECONDS = 1.0
FAILED_FILES_STATUS_LIMIT = 20


def get_version() -> str:
    """Get server version from package."""
    return "0.1.0"  # Could be read from pyproject.toml dynamically


def _safe_error(context: str, exc: Exception) -> str:
    """Return a client-safe error summary for status payloads."""
    return f"{context}: {type(exc).__name__}"


def _safe_recent_error(error: dict[str, Any]) -> dict[str, Any]:
    """Return a redacted error entry suitable for status payloads."""
    details = error.get("details") or {}
    safe_details: dict[str, Any] = {}
    if isinstance(details, dict):
        for key, value in details.items():
            if isinstance(value, str):
                safe_details[key] = "<redacted>"
            elif isinstance(value, (int, float, bool)) or value is None:
                safe_details[key] = value
            else:
                safe_details[key] = type(value).__name__
    return {
        "type": error.get("type"),
        "message": error.get("message"),
        "timestamp": error.get("timestamp"),
        "details": safe_details,
    }


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

    def _uses_spanner_native_embeddings(self) -> bool:
        """Return whether Spanner ML.PREDICT is the active embedding path."""
        return (
            settings.storage_backend == "spanner" and settings.spanner_embedding_mode == "spanner"
        )

    def _get_cached_store_stats(self) -> dict[str, Any] | None:
        """Return the store's last stats snapshot when the backend exposes one."""
        cached = getattr(store, "_stats_cache", None)
        if (
            isinstance(cached, tuple)
            and len(cached) == 2
            and isinstance(cached[1], dict)
        ):
            return cached[1]
        return None

    async def _get_store_stats_bounded(
        self,
        timeout_seconds: float,
    ) -> tuple[dict[str, Any] | None, str, str | None]:
        """Return store stats without letting a cache miss wedge status collection."""
        stats = self._get_cached_store_stats()
        if stats is not None:
            return stats, "cache", None
        try:
            stats = await asyncio.wait_for(store.get_stats_async(), timeout=timeout_seconds)
            return stats, "fresh", None
        except asyncio.TimeoutError:
            return None, "timeout", f"Store stats timed out after {timeout_seconds:.1f}s"

    async def _get_fts_available_bounded(self) -> tuple[bool | None, str, str | None]:
        """Return FTS availability without blocking the status server event loop."""
        stats = self._get_cached_store_stats()
        if stats is not None and "fts_index_available" in stats:
            return bool(stats.get("fts_index_available")), "cache", None
        try:
            loop = asyncio.get_running_loop()
            available = await asyncio.wait_for(
                loop.run_in_executor(None, store.has_fts_index),
                timeout=FTS_CHECK_TIMEOUT_SECONDS,
            )
            return bool(available), "fresh", None
        except asyncio.TimeoutError:
            return (
                None,
                "timeout",
                f"FTS index check timed out after {FTS_CHECK_TIMEOUT_SECONDS:.1f}s",
            )

    def _database_stats_payload(
        self,
        stats: dict[str, Any] | None,
        stats_source: str,
        degraded_reason: str | None = None,
    ) -> dict[str, Any]:
        """Format database stats consistently for health and full status payloads."""
        if stats is None:
            return {
                "status": "degraded",
                "total_chunks": 0,
                "stats_source": stats_source,
                "degraded_reason": degraded_reason or "Database stats unavailable",
            }

        result = {
            "status": "degraded" if degraded_reason or stats.get("error") else "ok",
            "total_chunks": stats.get("total_chunks", 0),
            "embedded_chunks": stats.get("embedded_chunks"),
            "awaiting_embedding": stats.get("awaiting_embedding"),
            "backfill_rate_per_min": stats.get("backfill_rate_per_min"),
            "backfill_eta_seconds": stats.get("backfill_eta_seconds"),
            "stats_source": stats_source,
        }
        if degraded_reason:
            result["degraded_reason"] = degraded_reason
        if stats.get("error"):
            result["degraded_reason"] = "Database stats unavailable"

        if settings.storage_backend == "lancedb":
            db_size = 0
            if settings.db_path.exists():
                for file_path in settings.db_path.rglob("*"):
                    if file_path.is_file():
                        db_size += file_path.stat().st_size
            result["database_size_bytes"] = db_size
            result["database_path"] = str(settings.db_path)

        for key in (
            "backend",
            "project",
            "instance",
            "database",
            "dimension",
            "fts_index_available",
            "vector_index_available",
            "vector_search_mode",
            "embedding_mode",
            "embedding_model_id",
        ):
            if key in stats:
                result[key] = stats[key]
        return result

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
                "message": f"Status collection failed: {type(e).__name__}",
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
            stats, stats_source, degraded_reason = await self._get_store_stats_bounded(
                HEALTH_STATS_TIMEOUT_SECONDS
            )
            checks["database"] = {
                "status": "degraded" if degraded_reason or stats is None else "ok",
                "chunks": stats.get("total_chunks", 0) if stats else 0,
                "stats_source": stats_source,
            }
            if degraded_reason:
                checks["database"]["degraded_reason"] = degraded_reason
            if stats and stats.get("error"):
                checks["database"]["status"] = "degraded"
                checks["database"]["degraded_reason"] = "Database stats unavailable"
        except Exception as e:
            checks["database"] = {
                "status": "degraded",
                "error": _safe_error("Database check failed", e),
            }
            db_error = str(e)
        else:
            db_error = stats.get("error") if stats else degraded_reason

        # Data integrity check (e.g., missing Lance files)
        if db_error and "LanceError(IO)" in str(db_error):
            checks["database_integrity"] = {
                "status": "error",
                "error": "LanceDB files missing or corrupted. Re-index required.",
            }

        # Embedder check
        try:
            if self._uses_spanner_native_embeddings():
                checks["embedder"] = {
                    "status": "ok",
                    "model_loaded": True,
                    "provider": "spanner",
                    "model_id": settings.spanner_embedding_model_id,
                    "message": "Using Spanner ML.PREDICT for embeddings",
                }
            else:
                embedder = get_embedder()
                checks["embedder"] = {
                    "status": "ok" if embedder else "error",
                    "model_loaded": embedder is not None,
                }
        except Exception as e:
            checks["embedder"] = {
                "status": "error",
                "error": _safe_error("Embedder check failed", e),
            }

        # File watcher check
        try:
            watchers = get_all_watchers()
            running_count = sum(1 for watcher in watchers if watcher.is_running)
            any_running = running_count > 0
            all_running = bool(watchers) and running_count == len(watchers)
            checks["file_watcher"] = {
                "status": "ok" if any_running else "stopped",
                "is_running": any_running,
                "all_sources_running": all_running,
                "running_sources": running_count,
                "total_sources": len(watchers),
            }
        except Exception as e:
            checks["file_watcher"] = {
                "status": "error",
                "error": _safe_error("File watcher check failed", e),
            }

        # Cache check
        try:
            cache = get_search_cache()
            checks["cache"] = {
                "status": "ok",
                "size": cache._lru.size,
            }
        except Exception as e:
            checks["cache"] = {"status": "error", "error": _safe_error("Cache check failed", e)}

        # FTS index check (degraded if not available, not error)
        try:
            fts_available, fts_source, fts_degraded_reason = await self._get_fts_available_bounded()
            if fts_available:
                fts_message = "Hybrid search enabled"
            elif fts_degraded_reason:
                fts_message = "Spanner full-text search index status unavailable"
            elif settings.storage_backend == "spanner":
                fts_message = "Spanner full-text search index unavailable; using vector-only search"
            else:
                fts_message = (
                    "Falling back to vector-only search (install tantivy for hybrid search)"
                )
            checks["fts_index"] = {
                "status": "ok" if fts_available else "degraded",
                "available": fts_available,
                "message": fts_message,
                "stats_source": fts_source,
            }
            if fts_degraded_reason:
                checks["fts_index"]["degraded_reason"] = fts_degraded_reason
        except Exception as e:
            checks["fts_index"] = {
                "status": "degraded",
                "error": _safe_error("FTS check failed", e),
            }

        # Overall status
        all_ok = all(check.get("status") == "ok" for check in checks.values())
        any_error = any(check.get("status") == "error" for check in checks.values())

        overall_status = "healthy" if all_ok else ("unhealthy" if any_error else "degraded")

        return {"status": overall_status, "checks": checks}

    async def _get_database_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        try:
            stats, stats_source, degraded_reason = await self._get_store_stats_bounded(
                FULL_STATS_TIMEOUT_SECONDS
            )
            return self._database_stats_payload(stats, stats_source, degraded_reason)
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {
                "status": "degraded",
                "error": _safe_error("Database stats failed", e),
                "stats_source": "error",
            }

    async def _get_indexing_status(self) -> dict[str, Any]:
        """Get indexing progress and status."""
        try:
            watchers = get_all_watchers()
            source_status: dict[str, dict[str, Any]] = {}
            for source_watcher in watchers:
                indexed = len(source_watcher.state.get_all_files())
                queue_size = source_watcher.queue.qsize()
                failed_files = (
                    source_watcher.failed_files() if hasattr(source_watcher, "failed_files") else []
                )
                source_status[source_watcher.source_name] = {
                    "files_discovered": indexed + queue_size,
                    "files_indexed": indexed,
                    "files_pending": queue_size,
                    "files_failed": source_watcher.failed_files_count,
                    "failed_files": failed_files[:FAILED_FILES_STATUS_LIMIT],
                    "failed_files_truncated": len(failed_files) > FAILED_FILES_STATUS_LIMIT,
                    "is_running": source_watcher.is_running,
                    "watch_path": str(source_watcher.projects_path),
                    "discovery_source": "watcher_state",
                }
            files_discovered = sum(s["files_discovered"] for s in source_status.values())
            files_indexed = sum(s["files_indexed"] for s in source_status.values())
            files_pending = sum(s["files_pending"] for s in source_status.values())
            files_failed = sum(s["files_failed"] for s in source_status.values())
            running_sources = sum(1 for watcher in watchers if watcher.is_running)

            return {
                "status": "active" if running_sources else "stopped",
                "files_discovered": files_discovered,
                "files_indexed": files_indexed,
                "files_pending": files_pending,
                "files_failed": files_failed,
                "running_sources": running_sources,
                "total_sources": len(watchers),
                "all_sources_running": bool(watchers) and running_sources == len(watchers),
                "sources": source_status,
            }
        except Exception as e:
            logger.error(f"Failed to get indexing status: {e}")
            return {"error": _safe_error("Indexing status failed", e)}

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
            return {"error": _safe_error("Performance metrics failed", e)}

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
            return {"error": _safe_error("Cache stats failed", e)}

    def _get_embedder_stats(self) -> dict[str, Any]:
        """Get embedder statistics."""
        try:
            # Determine dimension based on model (use store's mapping)
            from claude_history_rag.store import get_vector_dim

            dimension = get_vector_dim()
            if self._uses_spanner_native_embeddings():
                return {
                    "provider": "spanner",
                    "model": settings.embedding_model,
                    "model_id": settings.spanner_embedding_model_id,
                    "dimension": dimension,
                    "vertex_project": settings.vertex_project,
                    "vertex_location": settings.vertex_location,
                    "loaded": True,
                }

            embedder = get_embedder()

            return {
                "provider": settings.embedding_provider,
                "model": settings.embedding_model,
                "embedding_url": redact_url(settings.embedding_base_url),
                "dimension": dimension,
                "vertex_project": settings.vertex_project,
                "vertex_location": settings.vertex_location,
                "loaded": embedder is not None,
            }
        except Exception as e:
            logger.error(f"Failed to get embedder stats: {e}")
            return {"error": _safe_error("Embedder stats failed", e)}

    def _get_watcher_stats(self) -> dict[str, Any]:
        """Get file watcher statistics."""
        try:
            watchers = get_all_watchers()
            source_stats = {
                watcher.source_name: {
                    "is_running": watcher.is_running,
                    "queue_size": watcher.queue.qsize(),
                    "queue_max_size": watcher.queue.maxsize,
                    "failed_files_count": watcher.failed_files_count,
                    "watch_path": str(watcher.projects_path),
                }
                for watcher in watchers
            }
            queue_size = sum(source["queue_size"] for source in source_stats.values())
            failed_count = sum(source["failed_files_count"] for source in source_stats.values())
            running_sources = sum(1 for watcher in watchers if watcher.is_running)

            return {
                "is_running": running_sources > 0,
                "all_sources_running": bool(watchers) and running_sources == len(watchers),
                "running_sources": running_sources,
                "total_sources": len(watchers),
                "projects_path": str(settings.projects_path),
                "codex_sessions_path": str(settings.codex_sessions_path),
                "gemini_sessions_path": str(settings.gemini_sessions_path),
                "antigravity_sessions_path": str(settings.antigravity_sessions_path),
                "chatgpt_exports_path": str(settings.chatgpt_exports_path),
                "claude_app_exports_path": str(settings.claude_app_exports_path),
                "debounce_ms": watchers[0].debounce_ms if watchers else settings.debounce_delay,
                "queue_size": queue_size,
                "queue_max_size": sum(source["queue_max_size"] for source in source_stats.values()),
                "failed_files_count": failed_count,
                "sources": source_stats,
            }
        except Exception as e:
            logger.error(f"Failed to get watcher stats: {e}")
            return {"error": _safe_error("Watcher stats failed", e)}

    def _get_error_stats(self) -> dict[str, Any]:
        """Get error statistics."""
        return {
            "total": len(self.errors),
            "recent": [_safe_recent_error(error) for error in self.errors[-10:]],
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
            return {"error": _safe_error("Client registry stats failed", e)}

    def _get_configuration(self) -> dict[str, Any]:
        """Get current configuration."""
        return {
            "db_path": str(settings.db_path),
            "projects_path": str(settings.projects_path),
            "codex_sessions_path": str(settings.codex_sessions_path),
            "gemini_sessions_path": str(settings.gemini_sessions_path),
            "antigravity_sessions_path": str(settings.antigravity_sessions_path),
            "storage_backend": settings.storage_backend,
            "spanner_project": settings.spanner_project,
            "spanner_instance": settings.spanner_instance,
            "spanner_database": settings.spanner_database,
            "spanner_enable_full_text": settings.spanner_enable_full_text,
            "spanner_enable_vector_index": settings.spanner_enable_vector_index,
            "spanner_use_approx_vector_search": settings.spanner_use_approx_vector_search,
            "spanner_vector_index_leaves": settings.spanner_vector_index_leaves,
            "spanner_num_leaves_to_search": settings.spanner_num_leaves_to_search,
            "spanner_hybrid_candidate_limit": settings.spanner_hybrid_candidate_limit,
            "spanner_rrf_k": settings.spanner_rrf_k,
            "spanner_embedding_mode": settings.spanner_embedding_mode,
            "spanner_embedding_model_id": settings.spanner_embedding_model_id,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_url": redact_url(settings.embedding_base_url),
            "embedding_dimension": settings.embedding_dimension,
            "vertex_project": settings.vertex_project,
            "vertex_location": settings.vertex_location,
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
