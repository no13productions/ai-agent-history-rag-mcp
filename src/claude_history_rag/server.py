"""FastMCP server with RAG tools.

Supports two modes:
- Server mode: Direct local operations (embeddings + LanceDB)
- Client mode: Proxy requests to central server via API
"""

import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from claude_history_rag.config import settings
from claude_history_rag.errors import record_error
from claude_history_rag.time_filters import format_time_filter, parse_timeframe
from claude_history_rag.watcher import get_all_watchers

logger = logging.getLogger(__name__)


def _local_status_auth_headers() -> dict[str, str]:
    """Authorize the in-process snapshot call to the always-on daemon status server.

    When auth is enabled the daemon's ``/status`` endpoint requires a Bearer PSK,
    so an unauthenticated call gets 401 and the snapshot silently falls back to this
    (idle) MCP process's own watcher state -- which is why watcher_running read False.
    The MCP server and daemon share the same ``auth_state`` file, so the active key
    matches; read it straight from the file rather than via the auth manager to avoid
    a stray write to the daemon-owned ``auth.json``.
    """
    if not settings.auth_enabled:
        return {}
    psk = settings.server_psk
    if not psk:
        try:
            import json

            data = json.loads(settings.auth_state_path.read_text())
            psk = (data.get("active") or {}).get("key_plain")
        except Exception as e:
            logger.debug("Local status PSK unavailable: %s", type(e).__name__)
            return {}
    if not psk:
        return {}
    return {"Authorization": f"Bearer {psk}"}


async def _get_status_server_snapshot(detail_level: str = "full") -> dict[str, Any] | None:
    """Fetch the always-on daemon status server when this MCP process is lightweight."""
    if not settings.status_server_enabled:
        return None

    host = settings.status_server_host
    if host in {"", "0.0.0.0", "::"}:
        host = "127.0.0.1"

    try:
        import httpx

        timeout = httpx.Timeout(30.0, connect=1.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                f"http://{host}:{settings.status_server_port}/status",
                params={"detail": detail_level},
                headers=_local_status_auth_headers(),
            )
        if response.status_code != 200:
            return None
        data = response.json()
        if not isinstance(data, dict):
            return None
        return data
    except Exception as e:
        logger.debug("Daemon status snapshot unavailable: %s", type(e).__name__)
        return None


def _is_external_daemon_snapshot(snapshot: dict[str, Any]) -> bool:
    server = snapshot.get("server")
    if not isinstance(server, dict):
        return False
    return server.get("pid") not in {None, os.getpid()}


def _index_status_from_daemon_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    indexing = snapshot.get("indexing") if isinstance(snapshot.get("indexing"), dict) else {}
    file_watcher = (
        snapshot.get("file_watcher") if isinstance(snapshot.get("file_watcher"), dict) else {}
    )
    database = snapshot.get("database") if isinstance(snapshot.get("database"), dict) else {}
    embedder = snapshot.get("embedder") if isinstance(snapshot.get("embedder"), dict) else {}
    sources = indexing.get("sources") if isinstance(indexing.get("sources"), dict) else {}

    source_status = {
        name: {
            "watched_files": source.get("files_indexed", source.get("watched_files", 0)),
            "pending_files": source.get("files_pending", source.get("pending_files", 0)),
            "running": source.get("is_running", source.get("running", False)),
            "failed_files": source.get("files_failed", source.get("failed_files", 0)),
            "watch_path": source.get("watch_path"),
        }
        for name, source in sources.items()
        if isinstance(source, dict)
    }

    return {
        "mode": "server",
        "source": "daemon",
        "daemon_pid": snapshot.get("server", {}).get("pid"),
        "total_chunks": database.get("total_chunks", 0),
        "embedding_model_loaded": bool(embedder.get("loaded", True)),
        "watched_files": sum(source["watched_files"] for source in source_status.values()),
        "pending_files": sum(source["pending_files"] for source in source_status.values()),
        "sources": source_status,
        "watcher_running": bool(file_watcher.get("all_sources_running")),
        "status": snapshot.get("health", {}).get("status", snapshot.get("status", "unknown")),
        "cache_stats": snapshot.get("cache"),
        "storage_backend": database.get("backend", settings.storage_backend),
        "database": {
            key: value
            for key, value in database.items()
            if key
            in {
                "db_path",
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
                "awaiting_embedding",
                "backfill_rate_per_min",
                "backfill_eta_seconds",
            }
        },
    }


def _get_decision_engine():
    """Get decision engine with thread-safe lazy initialization.

    I1 fix: Uses the centralized get_decision_engine() from engine.py
    to avoid dual global instances. Thread safety is handled there.
    """
    # Lazy import to avoid circular imports
    from claude_history_rag.decision_engine import get_decision_engine

    return get_decision_engine()


def expand_file_query(file_path: str) -> list[str]:
    """Expand file path into multiple search terms for better matching.

    This implements simple query expansion without requiring an LLM.
    Validates input to prevent path traversal attacks.

    Args:
        file_path: File path to expand into search terms

    Returns:
        List of search terms derived from the file path (full path, basename,
        dirname, name without extension). Returns empty list if input is
        invalid or contains path traversal attempts.
    """
    # Validate input
    if not file_path or not file_path.strip():
        return []

    # Normalize path to resolve .. and detect traversal attempts
    normalized = os.path.normpath(file_path)

    # Reject if path tries to traverse (starts with .. after normalization)
    if normalized.startswith("..") or ".." in file_path:
        logger.warning(
            f"Path traversal attempt blocked: {file_path}",
            extra={"event": "security", "file_path": file_path},
        )
        return []

    terms = [file_path]
    base = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)

    if base:
        terms.append(base)
    if dir_name and dir_name != "/":  # Skip root directory as search term
        terms.append(dir_name)

    # Add without extension
    name_without_ext = os.path.splitext(base)[0]
    if name_without_ext and name_without_ext != base:
        terms.append(name_without_ext)

    # Remove duplicates while preserving order
    result = []
    seen: set[str] = set()
    for t in terms:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


mcp = FastMCP("ai-agent-history-rag")


async def _execute_search(
    query: str,
    query_vector: list[float],
    limit: int,
    project_filter: str | None,
) -> list[dict[str, Any]]:
    """Execute hybrid search - wrapper for decision engine."""
    from claude_history_rag.store import store

    return await store.hybrid_search_async(
        query=query,
        query_vector=query_vector,
        limit=limit,
        project_filter=project_filter,
    )


def _consume_search_type_marker(results: list[dict[str, Any]], default: str) -> str:
    """Remove internal backend search metadata and return the effective search type."""
    effective = default
    for result in results:
        marker = result.pop("_search_type", None)
        if marker in {"hybrid", "vector"}:
            effective = marker
    return effective


async def _embed_query(query: str) -> list[float]:
    """Embed query - wrapper for decision engine."""
    if settings.storage_backend == "spanner" and settings.spanner_embedding_mode == "spanner":
        from claude_history_rag.store import store

        if hasattr(store, "embed_query_text_async"):
            return await store.embed_query_text_async(query)

    from claude_history_rag.embedder import get_embedder

    embedder = get_embedder()
    return await embedder.embed_query(query)


@mcp.tool()
async def search_conversations(
    query: str,
    project_filter: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 5,
    use_hybrid: bool = True,
    enable_analysis: bool = True,
    enable_synthesis: bool = False,
    include_debug: bool = False,
) -> dict:
    """Search conversation history for relevant context.

    Use this to find:
    - Previous discussions about a topic
    - Decisions made in earlier sessions
    - Context that was compacted away

    Args:
        query: Natural language query
        project_filter: Limit to specific project path
        date_from: Inclusive lower timestamp bound. Accepts ISO-8601 datetime
            or date-only values such as 2026-06-13.
        date_to: Inclusive upper timestamp bound. Accepts ISO-8601 datetime
            or date-only values such as 2026-06-15.
        limit: Maximum results (default 5, min 1, max 50)
        use_hybrid: Use hybrid search (vector + BM25) for better results
            (default True)
        enable_analysis: Enable query analysis and result evaluation for
            improved relevance (default True). Adds 'analysis' and 'evaluation'
            to response.
        enable_synthesis: Enable result synthesis to combine multiple results
            into a coherent summary (default False). Adds 'synthesis' to response
            with key_points and deduplicated content.
        include_debug: Include detailed timing metrics and decision tracking
            in response (default False). Useful for debugging and performance
            analysis. Adds 'metrics' to response.

    Returns:
        Dict with results list and metadata. When enable_analysis=True, includes:
        - analysis: Query intent, detected technologies, key terms
        - evaluation: Relevance score, completeness assessment
        When enable_synthesis=True, includes:
        - synthesis: Primary content, key points, code snippets
        When include_debug=True, includes:
        - metrics: Timing data (query_analysis_ms, search_ms, etc.), decisions made
    """
    try:
        logger.debug(f"search_conversations: limit={limit}, analysis={enable_analysis}")

        if not query or not query.strip():
            logger.warning("Validation failure: empty query")
            return {"error": "Query cannot be empty", "results": []}

        # Limit query length to prevent memory issues
        if len(query) > 10000:
            logger.warning(f"Validation failure: query too long ({len(query)} chars)")
            return {"error": "Query too long (max 10000 chars)", "results": []}

        # Validate limit is an integer within bounds
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            logger.warning(f"Validation failure: invalid limit type ({type(limit).__name__})")
            return {
                "error": "Invalid limit: must be an integer",
                "results": [],
            }

        if limit < 1 or limit > 50:
            logger.warning(f"Validation failure: limit out of bounds ({limit})")
            return {
                "error": "Invalid limit: must be between 1 and 50",
                "results": [],
            }
        try:
            parsed_date_from, parsed_date_to = parse_timeframe(date_from, date_to)
        except ValueError as e:
            logger.warning("Validation failure: %s", e)
            return {"error": str(e), "results": []}
        normalized_date_from = format_time_filter(parsed_date_from)
        normalized_date_to = format_time_filter(parsed_date_to)

        # ============================================================
        # Client Mode: Proxy to central server
        # ============================================================
        if settings.is_client_mode:
            from claude_history_rag.api_client import ServerConnectionError, get_api_client

            api_client = get_api_client()
            if not api_client:
                return {"error": "Client mode not configured", "results": []}

            try:
                response = await api_client.search(
                    query=query,
                    limit=limit,
                    project_filter=project_filter,
                    date_from=normalized_date_from,
                    date_to=normalized_date_to,
                    use_hybrid=use_hybrid,
                    enable_analysis=enable_analysis,
                    enable_synthesis=enable_synthesis,
                    include_debug=include_debug,
                )
                return response.model_dump()
            except ServerConnectionError as e:
                logger.error(f"Server unavailable: {e}")
                return {"error": "Central server unavailable", "results": []}

        # ============================================================
        # Server Mode: Local processing
        # ============================================================
        from claude_history_rag.store import store

        effective_use_hybrid = use_hybrid and store.has_fts_index()
        search_type = "hybrid" if effective_use_hybrid else "vector"

        # Use decision engine for enhanced search
        if enable_analysis or enable_synthesis:
            engine = _get_decision_engine()

            async def execute_effective_search(
                search_query: str,
                search_vector: list[float],
                search_limit: int,
                search_project_filter: str | None,
            ) -> list[dict[str, Any]]:
                if effective_use_hybrid:
                    return await store.hybrid_search_async(
                        query=search_query,
                        query_vector=search_vector,
                        limit=search_limit,
                        project_filter=search_project_filter,
                        date_from=parsed_date_from,
                        date_to=parsed_date_to,
                    )
                return await store.search_async(
                    query_vector=search_vector,
                    limit=search_limit,
                    project_filter=search_project_filter,
                    date_from=parsed_date_from,
                    date_to=parsed_date_to,
                )

            # Pass enable_synthesis as parameter instead of mutating global state (T2 fix)
            result = await engine.search(
                query=query,
                search_func=execute_effective_search,
                embed_func=_embed_query,
                limit=limit,
                project_filter=project_filter,
                search_type=search_type,
                enable_synthesis=enable_synthesis,
                include_debug=include_debug,
            )
            if isinstance(result.get("results"), list):
                result["search_type"] = _consume_search_type_marker(
                    result["results"], result.get("search_type", search_type)
                )
            result["date_from"] = normalized_date_from
            result["date_to"] = normalized_date_to

            return result

        # Fall back to basic search if analysis disabled
        query_vector = await _embed_query(query)

        if effective_use_hybrid:
            # Use hybrid search with RRF reranking for better results
            results = await store.hybrid_search_async(
                query=query,
                query_vector=query_vector,
                limit=limit,
                project_filter=project_filter,
                date_from=parsed_date_from,
                date_to=parsed_date_to,
            )
        else:
            # Fall back to vector-only search
            results = await store.search_async(
                query_vector=query_vector,
                limit=limit,
                project_filter=project_filter,
                date_from=parsed_date_from,
                date_to=parsed_date_to,
            )

        return {
            "results": results,
            "count": len(results),
            "query": query,
            "search_type": _consume_search_type_marker(results, search_type),
            "cache_hit": False,
            "date_from": normalized_date_from,
            "date_to": normalized_date_to,
        }

    except FileNotFoundError:
        return {
            "error": "Index not initialized. Please wait for indexing.",
            "results": [],
        }
    except Exception as e:
        logger.exception(f"Search failed: {type(e).__name__} - {str(e)}")
        record_error(
            "search",
            f"Search failed: {type(e).__name__}",
            {"query_length": len(query), "error_type": type(e).__name__},
        )
        return {
            "error": f"Search error: {type(e).__name__}",
            "results": [],
        }


@mcp.tool()
async def search_file_changes(
    file_path: str | None = None,
    query: str | None = None,
    project_filter: str | None = None,
    operation_filter: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 10,
) -> dict:
    """Find file modifications in conversation history.

    Use this when user asks:
    - "What did we change in auth.dart?"
    - "Show me recent edits to the config files"
    - "What files did we create?"

    Args:
        file_path: Filter by file path (supports partial match)
        query: Semantic query about changes
        project_filter: Limit to specific project
        operation_filter: Filter by "edit" or "write"
        date_from: Inclusive lower timestamp bound. Accepts ISO-8601 datetime
            or date-only values such as 2026-06-13.
        date_to: Inclusive upper timestamp bound. Accepts ISO-8601 datetime
            or date-only values such as 2026-06-15.
        limit: Maximum results (default 10, min 1, max 50)

    Returns:
        Dict with file change results
    """
    try:
        logger.debug(f"search_file_changes: file_path={file_path}, operation={operation_filter}")

        # Validate query is not empty/whitespace if provided
        if query is not None and not query.strip():
            logger.warning("Validation failure: empty or whitespace query")
            return {
                "error": "Query cannot be empty or whitespace",
                "results": [],
            }

        # Validate query length to prevent resource exhaustion
        if query and len(query) > 10000:
            logger.warning(f"Validation failure: query too long ({len(query)} chars)")
            return {"error": "Query too long (max 10000 chars)", "results": []}

        # Validate limit is an integer within bounds
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            logger.warning(f"Validation failure: invalid limit type ({type(limit).__name__})")
            return {
                "error": "Invalid limit: must be an integer",
                "results": [],
            }

        if limit < 1 or limit > 50:
            logger.warning(f"Validation failure: limit out of bounds ({limit})")
            return {
                "error": "Invalid limit: must be between 1 and 50",
                "results": [],
            }

        # Validate operation_filter
        valid_operations = {"edit", "write"}
        if operation_filter and operation_filter not in valid_operations:
            logger.warning(f"Validation failure: invalid operation_filter '{operation_filter}'")
            return {
                "error": (
                    f"Invalid operation_filter '{operation_filter}': must be 'edit' or 'write'"
                ),
                "results": [],
            }
        try:
            parsed_date_from, parsed_date_to = parse_timeframe(date_from, date_to)
        except ValueError as e:
            logger.warning("Validation failure: %s", e)
            return {"error": str(e), "results": []}
        normalized_date_from = format_time_filter(parsed_date_from)
        normalized_date_to = format_time_filter(parsed_date_to)

        # ============================================================
        # Client Mode: Proxy to central server
        # ============================================================
        if settings.is_client_mode:
            from claude_history_rag.api_client import ServerConnectionError, get_api_client

            api_client = get_api_client()
            if not api_client:
                return {"error": "Client mode not configured", "results": []}

            try:
                response = await api_client.search_files(
                    file_path=file_path,
                    query=query,
                    project_filter=project_filter,
                    operation_filter=operation_filter,
                    date_from=normalized_date_from,
                    date_to=normalized_date_to,
                    limit=limit,
                )
                return response.model_dump()
            except ServerConnectionError as e:
                logger.error(f"Server unavailable: {e}")
                return {"error": "Central server unavailable", "results": []}

        # ============================================================
        # Server Mode: Local processing
        # ============================================================
        from claude_history_rag.store import store

        # Build search query with expansion if file_path provided
        if query:
            search_text = query
        elif file_path:
            # Use query expansion for better file matching
            expanded_terms = expand_file_query(file_path)
            if not expanded_terms:
                # File path was invalid (empty, traversal attempt, etc.)
                return {
                    "error": f"Invalid file path: {file_path}",
                    "results": [],
                }
            search_text = " ".join(expanded_terms)
        else:
            search_text = "file changes modifications edits"

        query_vector = await _embed_query(search_text)

        results = await store.search_async(
            query_vector=query_vector,
            limit=limit,
            project_filter=project_filter,
            chunk_type_filter="file_change",
            file_path_filter=file_path,
            operation_filter=operation_filter,
            date_from=parsed_date_from,
            date_to=parsed_date_to,
        )

        return {
            "results": results,
            "count": len(results),
            "file_path_filter": file_path,
            "operation_filter": operation_filter,
            "date_from": normalized_date_from,
            "date_to": normalized_date_to,
        }

    except FileNotFoundError:
        return {
            "error": "Index not initialized. Please wait for indexing.",
            "results": [],
        }
    except Exception as e:
        logger.exception(f"File search failed: {type(e).__name__}")
        record_error(
            "search",
            f"File search failed: {type(e).__name__}",
            {
                "has_file_path_filter": bool(file_path),
                "error_type": type(e).__name__,
            },
        )
        return {
            "error": f"File search error: {type(e).__name__}",
            "results": [],
        }


@mcp.tool()
async def get_session_summary(
    session_id: str | None = None,
    project_filter: str | None = None,
    count: int = 1,
) -> dict:
    """Get summary of conversation session(s).

    Use for:
    - "What did we work on in the last session?"
    - "Summarize our recent conversations"

    Args:
        session_id: Specific session ID, or None for recent
        project_filter: Limit to specific project
        count: Number of sessions to summarize

    Returns:
        Dict with session summaries
    """
    try:
        logger.debug(f"get_session_summary: session_id={session_id}, count={count}")

        # Ensure count is an integer
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = 1
        count = min(max(count, 1), 20)

        # ============================================================
        # Client Mode: Proxy to central server
        # ============================================================
        if settings.is_client_mode:
            from claude_history_rag.api_client import ServerConnectionError, get_api_client

            api_client = get_api_client()
            if not api_client:
                return {"error": "Client mode not configured", "summaries": []}

            try:
                response = await api_client.get_session_summary(
                    session_id=session_id,
                    project_filter=project_filter,
                    count=count,
                )
                return response.model_dump()
            except ServerConnectionError as e:
                logger.error(f"Server unavailable: {e}")
                return {"error": "Central server unavailable", "summaries": []}

        # ============================================================
        # Server Mode: Local processing
        # ============================================================
        from claude_history_rag.store import store

        query_vector = await _embed_query("session summary overview")

        # If filtering by session_id, fetch more results to ensure we get
        # enough after filtering. Use higher multiplier for sparse
        # session_id matches
        fetch_limit = count * 50 if session_id is not None else count * 3

        results = await store.search_async(
            query_vector=query_vector,
            limit=fetch_limit,
            project_filter=project_filter,
            chunk_type_filter="summary",
        )

        # Filter by session_id if specified
        if session_id is not None:
            session_id_str = str(session_id)
            results = [r for r in results if r.get("session_id") == session_id_str]
            # Warn if we didn't get enough results, regardless of fetch_limit saturation
            if len(results) < count:
                logger.warning(
                    f"Session filter returned {len(results)} summaries, "
                    f"requested {count}. Session may not exist or have fewer summaries."
                )

        return {
            "summaries": results[:count],
            "count": len(results[:count]),
        }

    except FileNotFoundError:
        return {
            "error": "Index not initialized. Please wait for indexing.",
            "summaries": [],
        }
    except Exception as e:
        logger.exception(f"Session summary failed: {type(e).__name__}")
        return {
            "error": f"Session summary error: {type(e).__name__}",
            "summaries": [],
        }


@mcp.tool()
async def get_index_status() -> dict:
    """Get status of the RAG index.

    Use when user asks about memory system health or
    why something isn't being found.

    Returns:
        Dict with index statistics including:
        - total_chunks: Number of indexed chunks
        - projects_indexed: Number of unique projects
        - watched_files: Number of files being tracked
        - pending_files: Number of files in queue for processing
        - status: Overall health status
        - cache_stats: Search cache statistics (if enabled)
    """
    try:
        logger.debug("get_index_status called")
        daemon_snapshot = await _get_status_server_snapshot(detail_level="full")
        if daemon_snapshot and _is_external_daemon_snapshot(daemon_snapshot):
            return _index_status_from_daemon_snapshot(daemon_snapshot)

        watchers = get_all_watchers()
        source_status = {
            watcher.source_name: {
                "watched_files": len(watcher.state.get_all_files()),
                "pending_files": watcher.queue.qsize(),
                "running": watcher.is_running,
                "failed_files": watcher.failed_files_count,
                "watch_path": str(watcher.projects_path),
            }
            for watcher in watchers
        }
        watched_files = sum(source["watched_files"] for source in source_status.values())
        pending_files = sum(source["pending_files"] for source in source_status.values())
        watcher_running = all(watcher.is_running for watcher in watchers)

        # ============================================================
        # Client Mode: Get status from central server
        # ============================================================
        if settings.is_client_mode:
            from claude_history_rag.api_client import ServerConnectionError, get_api_client
            from claude_history_rag.client_state import get_client_state_manager

            api_client = get_api_client()
            state_manager = get_client_state_manager()

            # Get local client state
            client_state = await state_manager.get_state()
            pending_uploads = len(client_state.pending_uploads)

            status = {
                "mode": "client",
                "server_url": settings.server_url,
                "machine_id": settings.machine_id,
                "watched_files": watched_files,
                "pending_files": pending_files,
                "sources": source_status,
                "pending_uploads": pending_uploads,
                "watcher_running": watcher_running,
                "connected": client_state.connected,
            }

            # Try to get server status
            if api_client:
                try:
                    server_status = await api_client.get_index_status()
                    status["server_status"] = server_status
                    status["status"] = "healthy" if client_state.connected else "degraded"
                except ServerConnectionError:
                    status["server_status"] = {"error": "Server unavailable"}
                    status["status"] = "degraded"

            return status

        # ============================================================
        # Server Mode: Local status
        # ============================================================
        from claude_history_rag.store import store

        stats = await store.get_stats_async()
        embedding_model_loaded = True
        if not (
            settings.storage_backend == "spanner" and settings.spanner_embedding_mode == "spanner"
        ):
            from claude_history_rag.embedder import get_embedder

            embedding_model_loaded = get_embedder().is_initialized

        # Get cache stats if decision engine is initialized (I1 fix: check via module)
        cache_stats = None
        try:
            from claude_history_rag.decision_engine import engine as de_module

            if de_module._global_engine is not None:
                cache_stats = await de_module._global_engine.get_cache_stats()
        except ImportError:
            pass

        return {
            "mode": "server",
            "total_chunks": stats["total_chunks"],
            "embedding_model_loaded": embedding_model_loaded,
            "watched_files": watched_files,
            "pending_files": pending_files,
            "sources": source_status,
            "watcher_running": watcher_running,
            "status": "healthy",
            "cache_stats": cache_stats,
            "storage_backend": stats.get("backend", settings.storage_backend),
            "database": {
                key: value
                for key, value in stats.items()
                if key
                in {
                    "db_path",
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
                }
            },
        }

    except Exception as e:
        logger.exception(f"Status check failed: {type(e).__name__}")
        return {
            "status": "error",
            "error": type(e).__name__,
        }


@mcp.tool()
async def get_server_status(detail_level: str = "basic") -> dict:
    """Get comprehensive MCP server status and health information.

    Use when you need to check server health, performance metrics,
    indexing progress, or debug issues with the memory system.

    Args:
        detail_level: "basic" for summary info, "full" for detailed metrics
                     including performance, cache stats, and errors

    Returns:
        Dict with comprehensive server status including:
        - server: Version, uptime, PID, platform info
        - health: Overall status (healthy/degraded/unhealthy) and component checks
        - database: Chunk counts, size (full detail only)
        - indexing: Progress, files pending/indexed/failed (full detail only)
        - performance: Memory, CPU, query metrics (full detail only)
        - cache: Hit rates, size (full detail only)
        - embedder: Model info, loaded status (full detail only)
        - file_watcher: Running status, queue info (full detail only)
        - errors: Recent errors and counts (full detail only)
        - configuration: Current settings (full detail only)
    """
    try:
        daemon_snapshot = await _get_status_server_snapshot(detail_level=detail_level)
        if daemon_snapshot and _is_external_daemon_snapshot(daemon_snapshot):
            daemon_snapshot["source"] = "daemon"
            return daemon_snapshot

        # Import here to avoid circular dependency
        from claude_history_rag.status import get_status_collector

        # Validate detail level
        if detail_level not in ["basic", "full"]:
            detail_level = "basic"

        collector = await get_status_collector()
        status = await collector.collect_status(detail_level=detail_level)

        return status

    except Exception as e:
        logger.exception(f"Failed to get server status: {type(e).__name__}")
        return {
            "error": "Failed to collect status",
            "error_type": type(e).__name__,
            "message": f"Status collection failed: {type(e).__name__}",
        }
