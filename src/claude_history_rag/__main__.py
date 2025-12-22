"""Entry point for the MCP server.

This is a lightweight MCP server that provides search tools.
The actual indexing is handled by the daemon (ai-agent-history-rag-daemon).

Two modes are supported:
1. Standalone mode (--standalone): MCP server runs its own watcher/indexer
2. Normal mode (default): MCP server is lightweight, expects daemon to be running
"""

import argparse
import asyncio
import contextlib
import logging
import signal
import sys

from claude_history_rag.config import OPTIMIZE_INTERVAL, settings
from claude_history_rag.decision_engine.cache import get_search_cache
from claude_history_rag.embedder import get_embedder
from claude_history_rag.server import mcp
from claude_history_rag.status_server import create_status_server
from claude_history_rag.store import store
from claude_history_rag.watcher import get_watcher

logger = logging.getLogger(__name__)


async def periodic_optimize(stop_event: asyncio.Event) -> None:
    """Periodically optimize the database."""
    while not stop_event.is_set():
        try:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=OPTIMIZE_INTERVAL)
            if stop_event.is_set():
                break
            logger.info("Running scheduled optimization...")
            await store.optimize_async()
        except Exception:
            logger.exception("Scheduled optimization failed")


async def run_server_standalone():
    """Run MCP server with built-in watcher (standalone mode).

    This is the original behavior where the MCP server handles everything.
    """
    watcher = get_watcher()
    cache = get_search_cache()
    stop_event = asyncio.Event()
    optimize_task: asyncio.Task | None = None
    status_server = None

    # Start status server if enabled
    if settings.status_server_enabled:
        try:
            status_server = create_status_server()
            await status_server.start()
        except Exception as e:
            logger.error(f"Failed to start status server: {e}", exc_info=True)

    # Start watcher
    await watcher.start()

    # Start cache maintenance
    await cache.start_maintenance()

    # Run initial optimization
    try:
        await store.optimize_async()
    except Exception as e:
        logger.error(f"Initial optimization failed: {type(e).__name__}")

    # Start periodic optimization task
    optimize_task = asyncio.create_task(periodic_optimize(stop_event))

    # Set up signal handlers
    loop = asyncio.get_running_loop()

    def handle_signal():
        logger.info("Received shutdown signal")
        stop_event.set()

    signal_handlers_installed = False
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, handle_signal)
            signal_handlers_installed = True
        except NotImplementedError:
            pass

    if not signal_handlers_installed:

        def sync_signal_handler(signum, frame):
            logger.info("Received shutdown signal (sync handler)")
            loop.call_soon_threadsafe(stop_event.set)

        signal.signal(signal.SIGINT, sync_signal_handler)
        signal.signal(signal.SIGTERM, sync_signal_handler)

    try:
        await mcp.run_stdio_async()
    finally:
        stop_event.set()

        if optimize_task:
            try:
                await asyncio.wait_for(optimize_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Optimization task did not finish in time, cancelling")
                optimize_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await optimize_task

        try:
            await asyncio.wait_for(watcher.stop(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning("Watcher stop timed out")

        if status_server:
            try:
                await status_server.stop()
            except Exception as e:
                logger.error(f"Status server stop failed: {type(e).__name__}")

        try:
            await cache.stop_maintenance()
        except Exception as e:
            logger.error(f"Cache maintenance stop failed: {type(e).__name__}")

        try:
            embedder = get_embedder()
            embedder.shutdown()
        except Exception as e:
            logger.error(f"Embedder shutdown failed: {type(e).__name__}")

        try:
            await store.close_async()
        except Exception as e:
            logger.error(f"Store close failed: {type(e).__name__}")


async def run_server_lightweight():
    """Run lightweight MCP server (expects daemon to handle indexing).

    This mode is optimized for quick startup and low resource usage.
    The daemon handles file watching, indexing, and the status server.
    """
    cache = get_search_cache()

    # Start cache maintenance (still needed for query caching)
    await cache.start_maintenance()

    try:
        await mcp.run_stdio_async()
    finally:
        try:
            await cache.stop_maintenance()
        except Exception as e:
            logger.error(f"Cache maintenance stop failed: {type(e).__name__}")

        try:
            embedder = get_embedder()
            embedder.shutdown()
        except Exception as e:
            logger.error(f"Embedder shutdown failed: {type(e).__name__}")

        try:
            await store.close_async()
        except Exception as e:
            logger.error(f"Store close failed: {type(e).__name__}")


def main():
    """Run the MCP server."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Agent History RAG MCP Server")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Run in standalone mode with built-in watcher/indexer (default: lightweight mode)",
    )
    args = parser.parse_args()

    # Set up dual logging: stderr for MCP + file for debugging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console handler (stderr only - NEVER stdout for STDIO MCP servers)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # File handler for detailed debugging (unbuffered for immediate writes)
    log_file = settings.db_path.parent / "claude-history-rag.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_stream = open(log_file, mode="a", buffering=1)  # noqa: SIM115
    file_handler = logging.StreamHandler(file_stream)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    mode = "standalone" if args.standalone else "lightweight"
    logger.info(
        f"Starting AI Agent History RAG MCP server ({mode} mode) | "
        f"db={settings.db_path} | "
        f"projects={settings.projects_path} | "
        f"embedding_model={settings.embedding_model} | "
        f"embedding_url={settings.embedding_base_url} | "
        f"log={log_file}"
    )

    try:
        if args.standalone:
            asyncio.run(run_server_standalone())
        else:
            asyncio.run(run_server_lightweight())
    except KeyboardInterrupt:
        logger.info("Server interrupted")


if __name__ == "__main__":
    main()
