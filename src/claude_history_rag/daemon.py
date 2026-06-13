"""Standalone daemon for indexing and status server.

This daemon runs independently of the MCP server and handles:
- File watching and indexing
- Status server (HTTP dashboard)
- Periodic optimization

Supports two modes:
- Server mode: Local embeddings + LanceDB storage + status server
- Client mode: File watching + chunk upload to central server

The MCP server queries the same LanceDB database populated by this daemon.
"""

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import sys
from pathlib import Path

from claude_history_rag.config import OPTIMIZE_INTERVAL, settings
from claude_history_rag.embedder import redact_url
from claude_history_rag.watcher import get_all_watchers

logger = logging.getLogger(__name__)

# PID file location
PID_FILE = settings.db_path.parent / "daemon.pid"


def is_daemon_running() -> tuple[bool, int | None]:
    """Check if daemon is already running.

    Returns:
        Tuple of (is_running, pid). pid is None if not running.
    """
    if not PID_FILE.exists():
        return False, None

    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process exists
        os.kill(pid, 0)
        # If PID matches current process (common in containers), treat as not running
        if pid == os.getpid():
            return False, None
        return True, pid
    except (ValueError, ProcessLookupError, PermissionError):
        # Invalid PID, process doesn't exist, or no permission
        # Clean up stale PID file
        PID_FILE.unlink(missing_ok=True)
        return False, None


def write_pid_file():
    """Write current PID to file."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def remove_pid_file():
    """Remove PID file."""
    PID_FILE.unlink(missing_ok=True)


async def periodic_optimize(stop_event: asyncio.Event) -> None:
    """Periodically optimize the database (server mode only)."""
    from claude_history_rag.store import store

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


async def periodic_backfill(stop_event: asyncio.Event) -> None:
    """Periodically backfill NULL embeddings via partitioned DML (deferred Spanner mode)."""
    from claude_history_rag.store import store

    backfill = getattr(store, "backfill_embeddings_async", None)
    if backfill is None:
        return  # backend has no deferred-embedding backfill (e.g. LanceDB)

    while not stop_event.is_set():
        try:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    stop_event.wait(), timeout=settings.spanner_backfill_interval_seconds
                )
            if stop_event.is_set():
                break
            updated = await backfill()
            if updated:
                logger.info("Embedding backfill updated %s rows", updated)
        except Exception:
            logger.exception("Scheduled embedding backfill failed")


async def run_daemon():
    """Run the daemon with file watcher, status server, and optimization.

    Handles both server mode and client mode.
    """
    watchers = get_all_watchers()
    stop_event = asyncio.Event()
    optimize_task: asyncio.Task | None = None
    backfill_task: asyncio.Task | None = None
    status_server = None
    cache = None

    # Detect mode
    is_server_mode = settings.is_server_mode
    mode_str = "SERVER" if is_server_mode else "CLIENT"

    # Write PID file
    write_pid_file()
    logger.info(f"[{mode_str}] Daemon started with PID {os.getpid()}")

    if is_server_mode:
        # Server mode imports
        from claude_history_rag.decision_engine.cache import get_search_cache
        from claude_history_rag.embedder import get_embedder
        from claude_history_rag.status import get_status_collector
        from claude_history_rag.status_server import create_status_server
        from claude_history_rag.store import store

        # Initialize status collector early so errors can be recorded
        await get_status_collector()
        cache = get_search_cache()
    else:
        # Client mode - log connection info
        logger.info(f"[CLIENT] Server URL: {settings.server_url}")
        logger.info(f"[CLIENT] Machine ID: {settings.machine_id}")

    try:
        # Start status server (server mode only - serves API + dashboard)
        if is_server_mode:
            try:
                from claude_history_rag.status_server import create_status_server

                status_server = create_status_server()
                await status_server.start()
                logger.info(
                    f"[SERVER] Dashboard: http://{settings.status_server_host}:{settings.status_server_port}/dashboard"
                )
                logger.info(
                    f"[SERVER] API: http://{settings.status_server_host}:{settings.status_server_port}/api/"
                )
            except Exception as e:
                logger.error(f"Failed to start status server: {e}", exc_info=True)

        # Start watcher (runs startup sync, then starts background tasks)
        # Works in both modes - client mode uploads, server mode embeds locally
        for history_watcher in watchers:
            await history_watcher.start()

        # Server mode: cache and optimization
        if is_server_mode:
            from claude_history_rag.store import store

            # Start cache maintenance
            await cache.start_maintenance()

            # Run initial optimization after startup sync
            try:
                await store.optimize_async()
            except Exception as e:
                logger.error(f"Initial optimization failed: {type(e).__name__}")

            # Start periodic optimization task
            optimize_task = asyncio.create_task(periodic_optimize(stop_event))

            # Start periodic embedding backfill (deferred Spanner-embedding mode only)
            if settings.storage_backend == "spanner" and settings.spanner_defer_embeddings:
                logger.info(
                    "[SERVER] Deferred embeddings enabled; backfilling every %ss via partitioned DML",
                    settings.spanner_backfill_interval_seconds,
                )
                backfill_task = asyncio.create_task(periodic_backfill(stop_event))

        # Set up signal handlers for graceful shutdown
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

        # Wait for stop signal
        logger.info(f"[{mode_str}] Daemon running. Press Ctrl+C to stop.")
        await stop_event.wait()

    finally:
        logger.info(f"[{mode_str}] Shutting down daemon...")

        # Stop optimize task (server mode only)
        if optimize_task:
            try:
                stop_event.set()
                await asyncio.wait_for(optimize_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Optimization task did not finish in time, cancelling")
                optimize_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await optimize_task

        # Stop embedding backfill task (deferred Spanner mode only)
        if backfill_task:
            try:
                stop_event.set()
                await asyncio.wait_for(backfill_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Embedding backfill task did not finish in time, cancelling")
                backfill_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await backfill_task

        for history_watcher in watchers:
            try:
                await asyncio.wait_for(history_watcher.stop(), timeout=15.0)
            except asyncio.TimeoutError:
                logger.warning("%s watcher stop timed out", history_watcher.source_name)

        # Stop status server (server mode only)
        if status_server:
            try:
                await status_server.stop()
            except Exception as e:
                logger.error(f"Status server stop failed: {type(e).__name__}")

        # Server mode cleanup
        if is_server_mode:
            from claude_history_rag.embedder import get_embedder
            from claude_history_rag.store import store

            # Stop cache maintenance
            if cache:
                try:
                    await cache.stop_maintenance()
                    logger.info("Cache maintenance stopped")
                except Exception as e:
                    logger.error(f"Cache maintenance stop failed: {type(e).__name__}")

            # Shutdown embedder thread pool
            try:
                embedder = get_embedder()
                embedder.shutdown()
                logger.info("Embedder shutdown complete")
            except Exception as e:
                logger.error(f"Embedder shutdown failed: {type(e).__name__}")

            # Close store connections
            try:
                await store.close_async()
                logger.info("Store closed successfully")
            except Exception as e:
                logger.error(f"Store close failed: {type(e).__name__}")

        # Client mode cleanup
        if not is_server_mode:
            from claude_history_rag.api_client import get_api_client

            api_client = get_api_client()
            if api_client:
                try:
                    await api_client.close()
                    logger.info("API client closed")
                except Exception as e:
                    logger.error(f"API client close failed: {type(e).__name__}")

        # Remove PID file
        remove_pid_file()
        logger.info(f"[{mode_str}] Daemon stopped")


def setup_logging(log_file: Path | None = None):
    """Set up logging for the daemon."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # File handler
    if log_file is None:
        log_file = settings.db_path.parent / "daemon.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_stream = open(log_file, mode="a", buffering=1)  # noqa: SIM115
    file_handler = logging.StreamHandler(file_stream)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if settings.log_level == "DEBUG" else logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Reduce noisy watchfiles debug logs
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    logging.getLogger("watchfiles.main").setLevel(logging.WARNING)

    return log_file


def cmd_start(args):
    """Start the daemon."""
    is_running, pid = is_daemon_running()
    if is_running:
        print(f"Daemon is already running (PID {pid})")
        # Return success to avoid restart loops in launchd/systemd
        return 0

    log_file = setup_logging()

    # Log startup info based on mode
    mode = "SERVER" if settings.is_server_mode else "CLIENT"
    if settings.is_server_mode:
        logger.info(
            f"Starting daemon [{mode}] | "
            f"storage_backend={settings.storage_backend} | "
            f"db={settings.db_path} | "
            f"spanner={settings.spanner_project}/{settings.spanner_instance}/"
            f"{settings.spanner_database} | "
            f"projects={settings.projects_path} | "
            f"embedding_provider={settings.embedding_provider} | "
            f"embedding_url={redact_url(settings.embedding_base_url)} | "
            f"embedding_model={settings.embedding_model} | "
            f"embedding_dimension={settings.embedding_dimension} | "
            f"log={log_file}"
        )
    else:
        logger.info(
            f"Starting daemon [{mode}] | "
            f"server_url={settings.server_url} | "
            f"machine_id={settings.machine_id} | "
            f"projects={settings.projects_path} | "
            f"log={log_file}"
        )

    try:
        asyncio.run(run_daemon())
    except KeyboardInterrupt:
        logger.info("Daemon interrupted")

    return 0


def cmd_stop(args):
    """Stop the daemon."""
    is_running, pid = is_daemon_running()
    if not is_running:
        print("Daemon is not running")
        return 1

    print(f"Stopping daemon (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for process to exit
        for _ in range(30):
            try:
                os.kill(pid, 0)
                import time

                time.sleep(0.5)
            except ProcessLookupError:
                print("Daemon stopped")
                return 0
        print("Daemon did not stop in time, sending SIGKILL...")
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        print("Daemon already stopped")
        remove_pid_file()
    except PermissionError:
        print(f"Permission denied to stop process {pid}")
        return 1

    return 0


def cmd_status(args):
    """Check daemon status."""
    is_running, pid = is_daemon_running()
    if is_running:
        print(f"Daemon is running (PID {pid})")
        print(
            f"Dashboard: http://{settings.status_server_host}:{settings.status_server_port}/dashboard"
        )
        return 0
    else:
        print("Daemon is not running")
        return 1


def cmd_restart(args):
    """Restart the daemon."""
    is_running, _ = is_daemon_running()
    if is_running:
        ret = cmd_stop(args)
        if ret != 0:
            return ret
        import time

        time.sleep(1)

    return cmd_start(args)


def main():
    """Main entry point for daemon CLI."""
    parser = argparse.ArgumentParser(
        description="AI Agent History RAG Daemon - Background indexing and status server"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the daemon")
    start_parser.set_defaults(func=cmd_start)

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the daemon")
    stop_parser.set_defaults(func=cmd_stop)

    # Status command
    status_parser = subparsers.add_parser("status", help="Check daemon status")
    status_parser.set_defaults(func=cmd_status)

    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart the daemon")
    restart_parser.set_defaults(func=cmd_restart)

    args = parser.parse_args()

    if args.command is None:
        # Default to start if no command given
        args.func = cmd_start

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
