#!/usr/bin/env python3
"""
Force Database Optimization Script

Triggers the backend-specific optimization routine (e.g., VACUUM for SQLite).
"""
import asyncio
import logging
import sys
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("force_optimize")

try:
    from claude_history_rag.config import settings
    from claude_history_rag.store import store
except ImportError:
    print("Error: Could not import application modules.")
    print("Make sure you are running this from the project root or with src/ in PYTHONPATH.")
    sys.exit(1)


async def force_optimize():
    parser = argparse.ArgumentParser(description="Force storage optimization")
    parser.add_argument("--backend", type=str, help="Override backend (sqlite/qdrant)", default=None)
    args = parser.parse_args()

    if args.backend:
        settings.storage_backend = args.backend

    print(f"Backend: {settings.storage_backend}")
    if settings.storage_backend == "sqlite":
        print(f"Database path: {settings.sqlite_db_path}")

    print("\nStarting optimization...")
    if settings.storage_backend == "sqlite":
        print("Running VACUUM to reclaim space...")
    elif settings.storage_backend == "qdrant":
        print("Triggering Qdrant optimization (if applicable)...")
    
    try:
        await store.initialize()
        await store.optimize_async()
        print("Optimization finished successfully.")
        
        # Print stats
        stats = await store.get_stats_async()
        print(f"\nStats: {stats}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        print(f"\nERROR: Optimization failed: {e}")
        sys.exit(1)
    finally:
        await store.close_async()

def main():
    asyncio.run(force_optimize())

if __name__ == "__main__":
    main()
