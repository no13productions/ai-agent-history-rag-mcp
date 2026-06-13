#!/usr/bin/env python3
"""
Force Database Optimization Script

This script forces an immediate optimization of the LanceDB database to reclaim space.
It uses aggressive cleanup settings (deleting unverified files) to fix bloated databases.
"""

import argparse
import asyncio
import logging
import sys
from datetime import timedelta
from pathlib import Path

# Set up logging before importing app modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("force_optimize")

try:
    # Delay importing store until we know the path
    import lancedb

    from claude_history_rag.config import settings
except ImportError:
    print("Error: Could not import application modules.")
    print("Make sure you are running this from the project root or with src/ in PYTHONPATH.")
    sys.exit(1)


async def force_optimize():
    parser = argparse.ArgumentParser(description="Force LanceDB optimization")
    parser.add_argument("--path", type=str, help="Path to LanceDB directory", default=None)
    parser.add_argument("--hours", type=int, help="Cleanup older than X hours", default=1)
    args = parser.parse_args()

    if args.path:
        db_path = Path(args.path)
        # Monkey patch settings for this run (though we use lancedb directly below)
        settings.db_path = db_path
    else:
        db_path = settings.db_path

    print(f"Database path: {db_path}")

    if not db_path.exists():
        print(f"Database does not exist at {db_path}")
        print("  - If running in Docker, run this script INSIDE the container:")
        print("    docker exec -it ai-agent-history-rag-server uv run scripts/force_optimize.py")
        print("  - Or specify path manually: uv run scripts/force_optimize.py --path /path/to/db")
        return

    print("Checking database size...")
    # Get initial size
    total_size = 0
    for p in db_path.rglob("*"):
        if p.is_file():
            total_size += p.stat().st_size

    print(f"Initial size: {total_size / (1024 * 1024 * 1024):.2f} GB")

    # Configure aggressive settings for this run
    cleanup_seconds = args.hours * 3600
    delete_unverified = True

    print("\nStarting optimization...")
    print(f"  - Cleanup older than: {cleanup_seconds} seconds ({args.hours} hours)")
    print(f"  - Delete unverified: {delete_unverified}")
    print("\nThis may take a few minutes depending on database size...")

    try:
        # Connect to specific path
        db = lancedb.connect(db_path)
        if "conversations" in db.table_names():
            table = db.open_table("conversations")

            # Run optimization
            table.optimize(
                cleanup_older_than=timedelta(seconds=cleanup_seconds),
                delete_unverified=delete_unverified,
            )

            print("Optimization command finished successfully.")

            # Check new size
            new_size = 0
            for p in db_path.rglob("*"):
                if p.is_file():
                    new_size += p.stat().st_size

            print(f"\nNew size: {new_size / (1024 * 1024 * 1024):.2f} GB")
            freed = total_size - new_size
            print(f"Space reclaimed: {freed / (1024 * 1024 * 1024):.2f} GB")
        else:
            print("Table 'conversations' not found in database.")

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        print(f"\nERROR: Optimization failed: {e}")
        sys.exit(1)


def main():
    asyncio.run(force_optimize())


if __name__ == "__main__":
    main()
