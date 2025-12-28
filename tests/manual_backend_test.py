
import asyncio
import os
import shutil
from pathlib import Path
from claude_history_rag.config import settings
from claude_history_rag.store import store

async def test_sqlite_backend():
    print("Testing SQLite Backend...")
    # Use temporary DB path
    test_db = Path("test_history.db")
    if test_db.exists():
        test_db.unlink()
    if test_db.parent.exists() and test_db.parent.name == "test_db_dir":
        shutil.rmtree(test_db.parent)

    settings.storage_backend = "sqlite"
    settings.sqlite_db_path = test_db
    
    # Initialize
    await store.initialize()
    
    # Add chunks
    chunks = [
        {
            "id": "1234567890abcdef", 
            "content": "Hello world test",
            "vector": [0.1] * 768, # dim 768
            "chunk_type": "text",
            "session_id": "sess1",
            "project_path": "/tmp",
            "project_name": "test",
            "file_path": "test.txt",
            "machine_id": "machine1",
            "timestamp": "2023-01-01T00:00:00"
        },
        {
            "id": "abcdef1234567890", 
            "content": "Another chunk here",
            "vector": [0.2] * 768,
            "chunk_type": "text",
            "session_id": "sess1",
            "project_path": "/tmp",
            "project_name": "test",
            "file_path": "other.txt",
            "machine_id": "machine1",
            "timestamp": "2023-01-01T00:01:00"
        }
    ]
    
    await store.add_chunks_async(chunks)
    print("Chunks added.")
    
    # Verify stats
    stats = await store.get_stats_async()
    print(f"Stats: {stats}")
    assert stats["total_chunks"] == 2
    
    # Search
    results = await store.search_async([0.1]*768, limit=1)
    print(f"Search Results: {len(results)}")
    assert len(results) == 1
    assert results[0]["id"] == "1234567890abcdef"
    print("Search verified.")
    
    # Filter search
    results = await store.search_async([0.1]*768, limit=10, file_path_filter="other.txt")
    print(f"Filtered Results: {len(results)}")
    assert len(results) == 1
    assert results[0]["id"] == "abcdef1234567890"
    print("Filter search verified.")
    
    # Optimize
    await store.optimize_async()
    print("Optimize ran.")
    
    await store.close_async()
    
    # Cleanup
    if test_db.exists():
        test_db.unlink()
    print("SQLite Test Passed!")

def main():
    asyncio.run(test_sqlite_backend())

if __name__ == "__main__":
    main()
