import asyncio
import logging
import os
from pathlib import Path
import sqlite3
# Mock settings before importing module
os.environ["CLAUDE_HISTORY_RAG_SQLITE_DB_PATH"] = str(Path.home() / ".claude-history-rag" / "test_repro.db")

from claude_history_rag.storage.sqlite import SQLiteBackend

logging.basicConfig(level=logging.ERROR)

async def test_repro():
    db_path = Path.home() / ".claude-history-rag" / "test_repro.db"
    if db_path.exists():
        os.remove(db_path)
    
    # Update config path as well
    # We need to rely on the backend reading it from settings, or we manually set it.
    # The environment variable set above might be validated? 
    # Yes, BaseSettings validators run on env vars too.
    # So we need to set the ENV VAR to the absolute path correctly before import OR reload config. 
    # But we already imported. `SQLiteBackend` reads `settings.sqlite_db_path`.
    
    # We can just override the instance variable.
    backend = SQLiteBackend()
    backend.db_path = db_path
    await backend.initialize()
    
    # 1. Valid insert
    try:
        await backend.add_chunks([{
            "id": "1", "content": "valid1", "chunk_type": "text", "vector": [0.1]*768
        }])
        print("Initial insert: OK")
    except Exception as e:
        print(f"Initial insert failed: {e}")
        return

    # 2. INTENTIONAL FAILURE
    # We will try to insert a chunk with missing required fields or causing some constraint violation.
    # But our code handles missing fields by skipping.
    # To force a SQL error, we can try to insert a chunk that violates constraints if we had strict ones, 
    # or we can mock the connection method, OR we can try to insert a string where logic expects something else?
    # Easier: duplicate primary key `id` if unchecked? 
    # Our SQL uses `INSERT OR REPLACE` so duplicates are overwritten.
    
    # 2. TEST FAILURE AND ROLLBACK LOGIC
    # We want to test that if an error occurs within add_chunks, rollback is called.
    # We will use unittest.mock to wrap the connection's execute method.
    from unittest.mock import MagicMock, AsyncMock, patch

    # We need to access the REAL add_chunks method on the backend object.
    # But we want `conn.execute` to fail on the SECOND insert.
    
    real_execute = backend.conn.execute
    
    # We need a robust mock that acts like aiosqlite execute
    class FailingExecute:
        def __init__(self, real_conn):
            self.conn = real_conn
            self.call_count = 0
            
        def __call__(self, sql, *args):
            # Pass through "BEGIN" and "COMMIT" and first insert
            self.call_count += 1
            # 1. BEGIN
            # 2. INSERT 1
            # 3. INSERT 2 -> FAIL
            
            # Simple heuristic checking sql
            if "INSERT" in sql:
                 if "fail_me" in str(args):
                      raise ValueError("Simulated Crash on specific item")
            
            return self.conn.execute(sql, *args)

    # Use patch on the instance's conn.execute? 
    # aiosqlite connection.execute returns a context manager (cursor).
    # It is an async method.
    
    # Easier approach: Just insert a chunk that causes a failure?
    # Trying to insert INVALID JSON into vector column? SQLite might accept it as string.
    # Trying to insert invalid types?
    # Let's try to patch `json.dumps` to fail for a specific item id?
    
    print("Attempting batch with failing item...")
    
    # We will create a special object that fails json serialization
    class Unserializable:
        def to_json(self): raise ValueError("Cannot serialize")
        
    try:
        # Item 1 is valid. Item 2 hasUnseriazliable metadata.
        # json.dumps is called on metadata.
        await backend.add_chunks([
            {"id":"valid_in_batch", "content": "valid", "vector": [0.1]*768},
            {"id":"fail_me", "content": "fail", "vector": [0.1]*768, "bad_meta": Unserializable()}
        ])
    except TypeError:
        # json.dumps raises TypeError for circular/unserializable
        print("Caught expected serialization error")
    except Exception as e:
        print(f"Caught expected error: {e}")

    # 3. VERIFY STATE
    print("Attempting recovery insert...")
    try:
        # If rollback happened, we should be able to insert again.
        await backend.add_chunks([{
            "id": "2", "content": "recovery", "chunk_type": "text", "vector": [0.2]*768
        }])
        print("Recovery insert: OK")
    except Exception as e:
        print(f"Recovery insert FAILED: {e}")
        
    # Check if "valid_in_batch" exists? 
    # Since we failed BEFORE commit (during loop), and rolled back, it should NOT exist.
    async with backend.conn.execute("SELECT id FROM conversation_chunks WHERE id='valid_in_batch'") as cursor:
        row = await cursor.fetchone()
        if row:
            print("Bug Confirmed: Partial data 'valid_in_batch' found (Atomicity isolation violation)")
        else:
            print("Success: Data consistent (valid_in_batch NOT found / rolled back)")

    await backend.close()
    if db_path.exists():
        os.remove(db_path)

if __name__ == "__main__":
    asyncio.run(test_repro())
