"""SQLite storage backend implementation using sqlite-vec."""

import json
import logging
import asyncio
from typing import Any
import sqlite3
import aiosqlite
import sqlite_vec

from claude_history_rag.config import settings
from claude_history_rag.store import get_current_vector_dim
from claude_history_rag.storage.protocol import StorageBackend

logger = logging.getLogger(__name__)


class SQLiteBackend(StorageBackend):
    """SQLite storage backend with vector search."""

    def __init__(self) -> None:
        """Initialize SQLite backend."""
        self.db_path = settings.sqlite_db_path
        self._vector_dim: int | None = None
        # Connection is managed per-call or persistent?
        # aiosqlite connection context manager is preferred, but for a backend object 
        # we might want to keep it open? Protocol says initialize/close.
        self.conn: aiosqlite.Connection | None = None
        self._init_lock = asyncio.Lock()

    @property
    def vector_dim(self) -> int:
        """Get vector dimension."""
        if self._vector_dim is None:
            self._vector_dim = get_current_vector_dim()
        return self._vector_dim

    async def initialize(self) -> None:
        """Initialize the backend connection and schema."""
        if self.conn:
             return

        async with self._init_lock:
             if self.conn:
                 return

             self.db_path.parent.mkdir(parents=True, exist_ok=True)
             
             self.conn = await aiosqlite.connect(self.db_path)
        
        # Load sqlite-vec extension
        await self.conn.enable_load_extension(True)
        await self.conn.load_extension(sqlite_vec.loadable_path())
        
        # Create tables
        # Metadata table
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_chunks (
                id TEXT PRIMARY KEY,
                content TEXT,
                chunk_type TEXT,
                session_id TEXT,
                project_path TEXT,
                project_name TEXT,
                file_path TEXT,
                machine_id TEXT,
                timestamp TEXT,
                metadata JSON
            )
        """)
        
        # Vector table
        # vec0 virtual table
        # We link via rowid. 
        # But wait, conversation_chunks uses TEXT PK. rowid is implicit.
        # We need to map `id` <-> `rowid`? 
        # Or store `id` in vector table? vec0 doesn't support auxiliary columns easily in all versions.
        # Standard pattern: 
        #   vec_table(rowid, vector)
        #   meta_table(rowid, metadata...)
        # We will assume rowid alignment or just simple integer PK in meta table.
        # Let's add an explicit INTEGER PRIMARY KEY to meta table to be safe and use that as the join key.
        # But `id` (the hex hash) is the external ID.
        # So: conversation_chunks (rowid INTEGER PRIMARY KEY, id TEXT UNIQUE, ...)
        
        # NOTE: sqlite-vec uses `rowid` for the vector table.
        
        await self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                content_vector float[{self.vector_dim}]
            )
        """)
        
        await self.conn.commit()
        logger.info(f"Connected to SQLite at {self.db_path}")

    async def close(self) -> None:
        """Close connections."""
        if self.conn:
            await self.conn.close()
            self.conn = None

    async def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Add chunks to storage."""
        if not self.conn:
            await self.initialize()
            
        try:
            await self.conn.execute("BEGIN")
            for chunk in chunks:
                chunk_id = chunk.get("id")
                vector = chunk.get("vector")
                if not chunk_id or not vector:
                    continue

                # Prepare metadata
                # Extract known fields, put rest in metadata JSON
                known_fields = {
                    "id", "content", "chunk_type", "session_id", 
                    "project_path", "project_name", "file_path", 
                    "machine_id", "timestamp"
                }
                meta_extra = {k: v for k, v in chunk.items() if k not in known_fields and k != "vector"}
                
                # Insert/Replace into main table
                # We use INSERT OR REPLACE to handle upserts
                # But if we replace, rowid changes? Yes.
                # So we must update vector table too with new rowid.
                
                # 1. UPSERT meta table
                # We need to retrieve rowid after insert
                cursor = await self.conn.execute("""
                    INSERT OR REPLACE INTO conversation_chunks 
                    (id, content, chunk_type, session_id, project_path, project_name, file_path, machine_id, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk_id,
                    chunk.get("content"),
                    chunk.get("chunk_type"),
                    chunk.get("session_id"),
                    chunk.get("project_path"),
                    chunk.get("project_name"),
                    chunk.get("file_path"),
                    chunk.get("machine_id"),
                    chunk.get("timestamp") if isinstance(chunk.get("timestamp"), str) else str(chunk.get("timestamp")),
                    json.dumps(meta_extra)
                ))
                
                row_id = cursor.lastrowid
                
                # 2. Insert into vector table using specific rowid
                # delete old if exists? "INSERT OR REPLACE" on vec0 works?
                await self.conn.execute("""
                    INSERT OR REPLACE INTO vec_chunks(rowid, content_vector)
                    VALUES (?, ?)
                """, (row_id, json.dumps(vector))) # sqlite-vec takes JSON float array or binary
                
            await self.conn.commit()
            logger.info(f"Added {len(chunks)} chunks to SQLite")
        except Exception as e:
            logger.error(f"Failed to add chunks to SQLite (rolling back): {e}")
            if self.conn:
                await self.conn.rollback()
            raise

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks."""
        if not self.conn:
            await self.initialize()

        # Build query
        # sqlite-vec search:
        # SELECT rowid, distance FROM vec_chunks WHERE content_vector MATCH ? ORDER BY distance LIMIT ?
        # Join with meta table
        
        where_clauses = []
        params = []
        
        # Handle filters
        if filters:
            for key, value in filters.items():
                if key == "file_path" and "%" in str(value):
                    where_clauses.append(f"c.file_path LIKE ?")
                    # unescape python-side escape sequences if needed, but SQL param passing handles quotes
                    # Wait, our filter value contains SQL wildcards (%) already
                    params.append(value)
                elif key in ["project_path", "chunk_type", "machine_id", "session_id", "file_path", "project_name"]:
                    where_clauses.append(f"c.{key} = ?")
                    params.append(value)
        
        where_sql = "AND ".join(where_clauses)
        if where_sql:
            where_sql = "AND " + where_sql
            
        # We need to supply vector param. sqlite-vec expects JSON string or blob?
        # Typically binding bytes or string works.
        
        # Subquery or Join?
        # Efficient vector search requires the vector scan to happen.
        # "knn_search" style:
        # SELECT ... FROM vec_chunks v JOIN conversation_chunks c ON v.rowid = c.rowid
        # WHERE v.content_vector MATCH ? AND {filters} ...
        # But `sqlite-vec` MATCH constraint might not support arbitrary ANDs well if not pushed down?
        # Actually `vec0` supports pre-filtering if using `k = ?` parameter or `limit`.
        # For simple integration: 
        # SELECT c.*, vec_distance_cosine(v.content_vector, ?) as score
        # FROM vec_chunks v
        # JOIN conversation_chunks c ON v.rowid = c.rowid
        # WHERE ...
        # ORDER BY score ASC LIMIT ?
        
        # Wait, naive scan is ok for small DB, but we want index usage.
        # sqlite-vec MATCH syntax:
        # rowid IN (SELECT rowid FROM vec_chunks WHERE content_vector MATCH ? k=?)
        
        # Complex filtered search in sqlite-vec is tricky.
        # Simplest approach for now: Brute force scan with `vec_distance_cosine` is surprisingly fast for <100k rows.
        # Let's try explicit distance calculation and sort, filtered by metadata first.
        
        sql = f"""
            SELECT 
                c.id, c.content, c.chunk_type, c.session_id, 
                c.project_path, c.project_name, c.file_path, 
                c.machine_id, c.timestamp, c.metadata,
                vec_distance_cosine(v.content_vector, ?) as distance
            FROM conversation_chunks c
            JOIN vec_chunks v ON c.rowid = v.rowid
            WHERE 1=1 {where_sql}
            ORDER BY distance ASC
            LIMIT ?
        """
        
        # Params: vector, filter_params..., limit
        full_params = [json.dumps(query_vector)] + params + [limit]
        
        async with self.conn.execute(sql, full_params) as cursor:
            rows = await cursor.fetchall()
            
        results = []
        for row in rows:
            # row: id, content, type, ..., distance
            # Convert to dict
            # sqlite3.Row not enabled by default in aiosqlite?
            # 0:id, 1:content, 2:chunk_type, 3:session_id, 4:project_path, 
            # 5:project_name, 6:file_path, 7:machine_id, 8:timestamp, 9:metadata, 10:distance
            
            meta = {}
            if row[9]:
                try:
                    meta = json.loads(row[9])
                except:
                    pass

            results.append({
                "id": row[0],
                "content": row[1],
                "chunk_type": row[2],
                "session_id": row[3],
                "project_path": row[4],
                "project_name": row[5],
                "file_path": row[6],
                "machine_id": row[7],
                "timestamp": row[8],
                "score": 1.0 - row[10], # distance to similarity
                **meta
            })
            
        return results

    async def delete(self, filters: dict[str, Any]) -> int:
        """Delete chunks."""
        if not self.conn:
            await self.initialize()
            
        where_clauses = []
        params = []
        for key, value in filters.items():
            if key in ["project_path", "chunk_type", "machine_id", "session_id"]:
                where_clauses.append(f"{key} = ?")
                params.append(value)
                
        if not where_clauses:
            return 0
            
        where_sql = " AND ".join(where_clauses)
        
        # We need to delete from both tables.
        # JOIN delete not supported in standard DELETE?
        # Delete from vec where rowid in (select rowid from chunks where ...)
        
        try:
            await self.conn.execute("BEGIN")
            # 1. Get rowids to delete
            cursor = await self.conn.execute(f"SELECT rowid FROM conversation_chunks WHERE {where_sql}", params)
            rows = await cursor.fetchall()
            rowids = [r[0] for r in rows]
            
            if not rowids:
                await self.conn.rollback() # Nothing to do
                return 0
                
            # 2. Delete
            placeholders = ",".join("?" for _ in rowids)
            await self.conn.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({placeholders})", rowids)
            await self.conn.execute(f"DELETE FROM conversation_chunks WHERE rowid IN ({placeholders})", rowids)
            
            await self.conn.commit()
            return len(rowids)
        except Exception as e:
            logger.error(f"Failed to delete from SQLite (rolling back): {e}")
            if self.conn:
                await self.conn.rollback()
            raise

    async def optimize(self) -> None:
        """Optimize."""
        if not self.conn:
            await self.initialize()
        await self.conn.execute("VACUUM")

    async def get_stats(self) -> dict[str, Any]:
        """Get stats."""
        if not self.conn:
             try:
                await self.initialize()
             except:
                return {"backend": "sqlite", "error": "disconnected"}
                
        try:
            cursor = await self.conn.execute("SELECT count(*) FROM conversation_chunks")
            row = await cursor.fetchone()
            count = row[0] if row else 0
            return {
                "backend": "sqlite",
                "total_chunks": count,
                "db_path": str(self.db_path)
            }
        except Exception as e:
            return {"backend": "sqlite", "error": str(e)}

    async def clear_all(self) -> int:
        """Clear all data."""
        if not self.conn:
            await self.initialize()
            
        await self.conn.execute("DELETE FROM conversation_chunks")
        await self.conn.execute("DELETE FROM vec_chunks")
        await self.conn.commit()
        return 0
