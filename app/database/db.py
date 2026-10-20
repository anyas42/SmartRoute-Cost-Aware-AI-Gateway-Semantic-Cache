from __future__ import annotations

import sqlite3
import os
import logging
from contextlib import contextmanager
from typing import Generator
from app.config import settings

logger = logging.getLogger("smartroute.database")

CREATE_REQUESTS_TABLE = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    query_text TEXT NOT NULL,
    answer TEXT NOT NULL,
    cache_hit INTEGER NOT NULL,
    cache_type TEXT NOT NULL,
    similarity_score REAL,
    selected_model TEXT NOT NULL,
    model_tier TEXT NOT NULL,
    routing_strategy TEXT NOT NULL,
    routing_reason TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    estimated_cost REAL NOT NULL,
    actual_cost REAL,
    cost_saved REAL NOT NULL,
    latency_ms REAL NOT NULL,
    namespace TEXT NOT NULL,
    provider TEXT NOT NULL,
    is_mock INTEGER NOT NULL,
    error TEXT
);
"""

CREATE_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS semantic_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    query_text TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    answer TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_tier TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    estimated_cost REAL NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    hit_count INTEGER DEFAULT 0,
    last_hit_at REAL
);
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(selected_model);
CREATE INDEX IF NOT EXISTS idx_requests_cache_hit ON requests(cache_hit);
CREATE INDEX IF NOT EXISTS idx_cache_namespace ON semantic_cache(namespace);
CREATE INDEX IF NOT EXISTS idx_cache_hash ON semantic_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON semantic_cache(expires_at);
"""


class DatabaseManager:
    """
    Manages SQLite database initialization and connections.
    Uses Write-Ahead Logging (WAL) for high-performance concurrent reads and writes.
    """
    def __init__(self, db_path: str = settings.DATABASE_PATH):
        self.db_path = db_path
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode and foreign keys
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    @contextmanager
    def session(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error during session: {e}", exc_info=True)
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True) if os.path.dirname(self.db_path) else None
        with self.session() as conn:
            conn.execute(CREATE_REQUESTS_TABLE)
            conn.execute(CREATE_CACHE_TABLE)
            conn.executescript(CREATE_INDEXES)
        logger.info(f"SQLite database initialized at {self.db_path}")


# Global database manager instance
db_manager = DatabaseManager()
