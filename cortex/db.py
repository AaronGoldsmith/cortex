"""Database module — SQLite + sqlite-vec schema, connection, and query helpers."""

import hashlib
import sqlite3
import struct
from typing import Optional

import sqlite_vec

from cortex.config import DB_PATH, EMBEDDING_DIM


def serialize_embedding(embedding) -> bytes:
    """Ensure embedding is in bytes format for sqlite-vec.

    Accepts either raw bytes (from embedder) or a list of floats.
    """
    if isinstance(embedding, bytes):
        return embedding
    return struct.pack(f"{len(embedding)}f", *embedding)


def _content_hash(content: str) -> str:
    """SHA-256 hex digest of content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def get_connection(db_path: str = None) -> sqlite3.Connection:
    """Return a sqlite3 connection with WAL mode, foreign keys, and sqlite-vec loaded."""
    path = db_path or str(DB_PATH)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.row_factory = sqlite3.Row
    return db


def init_db(db_path: str = None) -> None:
    """Create all tables if they don't exist."""
    conn = get_connection(db_path)
    try:
        conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                entry_type TEXT NOT NULL DEFAULT 'raw',
                source_model TEXT,
                source_project TEXT,
                session_id TEXT,
                confidence REAL DEFAULT 1.0,
                content_hash TEXT UNIQUE,
                created_at TEXT DEFAULT (datetime('now')),
                distilled_at TEXT
            );

            CREATE TABLE IF NOT EXISTS distillations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source_model TEXT,
                entry_count INTEGER,
                content_hash TEXT UNIQUE,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                distillation_id INTEGER REFERENCES distillations(id),
                entry_id INTEGER REFERENCES entries(id),
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER REFERENCES entries(id),
                distillation_id INTEGER REFERENCES distillations(id),
                helpful INTEGER,
                context TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS ingest_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS entry_vec
                USING vec0(embedding float[{EMBEDDING_DIM}]);

            CREATE VIRTUAL TABLE IF NOT EXISTS distill_vec
                USING vec0(embedding float[{EMBEDDING_DIM}]);
        """)
        conn.commit()
    finally:
        conn.close()


def insert_entry(
    conn: sqlite3.Connection,
    content: str,
    entry_type: str = "raw",
    source_model: str = None,
    source_project: str = None,
    session_id: str = None,
    confidence: float = 1.0,
    embedding: list[float] = None,
) -> Optional[int]:
    """Insert an entry and its embedding. Return entry id, or None if duplicate."""
    ch = _content_hash(content)
    try:
        cur = conn.execute(
            """INSERT INTO entries
               (content, entry_type, source_model, source_project, session_id, confidence, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (content, entry_type, source_model, source_project, session_id, confidence, ch),
        )
        entry_id = cur.lastrowid
    except sqlite3.IntegrityError:
        return None

    if embedding is not None:
        conn.execute(
            "INSERT INTO entry_vec (rowid, embedding) VALUES (?, ?)",
            (entry_id, serialize_embedding(embedding)),
        )

    conn.commit()
    return entry_id


def insert_distillation(
    conn: sqlite3.Connection,
    content: str,
    pattern_type: str,
    confidence: float = 1.0,
    source_model: str = None,
    entry_count: int = None,
    embedding: list[float] = None,
    source_entry_ids: list[int] = None,
) -> Optional[int]:
    """Insert a distillation, its embedding, and lineage rows. Return distillation id."""
    ch = _content_hash(content)
    try:
        cur = conn.execute(
            """INSERT INTO distillations
               (content, pattern_type, confidence, source_model, entry_count, content_hash)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (content, pattern_type, confidence, source_model, entry_count, ch),
        )
        distill_id = cur.lastrowid
    except sqlite3.IntegrityError:
        return None

    if embedding is not None:
        conn.execute(
            "INSERT INTO distill_vec (rowid, embedding) VALUES (?, ?)",
            (distill_id, serialize_embedding(embedding)),
        )

    if source_entry_ids:
        conn.executemany(
            "INSERT INTO lineage (distillation_id, entry_id) VALUES (?, ?)",
            [(distill_id, eid) for eid in source_entry_ids],
        )

    conn.commit()
    return distill_id


def get_undistilled_entries(conn: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    """Return entries where distilled_at IS NULL, ordered by created_at."""
    return conn.execute(
        "SELECT * FROM entries WHERE distilled_at IS NULL ORDER BY created_at LIMIT ?",
        (limit,),
    ).fetchall()


def mark_entries_distilled(conn: sqlite3.Connection, entry_ids: list[int]) -> None:
    """Set distilled_at = now for the given entry ids."""
    if not entry_ids:
        return
    placeholders = ",".join("?" for _ in entry_ids)
    conn.execute(
        f"UPDATE entries SET distilled_at = datetime('now') WHERE id IN ({placeholders})",
        entry_ids,
    )
    conn.commit()


def vector_search(
    conn: sqlite3.Connection,
    embedding: list[float],
    table: str = "entry_vec",
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Search a vec0 virtual table by cosine distance. Return list of (rowid, distance)."""
    rows = conn.execute(
        f"SELECT rowid, distance FROM {table} WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (serialize_embedding(embedding), top_k),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]
