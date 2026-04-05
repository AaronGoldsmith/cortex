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
                distilled_at TEXT,
                turn_index INTEGER
            );

            CREATE TABLE IF NOT EXISTS distillations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source_model TEXT,
                entry_count INTEGER,
                content_hash TEXT UNIQUE,
                created_at TEXT DEFAULT (datetime('now')),
                context_window INTEGER DEFAULT 0
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
        _migrate(conn)
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply schema migrations for existing databases."""
    # Add turn_index to entries if missing
    cols = {row[1] for row in conn.execute("PRAGMA table_info(entries)").fetchall()}
    if "turn_index" not in cols:
        conn.execute("ALTER TABLE entries ADD COLUMN turn_index INTEGER")

    # Add context_window to distillations if missing
    cols = {row[1] for row in conn.execute("PRAGMA table_info(distillations)").fetchall()}
    if "context_window" not in cols:
        conn.execute("ALTER TABLE distillations ADD COLUMN context_window INTEGER DEFAULT 0")

    # Create index after columns exist
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_entries_session_turn "
        "ON entries(session_id, turn_index)"
    )


def insert_entry(
    conn: sqlite3.Connection,
    content: str,
    entry_type: str = "raw",
    source_model: str = None,
    source_project: str = None,
    session_id: str = None,
    confidence: float = 1.0,
    embedding: list[float] = None,
    turn_index: int = None,
) -> Optional[int]:
    """Insert an entry and its embedding. Return entry id, or None if duplicate."""
    ch = _content_hash(content)
    try:
        cur = conn.execute(
            """INSERT INTO entries
               (content, entry_type, source_model, source_project, session_id, confidence, content_hash, turn_index)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (content, entry_type, source_model, source_project, session_id, confidence, ch, turn_index),
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
    context_window: int = 0,
) -> Optional[int]:
    """Insert a distillation, its embedding, and lineage rows. Return distillation id."""
    ch = _content_hash(content)
    try:
        cur = conn.execute(
            """INSERT INTO distillations
               (content, pattern_type, confidence, source_model, entry_count, content_hash, context_window)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (content, pattern_type, confidence, source_model, entry_count, ch, context_window),
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


def record_feedback(
    conn: sqlite3.Connection,
    entry_id: int = None,
    distillation_id: int = None,
    helpful: bool = True,
    context: str = None,
) -> int:
    """Insert a feedback row. Returns the feedback id.

    Validates that the referenced entry/distillation exists.
    Prevents duplicate feedback for the same (entry/distillation, context) combo.
    """
    if entry_id is None and distillation_id is None:
        raise ValueError("Must provide either entry_id or distillation_id")

    # Validate existence
    if entry_id is not None:
        row = conn.execute("SELECT id FROM entries WHERE id = ?", (entry_id,)).fetchone()
        if not row:
            raise ValueError(f"Entry {entry_id} not found")

    if distillation_id is not None:
        row = conn.execute("SELECT id FROM distillations WHERE id = ?", (distillation_id,)).fetchone()
        if not row:
            raise ValueError(f"Distillation {distillation_id} not found")

    # Check for duplicate feedback (same target + same context)
    existing = conn.execute(
        "SELECT id FROM feedback WHERE entry_id IS ? AND distillation_id IS ? AND context IS ?",
        (entry_id, distillation_id, context),
    ).fetchone()
    if existing:
        raise ValueError("Duplicate feedback already recorded for this item and context")

    cur = conn.execute(
        "INSERT INTO feedback (entry_id, distillation_id, helpful, context) VALUES (?, ?, ?, ?)",
        (entry_id, distillation_id, 1 if helpful else 0, context),
    )
    conn.commit()
    return cur.lastrowid


def adjust_confidence_from_feedback(
    conn: sqlite3.Connection,
    entry_id: int = None,
    distillation_id: int = None,
    helpful: bool = True,
) -> Optional[float]:
    """Adjust confidence of an entry or distillation based on feedback.

    Each 'helpful' bumps confidence by +0.05 (capped at 2.0).
    Each 'not helpful' reduces confidence by -0.1 (floored at 0.1).
    Returns the new confidence value.
    """
    if entry_id is not None:
        table = "entries"
        item_id = entry_id
    elif distillation_id is not None:
        table = "distillations"
        item_id = distillation_id
    else:
        return None

    row = conn.execute(f"SELECT confidence FROM {table} WHERE id = ?", (item_id,)).fetchone()
    if not row:
        return None

    current = row[0] or 1.0
    if helpful:
        new_conf = min(2.0, current + 0.05)
    else:
        new_conf = max(0.1, current - 0.1)

    new_conf = round(new_conf, 4)
    conn.execute(f"UPDATE {table} SET confidence = ? WHERE id = ?", (new_conf, item_id))
    conn.commit()
    return new_conf


def get_feedback_stats(conn: sqlite3.Connection) -> dict:
    """Return aggregate feedback statistics."""
    total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    helpful_count = conn.execute("SELECT COUNT(*) FROM feedback WHERE helpful = 1").fetchone()[0]

    helpful_rate = (helpful_count / total * 100) if total > 0 else 0.0

    # Top 5 most-helpful entries/distillations
    top_helpful = conn.execute("""
        SELECT entry_id, distillation_id, SUM(helpful) as helpful_sum, COUNT(*) as total
        FROM feedback
        GROUP BY entry_id, distillation_id
        ORDER BY helpful_sum DESC
        LIMIT 5
    """).fetchall()

    # Top 5 least-helpful (most "no" feedback)
    top_unhelpful = conn.execute("""
        SELECT entry_id, distillation_id,
               SUM(CASE WHEN helpful = 0 THEN 1 ELSE 0 END) as unhelpful_sum,
               COUNT(*) as total
        FROM feedback
        GROUP BY entry_id, distillation_id
        ORDER BY unhelpful_sum DESC
        LIMIT 5
    """).fetchall()

    # Breakdown by entry type / pattern type
    entry_breakdown = conn.execute("""
        SELECT e.entry_type, COUNT(*) as cnt,
               SUM(f.helpful) as helpful_sum
        FROM feedback f
        JOIN entries e ON f.entry_id = e.id
        WHERE f.entry_id IS NOT NULL
        GROUP BY e.entry_type
    """).fetchall()

    pattern_breakdown = conn.execute("""
        SELECT d.pattern_type, COUNT(*) as cnt,
               SUM(f.helpful) as helpful_sum
        FROM feedback f
        JOIN distillations d ON f.distillation_id = d.id
        WHERE f.distillation_id IS NOT NULL
        GROUP BY d.pattern_type
    """).fetchall()

    return {
        "total": total,
        "helpful_count": helpful_count,
        "unhelpful_count": total - helpful_count,
        "helpful_rate": round(helpful_rate, 1),
        "top_helpful": [
            {
                "id": f"D{r[1]}" if r[1] else f"E{r[0]}",
                "helpful": r[2],
                "total": r[3],
            }
            for r in top_helpful
        ],
        "top_unhelpful": [
            {
                "id": f"D{r[1]}" if r[1] else f"E{r[0]}",
                "unhelpful": r[2],
                "total": r[3],
            }
            for r in top_unhelpful
        ],
        "entry_type_breakdown": [
            {"type": r[0], "count": r[1], "helpful": r[2]}
            for r in entry_breakdown
        ],
        "pattern_type_breakdown": [
            {"type": r[0], "count": r[1], "helpful": r[2]}
            for r in pattern_breakdown
        ],
    }


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
