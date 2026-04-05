"""Goose ingestion provider — reads session history from Goose's SQLite database."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterator

from cortex.config import GOOSE_DB_PATH
from cortex.providers import register
from cortex.providers.base import IngestEntry

log = logging.getLogger(__name__)


def _extract_text(content_json: str) -> str:
    """Extract text content from Goose's content_json field."""
    try:
        blocks = json.loads(content_json)
    except (json.JSONDecodeError, TypeError):
        return ""
    if isinstance(blocks, list):
        texts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(t for t in texts if t).strip()
    return ""


def _get_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM ingest_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO ingest_state (key, value, updated_at) VALUES (?, ?, datetime('now')) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=datetime('now')",
        (key, value),
    )
    conn.commit()


class GooseProvider:
    """Ingestion provider for Goose (Block) session database."""

    name = "goose"

    def __init__(self, db_path: Path = None):
        self.db_path = Path(db_path) if db_path else GOOSE_DB_PATH

    def detect(self) -> bool:
        return self.db_path.exists()

    def iter_entries(self, conn: sqlite3.Connection) -> Iterator[IngestEntry]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Goose database not found: {self.db_path}")

        last_ts = _get_state(conn, "goose_last_timestamp")
        last_ts = int(last_ts) if last_ts else 0

        goose_conn = sqlite3.connect(str(self.db_path))
        max_ts = last_ts
        try:
            rows = goose_conn.execute(
                """SELECT m.message_id, m.session_id, m.role, m.content_json,
                          m.created_timestamp, s.working_dir, s.provider_name
                   FROM messages m
                   JOIN sessions s ON m.session_id = s.id
                   WHERE m.created_timestamp > ?
                   ORDER BY m.created_timestamp""",
                (last_ts,),
            ).fetchall()

            for row in rows:
                message_id, session_id, role, content_json, ts, working_dir, provider = row
                content = _extract_text(content_json)
                if not content or len(content) < 10:
                    continue

                project = Path(working_dir).name if working_dir else None
                source_model = f"goose/{provider}" if provider else "goose"

                if ts > max_ts:
                    max_ts = ts

                yield IngestEntry(
                    content=content,
                    entry_type="raw",
                    source_model=source_model,
                    source_project=project,
                    session_id=session_id,
                    confidence=1.0,
                    turn_index=None,
                )
        finally:
            goose_conn.close()

        if max_ts > last_ts:
            _set_state(conn, "goose_last_timestamp", str(max_ts))

    @property
    def metadata(self) -> dict:
        return {"db_path": str(self.db_path)}


register("goose", GooseProvider)
