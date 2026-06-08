"""Codex ingestion provider -- reads completed turns from local session JSONL."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterator

from cortex.config import CODEX_ARCHIVED_SESSIONS_DIR, CODEX_SESSIONS_DIR
from cortex.providers import register
from cortex.providers.base import IngestEntry

log = logging.getLogger(__name__)


def _state_key(session_path: Path) -> str:
    """Use the rollout filename so state follows sessions moved to the archive."""
    return f"codex:v2:session:{session_path.stem}"


def _load_state(conn: sqlite3.Connection, session_path: Path) -> dict:
    row = conn.execute(
        "SELECT value FROM ingest_state WHERE key = ?",
        (_state_key(session_path),),
    ).fetchone()
    if not row:
        return {}
    try:
        return json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return {}


def _save_state(conn: sqlite3.Connection, session_path: Path, state: dict) -> None:
    conn.execute(
        "INSERT INTO ingest_state (key, value, updated_at) VALUES (?, ?, datetime('now')) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=datetime('now')",
        (_state_key(session_path), json.dumps(state)),
    )
    conn.commit()


def _project_name(cwd: str | None) -> str | None:
    if not cwd:
        return None
    return Path(cwd).name or None


def _format_turn(user_text: str, assistant_text: str) -> str:
    return f"USER: {user_text.strip()}\nASSISTANT: {assistant_text.strip()}"


def _legacy_assistant_text(payload: dict) -> str | None:
    """Extract old-format assistant text while excluding embedded tool traffic."""
    if payload.get("type") != "message" or payload.get("role") != "assistant":
        return None
    content = payload.get("content")
    blocks = content if isinstance(content, list) else []
    texts = [
        block.get("text", "").strip()
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "output_text"
    ]
    text = "\n".join(part for part in texts if part).strip()
    if not text or text.startswith("[external_agent_tool_"):
        return None
    return text


class CodexProvider:
    """Ingest completed user/assistant turns from Codex session event streams."""

    name = "codex"

    def __init__(
        self,
        sessions_dir: Path | None = None,
        archived_sessions_dir: Path | None = None,
    ):
        self.sessions_dir = Path(sessions_dir) if sessions_dir else CODEX_SESSIONS_DIR
        self.archived_sessions_dir = (
            Path(archived_sessions_dir)
            if archived_sessions_dir
            else CODEX_ARCHIVED_SESSIONS_DIR
        )

    def detect(self) -> bool:
        return any(self._session_paths())

    def _session_paths(self) -> list[Path]:
        paths: dict[str, Path] = {}
        for root in (self.archived_sessions_dir, self.sessions_dir):
            if not root.exists():
                continue
            for path in root.rglob("*.jsonl"):
                # Prefer the active copy if a session exists in both locations.
                paths[path.stem] = path
        return sorted(paths.values())

    def iter_entries(self, conn: sqlite3.Connection) -> Iterator[IngestEntry]:
        for session_path in self._session_paths():
            try:
                size = session_path.stat().st_size
            except OSError:
                continue

            state = _load_state(conn, session_path)
            offset = int(state.get("offset", 0))
            if size < offset:
                state = {}
                offset = 0

            pending_user = state.get("pending_user")
            pending_assistant = state.get("pending_assistant")
            pending_legacy_assistant = state.get("pending_legacy_assistant")
            current_turn_id = state.get("current_turn_id")
            session_id = state.get("session_id", session_path.stem)
            source_project = state.get("source_project")
            turn_index = int(state.get("turn_index", 0))
            new_offset = offset

            try:
                with open(session_path, "rb") as session_file:
                    session_file.seek(offset)
                    while True:
                        line_start = session_file.tell()
                        raw = session_file.readline()
                        if not raw:
                            break
                        # Do not consume a partial line while Codex is writing it.
                        if not raw.endswith(b"\n"):
                            session_file.seek(line_start)
                            break
                        new_offset = session_file.tell()

                        try:
                            record = json.loads(raw.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue

                        record_type = record.get("type")
                        payload = record.get("payload", {})
                        if record_type == "session_meta":
                            session_id = payload.get("id") or session_id
                            source_project = _project_name(payload.get("cwd"))
                            continue
                        if record_type == "response_item":
                            legacy_text = _legacy_assistant_text(payload)
                            if legacy_text:
                                pending_legacy_assistant = legacy_text
                            continue
                        if record_type != "event_msg":
                            continue

                        event_type = payload.get("type")
                        if event_type == "task_started":
                            current_turn_id = payload.get("turn_id")
                            pending_user = None
                            pending_assistant = None
                            pending_legacy_assistant = None
                        elif event_type == "user_message":
                            pending_user = payload.get("message")
                        elif (
                            event_type == "agent_message"
                            and payload.get("phase") == "final_answer"
                        ):
                            pending_assistant = payload.get("message")
                        elif event_type == "task_complete":
                            completed_turn_id = payload.get("turn_id")
                            assistant_text = pending_assistant or pending_legacy_assistant
                            if (
                                pending_user
                                and assistant_text
                                and (
                                    not current_turn_id
                                    or not completed_turn_id
                                    or completed_turn_id == current_turn_id
                                )
                            ):
                                yield IngestEntry(
                                    content=_format_turn(pending_user, assistant_text),
                                    entry_type="raw",
                                    source_model="codex",
                                    source_project=source_project,
                                    session_id=session_id,
                                    confidence=1.0,
                                    turn_index=turn_index,
                                )
                                turn_index += 1
                            pending_user = None
                            pending_assistant = None
                            pending_legacy_assistant = None
                            current_turn_id = None
                        elif event_type == "turn_aborted":
                            pending_user = None
                            pending_assistant = None
                            pending_legacy_assistant = None
                            current_turn_id = None
            except OSError as exc:
                log.warning("Failed reading Codex session %s: %s", session_path, exc)
                continue

            _save_state(
                conn,
                session_path,
                {
                    "offset": new_offset,
                    "pending_user": pending_user,
                    "pending_assistant": pending_assistant,
                    "pending_legacy_assistant": pending_legacy_assistant,
                    "current_turn_id": current_turn_id,
                    "session_id": session_id,
                    "source_project": source_project,
                    "turn_index": turn_index,
                },
            )

    @property
    def metadata(self) -> dict:
        return {
            "sessions_dir": str(self.sessions_dir),
            "archived_sessions_dir": str(self.archived_sessions_dir),
        }


register("codex", CodexProvider)
