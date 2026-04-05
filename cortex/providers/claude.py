"""Claude Code ingestion provider — parses history.jsonl into normalized entries."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterator

from cortex.config import HISTORY_PATH, PROJECTS_DIR, STATE_PATH
from cortex.providers import register
from cortex.providers.base import IngestEntry

log = logging.getLogger(__name__)

# Cache for session turn lookups within a single ingest run
_session_turn_cache: dict[str, list] = {}


def _load_state(state_path: Path) -> dict:
    """Load cursor state from disk, or return defaults for first run."""
    try:
        return json.loads(state_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"last_line": 0, "last_session_id": None, "file_size": 0}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state))


def _resolve_turn_index(session_id: str, content: str, projects_dir: Path) -> int | None:
    """Find the turn index of a user message within its session JSONL."""
    if session_id in _session_turn_cache:
        turns = _session_turn_cache[session_id]
    else:
        from cortex.sessions import find_session_file, read_session_turns

        session_path = find_session_file(session_id, projects_dir)
        if session_path is None:
            _session_turn_cache[session_id] = []
            return None
        turns = read_session_turns(session_path)
        _session_turn_cache[session_id] = turns

    content_stripped = content.strip()
    for turn in turns:
        if turn.user_text == content_stripped:
            return turn.index
        if len(content_stripped) > 50 and turn.user_text.startswith(content_stripped[:50]):
            return turn.index
    return None


class ClaudeHistoryProvider:
    """Ingestion provider for Claude Code history.jsonl files."""

    name = "claude"

    def __init__(
        self,
        history_path: Path = None,
        state_path: Path = None,
        projects_dir: Path = None,
    ):
        self.history_path = Path(history_path) if history_path else HISTORY_PATH
        self.state_path = Path(state_path) if state_path else STATE_PATH
        self.projects_dir = projects_dir if projects_dir is not None else PROJECTS_DIR

    def detect(self) -> bool:
        return self.history_path.exists()

    def iter_entries(self, conn: sqlite3.Connection) -> Iterator[IngestEntry]:
        if not self.history_path.exists():
            raise FileNotFoundError(
                f"History file not found: {self.history_path}. "
                "Is Claude Code installed? Expected at ~/.claude/history.jsonl"
            )

        state = _load_state(self.state_path)
        file_size = self.history_path.stat().st_size

        # Detect file rotation (size shrank)
        if file_size < state.get("file_size", 0):
            log.info("History file shrank (%d -> %d), resetting cursor", state["file_size"], file_size)
            state["last_line"] = 0

        last_line = state["last_line"]

        with open(self.history_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if line_no < last_line:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("Malformed JSON at line %d, skipping", line_no)
                    continue

                content = entry.get("display", "")
                if not content or len(content) < 10:
                    continue

                session_id = entry.get("sessionId")
                turn_index = None
                if self.projects_dir and session_id:
                    turn_index = _resolve_turn_index(session_id, content, self.projects_dir)

                state["last_session_id"] = session_id
                last_line = line_no + 1

                yield IngestEntry(
                    content=content,
                    entry_type="raw",
                    source_model="claude",
                    source_project=entry.get("project"),
                    session_id=session_id,
                    confidence=1.0,
                    turn_index=turn_index,
                )

        state["last_line"] = last_line
        state["file_size"] = file_size
        _save_state(self.state_path, state)

    @property
    def metadata(self) -> dict:
        return {
            "history_path": str(self.history_path),
            "state_path": str(self.state_path),
            "projects_dir": str(self.projects_dir),
        }


register("claude", ClaudeHistoryProvider)
