"""Session log ingestion — parses Claude Code history.jsonl into the Cortex ledger."""

import json
import logging
from pathlib import Path

from cortex.db import insert_entry
from cortex.embedder import embed

log = logging.getLogger(__name__)


def _load_state(state_path: Path) -> dict:
    """Load cursor state from disk, or return defaults for first run."""
    try:
        return json.loads(state_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"last_line": 0, "last_session_id": None, "file_size": 0}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state))


def ingest_history(conn, history_path: Path, state_path: Path) -> dict:
    """Ingest new lines from history.jsonl into the ledger.

    Returns dict with keys: ingested, skipped, errors.
    """
    history_path = Path(history_path)
    state_path = Path(state_path)

    if not history_path.exists():
        raise FileNotFoundError(
            f"History file not found: {history_path}. "
            "Is Claude Code installed? Expected at ~/.claude/history.jsonl"
        )

    state = _load_state(state_path)
    file_size = history_path.stat().st_size

    # Detect file rotation (size shrank)
    if file_size < state.get("file_size", 0):
        log.info("History file shrank (%d -> %d), resetting cursor", state["file_size"], file_size)
        state["last_line"] = 0

    stats = {"ingested": 0, "skipped": 0, "errors": 0}
    last_line = state["last_line"]

    with open(history_path, "r", encoding="utf-8") as f:
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
                stats["errors"] += 1
                continue

            content = entry.get("display", "")
            if not content or len(content) < 10:
                stats["skipped"] += 1
                continue

            embedding = embed(content)
            result = insert_entry(
                conn,
                content=content,
                entry_type="raw",
                source_model="claude",
                source_project=entry.get("project"),
                session_id=entry.get("sessionId"),
                confidence=1.0,
                embedding=embedding,
            )
            if result is None:
                stats["skipped"] += 1  # duplicate
            else:
                stats["ingested"] += 1
            state["last_session_id"] = entry.get("sessionId")
            last_line = line_no + 1

    state["last_line"] = last_line
    state["file_size"] = file_size
    _save_state(state_path, state)
    return stats
