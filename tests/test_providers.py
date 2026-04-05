"""Tests for ingestion providers — Claude extraction and Goose provider."""

import json
import sqlite3

import pytest

from cortex.db import init_db, get_connection
from cortex.ingest import run_provider_ingest
from cortex.providers.claude import ClaudeHistoryProvider
from cortex.providers.goose import GooseProvider


@pytest.fixture
def db(tmp_path):
    """Fresh Cortex database for each test."""
    db_path = tmp_path / "cortex.db"
    init_db(db_path)
    conn = get_connection(db_path)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Claude provider tests
# ---------------------------------------------------------------------------


def test_claude_provider_iter_entries(db, tmp_path):
    """ClaudeHistoryProvider yields entries and the orchestrator ingests them."""
    history_path = tmp_path / "history.jsonl"
    state_path = tmp_path / "state.json"

    lines = []
    for i in range(3):
        lines.append(json.dumps({
            "display": f"This is test entry number {i} with enough content to pass filter",
            "sessionId": f"session-{i}",
            "project": "test-project",
        }))
    history_path.write_text("\n".join(lines) + "\n")

    provider = ClaudeHistoryProvider(history_path, state_path, projects_dir=None)
    stats = run_provider_ingest(db, provider)

    assert stats["ingested"] == 3
    assert stats["skipped"] == 0
    assert stats["errors"] == 0

    rows = db.execute("SELECT source_model, source_project FROM entries").fetchall()
    assert all(r[0] == "claude" for r in rows)
    assert all(r[1] == "test-project" for r in rows)


def test_claude_provider_cursor_state(db, tmp_path):
    """Second ingest adds nothing thanks to cursor state."""
    history_path = tmp_path / "history.jsonl"
    state_path = tmp_path / "state.json"

    lines = [json.dumps({
        "display": f"Entry {i} with sufficient content for ingestion",
        "sessionId": f"s-{i}",
        "project": "proj",
    }) for i in range(3)]
    history_path.write_text("\n".join(lines) + "\n")

    provider = ClaudeHistoryProvider(history_path, state_path, projects_dir=None)
    stats1 = run_provider_ingest(db, provider)
    assert stats1["ingested"] == 3

    provider2 = ClaudeHistoryProvider(history_path, state_path, projects_dir=None)
    stats2 = run_provider_ingest(db, provider2)
    assert stats2["ingested"] == 0


def test_claude_provider_detect(tmp_path):
    """detect() returns True only when history file exists."""
    provider = ClaudeHistoryProvider(
        history_path=tmp_path / "nonexistent.jsonl",
        state_path=tmp_path / "state.json",
    )
    assert provider.detect() is False

    (tmp_path / "history.jsonl").write_text("")
    provider2 = ClaudeHistoryProvider(
        history_path=tmp_path / "history.jsonl",
        state_path=tmp_path / "state.json",
    )
    assert provider2.detect() is True


# ---------------------------------------------------------------------------
# Goose provider tests
# ---------------------------------------------------------------------------


def _create_goose_db(path):
    """Create a minimal Goose-schema SQLite database for testing."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL DEFAULT '',
            working_dir TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            provider_name TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            role TEXT NOT NULL,
            content_json TEXT NOT NULL,
            created_timestamp INTEGER NOT NULL
        );
    """)
    return conn


def test_goose_provider_iter_entries(db, tmp_path):
    """GooseProvider reads messages from a Goose sessions.db and ingests them."""
    goose_db_path = tmp_path / "sessions.db"
    goose_conn = _create_goose_db(goose_db_path)

    goose_conn.execute(
        "INSERT INTO sessions (id, name, working_dir, provider_name) VALUES (?, ?, ?, ?)",
        ("session-1", "test", "/home/user/myproject", "openrouter"),
    )
    for i in range(5):
        content = json.dumps([{"type": "text", "text": f"Goose message number {i} with enough content to pass"}])
        goose_conn.execute(
            "INSERT INTO messages (message_id, session_id, role, content_json, created_timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"msg-{i}", "session-1", "user" if i % 2 == 0 else "assistant", content, 1000 + i),
        )
    goose_conn.commit()
    goose_conn.close()

    provider = GooseProvider(db_path=goose_db_path)
    stats = run_provider_ingest(db, provider)

    assert stats["ingested"] == 5
    assert stats["errors"] == 0

    rows = db.execute("SELECT source_model, source_project, session_id FROM entries").fetchall()
    assert all("goose" in r[0] for r in rows)
    assert all(r[1] == "myproject" for r in rows)
    assert all(r[2] == "session-1" for r in rows)


def test_goose_provider_cursor_state(db, tmp_path):
    """Second Goose ingest only picks up new messages."""
    goose_db_path = tmp_path / "sessions.db"
    goose_conn = _create_goose_db(goose_db_path)

    goose_conn.execute(
        "INSERT INTO sessions (id, name, working_dir, provider_name) VALUES (?, ?, ?, ?)",
        ("s1", "test", "/home/user/proj", "chatgpt_codex"),
    )
    for i in range(3):
        content = json.dumps([{"type": "text", "text": f"First batch message {i} with enough content here"}])
        goose_conn.execute(
            "INSERT INTO messages (message_id, session_id, role, content_json, created_timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"msg-{i}", "s1", "user", content, 1000 + i),
        )
    goose_conn.commit()

    provider = GooseProvider(db_path=goose_db_path)
    stats1 = run_provider_ingest(db, provider)
    assert stats1["ingested"] == 3

    # Add more messages
    for i in range(3, 6):
        content = json.dumps([{"type": "text", "text": f"Second batch message {i} with enough content here"}])
        goose_conn.execute(
            "INSERT INTO messages (message_id, session_id, role, content_json, created_timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"msg-{i}", "s1", "user", content, 2000 + i),
        )
    goose_conn.commit()
    goose_conn.close()

    provider2 = GooseProvider(db_path=goose_db_path)
    stats2 = run_provider_ingest(db, provider2)
    assert stats2["ingested"] == 3

    total = db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    assert total == 6


def test_goose_provider_detect(tmp_path):
    """detect() returns True only when Goose DB exists."""
    provider = GooseProvider(db_path=tmp_path / "nonexistent.db")
    assert provider.detect() is False

    goose_db_path = tmp_path / "sessions.db"
    _create_goose_db(goose_db_path).close()
    provider2 = GooseProvider(db_path=goose_db_path)
    assert provider2.detect() is True


def test_goose_provider_skips_short_content(db, tmp_path):
    """Messages with content shorter than 10 chars are skipped."""
    goose_db_path = tmp_path / "sessions.db"
    goose_conn = _create_goose_db(goose_db_path)

    goose_conn.execute(
        "INSERT INTO sessions (id, name, working_dir, provider_name) VALUES (?, ?, ?, ?)",
        ("s1", "test", "/tmp", "test_provider"),
    )
    # Short message
    goose_conn.execute(
        "INSERT INTO messages (message_id, session_id, role, content_json, created_timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        ("msg-short", "s1", "user", json.dumps([{"type": "text", "text": "hi"}]), 1000),
    )
    # Long enough message
    goose_conn.execute(
        "INSERT INTO messages (message_id, session_id, role, content_json, created_timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        ("msg-long", "s1", "user", json.dumps([{"type": "text", "text": "This is a long enough message to be ingested"}]), 1001),
    )
    goose_conn.commit()
    goose_conn.close()

    provider = GooseProvider(db_path=goose_db_path)
    stats = run_provider_ingest(db, provider)

    assert stats["ingested"] == 1


def test_goose_provider_source_model_includes_provider(db, tmp_path):
    """source_model includes the Goose provider name (e.g. goose/chatgpt_codex)."""
    goose_db_path = tmp_path / "sessions.db"
    goose_conn = _create_goose_db(goose_db_path)

    goose_conn.execute(
        "INSERT INTO sessions (id, name, working_dir, provider_name) VALUES (?, ?, ?, ?)",
        ("s1", "test", "/tmp/myproject", "chatgpt_codex"),
    )
    goose_conn.execute(
        "INSERT INTO messages (message_id, session_id, role, content_json, created_timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        ("msg-1", "s1", "user",
         json.dumps([{"type": "text", "text": "A sufficiently long message for the test"}]),
         1000),
    )
    goose_conn.commit()
    goose_conn.close()

    provider = GooseProvider(db_path=goose_db_path)
    run_provider_ingest(db, provider)

    row = db.execute("SELECT source_model FROM entries").fetchone()
    assert row[0] == "goose/chatgpt_codex"
