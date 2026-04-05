"""Tests for session JSONL reader and conversation-aware features."""

import json

import pytest

from cortex.db import (
    get_connection,
    get_undistilled_entries,
    init_db,
    insert_distillation,
    insert_entry,
)
from cortex.embedder import embed
from cortex.sessions import (
    Turn,
    find_session_file,
    format_context_window,
    format_turn,
    get_turn_context,
    read_session_turns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    """Fresh Cortex database in a temp directory."""
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = get_connection(db_path)
    yield conn
    conn.close()


@pytest.fixture
def session_file(tmp_path):
    """Create a mock session JSONL file with realistic conversation turns."""
    projects_dir = tmp_path / "projects"
    project_dir = projects_dir / "C--test-project"
    project_dir.mkdir(parents=True)

    session_id = "abc12345-1234-5678-9abc-def012345678"
    session_path = project_dir / f"{session_id}.jsonl"

    lines = [
        # System/metadata entries (should be skipped)
        {"type": "system", "message": {"role": "system", "content": "You are Claude"}},
        {"type": "file-history-snapshot", "message": {}},
        # Turn 0: user asks about auth, assistant responds with text
        {"type": "user", "message": {"role": "user", "content": "How do I fix the auth middleware?"}, "timestamp": "2026-01-01T10:00:00Z"},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Let me look at the auth middleware code."},
        ]}, "timestamp": "2026-01-01T10:00:05Z"},
        # Turn 1: assistant uses tool (should skip tool_use), then responds
        {"type": "user", "message": {"role": "user", "content": "The session tokens are stored wrong"}, "timestamp": "2026-01-01T10:01:00Z"},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "I need to check the session storage..."},
            {"type": "tool_use", "name": "Read", "id": "toolu_123", "input": {"path": "auth.py"}},
        ]}, "timestamp": "2026-01-01T10:01:05Z"},
        # Tool result (user message with list content — should be skipped)
        {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "file contents..."}]}},
        # Assistant responds with text after tool use
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "I see the issue. The session tokens are stored in plaintext. We should hash them."},
        ]}, "timestamp": "2026-01-01T10:01:30Z"},
        # Turn 2: user confirms
        {"type": "user", "message": {"role": "user", "content": "Yes, let's hash them with SHA-256"}, "timestamp": "2026-01-01T10:02:00Z"},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Done. I've updated the session storage to use SHA-256 hashing."},
        ]}, "timestamp": "2026-01-01T10:02:30Z"},
    ]

    session_path.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    return session_path, session_id, projects_dir


# ---------------------------------------------------------------------------
# Test: Turn extraction from session JSONL
# ---------------------------------------------------------------------------


def test_read_session_turns(session_file):
    """Verify that turns are correctly extracted and paired from a session JSONL."""
    session_path, _, _ = session_file
    turns = read_session_turns(session_path)

    assert len(turns) == 3, f"Expected 3 turns, got {len(turns)}"

    # Turn 0
    assert turns[0].index == 0
    assert "auth middleware" in turns[0].user_text
    assert "Let me look" in turns[0].assistant_text

    # Turn 1: user text paired with the text response (skipping tool_use/thinking)
    assert turns[1].index == 1
    assert "session tokens" in turns[1].user_text
    assert "hash them" in turns[1].assistant_text
    assert "tool_use" not in turns[1].assistant_text
    assert "thinking" not in turns[1].assistant_text

    # Turn 2
    assert turns[2].index == 2
    assert "SHA-256" in turns[2].user_text
    assert "Done" in turns[2].assistant_text


def test_read_session_turns_skips_tool_results(session_file):
    """Tool result messages (content is a list) should not create new turns."""
    session_path, _, _ = session_file
    turns = read_session_turns(session_path)

    # None of the turns should contain tool_result content
    for turn in turns:
        assert "tool_result" not in turn.user_text
        assert "file contents" not in turn.user_text


# ---------------------------------------------------------------------------
# Test: find_session_file
# ---------------------------------------------------------------------------


def test_find_session_file(session_file):
    """find_session_file should locate a session by UUID across project dirs."""
    _, session_id, projects_dir = session_file
    found = find_session_file(session_id, projects_dir)
    assert found is not None
    assert session_id in found.name


def test_find_session_file_missing(tmp_path):
    """Returns None for nonexistent session IDs."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    assert find_session_file("nonexistent-uuid", projects_dir) is None


# ---------------------------------------------------------------------------
# Test: Context window
# ---------------------------------------------------------------------------


def test_get_turn_context(session_file):
    """get_turn_context returns the right window around a focal turn."""
    session_path, _, _ = session_file

    # Window of 1 around turn 1 → turns 0, 1, 2
    context = get_turn_context(session_path, turn_index=1, window=1)
    assert len(context) == 3
    assert context[0].index == 0
    assert context[1].index == 1
    assert context[2].index == 2

    # Window of 1 around turn 0 → turns 0, 1 (clamped at start)
    context = get_turn_context(session_path, turn_index=0, window=1)
    assert len(context) == 2
    assert context[0].index == 0


def test_get_turn_context_zero_window(session_file):
    """Window of 0 returns empty list."""
    session_path, _, _ = session_file
    assert get_turn_context(session_path, turn_index=1, window=0) == []


# ---------------------------------------------------------------------------
# Test: Format helpers
# ---------------------------------------------------------------------------


def test_format_turn():
    """format_turn should mark focal turns with >>> prefix."""
    turn = Turn(index=0, user_text="hello", assistant_text="hi there")
    normal = format_turn(turn, focal=False)
    assert "   USER: hello" in normal
    focal = format_turn(turn, focal=True)
    assert ">>> USER: hello" in focal


# ---------------------------------------------------------------------------
# Test: DB schema — turn_index and context_window columns
# ---------------------------------------------------------------------------


def test_insert_entry_with_turn_index(db):
    """insert_entry should accept and store turn_index."""
    eid = insert_entry(
        db, content="Test entry with turn index",
        entry_type="raw", embedding=embed("test"),
        turn_index=5,
    )
    assert eid is not None
    row = db.execute("SELECT turn_index FROM entries WHERE id = ?", (eid,)).fetchone()
    assert row[0] == 5


def test_insert_entry_without_turn_index(db):
    """turn_index should be NULL when not provided."""
    eid = insert_entry(
        db, content="Test entry without turn index",
        entry_type="raw", embedding=embed("test"),
    )
    row = db.execute("SELECT turn_index FROM entries WHERE id = ?", (eid,)).fetchone()
    assert row[0] is None


def test_insert_distillation_with_context_window(db):
    """insert_distillation should accept and store context_window."""
    did = insert_distillation(
        db, content="Test pattern",
        pattern_type="workflow", confidence=0.8,
        embedding=embed("test pattern"),
        context_window=3,
    )
    assert did is not None
    row = db.execute("SELECT context_window FROM distillations WHERE id = ?", (did,)).fetchone()
    assert row[0] == 3


# ---------------------------------------------------------------------------
# Test: Distill with context window
# ---------------------------------------------------------------------------


def test_distill_with_context_window(db, session_file):
    """Distiller should include conversation context when context_window > 0."""
    session_path, session_id, projects_dir = session_file
    from cortex.distill import distill
    import cortex.config

    # Temporarily override PROJECTS_DIR for the test
    original_projects_dir = cortex.config.PROJECTS_DIR
    cortex.config.PROJECTS_DIR = projects_dir

    try:
        # Insert an entry with session_id and turn_index
        eid = insert_entry(
            db, content="The session tokens are stored wrong",
            entry_type="raw", source_model="claude",
            session_id=session_id,
            embedding=embed("session tokens stored wrong"),
            turn_index=1,
        )
        assert eid is not None

        # Track what prompt the LLM receives
        captured_prompts = []

        def mock_llm(prompt):
            captured_prompts.append(prompt)
            return json.dumps({"patterns": [{
                "content": "Session tokens should be hashed, not stored in plaintext",
                "pattern_type": "architecture",
                "confidence": 0.9,
                "source_entry_ids": [eid],
            }], "skipped": []})

        stats = distill(db, max_batches=1, batch_size=10, llm_call=mock_llm, context_window=1)
        assert stats["distillations"] == 1
        assert stats["errors"] == 0

        # Verify the prompt included conversation context
        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "Conversation context" in prompt
        assert "Focal message" in prompt
        assert "auth middleware" in prompt  # Turn 0 should be in context

        # Verify context_window was stored on the distillation
        dist = db.execute("SELECT context_window FROM distillations").fetchone()
        assert dist[0] == 1
    finally:
        cortex.config.PROJECTS_DIR = original_projects_dir


# ---------------------------------------------------------------------------
# Test: Ingest with turn_index resolution
# ---------------------------------------------------------------------------


def test_ingest_resolves_turn_index(db, session_file, tmp_path):
    """ingest_history should resolve turn_index when projects_dir is provided."""
    _, session_id, projects_dir = session_file
    from cortex.ingest import ingest_history

    history_path = tmp_path / "history.jsonl"
    state_path = tmp_path / "ingest_state.json"

    # Create a history entry that matches a turn in our session
    entry = {
        "display": "The session tokens are stored wrong",
        "sessionId": session_id,
        "project": "test-project",
    }
    history_path.write_text(json.dumps(entry) + "\n")

    stats = ingest_history(db, history_path, state_path, projects_dir=projects_dir)
    assert stats["ingested"] == 1

    # Check that turn_index was resolved
    row = db.execute("SELECT turn_index, session_id FROM entries").fetchone()
    assert row[0] == 1, f"Expected turn_index=1, got {row[0]}"
    assert row[1] == session_id
