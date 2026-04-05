"""Fundamental integration tests for Cortex.

These verify the CORE LOOP works end-to-end:
write -> distill -> query -> knowledge compounds.

NOT unit tests — these test the fundamental contracts of the system.
"""

import json
import sqlite3

import pytest

from cortex.db import (
    get_connection,
    get_undistilled_entries,
    init_db,
    insert_distillation,
    insert_entry,
    mark_entries_distilled,
    vector_search,
)
from cortex.embedder import embed, embed_query
from cortex.distill import distill
from cortex.query import query
from cortex.sanitize import has_secrets, sanitize
from cortex.ingest import ingest_history


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


# ---------------------------------------------------------------------------
# Test 1: Full Loop — write -> distill -> query
# ---------------------------------------------------------------------------


def test_full_loop_write_distill_query(db):
    """The fundamental test: entries go in, distillation extracts patterns,
    query finds the distilled knowledge."""

    entries = [
        "Use pdb.set_trace() to drop into an interactive debugger at any point in Python code.",
        "Python's breakpoint() builtin (3.7+) is cleaner than importing pdb directly.",
        "When debugging async Python code, use import asyncio; asyncio.get_event_loop().set_debug(True).",
        "The icecream library (ic()) is great for quick print-debugging without writing format strings.",
        "Python logging module with DEBUG level is better than print() for persistent debugging output.",
    ]

    entry_ids = []
    for text in entries:
        embedding = embed(text)
        eid = insert_entry(db, content=text, entry_type="raw", embedding=embedding)
        assert eid is not None, f"Failed to insert entry: {text[:40]}..."
        entry_ids.append(eid)

    # Verify all entries are undistilled
    undistilled = get_undistilled_entries(db)
    assert len(undistilled) == 5, "Expected 5 undistilled entries before distillation"

    # Mock LLM that returns a reasonable distillation
    def mock_llm(prompt):
        return json.dumps([
            {
                "content": (
                    "Python debugging techniques range from built-in tools "
                    "(pdb, breakpoint()) to third-party helpers (icecream). "
                    "For production code, prefer the logging module over print "
                    "statements. For async code, enable event loop debug mode."
                ),
                "pattern_type": "workflow",
                "confidence": 0.85,
            }
        ])

    stats = distill(db, max_batches=1, batch_size=10, llm_call=mock_llm)

    assert stats["distillations"] >= 1, "Distiller should have produced at least one distillation"
    assert stats["entries_processed"] == 5, "All 5 entries should have been processed"
    assert stats["errors"] == 0, "Distillation should not have errors"

    # Verify lineage: distillation links back to source entries
    lineage_rows = db.execute(
        "SELECT distillation_id, entry_id FROM lineage"
    ).fetchall()
    lineage_entry_ids = {row[1] for row in lineage_rows}
    assert lineage_entry_ids == set(entry_ids), (
        "Every source entry should appear in the lineage table"
    )

    # Verify original entries are marked as distilled
    undistilled_after = get_undistilled_entries(db)
    assert len(undistilled_after) == 0, "All entries should be marked distilled after distillation"

    # Query for debugging — should find the distillation
    results = query(db, "debugging", top_k=5)
    assert len(results) > 0, "Query for 'debugging' should return results"

    # The distillation should appear in results
    distillation_results = [r for r in results if r["kind"] == "distillation"]
    assert len(distillation_results) >= 1, (
        "Query should return the distilled pattern, not just raw entries"
    )


# ---------------------------------------------------------------------------
# Test 2: Semantic Relevance — search is meaningful, not random
# ---------------------------------------------------------------------------


def test_semantic_relevance(db):
    """Verify that semantic search actually ranks by meaning.
    React entries should rank higher for React queries,
    SQL entries should rank higher for database queries."""

    react_entries = [
        "React useEffect hook runs after render and handles side effects like API calls.",
        "React components re-render when state or props change; use React.memo to prevent unnecessary renders.",
        "The useState hook returns a state value and setter function for managing component state.",
    ]

    sql_entries = [
        "Adding an index on frequently-queried columns dramatically improves SELECT performance.",
        "Use EXPLAIN ANALYZE to identify slow query plans and missing indexes in PostgreSQL.",
        "Database connection pooling reduces overhead of creating new connections for each query.",
    ]

    for text in react_entries:
        insert_entry(db, content=text, entry_type="raw", embedding=embed(text))
    for text in sql_entries:
        insert_entry(db, content=text, entry_type="raw", embedding=embed(text))

    # Query for React-related topic
    react_results = query(db, "React hooks and component lifecycle", top_k=6, include_raw=True)
    assert len(react_results) >= 3, "Should return enough results to compare ranking"

    # Top results should be React entries
    top_3_contents = [r["content"] for r in react_results[:3]]
    react_in_top_3 = sum(1 for c in top_3_contents if any(kw in c for kw in ["React", "useState", "useEffect"]))
    assert react_in_top_3 >= 2, (
        f"At least 2 of top 3 results for 'React hooks' should be React entries, "
        f"got {react_in_top_3}. Top 3: {top_3_contents}"
    )

    # Query for SQL-related topic
    sql_results = query(db, "database performance and query optimization", top_k=6, include_raw=True)
    assert len(sql_results) >= 3, "Should return enough results to compare ranking"

    top_3_contents = [r["content"] for r in sql_results[:3]]
    sql_in_top_3 = sum(1 for c in top_3_contents if any(kw in c for kw in ["index", "EXPLAIN", "connection pool", "SELECT"]))
    assert sql_in_top_3 >= 2, (
        f"At least 2 of top 3 results for 'database performance' should be SQL entries, "
        f"got {sql_in_top_3}. Top 3: {top_3_contents}"
    )


# ---------------------------------------------------------------------------
# Test 3: Dedup Contract — content_hash prevents duplicates
# ---------------------------------------------------------------------------


def test_dedup_contract(db):
    """Identical content should be stored once. Different content should be stored separately."""

    content = "Always use virtual environments for Python projects to isolate dependencies."
    embedding = embed(content)

    first_id = insert_entry(db, content=content, entry_type="raw", embedding=embedding)
    assert first_id is not None, "First insert should succeed"

    second_id = insert_entry(db, content=content, entry_type="raw", embedding=embedding)
    assert second_id is None, "Duplicate content should return None (rejected by content_hash UNIQUE)"

    # Verify only one row exists
    count = db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    assert count == 1, f"Expected exactly 1 entry after duplicate insert, got {count}"

    # Slightly different content should insert fine
    different = "Always use virtual environments for Python projects to isolate dependencies!"
    diff_id = insert_entry(db, content=different, entry_type="raw", embedding=embed(different))
    assert diff_id is not None, "Different content (even by one char) should insert successfully"

    count = db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    assert count == 2, f"Expected 2 entries after inserting different content, got {count}"


# ---------------------------------------------------------------------------
# Test 4: Append-Only Contract — no DELETE or UPDATE on content
# ---------------------------------------------------------------------------


def test_append_only_contract(db):
    """Entries are append-only. Content cannot be deleted or updated via the db module.
    Only distilled_at is allowed to change (via mark_entries_distilled)."""

    content = "Use type hints in Python function signatures for better IDE support and docs."
    embedding = embed(content)
    eid = insert_entry(db, content=content, entry_type="raw", embedding=embedding)
    assert eid is not None

    # The db module does NOT expose delete or update-content functions.
    # Verify the data survives and content is unchanged after marking distilled.
    mark_entries_distilled(db, [eid])

    row = db.execute("SELECT content, distilled_at FROM entries WHERE id = ?", (eid,)).fetchone()
    assert row[0] == content, "Content should be unchanged after mark_entries_distilled"
    assert row[1] is not None, "distilled_at should be set after marking"

    # Direct SQL DELETE should be possible at the sqlite level (we can't prevent it
    # at the engine level), but the db module never exposes it. Verify the entry
    # persists through normal operations.
    undistilled = get_undistilled_entries(db)
    assert all(e[0] != eid for e in undistilled), (
        "A distilled entry should not appear in undistilled results"
    )

    # Entry should still exist in the table
    exists = db.execute("SELECT COUNT(*) FROM entries WHERE id = ?", (eid,)).fetchone()[0]
    assert exists == 1, "Entry should still exist after all operations — append-only"


# ---------------------------------------------------------------------------
# Test 5: Secret Sanitization — real patterns that appear in session logs
# ---------------------------------------------------------------------------


def test_secret_sanitization_real_patterns():
    """Test with REAL secret patterns that could leak through session logs."""

    # --- Secrets that MUST be caught ---

    anthropic_key = "My API key is sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJ"
    result = sanitize(anthropic_key)
    assert "sk-ant-api03" not in result, "Anthropic API key should be redacted"
    assert "REDACTED" in result, "Redacted text should contain REDACTED marker"

    github_pat = "export GITHUB_TOKEN=ghp_ABCDEFghijklmnopqrstuvwxyz0123456789ab"
    result = sanitize(github_pat)
    assert "ghp_" not in result, "GitHub PAT should be redacted"

    conn_string = "postgres://admin:secretpass123@db.example.com:5432/mydb"
    result = sanitize(conn_string)
    assert "secretpass123" not in result, "Connection string password should be redacted"

    private_key = """-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF7RO8yfNchZPmE2k+PLACEHOLDER
-----END RSA PRIVATE KEY-----"""
    result = sanitize(private_key)
    assert "BEGIN RSA PRIVATE KEY" not in result, "Private key block should be redacted"

    # Multiple secrets in one block
    multi = "Key: sk-ant-api03-xyz123abcdef456ghijklmno and token ghp_ABCDEFghijklmnopqrstuvwxyz0123456789ab"
    result = sanitize(multi)
    assert "sk-ant-" not in result, "All secrets should be caught in multi-secret text"
    assert "ghp_" not in result, "All secrets should be caught in multi-secret text"

    # has_secrets should agree
    assert has_secrets(anthropic_key), "has_secrets should detect Anthropic key"
    assert has_secrets(github_pat), "has_secrets should detect GitHub PAT"
    assert has_secrets(conn_string), "has_secrets should detect connection strings"

    # --- Non-secrets that should NOT be redacted ---

    normal_code = "def my_long_variable_name_for_config(): return 42"
    result = sanitize(normal_code)
    assert "my_long_variable_name_for_config" in result, (
        "Normal function names should not be redacted"
    )

    short_token = "session_id = 'abc123'"
    result = sanitize(short_token)
    assert "abc123" in result, "Short non-secret strings should not be redacted"


# ---------------------------------------------------------------------------
# Test 6: Ingest Idempotency — re-ingesting the same file doesn't duplicate
# ---------------------------------------------------------------------------


def test_ingest_idempotency(db, tmp_path):
    """Ingest is idempotent: running it twice on the same file yields the same entries.
    Adding new lines and re-ingesting only adds the new ones."""

    history_path = tmp_path / "history.jsonl"
    state_path = tmp_path / "ingest_state.json"

    # Create 5 fake history entries (display must be >= 10 chars)
    original_lines = []
    for i in range(5):
        entry = {
            "display": f"This is session log entry number {i} with enough content to be ingested",
            "sessionId": f"session-{i}",
            "project": "test-project",
        }
        original_lines.append(json.dumps(entry))

    history_path.write_text("\n".join(original_lines) + "\n")

    # First ingest
    stats1 = ingest_history(db, history_path, state_path)
    assert stats1["ingested"] == 5, f"First ingest should add 5 entries, got {stats1['ingested']}"

    count_after_first = db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    assert count_after_first == 5, f"DB should have 5 entries after first ingest, got {count_after_first}"

    # Second ingest of same file — should add nothing (cursor-based skip)
    stats2 = ingest_history(db, history_path, state_path)
    assert stats2["ingested"] == 0, (
        f"Second ingest of same file should add 0 entries, got {stats2['ingested']}"
    )

    count_after_second = db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    assert count_after_second == 5, (
        f"DB should still have 5 entries after re-ingest, got {count_after_second}"
    )

    # Append 3 new lines to the file
    new_lines = []
    for i in range(5, 8):
        entry = {
            "display": f"This is session log entry number {i} with enough content to be ingested",
            "sessionId": f"session-{i}",
            "project": "test-project",
        }
        new_lines.append(json.dumps(entry))

    with open(history_path, "a", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    # Third ingest — should only add the 3 new lines
    stats3 = ingest_history(db, history_path, state_path)
    assert stats3["ingested"] == 3, (
        f"Third ingest should add only 3 new entries, got {stats3['ingested']}"
    )

    count_after_third = db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    assert count_after_third == 8, (
        f"DB should have 8 entries after appending 3 new lines, got {count_after_third}"
    )
