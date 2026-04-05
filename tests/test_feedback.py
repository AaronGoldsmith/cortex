"""Tests for the cortex feedback feature.

Covers: record_feedback, adjust_confidence_from_feedback, get_feedback_stats,
CLI parsing, and query output hint.
"""

import pytest

from cortex.db import (
    adjust_confidence_from_feedback,
    get_connection,
    get_feedback_stats,
    init_db,
    insert_distillation,
    insert_entry,
    record_feedback,
)
from cortex.embedder import embed
from cortex.query import format_results, query


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
def sample_entry(db):
    """Insert a sample entry and return its id."""
    eid = insert_entry(
        db,
        content="Use pdb.set_trace() for debugging Python code.",
        entry_type="observation",
        embedding=embed("Use pdb.set_trace() for debugging Python code."),
    )
    return eid


@pytest.fixture
def sample_distillation(db):
    """Insert a sample distillation and return its id."""
    did = insert_distillation(
        db,
        content="Python debugging best practices include pdb and breakpoint().",
        pattern_type="workflow",
        confidence=0.9,
        embedding=embed("Python debugging best practices include pdb and breakpoint()."),
    )
    return did


# ---------------------------------------------------------------------------
# Test record_feedback
# ---------------------------------------------------------------------------


def test_record_feedback_entry(db, sample_entry):
    """Recording feedback for an entry succeeds."""
    fid = record_feedback(db, entry_id=sample_entry, helpful=True, context="Useful tip")
    assert fid is not None

    row = db.execute("SELECT * FROM feedback WHERE id = ?", (fid,)).fetchone()
    assert row is not None
    assert row["entry_id"] == sample_entry
    assert row["distillation_id"] is None
    assert row["helpful"] == 1
    assert row["context"] == "Useful tip"


def test_record_feedback_distillation(db, sample_distillation):
    """Recording feedback for a distillation succeeds."""
    fid = record_feedback(db, distillation_id=sample_distillation, helpful=False, context="Outdated")
    assert fid is not None

    row = db.execute("SELECT * FROM feedback WHERE id = ?", (fid,)).fetchone()
    assert row["distillation_id"] == sample_distillation
    assert row["helpful"] == 0


def test_record_feedback_no_id_raises(db):
    """Must provide either entry_id or distillation_id."""
    with pytest.raises(ValueError, match="Must provide either"):
        record_feedback(db)


def test_record_feedback_nonexistent_entry(db):
    """Feedback for a nonexistent entry raises."""
    with pytest.raises(ValueError, match="Entry 9999 not found"):
        record_feedback(db, entry_id=9999, helpful=True)


def test_record_feedback_nonexistent_distillation(db):
    """Feedback for a nonexistent distillation raises."""
    with pytest.raises(ValueError, match="Distillation 9999 not found"):
        record_feedback(db, distillation_id=9999, helpful=True)


def test_record_feedback_duplicate_rejected(db, sample_entry):
    """Same (entry_id, context) combo is rejected."""
    record_feedback(db, entry_id=sample_entry, helpful=True, context="Good")
    with pytest.raises(ValueError, match="Duplicate feedback"):
        record_feedback(db, entry_id=sample_entry, helpful=False, context="Good")


def test_record_feedback_different_context_allowed(db, sample_entry):
    """Same entry but different context is allowed."""
    record_feedback(db, entry_id=sample_entry, helpful=True, context="Good for X")
    fid2 = record_feedback(db, entry_id=sample_entry, helpful=False, context="Bad for Y")
    assert fid2 is not None


# ---------------------------------------------------------------------------
# Test adjust_confidence_from_feedback
# ---------------------------------------------------------------------------


def test_confidence_bump_on_helpful(db, sample_entry):
    """Helpful feedback bumps confidence by 0.05."""
    original = db.execute("SELECT confidence FROM entries WHERE id = ?", (sample_entry,)).fetchone()[0]
    new_conf = adjust_confidence_from_feedback(db, entry_id=sample_entry, helpful=True)
    assert new_conf == round(original + 0.05, 4)


def test_confidence_drop_on_unhelpful(db, sample_entry):
    """Unhelpful feedback drops confidence by 0.1."""
    original = db.execute("SELECT confidence FROM entries WHERE id = ?", (sample_entry,)).fetchone()[0]
    new_conf = adjust_confidence_from_feedback(db, entry_id=sample_entry, helpful=False)
    assert new_conf == round(original - 0.1, 4)


def test_confidence_capped_at_2(db, sample_entry):
    """Confidence cannot exceed 2.0."""
    db.execute("UPDATE entries SET confidence = 1.98 WHERE id = ?", (sample_entry,))
    db.commit()
    new_conf = adjust_confidence_from_feedback(db, entry_id=sample_entry, helpful=True)
    assert new_conf == 2.0


def test_confidence_floored_at_01(db, sample_entry):
    """Confidence cannot drop below 0.1."""
    db.execute("UPDATE entries SET confidence = 0.15 WHERE id = ?", (sample_entry,))
    db.commit()
    new_conf = adjust_confidence_from_feedback(db, entry_id=sample_entry, helpful=False)
    assert new_conf == 0.1


def test_confidence_distillation(db, sample_distillation):
    """Confidence adjustment works for distillations too."""
    new_conf = adjust_confidence_from_feedback(db, distillation_id=sample_distillation, helpful=True)
    assert new_conf == round(0.9 + 0.05, 4)


# ---------------------------------------------------------------------------
# Test get_feedback_stats
# ---------------------------------------------------------------------------


def test_feedback_stats_empty(db):
    """Stats on empty feedback table."""
    stats = get_feedback_stats(db)
    assert stats["total"] == 0
    assert stats["helpful_rate"] == 0.0


def test_feedback_stats_with_data(db, sample_entry, sample_distillation):
    """Stats aggregate correctly."""
    record_feedback(db, entry_id=sample_entry, helpful=True, context="a")
    record_feedback(db, entry_id=sample_entry, helpful=True, context="b")
    record_feedback(db, distillation_id=sample_distillation, helpful=False, context="c")

    stats = get_feedback_stats(db)
    assert stats["total"] == 3
    assert stats["helpful_count"] == 2
    assert stats["unhelpful_count"] == 1
    assert stats["helpful_rate"] == round(2 / 3 * 100, 1)
    assert len(stats["top_helpful"]) >= 1
    assert len(stats["top_unhelpful"]) >= 1


# ---------------------------------------------------------------------------
# Test query output hint
# ---------------------------------------------------------------------------


def test_format_results_includes_feedback_hint(db):
    """Query output includes the feedback tip line."""
    # Insert an entry so we get results
    content = "Use virtual environments for Python projects."
    insert_entry(db, content=content, entry_type="raw", embedding=embed(content))

    results = query(db, "Python virtual environments", top_k=3)
    output = format_results(results)
    assert "cortex feedback" in output
    assert "yes/no" in output


def test_format_results_empty_no_hint():
    """Empty results don't show the hint."""
    output = format_results([])
    assert output == "No knowledge found."
