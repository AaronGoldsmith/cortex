"""Tests for the Cortex eval framework."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cortex.eval import (
    CaseResult,
    EvalCase,
    EvalReport,
    _content_matches_any,
    _eval_single,
    load_eval_cases,
    run_eval,
    save_eval_cases,
)


# --- Fixtures ---


@pytest.fixture
def sample_cases():
    return [
        EvalCase(
            query="python testing patterns",
            expected_keywords=["pytest", "testing", "test"],
            anti_keywords=["javascript", "npm"],
            description="Query about Python testing returns relevant results",
        ),
        EvalCase(
            query="database migrations",
            expected_keywords=["migration", "schema", "sqlite"],
            anti_keywords=["frontend", "css"],
            description="Query about DB migrations returns DB-related content",
        ),
    ]


def _make_results(contents: list[str]) -> list[dict]:
    """Helper to build fake query results from content strings."""
    return [
        {
            "id": i,
            "content": c,
            "type": "observation",
            "source_model": "test",
            "source_project": None,
            "confidence": 1.0,
            "created_at": "2025-01-01",
            "kind": "entry",
            "similarity": 0.9,
            "recency": 0.8,
            "score": 0.85,
        }
        for i, c in enumerate(contents, 1)
    ]


# --- Test keyword matching ---


def test_content_matches_any_positive():
    assert _content_matches_any("Use pytest for testing Python code", ["pytest"])


def test_content_matches_any_case_insensitive():
    assert _content_matches_any("Run PYTEST with verbose flag", ["pytest"])


def test_content_matches_any_negative():
    assert not _content_matches_any("Use jest for testing JavaScript", ["pytest"])


def test_content_matches_any_empty_keywords():
    assert not _content_matches_any("anything here", [])


# --- Test eval scoring for relevant results ---


def test_eval_scores_relevant_results():
    """Eval correctly scores a case where results ARE relevant."""
    case = EvalCase(
        query="pytest fixtures",
        expected_keywords=["pytest", "fixture"],
        anti_keywords=["java"],
        description="Test pytest query",
    )
    results = _make_results([
        "Use pytest fixtures to set up test state",
        "The pytest fixture scope can be session or function",
        "Unrelated entry about cooking recipes",
        "Another pytest pattern for parametrize",
        "Entry about Java Spring Boot",
    ])

    cr = _eval_single(results, case)

    assert cr.relevance_hit is True
    assert cr.precision == pytest.approx(3 / 5)  # 3 of 5 mention pytest/fixture
    assert cr.noise == pytest.approx(1 / 5)  # 1 of 5 mentions java
    assert "pytest" in cr.top_result_preview.lower()


# --- Test eval detects noise/irrelevant results ---


def test_eval_detects_noise():
    """Eval correctly detects noise when results are irrelevant."""
    case = EvalCase(
        query="sqlite schema design",
        expected_keywords=["sqlite", "schema"],
        anti_keywords=["redis", "mongodb"],
        description="Test DB schema query",
    )
    results = _make_results([
        "Redis is great for caching sessions",
        "MongoDB aggregation pipeline tips",
        "How to configure Redis cluster",
        "Use MongoDB atlas for cloud hosting",
        "Redis pub/sub for real-time events",
    ])

    cr = _eval_single(results, case)

    assert cr.relevance_hit is False
    assert cr.precision == 0.0
    assert cr.noise == pytest.approx(5 / 5)  # all results match anti-keywords


def test_eval_empty_results():
    """Eval handles empty result set gracefully."""
    case = EvalCase(
        query="obscure topic",
        expected_keywords=["obscure"],
        anti_keywords=[],
        description="Empty results case",
    )
    cr = _eval_single([], case)

    assert cr.relevance_hit is False
    assert cr.precision == 0.0
    assert cr.noise == 0.0
    assert cr.top_result_preview == ""


# --- Test report summary formatting ---


def test_report_summary_format():
    """Eval report summary formats correctly for CLI display."""
    case_results = [
        CaseResult(
            case=EvalCase(
                query="q1",
                expected_keywords=["k1"],
                anti_keywords=[],
                description="First test case",
            ),
            relevance_hit=True,
            precision=0.8,
            noise=0.0,
            top_result_preview="Some relevant content about k1",
        ),
        CaseResult(
            case=EvalCase(
                query="q2",
                expected_keywords=["k2"],
                anti_keywords=["bad"],
                description="Second test case",
            ),
            relevance_hit=False,
            precision=0.0,
            noise=0.6,
            top_result_preview="Irrelevant content",
        ),
    ]

    report = EvalReport(
        timestamp="2025-06-01 12:00:00 UTC",
        total_cases=2,
        avg_relevance=0.5,
        avg_precision=0.4,
        avg_noise=0.3,
        case_results=case_results,
    )

    summary = report.summary()

    assert "Cortex Eval Report" in summary
    assert "2025-06-01" in summary
    assert "Cases:      2" in summary
    assert "50.0%" in summary  # relevance
    assert "40.0%" in summary  # precision
    assert "30.0%" in summary  # noise
    assert "[PASS] First test case" in summary
    assert "[FAIL] Second test case" in summary
    assert "lower is better" in summary


# --- Test load/save eval cases ---


def test_save_and_load_eval_cases(tmp_path):
    """Eval cases round-trip through JSON correctly."""
    cases = [
        EvalCase(
            query="test query",
            expected_keywords=["expected", "words"],
            anti_keywords=["unwanted"],
            description="A test eval case",
        ),
        EvalCase(
            query="another query",
            expected_keywords=["second"],
            anti_keywords=[],
            description="Second case",
        ),
    ]

    path = tmp_path / "eval_cases.json"
    save_eval_cases(path, cases)

    # Verify the file is valid JSON
    with open(path) as f:
        raw = json.load(f)
    assert len(raw) == 2
    assert raw[0]["query"] == "test query"
    assert raw[0]["expected_keywords"] == ["expected", "words"]

    # Load back
    loaded = load_eval_cases(path)
    assert len(loaded) == 2
    assert loaded[0].query == "test query"
    assert loaded[0].expected_keywords == ["expected", "words"]
    assert loaded[0].anti_keywords == ["unwanted"]
    assert loaded[0].description == "A test eval case"
    assert loaded[1].anti_keywords == []


def test_load_eval_cases_missing_file(tmp_path):
    """Loading from a nonexistent file returns empty list."""
    cases = load_eval_cases(tmp_path / "does_not_exist.json")
    assert cases == []


# --- Test run_eval integration ---


def test_run_eval_produces_report(sample_cases):
    """run_eval produces a complete EvalReport with correct structure."""
    mock_results = _make_results([
        "Always use pytest for testing Python applications",
        "Testing with pytest fixtures simplifies setup",
        "JavaScript npm test runner comparison",
    ])

    with patch("cortex.query.query", return_value=mock_results) as mock_query:
        # Use a sentinel for conn since we're mocking query
        report = run_eval("fake_conn", sample_cases, top_k=3)

    assert report.total_cases == 2
    assert len(report.case_results) == 2
    assert 0.0 <= report.avg_relevance <= 1.0
    assert 0.0 <= report.avg_precision <= 1.0
    assert 0.0 <= report.avg_noise <= 1.0
    assert "UTC" in report.timestamp
    # query was called once per case
    assert mock_query.call_count == 2
