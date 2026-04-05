"""Tests for the Cortex eval framework."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cortex.eval import (
    CaseResult,
    EvalCase,
    EvalReport,
    _case_key,
    _content_matches_any,
    _eval_single,
    _parse_adversarial_response,
    _parse_judge_response,
    llm_judge_eval,
    load_case_history,
    load_eval_cases,
    retire_stale_cases,
    run_eval,
    save_case_history,
    save_eval_cases,
    update_case_history,
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


# --- Test LLM judge response parsing ---


def test_parse_judge_response_valid():
    """Parses well-formed LLM judge JSON correctly."""
    response = json.dumps({
        "scores": [
            {"result_index": 0, "relevance": 4, "faithfulness": 5, "reasoning": "Directly relevant"},
            {"result_index": 1, "relevance": 2, "faithfulness": 3, "reasoning": "Partially related"},
        ]
    })
    scores = _parse_judge_response(response)
    assert len(scores) == 2
    assert scores[0]["relevance"] == 4
    assert scores[0]["faithfulness"] == 5
    assert scores[1]["relevance"] == 2
    assert scores[1]["faithfulness"] == 3


def test_parse_judge_response_with_surrounding_text():
    """Parses JSON even when wrapped in extra text."""
    response = 'Here is the scoring:\n{"scores": [{"result_index": 0, "relevance": 3, "faithfulness": 4, "reasoning": "ok"}]}\nDone.'
    scores = _parse_judge_response(response)
    assert len(scores) == 1
    assert scores[0]["relevance"] == 3


def test_parse_judge_response_clamps_values():
    """Values outside 1-5 are clamped."""
    response = json.dumps({
        "scores": [
            {"result_index": 0, "relevance": 0, "faithfulness": 7, "reasoning": ""},
        ]
    })
    scores = _parse_judge_response(response)
    assert scores[0]["relevance"] == 1
    assert scores[0]["faithfulness"] == 5


def test_parse_judge_response_invalid_json():
    """Returns empty list on garbage input."""
    assert _parse_judge_response("not json at all") == []


def test_parse_judge_response_missing_fields():
    """Skips entries missing relevance or faithfulness."""
    response = json.dumps({
        "scores": [
            {"result_index": 0, "relevance": 4},  # no faithfulness
            {"result_index": 1, "relevance": 3, "faithfulness": 5, "reasoning": "ok"},
        ]
    })
    scores = _parse_judge_response(response)
    assert len(scores) == 1
    assert scores[0]["relevance"] == 3


# --- Test llm_judge_eval ---


def test_llm_judge_eval_with_mock():
    """llm_judge_eval returns averaged scores from mock LLM call."""
    mock_response = json.dumps({
        "scores": [
            {"result_index": 0, "relevance": 5, "faithfulness": 4, "reasoning": "good"},
            {"result_index": 1, "relevance": 3, "faithfulness": 4, "reasoning": "ok"},
        ]
    })
    results = _make_results(["result one", "result two"])

    rel, faith = llm_judge_eval("test query", results, llm_call=lambda _: mock_response)

    assert rel == pytest.approx(4.0)
    assert faith == pytest.approx(4.0)


def test_llm_judge_eval_empty_results():
    """Returns None, None for empty results."""
    rel, faith = llm_judge_eval("test", [], llm_call=lambda _: "")
    assert rel is None
    assert faith is None


def test_llm_judge_eval_handles_failure():
    """Returns None, None when LLM call raises."""
    def failing_call(prompt):
        raise RuntimeError("LLM down")

    rel, faith = llm_judge_eval("test", _make_results(["x"]), llm_call=failing_call)
    assert rel is None
    assert faith is None


def test_llm_judge_eval_includes_answer_in_prompt():
    """When answer is provided, it appears in the prompt sent to LLM."""
    captured = {}
    def capture_call(prompt):
        captured["prompt"] = prompt
        return json.dumps({"scores": [{"result_index": 0, "relevance": 4, "faithfulness": 5, "reasoning": ""}]})

    llm_judge_eval("query", _make_results(["x"]), answer="the real answer", llm_call=capture_call)
    assert "the real answer" in captured["prompt"]


# --- Test run_eval with LLM judge ---


def test_run_eval_with_llm_judge(sample_cases):
    """run_eval with llm_judge=True populates LLM scores on case results."""
    mock_results = _make_results([
        "Always use pytest for testing Python applications",
        "Testing with pytest fixtures simplifies setup",
    ])

    mock_judge_response = json.dumps({
        "scores": [
            {"result_index": 0, "relevance": 5, "faithfulness": 5, "reasoning": "perfect"},
            {"result_index": 1, "relevance": 4, "faithfulness": 4, "reasoning": "good"},
        ]
    })

    with patch("cortex.query.query", return_value=mock_results):
        report = run_eval(
            "fake_conn", sample_cases, top_k=2,
            llm_judge=True, llm_call=lambda _: mock_judge_response,
        )

    assert report.avg_llm_relevance is not None
    assert report.avg_llm_faithfulness is not None
    assert report.avg_llm_relevance == pytest.approx(4.5)
    assert report.avg_llm_faithfulness == pytest.approx(4.5)
    # Each case result should have LLM scores
    for cr in report.case_results:
        assert cr.llm_relevance is not None
        assert cr.llm_faithfulness is not None


def test_run_eval_without_llm_judge_has_no_llm_scores(sample_cases):
    """run_eval without llm_judge leaves LLM fields as None."""
    mock_results = _make_results(["some content"])

    with patch("cortex.query.query", return_value=mock_results):
        report = run_eval("fake_conn", sample_cases, top_k=1)

    assert report.avg_llm_relevance is None
    assert report.avg_llm_faithfulness is None
    for cr in report.case_results:
        assert cr.llm_relevance is None
        assert cr.llm_faithfulness is None


def test_report_summary_includes_llm_scores():
    """Summary output includes LLM scores when present."""
    cr = CaseResult(
        case=EvalCase(query="q", expected_keywords=["k"], anti_keywords=[], description="test"),
        relevance_hit=True, precision=0.8, noise=0.0,
        top_result_preview="preview",
        llm_relevance=4.2, llm_faithfulness=3.8,
    )
    report = EvalReport(
        timestamp="2025-06-01 12:00:00 UTC",
        total_cases=1,
        avg_relevance=1.0, avg_precision=0.8, avg_noise=0.0,
        case_results=[cr],
        avg_llm_relevance=4.2, avg_llm_faithfulness=3.8,
    )
    summary = report.summary()
    assert "LLM Relevance:    4.20/5" in summary
    assert "LLM Faithfulness: 3.80/5" in summary
    assert "llm_rel=4.2" in summary
    assert "llm_faith=3.8" in summary


# --- Test case key derivation ---


def test_case_key_uses_description():
    case = EvalCase(
        query="some query",
        expected_keywords=["k"],
        anti_keywords=[],
        description="My unique case",
    )
    assert _case_key(case) == "My unique case"


def test_case_key_falls_back_to_query():
    case = EvalCase(
        query="fallback query",
        expected_keywords=["k"],
        anti_keywords=[],
        description="",
    )
    assert _case_key(case) == "fallback query"


# --- Test case history tracking ---


def test_load_save_case_history(tmp_path):
    path = tmp_path / "history.json"
    assert load_case_history(path) == {}

    history = {"case1": {"consecutive_passes": 3, "total_runs": 5, "last_run": "2026-04-01"}}
    save_case_history(path, history)

    loaded = load_case_history(path)
    assert loaded["case1"]["consecutive_passes"] == 3
    assert loaded["case1"]["total_runs"] == 5


def test_update_case_history_increments_on_pass():
    case = EvalCase(
        query="q", expected_keywords=["k"], anti_keywords=[], description="pass case"
    )
    report = EvalReport(
        timestamp="2026-04-05",
        total_cases=1,
        avg_relevance=1.0,
        avg_precision=0.8,
        avg_noise=0.0,
        case_results=[
            CaseResult(case=case, relevance_hit=True, precision=0.8, noise=0.0, top_result_preview="ok"),
        ],
    )
    history = {}
    update_case_history(history, report)
    assert history["pass case"]["consecutive_passes"] == 1
    assert history["pass case"]["total_runs"] == 1

    # Run again — should increment
    update_case_history(history, report)
    assert history["pass case"]["consecutive_passes"] == 2
    assert history["pass case"]["total_runs"] == 2


def test_update_case_history_resets_on_fail():
    case = EvalCase(
        query="q", expected_keywords=["k"], anti_keywords=[], description="fail case"
    )
    history = {"fail case": {"consecutive_passes": 5, "total_runs": 10, "last_run": "2026-04-01"}}

    # Failing report (relevance_hit=False)
    report = EvalReport(
        timestamp="2026-04-05",
        total_cases=1,
        avg_relevance=0.0,
        avg_precision=0.0,
        avg_noise=0.0,
        case_results=[
            CaseResult(case=case, relevance_hit=False, precision=0.0, noise=0.5, top_result_preview="bad"),
        ],
    )
    update_case_history(history, report)
    assert history["fail case"]["consecutive_passes"] == 0
    assert history["fail case"]["total_runs"] == 11


def test_update_case_history_resets_on_low_precision():
    """A case with relevance_hit=True but precision < 0.4 should reset."""
    case = EvalCase(
        query="q", expected_keywords=["k"], anti_keywords=[], description="low prec"
    )
    history = {"low prec": {"consecutive_passes": 3, "total_runs": 5, "last_run": "2026-04-01"}}

    report = EvalReport(
        timestamp="2026-04-05",
        total_cases=1,
        avg_relevance=1.0,
        avg_precision=0.2,
        avg_noise=0.0,
        case_results=[
            CaseResult(case=case, relevance_hit=True, precision=0.2, noise=0.0, top_result_preview="meh"),
        ],
    )
    update_case_history(history, report)
    assert history["low prec"]["consecutive_passes"] == 0


# --- Test retirement ---


def test_retire_stale_cases():
    cases = [
        EvalCase(query="q1", expected_keywords=["k"], anti_keywords=[], description="stable"),
        EvalCase(query="q2", expected_keywords=["k"], anti_keywords=[], description="unstable"),
        EvalCase(query="q3", expected_keywords=["k"], anti_keywords=[], description="borderline"),
    ]
    history = {
        "stable": {"consecutive_passes": 12, "total_runs": 20, "last_run": "2026-04-05"},
        "unstable": {"consecutive_passes": 3, "total_runs": 10, "last_run": "2026-04-05"},
        "borderline": {"consecutive_passes": 10, "total_runs": 15, "last_run": "2026-04-05"},
    }

    active, retired = retire_stale_cases(cases, history, threshold=10)

    assert len(retired) == 2  # stable (12) and borderline (10)
    assert len(active) == 1
    assert active[0].description == "unstable"
    retired_descs = {c.description for c in retired}
    assert "stable" in retired_descs
    assert "borderline" in retired_descs


def test_retire_no_history():
    """Cases with no history entry should not be retired."""
    cases = [
        EvalCase(query="q", expected_keywords=["k"], anti_keywords=[], description="new case"),
    ]
    active, retired = retire_stale_cases(cases, {}, threshold=10)
    assert len(active) == 1
    assert len(retired) == 0


# --- Test adversarial case parsing ---


def test_parse_adversarial_response_valid():
    response = """Here are some cases:
[
  {
    "query": "fly.io deploy issues",
    "expected_keywords": ["fly", "deploy"],
    "anti_keywords": ["heroku"],
    "description": "Cross-project fly.io retrieval",
    "answer": "fly deployment issues",
    "query_variants": ["fly deployment problems"]
  },
  {
    "query": "avoid sqlite pitfalls",
    "expected_keywords": ["sqlite", "pitfall"],
    "anti_keywords": [],
    "description": "Negation: sqlite mistakes",
    "answer": "common sqlite mistakes",
    "query_variants": []
  }
]
"""
    cases = _parse_adversarial_response(response)
    assert len(cases) == 2
    assert cases[0].query == "fly.io deploy issues"
    assert cases[0].description == "Cross-project fly.io retrieval"
    assert cases[1].anti_keywords == []


def test_parse_adversarial_response_invalid():
    assert _parse_adversarial_response("not json at all") == []
    assert _parse_adversarial_response("") == []


def test_parse_adversarial_response_partial():
    """Items missing required 'query' field should be skipped."""
    response = '[{"query": "good", "expected_keywords": ["k"]}, {"description": "no query"}]'
    cases = _parse_adversarial_response(response)
    assert len(cases) == 1
    assert cases[0].query == "good"
