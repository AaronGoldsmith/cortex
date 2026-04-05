"""Cortex eval — measure whether knowledge retrieval is improving over time."""

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from cortex.config import CORTEX_DIR

EVAL_CASES_PATH = CORTEX_DIR / "eval_cases.json"
EVAL_HISTORY_PATH = CORTEX_DIR / "eval_history.jsonl"


@dataclass
class EvalCase:
    query: str  # what to search for
    expected_keywords: list[str]  # words/phrases that SHOULD appear in results
    anti_keywords: list[str]  # words/phrases that should NOT dominate results
    description: str  # human-readable description of what this tests
    answer: str = ""  # ground truth answer (for Q&A evals)
    query_variants: list[str] = field(default_factory=list)  # paraphrased alternatives

    def pick_query(self) -> str:
        """Return a randomly selected query variant, or the primary query."""
        if not self.query_variants:
            return self.query
        return random.choice([self.query] + self.query_variants)


@dataclass
class CaseResult:
    case: EvalCase
    relevance_hit: bool  # did any result contain expected keywords?
    precision: float  # what fraction of results were relevant?
    noise: float  # what fraction matched anti-keywords?
    top_result_preview: str  # first result content truncated


@dataclass
class EvalReport:
    timestamp: str
    total_cases: int
    avg_relevance: float  # 0-1, fraction of cases where >=1 expected keyword hit
    avg_precision: float  # 0-1, fraction of results containing expected keywords
    avg_noise: float  # 0-1, fraction of results matching anti-keywords (lower=better)
    case_results: list[CaseResult] = field(default_factory=list)

    def summary(self) -> str:
        """Format as CLI-friendly summary."""
        lines = [
            f"Cortex Eval Report — {self.timestamp}",
            f"  Cases:      {self.total_cases}",
            f"  Relevance:  {self.avg_relevance:.1%}",
            f"  Precision:  {self.avg_precision:.1%}",
            f"  Noise:      {self.avg_noise:.1%}  (lower is better)",
            "",
        ]
        for cr in self.case_results:
            status = "PASS" if cr.relevance_hit else "FAIL"
            lines.append(f"  [{status}] {cr.case.description}")
            lines.append(
                f"         precision={cr.precision:.0%}  noise={cr.noise:.0%}"
            )
            if cr.top_result_preview:
                preview = cr.top_result_preview
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                lines.append(f"         top: {preview}")
            lines.append("")
        return "\n".join(lines)


def load_eval_cases(path: Path) -> list[EvalCase]:
    """Load eval cases from a JSON file."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [
        EvalCase(
            query=c["query"],
            expected_keywords=c["expected_keywords"],
            anti_keywords=c.get("anti_keywords", []),
            description=c.get("description", c["query"]),
            answer=c.get("answer", ""),
            query_variants=c.get("query_variants", []),
        )
        for c in data
    ]


def save_eval_cases(path: Path, cases: list[EvalCase]) -> None:
    """Save eval cases to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(c) for c in cases], f, indent=2)


def _content_matches_any(content: str, keywords: list[str]) -> bool:
    """Check if content contains any of the given keywords (case-insensitive)."""
    content_lower = content.lower()
    return any(kw.lower() in content_lower for kw in keywords)


def _eval_single(results: list[dict], case: EvalCase) -> CaseResult:
    """Score a single eval case against query results."""
    if not results:
        return CaseResult(
            case=case,
            relevance_hit=False,
            precision=0.0,
            noise=0.0,
            top_result_preview="",
        )

    relevant_count = 0
    noisy_count = 0

    for r in results:
        content = r.get("content", "")
        if _content_matches_any(content, case.expected_keywords):
            relevant_count += 1
        if case.anti_keywords and _content_matches_any(content, case.anti_keywords):
            noisy_count += 1

    total = len(results)
    top_preview = results[0].get("content", "")[:120] if results else ""

    return CaseResult(
        case=case,
        relevance_hit=relevant_count > 0,
        precision=relevant_count / total,
        noise=noisy_count / total,
        top_result_preview=top_preview,
    )


def run_eval(conn, cases: list[EvalCase], top_k: int = 5) -> EvalReport:
    """Run all eval cases against current DB state. Return scored report.

    When a case has query_variants, a random variant is selected each run
    so the eval doesn't overfit to a single phrasing.
    """
    from cortex.query import query

    case_results = []
    for case in cases:
        selected_query = case.pick_query()
        results = query(conn, selected_query, top_k=top_k)
        cr = _eval_single(results, case)
        case_results.append(cr)

    total = len(case_results)
    avg_relevance = sum(cr.relevance_hit for cr in case_results) / total if total else 0.0
    avg_precision = sum(cr.precision for cr in case_results) / total if total else 0.0
    avg_noise = sum(cr.noise for cr in case_results) / total if total else 0.0

    return EvalReport(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        total_cases=total,
        avg_relevance=avg_relevance,
        avg_precision=avg_precision,
        avg_noise=avg_noise,
        case_results=case_results,
    )


def _project_variants(name: str) -> list[str]:
    """Generate query paraphrases for a project name."""
    return [
        f"tell me about {name}",
        f"{name} overview",
        f"what's in cortex about {name}?",
        f"{name} details and context",
        f"summarize {name}",
    ]


def _knowledge_variants(topic: str) -> list[str]:
    """Generate query paraphrases for 'what do I know about X?'."""
    return [
        f"tell me about {topic}",
        f"{topic} overview",
        f"summarize {topic} knowledge",
        f"what's in cortex about {topic}?",
        f"{topic} details and context",
    ]


def _avoidance_variants(topic: str) -> list[str]:
    """Generate query paraphrases for 'what should I avoid when X?'."""
    return [
        f"{topic} pitfalls",
        f"common mistakes with {topic}",
        f"warnings about {topic}",
        f"{topic} best practices",
        f"gotchas for {topic}",
    ]


def generate_eval_cases(conn) -> list[EvalCase]:
    """Auto-generate eval cases from existing entries and distillations.

    Examines the topics present in the DB and creates eval cases that
    test whether querying those topics returns relevant results.
    Each case includes query variants so the eval rotates phrasings.
    """
    cases = []

    # Gather distinct projects with entries
    projects = conn.execute(
        "SELECT DISTINCT source_project FROM entries "
        "WHERE source_project IS NOT NULL LIMIT 20"
    ).fetchall()

    for row in projects:
        project = row[0] if isinstance(row, (tuple, list)) else row["source_project"]
        if not project:
            continue
        # Use the project name as a natural query
        project_name = Path(project).name if "/" in project or "\\" in project else project
        cases.append(
            EvalCase(
                query=f"{project_name} project",
                expected_keywords=[project_name.lower()],
                anti_keywords=[],
                description=f"Query for project '{project_name}' returns its entries",
                query_variants=_project_variants(project_name),
            )
        )

    # Gather distinct entry types with sample content
    entry_types = conn.execute(
        "SELECT DISTINCT entry_type FROM entries WHERE entry_type != 'raw' LIMIT 10"
    ).fetchall()

    for row in entry_types:
        etype = row[0] if isinstance(row, (tuple, list)) else row["entry_type"]
        # Grab a sample entry of this type to extract keywords
        sample = conn.execute(
            "SELECT content FROM entries WHERE entry_type = ? LIMIT 1",
            (etype,),
        ).fetchone()
        if sample:
            content = sample[0] if isinstance(sample, (tuple, list)) else sample["content"]
            # Extract first meaningful words as keywords
            words = [w for w in content.split()[:20] if len(w) > 3]
            if words:
                keyword = words[0].strip(".,;:!?\"'").lower()
                cases.append(
                    EvalCase(
                        query=content[:80],
                        expected_keywords=[keyword],
                        anti_keywords=[],
                        description=f"Query from '{etype}' entry returns related content",
                    )
                )

    # Gather distillation pattern types
    patterns = conn.execute(
        "SELECT DISTINCT pattern_type FROM distillations LIMIT 10"
    ).fetchall()

    for row in patterns:
        ptype = row[0] if isinstance(row, (tuple, list)) else row["pattern_type"]
        sample = conn.execute(
            "SELECT content FROM distillations WHERE pattern_type = ? LIMIT 1",
            (ptype,),
        ).fetchone()
        if sample:
            content = sample[0] if isinstance(sample, (tuple, list)) else sample["content"]
            words = [w for w in content.split()[:20] if len(w) > 3]
            if words:
                keyword = words[0].strip(".,;:!?\"'").lower()
                cases.append(
                    EvalCase(
                        query=content[:80],
                        expected_keywords=[keyword],
                        anti_keywords=[],
                        description=f"Query from '{ptype}' distillation returns related content",
                    )
                )

    return cases


def snapshot_eval(report: EvalReport, history_path: Path = EVAL_HISTORY_PATH) -> None:
    """Append an eval report to the history log for tracking over time."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": report.timestamp,
        "total_cases": report.total_cases,
        "avg_relevance": report.avg_relevance,
        "avg_precision": report.avg_precision,
        "avg_noise": report.avg_noise,
    }
    with open(history_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_eval_history(history_path: Path = EVAL_HISTORY_PATH) -> list[dict]:
    """Load eval history for trend comparison."""
    if not history_path.exists():
        return []
    entries = []
    with open(history_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def compare_evals(history: list[dict]) -> str:
    """Format eval history as a trend comparison."""
    if len(history) < 2:
        return "Not enough history to compare (need at least 2 runs)."

    lines = ["Eval History (last 10 runs):", ""]
    lines.append(f"  {'Timestamp':<28} {'Relevance':>10} {'Precision':>10} {'Noise':>10}")
    lines.append(f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 10}")

    for entry in history[-10:]:
        lines.append(
            f"  {entry['timestamp']:<28} "
            f"{entry['avg_relevance']:>9.1%} "
            f"{entry['avg_precision']:>9.1%} "
            f"{entry['avg_noise']:>9.1%}"
        )

    # Compute delta between first and last
    first, last = history[0], history[-1]
    rel_delta = last["avg_relevance"] - first["avg_relevance"]
    prec_delta = last["avg_precision"] - first["avg_precision"]
    noise_delta = last["avg_noise"] - first["avg_noise"]

    lines.append("")
    lines.append(
        f"  Trend: relevance {'+' if rel_delta >= 0 else ''}{rel_delta:.1%}, "
        f"precision {'+' if prec_delta >= 0 else ''}{prec_delta:.1%}, "
        f"noise {'+' if noise_delta >= 0 else ''}{noise_delta:.1%}"
    )
    better = rel_delta > 0 or prec_delta > 0 or noise_delta < 0
    lines.append(f"  Overall: {'IMPROVING' if better else 'DEGRADING or STABLE'}")

    return "\n".join(lines)


def seed_qa_cases(conn) -> list[EvalCase]:
    """Generate Q&A eval cases from memory-sourced entries (high-confidence, curated).

    These are real knowledge assertions: "if someone asks X, Cortex should know Y."
    Uses memory files (confidence > 1.0) as ground truth since they're human-curated.
    Each case includes query_variants so the eval rotates phrasings per run.
    """
    cases = []
    seen_queries = set()

    # Pull curated entries (memory files have confidence 1.2)
    rows = conn.execute(
        "SELECT content, entry_type, source_project FROM entries "
        "WHERE confidence > 1.0 AND length(content) > 30 "
        "ORDER BY created_at DESC LIMIT 30"
    ).fetchall()

    for row in rows:
        content = row[0] if isinstance(row, (tuple, list)) else row["content"]
        entry_type = row[1] if isinstance(row, (tuple, list)) else row["entry_type"]
        project = row[2] if isinstance(row, (tuple, list)) else row["source_project"]

        # Strip [name] prefix if present
        clean = content
        if clean.startswith("[") and "] " in clean:
            clean = clean.split("] ", 1)[1]

        # Generate a natural question from the content
        # Use first sentence or first 80 chars as the "answer"
        answer = clean.split(".")[0].strip() if "." in clean else clean[:80]

        # Build query and variants based on entry type
        topic = _extract_topic(clean)
        if entry_type == "correction":
            query = f"what should I avoid when {topic}?"
            variants = _avoidance_variants(topic)
        elif project:
            query = f"what do I know about {project}?"
            variants = _knowledge_variants(project)
        else:
            query = f"what do I know about {topic}?"
            variants = _knowledge_variants(topic)

        # Expected keywords from the answer
        keywords = [w.lower().strip(".,;:!?\"'()[]") for w in answer.split() if len(w) > 4][:3]
        if not keywords:
            continue

        # Dedup by query text
        if query in seen_queries:
            continue
        seen_queries.add(query)

        cases.append(
            EvalCase(
                query=query,
                expected_keywords=keywords,
                anti_keywords=[],
                description=f"Q&A: {query[:60]}",
                answer=answer,
                query_variants=variants,
            )
        )

    return cases


def _extract_topic(text: str) -> str:
    """Pull a short topic phrase from text for question generation."""
    words = [w for w in text.split()[:8] if len(w) > 3]
    return " ".join(words[:4]).lower().rstrip(".,;:!?")


def backfill_variants(cases: list[EvalCase]) -> int:
    """Add query_variants to existing cases that don't have them.

    Infers the variant template from the query pattern. Returns the
    number of cases that were updated.
    """
    updated = 0
    for case in cases:
        if case.query_variants:
            continue

        q = case.query.lower()

        # Project query pattern: "{name} project"
        if q.endswith(" project"):
            name = case.query.rsplit(" ", 1)[0]
            case.query_variants = _project_variants(name)
            updated += 1

        # Knowledge query pattern: "what do I know about {topic}?"
        elif "what do i know about" in q:
            topic = case.query.split("about", 1)[1].strip().rstrip("?")
            case.query_variants = _knowledge_variants(topic)
            updated += 1

        # Avoidance query pattern: "what should I avoid when {topic}?"
        elif "what should i avoid" in q:
            topic = case.query.split("when", 1)[1].strip().rstrip("?") if "when" in q else case.query
            case.query_variants = _avoidance_variants(topic)
            updated += 1

        # Freeform topic query — generate generic rephrasings
        else:
            topic = case.query.rstrip("?").strip()
            case.query_variants = [
                f"tell me about {topic}",
                f"{topic} explained",
                f"details on {topic}",
                f"context for {topic}",
            ]
            updated += 1

    return updated


# ---------------------------------------------------------------------------
# Context A/B comparison eval
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are a blind quality judge for knowledge distillations. You will see two distillation outputs (A and B) produced from the same source entries. You do NOT know which method produced which output. Score each on three dimensions (1-5):

- Specificity: Does it reference concrete details vs vague generalizations?
- Actionability: Could someone act on this pattern?
- Grounding: Does it feel grounded in a real situation?

Source entries:
{entries}

--- Distillation A ---
{distillation_a}

--- Distillation B ---
{distillation_b}

Respond with ONLY a JSON object:
{{
  "a": {{"specificity": N, "actionability": N, "grounding": N}},
  "b": {{"specificity": N, "actionability": N, "grounding": N}}
}}
"""


def compare_context(conn, sample_size=20, context_window=2, llm_call=None) -> dict:
    """A/B eval comparing distillation quality with vs without conversation context.

    Three tests:
    1. Quality Lift — re-distill entries with and without context, judge both.
    2. Rescue Rate — try distilling previously-skipped entries with context.
    3. Blind Judge — LLM scores each pair on specificity/actionability/grounding.

    Returns dict with quality_lift, rescue, and summary keys.
    """
    from cortex.distill import (
        DISTILL_PROMPT,
        _default_llm_call,
        _get_conversation_context,
        _parse_response,
    )
    from cortex.sanitize import sanitize

    if llm_call is None:
        llm_call = _default_llm_call

    result = {
        "quality_lift": {"sample_size": 0, "pairs": [], "avg_old_score": 0.0, "avg_new_score": 0.0},
        "rescue": {"sample_size": 0, "rescued": 0, "rescue_rate": 0.0, "examples": []},
        "summary": "",
    }

    # ------------------------------------------------------------------
    # Test 1 + 3: Quality Lift + Blind Judge
    # ------------------------------------------------------------------
    distilled_rows = conn.execute(
        "SELECT e.id, e.content, e.session_id, e.turn_index, e.source_project, e.created_at "
        "FROM entries e "
        "WHERE e.distilled_at IS NOT NULL "
        "  AND e.distilled_at NOT LIKE 'skipped:%' "
        "  AND e.turn_index IS NOT NULL "
        "ORDER BY RANDOM() LIMIT ?",
        (sample_size,),
    ).fetchall()

    pairs = []
    for row in distilled_rows:
        eid, content, session_id, turn_index, project, created = (
            row[0], row[1], row[2], row[3], row[4], row[5],
        )

        sanitized = sanitize(content)
        proj_short = str(project or "unknown")
        if "\\" in proj_short or "/" in proj_short:
            proj_short = proj_short.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]

        header = f"[Entry {eid}] (project: {proj_short}, date: {created})"

        # Prompt WITHOUT context
        no_ctx_text = f"{header}: {sanitized}"
        prompt_no_ctx = DISTILL_PROMPT.format(entries=no_ctx_text)

        # Prompt WITH context
        ctx_text_block = _get_conversation_context(session_id, turn_index, context_window)
        if ctx_text_block:
            with_ctx_text = (
                f"{header}:\n"
                f"  Conversation context:\n{ctx_text_block}\n"
                f"  >>> Focal message: {sanitized}"
            )
        else:
            with_ctx_text = no_ctx_text  # fallback — no session file
        prompt_with_ctx = DISTILL_PROMPT.format(entries=with_ctx_text)

        try:
            old_resp = llm_call(prompt_no_ctx)
            new_resp = llm_call(prompt_with_ctx)
        except Exception:
            continue

        # Extract the text of each distillation for judging
        try:
            old_parsed = _parse_response(old_resp)
            new_parsed = _parse_response(new_resp)
        except Exception:
            continue

        def _patterns_text(parsed):
            if isinstance(parsed, dict):
                pats = parsed.get("patterns", [])
            else:
                pats = parsed
            return "\n".join(p.get("content", "") for p in pats if isinstance(p, dict))

        old_text = _patterns_text(old_parsed)
        new_text = _patterns_text(new_parsed)

        if not old_text and not new_text:
            continue

        # Test 3: Blind Judge — randomize which is A vs B
        import random as _rand
        if _rand.random() < 0.5:
            a_text, b_text, a_label, b_label = old_text, new_text, "old", "new"
        else:
            a_text, b_text, a_label, b_label = new_text, old_text, "new", "old"

        judge_prompt = JUDGE_PROMPT.format(
            entries=sanitized[:500],
            distillation_a=a_text,
            distillation_b=b_text,
        )

        scores = {"old": {"specificity": 3, "actionability": 3, "grounding": 3},
                  "new": {"specificity": 3, "actionability": 3, "grounding": 3}}
        try:
            judge_resp = llm_call(judge_prompt)
            judge_data = json.loads(
                judge_resp.strip() if judge_resp.strip().startswith("{")
                else judge_resp[judge_resp.index("{"):judge_resp.rindex("}") + 1]
            )
            scores[a_label] = judge_data.get("a", scores[a_label])
            scores[b_label] = judge_data.get("b", scores[b_label])
        except Exception:
            pass  # keep default scores

        pairs.append({
            "entry_id": eid,
            "old": old_text[:300],
            "new": new_text[:300],
            "scores": scores,
        })

    # Compute averages
    def _avg_total(score_dict):
        return sum(score_dict.values()) / max(len(score_dict), 1)

    if pairs:
        avg_old = sum(_avg_total(p["scores"]["old"]) for p in pairs) / len(pairs)
        avg_new = sum(_avg_total(p["scores"]["new"]) for p in pairs) / len(pairs)
    else:
        avg_old = avg_new = 0.0

    result["quality_lift"] = {
        "sample_size": len(pairs),
        "pairs": pairs,
        "avg_old_score": round(avg_old, 2),
        "avg_new_score": round(avg_new, 2),
    }

    # ------------------------------------------------------------------
    # Test 2: Rescue Rate
    # ------------------------------------------------------------------
    skipped_rows = conn.execute(
        "SELECT e.id, e.content, e.session_id, e.turn_index, e.source_project, e.created_at "
        "FROM entries e "
        "WHERE e.distilled_at LIKE 'skipped:%' "
        "  AND e.turn_index IS NOT NULL "
        "ORDER BY RANDOM() LIMIT ?",
        (sample_size,),
    ).fetchall()

    rescued_examples = []
    for row in skipped_rows:
        eid, content, session_id, turn_index, project, created = (
            row[0], row[1], row[2], row[3], row[4], row[5],
        )

        sanitized = sanitize(content)
        proj_short = str(project or "unknown")
        if "\\" in proj_short or "/" in proj_short:
            proj_short = proj_short.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]

        header = f"[Entry {eid}] (project: {proj_short}, date: {created})"
        ctx_text_block = _get_conversation_context(session_id, turn_index, context_window)
        if ctx_text_block:
            entry_text = (
                f"{header}:\n"
                f"  Conversation context:\n{ctx_text_block}\n"
                f"  >>> Focal message: {sanitized}"
            )
        else:
            entry_text = f"{header}: {sanitized}"

        prompt = DISTILL_PROMPT.format(entries=entry_text)
        try:
            resp = llm_call(prompt)
            parsed = _parse_response(resp)
        except Exception:
            continue

        if isinstance(parsed, dict):
            patterns = parsed.get("patterns", [])
        else:
            patterns = parsed

        if patterns:
            rescued_examples.append({
                "entry_id": eid,
                "content": content[:200],
                "pattern": patterns[0].get("content", "")[:200],
            })

    rescue_total = len(skipped_rows)
    result["rescue"] = {
        "sample_size": rescue_total,
        "rescued": len(rescued_examples),
        "rescue_rate": round(len(rescued_examples) / max(rescue_total, 1), 2),
        "examples": rescued_examples,
    }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    lines = []
    ql = result["quality_lift"]
    rs = result["rescue"]
    lines.append(f"Quality Lift: {ql['sample_size']} pairs evaluated")
    lines.append(f"  Avg score WITHOUT context: {ql['avg_old_score']:.2f}/5")
    lines.append(f"  Avg score WITH context:    {ql['avg_new_score']:.2f}/5")
    if ql["avg_old_score"] > 0:
        delta = ql["avg_new_score"] - ql["avg_old_score"]
        pct = delta / ql["avg_old_score"] * 100
        lines.append(f"  Delta: {'+' if delta >= 0 else ''}{delta:.2f} ({'+' if pct >= 0 else ''}{pct:.1f}%)")
    lines.append(f"Rescue Rate: {rs['rescued']}/{rs['sample_size']} previously-skipped entries produced patterns with context")
    if rs["sample_size"] > 0:
        lines.append(f"  Rate: {rs['rescue_rate']:.0%}")
    result["summary"] = "\n".join(lines)

    return result


def format_context_comparison(result: dict) -> str:
    """Format compare_context() output for CLI display."""
    lines = ["=" * 60, "Context A/B Comparison", "=" * 60, ""]

    # Quality Lift
    ql = result["quality_lift"]
    lines.append(f"TEST 1 & 3: Quality Lift + Blind Judge ({ql['sample_size']} pairs)")
    lines.append("-" * 40)
    lines.append(f"  Avg score WITHOUT context: {ql['avg_old_score']:.2f}/5")
    lines.append(f"  Avg score WITH context:    {ql['avg_new_score']:.2f}/5")
    if ql["avg_old_score"] > 0:
        delta = ql["avg_new_score"] - ql["avg_old_score"]
        pct = delta / ql["avg_old_score"] * 100
        lines.append(f"  Delta: {'+' if delta >= 0 else ''}{delta:.2f} ({'+' if pct >= 0 else ''}{pct:.1f}%)")
    lines.append("")

    for i, pair in enumerate(ql["pairs"][:10]):  # show first 10
        old_s = pair["scores"]["old"]
        new_s = pair["scores"]["new"]
        old_avg = sum(old_s.values()) / max(len(old_s), 1)
        new_avg = sum(new_s.values()) / max(len(new_s), 1)
        marker = "+" if new_avg > old_avg else ("-" if new_avg < old_avg else "=")
        lines.append(f"  [{marker}] Entry {pair['entry_id']}: "
                      f"old={old_avg:.1f} new={new_avg:.1f}")
        lines.append(f"      Old: {pair['old'][:80]}{'...' if len(pair['old']) > 80 else ''}")
        lines.append(f"      New: {pair['new'][:80]}{'...' if len(pair['new']) > 80 else ''}")
        lines.append("")

    # Rescue Rate
    rs = result["rescue"]
    lines.append(f"TEST 2: Rescue Rate ({rs['sample_size']} skipped entries)")
    lines.append("-" * 40)
    lines.append(f"  Rescued: {rs['rescued']}/{rs['sample_size']}")
    if rs["sample_size"] > 0:
        lines.append(f"  Rate: {rs['rescue_rate']:.0%}")
    lines.append("")

    for ex in rs["examples"][:5]:  # show first 5
        lines.append(f"  Entry {ex['entry_id']}:")
        lines.append(f"    Source:  {ex['content'][:80]}{'...' if len(ex['content']) > 80 else ''}")
        lines.append(f"    Pattern: {ex['pattern'][:80]}{'...' if len(ex['pattern']) > 80 else ''}")
        lines.append("")

    # Summary
    lines.append("=" * 60)
    lines.append("SUMMARY")
    lines.append("=" * 60)
    lines.append(result["summary"])

    return "\n".join(lines)
