"""Cortex eval — measure whether knowledge retrieval is improving over time."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from cortex.config import CORTEX_DIR

EVAL_CASES_PATH = CORTEX_DIR / "eval_cases.json"


@dataclass
class EvalCase:
    query: str  # what to search for
    expected_keywords: list[str]  # words/phrases that SHOULD appear in results
    anti_keywords: list[str]  # words/phrases that should NOT dominate results
    description: str  # human-readable description of what this tests


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
    """Run all eval cases against current DB state. Return scored report."""
    from cortex.query import query

    case_results = []
    for case in cases:
        results = query(conn, case.query, top_k=top_k)
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


def generate_eval_cases(conn) -> list[EvalCase]:
    """Auto-generate eval cases from existing entries and distillations.

    Examines the topics present in the DB and creates eval cases that
    test whether querying those topics returns relevant results.
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
