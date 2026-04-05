"""Cortex query — semantic search across the knowledge ledger."""

import logging
import math
from datetime import datetime, timezone

from cortex.config import (
    CONFIDENCE_WEIGHT,
    DEFAULT_TOP_K,
    DISTILLATION_BOOST,
    PROJECTS_DIR,
    RECENCY_WEIGHT,
    SIMILARITY_WEIGHT,
)
from cortex.db import vector_search
from cortex.embedder import embed_query

log = logging.getLogger(__name__)


def query(conn, text, top_k=DEFAULT_TOP_K, project_filter=None):
    """Search the ledger for entries and distillations relevant to the query text.

    Returns a ranked list of results combining semantic similarity,
    confidence, and recency.
    """
    query_embedding = embed_query(text)

    # Search both entries and distillations
    entry_hits = vector_search(conn, query_embedding, "entry_vec", top_k * 2)
    distill_hits = vector_search(conn, query_embedding, "distill_vec", top_k * 2)

    results = []

    # Fetch entry details
    for entry_id, distance in entry_hits:
        row = conn.execute(
            "SELECT id, content, entry_type, source_model, source_project, "
            "confidence, created_at FROM entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        if row and (project_filter is None or row[4] == project_filter):
            results.append(_score_result(row, distance, kind="entry"))

    # Fetch distillation details with source context
    for dist_id, distance in distill_hits:
        row = conn.execute(
            "SELECT id, content, pattern_type, source_model, NULL, "
            "confidence, created_at, context_window FROM distillations WHERE id = ?",
            (dist_id,),
        ).fetchone()
        if row:
            result = _score_result(row, distance, kind="distillation")
            result["source_context"] = _get_source_context(conn, dist_id, row[7] or 0)
            results.append(result)

    # Sort by combined score descending
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def _score_result(row, distance, kind):
    """Compute a combined score from similarity, confidence, and recency."""
    # sqlite-vec returns L2 distance; convert to cosine similarity for unit vectors:
    # cosine_sim = 1 - (L2_dist² / 2)
    similarity = max(0.0, 1.0 - (distance ** 2) / 2.0)
    confidence = row[5] or 1.0

    # Recency: exponential decay, half-life of 30 days
    created_at = datetime.fromisoformat(row[6]) if row[6] else datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - created_at).days
    recency = math.exp(-0.693 * age_days / 30)  # 0.693 = ln(2)

    score = (
        SIMILARITY_WEIGHT * similarity
        + CONFIDENCE_WEIGHT * confidence
        + RECENCY_WEIGHT * recency
        + (DISTILLATION_BOOST if kind == "distillation" else 0.0)
    )

    return {
        "id": row[0],
        "content": row[1],
        "type": row[2],
        "source_model": row[3],
        "source_project": row[4],
        "confidence": confidence,
        "created_at": row[6],
        "kind": kind,
        "similarity": round(similarity, 4),
        "recency": round(recency, 4),
        "score": round(score, 4),
    }


def _get_source_context(conn, distillation_id: int, context_window: int) -> list[dict]:
    """Follow lineage from a distillation to its source entries' conversation context.

    Returns a list of dicts with keys: entry_id, user_text, assistant_text.
    Only includes entries where we can resolve the session JSONL.
    """
    from cortex.sessions import find_session_file, read_session_turns

    rows = conn.execute(
        "SELECT e.id, e.content, e.session_id, e.turn_index "
        "FROM lineage l JOIN entries e ON l.entry_id = e.id "
        "WHERE l.distillation_id = ?",
        (distillation_id,),
    ).fetchall()

    contexts = []
    for row in rows:
        entry_id, content, session_id, turn_index = row[0], row[1], row[2], row[3]
        ctx = {"entry_id": entry_id, "user_text": content, "assistant_text": None}

        if session_id and turn_index is not None:
            try:
                session_path = find_session_file(session_id, PROJECTS_DIR)
                if session_path:
                    turns = read_session_turns(session_path)
                    for turn in turns:
                        if turn.index == turn_index:
                            ctx["user_text"] = turn.user_text
                            ctx["assistant_text"] = turn.assistant_text
                            break
            except Exception:
                pass  # Graceful degradation

        contexts.append(ctx)
    return contexts


def format_results(results):
    """Format query results for CLI display."""
    if not results:
        return "No knowledge found."

    lines = []
    for i, r in enumerate(results, 1):
        prefix = "D" if r["kind"] == "distillation" else "E"
        source = r["source_project"] or r["source_model"] or "unknown"
        lines.append(
            f"  [{prefix}{r['id']}] (score: {r['score']:.2f}, "
            f"conf: {r['confidence']:.1f}) {source}"
        )
        # Truncate long content for display
        content = r["content"]
        if len(content) > 200:
            content = content[:197] + "..."
        lines.append(f"    {content}")

        # Show source conversation context for distillations
        source_ctx = r.get("source_context", [])
        if source_ctx:
            for ctx in source_ctx[:2]:  # Show at most 2 source turns
                user = ctx["user_text"]
                if len(user) > 120:
                    user = user[:117] + "..."
                lines.append(f"    Source:")
                lines.append(f"      USER: {user}")
                if ctx.get("assistant_text"):
                    asst = ctx["assistant_text"]
                    if len(asst) > 200:
                        asst = asst[:197] + "..."
                    lines.append(f"      ASSISTANT: {asst}")

        lines.append("")

    lines.append("Tip: Run 'cortex feedback <id> yes/no' to rate these results")
    return "\n".join(lines)
