"""Cortex query — semantic search across the knowledge ledger."""

import math
from datetime import datetime, timezone

from cortex.config import (
    CONFIDENCE_WEIGHT,
    DEFAULT_TOP_K,
    RECENCY_WEIGHT,
    SIMILARITY_WEIGHT,
)
from cortex.db import vector_search
from cortex.embedder import embed_query


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

    # Fetch distillation details
    for dist_id, distance in distill_hits:
        row = conn.execute(
            "SELECT id, content, pattern_type, source_model, NULL, "
            "confidence, created_at FROM distillations WHERE id = ?",
            (dist_id,),
        ).fetchone()
        if row:
            results.append(_score_result(row, distance, kind="distillation"))

    # Sort by combined score descending
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def _score_result(row, distance, kind):
    """Compute a combined score from similarity, confidence, and recency."""
    similarity = max(0.0, 1.0 - distance)  # sqlite-vec returns L2 distance
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
        lines.append("")

    return "\n".join(lines)
