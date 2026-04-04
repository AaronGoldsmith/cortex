"""Cortex distiller — extract patterns from raw entries using LLM."""

import json
import logging
import os

from cortex.db import get_undistilled_entries, insert_distillation, mark_entries_distilled
from cortex.embedder import embed
from cortex.sanitize import sanitize

logger = logging.getLogger("cortex.distill")

DISTILL_PROMPT = """You are a knowledge distiller. Given a batch of raw observations from AI coding sessions, extract the key patterns, insights, or recurring themes.

For each pattern you identify:
1. Write a clear, concise description of the pattern
2. Classify it as one of: workflow, debugging, architecture, preference, domain_knowledge, tool_usage, anti_pattern
3. Rate your confidence (0.0 to 1.0) that this is a genuine, reusable pattern

Respond with a JSON array of objects:
[
  {{
    "content": "description of the pattern",
    "pattern_type": "workflow|debugging|architecture|preference|domain_knowledge|tool_usage|anti_pattern",
    "confidence": 0.85
  }}
]

Only extract patterns that appear across multiple entries or represent significant insights.
If the entries are too sparse or unrelated to form patterns, return an empty array [].

Raw entries:
{entries}"""


def distill(conn, max_batches=10, batch_size=10, llm_call=None):
    """Run one distillation pass over undistilled entries.

    Args:
        conn: sqlite3 connection
        max_batches: maximum number of batches to process (cost control)
        batch_size: entries per batch
        llm_call: callable(prompt) -> str. If None, uses default Claude via subprocess.

    Returns:
        dict with stats: {"batches": N, "distillations": M, "entries_processed": P, "errors": E}
    """
    if llm_call is None:
        llm_call = _default_llm_call

    entries = get_undistilled_entries(conn, limit=max_batches * batch_size)
    if not entries:
        return {"batches": 0, "distillations": 0, "entries_processed": 0, "errors": 0}

    stats = {"batches": 0, "distillations": 0, "entries_processed": 0, "errors": 0}

    # Process in batches
    for i in range(0, len(entries), batch_size):
        if stats["batches"] >= max_batches:
            break

        batch = entries[i : i + batch_size]
        batch_ids = [e[0] for e in batch]
        batch_texts = [e[1] for e in batch]

        # Sanitize before sending to LLM
        sanitized = [sanitize(t) for t in batch_texts]
        entries_text = "\n---\n".join(
            f"[Entry {eid}]: {text}" for eid, text in zip(batch_ids, sanitized)
        )

        prompt = DISTILL_PROMPT.format(entries=entries_text)

        try:
            response = llm_call(prompt)
            patterns = _parse_response(response)

            for pattern in patterns:
                embedding = embed(pattern["content"])
                insert_distillation(
                    conn,
                    content=pattern["content"],
                    pattern_type=pattern.get("pattern_type", "unknown"),
                    confidence=pattern.get("confidence", 0.5),
                    source_model="distiller",
                    entry_count=len(batch),
                    embedding=embedding,
                    source_entry_ids=batch_ids,
                )
                stats["distillations"] += 1

            mark_entries_distilled(conn, batch_ids)
            stats["entries_processed"] += len(batch)
            stats["batches"] += 1

        except Exception as e:
            logger.warning("Distill batch %d failed: %s", stats["batches"], e)
            stats["errors"] += 1
            stats["batches"] += 1
            continue

    return stats


def _parse_response(response):
    """Parse LLM response as JSON array of patterns."""
    text = response.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    patterns = json.loads(text)
    if not isinstance(patterns, list):
        raise ValueError("Expected JSON array")

    # Validate each pattern
    valid = []
    for p in patterns:
        if isinstance(p, dict) and "content" in p and p["content"].strip():
            valid.append(p)

    return valid


def _default_llm_call(prompt):
    """Default LLM call via Claude CLI subprocess."""
    import subprocess

    result = subprocess.run(
        ["claude", "--print", "--model", "haiku", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed: {result.stderr}")
    return result.stdout
