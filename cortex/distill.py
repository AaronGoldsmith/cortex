"""Cortex distiller — extract patterns from raw entries using LLM."""

import json
import logging
import os

from cortex.db import get_undistilled_entries, insert_distillation, mark_entries_distilled
from cortex.embedder import embed
from cortex.sanitize import has_secrets, sanitize

logger = logging.getLogger("cortex.distill")

DISTILL_PROMPT = """You are a knowledge distiller. Given a batch of raw observations from AI coding sessions, extract the key patterns, insights, or recurring themes.

For each pattern you identify:
1. Write a clear, concise description of the pattern
2. Classify it as one of: workflow, debugging, architecture, preference, domain_knowledge, tool_usage, anti_pattern
3. Rate your confidence (0.0 to 1.0) that this is a genuine, reusable pattern

For entries that lack sufficient context to extract anything meaningful (e.g., isolated questions like "did it finish?" or vague commands with no surrounding context), mark them as skipped.

Respond with a JSON object containing two arrays:
{{
  "patterns": [
    {{
      "content": "description of the pattern",
      "pattern_type": "workflow|debugging|architecture|preference|domain_knowledge|tool_usage|anti_pattern",
      "confidence": 0.85,
      "source_entry_ids": [1, 3]
    }}
  ],
  "skipped": [
    {{
      "entry_id": 2,
      "reason": "isolated question without context"
    }}
  ]
}}

Rules:
- Only extract patterns that appear across multiple entries or represent significant insights.
- If ALL entries lack context, return {{"patterns": [], "skipped": [...]}}
- Be specific in source_entry_ids — only link entries that actually support the pattern.
- Skipped entries are NOT lost — they will be re-processed later when more context is available.

Output ONLY the JSON object — no prose, no summary, no markdown fences.

Raw entries:
{entries}"""


def distill(conn, max_batches=10, batch_size=10, llm_call=None, dry_run=False, context_window=0):
    """Run one distillation pass over undistilled entries.

    Args:
        conn: sqlite3 connection
        max_batches: maximum number of batches to process (cost control)
        batch_size: entries per batch
        llm_call: callable(prompt) -> str. If None, uses default Claude via subprocess.
        dry_run: if True, show what would be done without making LLM calls or writing.
        context_window: number of surrounding conversation turns to include (0 = none).

    Returns:
        dict with stats: {"batches": N, "distillations": M, "entries_processed": P, "errors": E}
        If dry_run, also includes "plan": list of batch previews.
    """
    if llm_call is None:
        llm_call = _default_llm_call

    # Cap context window to prevent token blowout
    context_window = min(context_window, 20)

    entries = get_undistilled_entries(conn, limit=max_batches * batch_size)
    if not entries:
        return {"batches": 0, "distillations": 0, "entries_processed": 0, "errors": 0}

    stats = {"batches": 0, "distillations": 0, "entries_processed": 0, "errors": 0}
    if dry_run:
        stats["plan"] = []

    # Process in batches
    for i in range(0, len(entries), batch_size):
        if stats["batches"] >= max_batches:
            break

        batch = entries[i : i + batch_size]
        batch_ids = [e[0] for e in batch]
        batch_texts = [e[1] for e in batch]

        # Sanitize before sending to LLM
        sanitized = [sanitize(t) for t in batch_texts]

        if dry_run:
            has_redactions = any(has_secrets(t) for t in batch_texts)
            stats["plan"].append({
                "batch": stats["batches"] + 1,
                "entry_count": len(batch),
                "entry_ids": batch_ids,
                "has_secrets_redacted": has_redactions,
                "has_context_window": context_window > 0,
                "context_window": context_window,
                "preview": [t[:80] + "..." if len(t) > 80 else t for t in sanitized[:3]],
            })
            stats["entries_processed"] += len(batch)
            stats["batches"] += 1
            continue
        # Include project and date context for each entry
        entry_lines = []
        for entry, text in zip(batch, sanitized):
            eid = entry[0]
            project = entry[4] or "unknown"  # source_project
            created = entry[8] or "unknown"  # created_at
            session_id = entry[5]  # session_id
            turn_index = entry[10] if len(entry) > 10 else None  # turn_index
            # Shorten project path to just the directory name
            if "\\" in str(project) or "/" in str(project):
                project = str(project).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]

            entry_header = f"[Entry {eid}] (project: {project}, date: {created})"

            # Pull conversation context if available
            if context_window > 0 and session_id and turn_index is not None:
                context_text = _get_conversation_context(session_id, turn_index, context_window)
                if context_text:
                    entry_lines.append(
                        f"{entry_header}:\n"
                        f"  Conversation context:\n{context_text}\n"
                        f"  >>> Focal message: {text}"
                    )
                    continue

            entry_lines.append(f"{entry_header}: {text}")
        entries_text = "\n---\n".join(entry_lines)

        prompt = DISTILL_PROMPT.format(entries=entries_text)

        try:
            response = llm_call(prompt)
            result = _parse_response(response)

            # Handle both old format (list) and new format (dict with patterns/skipped)
            if isinstance(result, dict):
                patterns = result.get("patterns", [])
                skipped = result.get("skipped", [])
            else:
                patterns = result
                skipped = []

            for pattern in patterns:
                embedding = embed(pattern["content"])
                # Use specific source IDs if provided, else fall back to whole batch
                source_ids = pattern.get("source_entry_ids", batch_ids)
                # Validate IDs are in this batch
                source_ids = [eid for eid in source_ids if eid in batch_ids] or batch_ids
                insert_distillation(
                    conn,
                    content=pattern["content"],
                    pattern_type=pattern.get("pattern_type", "unknown"),
                    confidence=pattern.get("confidence", 0.5),
                    source_model="distiller",
                    entry_count=len(source_ids),
                    embedding=embedding,
                    source_entry_ids=source_ids,
                    context_window=context_window,
                )
                stats["distillations"] += 1

            # Mark skipped entries with reason (re-processable later)
            skipped_ids = []
            for skip in skipped:
                eid = skip.get("entry_id")
                reason = skip.get("reason", "insufficient_context")
                if eid in batch_ids:
                    conn.execute(
                        "UPDATE entries SET distilled_at = ? WHERE id = ?",
                        (f"skipped:{reason}", eid),
                    )
                    skipped_ids.append(eid)
            stats["skipped"] = stats.get("skipped", 0) + len(skipped_ids)

            # Mark successfully distilled entries (exclude skipped)
            distilled_ids = [eid for eid in batch_ids if eid not in skipped_ids]
            mark_entries_distilled(conn, distilled_ids)
            conn.commit()

            stats["entries_processed"] += len(batch)
            stats["batches"] += 1

        except Exception as e:
            logger.warning("Distill batch %d failed: %s", stats["batches"], e)
            stats["errors"] += 1
            stats["batches"] += 1
            continue

    return stats


def _get_conversation_context(session_id: str, turn_index: int, window: int) -> str | None:
    """Pull surrounding conversation turns from a session JSONL.

    Returns formatted context string, or None if session file not found.
    """
    from cortex.config import PROJECTS_DIR
    from cortex.sessions import find_session_file, format_turn, get_turn_context

    session_path = find_session_file(session_id, PROJECTS_DIR)
    if session_path is None:
        return None

    context_turns = get_turn_context(session_path, turn_index, window)
    if not context_turns:
        return None

    lines = []
    for turn in context_turns:
        text = format_turn(turn, focal=(turn.index == turn_index))
        lines.append(sanitize(text))
        lines.append("")
    return "\n".join(lines)


def _parse_response(response):
    """Parse LLM response as JSON. Handles both formats:
    - Old: JSON array of patterns
    - New: JSON object with "patterns" and "skipped" arrays
    Tolerates extra text before/after the JSON.
    """
    text = response.strip()

    # Handle markdown code blocks
    if "```" in text:
        # Extract content between first ``` and last ```
        parts = text.split("```")
        if len(parts) >= 3:
            # Take the content block, strip language tag
            block = parts[1]
            if block.startswith("json"):
                block = block[4:]
            text = block.strip()

    # Find the JSON object or array in the text
    # Look for first { or [ and find matching close
    json_start = -1
    for i, ch in enumerate(text):
        if ch in ('{', '['):
            json_start = i
            break

    if json_start == -1:
        raise ValueError("No JSON found in response")

    # Try parsing from the start of JSON, progressively trimming trailing junk
    parse_text = text[json_start:]
    parsed = None
    last_error = None
    for end_offset in range(0, min(200, len(parse_text))):
        try_text = parse_text if end_offset == 0 else parse_text[:-(end_offset)]
        try:
            parsed = json.loads(try_text)
            break
        except json.JSONDecodeError as e:
            last_error = e
            continue

    if parsed is None:
        raise last_error or ValueError("Failed to parse JSON from response")

    # New format: dict with patterns and skipped
    if isinstance(parsed, dict) and "patterns" in parsed:
        valid_patterns = []
        for p in parsed.get("patterns", []):
            if isinstance(p, dict) and "content" in p and p["content"].strip():
                valid_patterns.append(p)
        valid_skipped = []
        for s in parsed.get("skipped", []):
            if isinstance(s, dict) and "entry_id" in s:
                valid_skipped.append(s)
        return {"patterns": valid_patterns, "skipped": valid_skipped}

    # Old format: plain list of patterns
    if isinstance(parsed, list):
        valid = []
        for p in parsed:
            if isinstance(p, dict) and "content" in p and p["content"].strip():
                valid.append(p)
        return valid

    raise ValueError(f"Unexpected response format: {type(parsed)}")


def _default_llm_call(prompt):
    """Default LLM call via Claude CLI subprocess."""
    import subprocess
   # todo: experiment with using two additional flags for constructing json schema
    result = subprocess.run(
        ["claude", "--agent", "signal-distiller", "--no-session-persistence", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=200,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed: {result.stderr}")
    return result.stdout
