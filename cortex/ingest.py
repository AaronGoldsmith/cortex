"""Ingestion — parses AI tool session history, memory files, and subagent logs into the Cortex ledger."""

import json
import logging
import re
from pathlib import Path

from cortex.db import insert_entry
from cortex.embedder import embed
from cortex.providers.base import IngestEntry

log = logging.getLogger(__name__)

# Map memory file frontmatter types to Cortex entry types
_MEMORY_TYPE_MAP = {
    "user": "observation",
    "feedback": "correction",
    "project": "observation",
    "reference": "recommendation",
}


def run_provider_ingest(conn, provider) -> dict:
    """Generic ingest loop: iterate provider entries, embed, insert, return stats."""
    stats = {"ingested": 0, "skipped": 0, "errors": 0}
    for entry in provider.iter_entries(conn):
        try:
            embedding = embed(entry.content)
            result = insert_entry(
                conn,
                content=entry.content,
                entry_type=entry.entry_type,
                source_model=entry.source_model,
                source_project=entry.source_project,
                session_id=entry.session_id,
                confidence=entry.confidence,
                embedding=embedding,
                turn_index=entry.turn_index,
            )
            if result is None:
                stats["skipped"] += 1
            else:
                stats["ingested"] += 1
        except Exception as e:
            log.warning("Error ingesting entry: %s", e)
            stats["errors"] += 1
    return stats


def ingest_history(conn, history_path: Path, state_path: Path, projects_dir: Path = None) -> dict:
    """Ingest new lines from history.jsonl into the ledger.

    Backward-compatible facade — delegates to ClaudeHistoryProvider.
    Returns dict with keys: ingested, skipped, errors.
    """
    from cortex.providers.claude import ClaudeHistoryProvider

    provider = ClaudeHistoryProvider(
        history_path=history_path,
        state_path=state_path,
        projects_dir=projects_dir,
    )
    return run_provider_ingest(conn, provider)


def _parse_memory_frontmatter(text):
    """Extract frontmatter fields from a memory .md file."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not match:
        return {}, text

    frontmatter = {}
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip()

    return frontmatter, match.group(2).strip()


def ingest_memory_files(conn, projects_dir: Path) -> dict:
    """Ingest .claude/projects/*/memory/*.md files into the ledger.

    Memory files are already curated knowledge — they get higher confidence (1.2).
    Returns dict with keys: ingested, skipped, errors.
    """
    projects_dir = Path(projects_dir)
    stats = {"ingested": 0, "skipped": 0, "errors": 0}

    if not projects_dir.exists():
        log.warning("Projects directory not found: %s", projects_dir)
        return stats

    for md_file in sorted(projects_dir.glob("*/memory/*.md")):
        if md_file.name == "MEMORY.md":
            stats["skipped"] += 1
            continue

        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception as e:
            log.warning("Failed to read %s: %s", md_file, e)
            stats["errors"] += 1
            continue

        frontmatter, body = _parse_memory_frontmatter(text)
        if not body or len(body) < 10:
            stats["skipped"] += 1
            continue

        # Extract project name from directory path (C--Users-aargo-Development-mobius → mobius)
        project_dir_name = md_file.parent.parent.name
        project_name = project_dir_name.rsplit("-", 1)[-1] if "-" in project_dir_name else project_dir_name

        memory_type = frontmatter.get("type", "observation")
        entry_type = _MEMORY_TYPE_MAP.get(memory_type, "observation")
        name = frontmatter.get("name", md_file.stem)

        # Combine name + body for richer embedding
        content = f"[{name}] {body}" if name else body

        embedding = embed(content)
        result = insert_entry(
            conn,
            content=content,
            entry_type=entry_type,
            source_model="memory",
            source_project=project_name,
            session_id=None,
            confidence=1.2,  # curated knowledge gets a boost
            embedding=embedding,
        )
        if result is None:
            stats["skipped"] += 1
        else:
            stats["ingested"] += 1

    return stats


def ingest_subagent_logs(conn, projects_dir: Path) -> dict:
    """Ingest subagent JSONL conversation logs into the ledger.

    Extracts user prompts and assistant text responses from subagent sessions.
    Returns dict with keys: ingested, skipped, errors.
    """
    projects_dir = Path(projects_dir)
    stats = {"ingested": 0, "skipped": 0, "errors": 0}

    if not projects_dir.exists():
        log.warning("Projects directory not found: %s", projects_dir)
        return stats

    for jsonl_file in sorted(projects_dir.glob("*/*/subagents/agent-*.jsonl")):
        # Extract project name from path
        project_dir_name = jsonl_file.parts[-4]  # the project directory
        project_name = project_dir_name.rsplit("-", 1)[-1] if "-" in project_dir_name else project_dir_name

        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        stats["errors"] += 1
                        continue

                    # Extract content from user messages (the prompts/tasks)
                    msg = entry.get("message", {})
                    role = msg.get("role") or entry.get("type")

                    if role == "user":
                        content = msg.get("content", "") if isinstance(msg.get("content"), str) else ""
                        if not content or len(content) < 20:
                            stats["skipped"] += 1
                            continue

                        # Skip system-generated preambles that are just agent setup
                        if content.startswith("You are working in"):
                            # Still ingest if there's substantial content after the preamble
                            lines = content.split("\n", 2)
                            if len(lines) > 2 and len(lines[2]) > 20:
                                content = lines[2]
                            else:
                                stats["skipped"] += 1
                                continue

                    elif role == "assistant":
                        # Extract text from assistant content blocks
                        raw_content = msg.get("content", [])
                        if isinstance(raw_content, list):
                            texts = [c.get("text", "") for c in raw_content if c.get("type") == "text"]
                            content = " ".join(texts).strip()
                        elif isinstance(raw_content, str):
                            content = raw_content.strip()
                        else:
                            stats["skipped"] += 1
                            continue

                        if not content or len(content) < 20:
                            stats["skipped"] += 1
                            continue

                        # Skip very long assistant responses (tool dumps, code blocks)
                        if len(content) > 2000:
                            content = content[:2000]
                    else:
                        stats["skipped"] += 1
                        continue

                    agent_id = entry.get("agentId", jsonl_file.stem)
                    embedding = embed(content)
                    result = insert_entry(
                        conn,
                        content=content,
                        entry_type="raw",
                        source_model="claude",
                        source_project=project_name,
                        session_id=agent_id,
                        confidence=1.0,
                        embedding=embedding,
                    )
                    if result is None:
                        stats["skipped"] += 1
                    else:
                        stats["ingested"] += 1

        except Exception as e:
            log.warning("Failed to process %s: %s", jsonl_file, e)
            stats["errors"] += 1

    return stats


def backfill_turn_indices(conn, projects_dir: Path) -> dict:
    """Backfill turn_index for existing entries that have session_id but no turn_index.

    Returns dict with keys: updated, skipped.
    """
    from cortex.providers.claude import _resolve_turn_index, _session_turn_cache

    rows = conn.execute(
        "SELECT id, content, session_id FROM entries "
        "WHERE session_id IS NOT NULL AND turn_index IS NULL"
    ).fetchall()

    stats = {"updated": 0, "skipped": 0}
    for row in rows:
        entry_id, content, session_id = row[0], row[1], row[2]
        turn_index = _resolve_turn_index(session_id, content, projects_dir)
        if turn_index is not None:
            conn.execute(
                "UPDATE entries SET turn_index = ? WHERE id = ?",
                (turn_index, entry_id),
            )
            stats["updated"] += 1
        else:
            stats["skipped"] += 1

    conn.commit()
    _session_turn_cache.clear()
    return stats
