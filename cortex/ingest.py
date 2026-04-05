"""Ingestion — parses Claude Code history, memory files, and subagent logs into the Cortex ledger."""

import json
import logging
import re
from pathlib import Path

from cortex.db import insert_entry
from cortex.embedder import embed

log = logging.getLogger(__name__)

# Cache for session turn lookups within a single ingest run
_session_turn_cache: dict[str, list] = {}

# Map memory file frontmatter types to Cortex entry types
_MEMORY_TYPE_MAP = {
    "user": "observation",
    "feedback": "correction",
    "project": "observation",
    "reference": "recommendation",
}


def _load_state(state_path: Path) -> dict:
    """Load cursor state from disk, or return defaults for first run."""
    try:
        return json.loads(state_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"last_line": 0, "last_session_id": None, "file_size": 0}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state))


def ingest_history(conn, history_path: Path, state_path: Path, projects_dir: Path = None) -> dict:
    """Ingest new lines from history.jsonl into the ledger.

    If projects_dir is provided, resolves turn_index for each entry by looking
    up its position in the corresponding session JSONL file.

    Returns dict with keys: ingested, skipped, errors.
    """
    history_path = Path(history_path)
    state_path = Path(state_path)

    if not history_path.exists():
        raise FileNotFoundError(
            f"History file not found: {history_path}. "
            "Is Claude Code installed? Expected at ~/.claude/history.jsonl"
        )

    state = _load_state(state_path)
    file_size = history_path.stat().st_size

    # Detect file rotation (size shrank)
    if file_size < state.get("file_size", 0):
        log.info("History file shrank (%d -> %d), resetting cursor", state["file_size"], file_size)
        state["last_line"] = 0

    stats = {"ingested": 0, "skipped": 0, "errors": 0}
    last_line = state["last_line"]

    with open(history_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < last_line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Malformed JSON at line %d, skipping", line_no)
                stats["errors"] += 1
                continue

            content = entry.get("display", "")
            if not content or len(content) < 10:
                stats["skipped"] += 1
                continue

            session_id = entry.get("sessionId")
            turn_index = None
            if projects_dir and session_id:
                turn_index = _resolve_turn_index(session_id, content, projects_dir)

            embedding = embed(content)
            result = insert_entry(
                conn,
                content=content,
                entry_type="raw",
                source_model="claude",
                source_project=entry.get("project"),
                session_id=session_id,
                confidence=1.0,
                embedding=embedding,
                turn_index=turn_index,
            )
            if result is None:
                stats["skipped"] += 1  # duplicate
            else:
                stats["ingested"] += 1
            state["last_session_id"] = session_id
            last_line = line_no + 1

    state["last_line"] = last_line
    state["file_size"] = file_size
    _save_state(state_path, state)
    return stats


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


def _resolve_turn_index(session_id: str, content: str, projects_dir: Path) -> int | None:
    """Find the turn index of a user message within its session JSONL.

    Uses the sessions module to parse turns, then matches by content.
    Results are cached per session_id within the process.
    """
    if session_id in _session_turn_cache:
        turns = _session_turn_cache[session_id]
    else:
        from cortex.sessions import find_session_file, read_session_turns

        session_path = find_session_file(session_id, projects_dir)
        if session_path is None:
            _session_turn_cache[session_id] = []
            return None
        turns = read_session_turns(session_path)
        _session_turn_cache[session_id] = turns

    # Match by content prefix (history.jsonl display may be truncated)
    content_stripped = content.strip()
    for turn in turns:
        if turn.user_text == content_stripped:
            return turn.index
        # Fallback: prefix match for truncated content
        if len(content_stripped) > 50 and turn.user_text.startswith(content_stripped[:50]):
            return turn.index
    return None


def backfill_turn_indices(conn, projects_dir: Path) -> dict:
    """Backfill turn_index for existing entries that have session_id but no turn_index.

    Returns dict with keys: updated, skipped.
    """
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
