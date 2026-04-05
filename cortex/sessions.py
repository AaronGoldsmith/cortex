"""Session JSONL reader — extracts conversation turns from Claude Code session files."""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

log = logging.getLogger(__name__)


class Turn(NamedTuple):
    """A single conversation turn: one user message paired with the assistant response."""
    index: int
    user_text: str
    assistant_text: str
    timestamp: Optional[str] = None


def find_session_file(session_id: str, projects_dir: Path) -> Optional[Path]:
    """Find the session JSONL file for a given session ID.

    Searches all project directories under projects_dir for a file named
    {session_id}.jsonl.
    """
    if not session_id or not projects_dir.exists():
        return None

    for match in projects_dir.glob(f"*/{session_id}.jsonl"):
        return match
    return None


def read_session_turns(session_path: Path) -> list[Turn]:
    """Parse a session JSONL file into ordered conversation turns.

    Walks the file line by line, extracts user text messages (skipping
    tool_result blocks) and assistant text blocks (skipping tool_use and
    thinking blocks), then pairs them into Turn objects.
    """
    return _read_turns_cached(str(session_path), session_path.stat().st_mtime)


@lru_cache(maxsize=32)
def _read_turns_cached(path_str: str, _mtime: float) -> list[Turn]:
    """Cached implementation — cache key includes mtime for invalidation."""
    path = Path(path_str)
    if not path.exists():
        return []

    turns = []
    pending_user_text = None
    pending_timestamp = None
    turn_index = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type", "")
            msg = entry.get("message", {})
            role = msg.get("role", "")

            # User text message (not tool results)
            if entry_type == "user" or role == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content.strip()) > 0:
                    # If we had a pending user message without a response, emit it solo
                    if pending_user_text is not None:
                        turns.append(Turn(
                            index=turn_index,
                            user_text=pending_user_text,
                            assistant_text="",
                            timestamp=pending_timestamp,
                        ))
                        turn_index += 1
                    pending_user_text = content.strip()
                    pending_timestamp = entry.get("timestamp")
                # Skip tool_result blocks (content is a list)

            # Assistant text response
            elif entry_type == "assistant" or role == "assistant":
                raw_content = msg.get("content", [])
                assistant_text = _extract_assistant_text(raw_content)
                if assistant_text and pending_user_text is not None:
                    turns.append(Turn(
                        index=turn_index,
                        user_text=pending_user_text,
                        assistant_text=assistant_text,
                        timestamp=pending_timestamp,
                    ))
                    turn_index += 1
                    pending_user_text = None
                    pending_timestamp = None

    # Emit any trailing user message without a response
    if pending_user_text is not None:
        turns.append(Turn(
            index=turn_index,
            user_text=pending_user_text,
            assistant_text="",
            timestamp=pending_timestamp,
        ))

    return turns


def _extract_assistant_text(content) -> str:
    """Extract only text blocks from assistant message content.

    Skips tool_use and thinking blocks.
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "").strip()
            if text:
                texts.append(text)
    return "\n".join(texts)


def get_turn_context(session_path: Path, turn_index: int, window: int) -> list[Turn]:
    """Get a window of turns around a focal turn index.

    Returns turns from [turn_index - window, turn_index + window] inclusive,
    clamped to session bounds.
    """
    if window <= 0:
        return []
    turns = read_session_turns(session_path)
    if not turns:
        return []
    start = max(0, turn_index - window)
    end = min(len(turns), turn_index + window + 1)
    return turns[start:end]


def format_turn(turn: Turn, focal: bool = False) -> str:
    """Format a single turn for display or prompt inclusion."""
    marker = ">>>" if focal else "   "
    lines = [f"{marker} USER: {turn.user_text}"]
    if turn.assistant_text:
        lines.append(f"{marker} ASSISTANT: {turn.assistant_text}")
    return "\n".join(lines)


def format_context_window(turns: list[Turn], focal_index: int) -> str:
    """Format a window of turns with the focal turn highlighted."""
    lines = []
    for turn in turns:
        lines.append(format_turn(turn, focal=(turn.index == focal_index)))
        lines.append("---")
    return "\n".join(lines)
