"""Cortex MCP Server — the History Channel.

Hybrid architecture:
- PULL: search_project_history tool for explicit queries
- PUSH: Background monitor tails history.jsonl and pushes high-relevance
  context via notifications/claude/channel

Boot:
  claude --dangerously-load-development-channels "python -m cortex.mcp_server"
"""

import asyncio
import json
import os
import time as _time

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.lowlevel.server import NotificationOptions
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification

from cortex.config import DB_PATH, HISTORY_PATH
from cortex.db import get_connection
from cortex.query import query

RELEVANCE_THRESHOLD = 0.85

mcp = FastMCP("cortex-history-channel")

# --- Session state for background push ---
_write_stream_ref = None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_result(result: dict) -> str:
    """Format a single query result for LLM consumption."""
    kind = result.get("kind", "entry")
    lines = []

    if kind == "distillation":
        lines.append(f"## Distilled Pattern ({result.get('type', 'unknown')})")
    else:
        lines.append(f"## Entry ({result.get('type', 'unknown')})")

    meta_parts = []
    if result.get("source_project"):
        meta_parts.append(f"project: {result['source_project']}")
    if result.get("source_model"):
        meta_parts.append(f"model: {result['source_model']}")
    if result.get("created_at"):
        meta_parts.append(f"date: {result['created_at']}")
    meta_parts.append(f"score: {result.get('score', 0)}")
    lines.append(" | ".join(meta_parts))

    lines.append("")
    lines.append(result.get("content", ""))

    source_context = result.get("source_context")
    if source_context:
        lines.append("")
        lines.append("### Source Conversation Context")
        for turn in source_context:
            if turn.get("user_text"):
                lines.append(f"\n**User:** {turn['user_text']}")
            if turn.get("assistant_text"):
                asst = turn["assistant_text"]
                if len(asst) > 800:
                    asst = asst[:797] + "..."
                lines.append(f"\n**Assistant:** {asst}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PULL — explicit tool
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_project_history(
    query_text: str,
    project_name: str = None,
    ctx: Context = None,
) -> str:
    """Search the Cortex knowledge ledger for past debugging sessions,
    architectural decisions, and distilled patterns.

    Args:
        query_text: Natural language query describing what you're looking for.
        project_name: Optional project directory name to filter results by.
    """
    if not DB_PATH.exists():
        return "Cortex database not initialized. Run `cortex init` first."

    conn = get_connection()
    try:
        results = query(conn, query_text, top_k=4, project_filter=project_name)
    finally:
        conn.close()

    if not results:
        return "No relevant history found."

    formatted = [_format_result(r) for r in results]
    header = f"Found {len(results)} relevant result(s) from project history:\n"
    return header + "\n---\n".join(formatted)


# ---------------------------------------------------------------------------
# PUSH — background monitor + channel notifications
# ---------------------------------------------------------------------------

async def _send_channel_event(content: str, meta: dict | None = None):
    """Emit a notifications/claude/channel event over the write stream."""
    if _write_stream_ref is None:
        return

    params = {"content": content}
    if meta:
        params["meta"] = meta

    notification = JSONRPCNotification(
        jsonrpc="2.0",
        method="notifications/claude/channel",
        params=params,
    )
    await _write_stream_ref.send(
        SessionMessage(message=JSONRPCMessage(notification))
    )

    # Write sidecar for statusline visibility
    try:
        signal_file = DB_PATH.parent / "last_signal.json"
        signal_file.write_text(json.dumps({
            "ts": int(_time.time()),
            "score": meta.get("score", "?") if meta else "?",
            "snippet": content[:80].replace("\n", " "),
        }))
    except OSError:
        pass


async def _evaluate_and_push(user_text: str):
    """Query Cortex for distillations and push a channel event if above threshold."""
    if not DB_PATH.exists():
        return

    conn = get_connection()
    try:
        results = query(conn, user_text, top_k=4)
    finally:
        conn.close()

    # Only push distillations — raw entries are noisy and not actionable context
    distillations = [r for r in results if r.get("kind") == "distillation"]
    if not distillations:
        return

    top = distillations[0]
    score = top.get("score", 0)
    if score < RELEVANCE_THRESHOLD:
        return

    await _send_channel_event(
        content=_format_result(top),
        meta={
            "severity": "info",
            "score": str(score),
            "kind": "distillation",
            "type": top.get("type", "unknown"),
        },
    )


async def _monitor_active_session():
    """Tail history.jsonl and push high-relevance context as channel events."""
    if not HISTORY_PATH.exists():
        return

    file_size = os.path.getsize(HISTORY_PATH)
    seen = set()  # dedup on first 100 chars

    while True:
        await asyncio.sleep(2)

        try:
            current_size = os.path.getsize(HISTORY_PATH)
        except OSError:
            continue

        if current_size <= file_size:
            file_size = current_size  # handle truncation
            continue

        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                f.seek(file_size)
                new_data = f.read()
                file_size = current_size
        except OSError:
            continue

        for line in new_data.splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial line from mid-write

            user_text = entry.get("display", "")
            if len(user_text) < 15:
                continue

            key = user_text[:100]
            if key in seen:
                continue
            seen.add(key)
            if len(seen) > 500:
                seen.clear()

            await _evaluate_and_push(user_text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _run():
    """Run the MCP server with claude/channel capability advertised."""
    from mcp.server.stdio import stdio_server

    low_level = mcp._mcp_server

    # Build init options with the experimental claude/channel capability
    init_options = low_level.create_initialization_options(
        notification_options=NotificationOptions(),
        experimental_capabilities={"claude/channel": {}},
    )

    async def _main():
        global _write_stream_ref
        async with stdio_server() as (read_stream, write_stream):
            _write_stream_ref = write_stream
            asyncio.create_task(_monitor_active_session())
            await low_level.run(read_stream, write_stream, init_options)

    asyncio.run(_main())


if __name__ == "__main__":
    _run()
