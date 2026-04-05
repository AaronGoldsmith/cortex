"""Cortex MCP Server — the History Channel.

Surfaces past debugging sessions, architectural decisions, and distilled
patterns to an active AI session via the Model Context Protocol.
"""

from mcp.server.fastmcp import FastMCP

from cortex.config import DB_PATH
from cortex.db import get_connection
from cortex.query import query

mcp = FastMCP("cortex-history-channel")


def _format_result(result: dict) -> str:
    """Format a single query result for LLM consumption."""
    kind = result.get("kind", "entry")
    lines = []

    # Header
    if kind == "distillation":
        lines.append(f"## Distilled Pattern ({result.get('type', 'unknown')})")
    else:
        lines.append(f"## Entry ({result.get('type', 'unknown')})")

    # Metadata line
    meta_parts = []
    if result.get("source_project"):
        meta_parts.append(f"project: {result['source_project']}")
    if result.get("source_model"):
        meta_parts.append(f"model: {result['source_model']}")
    if result.get("created_at"):
        meta_parts.append(f"date: {result['created_at']}")
    meta_parts.append(f"score: {result.get('score', 0)}")
    lines.append(" | ".join(meta_parts))

    # Content
    lines.append("")
    lines.append(result.get("content", ""))

    # Source conversation context (distillations only)
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


@mcp.tool()
def search_project_history(query_text: str, project_name: str = None) -> str:
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
