"""Cortex CLI — knowledge ledger commands."""

import json
import os
import subprocess
import sys

import click

from cortex.config import (
    CORTEX_DIR,
    DB_PATH,
    DEFAULT_TOP_K,
    ENTRY_TYPES,
    HISTORY_PATH,
    STATE_PATH,
)


@click.group()
def main():
    """Cortex — persistent, model-agnostic knowledge ledger."""
    pass


@main.command()
def init():
    """Initialize Cortex: create ~/.cortex/ and set up the database."""
    from cortex.db import init_db

    CORTEX_DIR.mkdir(parents=True, exist_ok=True)
    init_db(DB_PATH)
    click.echo(f"Cortex initialized at {CORTEX_DIR}")
    click.echo(f"Database: {DB_PATH}")


@main.command("write")
@click.argument("content")
@click.option("--type", "entry_type", default="observation", type=click.Choice(sorted(ENTRY_TYPES)))
@click.option("--model", "source_model", default=None, help="Source model name")
@click.option("--project", "source_project", default=None, help="Source project path")
@click.option("--confidence", default=1.0, type=float, help="Confidence score 0.0-1.0")
def write_entry(content, entry_type, source_model, source_project, confidence):
    """Write an entry to the knowledge ledger."""
    from cortex.db import get_connection, insert_entry
    from cortex.embedder import embed

    _ensure_initialized()
    conn = get_connection(DB_PATH)
    embedding = embed(content)
    entry_id = insert_entry(
        conn,
        content=content,
        entry_type=entry_type,
        source_model=source_model,
        source_project=source_project,
        session_id=None,
        confidence=confidence,
        embedding=embedding,
    )
    if entry_id:
        click.echo(f"Entry {entry_id} written.")
    else:
        click.echo("Duplicate entry, skipped.")
    conn.close()


@main.command()
@click.argument("text")
@click.option("-k", "--top-k", default=DEFAULT_TOP_K, help="Number of results")
@click.option("--project", default=None, help="Filter by project path")
def query(text, top_k, project):
    """Search the knowledge ledger."""
    from cortex.db import get_connection
    from cortex.query import format_results
    from cortex.query import query as do_query

    _ensure_initialized()
    conn = get_connection(DB_PATH)
    results = do_query(conn, text, top_k=top_k, project_filter=project)
    click.echo(format_results(results))
    conn.close()


@main.command()
@click.option("--source", type=click.Choice(["claude", "goose", "all"]), default="claude",
              help="Data source to ingest from (default: claude)")
@click.option("--background", is_flag=True, help="Run in background (for hooks)")
@click.option("--memory", is_flag=True, help="Also ingest .claude/projects/*/memory/*.md files")
@click.option("--subagents", is_flag=True, help="Also ingest subagent conversation logs")
@click.option("--all", "ingest_all", is_flag=True, help="Ingest all sub-sources for the selected provider")
@click.option("--backfill-turns", is_flag=True, help="Populate turn_index on existing entries by matching them to session JSONL files")
def ingest(source, background, memory, subagents, ingest_all, backfill_turns):
    """Ingest AI tool session history into the ledger."""
    if background:
        # Spawn self as background process
        click.echo("cortex: ingesting session...", err=True)
        popen_kwargs = {
            "stdout": open(CORTEX_DIR / "ingest.log", "a"),
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.DETACHED_PROCESS
        else:
            popen_kwargs["start_new_session"] = True
        subprocess.Popen(
            [sys.executable, "-m", "cortex.cli", "ingest"],
            **popen_kwargs,
        )
        return

    from cortex.config import PROJECTS_DIR
    from cortex.db import get_connection
    from cortex.ingest import backfill_turn_indices, ingest_history, ingest_memory_files, ingest_subagent_logs, run_provider_ingest
    from cortex.providers import discover_providers, get_provider

    _ensure_initialized()
    conn = get_connection(DB_PATH)
    try:
        if backfill_turns:
            bstats = backfill_turn_indices(conn, PROJECTS_DIR)
            click.echo(
                f"Backfill -- Updated: {bstats['updated']}, "
                f"Skipped: {bstats['skipped']}"
            )
            return

        # Warn if Claude-specific flags used with non-Claude source
        if source not in ("claude", "all") and (memory or subagents):
            click.echo(f"Warning: --memory/--subagents only apply to Claude source, ignoring", err=True)

        # Determine which providers to run
        if source == "all":
            sources = discover_providers()
            if not sources:
                click.echo("No providers detected on this machine.", err=True)
                sys.exit(1)
        else:
            sources = [source]

        for src in sources:
            if src == "claude":
                # Claude uses the facade for backward compat (state file, turn resolution)
                stats = ingest_history(conn, HISTORY_PATH, STATE_PATH, projects_dir=PROJECTS_DIR)
                click.echo(
                    f"Claude  -- Ingested: {stats['ingested']}, "
                    f"Skipped: {stats['skipped']}, "
                    f"Errors: {stats['errors']}"
                )

                if memory or ingest_all:
                    mstats = ingest_memory_files(conn, PROJECTS_DIR)
                    click.echo(
                        f"Memory  -- Ingested: {mstats['ingested']}, "
                        f"Skipped: {mstats['skipped']}, "
                        f"Errors: {mstats['errors']}"
                    )

                if subagents or ingest_all:
                    sstats = ingest_subagent_logs(conn, PROJECTS_DIR)
                    click.echo(
                        f"Agents  -- Ingested: {sstats['ingested']}, "
                        f"Skipped: {sstats['skipped']}, "
                        f"Errors: {sstats['errors']}"
                    )
            else:
                provider = get_provider(src)
                if not provider.detect():
                    click.echo(f"{src.title()} -- not detected, skipping")
                    continue
                stats = run_provider_ingest(conn, provider)
                label = src.title().ljust(7)
                click.echo(
                    f"{label}-- Ingested: {stats['ingested']}, "
                    f"Skipped: {stats['skipped']}, "
                    f"Errors: {stats['errors']}"
                )

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        conn.close()


@main.command()
@click.option("--max-batches", default=10, help="Maximum number of batches to process (limits LLM cost)")
@click.option("--batch-size", default=10, help="Entries per batch")
@click.option("--dry-run", is_flag=True, help="Show what would be distilled without making LLM calls")
@click.option("--context-window", default=3, type=int, help="Include N surrounding turns from the session as LLM context (0=off, max 20)")
def distill(max_batches, batch_size, dry_run, context_window):
    """Extract patterns from raw entries using an LLM.

    Reads undistilled entries in batches, sanitizes secrets, sends them to an
    LLM to identify recurring patterns and insights, then writes the results
    back as distillations with lineage to source entries. Each batch = 1 LLM
    API call. Use --dry-run to preview before spending tokens. Use
    --context-window N to include N surrounding conversation turns from the
    original session, giving the distiller richer dialogue context.
    """
    from cortex.db import get_connection
    from cortex.distill import distill as do_distill

    _ensure_initialized()
    conn = get_connection(DB_PATH)
    stats = do_distill(conn, max_batches=max_batches, batch_size=batch_size, dry_run=dry_run, context_window=context_window)

    if dry_run:
        click.echo(f"DRY RUN — {stats['batches']} batches, {stats['entries_processed']} entries")
        click.echo(f"Estimated LLM calls: {stats['batches']}")
        click.echo()
        for batch_info in stats.get("plan", []):
            click.echo(f"  Batch {batch_info['batch']}: {batch_info['entry_count']} entries "
                       f"(IDs: {batch_info['entry_ids'][:5]}{'...' if len(batch_info['entry_ids']) > 5 else ''})")
            if batch_info["has_secrets_redacted"]:
                click.echo(f"    ! Contains secrets that will be redacted")
            for preview in batch_info["preview"]:
                click.echo(f"    - {preview}")
            click.echo()
    else:
        skipped = stats.get('skipped', 0)
        click.echo(
            f"Batches: {stats['batches']}, "
            f"Distillations: {stats['distillations']}, "
            f"Entries processed: {stats['entries_processed']}, "
            f"Skipped (insufficient context): {skipped}, "
            f"Errors: {stats['errors']}"
        )
    conn.close()


@main.command()
@click.argument("distillation_id", type=int)
@click.option("--window", default=5, type=int, help="Number of conversation turns to show before and after each source entry")
def trace(distillation_id, window):
    """Trace a distillation back to its source entries and conversation.

    Given a distillation ID, shows its content, the source entries that
    produced it (via lineage), and the surrounding conversation turns from
    the original session JSONL files so you can see the full context.
    """
    from cortex.config import PROJECTS_DIR
    from cortex.db import get_connection
    from cortex.sessions import find_session_file, format_turn, get_turn_context, read_session_turns

    _ensure_initialized()
    conn = get_connection(DB_PATH)

    # Fetch the distillation
    dist = conn.execute(
        "SELECT id, content, pattern_type, confidence, context_window, created_at "
        "FROM distillations WHERE id = ?",
        (distillation_id,),
    ).fetchone()
    if not dist:
        click.echo(f"Distillation {distillation_id} not found.")
        conn.close()
        return

    click.echo(f"Distillation D{dist[0]}")
    click.echo(f"  Pattern:    {dist[1]}")
    click.echo(f"  Type:       {dist[2]}")
    click.echo(f"  Confidence: {dist[3]}")
    click.echo(f"  Context window used: {dist[4] or 0}")
    click.echo(f"  Created:    {dist[5]}")
    click.echo()

    # Fetch source entries via lineage
    rows = conn.execute(
        "SELECT e.id, e.content, e.session_id, e.turn_index, e.source_project, e.created_at "
        "FROM lineage l JOIN entries e ON l.entry_id = e.id "
        "WHERE l.distillation_id = ? ORDER BY e.created_at",
        (distillation_id,),
    ).fetchall()

    if not rows:
        click.echo("  No source entries found in lineage.")
        conn.close()
        return

    def _safe_echo(text):
        """Echo text, replacing unencodable chars for Windows terminals."""
        try:
            click.echo(text)
        except UnicodeEncodeError:
            click.echo(text.encode("ascii", errors="replace").decode("ascii"))

    _safe_echo(f"Source entries ({len(rows)}):")
    for row in rows:
        eid, content, session_id, turn_index, project, created = row[0], row[1], row[2], row[3], row[4], row[5]
        _safe_echo(f"\n  Entry E{eid} (project: {project}, date: {created})")
        content_preview = content[:150] + "..." if len(content) > 150 else content
        _safe_echo(f"    Content: {content_preview}")

        if session_id and turn_index is not None:
            session_path = find_session_file(session_id, PROJECTS_DIR)
            if session_path:
                context_turns = get_turn_context(session_path, turn_index, window)
                if context_turns:
                    _safe_echo(f"    Conversation (turn {turn_index} +-{window}):")
                    for turn in context_turns:
                        marker = ">>>" if turn.index == turn_index else "   "
                        user_text = turn.user_text[:120] + "..." if len(turn.user_text) > 120 else turn.user_text
                        _safe_echo(f"      {marker} [{turn.index}] USER: {user_text}")
                        if turn.assistant_text:
                            asst = turn.assistant_text[:200] + "..." if len(turn.assistant_text) > 200 else turn.assistant_text
                            _safe_echo(f"      {marker}      ASSISTANT: {asst}")
                else:
                    click.echo(f"    (no conversation context available)")
            else:
                click.echo(f"    Session file not found for {session_id}")
        else:
            click.echo(f"    (no turn_index — run 'cortex ingest --backfill-turns' to resolve)")

    conn.close()


@main.command("eval")
@click.option("--generate", is_flag=True, help="Auto-generate retrieval eval cases from existing entries")
@click.option("--seed-qa", is_flag=True, help="Generate Q&A eval cases from curated memory entries")
@click.option("--top-k", default=5, help="Results per query to evaluate")
@click.option("--history", is_flag=True, help="Show eval trend over time")
@click.option("--backfill-variants", is_flag=True, help="Add query variants to existing cases that lack them")
@click.option("--compare-context", "compare_context_flag", is_flag=True, help="A/B eval: distillation quality with vs without conversation context")
@click.option("--llm-judge", is_flag=True, help="Score results with LLM judge on relevance/faithfulness (1-5)")
def run_eval(generate, seed_qa, top_k, history, backfill_variants, compare_context_flag, llm_judge):
    """Run evaluation suite against current knowledge base."""
    from cortex.db import get_connection
    from cortex.eval import (
        EVAL_CASES_PATH,
        backfill_variants as do_backfill,
        compare_evals,
        generate_eval_cases,
        load_eval_cases,
        load_eval_history,
        run_eval,
        save_eval_cases,
        seed_qa_cases,
        snapshot_eval,
    )

    _ensure_initialized()
    conn = get_connection(DB_PATH)

    if compare_context_flag:
        from cortex.eval import compare_context, format_context_comparison
        result = compare_context(conn, sample_size=20, context_window=2)
        click.echo(format_context_comparison(result))
        conn.close()
        return

    if history:
        h = load_eval_history()
        click.echo(compare_evals(h))
        conn.close()
        return

    if generate or seed_qa:
        cases = load_eval_cases(EVAL_CASES_PATH) if EVAL_CASES_PATH.exists() else []

        if generate:
            new = generate_eval_cases(conn)
            cases.extend(new)
            click.echo(f"Generated {len(new)} retrieval eval cases")

        if seed_qa:
            qa = seed_qa_cases(conn)
            cases.extend(qa)
            click.echo(f"Generated {len(qa)} Q&A eval cases")

        if not cases:
            click.echo("Not enough entries to generate eval cases.")
            conn.close()
            return

        save_eval_cases(EVAL_CASES_PATH, cases)
        click.echo(f"Total: {len(cases)} eval cases → {EVAL_CASES_PATH}")

    cases = load_eval_cases(EVAL_CASES_PATH)
    if not cases:
        click.echo("No eval cases found. Run 'cortex eval --generate' first.")
        conn.close()
        return

    if backfill_variants:
        count = do_backfill(cases)
        save_eval_cases(EVAL_CASES_PATH, cases)
        click.echo(f"Backfilled variants on {count} cases")

    report = run_eval(conn, cases, top_k=top_k, llm_judge=llm_judge)
    click.echo(report.summary())

    # Auto-snapshot for trend tracking
    snapshot_eval(report)
    click.echo(f"(Snapshot saved to eval history)")

    conn.close()


@main.command()
@click.option("--diagnose", is_flag=True, help="Analyze failing eval cases and output structured diagnosis")
@click.option("--update-case", type=int, default=None, help="Update eval case by index (0-based)")
@click.option("--query", default=None, help="New query for updated case")
@click.option("--keywords", default=None, help="Comma-separated expected keywords for updated case")
@click.option("--variants", default=None, help="Comma-separated query variants for updated case")
@click.option("--remove-case", type=int, default=None, help="Remove eval case by index (0-based)")
@click.option("--adjust-confidence", nargs=2, type=click.Tuple([int, float]), default=None,
              help="Adjust entry confidence: ENTRY_ID NEW_CONFIDENCE")
def improve(diagnose, update_case, query, keywords, variants, remove_case, adjust_confidence):
    """Tools for the eval-auditor agent to diagnose and fix weak spots."""
    from cortex.db import get_connection
    from cortex.eval import EVAL_CASES_PATH, load_eval_cases, run_eval, save_eval_cases

    _ensure_initialized()

    if diagnose:
        conn = get_connection(DB_PATH)
        cases = load_eval_cases(EVAL_CASES_PATH)
        if not cases:
            click.echo("No eval cases. Run 'cortex eval --generate --seed-qa' first.")
            conn.close()
            return

        report = run_eval(conn, cases, top_k=5)

        # Output structured diagnosis for the agent
        failures = []
        for i, cr in enumerate(report.case_results):
            if not cr.relevance_hit or cr.precision < 0.4:
                failures.append({
                    "index": i,
                    "query": cr.case.query,
                    "description": cr.case.description,
                    "expected_keywords": cr.case.expected_keywords,
                    "answer": cr.case.answer,
                    "relevance_hit": cr.relevance_hit,
                    "precision": cr.precision,
                    "noise": cr.noise,
                    "top_result": cr.top_result_preview,
                })

        click.echo(json.dumps({
            "summary": {
                "total_cases": report.total_cases,
                "avg_relevance": report.avg_relevance,
                "avg_precision": report.avg_precision,
                "avg_noise": report.avg_noise,
                "failing_count": len(failures),
            },
            "failures": failures,
        }, indent=2))
        conn.close()
        return

    if update_case is not None:
        cases = load_eval_cases(EVAL_CASES_PATH)
        if update_case >= len(cases):
            click.echo(f"Case index {update_case} out of range (max {len(cases) - 1})")
            return
        if query:
            cases[update_case].query = query
        if keywords:
            cases[update_case].expected_keywords = [k.strip() for k in keywords.split(",")]
        if variants:
            cases[update_case].query_variants = [v.strip() for v in variants.split(",")]
        save_eval_cases(EVAL_CASES_PATH, cases)
        click.echo(f"Updated case {update_case}: {cases[update_case].description}")
        return

    if remove_case is not None:
        cases = load_eval_cases(EVAL_CASES_PATH)
        if remove_case >= len(cases):
            click.echo(f"Case index {remove_case} out of range (max {len(cases) - 1})")
            return
        removed = cases.pop(remove_case)
        save_eval_cases(EVAL_CASES_PATH, cases)
        click.echo(f"Removed case {remove_case}: {removed.description}")
        return

    if adjust_confidence:
        entry_id, new_conf = adjust_confidence
        conn = get_connection(DB_PATH)
        conn.execute(
            "UPDATE entries SET confidence = ? WHERE id = ?",
            (new_conf, int(entry_id)),
        )
        conn.commit()
        click.echo(f"Entry {int(entry_id)} confidence -> {new_conf}")
        conn.close()
        return

    click.echo("Use --diagnose, --update-case, --remove-case, or --adjust-confidence")


@main.command()
def status():
    """Show Cortex status: entry counts, last ingest/distill, DB size."""
    _ensure_initialized()

    from cortex.db import get_connection

    conn = get_connection(DB_PATH)

    entry_count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    distill_count = conn.execute("SELECT COUNT(*) FROM distillations").fetchone()[0]
    undistilled = conn.execute(
        "SELECT COUNT(*) FROM entries WHERE distilled_at IS NULL"
    ).fetchone()[0]

    # Entries by project
    projects = conn.execute(
        "SELECT source_project, COUNT(*) FROM entries "
        "GROUP BY source_project ORDER BY COUNT(*) DESC LIMIT 5"
    ).fetchall()

    # Last ingest/distill times
    last_entry = conn.execute(
        "SELECT created_at FROM entries ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    last_distill = conn.execute(
        "SELECT created_at FROM distillations ORDER BY created_at DESC LIMIT 1"
    ).fetchone()

    # DB size
    db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0

    click.echo(f"Cortex Status")
    click.echo(f"  Database:        {DB_PATH}")
    click.echo(f"  DB size:         {db_size / 1024:.1f} KB")
    click.echo(f"  Entries:         {entry_count}")
    click.echo(f"  Distillations:   {distill_count}")
    click.echo(f"  Undistilled:     {undistilled}")
    click.echo(f"  Last entry:      {last_entry[0] if last_entry else 'never'}")
    click.echo(f"  Last distill:    {last_distill[0] if last_distill else 'never'}")

    if projects:
        click.echo(f"  Top projects:")
        for proj, count in projects:
            click.echo(f"    {proj or '(no project)'}: {count}")

    conn.close()


def _ensure_initialized():
    """Check that Cortex has been initialized."""
    if not DB_PATH.exists():
        click.echo("Cortex not initialized. Run 'cortex init' first.", err=True)
        sys.exit(1)
