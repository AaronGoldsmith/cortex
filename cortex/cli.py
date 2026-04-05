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
@click.option("--background", is_flag=True, help="Run in background (for hooks)")
@click.option("--memory", is_flag=True, help="Also ingest .claude/projects/*/memory/*.md files")
@click.option("--subagents", is_flag=True, help="Also ingest subagent conversation logs")
@click.option("--all", "ingest_all", is_flag=True, help="Ingest all sources (history + memory + subagents)")
def ingest(background, memory, subagents, ingest_all):
    """Ingest Claude Code session history into the ledger."""
    if background:
        # Spawn self as background process
        click.echo("cortex: ingesting session...", err=True)
        subprocess.Popen(
            [sys.executable, "-m", "cortex.cli", "ingest"],
            stdout=open(CORTEX_DIR / "ingest.log", "a"),
            stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS if os.name == "nt" else 0,
        )
        return

    from cortex.config import PROJECTS_DIR
    from cortex.db import get_connection
    from cortex.ingest import ingest_history, ingest_memory_files, ingest_subagent_logs

    _ensure_initialized()
    conn = get_connection(DB_PATH)
    try:
        # Always ingest main history
        stats = ingest_history(conn, HISTORY_PATH, STATE_PATH)
        click.echo(
            f"History — Ingested: {stats['ingested']}, "
            f"Skipped: {stats['skipped']}, "
            f"Errors: {stats['errors']}"
        )

        if memory or ingest_all:
            mstats = ingest_memory_files(conn, PROJECTS_DIR)
            click.echo(
                f"Memory  — Ingested: {mstats['ingested']}, "
                f"Skipped: {mstats['skipped']}, "
                f"Errors: {mstats['errors']}"
            )

        if subagents or ingest_all:
            sstats = ingest_subagent_logs(conn, PROJECTS_DIR)
            click.echo(
                f"Agents  — Ingested: {sstats['ingested']}, "
                f"Skipped: {sstats['skipped']}, "
                f"Errors: {sstats['errors']}"
            )

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        conn.close()


@main.command()
@click.option("--max-batches", default=10, help="Max batches to process (cost control)")
@click.option("--batch-size", default=10, help="Entries per batch")
def distill(max_batches, batch_size):
    """Run distillation over undistilled entries."""
    from cortex.db import get_connection
    from cortex.distill import distill as do_distill

    _ensure_initialized()
    conn = get_connection(DB_PATH)
    stats = do_distill(conn, max_batches=max_batches, batch_size=batch_size)
    click.echo(
        f"Batches: {stats['batches']}, "
        f"Distillations: {stats['distillations']}, "
        f"Entries processed: {stats['entries_processed']}, "
        f"Errors: {stats['errors']}"
    )
    conn.close()


@main.command("eval")
@click.option("--generate", is_flag=True, help="Auto-generate retrieval eval cases from existing entries")
@click.option("--seed-qa", is_flag=True, help="Generate Q&A eval cases from curated memory entries")
@click.option("--top-k", default=5, help="Results per query to evaluate")
@click.option("--history", is_flag=True, help="Show eval trend over time")
def run_eval(generate, seed_qa, top_k, history):
    """Run evaluation suite against current knowledge base."""
    from cortex.db import get_connection
    from cortex.eval import (
        EVAL_CASES_PATH,
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

    report = run_eval(conn, cases, top_k=top_k)
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
@click.option("--remove-case", type=int, default=None, help="Remove eval case by index (0-based)")
@click.option("--adjust-confidence", nargs=2, type=click.Tuple([int, float]), default=None,
              help="Adjust entry confidence: ENTRY_ID NEW_CONFIDENCE")
def improve(diagnose, update_case, query, keywords, remove_case, adjust_confidence):
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
        click.echo(f"Entry {int(entry_id)} confidence → {new_conf}")
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
