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
def ingest(background):
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

    from cortex.db import get_connection
    from cortex.ingest import ingest_history

    _ensure_initialized()
    conn = get_connection(DB_PATH)
    try:
        stats = ingest_history(conn, HISTORY_PATH, STATE_PATH)
        click.echo(
            f"Ingested: {stats['ingested']}, "
            f"Skipped: {stats['skipped']}, "
            f"Errors: {stats['errors']}"
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
@click.option("--generate", is_flag=True, help="Auto-generate eval cases from existing entries")
@click.option("--top-k", default=5, help="Results per query to evaluate")
def run_eval(generate, top_k):
    """Run evaluation suite against current knowledge base."""
    from cortex.db import get_connection
    from cortex.eval import (
        EVAL_CASES_PATH,
        generate_eval_cases,
        load_eval_cases,
        run_eval,
        save_eval_cases,
    )

    _ensure_initialized()
    conn = get_connection(DB_PATH)

    if generate:
        cases = generate_eval_cases(conn)
        if not cases:
            click.echo("Not enough entries to generate eval cases.")
            conn.close()
            return
        save_eval_cases(EVAL_CASES_PATH, cases)
        click.echo(f"Generated {len(cases)} eval cases → {EVAL_CASES_PATH}")

    cases = load_eval_cases(EVAL_CASES_PATH)
    if not cases:
        click.echo("No eval cases found. Run 'cortex eval --generate' first.")
        conn.close()
        return

    report = run_eval(conn, cases, top_k=top_k)
    click.echo(report.summary())
    conn.close()


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
