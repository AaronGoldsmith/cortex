# Cortex

Persistent, model-agnostic knowledge ledger that compounds intelligence across AI sessions.

## Architecture

Three components forming a loop:
- **Ledger** — append-only SQLite + sqlite-vec store at `~/.cortex/cortex.db`
- **Distiller** — spare-tokens job that extracts patterns from raw entries
- **Tap** — semantic search query interface (`cortex query`)

## Invariants

- Append-only: entries are never deleted or mutated. Corrections are new entries referencing originals.
- Local-first: the DB never leaves the machine. Only the distiller sends content to external LLMs, after sanitization.
- Model-agnostic: any LLM can read/write via the CLI. No Anthropic-specific dependencies.
- Idempotent ingestion: running `cortex ingest` multiple times produces the same result.
- Lineage tracking: every distillation links back to the source entries that produced it.
- Conversation context on demand (Option B): assistant responses are not stored in the DB. Conversation context is pulled at runtime from session JSONL files, referenced via turn_index on entries.

## Tech Stack

- Python 3.11+
- SQLite with WAL mode + foreign keys
- sqlite-vec for vector similarity search
- sentence-transformers (all-MiniLM-L6-v2) for local embeddings — zero API cost
- Click for CLI

## CLI Commands

- `cortex init` — create ~/.cortex/ and initialize DB
- `cortex write` — add an entry to the ledger
- `cortex query` — semantic search across entries and distillations
- `cortex ingest` — ingest session history from AI tools. Flags: `--source` (claude|goose|all), `--memory`, `--subagents`, `--all`, `--backfill-turns`
- `cortex distill` — cluster raw entries, sanitize, call LLM, write distillations. Flags: `--context-window N`
- `cortex eval` — run evaluation suite. Flags: `--generate`, `--seed-qa`, `--history`
- `cortex improve` — diagnostic tools for the eval-auditor agent. Flags: `--diagnose`, `--update-case`, `--remove-case`, `--adjust-confidence`
- `cortex trace` — show the conversation context around a specific entry or distillation
- `cortex status` — show entry counts, last ingest/distill times, DB size

## Agents

- **eval-auditor** (`.claude/agents/eval-auditor.md`): If eval results show low relevance or precision, spawn this agent to diagnose failures, fix eval cases, add missing knowledge, and adjust confidence scores.

## Project Structure

```
cortex/
  __init__.py
  cli.py        — Click CLI entry point
  config.py     — Paths, model version, constants
  db.py         — Schema, migrations, connection management
  embedder.py   — sentence-transformers wrapper
  ingest.py     — Orchestrator + Claude-specific memory/subagent ingest
  distill.py    — Clustering, sanitization, LLM calls, write-back
  query.py      — Vector search + ranking
  sanitize.py   — Secret detection/redaction before LLM calls
  sessions.py   — Runtime conversation context loader from session JSONL files
  providers/
    __init__.py  — Provider registry (get_provider, discover_providers)
    base.py      — IngestEntry NamedTuple + IngestProvider Protocol
    claude.py    — Claude Code history.jsonl provider
    goose.py     — Goose (Block) sessions.db provider
tests/
  test_core_loop.py
  test_eval.py
  test_providers.py
  test_sessions.py
```

## Testing

Run tests with: `pytest tests/ -v`

## Executing CLI Commands
You do *not* need to prefix with `python -m`. Execute with `cortex <command> <arg>` directly