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
- `cortex ingest` — parse Claude Code history.jsonl, deduplicate, embed, store
- `cortex distill` — cluster raw entries, sanitize, call LLM, write distillations
- `cortex status` — show entry counts, last ingest/distill times, DB size

## Project Structure

```
cortex/
  __init__.py
  cli.py        — Click CLI entry point
  config.py     — Paths, model version, constants
  db.py         — Schema, migrations, connection management
  embedder.py   — sentence-transformers wrapper
  ingest.py     — history.jsonl parser + dedup
  distill.py    — Clustering, sanitization, LLM calls, write-back
  query.py      — Vector search + ranking
  sanitize.py   — Secret detection/redaction before LLM calls
tests/
  test_db.py
  test_embedder.py
  test_ingest.py
  test_distill.py
  test_query.py
  test_sanitize.py
```

## Testing

Run tests with: `pytest tests/ -v`
