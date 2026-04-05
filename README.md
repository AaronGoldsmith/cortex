# Cortex

Persistent, model-agnostic knowledge ledger that compounds intelligence across AI sessions.

Cortex ingests session history from multiple AI coding tools, embeds it locally, and serves it back via semantic search. Over time, a distillation layer extracts recurring patterns and insights. The result is a personal knowledge base that grows smarter the more you use AI tools.

## Quick Start

```bash
cortex init          # create ~/.cortex/ and initialize the DB
cortex ingest        # pull in Claude Code session history
cortex query "how do I deploy to fly.io"
```

## How It Works

```
  AI Sessions ──> Ingest ──> Ledger (SQLite + embeddings)
                                │
                          Distill (LLM)
                                │
                          Patterns & Insights
                                │
                    Query ◄─────┘
```

1. **Ingest** parses session files from AI tools and stores them as entries with local embeddings (all-MiniLM-L6-v2, zero API cost).
2. **Distill** clusters raw entries, sanitizes secrets, sends batches to an LLM, and writes back distillations with lineage to source entries.
3. **Query** runs semantic search across both entries and distillations, ranking by similarity, confidence, and recency.

## Supported Sources

| Tool | Format | Flag |
|------|--------|------|
| Claude Code | `~/.claude/history.jsonl` + memory files + subagent logs | `--source claude` (default) |
| Goose (Block) | `sessions.db` SQLite database | `--source goose` |

```bash
cortex ingest                    # Claude Code only (default)
cortex ingest --source goose     # Goose only
cortex ingest --source all       # all detected tools
cortex ingest --all              # Claude history + memory files + subagent logs
```

Adding a new source means writing a provider module in `cortex/providers/` that implements `detect()` and `iter_entries()`.

## Commands

### Ingest

```bash
cortex ingest                          # Claude history (default)
cortex ingest --source goose           # Goose sessions
cortex ingest --source all             # all detected providers
cortex ingest --memory                 # also ingest Claude memory .md files
cortex ingest --subagents              # also ingest Claude subagent logs
cortex ingest --all                    # history + memory + subagents
cortex ingest --backfill-turns         # resolve turn_index on existing entries
cortex ingest --background             # run as background process (for hooks)
```

Ingestion is idempotent -- running it multiple times produces the same result.

### Query

```bash
cortex query "sqlite WAL mode gotchas"
cortex query "discord bot tokens" -k 10
cortex query "deployment" --project sol
```

Returns entries ranked by semantic similarity, confidence score, and recency.

### Write

```bash
cortex write "sqlite-vec requires enable_load_extension(True) before loading" \
  --type observation --model claude --project cortex

cortex write "User prefers lightweight scripts over MCP servers" \
  --type correction --model claude
```

Entry types: `raw`, `observation`, `recommendation`, `correction`, `pattern`

### Distill

```bash
cortex distill --dry-run               # preview without spending tokens
cortex distill --max-batches 5         # limit LLM calls
cortex distill --context-window 0      # disable conversation context (default: 3)
```

Each batch = 1 LLM API call. Secrets are redacted before anything leaves the machine.

### Trace

```bash
cortex trace 42                        # show distillation D42's source entries
cortex trace 42 --window 10            # wider conversation context
```

Follow a distillation back to the raw entries and original conversation that produced it.

### Eval

```bash
cortex eval --generate                 # auto-generate eval cases from entries
cortex eval --seed-qa                  # generate Q&A cases from memory entries
cortex eval                            # run eval suite
cortex eval --history                  # show score trend over time
```

### Status

```bash
cortex status
```

Shows entry count, distillation count, undistilled entries, top projects, DB size.

## Using Cortex from AI Agents

Cortex is designed to be called by AI agents during their sessions. The `/cortex` skill (if installed) teaches agents when and how to use it, but the core patterns are:

### Querying at session start

Before starting a non-trivial task, agents should check for prior knowledge:

```bash
cortex query "brief description of the task or domain" 2>/dev/null
```

This surfaces past learnings, debug patterns, and gotchas from previous sessions -- even sessions with different AI tools.

### Writing discoveries

When an agent encounters something reusable across sessions or projects:

```bash
cortex write "what was learned" --type observation --model <model-name> --project <project>
```

**Good writes:** debugging insights, cross-project patterns, tool gotchas, things that failed and why.

**Skip:** trivial observations, user preferences (use Claude Memory for those), ephemeral task state.

### Cortex vs Claude Memory

| What | Cortex | Claude Memory |
|------|--------|---------------|
| Cross-project patterns | Yes | No |
| Debugging techniques | Yes | No |
| Tool/library gotchas | Yes | No |
| User preferences/style | No | Yes |
| Project-specific conventions | No | Yes |
| Architecture decisions | No | Yes |

**Rule of thumb:** Claude Memory is *who the user is* and *how this project works*. Cortex is *what we've learned about how things work in general*.

### Agent-driven maintenance

Two specialized agents keep Cortex healthy:

**signal-distiller** -- extracts patterns from raw entries via LLM distillation:
```bash
cortex distill --max-batches 5 --batch-size 10
```

**eval-auditor** -- diagnoses weak eval results and fixes them:
```bash
cortex improve --diagnose              # get structured failure data
cortex improve --update-case 3 --query "better query" --keywords "k1,k2"
cortex improve --adjust-confidence 42 1.5
```

## Prerequisites

- **Python 3.11+**
- **Claude Code CLI** — `cortex distill` shells out to `claude` as a subprocess to run the `signal-distiller` agent. Install Claude Code and ensure the `claude` binary is on your PATH before running distillation.
- **sentence-transformers** — the embedding model (`all-MiniLM-L6-v2`) downloads automatically on first use (~80MB). No API key needed.

## Installation

```bash
pip install -e .
```

## Recommended Hooks

Cortex works best when ingestion happens automatically. Add these to your Claude Code settings (`~/.claude/settings.json`) under `"hooks"`:

### Auto-ingest on session end

Runs `cortex ingest` in the background every time a Claude Code session closes, keeping the ledger up to date without manual effort.

```json
"SessionEnd": [
  {
    "hooks": [
      {
        "type": "command",
        "command": "cortex ingest --background",
        "timeout": 10
      }
    ]
  }
]
```

### Auto-run tests after edits

Runs pytest after any file edit or write, catching regressions immediately.

```json
"PostToolUse": [
  {
    "matcher": "Edit|Write",
    "hooks": [
      {
        "type": "command",
        "command": "if [ -f pytest.ini ] || [ -f pyproject.toml ] || [ -d tests ]; then pytest --tb=short -q 2>/dev/null || echo 'TESTS FAILED'; fi",
        "timeout": 30,
        "statusMessage": "Running tests..."
      }
    ]
  }
]
```

## Invariants

- **Append-only**: entries are never deleted or mutated. Corrections are new entries.
- **Local-first**: the DB never leaves the machine. Only the distiller sends content to external LLMs, after sanitization.
- **Model-agnostic**: any LLM can read/write via the CLI.
- **Idempotent ingestion**: re-running ingest produces the same result.
- **Lineage tracking**: every distillation links back to its source entries.
