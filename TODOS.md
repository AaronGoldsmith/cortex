# Cortex TODOs

## v0.2

- [ ] **Feedback loop** (`cortex feedback`) — track whether surfaced knowledge was helpful. Schema has the table; need CLI command + Skill.md integration to close the loop. Depends on: v0.1 working.

- [ ] **Skill.md for Claude Code** — `.claude/skills/cortex/SKILL.md` that teaches sessions to query Cortex before tasks and write notable learnings. Lightweight alternative to MCP server.

- [ ] **Backfill importers** — import existing knowledge from parallel-feud discoveries.json, .claude/memory/ files, and Sol exports. Each is a small script.

## v0.3

- [ ] **Multi-model distillation** — Mobius-style competition where 2-3 models distill independently, best output wins. Reduces single-model bias.

- [ ] **Meta-distillation** — allow distiller to read first-level distillations and extract higher-order patterns. Schema supports it via lineage; needs uncapped depth + cycle detection.

## Someday

- [ ] **MCP server** — expose cortex query/write as MCP tools for non-Claude LLMs
- [ ] **Knowledge graph UI** — adapt parallel-feud's graph.html for Cortex entries
- [ ] **Cross-project linking** — surface connections between entries from different projects
