# Cortex TODOs

## v0.2

- [x] **Skill.md for Claude Code** — done, lives at `~/.claude/skills/cortex/SKILL.md`
- [x] **Memory file ingestion** — done, `cortex ingest --memory` (45 files ingested)
- [ ] **Feedback loop** (`cortex feedback`) — track whether surfaced knowledge was helpful. Schema has the table; need CLI command + Skill.md integration to close the loop.
- [ ] **Subagent log ingestion** — `cortex ingest --subagents` code is written but untested on full dataset (813 files). Run and eval to check noise impact.
- [ ] **LLM-as-judge evals** (Gemini feedback) — replace keyword matching with rubric-based scoring. Measure faithfulness and relevance on 1-5 scale instead of binary keyword hits. Critical as knowledge base grows.
- [ ] **Eval case lifecycle** (Gemini feedback) — auto-retire cases that pass 10x in a row; auto-generate adversarial cases every ~100 new entries.

## v0.3

- [ ] **Multi-view distillation** (Codex feedback) — instead of flat text summaries, distill into structured types: atomic facts, principles, triggers+actions, caveats. Each with provenance links.
- [ ] **Factorized confidence** (Codex feedback) — replace single float with: evidence strength, recency, scope (global vs project), stability. Composite score per query context.
- [ ] **Multi-model distillation** — Mobius-style competition where 2-3 models distill independently, best output wins.
- [ ] **Meta-distillation** — re-distill existing patterns every ~500 entries to find meta-patterns. Schema supports it via lineage; needs uncapped depth + cycle detection.
- [ ] **Small-to-big retrieval** (Gemini feedback) — store pattern snippets as vectors but link to full session context. Retrieve snippet, feed parent context to LLM.

## v1.0 — Skill Formation

- [ ] **Skill formation layer** (Codex feedback, highest impact) — turn repeated distillation patterns into executable artifacts (prompts, checklists, code snippets) that get actively injected into future sessions. "Recall -> Apply -> Measure -> Refine" loop. This is what makes the system compound behavior, not just data.
- [ ] **Knowledge Reuse Rate** (Gemini feedback) — track whether retrieved knowledge actually gets used in session output. The real utility metric.
- [ ] **Model-specific format adapters** (Codex feedback) — prompt templates per model for how memory is surfaced (system vs tool vs context).

## Someday

- [ ] **MCP server** — expose cortex query/write as MCP tools for non-Claude LLMs
- [ ] **Knowledge graph UI** — adapt parallel-feud's graph.html for Cortex entries
- [ ] **Cross-project linking** — surface connections between entries from different projects
- [ ] **Auto-distill on light sessions** — use session token counts from ~/.claude/usage-data/session-meta/ to trigger distillation after low-usage sessions
