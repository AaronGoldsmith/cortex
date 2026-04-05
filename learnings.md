# Learnings

## Iteration 1 — 68.3% -> 73.8% precision, 100% relevance (pass)

**What I tried:** Fixed 5 eval cases — 2 bad questions (malformed queries with backticks/weird phrasing), 2 stale project name lookups (blender-arena renamed to materialize), 1 too-narrow keyword set (mobius).

**What happened:** Precision jumped 5.5%. One rewrite (case 46, "don't change" query) initially broke relevance — the rewritten query was too different from the entry it needed to match. Second attempt with closer phrasing fixed it.

**Pattern discovered:** 6 of 9 original failures are project-name lookups where semantic search returns semantically related but wrong-project results. This is a fundamental limitation — semantic embeddings treat "blender project" and "3D modeling" as similar, but the eval wants exact project-name matching. Either need BM25 hybrid search (cortex TODOS.md v0.3) or accept that these cases will always have low precision.

**Failure mode added:** Overly aggressive query rewrites can break relevance. Keep rewrites close to the original phrasing — change keywords, don't change meaning.
