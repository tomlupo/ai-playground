# Decisions Log

Record architectural and design decisions that persist across sessions.

## Format

```
### YYYY-MM-DD: Decision Title
**Context:** Why this decision was needed
**Decision:** What was decided
**Alternatives considered:** What else was evaluated
**Consequences:** Expected impact
```

## Decisions

### 2026-01-30: Use uv for all package management
**Context:** Need consistent Python environment management across tools
**Decision:** Use `uv` exclusively, never raw pip
**Alternatives considered:** pip, poetry, conda
**Consequences:** Faster installs, consistent lockfiles, PEP 723 inline metadata support

### 2026-01-30: Paper replication methodology — use skills, not raw WebFetch
**Context:** Replicating arXiv:2512.12924v1 (hypothesis-driven trading). First implementation produced 6.42% annualized vs paper's 0.55% — completely wrong due to missed capital constraints. Used raw WebFetch instead of available skills.
**Decision:** For paper replications, always use this skill chain: `paper-reading` (3-pass structured extraction) → `qrd` (spec with acceptance criteria) → `market-data-fetcher` (data) → implement → `statistical-analysis` (stats) → `/llm-external-review:code` (review). Never use raw WebFetch for paper reading when `paper-reading` skill exists.
**Alternatives considered:** (1) Raw WebFetch with detailed prompts — failed, missed 80% cash reserve and Eq.17 sizing on first pass. (2) Two-pass WebFetch — worked but redundant given paper-reading skill's built-in three-pass approach.
**Consequences:** Second WebFetch pass caught 80% cash reserve rule, 30d hold cap, Eq.17 sizing formula. Fixed replication: 6.42% → 1.75% annualized. The paper-reading skill would have caught these on the first pass via its structured methodology template.

### 2026-01-30: Always evaluate available skills before implementation
**Context:** 16+ skills available in `.claude/skills/`, none were used during the arXiv replication. The `forced-eval` hook exists to enforce skill evaluation but was not effective.
**Decision:** Before any non-trivial task, scan `.claude/skills/` and select applicable skills. For quant research: paper-reading, qrd, market-data-fetcher, statistical-analysis, context7 are the standard chain.
**Alternatives considered:** Ad-hoc tool selection — leads to reinventing capabilities that already exist (hand-coded stats, raw yfinance, unstructured paper reading).
**Consequences:** Should reduce iteration cycles (v1→v2 fix rounds) and improve first-pass accuracy.
