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

### 2026-01-30: Paper replication methodology — two-pass approach
**Context:** Replicating arXiv:2512.12924v1 (hypothesis-driven trading). First implementation produced 6.42% annualized vs paper's 0.55% — completely wrong due to missed capital constraints.
**Decision:** Always do two WebFetch passes on a paper: (1) full content extraction, (2) targeted re-read for exact quantitative parameters (position sizing formulas, capital reserves, holding limits, trade counts). Cross-check replication output counts against paper's reported values.
**Alternatives considered:** Single-pass reading with "be detailed" prompt. Failed because papers embed critical constraints in single sentences.
**Consequences:** Second pass caught 80% cash reserve rule, 30d hold cap, Eq.17 sizing formula. Fixed replication went from 6.42% → 1.75% annualized, much closer to paper's 0.55%.
