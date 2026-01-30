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
