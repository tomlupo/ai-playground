@CLAUDE.md

## Agent Guidelines

All agents should follow the rules in `.claude/rules/` and the conventions in `CLAUDE.md`.

### Available Agents

- **research-paper-analyst** — PDF document analysis combining extraction and structured reading methodology. Use for research papers, financial reports, fund cards, prospectuses.

### Agent Conventions

- Follow `.claude/rules/work-organization.md` for file placement
- Follow `.claude/rules/python-rules.md` for Python code
- Save exploratory work to `scratch/{agent-name}/`
- Save summaries and outputs to appropriate directories
- Use `docs/memory/` to persist cross-session knowledge
