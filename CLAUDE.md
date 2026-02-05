# AI Playground

Quick experimentation repo for Python tools and ideas. Optimized for Claude Code Cloud.

**Defaults:** Python, uv package manager, quant research focus.

**Core rule:** Build what user asks. Test if code works (actually run the code and verify results).

**Output:** Always produce output.

- **Save artifacts** to `output/`: charts/plots (PNG, SVG), reports (Markdown, HTML), data (JSON). Use timestamped filenames: `{name}_{YYYYMMDD_HHMMSS}.png`.
- **Print** a short summary to stdout for quick review.
- **Always send** a summary report with the `/gist-report` skill.

## Project Structure

```
ai-playground/
├── tools/                    # Each tool gets its own subfolder
│   └── {tool-name}/
│       ├── main.py          # Entry point
│       ├── pyproject.toml   # Tool-specific dependencies (optional)
│       └── README.md        # What it does, how to use
├── shared/                   # Shared utilities across tools
├── data/                     # Sample data for experiments
├── output/                  # Generated output, reports, charts
└── .claude/skills/          # Project-specific skills
```

## Python Environment

- **Always use uv** for package management, never pip directly
- Run scripts: `uv run python tools/{tool}/main.py`
- Add dependencies: `uv add <package>` (project-wide) or use inline script metadata
- For tool-specific deps, use PEP 723 inline metadata in the script
- **Optional extras**: `uv sync --extra finance` (quant stack: yfinance, quantstats, ffn, vectorbt, cvxpy, riskfolio-lib, arch, statsmodels, ta), `uv sync --extra ml` (scikit-learn, scipy)

## Creating New Tools

When asked to create a new tool:

1. Create subfolder: `tools/{descriptive-name}/`
2. Create `main.py` with PEP 723 metadata for dependencies
3. Create `README.md` with purpose and usage
4. Run and test the tool
5. If requested, create a gist with the results

## Running Tools

```bash
# Run a tool
uv run tools/{tool-name}/main.py

# Run with arguments
uv run tools/{tool-name}/main.py --input data/sample.csv

# Run tests if present
uv run pytest tools/{tool-name}/
```

## Coding Rules

Generic coding standards are in `.claude/rules/`:

- `code-quality.md` — Constants, naming, DRY, SRP, encapsulation
- `general-rules.md` — Task workflow, communication, problem-solving
- `work-organization.md` — Directory structure, scratch-first rule
- `python-rules.md` — uv, Black, type hints, formatting conventions

## Session Memory

Persist knowledge across ephemeral cloud sessions using `docs/memory/`:

- `decisions.md` — Architectural and design decisions
- `patterns.md` — Recurring patterns and conventions
- `issues.md` — Known issues and workarounds

**Convention:** At end of significant sessions, save key learnings to these files.

## MCP Alternatives (Cloud)

MCP servers don't work on Claude Remote. Use these alternatives:

| MCP | Alternative |
| --- | --- |
| `fetch` | Built-in `WebFetch`/`WebSearch`, or `curl` |
| `sequential-thinking` | Native extended thinking + `forced-eval` hook |
| `memory` | `docs/memory/` directory (decisions.md, patterns.md, issues.md) |
| `firecrawl` | crawl4ai |


## Compound Engineering Workflow

80% planning and review, 20% execution. Each unit of work should make subsequent units easier.

### Core Loop

| Command | What it does |
| --- | --- |
| `/workflows:brainstorm` | Explore requirements and approaches through collaborative dialogue. Creates `docs/brainstorms/` doc. |
| `/workflows:plan` | Transform feature descriptions into structured plans. Runs parallel research agents (repo-research, best-practices, framework-docs). Creates `docs/plans/` doc. |
| `/workflows:work` | Execute plans with TodoWrite tracking, incremental commits, optional reviewer agents. Creates PR when done. |
| `/workflows:review` | Multi-agent code review (13+ parallel reviewers: security-sentinel, performance-oracle, architecture-strategist, etc.). Creates prioritized P1/P2/P3 todos. |
| `/workflows:compound` | Document solved problems as searchable knowledge in `docs/solutions/[category]/`. Runs 6 parallel subagents. |

### Headless Quant Research Pattern

```bash
# Create specification
/qrd momentum-signal

# Plan with parallel research
/workflows:plan momentum signal implementation

# Execute the plan
/workflows:work

# Multi-agent review
/workflows:review

# Compound learnings
/workflows:compound
```

## Creating Gists

When asked to share via gist:

```bash
# Single file
gh gist create tools/{tool}/main.py --public -d "Description"

# Multiple files
gh gist create tools/{tool}/main.py tools/{tool}/README.md --public -d "Description"

# With output
gh gist create tools/{tool}/main.py output/{result}.md --public -d "Tool + Results"
```

Return the gist URL to the user.

## Output Guidelines

- Save charts/plots to `output/` as PNG or SVG
- Save reports to `output/` as Markdown
- Print summary to stdout for quick review
- Include timestamp in output filenames: `{name}_{YYYYMMDD_HHMMSS}.png`

## Code Style

- Type hints required
- Docstrings for public functions
- Use `rich` for console output when appropriate
- Use `typer` or `click` for CLI tools
- Keep tools self-contained (single main.py when possible)

## Claude Code Web Optimization

This repository is optimized for Claude Code Web (cloud-based sessions).

### Environment Setup

1. **Network Access**: Use "Full Internet" for market-data-fetcher access to:
  
  - `stooq.pl` (Polish markets)
  - `api.nbp.pl` (PLN FX rates)
  - `query1.finance.yahoo.com` (Yahoo Finance)
2. **API Keys**: Set in environment configuration (not `.env` file):
  
  - `TIINGO_API_KEY` - Optional, for Tiingo data
  - `FRED_API_KEY` - Optional, for FRED economic data
  - `CONTEXT7_API_KEY` - For context7 skill
3. **SessionStart Hook**: `scripts/setup-env.sh` auto-installs:
  
  - `uv` package manager
  - `gh` GitHub CLI
  - Pre-downloads NLP models

See `docs/claude-code-web-setup.md` for detailed configuration.

### Best Patterns for Cloud

- **Parallel execution**: Use `&` prefix to spawn parallel data fetching tasks
- **QRD specs**: Use `/qrd` skill to create specifications, then execute autonomously
- **PDF processing**: Upload PDFs to repo first, then use pdf-skill
- **Gist sharing**: Use `/gist-report` to share results

### Limitations & Workarounds

- **Network limited**: Use `data/samples/` for cached market data (WIG20, SPY, BTC-USD)
- **MCP servers**: Don't work in Cloud Web. Use `context7` skill CLI instead:
  
  ```bash
  python3 .claude/skills/context7/scripts/context7.py search "react"
  ```
  
- **Local paths**: `/gist-transcript` references local paths, not cloud-compatible