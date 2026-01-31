# AI Playground

Quick experimentation repo for Python tools and ideas. Optimized for Claude Code Cloud.

**Defaults:** Python, uv package manager, quant research focus.

**Core rule:** Build what user asks. Test if code works (actually run the code and verify results).

**Skill rule:** NEVER build from scratch what a skill already does. Before implementing, check `.claude/skills/` for applicable skills. Use them.

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
├── outputs/                  # Generated outputs, reports, charts
└── .claude/skills/          # Project-specific skills
```

## Python Environment

- **Always use uv** for package management, never pip directly
- Run scripts: `uv run python tools/{tool}/main.py`
- Add dependencies: `uv add <package>` (project-wide) or use inline script metadata
- For tool-specific deps, use PEP 723 inline metadata in the script

### PEP 723 Inline Script Metadata (Preferred for Tools)

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2.0",
#     "requests",
# ]
# ///
```

Then run with: `uv run tools/{tool}/main.py`

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

## Skill Selection (Mandatory)

Before starting any non-trivial task, match the task to available skills:

| Task Type | Required Skill | Never Do Instead |
|-----------|---------------|------------------|
| Read a research paper | `paper-reading` (3-pass) | Raw WebFetch with ad-hoc prompts |
| Fetch market/financial data | `market-data-fetcher` | Raw yfinance/requests |
| Quant research specification | `qrd` | Jump straight to coding |
| Statistical hypothesis testing | `statistical-analysis` | Hand-code t-tests/bootstrap |
| Library documentation lookup | `context7` (MCP or CLI) | Rely on training data |
| PDF document analysis | `pdf-skill` + `research-paper-analyst` agent | Manual text extraction |
| Data profiling | `exploratory-data-analysis` | Ad-hoc pandas describe() |
| Code review | `/llm-external-review:code` | Self-review only |
| Multi-perspective analysis | `/council:ask` or `/council:debate` | Single-model answer |

### Enforcement

1. At task start, list which skills apply (even if the answer is zero)
2. If a skill exists for a subtask, use it — do not reimplement
3. The `forced-eval` hook checks this — do not bypass it
4. If skipping an applicable skill, state why explicitly

## Session Memory

Persist knowledge across ephemeral cloud sessions using `docs/memory/`:
- `decisions.md` — Architectural and design decisions
- `patterns.md` — Recurring patterns and conventions
- `issues.md` — Known issues and workarounds

**Convention:** At end of significant sessions, save key learnings to these files.

## MCP Alternatives (Cloud)

MCP servers don't work on Claude Remote. Use these alternatives:

| MCP | Alternative |
|-----|-------------|
| `fetch` | Built-in `WebFetch`/`WebSearch`, or `curl` |
| `sequential-thinking` | Native extended thinking + `forced-eval` hook |
| `memory` | `docs/memory/` directory (decisions.md, patterns.md, issues.md) |
| `firecrawl` | `uv run python -c "import httpx; ..."` + BeautifulSoup |
| `postgres` | `uv run python` with psycopg2/sqlalchemy + `sql-patterns` skill |

## Multi-Model Review

Use external AI models for second opinions:
- `/review` — Code review via Codex/Gemini
- `/compare` — Multi-model review comparison
- `/architecture` — Architecture review (Gemini 1M context)
- `/security` — Security-focused review (Codex)
- `/ask` — Multi-model question answering
- `/debate` — Structured multi-model debate
- `/decide` — Decision support with pros/cons from multiple models
- `/brainstorm` — Collaborative brainstorming across models

## Compound Engineering Workflow

**Plugin:** `compound-engineering@every-marketplace` — 80% planning and review, 20% execution. Each unit of work should make subsequent units easier.

### Core Loop: `/workflows:plan` → `/workflows:work` → `/workflows:review` → `/workflows:compound`

| Command | What it does |
|---------|-------------|
| `/workflows:brainstorm` | Explore requirements and approaches through collaborative dialogue. Creates `docs/brainstorms/` doc. |
| `/workflows:plan` | Transform feature descriptions into structured plans. Runs parallel research agents (repo-research, best-practices, framework-docs). Creates `docs/plans/` doc. |
| `/workflows:work` | Execute plans with TodoWrite tracking, incremental commits, optional reviewer agents. Creates PR when done. |
| `/workflows:review` | Multi-agent code review (13+ parallel reviewers: security-sentinel, performance-oracle, architecture-strategist, etc.). Creates prioritized P1/P2/P3 todos. |
| `/workflows:compound` | Document solved problems as searchable knowledge in `docs/solutions/[category]/`. Runs 6 parallel subagents. |

Additional utilities: `/deepen-plan`, `/plan_review`, `/triage`, `/resolve_parallel`, `/changelog`

### Quant Research (RALPH Loop)
1. **Research** → `paper-reading` skill + `/council:ask` (form hypothesis)
2. **Act** → `qrd` spec → `market-data-fetcher` for data → `/workflows:plan` + `/workflows:work` in `tools/{name}/`
3. **Learn** → `statistical-analysis` skill + `/workflows:review` results against acceptance criteria
4. **Plan** → Refine approach or pivot
5. **Hypothesize** → `/workflows:compound` learnings, start next iteration

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

### Paper Replication Workflow

When replicating an academic paper:

1. **`paper-reading`** — Three-pass extraction:
   - Pass 1: Quick assessment (title, abstract, methodology type)
   - Pass 2: Technical summary (ALL formulas, parameters, constraints)
   - Pass 3: Critical analysis (assumptions, limitations, acceptance criteria)
2. **`qrd`** — Create spec with measurable acceptance criteria from paper's reported results (e.g., "total trades ~140, beta ~0.06, Sharpe ~0.33")
3. **`market-data-fetcher`** — Data acquisition (cached, with fallbacks)
4. **Implement** against the spec, not from memory of the paper
5. **Run and validate** against acceptance criteria — flag any metric >2x off
6. **`statistical-analysis`** — Proper significance tests with APA reporting
7. **`/llm-external-review:code`** — External review for logic errors
8. **`/workflows:compound`** — Document learnings

**Critical:** Papers bury constraints in single sentences (e.g., "80% remains in cash"). The `paper-reading` skill's structured extraction catches these. Raw WebFetch does not.

## Creating Gists

When asked to share via gist:

```bash
# Single file
gh gist create tools/{tool}/main.py --public -d "Description"

# Multiple files
gh gist create tools/{tool}/main.py tools/{tool}/README.md --public -d "Description"

# With output
gh gist create tools/{tool}/main.py outputs/{result}.md --public -d "Tool + Results"
```

Return the gist URL to the user.

## Output Guidelines

- Save charts/plots to `outputs/` as PNG or SVG
- Save reports to `outputs/` as Markdown
- Print summary to stdout for quick review
- Include timestamp in output filenames: `{name}_{YYYYMMDD_HHMMSS}.png`

## Code Style

- Type hints required
- Docstrings for public functions
- Use `rich` for console output when appropriate
- Use `typer` or `click` for CLI tools
- Keep tools self-contained (single main.py when possible)

## Testing Ideas

When experimenting:
1. Start simple, iterate fast
2. Print intermediate results
3. Save interesting outputs
4. Document what worked in the tool's README

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

### Parallel Task Templates

For multi-symbol analysis:
```
& Fetch WIG20 data and create momentum analysis
& Fetch SPY data and create momentum analysis
& Fetch BTC-USD data and create momentum analysis
```

For multi-perspective review:
```
& /workflows:review --focus security
& /workflows:review --focus performance
```
