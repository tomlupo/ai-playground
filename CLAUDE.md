# AI Playground

Quick experimentation repo for Python tools and ideas. Optimized for Claude Code Cloud.

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

## Compound Engineering Workflow

This repo uses compound-engineering for knowledge amplification across sessions.

### Core Commands
- `/workflows:brainstorm` - Explore approaches with multi-perspective analysis
- `/workflows:plan` - Create structured implementation plan
- `/workflows:work` - Execute systematically with verification
- `/workflows:review` - Multi-agent code review (security, performance, correctness)
- `/workflows:compound` - Document solved problems for future reuse

### For Signal Research (RALPH Loop)
1. **Research** → `/workflows:brainstorm` (form hypothesis)
2. **Act** → `/workflows:plan` + `/workflows:work` (implement)
3. **Learn** → `/iterate check` (validate against criteria)
4. **Plan** → Refine approach or pivot
5. **Hypothesize** → Start next iteration

### Context Preservation
```bash
/handoff  # Save context before /clear or session end
/resume   # Continue from saved state in new session
```

### Headless Quant Research Pattern
```bash
# Create specification
/qrd momentum-signal

# Execute autonomously
/workflows:work --spec outputs/qrd/momentum-signal.md

# Document learnings
/workflows:compound --topic momentum-calculation
```

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
