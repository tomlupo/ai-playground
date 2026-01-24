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
