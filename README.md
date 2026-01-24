# AI Playground 🧪

Quick experimentation repo for Python tools and ideas. Optimized for **Claude Code Cloud**.

## Quick Start

```bash
# Create a new tool
"Create a tool that fetches BTC price and plots 30-day history"

# Run an experiment
"Quick experiment: test if yfinance still works with Polish tickers"

# Share results
"Create a gist with the tool and output chart"
```

## Structure

```
tools/           # Each tool in its own folder
├── {tool-name}/
│   ├── main.py      # Entry point with PEP 723 deps
│   └── README.md    # What it does
├── scratch/         # Quick experiments (disposable)

shared/          # Reusable utilities
data/            # Sample/test data
outputs/         # Generated charts, reports
```

## Features

- **uv** for fast Python package management
- **PEP 723** inline script metadata (no separate requirements.txt)
- **Skills** for consistent tool creation
- **GitHub Gist** integration for sharing

## Skills Available

| Skill | Purpose |
|-------|---------|
| `create-tool` | Create new tool with proper structure |
| `share-gist` | Share code/outputs via GitHub Gist |
| `quick-experiment` | Fast prototyping without full structure |
| `finance-experiment` | Quant/finance patterns and snippets |

## Example Tool

```python
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "rich"]
# ///
"""Fetch and display something."""

from rich import print
import httpx

def main():
    resp = httpx.get("https://api.example.com/data")
    print(f"[green]Result:[/] {resp.json()}")

if __name__ == "__main__":
    main()
```

Run: `uv run tools/my-tool/main.py`

## License

MIT - experiment freely!
