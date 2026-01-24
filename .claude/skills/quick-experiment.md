---
name: quick-experiment
description: Run a quick Python experiment without creating a full tool structure
---

# Quick Experiment Skill

For rapid prototyping when you don't need a full tool structure.

## When to Use

- Testing a quick idea
- One-off data exploration
- API testing
- Validating a concept before building a full tool

## Pattern: Inline Script

Create a standalone script in `tools/scratch/` with PEP 723 metadata:

```python
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "rich",
# ]
# ///
"""Quick experiment: {description}"""

from rich import print

# Experiment code here
result = ...

print(f"[green]Result:[/] {result}")
```

Run with:
```bash
uv run tools/scratch/{experiment}.py
```

## Pattern: REPL-style

For interactive exploration:

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('data/sample.csv')
print(df.describe())
"
```

## Output

- Print results to console for quick review
- If worth keeping, save to `outputs/experiments/`
- If it grows into something useful, promote to a full tool

## Cleanup

Scratch experiments in `tools/scratch/` can be deleted freely. Only promote valuable ones to proper tool folders.
