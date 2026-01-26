---
description: "Parallel Research Task Template"
---

# Parallel Research Template

This template guides parallel execution of research tasks in Claude Code Web.

## Usage

Use the `&` prefix to spawn parallel tasks. Each runs in a separate cloud session with results merged upon completion.

## Multi-Symbol Market Analysis

```
& Fetch WIG20 price data (2024) and create momentum analysis
& Fetch SPY price data (2024) and create momentum analysis
& Fetch BTC-USD price data (2024) and create momentum analysis
```

## Multi-Source Data Comparison

```
& Fetch PKO from Stooq, analyze data quality
& Fetch PKO from Yahoo, analyze data quality
& Compare results and report discrepancies
```

## Multi-Perspective Code Review

```
& /workflows:review --focus security
& /workflows:review --focus performance
& /workflows:review --focus conventions
```

## Parallel Bug Resolution

```
& /resolve_parallel #123
& /resolve_parallel #124
& /resolve_parallel #125
```

## Research + Documentation

```
& Research current React 19 patterns and create summary
& Research Next.js 15 app router and create summary
& Create combined best practices guide
```

## Tips

1. **Independent tasks**: Use `&` for tasks that don't depend on each other
2. **Merge results**: Final session combines outputs from parallel tasks
3. **Shared data**: Upload data files to repo before parallel execution
4. **Output files**: Each task should save to unique output paths:
   - `outputs/WIG20_analysis_20240301.md`
   - `outputs/SPY_analysis_20240301.md`

## Sample Data Available

For offline/restricted network scenarios:
- `data/samples/WIG20.csv` - Polish WIG20 index
- `data/samples/SPY.csv` - S&P 500 ETF
- `data/samples/BTC-USD.csv` - Bitcoin/USD

## Combining Results

After parallel tasks complete:

```python
# Merge analysis results
from pathlib import Path
import pandas as pd

analyses = list(Path('outputs').glob('*_analysis_*.md'))
for f in analyses:
    print(f"=== {f.name} ===")
    print(f.read_text()[:500])
```
