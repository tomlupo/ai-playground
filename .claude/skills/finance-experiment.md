---
name: finance-experiment
description: Quick finance/quant experiments with common patterns for returns, performance, and data fetching
---

# Finance Experiment Skill

Common patterns for quantitative finance experiments.

## Quick Data Fetch

```python
# /// script
# dependencies = ["yfinance", "pandas"]
# ///
import yfinance as yf
import pandas as pd

# Single ticker
msft = yf.Ticker("MSFT")
hist = msft.history(period="1y")

# Multiple tickers
tickers = yf.download(["AAPL", "GOOGL", "MSFT"], period="1y")
prices = tickers["Adj Close"]
returns = prices.pct_change().dropna()
```

## Returns Analysis

```python
# /// script
# dependencies = ["pandas", "numpy", "quantstats"]
# ///
import quantstats as qs

# Generate tearsheet
qs.reports.html(returns, benchmark="SPY", output="outputs/tearsheet.html")

# Quick stats
print(qs.stats.sharpe(returns))
print(qs.stats.max_drawdown(returns))
print(qs.stats.cagr(returns))
```

## Performance Attribution Pattern

```python
# /// script
# dependencies = ["pandas", "numpy"]
# ///
import pandas as pd
import numpy as np

def brinson_attribution(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    """Single-period Brinson attribution."""
    allocation = ((portfolio_weights - benchmark_weights) * benchmark_returns).sum()
    selection = (benchmark_weights * (portfolio_returns - benchmark_returns)).sum()
    interaction = ((portfolio_weights - benchmark_weights) * (portfolio_returns - benchmark_returns)).sum()
    
    return {
        "allocation": allocation,
        "selection": selection,
        "interaction": interaction,
        "total": allocation + selection + interaction,
    }
```

## Visualization Pattern

```python
# /// script
# dependencies = ["matplotlib", "pandas"]
# ///
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Cumulative returns
(1 + returns).cumprod().plot(ax=axes[0, 0], title="Cumulative Returns")

# Rolling Sharpe
rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
rolling_sharpe.plot(ax=axes[0, 1], title="Rolling Sharpe (1Y)")

# Drawdown
cumulative = (1 + returns).cumprod()
drawdown = cumulative / cumulative.cummax() - 1
drawdown.plot(ax=axes[1, 0], title="Drawdown")

# Monthly returns heatmap
# ...

plt.tight_layout()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"outputs/analysis_{timestamp}.png", dpi=150)
plt.close()
```

## Install Finance Dependencies

If needed:
```bash
uv add yfinance quantstats empyrical scipy
```

Or use inline metadata in individual scripts.
