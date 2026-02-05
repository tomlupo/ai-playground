# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "matplotlib>=3.8",
#     "yfinance>=0.2",
#     "pyyaml>=6.0",
#     "rich>=13.0",
# ]
# ///
"""
Quant Research Template

Usage:
    uv run tools/_templates/quant-research/main.py --config config.yaml
    uv run tools/_templates/quant-research/main.py --symbol SPY --rsi-period 2 --rsi-threshold 10

See README.md for full documentation.
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

# Add project root to path for shared imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.indicators import cagr, ibs, max_drawdown, rsi, sharpe_ratio

console = Console()

# --- Constants ---
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
OUTPUT_DIR = PROJECT_ROOT / "output"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# --- Data Loading ---
def load_data(symbol: str) -> pd.DataFrame:
    """Load cached data from data/samples/ or download if not available."""
    cached_path = SAMPLES_DIR / f"{symbol}.csv"

    if cached_path.exists():
        console.print(f"[green]Loading cached data:[/] {cached_path}")
        df = pd.read_csv(cached_path, parse_dates=["Date"], index_col="Date")
    else:
        console.print(f"[yellow]Downloading {symbol} data...[/]")
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(period="10y")
        df.index.name = "Date"
        # Cache for next time
        df.to_csv(cached_path)

    return df


# --- Strategy Logic ---
def generate_signals(
    df: pd.DataFrame,
    rsi_period: int = 2,
    rsi_threshold: float = 10,
    ibs_threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Generate trading signals based on strategy rules.

    Modify this function to implement your strategy.
    """
    df = df.copy()

    # Calculate indicators
    df["rsi"] = rsi(df["Close"], period=rsi_period)
    df["ibs"] = ibs(df["High"], df["Low"], df["Close"])

    # Generate signals: 1 = long, 0 = flat
    df["signal"] = 0
    df.loc[(df["rsi"] < rsi_threshold) & (df["ibs"] < ibs_threshold), "signal"] = 1

    # Exit on close the next day (simple implementation)
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


# --- Backtesting ---
def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Run vectorized backtest."""
    df = df.copy()

    # Calculate returns
    df["returns"] = df["Close"].pct_change()
    df["strategy_returns"] = df["position"] * df["returns"]

    # Equity curves
    df["buy_hold_equity"] = (1 + df["returns"]).cumprod()
    df["strategy_equity"] = (1 + df["strategy_returns"]).cumprod()

    return df


# --- Performance Metrics ---
def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate performance metrics."""
    strat_returns = df["strategy_returns"].dropna()
    bh_returns = df["returns"].dropna()

    metrics = {
        "Strategy CAGR": f"{cagr(strat_returns):.2%}",
        "Buy & Hold CAGR": f"{cagr(bh_returns):.2%}",
        "Strategy Sharpe": f"{sharpe_ratio(strat_returns):.2f}",
        "Buy & Hold Sharpe": f"{sharpe_ratio(bh_returns):.2f}",
        "Strategy Max DD": f"{max_drawdown(strat_returns):.2%}",
        "Buy & Hold Max DD": f"{max_drawdown(bh_returns):.2%}",
        "Total Trades": int(df["signal"].sum()),
        "Win Rate": f"{(df.loc[df['position'] == 1, 'returns'] > 0).mean():.2%}",
    }

    return metrics


# --- Visualization ---
def plot_equity_curves(df: pd.DataFrame, symbol: str, output_path: Path) -> None:
    """Plot equity curves comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df.index, df["buy_hold_equity"], label="Buy & Hold", alpha=0.7)
    ax.plot(df.index, df["strategy_equity"], label="Strategy", alpha=0.9)

    ax.set_title(f"Equity Curves: {symbol}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($1 initial)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    console.print(f"[green]Saved chart:[/] {output_path}")


# --- Output ---
def display_results(metrics: dict, symbol: str) -> None:
    """Display results using rich table."""
    table = Table(title=f"Performance Metrics: {symbol}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for metric, value in metrics.items():
        table.add_row(metric, str(value))

    console.print(table)


def save_report(df: pd.DataFrame, metrics: dict, symbol: str, config: dict) -> Path:
    """Save markdown report to output/."""
    output_path = OUTPUT_DIR / f"quant_research_{symbol}_{TIMESTAMP}.md"

    report = f"""# Quant Research Report: {symbol}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Configuration

```yaml
{yaml.dump(config, default_flow_style=False)}
```

## Performance Metrics

| Metric | Value |
|--------|-------|
"""
    for metric, value in metrics.items():
        report += f"| {metric} | {value} |\n"

    report += f"""
## Data Summary

- Period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}
- Total bars: {len(df):,}
- Trading days with position: {int(df['position'].sum()):,}

## Equity Curve

![Equity Curves](quant_research_{symbol}_{TIMESTAMP}.png)
"""

    output_path.write_text(report)
    console.print(f"[green]Saved report:[/] {output_path}")
    return output_path


# --- Main ---
def load_config(config_path: Path | None) -> dict:
    """Load configuration from YAML file or use defaults."""
    defaults = {
        "symbol": "SPY",
        "rsi_period": 2,
        "rsi_threshold": 10,
        "ibs_threshold": 0.2,
    }

    if config_path and config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
            defaults.update(user_config)

    return defaults


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Quant Research Template")
    parser.add_argument("--config", type=Path, help="Path to config.yaml")
    parser.add_argument("--symbol", type=str, help="Ticker symbol")
    parser.add_argument("--rsi-period", type=int, help="RSI lookback period")
    parser.add_argument("--rsi-threshold", type=float, help="RSI entry threshold")
    parser.add_argument("--ibs-threshold", type=float, help="IBS entry threshold")
    args = parser.parse_args()

    # Load config (file -> CLI overrides)
    config = load_config(args.config)

    # Apply CLI overrides
    if args.symbol:
        config["symbol"] = args.symbol
    if args.rsi_period:
        config["rsi_period"] = args.rsi_period
    if args.rsi_threshold:
        config["rsi_threshold"] = args.rsi_threshold
    if args.ibs_threshold:
        config["ibs_threshold"] = args.ibs_threshold

    console.print(f"[bold]Running quant research for {config['symbol']}[/]")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Run pipeline
    df = load_data(config["symbol"])
    df = generate_signals(
        df,
        rsi_period=config["rsi_period"],
        rsi_threshold=config["rsi_threshold"],
        ibs_threshold=config["ibs_threshold"],
    )
    df = backtest(df)

    # Calculate and display metrics
    metrics = calculate_metrics(df)
    display_results(metrics, config["symbol"])

    # Save outputs
    chart_path = OUTPUT_DIR / f"quant_research_{config['symbol']}_{TIMESTAMP}.png"
    plot_equity_curves(df, config["symbol"], chart_path)
    save_report(df, metrics, config["symbol"], config)


if __name__ == "__main__":
    main()
