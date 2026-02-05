# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "matplotlib>=3.8",
# ]
# ///
"""
Base infrastructure for PAL strategy backtesting.

All strategies share this common backtest engine.
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.indicators import cagr, max_drawdown, sharpe_ratio

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass
class StrategyResult:
    """Container for strategy backtest results."""

    name: str
    params: dict
    cagr: float
    max_dd: float
    sharpe: float
    total_trades: int
    win_rate: float
    avg_trade_return: float
    exposure: float  # % of time in market
    equity_curve: pd.Series
    returns: pd.Series
    train_metrics: dict | None = None
    test_metrics: dict | None = None


@dataclass
class TargetMetrics:
    """PAL published target metrics."""

    name: str
    cagr: float
    max_dd: float
    sharpe: float

    def is_match(self, result: StrategyResult, tolerance: dict = None) -> bool:
        """Check if result matches target within tolerance."""
        if tolerance is None:
            tolerance = {"cagr": 0.03, "max_dd": 0.10, "sharpe": 0.15}

        cagr_ok = abs(result.cagr - self.cagr) <= tolerance["cagr"]
        dd_ok = abs(result.max_dd - self.max_dd) <= tolerance["max_dd"]
        sharpe_ok = abs(result.sharpe - self.sharpe) <= tolerance["sharpe"]

        return cagr_ok and dd_ok and sharpe_ok


# PAL Target Metrics
TARGETS = {
    "ETFMR": TargetMetrics("ETFMR", cagr=0.10, max_dd=-0.229, sharpe=0.82),
    "MRETF": TargetMetrics("MRETF", cagr=0.052, max_dd=-0.093, sharpe=0.78),
    "B2S2ETF": TargetMetrics("B2S2ETF", cagr=0.091, max_dd=-0.306, sharpe=0.63),
    "MRMOM": TargetMetrics("MRMOM", cagr=0.103, max_dd=-0.168, sharpe=1.15),
    "ETFSEAS": TargetMetrics("ETFSEAS", cagr=0.073, max_dd=-0.134, sharpe=0.83),
}


def load_data(symbol: str) -> pd.DataFrame:
    """Load cached data from data/samples/."""
    cached_path = SAMPLES_DIR / f"{symbol}.csv"
    if not cached_path.exists():
        raise FileNotFoundError(f"No cached data for {symbol} at {cached_path}")

    df = pd.read_csv(cached_path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()

    # Ensure required columns
    required = ["Close", "High", "Low", "Open"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def load_multi_asset(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Load multiple assets and align dates."""
    dfs = {sym: load_data(sym) for sym in symbols}

    # Find common date range
    common_start = max(df.index.min() for df in dfs.values())
    common_end = min(df.index.max() for df in dfs.values())

    # Align to common dates
    for sym in dfs:
        dfs[sym] = dfs[sym].loc[common_start:common_end]

    return dfs


def backtest_signals(
    prices: pd.Series,
    signals: pd.Series,
    cost_bps: float = 10,
) -> tuple[pd.Series, pd.Series]:
    """
    Run vectorized backtest given signals.

    Args:
        prices: Close prices
        signals: Position signals (1=long, 0=flat)
        cost_bps: Transaction cost in basis points per side

    Returns:
        (strategy_returns, equity_curve)
    """
    returns = prices.pct_change()

    # Position is signal shifted by 1 (execute next day)
    position = signals.shift(1).fillna(0)

    # Calculate trades (position changes)
    trades = position.diff().abs()

    # Cost per trade (both sides)
    costs = trades * (cost_bps / 10000)

    # Strategy returns
    strategy_returns = position * returns - costs

    # Equity curve
    equity = (1 + strategy_returns).cumprod()

    return strategy_returns, equity


def calculate_metrics(returns: pd.Series, equity: pd.Series = None) -> dict:
    """Calculate standard performance metrics."""
    returns = returns.dropna()

    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252

    metrics = {
        "cagr": cagr(returns),
        "max_dd": max_drawdown(returns),
        "sharpe": sharpe_ratio(returns),
        "total_return": total_return,
        "years": years,
        "volatility": returns.std() * np.sqrt(252),
    }

    return metrics


def split_data(
    df: pd.DataFrame,
    train_end: str = "2020-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets."""
    train = df.loc[:train_end]
    test = df.loc[train_end:].iloc[1:]  # Exclude train_end from test
    return train, test


def optimize_grid(
    data: pd.DataFrame | dict[str, pd.DataFrame],
    signal_func: Callable,
    param_grid: dict,
    cost_bps: float = 10,
    metric: str = "sharpe",
    top_n: int = 10,
) -> list[tuple[dict, dict]]:
    """
    Grid search optimization.

    Args:
        data: Single DataFrame or dict of DataFrames for multi-asset
        signal_func: Function(data, **params) -> pd.Series of signals
        param_grid: Dict of param_name -> list of values
        cost_bps: Transaction cost
        metric: Metric to optimize ('sharpe', 'cagr', 'max_dd')
        top_n: Return top N results

    Returns:
        List of (params, metrics) tuples sorted by metric
    """
    import itertools

    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    results = []
    total = len(combinations)

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        try:
            # Generate signals
            signals = signal_func(data, **params)

            # Get prices for backtest
            if isinstance(data, dict):
                # Multi-asset: combine returns
                all_returns = []
                for sym, df in data.items():
                    sym_signals = signals.get(sym, pd.Series(0, index=df.index))
                    sym_returns, _ = backtest_signals(df["Close"], sym_signals, cost_bps)
                    all_returns.append(sym_returns)

                # Equal weight combine
                combined_returns = pd.concat(all_returns, axis=1).mean(axis=1)
                equity = (1 + combined_returns).cumprod()
                strategy_returns = combined_returns
            else:
                strategy_returns, equity = backtest_signals(
                    data["Close"], signals, cost_bps
                )

            metrics = calculate_metrics(strategy_returns, equity)
            results.append((params, metrics))

        except Exception as e:
            # Skip invalid parameter combinations
            continue

        # Progress indicator every 10%
        if (i + 1) % max(1, total // 10) == 0:
            print(f"  Progress: {i + 1}/{total} ({100 * (i + 1) / total:.0f}%)")

    # Sort by metric (descending for sharpe/cagr, ascending for max_dd)
    if metric == "max_dd":
        results.sort(key=lambda x: x[1][metric], reverse=True)  # Less negative is better
    else:
        results.sort(key=lambda x: x[1][metric], reverse=True)

    return results[:top_n]


def create_result(
    name: str,
    params: dict,
    returns: pd.Series,
    signals: pd.Series,
) -> StrategyResult:
    """Create StrategyResult from backtest."""
    metrics = calculate_metrics(returns)
    equity = (1 + returns).cumprod()

    # Calculate trade statistics
    trades = signals.diff().abs()
    total_trades = int(trades.sum() / 2)  # Entry + exit = 1 trade

    # Position returns
    position = signals.shift(1).fillna(0)
    position_returns = returns[position > 0]
    win_rate = (position_returns > 0).mean() if len(position_returns) > 0 else 0
    avg_trade = position_returns.mean() if len(position_returns) > 0 else 0

    # Exposure
    exposure = position.mean()

    return StrategyResult(
        name=name,
        params=params,
        cagr=metrics["cagr"],
        max_dd=metrics["max_dd"],
        sharpe=metrics["sharpe"],
        total_trades=total_trades,
        win_rate=win_rate,
        avg_trade_return=avg_trade,
        exposure=exposure,
        equity_curve=equity,
        returns=returns,
    )


def plot_equity_curve(
    result: StrategyResult,
    benchmark: pd.Series = None,
    target: TargetMetrics = None,
    output_path: Path = None,
) -> None:
    """Plot equity curve with optional benchmark comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    # Equity curve
    ax1 = axes[0]
    ax1.plot(result.equity_curve.index, result.equity_curve, label=result.name, linewidth=1.5)

    if benchmark is not None:
        bh_equity = (1 + benchmark.pct_change()).cumprod()
        ax1.plot(bh_equity.index, bh_equity, label="Buy & Hold", alpha=0.7, linewidth=1)

    ax1.set_title(f"{result.name} Equity Curve")
    ax1.set_ylabel("Equity ($1 initial)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Add metrics annotation
    metrics_text = (
        f"CAGR: {result.cagr:.1%}\n"
        f"Max DD: {result.max_dd:.1%}\n"
        f"Sharpe: {result.sharpe:.2f}\n"
        f"Trades: {result.total_trades}"
    )
    if target:
        metrics_text += f"\n\nTarget:\nCAGR: {target.cagr:.1%}\nDD: {target.max_dd:.1%}\nSharpe: {target.sharpe:.2f}"

    ax1.text(
        0.02, 0.98, metrics_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Drawdown
    ax2 = axes[1]
    equity = result.equity_curve
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.5, color="red")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")

    plt.close()


def format_comparison_table(results: list[StrategyResult], targets: dict) -> str:
    """Format results comparison as markdown table."""
    lines = [
        "| Strategy | CAGR | Max DD | Sharpe | Trades | Win Rate | Match |",
        "|----------|------|--------|--------|--------|----------|-------|",
    ]

    for r in results:
        target = targets.get(r.name)
        match = "✓" if target and target.is_match(r) else "✗"

        lines.append(
            f"| {r.name} | {r.cagr:.1%} | {r.max_dd:.1%} | {r.sharpe:.2f} | "
            f"{r.total_trades} | {r.win_rate:.1%} | {match} |"
        )

    # Add targets row
    lines.append("|----------|------|--------|--------|--------|----------|-------|")
    lines.append("| **Targets** |")
    for name, t in targets.items():
        lines.append(f"| {name} (target) | {t.cagr:.1%} | {t.max_dd:.1%} | {t.sharpe:.2f} | - | - | - |")

    return "\n".join(lines)
