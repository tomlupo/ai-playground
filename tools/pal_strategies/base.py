# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "numpy>=1.24", "matplotlib>=3.8"]
# ///
"""
Base infrastructure for PAL strategy backtesting.

Provides common utilities for:
- Data loading from cached samples
- Train/test splitting with date-based separation
- Backtest engine with transaction costs
- Performance metric calculation
- Grid search optimization
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path for shared imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.indicators import cagr, max_drawdown, sharpe_ratio

# Constants
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Default train/test split dates
TRAIN_START = "2005-01-01"
TRAIN_END = "2020-12-31"
TEST_START = "2021-01-01"
TEST_END = "2025-12-31"

# Transaction cost in basis points per side
DEFAULT_COST_BPS = 10


@dataclass
class Backtest:
    """Container for backtest results."""

    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    signals: pd.Series
    trades: pd.DataFrame

    @property
    def cagr(self) -> float:
        """Compound Annual Growth Rate."""
        return cagr(self.returns)

    @property
    def sharpe(self) -> float:
        """Sharpe Ratio (risk-free = 2%)."""
        return sharpe_ratio(self.returns, rf=0.02)

    @property
    def max_dd(self) -> float:
        """Maximum Drawdown (negative)."""
        return max_drawdown(self.returns)

    @property
    def num_trades(self) -> int:
        """Number of round-trip trades."""
        return len(self.trades) if self.trades is not None else 0

    @property
    def win_rate(self) -> float:
        """Percentage of winning trades."""
        if self.trades is None or len(self.trades) == 0:
            return 0.0
        return (self.trades["pnl"] > 0).mean()

    @property
    def avg_trade(self) -> float:
        """Average trade return."""
        if self.trades is None or len(self.trades) == 0:
            return 0.0
        return self.trades["pnl"].mean()

    def summary(self) -> dict:
        """Return dictionary of key metrics."""
        return {
            "CAGR": self.cagr,
            "Sharpe": self.sharpe,
            "Max DD": self.max_dd,
            "Trades": self.num_trades,
            "Win Rate": self.win_rate,
            "Avg Trade": self.avg_trade,
        }

    def metrics_str(self) -> str:
        """Formatted string of key metrics."""
        return (
            f"CAGR: {self.cagr:.2%} | "
            f"Sharpe: {self.sharpe:.2f} | "
            f"MaxDD: {self.max_dd:.2%} | "
            f"Trades: {self.num_trades}"
        )


def load_data(symbol: str) -> pd.DataFrame:
    """
    Load OHLCV data from cached samples.

    Args:
        symbol: Ticker symbol (e.g., "SPY", "QQQ")

    Returns:
        DataFrame with OHLCV columns, Date index
    """
    path = SAMPLES_DIR / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No cached data for {symbol} at {path}")

    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()

    # Ensure column names are consistent
    df.columns = [c.title() for c in df.columns]

    return df


def load_multi_asset(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load data for multiple assets.

    Args:
        symbols: List of ticker symbols

    Returns:
        Dictionary mapping symbol -> DataFrame
    """
    return {s: load_data(s) for s in symbols}


def split_train_test(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing periods.

    Args:
        df: Full dataset
        train_end: Last date of training period
        test_start: First date of testing period

    Returns:
        (train_df, test_df)
    """
    train = df[df.index <= train_end].copy()
    test = df[df.index >= test_start].copy()
    return train, test


def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    cost_bps: float = DEFAULT_COST_BPS,
) -> Backtest:
    """
    Run vectorized backtest on a single asset.

    Args:
        prices: Close prices
        signals: Trading signals (1=long, 0=flat)
        cost_bps: Transaction cost in basis points per side

    Returns:
        Backtest results object
    """
    # Align data
    df = pd.DataFrame({"price": prices, "signal": signals}).dropna()

    # Position is signal from previous day (trade at close, hold next day)
    df["position"] = df["signal"].shift(1).fillna(0)

    # Calculate returns
    df["returns"] = df["price"].pct_change()

    # Strategy returns (gross)
    df["strat_returns"] = df["position"] * df["returns"]

    # Transaction costs - cost on position changes
    df["pos_change"] = df["position"].diff().abs()
    cost_per_trade = cost_bps / 10000  # Convert bps to decimal
    df["costs"] = df["pos_change"] * cost_per_trade

    # Net returns
    df["net_returns"] = df["strat_returns"] - df["costs"]

    # Equity curve
    df["equity"] = (1 + df["net_returns"]).cumprod()

    # Extract trades
    trades = _extract_trades(df)

    return Backtest(
        equity=df["equity"],
        returns=df["net_returns"],
        positions=df["position"],
        signals=df["signal"],
        trades=trades,
    )


def run_multi_asset_backtest(
    prices_dict: dict[str, pd.Series],
    signals_dict: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
) -> Backtest:
    """
    Run backtest on multiple assets with equal or custom weighting.

    Args:
        prices_dict: Dictionary of symbol -> close prices
        signals_dict: Dictionary of symbol -> signals
        weights: Optional custom weights (default: equal weight)
        cost_bps: Transaction cost in basis points

    Returns:
        Combined portfolio backtest results
    """
    symbols = list(prices_dict.keys())
    n_assets = len(symbols)

    if weights is None:
        weights = {s: 1.0 / n_assets for s in symbols}

    # Run individual backtests
    backtests = {}
    for symbol in symbols:
        bt = run_backtest(
            prices_dict[symbol],
            signals_dict[symbol],
            cost_bps=cost_bps,
        )
        backtests[symbol] = bt

    # Combine returns with weights
    all_returns = pd.DataFrame({s: bt.returns for s, bt in backtests.items()})
    weighted_returns = sum(
        all_returns[s] * weights[s] for s in symbols
    ).fillna(0)

    # Combined equity
    combined_equity = (1 + weighted_returns).cumprod()

    # Combine signals and positions
    all_signals = pd.DataFrame({s: bt.signals for s, bt in backtests.items()}).fillna(0)
    all_positions = pd.DataFrame({s: bt.positions for s, bt in backtests.items()}).fillna(0)

    # Combine trades
    all_trades = pd.concat(
        [bt.trades.assign(symbol=s) for s, bt in backtests.items() if bt.trades is not None],
        ignore_index=True,
    )

    return Backtest(
        equity=combined_equity,
        returns=weighted_returns,
        positions=all_positions.sum(axis=1) / n_assets,  # Average position
        signals=all_signals.sum(axis=1) / n_assets,
        trades=all_trades,
    )


def _extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Extract individual trades from backtest DataFrame."""
    trades = []
    in_trade = False
    entry_date = None
    entry_price = None

    for date, row in df.iterrows():
        if not in_trade and row["position"] == 1:
            # Entry
            in_trade = True
            entry_date = date
            entry_price = row["price"]
        elif in_trade and row["position"] == 0:
            # Exit
            exit_price = row["price"]
            pnl = (exit_price / entry_price) - 1
            trades.append({
                "entry_date": entry_date,
                "exit_date": date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "days": (date - entry_date).days,
            })
            in_trade = False
            entry_date = None
            entry_price = None

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def optimize_strategy(
    data: pd.DataFrame | dict[str, pd.DataFrame],
    signal_func: Callable,
    param_grid: dict[str, list],
    cost_bps: float = DEFAULT_COST_BPS,
    metric: str = "sharpe",
    n_best: int = 5,
) -> list[tuple[dict, Backtest]]:
    """
    Grid search for optimal strategy parameters.

    Args:
        data: Single DataFrame or dict of DataFrames for multi-asset
        signal_func: Function(data, **params) -> signals
        param_grid: Dictionary of parameter -> list of values to test
        cost_bps: Transaction cost
        metric: Optimization target ("sharpe", "cagr", "calmar")
        n_best: Number of top results to return

    Returns:
        List of (params, Backtest) tuples sorted by metric
    """
    results = []

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for values in product(*param_values):
        params = dict(zip(param_names, values))

        try:
            # Generate signals
            signals = signal_func(data, **params)

            # Run backtest
            if isinstance(data, dict):
                # Multi-asset
                prices = {s: df["Close"] for s, df in data.items()}
                bt = run_multi_asset_backtest(prices, signals, cost_bps=cost_bps)
            else:
                # Single asset
                bt = run_backtest(data["Close"], signals, cost_bps=cost_bps)

            # Score based on metric
            if metric == "sharpe":
                score = bt.sharpe
            elif metric == "cagr":
                score = bt.cagr
            elif metric == "calmar":
                score = bt.cagr / abs(bt.max_dd) if bt.max_dd != 0 else 0
            else:
                score = bt.sharpe

            results.append((params, bt, score))

        except Exception as e:
            # Skip invalid parameter combinations
            continue

    # Sort by score descending
    results.sort(key=lambda x: x[2], reverse=True)

    # Return top n without the score
    return [(params, bt) for params, bt, _ in results[:n_best]]


def plot_equity_comparison(
    backtests: dict[str, Backtest],
    title: str,
    save_path: Path | None = None,
) -> None:
    """
    Plot multiple equity curves for comparison.

    Args:
        backtests: Dictionary of name -> Backtest
        title: Chart title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, bt in backtests.items():
        ax.plot(bt.equity.index, bt.equity.values, label=name, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($1 initial)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.close()


def format_comparison_table(
    strategies: dict[str, dict],
    targets: dict[str, dict],
) -> str:
    """
    Format comparison of achieved vs target metrics as markdown table.

    Args:
        strategies: Dict of strategy name -> achieved metrics
        targets: Dict of strategy name -> target metrics

    Returns:
        Markdown formatted table
    """
    lines = [
        "| Strategy | Target CAGR | Achieved CAGR | Target MDD | Achieved MDD | Target Sharpe | Achieved Sharpe | Match |",
        "|----------|-------------|---------------|------------|--------------|---------------|-----------------|-------|",
    ]

    for name in strategies:
        achieved = strategies[name]
        target = targets.get(name, {})

        # Check if within tolerance
        cagr_ok = abs(achieved["CAGR"] - target.get("CAGR", 0)) <= 0.03
        mdd_ok = abs(achieved["Max DD"] - target.get("MDD", 0)) <= 0.10
        sharpe_ok = abs(achieved["Sharpe"] - target.get("Sharpe", 0)) <= 0.15

        match = "✓" if (cagr_ok and mdd_ok and sharpe_ok) else "✗"

        lines.append(
            f"| {name} | "
            f"{target.get('CAGR', 0):.1%} | {achieved['CAGR']:.1%} | "
            f"{target.get('MDD', 0):.1%} | {achieved['Max DD']:.1%} | "
            f"{target.get('Sharpe', 0):.2f} | {achieved['Sharpe']:.2f} | "
            f"{match} |"
        )

    return "\n".join(lines)
