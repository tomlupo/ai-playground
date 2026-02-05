# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "matplotlib>=3.8",
# ]
# ///
"""
B2S2ETF Strategy - Buy 2 Sell 2 ETF Mean-Reversion

Target Metrics (PAL):
- CAGR: 9.1%
- Max DD: -30.6%
- Sharpe: 0.63

Hypothesis: "Buy 2 Sell 2" likely means:
- Buy after 2 consecutive down days
- Sell after 2 consecutive up days (or fixed period)

Single asset: SPY only

Usage:
    uv run tools/pal-strategies/b2s2etf.py
    uv run tools/pal-strategies/b2s2etf.py --optimize
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.indicators import ibs, rsi
from base import (
    OUTPUT_DIR,
    TARGETS,
    backtest_signals,
    calculate_metrics,
    create_result,
    load_data,
    plot_equity_curve,
    split_data,
)

STRATEGY_NAME = "B2S2ETF"
SYMBOL = "SPY"
TARGET = TARGETS[STRATEGY_NAME]


def count_consecutive(returns: pd.Series, direction: str = "down") -> pd.Series:
    """Count consecutive up or down days (vectorized)."""
    if direction == "down":
        is_dir = (returns < 0).astype(int)
    else:
        is_dir = (returns > 0).astype(int)

    # Group consecutive runs
    groups = (is_dir != is_dir.shift()).cumsum()
    counts = is_dir.groupby(groups).cumsum()

    return counts


def generate_signals_v1(
    df: pd.DataFrame,
    down_days: int = 2,
    up_days: int = 2,
) -> pd.Series:
    """
    Version 1: Pure "Buy N down, Sell N up" pattern.

    Entry: After N consecutive down days
    Exit: After M consecutive up days
    """
    returns = df["Close"].pct_change()

    # Count consecutive down/up days
    down_count = count_consecutive(returns, "down")
    up_count = count_consecutive(returns, "up")

    # Entry: N consecutive down days completed
    entry = down_count >= down_days

    # Exit: M consecutive up days completed
    exit_sig = up_count >= up_days

    # Build position with state tracking
    signal_events = pd.Series(0, index=df.index)
    signal_events[entry] = 1
    signal_events[exit_sig] = -1

    state = signal_events.replace(0, np.nan).ffill().fillna(0)
    position = np.where(state == 1, 1.0, 0.0)

    return pd.Series(position, index=df.index)


def generate_signals_v2(
    df: pd.DataFrame,
    down_days: int = 2,
    hold_days: int = 2,
) -> pd.Series:
    """
    Version 2: "Buy N down, Hold M days" pattern.

    Entry: After N consecutive down days
    Exit: After M days (fixed holding period)
    """
    returns = df["Close"].pct_change()
    down_count = count_consecutive(returns, "down")

    # Entry signal
    entry = (down_count >= down_days).astype(int)

    # Hold for M days after entry
    position = entry.rolling(hold_days, min_periods=1).max()

    return pd.Series(position, index=df.index)


def generate_signals_v3(
    df: pd.DataFrame,
    down_days: int = 2,
    rsi_period: int = 2,
    rsi_entry: float = 30,
    hold_days: int = 2,
) -> pd.Series:
    """
    Version 3: Consecutive down + RSI filter.

    Entry: N consecutive down days AND RSI < threshold
    Exit: After M days
    """
    returns = df["Close"].pct_change()
    down_count = count_consecutive(returns, "down")
    rsi_val = rsi(df["Close"], period=rsi_period)

    # Entry: down days + RSI filter
    entry = (down_count >= down_days) & (rsi_val < rsi_entry)
    entry = entry.astype(int)

    # Hold for M days
    position = entry.rolling(hold_days, min_periods=1).max()

    return pd.Series(position, index=df.index)


def generate_signals_v4(
    df: pd.DataFrame,
    down_days: int = 2,
    ibs_entry: float = 0.3,
    hold_days: int = 2,
) -> pd.Series:
    """
    Version 4: Consecutive down + IBS filter.

    Entry: N consecutive down days AND IBS < threshold
    Exit: After M days
    """
    returns = df["Close"].pct_change()
    down_count = count_consecutive(returns, "down")
    ibs_val = ibs(df["High"], df["Low"], df["Close"])

    # Entry: down days + IBS filter
    entry = (down_count >= down_days) & (ibs_val < ibs_entry)
    entry = entry.astype(int)

    # Hold for M days
    position = entry.rolling(hold_days, min_periods=1).max()

    return pd.Series(position, index=df.index)


def generate_signals_v5(
    df: pd.DataFrame,
    down_days: int = 2,
    rsi_period: int = 2,
    rsi_entry: float = 30,
    up_days: int = 2,
) -> pd.Series:
    """
    Version 5: Consecutive down + RSI entry, consecutive up exit.

    Entry: N consecutive down days AND RSI < threshold
    Exit: M consecutive up days
    """
    returns = df["Close"].pct_change()
    down_count = count_consecutive(returns, "down")
    up_count = count_consecutive(returns, "up")
    rsi_val = rsi(df["Close"], period=rsi_period)

    # Entry: down days + RSI
    entry = (down_count >= down_days) & (rsi_val < rsi_entry)

    # Exit: up days
    exit_sig = up_count >= up_days

    # Build position
    signal_events = pd.Series(0, index=df.index)
    signal_events[entry] = 1
    signal_events[exit_sig] = -1

    state = signal_events.replace(0, np.nan).ffill().fillna(0)
    position = np.where(state == 1, 1.0, 0.0)

    return pd.Series(position, index=df.index)


def generate_signals_v6(
    df: pd.DataFrame,
    down_days: int = 2,
    pct_decline: float = -0.02,
    hold_days: int = 3,
) -> pd.Series:
    """
    Version 6: Consecutive down + magnitude filter.

    Entry: N consecutive down days AND total decline > threshold
    Exit: After M days
    """
    returns = df["Close"].pct_change()
    down_count = count_consecutive(returns, "down")

    # Total decline over the down period
    total_decline = returns.rolling(down_days).sum()

    # Entry: consecutive down + significant decline
    entry = (down_count >= down_days) & (total_decline < pct_decline)
    entry = entry.astype(int)

    # Hold for M days
    position = entry.rolling(hold_days, min_periods=1).max()

    return pd.Series(position, index=df.index)


def backtest_strategy(
    df: pd.DataFrame,
    signals: pd.Series,
    cost_bps: float = 10,
) -> tuple[pd.Series, pd.Series]:
    """Run backtest for single asset."""
    return backtest_signals(df["Close"], signals, cost_bps)


def optimize(df: pd.DataFrame, version: str = "v1") -> list[tuple]:
    """Run parameter optimization."""
    print(f"\nOptimizing {STRATEGY_NAME} - Version {version}")

    if version == "v1":
        param_grid = {
            "down_days": [2, 3, 4, 5],
            "up_days": [1, 2, 3, 4, 5],
        }
        signal_func = generate_signals_v1
    elif version == "v2":
        param_grid = {
            "down_days": [2, 3, 4, 5],
            "hold_days": [1, 2, 3, 5, 7, 10],
        }
        signal_func = generate_signals_v2
    elif version == "v3":
        param_grid = {
            "down_days": [2, 3, 4],
            "rsi_period": [2, 3, 5],
            "rsi_entry": [20, 30, 40, 50],
            "hold_days": [1, 2, 3, 5],
        }
        signal_func = generate_signals_v3
    elif version == "v4":
        param_grid = {
            "down_days": [2, 3, 4],
            "ibs_entry": [0.2, 0.3, 0.4, 0.5],
            "hold_days": [1, 2, 3, 5],
        }
        signal_func = generate_signals_v4
    elif version == "v5":
        param_grid = {
            "down_days": [2, 3, 4],
            "rsi_period": [2, 3],
            "rsi_entry": [20, 30, 40, 50],
            "up_days": [1, 2, 3],
        }
        signal_func = generate_signals_v5
    elif version == "v6":
        param_grid = {
            "down_days": [2, 3, 4],
            "pct_decline": [-0.01, -0.02, -0.03, -0.04],
            "hold_days": [1, 2, 3, 5],
        }
        signal_func = generate_signals_v6
    else:
        raise ValueError(f"Unknown version: {version}")

    # Grid search
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    results = []
    total = len(combinations)
    print(f"Testing {total} parameter combinations...")

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        try:
            signals = signal_func(df, **params)
            returns, equity = backtest_strategy(df, signals)
            metrics = calculate_metrics(returns)

            results.append({
                "params": params,
                "cagr": metrics["cagr"],
                "max_dd": metrics["max_dd"],
                "sharpe": metrics["sharpe"],
            })
        except Exception:
            continue

        if (i + 1) % max(1, total // 5) == 0:
            print(f"  Progress: {i + 1}/{total}")

    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\nTop 10 results:")
    print("-" * 80)
    for r in results[:10]:
        target_match = TARGET.is_match(
            type("R", (), {"cagr": r["cagr"], "max_dd": r["max_dd"], "sharpe": r["sharpe"]})()
        )
        match_str = "✓ MATCH" if target_match else ""
        print(
            f"  CAGR: {r['cagr']:>6.1%} | DD: {r['max_dd']:>7.1%} | "
            f"Sharpe: {r['sharpe']:.2f} | {r['params']} {match_str}"
        )

    return results


def run_strategy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
    params: dict = None,
    version: str = "v1",
) -> tuple:
    """Run strategy with given parameters."""
    signal_funcs = {
        "v1": generate_signals_v1,
        "v2": generate_signals_v2,
        "v3": generate_signals_v3,
        "v4": generate_signals_v4,
        "v5": generate_signals_v5,
        "v6": generate_signals_v6,
    }

    if params is None:
        # Default parameters
        if version == "v1":
            params = {"down_days": 2, "up_days": 2}
        elif version == "v2":
            params = {"down_days": 2, "hold_days": 2}
        elif version == "v3":
            params = {"down_days": 2, "rsi_period": 2, "rsi_entry": 30, "hold_days": 2}
        elif version == "v4":
            params = {"down_days": 2, "ibs_entry": 0.3, "hold_days": 2}
        elif version == "v5":
            params = {"down_days": 2, "rsi_period": 2, "rsi_entry": 30, "up_days": 2}
        elif version == "v6":
            params = {"down_days": 2, "pct_decline": -0.02, "hold_days": 3}

    signal_func = signal_funcs[version]

    # Train period
    train_signals = signal_func(train_df, **params)
    train_returns, train_equity = backtest_strategy(train_df, train_signals)
    train_metrics = calculate_metrics(train_returns)

    # Test period
    test_metrics = None
    if test_df is not None:
        test_signals = signal_func(test_df, **params)
        test_returns, test_equity = backtest_strategy(test_df, test_signals)
        test_metrics = calculate_metrics(test_returns)

    return train_returns, train_metrics, test_metrics, params


def main():
    import argparse

    parser = argparse.ArgumentParser(description=f"{STRATEGY_NAME} Strategy")
    parser.add_argument("--optimize", action="store_true", help="Run optimization")
    parser.add_argument("--version", default="all", help="Signal version (v1-v6 or 'all')")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  {STRATEGY_NAME} - Buy 2 Sell 2 ETF Strategy")
    print(f"  Target: CAGR {TARGET.cagr:.1%}, DD {TARGET.max_dd:.1%}, Sharpe {TARGET.sharpe:.2f}")
    print(f"{'='*60}")

    # Load data
    print(f"\nLoading {SYMBOL}...")
    df = load_data(SYMBOL)
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Split into train/test
    train_df, test_df = split_data(df, train_end="2020-12-31")
    print(f"  Train: {train_df.index.min()} to {train_df.index.max()}")
    print(f"  Test:  {test_df.index.min()} to {test_df.index.max()}")

    # Determine versions to run
    versions = ["v1", "v2", "v3", "v4", "v5", "v6"] if args.version == "all" else [args.version]

    best_result = None
    best_sharpe = -np.inf

    for version in versions:
        print(f"\n{'='*60}")
        print(f"  Testing Version {version}")
        print(f"{'='*60}")

        if args.optimize:
            results = optimize(train_df, version=version)
            if results:
                best_params = results[0]["params"]
            else:
                continue
        else:
            best_params = None

        # Run with best params
        train_returns, train_metrics, test_metrics, params = run_strategy(
            train_df, test_df, params=best_params, version=version
        )

        print(f"\n{version} Results:")
        print(f"  Parameters: {params}")
        print(f"\n  TRAIN Period:")
        print(f"    CAGR:   {train_metrics['cagr']:>7.1%}  (target: {TARGET.cagr:.1%})")
        print(f"    Max DD: {train_metrics['max_dd']:>7.1%}  (target: {TARGET.max_dd:.1%})")
        print(f"    Sharpe: {train_metrics['sharpe']:>7.2f}  (target: {TARGET.sharpe:.2f})")

        if test_metrics:
            print(f"\n  TEST Period:")
            print(f"    CAGR:   {test_metrics['cagr']:>7.1%}")
            print(f"    Max DD: {test_metrics['max_dd']:>7.1%}")
            print(f"    Sharpe: {test_metrics['sharpe']:>7.2f}")

        # Track best overall
        if train_metrics["sharpe"] > best_sharpe:
            best_sharpe = train_metrics["sharpe"]
            best_result = {
                "version": version,
                "params": params,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "returns": train_returns,
            }

    # Final report for best version
    if best_result:
        print(f"\n{'='*60}")
        print(f"  BEST RESULT: Version {best_result['version']}")
        print(f"{'='*60}")
        print(f"  Parameters: {best_result['params']}")
        print(f"  Train Sharpe: {best_result['train_metrics']['sharpe']:.2f}")

        # Check target match
        result_obj = type("R", (), {
            "cagr": best_result["train_metrics"]["cagr"],
            "max_dd": best_result["train_metrics"]["max_dd"],
            "sharpe": best_result["train_metrics"]["sharpe"],
        })()

        if TARGET.is_match(result_obj):
            print(f"\n  ✓ MATCHES TARGET METRICS!")
        else:
            print(f"\n  ✗ Does not match target (within tolerance)")

        # Save equity curve plot
        output_dir = OUTPUT_DIR / "claude-reverse-engineer-strategies-ziZsY" / "pal-reverse-engineering"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create full period result for plotting
        signal_funcs = {
            "v1": generate_signals_v1,
            "v2": generate_signals_v2,
            "v3": generate_signals_v3,
            "v4": generate_signals_v4,
            "v5": generate_signals_v5,
            "v6": generate_signals_v6,
        }
        full_signals = signal_funcs[best_result["version"]](df, **best_result["params"])
        full_returns, full_equity = backtest_strategy(df, full_signals)

        result = create_result(
            STRATEGY_NAME,
            best_result["params"],
            full_returns,
            full_signals,
        )

        plot_equity_curve(
            result,
            benchmark=df["Close"],
            target=TARGET,
            output_path=output_dir / f"{STRATEGY_NAME.lower()}_equity.png",
        )

        return best_result

    return None


if __name__ == "__main__":
    main()
