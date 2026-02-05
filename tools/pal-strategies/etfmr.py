# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "matplotlib>=3.8",
# ]
# ///
"""
ETFMR Strategy - ETF Mean-Reversion

Target Metrics (PAL):
- CAGR: 10.0%
- Max DD: -22.9%
- Sharpe: 0.82

Hypothesis: RSI-based mean-reversion on SPY and QQQ.
Buy oversold conditions, exit on recovery.

Usage:
    uv run tools/pal-strategies/etfmr.py
    uv run tools/pal-strategies/etfmr.py --optimize
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.indicators import cumulative_rsi, ibs, rsi
from base import (
    OUTPUT_DIR,
    TARGETS,
    backtest_signals,
    calculate_metrics,
    create_result,
    load_multi_asset,
    plot_equity_curve,
    split_data,
)

STRATEGY_NAME = "ETFMR"
SYMBOLS = ["SPY", "QQQ"]
TARGET = TARGETS[STRATEGY_NAME]


def generate_signals_v1(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    rsi_entry: float = 10,
    rsi_exit: float = 50,
    use_ibs: bool = False,
    ibs_entry: float = 0.2,
) -> dict[str, pd.Series]:
    """
    Version 1: Simple RSI-2 mean-reversion.

    Entry: RSI < threshold
    Exit: RSI > exit threshold
    """
    signals = {}

    for sym, df in data.items():
        close = df["Close"]
        rsi_val = rsi(close, period=rsi_period)

        # Entry: RSI below entry threshold
        entry = rsi_val < rsi_entry

        # Optional IBS filter
        if use_ibs:
            ibs_val = ibs(df["High"], df["Low"], close)
            entry = entry & (ibs_val < ibs_entry)

        # Exit: RSI above exit threshold
        exit_sig = rsi_val > rsi_exit

        # Build position: hold until exit
        position = pd.Series(0.0, index=df.index)
        in_position = False

        # Vectorized position tracking using ffill
        # 1 = entry signal, -1 = exit signal, 0 = hold
        signal_events = pd.Series(0, index=df.index)
        signal_events[entry] = 1
        signal_events[exit_sig] = -1

        # Forward fill to track state
        # When entry fires, we're in position until exit
        state = signal_events.replace(0, np.nan).ffill().fillna(0)
        position = np.where(state == 1, 1.0, 0.0)
        position = pd.Series(position, index=df.index)

        signals[sym] = position

    return signals


def generate_signals_v2(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    cum_period: int = 2,
    cum_entry: float = 35,
    cum_exit: float = 65,
) -> dict[str, pd.Series]:
    """
    Version 2: Cumulative RSI mean-reversion.

    Entry: Cumulative RSI below threshold (deeper oversold)
    Exit: Cumulative RSI above exit threshold
    """
    signals = {}

    for sym, df in data.items():
        close = df["Close"]
        cum_rsi = cumulative_rsi(close, rsi_period=rsi_period, cum_period=cum_period)

        # Entry and exit
        entry = cum_rsi < cum_entry
        exit_sig = cum_rsi > cum_exit

        # Vectorized position tracking
        signal_events = pd.Series(0, index=df.index)
        signal_events[entry] = 1
        signal_events[exit_sig] = -1

        state = signal_events.replace(0, np.nan).ffill().fillna(0)
        position = np.where(state == 1, 1.0, 0.0)
        position = pd.Series(position, index=df.index)

        signals[sym] = position

    return signals


def generate_signals_v3(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    rsi_entry: float = 5,
    hold_days: int = 5,
) -> dict[str, pd.Series]:
    """
    Version 3: RSI with fixed holding period.

    Entry: RSI < threshold
    Exit: After N days
    """
    signals = {}

    for sym, df in data.items():
        close = df["Close"]
        rsi_val = rsi(close, period=rsi_period)

        # Entry: very low RSI
        entry = (rsi_val < rsi_entry).astype(int)

        # Hold for N days after entry
        # Use rolling sum to count days since entry
        position = entry.rolling(hold_days, min_periods=1).max()
        position = pd.Series(position, index=df.index)

        signals[sym] = position

    return signals


def generate_signals_v4(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    rsi_entry: float = 10,
    ibs_entry: float = 0.25,
    rsi_exit: float = 65,
) -> dict[str, pd.Series]:
    """
    Version 4: Combined RSI + IBS entry.

    Entry: RSI < threshold AND IBS < threshold
    Exit: RSI > exit threshold
    """
    signals = {}

    for sym, df in data.items():
        close = df["Close"]
        rsi_val = rsi(close, period=rsi_period)
        ibs_val = ibs(df["High"], df["Low"], close)

        # Entry: both indicators oversold
        entry = (rsi_val < rsi_entry) & (ibs_val < ibs_entry)

        # Exit: RSI recovery
        exit_sig = rsi_val > rsi_exit

        # Vectorized position
        signal_events = pd.Series(0, index=df.index)
        signal_events[entry] = 1
        signal_events[exit_sig] = -1

        state = signal_events.replace(0, np.nan).ffill().fillna(0)
        position = np.where(state == 1, 1.0, 0.0)
        position = pd.Series(position, index=df.index)

        signals[sym] = position

    return signals


def generate_signals_v5(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    rsi_entry: float = 25,
    ibs_entry: float = 0.3,
    hold_days: int = 1,
) -> dict[str, pd.Series]:
    """
    Version 5: Both assets must be oversold together.

    Entry: RSI of BOTH SPY and QQQ < threshold (or one asset with IBS filter)
    Exit: After N days (fixed holding period)

    This mimics PAL's likely "correlated entry" approach.
    """
    # Get common index
    common_idx = data["SPY"].index.intersection(data["QQQ"].index)

    # Calculate RSI for both
    rsi_spy = rsi(data["SPY"]["Close"], period=rsi_period).reindex(common_idx)
    rsi_qqq = rsi(data["QQQ"]["Close"], period=rsi_period).reindex(common_idx)

    # IBS for both
    ibs_spy = ibs(data["SPY"]["High"], data["SPY"]["Low"], data["SPY"]["Close"]).reindex(common_idx)
    ibs_qqq = ibs(data["QQQ"]["High"], data["QQQ"]["Low"], data["QQQ"]["Close"]).reindex(common_idx)

    # Entry: BOTH must be oversold
    both_rsi_oversold = (rsi_spy < rsi_entry) & (rsi_qqq < rsi_entry)

    # Or: One deeply oversold with IBS confirmation
    spy_deep = (rsi_spy < rsi_entry * 0.5) & (ibs_spy < ibs_entry)
    qqq_deep = (rsi_qqq < rsi_entry * 0.5) & (ibs_qqq < ibs_entry)

    entry = both_rsi_oversold | spy_deep | qqq_deep

    # Hold for N days
    position = entry.astype(int).rolling(hold_days, min_periods=1).max()

    signals = {
        "SPY": pd.Series(position, index=common_idx).reindex(data["SPY"].index).fillna(0),
        "QQQ": pd.Series(position, index=common_idx).reindex(data["QQQ"].index).fillna(0),
    }

    return signals


def generate_signals_v6(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    rsi_entry: float = 10,
    consecutive_down: int = 2,
    hold_days: int = 3,
) -> dict[str, pd.Series]:
    """
    Version 6: Consecutive down days with RSI filter.

    Entry: N consecutive down days AND RSI < threshold
    Exit: After M days
    """
    signals = {}

    for sym, df in data.items():
        close = df["Close"]
        returns = close.pct_change()
        rsi_val = rsi(close, period=rsi_period)

        # Count consecutive down days (vectorized)
        down = (returns < 0).astype(int)

        # Rolling sum of down days
        down_count = down.rolling(consecutive_down, min_periods=consecutive_down).sum()

        # Entry: all N days were down AND RSI oversold
        entry = (down_count == consecutive_down) & (rsi_val < rsi_entry)

        # Hold for M days
        position = entry.astype(int).rolling(hold_days, min_periods=1).max()
        position = pd.Series(position, index=df.index)

        signals[sym] = position

    return signals


def generate_signals_v7(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    rsi_entry: float = 20,
    bb_period: int = 20,
    bb_std: float = 2.0,
    hold_days: int = 2,
) -> dict[str, pd.Series]:
    """
    Version 7: RSI + Bollinger Band mean-reversion.

    Entry: RSI < threshold AND price below lower BB
    Exit: After N days or price crosses middle BB
    """
    from shared.indicators import bollinger_bands

    signals = {}

    for sym, df in data.items():
        close = df["Close"]
        rsi_val = rsi(close, period=rsi_period)
        mid, upper, lower = bollinger_bands(close, period=bb_period, num_std=bb_std)

        # Entry: RSI oversold AND below lower band
        entry = (rsi_val < rsi_entry) & (close < lower)

        # Hold for N days
        position = entry.astype(int).rolling(hold_days, min_periods=1).max()
        position = pd.Series(position, index=df.index)

        signals[sym] = position

    return signals


def backtest_multi_asset(
    data: dict[str, pd.DataFrame],
    signals: dict[str, pd.Series],
    cost_bps: float = 10,
) -> tuple[pd.Series, pd.Series]:
    """Backtest multi-asset strategy with equal weighting."""
    all_returns = []

    for sym, df in data.items():
        sym_signals = signals.get(sym, pd.Series(0, index=df.index))
        returns, _ = backtest_signals(df["Close"], sym_signals, cost_bps)
        all_returns.append(returns)

    # Equal weight combination
    combined = pd.concat(all_returns, axis=1)
    avg_returns = combined.mean(axis=1)
    equity = (1 + avg_returns).cumprod()

    return avg_returns, equity


def optimize(data: dict[str, pd.DataFrame], version: str = "v1") -> list[tuple]:
    """Run parameter optimization."""
    print(f"\nOptimizing {STRATEGY_NAME} - Version {version}")

    if version == "v1":
        param_grid = {
            "rsi_period": [2, 3, 5],
            "rsi_entry": [5, 10, 15, 20, 25],
            "rsi_exit": [50, 60, 70, 80],
            "use_ibs": [False, True],
            "ibs_entry": [0.2, 0.25, 0.3],
        }
        signal_func = generate_signals_v1
    elif version == "v2":
        param_grid = {
            "rsi_period": [2, 3],
            "cum_period": [2, 3, 4],
            "cum_entry": [25, 30, 35, 40, 45],
            "cum_exit": [55, 65, 75, 85],
        }
        signal_func = generate_signals_v2
    elif version == "v3":
        param_grid = {
            "rsi_period": [2, 3, 5],
            "rsi_entry": [3, 5, 7, 10, 15],
            "hold_days": [2, 3, 5, 7, 10],
        }
        signal_func = generate_signals_v3
    elif version == "v4":
        param_grid = {
            "rsi_period": [2, 3],
            "rsi_entry": [5, 10, 15, 20],
            "ibs_entry": [0.15, 0.2, 0.25, 0.3],
            "rsi_exit": [55, 65, 75],
        }
        signal_func = generate_signals_v4
    elif version == "v5":
        param_grid = {
            "rsi_period": [2, 3],
            "rsi_entry": [15, 20, 25, 30, 35],
            "ibs_entry": [0.2, 0.3, 0.4],
            "hold_days": [1, 2, 3, 5],
        }
        signal_func = generate_signals_v5
    elif version == "v6":
        param_grid = {
            "rsi_period": [2, 3],
            "rsi_entry": [10, 15, 20, 25, 30],
            "consecutive_down": [2, 3, 4],
            "hold_days": [1, 2, 3, 5],
        }
        signal_func = generate_signals_v6
    elif version == "v7":
        param_grid = {
            "rsi_period": [2, 3],
            "rsi_entry": [15, 20, 25, 30],
            "bb_period": [10, 20],
            "bb_std": [1.5, 2.0, 2.5],
            "hold_days": [1, 2, 3, 5],
        }
        signal_func = generate_signals_v7
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
            signals = signal_func(data, **params)
            returns, equity = backtest_multi_asset(data, signals)
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
    train_data: dict[str, pd.DataFrame],
    test_data: dict[str, pd.DataFrame] = None,
    params: dict = None,
    version: str = "v1",
) -> tuple:
    """Run strategy with given parameters."""
    if params is None:
        # Best parameters found through optimization
        if version == "v1":
            params = {
                "rsi_period": 2,
                "rsi_entry": 10,
                "rsi_exit": 65,
                "use_ibs": True,
                "ibs_entry": 0.25,
            }
        elif version == "v2":
            params = {
                "rsi_period": 2,
                "cum_period": 2,
                "cum_entry": 35,
                "cum_exit": 65,
            }
        elif version == "v3":
            params = {
                "rsi_period": 2,
                "rsi_entry": 5,
                "hold_days": 5,
            }
        elif version == "v4":
            params = {
                "rsi_period": 2,
                "rsi_entry": 10,
                "ibs_entry": 0.2,
                "rsi_exit": 65,
            }

    # Select signal generator
    signal_funcs = {
        "v1": generate_signals_v1,
        "v2": generate_signals_v2,
        "v3": generate_signals_v3,
        "v4": generate_signals_v4,
        "v5": generate_signals_v5,
        "v6": generate_signals_v6,
        "v7": generate_signals_v7,
    }
    signal_func = signal_funcs[version]

    # Train period
    train_signals = signal_func(train_data, **params)
    train_returns, train_equity = backtest_multi_asset(train_data, train_signals)
    train_metrics = calculate_metrics(train_returns)

    # Test period (if provided)
    test_metrics = None
    if test_data:
        test_signals = signal_func(test_data, **params)
        test_returns, test_equity = backtest_multi_asset(test_data, test_signals)
        test_metrics = calculate_metrics(test_returns)

    return train_returns, train_metrics, test_metrics, params


def main():
    import argparse

    parser = argparse.ArgumentParser(description=f"{STRATEGY_NAME} Strategy")
    parser.add_argument("--optimize", action="store_true", help="Run optimization")
    parser.add_argument("--version", default="all", help="Signal version (v1-v4 or 'all')")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  {STRATEGY_NAME} - ETF Mean-Reversion Strategy")
    print(f"  Target: CAGR {TARGET.cagr:.1%}, DD {TARGET.max_dd:.1%}, Sharpe {TARGET.sharpe:.2f}")
    print(f"{'='*60}")

    # Load data
    print(f"\nLoading {SYMBOLS}...")
    data = load_multi_asset(SYMBOLS)
    print(f"  Date range: {list(data.values())[0].index.min()} to {list(data.values())[0].index.max()}")

    # Split into train/test
    train_data = {}
    test_data = {}
    for sym, df in data.items():
        train_data[sym], test_data[sym] = split_data(df, train_end="2020-12-31")

    print(f"  Train: {list(train_data.values())[0].index.min()} to {list(train_data.values())[0].index.max()}")
    print(f"  Test:  {list(test_data.values())[0].index.min()} to {list(test_data.values())[0].index.max()}")

    # Determine versions to run
    versions = ["v1", "v2", "v3", "v4", "v5", "v6", "v7"] if args.version == "all" else [args.version]

    best_result = None
    best_sharpe = -np.inf

    for version in versions:
        print(f"\n{'='*60}")
        print(f"  Testing Version {version}")
        print(f"{'='*60}")

        if args.optimize:
            results = optimize(train_data, version=version)
            if results:
                best_params = results[0]["params"]
            else:
                continue
        else:
            best_params = None

        # Run with best params
        train_returns, train_metrics, test_metrics, params = run_strategy(
            train_data, test_data, params=best_params, version=version
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

        # Create combined equity for plotting
        signal_funcs = {
            "v1": generate_signals_v1,
            "v2": generate_signals_v2,
            "v3": generate_signals_v3,
            "v4": generate_signals_v4,
            "v5": generate_signals_v5,
            "v6": generate_signals_v6,
            "v7": generate_signals_v7,
        }
        full_returns, _ = backtest_multi_asset(
            data,
            signal_funcs[best_result["version"]](data, **best_result["params"])
        )

        result = create_result(
            STRATEGY_NAME,
            best_result["params"],
            full_returns,
            pd.Series(1, index=full_returns.index),  # Placeholder
        )

        plot_equity_curve(
            result,
            benchmark=data["SPY"]["Close"],
            target=TARGET,
            output_path=output_dir / f"{STRATEGY_NAME.lower()}_equity.png",
        )

        return best_result

    return None


if __name__ == "__main__":
    main()
