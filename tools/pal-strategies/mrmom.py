# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "matplotlib>=3.8",
# ]
# ///
"""
MRMOM Strategy - Mean-Reversion/Momentum Regime Switching

Target Metrics (PAL):
- CAGR: 10.3%
- Max DD: -16.8%
- Sharpe: 1.15 (HIGHEST of all PAL strategies)

Hypothesis: Uses Momersion indicator to detect regime:
- Momersion > 50: Momentum regime -> use trend-following rules
- Momersion < 50: Mean-reversion regime -> use MR rules

Assets: SPY, QQQ, TLT, GLD (rotate based on regime)

Usage:
    uv run tools/pal-strategies/mrmom.py
    uv run tools/pal-strategies/mrmom.py --optimize
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.indicators import ibs, momersion, rate_of_change, rsi
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

STRATEGY_NAME = "MRMOM"
SYMBOLS = ["SPY", "QQQ", "TLT", "GLD"]
TARGET = TARGETS[STRATEGY_NAME]


def generate_signals_v1(
    data: dict[str, pd.DataFrame],
    mom_lookback: int = 252,
    mom_threshold: float = 50,
    rsi_period: int = 2,
    mr_rsi_entry: float = 20,
    mr_rsi_exit: float = 65,
    mom_roc_period: int = 20,
    mom_roc_threshold: float = 0.0,
) -> dict[str, pd.Series]:
    """
    Version 1: Basic regime switching with Momersion.

    In MR regime (Momersion < threshold):
        - Use RSI mean-reversion on SPY/QQQ
        - TLT as risk-off allocation

    In MOM regime (Momersion > threshold):
        - Use momentum (ROC) for asset selection
        - Allocate to strongest positive momentum
    """
    # Calculate Momersion on SPY (as market regime proxy)
    spy_returns = data["SPY"]["Close"].pct_change()
    mom_indicator = momersion(spy_returns, lookback=mom_lookback)

    # Get common index
    common_idx = data["SPY"].index

    signals = {}

    for sym, df in data.items():
        close = df["Close"]
        rsi_val = rsi(close, period=rsi_period)
        roc_val = rate_of_change(close, period=mom_roc_period)

        # MR regime signals (Momersion < threshold)
        mr_regime = mom_indicator < mom_threshold

        if sym in ["SPY", "QQQ"]:
            # RSI-based MR entry
            mr_entry = (rsi_val < mr_rsi_entry) & mr_regime
            mr_exit = (rsi_val > mr_rsi_exit) | ~mr_regime
        elif sym == "TLT":
            # TLT: buy in MR regime when equities not oversold
            spy_rsi = rsi(data["SPY"]["Close"], period=rsi_period)
            mr_entry = mr_regime & (spy_rsi > mr_rsi_entry)
            mr_exit = ~mr_regime
        else:  # GLD
            # GLD: very selective, only in extreme MR with low equity momentum
            spy_roc = rate_of_change(data["SPY"]["Close"], period=mom_roc_period)
            mr_entry = mr_regime & (spy_roc < -0.05)
            mr_exit = ~mr_regime | (spy_roc > 0)

        # MOM regime signals (Momersion > threshold)
        mom_regime = mom_indicator >= mom_threshold

        if sym in ["SPY", "QQQ"]:
            # Momentum: buy on positive ROC
            mom_entry = (roc_val > mom_roc_threshold) & mom_regime
            mom_exit = (roc_val < 0) | ~mom_regime
        elif sym == "TLT":
            # TLT: avoid in momentum regime (risk-on)
            mom_entry = pd.Series(False, index=df.index)
            mom_exit = pd.Series(True, index=df.index)
        else:  # GLD
            # GLD: buy on strong momentum only
            mom_entry = (roc_val > 0.03) & mom_regime
            mom_exit = (roc_val < 0) | ~mom_regime

        # Combine regimes
        entry = mr_entry | mom_entry
        exit_sig = mr_exit & mom_exit

        # Build position
        signal_events = pd.Series(0, index=df.index)
        signal_events[entry] = 1
        signal_events[exit_sig] = -1

        state = signal_events.replace(0, np.nan).ffill().fillna(0)
        position = np.where(state == 1, 1.0, 0.0)

        signals[sym] = pd.Series(position, index=df.index)

    return signals


def generate_signals_v2(
    data: dict[str, pd.DataFrame],
    mom_lookback: int = 126,
    mom_threshold: float = 50,
    rsi_period: int = 2,
    mr_rsi_entry: float = 15,
    hold_days: int = 3,
    mom_sma_period: int = 200,
) -> dict[str, pd.Series]:
    """
    Version 2: Simplified regime switching.

    In MR regime: RSI mean-reversion on equities with fixed holding
    In MOM regime: Trend-following (price > SMA)
    TLT/GLD: Risk-off allocation based on regime
    """
    spy_returns = data["SPY"]["Close"].pct_change()
    mom_indicator = momersion(spy_returns, lookback=mom_lookback)

    signals = {}

    for sym, df in data.items():
        close = df["Close"]
        rsi_val = rsi(close, period=rsi_period)
        sma = close.rolling(mom_sma_period).mean()

        # Regime detection
        mr_regime = mom_indicator < mom_threshold
        mom_regime = mom_indicator >= mom_threshold

        if sym in ["SPY", "QQQ"]:
            # MR: RSI oversold
            mr_entry = (rsi_val < mr_rsi_entry) & mr_regime

            # MOM: Price above SMA
            mom_entry = (close > sma) & mom_regime

            # Combined entry with holding period
            entry = mr_entry | mom_entry
            position = entry.astype(int).rolling(hold_days, min_periods=1).max()

        elif sym == "TLT":
            # TLT: Inverse to equity regime
            # Buy in MR regime OR when equities weak in MOM regime
            spy_close = data["SPY"]["Close"]
            spy_sma = spy_close.rolling(mom_sma_period).mean()

            entry = mr_regime | ((mom_regime) & (spy_close < spy_sma))
            position = entry.astype(int)

        else:  # GLD
            # GLD: Diversifier, limited allocation
            spy_returns = data["SPY"]["Close"].pct_change()
            recent_vol = spy_returns.rolling(20).std() * np.sqrt(252)

            # Buy GLD in high vol environments
            entry = recent_vol > 0.20
            position = entry.astype(int)

        signals[sym] = pd.Series(position, index=df.index).fillna(0)

    return signals


def generate_signals_v3(
    data: dict[str, pd.DataFrame],
    mom_lookback: int = 252,
    mr_threshold: float = 45,
    mom_threshold: float = 55,
    rsi_period: int = 2,
    rsi_entry: float = 20,
    rsi_exit: float = 50,
    hold_days: int = 5,
) -> dict[str, pd.Series]:
    """
    Version 3: Three-state regime with neutral zone.

    MR regime (Momersion < mr_threshold): Full MR allocation
    Neutral (mr_threshold <= Momersion <= mom_threshold): Reduced exposure
    MOM regime (Momersion > mom_threshold): Momentum allocation
    """
    spy_returns = data["SPY"]["Close"].pct_change()
    mom_indicator = momersion(spy_returns, lookback=mom_lookback)

    signals = {}

    # Regime states
    mr_regime = mom_indicator < mr_threshold
    neutral_regime = (mom_indicator >= mr_threshold) & (mom_indicator <= mom_threshold)
    mom_regime = mom_indicator > mom_threshold

    for sym, df in data.items():
        close = df["Close"]
        returns = close.pct_change()
        rsi_val = rsi(close, period=rsi_period)

        if sym in ["SPY", "QQQ"]:
            # MR regime: RSI entry
            mr_entry = (rsi_val < rsi_entry) & mr_regime
            mr_exit = rsi_val > rsi_exit

            # MOM regime: Stay long if positive momentum
            mom_12m = close / close.shift(252) - 1
            mom_entry = (mom_12m > 0) & mom_regime
            mom_exit = mom_12m < -0.1

            # Neutral: Reduced MR with tighter threshold
            neutral_entry = (rsi_val < rsi_entry * 0.5) & neutral_regime

            # Combine
            entry = mr_entry | mom_entry | neutral_entry
            exit_sig = mr_exit & mom_exit

            # Position with state tracking
            signal_events = pd.Series(0, index=df.index)
            signal_events[entry] = 1
            signal_events[exit_sig] = -1

            state = signal_events.replace(0, np.nan).ffill().fillna(0)
            position = np.where(state == 1, 1.0, 0.0)

        elif sym == "TLT":
            # TLT: Risk-off, inversely correlated
            spy_rsi = rsi(data["SPY"]["Close"], period=rsi_period)

            # Buy TLT in MR regime when SPY not oversold
            # Or in neutral regime as hedge
            entry = (mr_regime & (spy_rsi > rsi_entry)) | neutral_regime
            position = entry.astype(float)

        else:  # GLD
            # GLD: Tail hedge
            spy_returns = data["SPY"]["Close"].pct_change()
            spy_vol = spy_returns.rolling(20).std() * np.sqrt(252)

            # Buy GLD in high vol or MR regime
            entry = (spy_vol > 0.25) | mr_regime
            position = entry.astype(float) * 0.5  # Half weight

        signals[sym] = pd.Series(position, index=df.index).fillna(0)

    return signals


def generate_signals_v4(
    data: dict[str, pd.DataFrame],
    mom_lookback: int = 126,
    mom_threshold: float = 50,
    rsi_period: int = 3,
    rsi_entry: float = 25,
    ibs_entry: float = 0.3,
    hold_days: int = 2,
) -> dict[str, pd.Series]:
    """
    Version 4: Dual-filter MR with momentum overlay.

    MR regime: RSI + IBS combined filter
    MOM regime: Simple trend following
    """
    spy_returns = data["SPY"]["Close"].pct_change()
    mom_indicator = momersion(spy_returns, lookback=mom_lookback)

    signals = {}

    for sym, df in data.items():
        close = df["Close"]
        rsi_val = rsi(close, period=rsi_period)
        ibs_val = ibs(df["High"], df["Low"], close)

        mr_regime = mom_indicator < mom_threshold
        mom_regime = ~mr_regime

        if sym in ["SPY", "QQQ"]:
            # MR: Dual filter
            mr_entry = (rsi_val < rsi_entry) & (ibs_val < ibs_entry) & mr_regime

            # MOM: Above 50-day SMA
            sma50 = close.rolling(50).mean()
            mom_entry = (close > sma50) & mom_regime

            entry = mr_entry | mom_entry
            position = entry.astype(int).rolling(hold_days, min_periods=1).max()

        elif sym == "TLT":
            # TLT: Buy in MR, or when SPY below SMA in MOM
            spy_close = data["SPY"]["Close"]
            spy_sma = spy_close.rolling(200).mean()

            entry = mr_regime | (mom_regime & (spy_close < spy_sma))
            position = entry.astype(int)

        else:  # GLD
            # GLD: Vol-based allocation
            spy_returns = data["SPY"]["Close"].pct_change()
            spy_vol = spy_returns.rolling(20).std()
            vol_high = spy_vol > spy_vol.rolling(60).mean()

            entry = vol_high & mr_regime
            position = entry.astype(int)

        signals[sym] = pd.Series(position, index=df.index).fillna(0)

    return signals


def generate_signals_v5(
    data: dict[str, pd.DataFrame],
    mom_lookback: int = 252,
    mom_threshold: float = 50,
    rsi_period: int = 2,
    rsi_entry: float = 10,
    allocation_method: str = "equal",
) -> dict[str, pd.Series]:
    """
    Version 5: Pure regime rotation with risk parity flavor.

    MR regime: SPY+QQQ MR + TLT
    MOM regime: SPY+QQQ momentum + GLD as diversifier
    """
    spy_returns = data["SPY"]["Close"].pct_change()
    mom_indicator = momersion(spy_returns, lookback=mom_lookback)

    signals = {}

    mr_regime = mom_indicator < mom_threshold
    mom_regime = ~mr_regime

    for sym, df in data.items():
        close = df["Close"]
        rsi_val = rsi(close, period=rsi_period)

        if sym in ["SPY", "QQQ"]:
            # MR: RSI oversold
            mr_entry = (rsi_val < rsi_entry) & mr_regime

            # MOM: 12-month momentum positive
            mom_12m = close / close.shift(252) - 1
            mom_entry = (mom_12m > 0) & mom_regime

            # Combined
            entry = mr_entry | mom_entry
            position = entry.astype(float)

        elif sym == "TLT":
            # TLT: Risk-off in MR regime
            position = mr_regime.astype(float)

        else:  # GLD
            # GLD: Risk-off in MOM regime (inflation/tail hedge)
            position = mom_regime.astype(float) * 0.5

        signals[sym] = pd.Series(position, index=df.index).fillna(0)

    return signals


def backtest_multi_asset(
    data: dict[str, pd.DataFrame],
    signals: dict[str, pd.Series],
    cost_bps: float = 10,
) -> tuple[pd.Series, pd.Series]:
    """Backtest multi-asset strategy with dynamic weighting."""
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
            "mom_lookback": [126, 252],
            "mom_threshold": [45, 50, 55],
            "rsi_period": [2, 3],
            "mr_rsi_entry": [10, 15, 20, 25],
            "mr_rsi_exit": [55, 65, 75],
        }
        signal_func = generate_signals_v1
    elif version == "v2":
        param_grid = {
            "mom_lookback": [63, 126, 252],
            "mom_threshold": [45, 50, 55],
            "rsi_period": [2, 3],
            "mr_rsi_entry": [10, 15, 20],
            "hold_days": [1, 2, 3, 5],
        }
        signal_func = generate_signals_v2
    elif version == "v3":
        param_grid = {
            "mom_lookback": [126, 252],
            "mr_threshold": [40, 45, 50],
            "mom_threshold": [50, 55, 60],
            "rsi_entry": [15, 20, 25],
            "hold_days": [3, 5, 7],
        }
        signal_func = generate_signals_v3
    elif version == "v4":
        param_grid = {
            "mom_lookback": [63, 126, 252],
            "mom_threshold": [45, 50, 55],
            "rsi_period": [2, 3],
            "rsi_entry": [20, 25, 30],
            "ibs_entry": [0.2, 0.3, 0.4],
            "hold_days": [1, 2, 3],
        }
        signal_func = generate_signals_v4
    elif version == "v5":
        param_grid = {
            "mom_lookback": [126, 189, 252],
            "mom_threshold": [40, 45, 50, 55, 60],
            "rsi_period": [2, 3],
            "rsi_entry": [5, 10, 15, 20],
        }
        signal_func = generate_signals_v5
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
        except Exception as e:
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
    signal_funcs = {
        "v1": generate_signals_v1,
        "v2": generate_signals_v2,
        "v3": generate_signals_v3,
        "v4": generate_signals_v4,
        "v5": generate_signals_v5,
    }

    if params is None:
        # Default parameters
        defaults = {
            "v1": {"mom_lookback": 252, "mom_threshold": 50, "rsi_period": 2, "mr_rsi_entry": 20, "mr_rsi_exit": 65},
            "v2": {"mom_lookback": 126, "mom_threshold": 50, "rsi_period": 2, "mr_rsi_entry": 15, "hold_days": 3},
            "v3": {"mom_lookback": 252, "mr_threshold": 45, "mom_threshold": 55, "rsi_entry": 20, "hold_days": 5},
            "v4": {"mom_lookback": 126, "mom_threshold": 50, "rsi_period": 3, "rsi_entry": 25, "ibs_entry": 0.3, "hold_days": 2},
            "v5": {"mom_lookback": 252, "mom_threshold": 50, "rsi_period": 2, "rsi_entry": 10},
        }
        params = defaults.get(version, {})

    signal_func = signal_funcs[version]

    # Train period
    train_signals = signal_func(train_data, **params)
    train_returns, train_equity = backtest_multi_asset(train_data, train_signals)
    train_metrics = calculate_metrics(train_returns)

    # Test period
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
    parser.add_argument("--version", default="all", help="Signal version (v1-v5 or 'all')")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  {STRATEGY_NAME} - MR/Momentum Regime Switching Strategy")
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
    versions = ["v1", "v2", "v3", "v4", "v5"] if args.version == "all" else [args.version]

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

        # Create full period result
        signal_funcs = {
            "v1": generate_signals_v1,
            "v2": generate_signals_v2,
            "v3": generate_signals_v3,
            "v4": generate_signals_v4,
            "v5": generate_signals_v5,
        }
        full_returns, full_equity = backtest_multi_asset(
            data,
            signal_funcs[best_result["version"]](data, **best_result["params"])
        )

        result = create_result(
            STRATEGY_NAME,
            best_result["params"],
            full_returns,
            pd.Series(1, index=full_returns.index),
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
