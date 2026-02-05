# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "numpy>=1.24", "matplotlib>=3.8", "rich>=13.0"]
# ///
"""
MRETF Strategy - Reverse Engineered from Price Action Lab

Target Metrics (PAL):
- CAGR: 5.2%
- Max Drawdown: -9.3%
- Sharpe Ratio: 0.78

Strategy Type: Mean-reversion + breakouts on SPY, QQQ, TLT (long-only)

Hypothesis: Mean-reversion entries on SPY/QQQ with TLT as risk-off allocation.
Possibly includes breakout component or rotation based on strength.
Lower CAGR but very tight drawdown suggests defensive TLT allocation.
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.indicators import ibs, rsi
from tools.pal_strategies.base import (
    Backtest,
    load_data,
    run_backtest,
    run_multi_asset_backtest,
    split_train_test,
)

console = Console()

TARGET = {
    "CAGR": 0.052,
    "MDD": -0.093,
    "Sharpe": 0.78,
}

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_signals_mr_with_tlt(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    rsi_entry: float = 10,
    rsi_exit: float = 60,
    tlt_allocation: float = 0.5,
) -> dict[str, pd.Series]:
    """
    Mean-reversion on SPY/QQQ with constant TLT allocation.

    Entry: RSI < threshold on SPY or QQQ
    Exit: RSI > exit threshold
    TLT: Always hold fixed allocation as hedge
    """
    signals = {}

    for symbol in ["SPY", "QQQ"]:
        if symbol not in data:
            continue

        df = data[symbol]
        rsi_val = rsi(df["Close"], period=rsi_period)

        signal = pd.Series(0, index=df.index)
        in_position = False

        for i in range(1, len(df)):
            if not in_position:
                if rsi_val.iloc[i] < rsi_entry:
                    signal.iloc[i] = 1
                    in_position = True
            else:
                if rsi_val.iloc[i] > rsi_exit:
                    signal.iloc[i] = 0
                    in_position = False
                else:
                    signal.iloc[i] = 1

        signals[symbol] = signal

    # TLT: constant allocation (always long)
    if "TLT" in data:
        signals["TLT"] = pd.Series(1, index=data["TLT"].index)

    return signals


def generate_signals_rotation_mr(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    rsi_entry: float = 15,
    rsi_exit: float = 55,
    lookback: int = 20,
) -> dict[str, pd.Series]:
    """
    Rotation strategy: buy most oversold asset, switch to TLT when none oversold.

    Entry: Buy asset with lowest RSI if below threshold
    Exit: When RSI recovers, rotate to next oversold or TLT
    """
    equity_symbols = ["SPY", "QQQ"]

    # Calculate RSI for each asset
    rsi_df = pd.DataFrame()
    for symbol in equity_symbols:
        if symbol in data:
            rsi_df[symbol] = rsi(data[symbol]["Close"], period=rsi_period)

    # Find dates where data overlaps
    common_dates = rsi_df.dropna().index

    signals = {s: pd.Series(0.0, index=data[s].index) for s in data.keys()}

    current_position = None  # Track current holding

    for date in common_dates:
        # Get RSI values for this date
        rsi_values = {s: rsi_df.loc[date, s] for s in equity_symbols if s in rsi_df.columns}

        # Find most oversold equity
        oversold = {s: v for s, v in rsi_values.items() if v < rsi_entry}

        if current_position is None or current_position == "TLT":
            # Looking to enter equity
            if oversold:
                # Buy most oversold
                most_oversold = min(oversold, key=oversold.get)
                signals[most_oversold].loc[date] = 1.0
                current_position = most_oversold
            else:
                # Stay in TLT
                if "TLT" in signals:
                    signals["TLT"].loc[date] = 1.0
                current_position = "TLT"
        else:
            # Currently in equity position
            current_rsi = rsi_values.get(current_position, 100)

            if current_rsi > rsi_exit:
                # Exit and check for new entry
                if oversold:
                    most_oversold = min(oversold, key=oversold.get)
                    signals[most_oversold].loc[date] = 1.0
                    current_position = most_oversold
                else:
                    if "TLT" in signals:
                        signals["TLT"].loc[date] = 1.0
                    current_position = "TLT"
            else:
                # Stay in position
                signals[current_position].loc[date] = 1.0

    return signals


def generate_signals_tlt_filter(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    rsi_entry: float = 15,
    rsi_exit: float = 60,
    tlt_roc_period: int = 20,
    tlt_threshold: float = 0.0,
) -> dict[str, pd.Series]:
    """
    Trade SPY/QQQ only when TLT momentum is positive (risk-on).
    When TLT momentum negative (risk-off), hold TLT.
    """
    signals = {}

    # TLT momentum filter
    tlt_roc = data["TLT"]["Close"].pct_change(tlt_roc_period)
    risk_on = tlt_roc < tlt_threshold  # Bonds falling = risk-on

    for symbol in ["SPY", "QQQ"]:
        if symbol not in data:
            continue

        df = data[symbol]
        rsi_val = rsi(df["Close"], period=rsi_period)

        signal = pd.Series(0, index=df.index)
        in_position = False

        for i in range(1, len(df)):
            date = df.index[i]
            is_risk_on = risk_on.loc[date] if date in risk_on.index else True

            if not in_position:
                if rsi_val.iloc[i] < rsi_entry and is_risk_on:
                    signal.iloc[i] = 1
                    in_position = True
            else:
                if rsi_val.iloc[i] > rsi_exit or not is_risk_on:
                    signal.iloc[i] = 0
                    in_position = False
                else:
                    signal.iloc[i] = 1

        signals[symbol] = signal

    # TLT: hold when not in equities (inverse of equity signals)
    if "TLT" in data:
        equity_in_trade = signals.get("SPY", pd.Series(0, index=data["TLT"].index)).reindex(data["TLT"].index, fill_value=0)
        if "QQQ" in signals:
            equity_in_trade = equity_in_trade | signals["QQQ"].reindex(data["TLT"].index, fill_value=0)
        signals["TLT"] = (1 - equity_in_trade).clip(0, 1)

    return signals


def generate_signals_breakout_mr_combo(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    rsi_entry: float = 15,
    rsi_exit: float = 55,
    breakout_period: int = 20,
) -> dict[str, pd.Series]:
    """
    Combines mean-reversion and breakout:
    - Mean-reversion: Buy on RSI < threshold
    - Breakout: Buy on new high (TLT only for momentum)
    """
    signals = {}

    # SPY and QQQ: mean-reversion only
    for symbol in ["SPY", "QQQ"]:
        if symbol not in data:
            continue

        df = data[symbol]
        rsi_val = rsi(df["Close"], period=rsi_period)

        signal = pd.Series(0, index=df.index)
        in_position = False

        for i in range(1, len(df)):
            if not in_position:
                if rsi_val.iloc[i] < rsi_entry:
                    signal.iloc[i] = 1
                    in_position = True
            else:
                if rsi_val.iloc[i] > rsi_exit:
                    signal.iloc[i] = 0
                    in_position = False
                else:
                    signal.iloc[i] = 1

        signals[symbol] = signal

    # TLT: breakout (new high) + always some allocation
    if "TLT" in data:
        df = data["TLT"]
        high_n = df["Close"].rolling(breakout_period).max()
        at_high = df["Close"] >= high_n

        signal = pd.Series(0.5, index=df.index)  # Base 50% allocation
        signal[at_high] = 1.0  # Full allocation on breakout

        signals["TLT"] = signal

    return signals


def generate_signals_adaptive_allocation(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    rsi_entry: float = 20,
    hold_days: int = 3,
    vol_lookback: int = 20,
) -> dict[str, pd.Series]:
    """
    Adaptive allocation based on volatility:
    - High equity volatility: More TLT
    - Entry on RSI oversold, fixed holding period
    """
    signals = {}

    # Calculate equity volatility
    spy_vol = data["SPY"]["Close"].pct_change().rolling(vol_lookback).std() * np.sqrt(252)

    for symbol in ["SPY", "QQQ"]:
        if symbol not in data:
            continue

        df = data[symbol]
        rsi_val = rsi(df["Close"], period=rsi_period)

        signal = pd.Series(0, index=df.index)
        hold_counter = 0

        for i in range(1, len(df)):
            if hold_counter > 0:
                signal.iloc[i] = 1
                hold_counter -= 1
            elif rsi_val.iloc[i] < rsi_entry:
                signal.iloc[i] = 1
                hold_counter = hold_days

        signals[symbol] = signal

    # TLT: inversely proportional to equity signals
    if "TLT" in data:
        # Higher allocation when not in equities
        equity_exposure = (
            signals.get("SPY", pd.Series(0, index=data["TLT"].index)).reindex(data["TLT"].index, fill_value=0) +
            signals.get("QQQ", pd.Series(0, index=data["TLT"].index)).reindex(data["TLT"].index, fill_value=0)
        ) / 2
        signals["TLT"] = (1 - equity_exposure * 0.5).clip(0.5, 1)

    return signals


def run_optimization(train_data: dict[str, pd.DataFrame], verbose: bool = True) -> tuple[dict, Backtest]:
    """Run optimization to find best parameters."""
    best_result = None
    best_params = None
    best_score = -np.inf

    prices = {s: df["Close"] for s, df in train_data.items()}

    # Hypothesis 1: MR with constant TLT
    if verbose:
        console.print("[yellow]Testing Hypothesis 1: MR with constant TLT allocation[/]")

    for rsi_period in [2, 3]:
        for rsi_entry in [10, 15, 20, 25]:
            for rsi_exit in [50, 55, 60, 65]:
                signals = generate_signals_mr_with_tlt(
                    train_data, rsi_period, rsi_entry, rsi_exit
                )
                bt = run_multi_asset_backtest(prices, signals)

                # Prioritize low drawdown strategies
                score = bt.sharpe + (0.1 if bt.max_dd > -0.15 else 0)

                if score > best_score and bt.num_trades > 20:
                    best_score = score
                    best_params = {
                        "type": "mr_with_tlt",
                        "rsi_period": rsi_period,
                        "rsi_entry": rsi_entry,
                        "rsi_exit": rsi_exit,
                    }
                    best_result = bt

    # Hypothesis 2: Rotation MR
    if verbose:
        console.print("[yellow]Testing Hypothesis 2: Rotation mean-reversion[/]")

    for rsi_period in [2, 3]:
        for rsi_entry in [10, 15, 20]:
            for rsi_exit in [50, 55, 60]:
                signals = generate_signals_rotation_mr(
                    train_data, rsi_period, rsi_entry, rsi_exit
                )
                bt = run_multi_asset_backtest(prices, signals)

                score = bt.sharpe + (0.1 if bt.max_dd > -0.15 else 0)

                if score > best_score and bt.num_trades > 20:
                    best_score = score
                    best_params = {
                        "type": "rotation_mr",
                        "rsi_period": rsi_period,
                        "rsi_entry": rsi_entry,
                        "rsi_exit": rsi_exit,
                    }
                    best_result = bt

    # Hypothesis 3: TLT filter
    if verbose:
        console.print("[yellow]Testing Hypothesis 3: TLT momentum filter[/]")

    for rsi_period in [2, 3]:
        for rsi_entry in [10, 15, 20]:
            for rsi_exit in [55, 60, 65]:
                for tlt_roc_period in [10, 20, 30]:
                    signals = generate_signals_tlt_filter(
                        train_data, rsi_period, rsi_entry, rsi_exit, tlt_roc_period
                    )
                    bt = run_multi_asset_backtest(prices, signals)

                    score = bt.sharpe + (0.1 if bt.max_dd > -0.15 else 0)

                    if score > best_score and bt.num_trades > 20:
                        best_score = score
                        best_params = {
                            "type": "tlt_filter",
                            "rsi_period": rsi_period,
                            "rsi_entry": rsi_entry,
                            "rsi_exit": rsi_exit,
                            "tlt_roc_period": tlt_roc_period,
                        }
                        best_result = bt

    # Hypothesis 4: Breakout + MR combo
    if verbose:
        console.print("[yellow]Testing Hypothesis 4: Breakout + MR combo[/]")

    for rsi_period in [2, 3]:
        for rsi_entry in [10, 15, 20]:
            for rsi_exit in [50, 55, 60]:
                for breakout_period in [20, 50]:
                    signals = generate_signals_breakout_mr_combo(
                        train_data, rsi_period, rsi_entry, rsi_exit, breakout_period
                    )
                    bt = run_multi_asset_backtest(prices, signals)

                    score = bt.sharpe + (0.1 if bt.max_dd > -0.15 else 0)

                    if score > best_score and bt.num_trades > 20:
                        best_score = score
                        best_params = {
                            "type": "breakout_mr_combo",
                            "rsi_period": rsi_period,
                            "rsi_entry": rsi_entry,
                            "rsi_exit": rsi_exit,
                            "breakout_period": breakout_period,
                        }
                        best_result = bt

    # Hypothesis 5: Adaptive allocation
    if verbose:
        console.print("[yellow]Testing Hypothesis 5: Adaptive allocation[/]")

    for rsi_period in [2, 3]:
        for rsi_entry in [15, 20, 25]:
            for hold_days in [2, 3, 4, 5]:
                signals = generate_signals_adaptive_allocation(
                    train_data, rsi_period, rsi_entry, hold_days
                )
                bt = run_multi_asset_backtest(prices, signals)

                score = bt.sharpe + (0.1 if bt.max_dd > -0.15 else 0)

                if score > best_score and bt.num_trades > 20:
                    best_score = score
                    best_params = {
                        "type": "adaptive_allocation",
                        "rsi_period": rsi_period,
                        "rsi_entry": rsi_entry,
                        "hold_days": hold_days,
                    }
                    best_result = bt

    if verbose:
        console.print(f"[green]Best parameters:[/] {best_params}")
        console.print(f"[green]Best Sharpe:[/] {best_score:.3f}")

    return best_params, best_result


def apply_strategy(data: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.Series]:
    """Apply strategy with given parameters."""
    if params["type"] == "mr_with_tlt":
        return generate_signals_mr_with_tlt(
            data, params["rsi_period"], params["rsi_entry"], params["rsi_exit"]
        )
    elif params["type"] == "rotation_mr":
        return generate_signals_rotation_mr(
            data, params["rsi_period"], params["rsi_entry"], params["rsi_exit"]
        )
    elif params["type"] == "tlt_filter":
        return generate_signals_tlt_filter(
            data, params["rsi_period"], params["rsi_entry"],
            params["rsi_exit"], params["tlt_roc_period"]
        )
    elif params["type"] == "breakout_mr_combo":
        return generate_signals_breakout_mr_combo(
            data, params["rsi_period"], params["rsi_entry"],
            params["rsi_exit"], params["breakout_period"]
        )
    elif params["type"] == "adaptive_allocation":
        return generate_signals_adaptive_allocation(
            data, params["rsi_period"], params["rsi_entry"], params["hold_days"]
        )
    else:
        raise ValueError(f"Unknown strategy type: {params['type']}")


def main():
    """Main entry point."""
    console.print("[bold blue]MRETF Strategy - Reverse Engineering[/]")
    console.print(f"Target: CAGR {TARGET['CAGR']:.1%}, MDD {TARGET['MDD']:.1%}, Sharpe {TARGET['Sharpe']:.2f}")
    console.print("-" * 60)

    # Load data
    console.print("\n[cyan]Loading data...[/]")
    spy = load_data("SPY")
    qqq = load_data("QQQ")
    tlt = load_data("TLT")
    full_data = {"SPY": spy, "QQQ": qqq, "TLT": tlt}

    # Split
    train_spy, test_spy = split_train_test(spy)
    train_qqq, test_qqq = split_train_test(qqq)
    train_tlt, test_tlt = split_train_test(tlt)
    train_data = {"SPY": train_spy, "QQQ": train_qqq, "TLT": train_tlt}
    test_data = {"SPY": test_spy, "QQQ": test_qqq, "TLT": test_tlt}

    console.print(f"Training period: {train_spy.index.min().date()} to {train_spy.index.max().date()}")
    console.print(f"Test period: {test_spy.index.min().date()} to {test_spy.index.max().date()}")

    # Optimize
    console.print("\n[cyan]Running optimization on training data...[/]")
    best_params, train_bt = run_optimization(train_data)

    # Training results
    console.print("\n[bold]Training Period Results:[/]")
    metrics = train_bt.summary()
    table = Table(title="Training Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Achieved", style="green")
    table.add_column("Target", style="yellow")

    table.add_row("CAGR", f"{metrics['CAGR']:.2%}", f"{TARGET['CAGR']:.1%}")
    table.add_row("Max DD", f"{metrics['Max DD']:.2%}", f"{TARGET['MDD']:.1%}")
    table.add_row("Sharpe", f"{metrics['Sharpe']:.2f}", f"{TARGET['Sharpe']:.2f}")
    table.add_row("Trades", f"{metrics['Trades']}", "-")
    console.print(table)

    # Test validation
    console.print("\n[cyan]Validating on test data...[/]")
    test_prices = {s: df["Close"] for s, df in test_data.items()}
    test_signals = apply_strategy(test_data, best_params)
    test_bt = run_multi_asset_backtest(test_prices, test_signals)

    console.print("\n[bold]Test Period Results:[/]")
    test_metrics = test_bt.summary()
    table = Table(title="Test Metrics (Out-of-Sample)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("CAGR", f"{test_metrics['CAGR']:.2%}")
    table.add_row("Max DD", f"{test_metrics['Max DD']:.2%}")
    table.add_row("Sharpe", f"{test_metrics['Sharpe']:.2f}")
    table.add_row("Trades", f"{test_metrics['Trades']}")
    console.print(table)

    # Full period
    console.print("\n[cyan]Running on full period...[/]")
    full_prices = {s: df["Close"] for s, df in full_data.items()}
    full_signals = apply_strategy(full_data, best_params)
    full_bt = run_multi_asset_backtest(full_prices, full_signals)
    full_metrics = full_bt.summary()

    console.print("\n[bold]Full Period Results:[/]")
    console.print(f"CAGR: {full_metrics['CAGR']:.2%} (target: {TARGET['CAGR']:.1%})")
    console.print(f"Max DD: {full_metrics['Max DD']:.2%} (target: {TARGET['MDD']:.1%})")
    console.print(f"Sharpe: {full_metrics['Sharpe']:.2f} (target: {TARGET['Sharpe']:.2f})")

    # Check tolerance
    cagr_ok = abs(full_metrics["CAGR"] - TARGET["CAGR"]) <= 0.03
    mdd_ok = abs(full_metrics["Max DD"] - TARGET["MDD"]) <= 0.10
    sharpe_ok = abs(full_metrics["Sharpe"] - TARGET["Sharpe"]) <= 0.15

    if cagr_ok and mdd_ok and sharpe_ok:
        console.print("\n[bold green]✓ Strategy successfully reverse-engineered![/]")
    else:
        console.print("\n[bold yellow]Strategy partially matched target metrics[/]")

    # Save outputs
    output_dir = PROJECT_ROOT / "output" / "claude-reverse-engineer-strategies-ziZsY" / "pal-reverse-engineering"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(full_bt.equity.index, full_bt.equity.values, label="MRETF Strategy", color="blue")

    # Buy & hold comparison
    bh = (spy["Close"]/spy["Close"].iloc[0] + qqq["Close"]/qqq["Close"].iloc[0] + tlt["Close"]/tlt["Close"].iloc[0]) / 3
    ax.plot(bh.index, bh.values, label="SPY+QQQ+TLT Equal Weight", color="gray", alpha=0.5)

    ax.set_title(f"MRETF Strategy Equity Curve\nCAGR: {full_metrics['CAGR']:.1%}, Sharpe: {full_metrics['Sharpe']:.2f}, MaxDD: {full_metrics['Max DD']:.1%}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($1 initial)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    chart_path = output_dir / f"mretf_equity_{TIMESTAMP}.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    console.print(f"\n[green]Saved equity curve:[/] {chart_path}")

    # Report
    report = f"""# MRETF Strategy - Reverse Engineered

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Target Metrics (PAL)

| Metric | Target |
|--------|--------|
| CAGR | {TARGET['CAGR']:.1%} |
| Max Drawdown | {TARGET['MDD']:.1%} |
| Sharpe Ratio | {TARGET['Sharpe']:.2f} |

## Achieved Metrics

| Metric | Achieved | Target | Match |
|--------|----------|--------|-------|
| CAGR | {full_metrics['CAGR']:.2%} | {TARGET['CAGR']:.1%} | {'✓' if cagr_ok else '✗'} |
| Max DD | {full_metrics['Max DD']:.2%} | {TARGET['MDD']:.1%} | {'✓' if mdd_ok else '✗'} |
| Sharpe | {full_metrics['Sharpe']:.2f} | {TARGET['Sharpe']:.2f} | {'✓' if sharpe_ok else '✗'} |

## Strategy Rules

**Type:** {best_params['type']}

**Parameters:**
```python
{best_params}
```

## Validation Status

**Overall:** {'✓ Successfully reverse-engineered' if (cagr_ok and mdd_ok and sharpe_ok) else '⚠ Partially matched'}
"""

    report_path = output_dir / f"mretf_report_{TIMESTAMP}.md"
    report_path.write_text(report)
    console.print(f"[green]Saved report:[/] {report_path}")

    return best_params, full_metrics


if __name__ == "__main__":
    params, metrics = main()
