# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "numpy>=1.24", "matplotlib>=3.8", "rich>=13.0"]
# ///
"""
B2S2ETF Strategy - Reverse Engineered from Price Action Lab

Target Metrics (PAL):
- CAGR: 9.1%
- Max Drawdown: -30.6%
- Sharpe Ratio: 0.63

Strategy Type: Mean-reversion on SPY only (long-only)

Hypothesis: "B2S2" likely means "Buy 2, Sell 2" - buy after 2 consecutive down days,
sell after 2 consecutive up days (or fixed holding period).
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.indicators import consecutive_days, cumulative_rsi, ibs, rsi
from tools.pal_strategies.base import (
    Backtest,
    load_data,
    run_backtest,
    split_train_test,
)

console = Console()

# Target metrics from PAL
TARGET = {
    "CAGR": 0.091,
    "MDD": -0.306,
    "Sharpe": 0.63,
}

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_signals_consecutive(
    df: pd.DataFrame,
    down_days: int = 2,
    up_days: int = 2,
) -> pd.Series:
    """
    Classic B2S2: Buy after N down days, sell after N up days.

    Args:
        df: OHLCV DataFrame
        down_days: Number of consecutive down days to trigger entry
        up_days: Number of consecutive up days to trigger exit

    Returns:
        Signal series (1=long, 0=flat)
    """
    returns = df["Close"].pct_change()

    signal = pd.Series(0, index=df.index)
    in_position = False
    consecutive_down = 0
    consecutive_up = 0

    for i in range(1, len(df)):
        ret = returns.iloc[i]

        # Count consecutive days
        if ret < 0:
            consecutive_down += 1
            consecutive_up = 0
        elif ret > 0:
            consecutive_up += 1
            consecutive_down = 0
        else:
            consecutive_down = 0
            consecutive_up = 0

        if not in_position:
            # Entry: N consecutive down days
            if consecutive_down >= down_days:
                signal.iloc[i] = 1
                in_position = True
        else:
            # Exit: N consecutive up days
            if consecutive_up >= up_days:
                signal.iloc[i] = 0
                in_position = False
            else:
                signal.iloc[i] = 1

    return signal


def generate_signals_b2s2_rsi(
    df: pd.DataFrame,
    down_days: int = 2,
    rsi_period: int = 2,
    rsi_threshold: float = 30,
    hold_days: int = 2,
) -> pd.Series:
    """
    B2S2 with RSI filter: Buy after N down days AND RSI < threshold.

    Args:
        df: OHLCV DataFrame
        down_days: Consecutive down days for entry
        rsi_period: RSI period
        rsi_threshold: RSI must be below this for entry
        hold_days: Fixed holding period for exit

    Returns:
        Signal series
    """
    returns = df["Close"].pct_change()
    rsi_val = rsi(df["Close"], period=rsi_period)

    signal = pd.Series(0, index=df.index)
    hold_counter = 0
    consecutive_down = 0

    for i in range(1, len(df)):
        ret = returns.iloc[i]

        # Count consecutive down days
        if ret < 0:
            consecutive_down += 1
        else:
            consecutive_down = 0

        if hold_counter > 0:
            signal.iloc[i] = 1
            hold_counter -= 1
        elif consecutive_down >= down_days and rsi_val.iloc[i] < rsi_threshold:
            signal.iloc[i] = 1
            hold_counter = hold_days

    return signal


def generate_signals_b2s2_ibs(
    df: pd.DataFrame,
    down_days: int = 2,
    ibs_threshold: float = 0.3,
    exit_days: int = 2,
) -> pd.Series:
    """
    B2S2 with IBS filter: Buy after N down days AND IBS < threshold.

    Args:
        df: OHLCV DataFrame
        down_days: Consecutive down days for entry
        ibs_threshold: IBS must be below this for entry
        exit_days: Fixed holding period or consecutive up days

    Returns:
        Signal series
    """
    returns = df["Close"].pct_change()
    ibs_val = ibs(df["High"], df["Low"], df["Close"])

    signal = pd.Series(0, index=df.index)
    hold_counter = 0
    consecutive_down = 0

    for i in range(1, len(df)):
        ret = returns.iloc[i]

        if ret < 0:
            consecutive_down += 1
        else:
            consecutive_down = 0

        if hold_counter > 0:
            signal.iloc[i] = 1
            hold_counter -= 1
        elif consecutive_down >= down_days and ibs_val.iloc[i] < ibs_threshold:
            signal.iloc[i] = 1
            hold_counter = exit_days

    return signal


def generate_signals_b2s2_combined(
    df: pd.DataFrame,
    down_days: int = 2,
    rsi_period: int = 2,
    rsi_threshold: float = 30,
    ibs_threshold: float = 0.3,
    rsi_exit: float = 70,
) -> pd.Series:
    """
    B2S2 with both RSI and IBS filters.

    Args:
        df: OHLCV DataFrame
        down_days: Consecutive down days for entry
        rsi_period: RSI period
        rsi_threshold: RSI entry threshold
        ibs_threshold: IBS entry threshold
        rsi_exit: RSI exit threshold

    Returns:
        Signal series
    """
    returns = df["Close"].pct_change()
    rsi_val = rsi(df["Close"], period=rsi_period)
    ibs_val = ibs(df["High"], df["Low"], df["Close"])

    signal = pd.Series(0, index=df.index)
    in_position = False
    consecutive_down = 0

    for i in range(1, len(df)):
        ret = returns.iloc[i]

        if ret < 0:
            consecutive_down += 1
        else:
            consecutive_down = 0

        if not in_position:
            # Entry: down days + RSI + IBS
            entry = (
                consecutive_down >= down_days and
                rsi_val.iloc[i] < rsi_threshold and
                ibs_val.iloc[i] < ibs_threshold
            )
            if entry:
                signal.iloc[i] = 1
                in_position = True
        else:
            # Exit: RSI recovers
            if rsi_val.iloc[i] > rsi_exit:
                signal.iloc[i] = 0
                in_position = False
            else:
                signal.iloc[i] = 1

    return signal


def generate_signals_b2s2_sma(
    df: pd.DataFrame,
    down_days: int = 2,
    up_days: int = 2,
    sma_period: int = 200,
    require_above_sma: bool = True,
) -> pd.Series:
    """
    B2S2 with trend filter - only trade above SMA.

    Args:
        df: OHLCV DataFrame
        down_days: Consecutive down days for entry
        up_days: Consecutive up days for exit
        sma_period: SMA period for trend filter
        require_above_sma: Only trade when price > SMA

    Returns:
        Signal series
    """
    returns = df["Close"].pct_change()
    sma = df["Close"].rolling(sma_period).mean()

    signal = pd.Series(0, index=df.index)
    in_position = False
    consecutive_down = 0
    consecutive_up = 0

    for i in range(1, len(df)):
        ret = returns.iloc[i]

        if ret < 0:
            consecutive_down += 1
            consecutive_up = 0
        elif ret > 0:
            consecutive_up += 1
            consecutive_down = 0
        else:
            consecutive_down = 0
            consecutive_up = 0

        above_sma = df["Close"].iloc[i] > sma.iloc[i] if pd.notna(sma.iloc[i]) else False

        if not in_position:
            trend_ok = above_sma if require_above_sma else True
            if consecutive_down >= down_days and trend_ok:
                signal.iloc[i] = 1
                in_position = True
        else:
            if consecutive_up >= up_days:
                signal.iloc[i] = 0
                in_position = False
            else:
                signal.iloc[i] = 1

    return signal


def run_optimization(train_df: pd.DataFrame, verbose: bool = True) -> tuple[dict, Backtest]:
    """Run optimization to find best parameters."""
    best_result = None
    best_params = None
    best_score = -np.inf

    # Hypothesis 1: Classic B2S2
    if verbose:
        console.print("[yellow]Testing Hypothesis 1: Classic B2S2[/]")

    for down_days in [2, 3, 4]:
        for up_days in [1, 2, 3]:
            signals = generate_signals_consecutive(train_df, down_days, up_days)
            bt = run_backtest(train_df["Close"], signals)

            if bt.sharpe > best_score and bt.num_trades > 30:
                best_score = bt.sharpe
                best_params = {
                    "type": "consecutive",
                    "down_days": down_days,
                    "up_days": up_days,
                }
                best_result = bt

    # Hypothesis 2: B2S2 + RSI
    if verbose:
        console.print("[yellow]Testing Hypothesis 2: B2S2 + RSI[/]")

    for down_days in [2, 3]:
        for rsi_period in [2, 3, 5]:
            for rsi_threshold in [20, 30, 40, 50]:
                for hold_days in [1, 2, 3, 4, 5]:
                    signals = generate_signals_b2s2_rsi(
                        train_df, down_days, rsi_period, rsi_threshold, hold_days
                    )
                    bt = run_backtest(train_df["Close"], signals)

                    if bt.sharpe > best_score and bt.num_trades > 30:
                        best_score = bt.sharpe
                        best_params = {
                            "type": "b2s2_rsi",
                            "down_days": down_days,
                            "rsi_period": rsi_period,
                            "rsi_threshold": rsi_threshold,
                            "hold_days": hold_days,
                        }
                        best_result = bt

    # Hypothesis 3: B2S2 + IBS
    if verbose:
        console.print("[yellow]Testing Hypothesis 3: B2S2 + IBS[/]")

    for down_days in [2, 3]:
        for ibs_threshold in [0.2, 0.3, 0.4, 0.5]:
            for exit_days in [1, 2, 3, 4, 5]:
                signals = generate_signals_b2s2_ibs(
                    train_df, down_days, ibs_threshold, exit_days
                )
                bt = run_backtest(train_df["Close"], signals)

                if bt.sharpe > best_score and bt.num_trades > 30:
                    best_score = bt.sharpe
                    best_params = {
                        "type": "b2s2_ibs",
                        "down_days": down_days,
                        "ibs_threshold": ibs_threshold,
                        "exit_days": exit_days,
                    }
                    best_result = bt

    # Hypothesis 4: B2S2 + RSI + IBS
    if verbose:
        console.print("[yellow]Testing Hypothesis 4: B2S2 + RSI + IBS[/]")

    for down_days in [2, 3]:
        for rsi_period in [2, 3]:
            for rsi_threshold in [25, 30, 35, 40]:
                for ibs_threshold in [0.25, 0.3, 0.35, 0.4]:
                    for rsi_exit in [60, 65, 70, 75]:
                        signals = generate_signals_b2s2_combined(
                            train_df, down_days, rsi_period, rsi_threshold, ibs_threshold, rsi_exit
                        )
                        bt = run_backtest(train_df["Close"], signals)

                        if bt.sharpe > best_score and bt.num_trades > 30:
                            best_score = bt.sharpe
                            best_params = {
                                "type": "b2s2_combined",
                                "down_days": down_days,
                                "rsi_period": rsi_period,
                                "rsi_threshold": rsi_threshold,
                                "ibs_threshold": ibs_threshold,
                                "rsi_exit": rsi_exit,
                            }
                            best_result = bt

    # Hypothesis 5: B2S2 + SMA trend filter
    if verbose:
        console.print("[yellow]Testing Hypothesis 5: B2S2 + SMA trend filter[/]")

    for down_days in [2, 3, 4]:
        for up_days in [1, 2, 3]:
            for sma_period in [50, 100, 200]:
                for above_sma in [True, False]:
                    signals = generate_signals_b2s2_sma(
                        train_df, down_days, up_days, sma_period, above_sma
                    )
                    bt = run_backtest(train_df["Close"], signals)

                    if bt.sharpe > best_score and bt.num_trades > 30:
                        best_score = bt.sharpe
                        best_params = {
                            "type": "b2s2_sma",
                            "down_days": down_days,
                            "up_days": up_days,
                            "sma_period": sma_period,
                            "require_above_sma": above_sma,
                        }
                        best_result = bt

    if verbose:
        console.print(f"[green]Best parameters:[/] {best_params}")
        console.print(f"[green]Best Sharpe:[/] {best_score:.3f}")

    return best_params, best_result


def apply_strategy(df: pd.DataFrame, params: dict) -> pd.Series:
    """Apply strategy with given parameters."""
    if params["type"] == "consecutive":
        return generate_signals_consecutive(df, params["down_days"], params["up_days"])
    elif params["type"] == "b2s2_rsi":
        return generate_signals_b2s2_rsi(
            df, params["down_days"], params["rsi_period"],
            params["rsi_threshold"], params["hold_days"]
        )
    elif params["type"] == "b2s2_ibs":
        return generate_signals_b2s2_ibs(
            df, params["down_days"], params["ibs_threshold"], params["exit_days"]
        )
    elif params["type"] == "b2s2_combined":
        return generate_signals_b2s2_combined(
            df, params["down_days"], params["rsi_period"],
            params["rsi_threshold"], params["ibs_threshold"], params["rsi_exit"]
        )
    elif params["type"] == "b2s2_sma":
        return generate_signals_b2s2_sma(
            df, params["down_days"], params["up_days"],
            params["sma_period"], params["require_above_sma"]
        )
    else:
        raise ValueError(f"Unknown strategy type: {params['type']}")


def main():
    """Main entry point."""
    console.print("[bold blue]B2S2ETF Strategy - Reverse Engineering[/]")
    console.print(f"Target: CAGR {TARGET['CAGR']:.1%}, MDD {TARGET['MDD']:.1%}, Sharpe {TARGET['Sharpe']:.2f}")
    console.print("-" * 60)

    # Load data
    console.print("\n[cyan]Loading data...[/]")
    spy = load_data("SPY")

    # Split into train/test
    train_df, test_df = split_train_test(spy)

    console.print(f"Training period: {train_df.index.min().date()} to {train_df.index.max().date()}")
    console.print(f"Test period: {test_df.index.min().date()} to {test_df.index.max().date()}")

    # Run optimization
    console.print("\n[cyan]Running optimization on training data...[/]")
    best_params, train_bt = run_optimization(train_df)

    # Display training results
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
    table.add_row("Win Rate", f"{metrics['Win Rate']:.1%}", "-")
    console.print(table)

    # Validate on test data
    console.print("\n[cyan]Validating on test data...[/]")
    test_signals = apply_strategy(test_df, best_params)
    test_bt = run_backtest(test_df["Close"], test_signals)

    console.print("\n[bold]Test Period Results:[/]")
    test_metrics = test_bt.summary()
    table = Table(title="Test Metrics (Out-of-Sample)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("CAGR", f"{test_metrics['CAGR']:.2%}")
    table.add_row("Max DD", f"{test_metrics['Max DD']:.2%}")
    table.add_row("Sharpe", f"{test_metrics['Sharpe']:.2f}")
    table.add_row("Trades", f"{test_metrics['Trades']}")
    table.add_row("Win Rate", f"{test_metrics['Win Rate']:.1%}")
    console.print(table)

    # Run on full period
    console.print("\n[cyan]Running on full period...[/]")
    full_signals = apply_strategy(spy, best_params)
    full_bt = run_backtest(spy["Close"], full_signals)
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

    # Plot equity curve
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(full_bt.equity.index, full_bt.equity.values, label="B2S2ETF Strategy", color="blue")

    spy_bh = spy["Close"] / spy["Close"].iloc[0]
    ax.plot(spy_bh.index, spy_bh.values, label="SPY Buy & Hold", color="gray", alpha=0.5)

    ax.set_title(f"B2S2ETF Strategy Equity Curve\nCAGR: {full_metrics['CAGR']:.1%}, Sharpe: {full_metrics['Sharpe']:.2f}, MaxDD: {full_metrics['Max DD']:.1%}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($1 initial)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    chart_path = output_dir / f"b2s2etf_equity_{TIMESTAMP}.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    console.print(f"\n[green]Saved equity curve:[/] {chart_path}")

    # Save report
    report = f"""# B2S2ETF Strategy - Reverse Engineered

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Target Metrics (PAL)

| Metric | Target |
|--------|--------|
| CAGR | {TARGET['CAGR']:.1%} |
| Max Drawdown | {TARGET['MDD']:.1%} |
| Sharpe Ratio | {TARGET['Sharpe']:.2f} |

## Achieved Metrics

### Full Period (2005-2025)

| Metric | Achieved | Target | Match |
|--------|----------|--------|-------|
| CAGR | {full_metrics['CAGR']:.2%} | {TARGET['CAGR']:.1%} | {'✓' if cagr_ok else '✗'} |
| Max DD | {full_metrics['Max DD']:.2%} | {TARGET['MDD']:.1%} | {'✓' if mdd_ok else '✗'} |
| Sharpe | {full_metrics['Sharpe']:.2f} | {TARGET['Sharpe']:.2f} | {'✓' if sharpe_ok else '✗'} |
| Trades | {full_metrics['Trades']} | - | - |
| Win Rate | {full_metrics['Win Rate']:.1%} | - | - |

## Strategy Rules

**Type:** {best_params['type']}

**Parameters:**
```python
{best_params}
```

## Validation Status

**Overall:** {'✓ Successfully reverse-engineered' if (cagr_ok and mdd_ok and sharpe_ok) else '⚠ Partially matched'}

## Equity Curve

![B2S2ETF Equity Curve](b2s2etf_equity_{TIMESTAMP}.png)
"""

    report_path = output_dir / f"b2s2etf_report_{TIMESTAMP}.md"
    report_path.write_text(report)
    console.print(f"[green]Saved report:[/] {report_path}")

    return best_params, full_metrics


if __name__ == "__main__":
    params, metrics = main()
