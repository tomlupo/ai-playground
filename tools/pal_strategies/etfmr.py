# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "numpy>=1.24", "matplotlib>=3.8", "rich>=13.0"]
# ///
"""
ETFMR Strategy - Reverse Engineered from Price Action Lab

Target Metrics (PAL):
- CAGR: 10.0%
- Max Drawdown: -22.9%
- Sharpe Ratio: 0.82

Strategy Type: Mean-reversion on SPY and QQQ (long-only)

Hypothesis: RSI-2/RSI-3 based mean reversion with possible IBS filter.
When both ETFs are oversold, buy equal weight. Exit when RSI recovers.
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

from shared.indicators import cumulative_rsi, ibs, rsi
from tools.pal_strategies.base import (
    Backtest,
    load_data,
    optimize_strategy,
    plot_equity_comparison,
    run_backtest,
    run_multi_asset_backtest,
    split_train_test,
)

console = Console()

# Target metrics from PAL
TARGET = {
    "CAGR": 0.10,
    "MDD": -0.229,
    "Sharpe": 0.82,
}

# Timestamp for outputs
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_signals_rsi_combined(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    rsi_entry: float = 10,
    rsi_exit: float = 70,
    require_both: bool = True,
) -> dict[str, pd.Series]:
    """
    RSI-based mean reversion signals.

    Entry: RSI < entry_threshold (on both assets if require_both=True)
    Exit: RSI > exit_threshold

    Args:
        data: Dict of symbol -> DataFrame
        rsi_period: RSI lookback period
        rsi_entry: Entry threshold (oversold)
        rsi_exit: Exit threshold (overbought/recovered)
        require_both: Require both assets to be oversold

    Returns:
        Dict of symbol -> signals
    """
    signals = {}

    # Calculate RSI for each asset
    rsi_values = {}
    for symbol, df in data.items():
        rsi_values[symbol] = rsi(df["Close"], period=rsi_period)

    # Determine combined entry condition
    if require_both:
        # Both must be oversold
        combined_entry = pd.DataFrame(rsi_values).max(axis=1) < rsi_entry
    else:
        # Either oversold
        combined_entry = pd.DataFrame(rsi_values).min(axis=1) < rsi_entry

    for symbol, df in data.items():
        rsi_val = rsi_values[symbol]

        # Create signal series
        signal = pd.Series(0, index=df.index)

        # Track position state
        in_position = False

        for i in range(1, len(df)):
            date = df.index[i]

            if not in_position:
                # Check entry
                if combined_entry.iloc[i]:
                    signal.iloc[i] = 1
                    in_position = True
            else:
                # Check exit
                if rsi_val.iloc[i] > rsi_exit:
                    signal.iloc[i] = 0
                    in_position = False
                else:
                    signal.iloc[i] = 1  # Stay in position

        signals[symbol] = signal

    return signals


def generate_signals_rsi_ibs(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    rsi_entry: float = 10,
    ibs_entry: float = 0.25,
    hold_days: int = 1,
) -> dict[str, pd.Series]:
    """
    RSI + IBS combined mean reversion signals.

    Entry: RSI < threshold AND IBS < threshold
    Exit: After N days

    Args:
        data: Dict of symbol -> DataFrame
        rsi_period: RSI lookback period
        rsi_entry: RSI entry threshold
        ibs_entry: IBS entry threshold
        hold_days: Number of days to hold after entry

    Returns:
        Dict of symbol -> signals
    """
    signals = {}

    for symbol, df in data.items():
        rsi_val = rsi(df["Close"], period=rsi_period)
        ibs_val = ibs(df["High"], df["Low"], df["Close"])

        # Entry condition
        entry = (rsi_val < rsi_entry) & (ibs_val < ibs_entry)

        # Create signal with hold period
        signal = pd.Series(0, index=df.index)
        hold_counter = 0

        for i in range(len(df)):
            if hold_counter > 0:
                signal.iloc[i] = 1
                hold_counter -= 1
            elif entry.iloc[i]:
                signal.iloc[i] = 1
                hold_counter = hold_days

        signals[symbol] = signal

    return signals


def generate_signals_cumrsi(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    cum_period: int = 2,
    cum_entry: float = 35,
    rsi_exit: float = 65,
) -> dict[str, pd.Series]:
    """
    Cumulative RSI mean reversion signals.

    Entry: Cumulative RSI < threshold (deeper oversold)
    Exit: RSI > exit threshold

    Args:
        data: Dict of symbol -> DataFrame
        rsi_period: RSI period
        cum_period: Cumulative period
        cum_entry: Cumulative RSI entry threshold
        rsi_exit: Exit threshold

    Returns:
        Dict of symbol -> signals
    """
    signals = {}

    for symbol, df in data.items():
        rsi_val = rsi(df["Close"], period=rsi_period)
        cum_rsi = cumulative_rsi(df["Close"], rsi_period=rsi_period, cum_period=cum_period)

        signal = pd.Series(0, index=df.index)
        in_position = False

        for i in range(1, len(df)):
            if not in_position:
                if cum_rsi.iloc[i] < cum_entry:
                    signal.iloc[i] = 1
                    in_position = True
            else:
                if rsi_val.iloc[i] > rsi_exit:
                    signal.iloc[i] = 0
                    in_position = False
                else:
                    signal.iloc[i] = 1

        signals[symbol] = signal

    return signals


def generate_signals_adaptive(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    rsi_entry_spy: float = 10,
    rsi_entry_qqq: float = 15,
    rsi_exit: float = 65,
    sma_period: int = 200,
    require_above_sma: bool = True,
) -> dict[str, pd.Series]:
    """
    Adaptive RSI with trend filter.

    Entry: RSI < threshold (different for each asset) AND price > SMA
    Exit: RSI > exit threshold

    Args:
        data: Dict of symbol -> DataFrame
        rsi_period: RSI lookback
        rsi_entry_spy: SPY entry threshold
        rsi_entry_qqq: QQQ entry threshold (usually more volatile)
        rsi_exit: Exit threshold
        sma_period: Trend filter period
        require_above_sma: Only trade above SMA

    Returns:
        Dict of symbol -> signals
    """
    signals = {}

    entry_thresholds = {"SPY": rsi_entry_spy, "QQQ": rsi_entry_qqq}

    for symbol, df in data.items():
        rsi_val = rsi(df["Close"], period=rsi_period)
        sma = df["Close"].rolling(sma_period).mean()
        entry_thresh = entry_thresholds.get(symbol, rsi_entry_spy)

        signal = pd.Series(0, index=df.index)
        in_position = False

        for i in range(1, len(df)):
            if not in_position:
                # Entry conditions
                rsi_ok = rsi_val.iloc[i] < entry_thresh
                trend_ok = df["Close"].iloc[i] > sma.iloc[i] if require_above_sma else True

                if rsi_ok and trend_ok:
                    signal.iloc[i] = 1
                    in_position = True
            else:
                if rsi_val.iloc[i] > rsi_exit:
                    signal.iloc[i] = 0
                    in_position = False
                else:
                    signal.iloc[i] = 1

        signals[symbol] = signal

    return signals


def run_optimization(
    train_data: dict[str, pd.DataFrame],
    verbose: bool = True,
) -> tuple[dict, Backtest]:
    """
    Run optimization to find best parameters.

    Tests multiple hypothesis:
    1. Simple RSI with both assets oversold
    2. RSI + IBS combination
    3. Cumulative RSI
    4. Adaptive RSI with trend filter
    """
    best_result = None
    best_params = None
    best_score = -np.inf

    prices = {s: df["Close"] for s, df in train_data.items()}

    # Hypothesis 1: Simple RSI (both oversold)
    if verbose:
        console.print("[yellow]Testing Hypothesis 1: Simple RSI (both oversold)[/]")

    for rsi_period in [2, 3]:
        for rsi_entry in [5, 10, 15, 20]:
            for rsi_exit in [50, 60, 70, 80]:
                signals = generate_signals_rsi_combined(
                    train_data,
                    rsi_period=rsi_period,
                    rsi_entry=rsi_entry,
                    rsi_exit=rsi_exit,
                    require_both=True,
                )
                bt = run_multi_asset_backtest(prices, signals)

                if bt.sharpe > best_score and bt.num_trades > 50:
                    best_score = bt.sharpe
                    best_params = {
                        "type": "rsi_combined",
                        "rsi_period": rsi_period,
                        "rsi_entry": rsi_entry,
                        "rsi_exit": rsi_exit,
                        "require_both": True,
                    }
                    best_result = bt

    # Hypothesis 1b: Simple RSI (either oversold)
    for rsi_period in [2, 3]:
        for rsi_entry in [5, 10, 15]:
            for rsi_exit in [50, 60, 70]:
                signals = generate_signals_rsi_combined(
                    train_data,
                    rsi_period=rsi_period,
                    rsi_entry=rsi_entry,
                    rsi_exit=rsi_exit,
                    require_both=False,
                )
                bt = run_multi_asset_backtest(prices, signals)

                if bt.sharpe > best_score and bt.num_trades > 50:
                    best_score = bt.sharpe
                    best_params = {
                        "type": "rsi_combined",
                        "rsi_period": rsi_period,
                        "rsi_entry": rsi_entry,
                        "rsi_exit": rsi_exit,
                        "require_both": False,
                    }
                    best_result = bt

    # Hypothesis 2: RSI + IBS
    if verbose:
        console.print("[yellow]Testing Hypothesis 2: RSI + IBS[/]")

    for rsi_period in [2, 3]:
        for rsi_entry in [10, 15, 20, 25]:
            for ibs_entry in [0.15, 0.2, 0.25, 0.3]:
                for hold_days in [1, 2, 3]:
                    signals = generate_signals_rsi_ibs(
                        train_data,
                        rsi_period=rsi_period,
                        rsi_entry=rsi_entry,
                        ibs_entry=ibs_entry,
                        hold_days=hold_days,
                    )
                    bt = run_multi_asset_backtest(prices, signals)

                    if bt.sharpe > best_score and bt.num_trades > 50:
                        best_score = bt.sharpe
                        best_params = {
                            "type": "rsi_ibs",
                            "rsi_period": rsi_period,
                            "rsi_entry": rsi_entry,
                            "ibs_entry": ibs_entry,
                            "hold_days": hold_days,
                        }
                        best_result = bt

    # Hypothesis 3: Cumulative RSI
    if verbose:
        console.print("[yellow]Testing Hypothesis 3: Cumulative RSI[/]")

    for rsi_period in [2, 3]:
        for cum_period in [2, 3]:
            for cum_entry in [25, 30, 35, 40, 45]:
                for rsi_exit in [55, 60, 65, 70]:
                    signals = generate_signals_cumrsi(
                        train_data,
                        rsi_period=rsi_period,
                        cum_period=cum_period,
                        cum_entry=cum_entry,
                        rsi_exit=rsi_exit,
                    )
                    bt = run_multi_asset_backtest(prices, signals)

                    if bt.sharpe > best_score and bt.num_trades > 50:
                        best_score = bt.sharpe
                        best_params = {
                            "type": "cumrsi",
                            "rsi_period": rsi_period,
                            "cum_period": cum_period,
                            "cum_entry": cum_entry,
                            "rsi_exit": rsi_exit,
                        }
                        best_result = bt

    # Hypothesis 4: Adaptive RSI with trend filter
    if verbose:
        console.print("[yellow]Testing Hypothesis 4: Adaptive RSI with trend filter[/]")

    for rsi_period in [2, 3]:
        for rsi_entry in [5, 10, 15]:
            for rsi_exit in [55, 60, 65, 70]:
                for sma_period in [200]:
                    for above_sma in [True, False]:
                        signals = generate_signals_adaptive(
                            train_data,
                            rsi_period=rsi_period,
                            rsi_entry_spy=rsi_entry,
                            rsi_entry_qqq=rsi_entry + 5,  # QQQ more volatile
                            rsi_exit=rsi_exit,
                            sma_period=sma_period,
                            require_above_sma=above_sma,
                        )
                        bt = run_multi_asset_backtest(prices, signals)

                        if bt.sharpe > best_score and bt.num_trades > 50:
                            best_score = bt.sharpe
                            best_params = {
                                "type": "adaptive",
                                "rsi_period": rsi_period,
                                "rsi_entry_spy": rsi_entry,
                                "rsi_entry_qqq": rsi_entry + 5,
                                "rsi_exit": rsi_exit,
                                "sma_period": sma_period,
                                "require_above_sma": above_sma,
                            }
                            best_result = bt

    if verbose:
        console.print(f"[green]Best parameters:[/] {best_params}")
        console.print(f"[green]Best Sharpe:[/] {best_score:.3f}")

    return best_params, best_result


def validate_on_test(
    test_data: dict[str, pd.DataFrame],
    params: dict,
) -> Backtest:
    """Run strategy with given parameters on test data."""
    prices = {s: df["Close"] for s, df in test_data.items()}

    if params["type"] == "rsi_combined":
        signals = generate_signals_rsi_combined(
            test_data,
            rsi_period=params["rsi_period"],
            rsi_entry=params["rsi_entry"],
            rsi_exit=params["rsi_exit"],
            require_both=params["require_both"],
        )
    elif params["type"] == "rsi_ibs":
        signals = generate_signals_rsi_ibs(
            test_data,
            rsi_period=params["rsi_period"],
            rsi_entry=params["rsi_entry"],
            ibs_entry=params["ibs_entry"],
            hold_days=params["hold_days"],
        )
    elif params["type"] == "cumrsi":
        signals = generate_signals_cumrsi(
            test_data,
            rsi_period=params["rsi_period"],
            cum_period=params["cum_period"],
            cum_entry=params["cum_entry"],
            rsi_exit=params["rsi_exit"],
        )
    elif params["type"] == "adaptive":
        signals = generate_signals_adaptive(
            test_data,
            rsi_period=params["rsi_period"],
            rsi_entry_spy=params["rsi_entry_spy"],
            rsi_entry_qqq=params["rsi_entry_qqq"],
            rsi_exit=params["rsi_exit"],
            sma_period=params["sma_period"],
            require_above_sma=params["require_above_sma"],
        )
    else:
        raise ValueError(f"Unknown signal type: {params['type']}")

    return run_multi_asset_backtest(prices, signals)


def run_full_period(
    full_data: dict[str, pd.DataFrame],
    params: dict,
) -> Backtest:
    """Run strategy on full data period for final equity curve."""
    prices = {s: df["Close"] for s, df in full_data.items()}

    if params["type"] == "rsi_combined":
        signals = generate_signals_rsi_combined(
            full_data,
            rsi_period=params["rsi_period"],
            rsi_entry=params["rsi_entry"],
            rsi_exit=params["rsi_exit"],
            require_both=params["require_both"],
        )
    elif params["type"] == "rsi_ibs":
        signals = generate_signals_rsi_ibs(
            full_data,
            rsi_period=params["rsi_period"],
            rsi_entry=params["rsi_entry"],
            ibs_entry=params["ibs_entry"],
            hold_days=params["hold_days"],
        )
    elif params["type"] == "cumrsi":
        signals = generate_signals_cumrsi(
            full_data,
            rsi_period=params["rsi_period"],
            cum_period=params["cum_period"],
            cum_entry=params["cum_entry"],
            rsi_exit=params["rsi_exit"],
        )
    elif params["type"] == "adaptive":
        signals = generate_signals_adaptive(
            full_data,
            rsi_period=params["rsi_period"],
            rsi_entry_spy=params["rsi_entry_spy"],
            rsi_entry_qqq=params["rsi_entry_qqq"],
            rsi_exit=params["rsi_exit"],
            sma_period=params["sma_period"],
            require_above_sma=params["require_above_sma"],
        )
    else:
        raise ValueError(f"Unknown signal type: {params['type']}")

    return run_multi_asset_backtest(prices, signals)


def main():
    """Main entry point."""
    console.print("[bold blue]ETFMR Strategy - Reverse Engineering[/]")
    console.print(f"Target: CAGR {TARGET['CAGR']:.1%}, MDD {TARGET['MDD']:.1%}, Sharpe {TARGET['Sharpe']:.2f}")
    console.print("-" * 60)

    # Load data
    console.print("\n[cyan]Loading data...[/]")
    spy = load_data("SPY")
    qqq = load_data("QQQ")
    full_data = {"SPY": spy, "QQQ": qqq}

    # Split into train/test
    train_spy, test_spy = split_train_test(spy)
    train_qqq, test_qqq = split_train_test(qqq)
    train_data = {"SPY": train_spy, "QQQ": train_qqq}
    test_data = {"SPY": test_spy, "QQQ": test_qqq}

    console.print(f"Training period: {train_spy.index.min().date()} to {train_spy.index.max().date()}")
    console.print(f"Test period: {test_spy.index.min().date()} to {test_spy.index.max().date()}")

    # Run optimization on training data
    console.print("\n[cyan]Running optimization on training data...[/]")
    best_params, train_bt = run_optimization(train_data)

    # Display training results
    console.print("\n[bold]Training Period Results:[/]")
    table = Table(title="Training Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Achieved", style="green")
    table.add_column("Target", style="yellow")
    table.add_column("Delta", style="white")

    metrics = train_bt.summary()
    table.add_row("CAGR", f"{metrics['CAGR']:.2%}", f"{TARGET['CAGR']:.2%}",
                  f"{(metrics['CAGR'] - TARGET['CAGR']):.2%}")
    table.add_row("Max DD", f"{metrics['Max DD']:.2%}", f"{TARGET['MDD']:.2%}",
                  f"{(metrics['Max DD'] - TARGET['MDD']):.2%}")
    table.add_row("Sharpe", f"{metrics['Sharpe']:.2f}", f"{TARGET['Sharpe']:.2f}",
                  f"{(metrics['Sharpe'] - TARGET['Sharpe']):.2f}")
    table.add_row("Trades", f"{metrics['Trades']}", "-", "-")
    table.add_row("Win Rate", f"{metrics['Win Rate']:.1%}", "-", "-")
    console.print(table)

    # Validate on test data
    console.print("\n[cyan]Validating on test data...[/]")
    test_bt = validate_on_test(test_data, best_params)

    console.print("\n[bold]Test Period Results:[/]")
    table = Table(title="Test Metrics (Out-of-Sample)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    test_metrics = test_bt.summary()
    table.add_row("CAGR", f"{test_metrics['CAGR']:.2%}")
    table.add_row("Max DD", f"{test_metrics['Max DD']:.2%}")
    table.add_row("Sharpe", f"{test_metrics['Sharpe']:.2f}")
    table.add_row("Trades", f"{test_metrics['Trades']}")
    table.add_row("Win Rate", f"{test_metrics['Win Rate']:.1%}")
    console.print(table)

    # Run on full period for final results
    console.print("\n[cyan]Running on full period...[/]")
    full_bt = run_full_period(full_data, best_params)

    console.print("\n[bold]Full Period Results:[/]")
    full_metrics = full_bt.summary()
    console.print(f"CAGR: {full_metrics['CAGR']:.2%} (target: {TARGET['CAGR']:.2%})")
    console.print(f"Max DD: {full_metrics['Max DD']:.2%} (target: {TARGET['MDD']:.2%})")
    console.print(f"Sharpe: {full_metrics['Sharpe']:.2f} (target: {TARGET['Sharpe']:.2f})")

    # Check if within tolerance
    cagr_ok = abs(full_metrics["CAGR"] - TARGET["CAGR"]) <= 0.03
    mdd_ok = abs(full_metrics["Max DD"] - TARGET["MDD"]) <= 0.10
    sharpe_ok = abs(full_metrics["Sharpe"] - TARGET["Sharpe"]) <= 0.15

    if cagr_ok and mdd_ok and sharpe_ok:
        console.print("\n[bold green]✓ Strategy successfully reverse-engineered![/]")
    else:
        console.print("\n[bold yellow]Strategy partially matched target metrics[/]")
        if not cagr_ok:
            console.print(f"  - CAGR outside tolerance (±3%)")
        if not mdd_ok:
            console.print(f"  - Max DD outside tolerance (±10%)")
        if not sharpe_ok:
            console.print(f"  - Sharpe outside tolerance (±0.15)")

    # Save results
    output_dir = PROJECT_ROOT / "output" / "claude-reverse-engineer-strategies-ziZsY" / "pal-reverse-engineering"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot equity curve
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(full_bt.equity.index, full_bt.equity.values, label="ETFMR Strategy", color="blue")

    # Add buy & hold for comparison
    spy_bh = (spy["Close"] / spy["Close"].iloc[0])
    qqq_bh = (qqq["Close"] / qqq["Close"].iloc[0])
    combined_bh = (spy_bh + qqq_bh) / 2

    ax.plot(combined_bh.index, combined_bh.values, label="SPY+QQQ Buy & Hold", color="gray", alpha=0.5)

    ax.set_title(f"ETFMR Strategy Equity Curve\nCAGR: {full_metrics['CAGR']:.1%}, Sharpe: {full_metrics['Sharpe']:.2f}, MaxDD: {full_metrics['Max DD']:.1%}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($1 initial)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    chart_path = output_dir / f"etfmr_equity_{TIMESTAMP}.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    console.print(f"\n[green]Saved equity curve:[/] {chart_path}")

    # Save report
    report = f"""# ETFMR Strategy - Reverse Engineered

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

### Training Period (2005-2020)

| Metric | Value |
|--------|-------|
| CAGR | {metrics['CAGR']:.2%} |
| Max DD | {metrics['Max DD']:.2%} |
| Sharpe | {metrics['Sharpe']:.2f} |
| Trades | {metrics['Trades']} |

### Test Period (2021-2025)

| Metric | Value |
|--------|-------|
| CAGR | {test_metrics['CAGR']:.2%} |
| Max DD | {test_metrics['Max DD']:.2%} |
| Sharpe | {test_metrics['Sharpe']:.2f} |
| Trades | {test_metrics['Trades']} |

## Strategy Rules

**Type:** {best_params['type']}

**Parameters:**
```python
{best_params}
```

## Description

"""
    if best_params["type"] == "rsi_combined":
        report += f"""
- **Entry:** RSI-{best_params['rsi_period']} < {best_params['rsi_entry']} on {'both' if best_params['require_both'] else 'either'} SPY and QQQ
- **Exit:** RSI-{best_params['rsi_period']} > {best_params['rsi_exit']}
- **Position:** Equal weight in both assets when in trade
"""
    elif best_params["type"] == "rsi_ibs":
        report += f"""
- **Entry:** RSI-{best_params['rsi_period']} < {best_params['rsi_entry']} AND IBS < {best_params['ibs_entry']}
- **Exit:** After {best_params['hold_days']} days
- **Position:** Equal weight in both assets when in trade
"""
    elif best_params["type"] == "cumrsi":
        report += f"""
- **Entry:** Cumulative RSI-{best_params['rsi_period']} (sum over {best_params['cum_period']} days) < {best_params['cum_entry']}
- **Exit:** RSI-{best_params['rsi_period']} > {best_params['rsi_exit']}
- **Position:** Equal weight in both assets when in trade
"""
    elif best_params["type"] == "adaptive":
        report += f"""
- **Entry SPY:** RSI-{best_params['rsi_period']} < {best_params['rsi_entry_spy']}{' AND price > SMA-' + str(best_params['sma_period']) if best_params['require_above_sma'] else ''}
- **Entry QQQ:** RSI-{best_params['rsi_period']} < {best_params['rsi_entry_qqq']}{' AND price > SMA-' + str(best_params['sma_period']) if best_params['require_above_sma'] else ''}
- **Exit:** RSI-{best_params['rsi_period']} > {best_params['rsi_exit']}
- **Position:** Equal weight in both assets when in trade
"""

    report += f"""
## Equity Curve

![ETFMR Equity Curve](etfmr_equity_{TIMESTAMP}.png)

## Validation Status

**Overall:** {'✓ Successfully reverse-engineered' if (cagr_ok and mdd_ok and sharpe_ok) else '⚠ Partially matched'}

- CAGR: {'✓ Within ±3%' if cagr_ok else '✗ Outside tolerance'}
- Max DD: {'✓ Within ±10%' if mdd_ok else '✗ Outside tolerance'}
- Sharpe: {'✓ Within ±0.15' if sharpe_ok else '✗ Outside tolerance'}
"""

    report_path = output_dir / f"etfmr_report_{TIMESTAMP}.md"
    report_path.write_text(report)
    console.print(f"[green]Saved report:[/] {report_path}")

    return best_params, full_metrics


if __name__ == "__main__":
    params, metrics = main()
