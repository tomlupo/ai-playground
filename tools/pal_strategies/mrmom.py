# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "numpy>=1.24", "matplotlib>=3.8", "rich>=13.0"]
# ///
"""
MRMOM Strategy - Reverse Engineered from Price Action Lab

Target Metrics (PAL):
- CAGR: 10.3%
- Max Drawdown: -16.8%
- Sharpe Ratio: 1.15

Strategy Type: Regime switching on SPY, QQQ, TLT, GLD (long-only)

Hypothesis: Uses Momersion indicator to detect market regime.
- Momersion > 50: Use momentum rules (buy on strength)
- Momersion < 50: Use mean-reversion rules (buy on weakness)
Allocate across assets based on regime.

This is the BEST risk-adjusted strategy in the PAL portfolio (Sharpe 1.15).
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

from shared.indicators import momersion, regime_momersion, rsi
from tools.pal_strategies.base import (
    Backtest,
    load_data,
    run_backtest,
    run_multi_asset_backtest,
    split_train_test,
)

console = Console()

TARGET = {
    "CAGR": 0.103,
    "MDD": -0.168,
    "Sharpe": 1.15,
}

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_signals_momersion_regime(
    data: dict[str, pd.DataFrame],
    mom_lookback: int = 252,
    mom_threshold: float = 50,
    rsi_period: int = 2,
    rsi_mr_entry: float = 15,
    rsi_mom_entry: float = 70,
    rsi_exit: float = 50,
) -> dict[str, pd.Series]:
    """
    Momersion-based regime switching.

    In MR regime (Momersion < threshold): Buy when RSI < rsi_mr_entry
    In Mom regime (Momersion > threshold): Buy when RSI > rsi_mom_entry
    """
    signals = {}

    # Calculate momersion on SPY as regime indicator
    spy_returns = data["SPY"]["Close"].pct_change()
    mom_indicator = momersion(spy_returns, lookback=mom_lookback)

    for symbol in data.keys():
        df = data[symbol]
        rsi_val = rsi(df["Close"], period=rsi_period)

        signal = pd.Series(0, index=df.index)
        in_position = False

        for i in range(1, len(df)):
            date = df.index[i]

            # Get regime
            mom_val = mom_indicator.loc[date] if date in mom_indicator.index else 50
            is_momentum = mom_val > mom_threshold

            if not in_position:
                # Entry based on regime
                if is_momentum:
                    # Momentum regime: buy on strength
                    if rsi_val.iloc[i] > rsi_mom_entry:
                        signal.iloc[i] = 1
                        in_position = True
                else:
                    # MR regime: buy on weakness
                    if rsi_val.iloc[i] < rsi_mr_entry:
                        signal.iloc[i] = 1
                        in_position = True
            else:
                # Exit when RSI crosses 50 (neutral)
                if (is_momentum and rsi_val.iloc[i] < rsi_exit) or \
                   (not is_momentum and rsi_val.iloc[i] > rsi_exit):
                    signal.iloc[i] = 0
                    in_position = False
                else:
                    signal.iloc[i] = 1

        signals[symbol] = signal

    return signals


def generate_signals_asset_rotation(
    data: dict[str, pd.DataFrame],
    mom_lookback: int = 252,
    mom_threshold: float = 50,
    roc_period: int = 20,
) -> dict[str, pd.Series]:
    """
    Rotate between assets based on regime and momentum.

    In MR regime: Favor TLT and GLD (defensive)
    In Mom regime: Favor SPY and QQQ (risk-on)
    Within category: Pick asset with best momentum
    """
    signals = {s: pd.Series(0.0, index=data[s].index) for s in data.keys()}

    # Calculate indicators
    spy_returns = data["SPY"]["Close"].pct_change()
    mom_indicator = momersion(spy_returns, lookback=mom_lookback)

    # Rate of change for momentum ranking
    roc = {s: data[s]["Close"].pct_change(roc_period) for s in data.keys()}

    risk_on_assets = ["SPY", "QQQ"]
    risk_off_assets = ["TLT", "GLD"]

    common_dates = data["SPY"].index

    for date in common_dates:
        if date not in mom_indicator.index:
            continue

        mom_val = mom_indicator.loc[date]
        is_momentum = mom_val > mom_threshold

        if is_momentum:
            # Risk-on: choose between SPY and QQQ
            assets = risk_on_assets
        else:
            # Risk-off: choose between TLT and GLD
            assets = risk_off_assets

        # Pick asset with best momentum
        roc_values = {s: roc[s].loc[date] for s in assets if date in roc[s].index and pd.notna(roc[s].loc[date])}

        if roc_values:
            best_asset = max(roc_values, key=roc_values.get)
            signals[best_asset].loc[date] = 1.0

    return signals


def generate_signals_dual_regime(
    data: dict[str, pd.DataFrame],
    mom_lookback: int = 252,
    mom_threshold: float = 50,
    rsi_period: int = 3,
    rsi_entry: float = 20,
    rsi_exit: float = 60,
    hold_days: int = 3,
) -> dict[str, pd.Series]:
    """
    Dual regime with fixed holding period.

    In MR regime: Buy all assets on RSI < entry
    In Mom regime: Don't trade (or stay in)
    Fixed holding period for exits.
    """
    signals = {}

    spy_returns = data["SPY"]["Close"].pct_change()
    mom_indicator = momersion(spy_returns, lookback=mom_lookback)

    for symbol in data.keys():
        df = data[symbol]
        rsi_val = rsi(df["Close"], period=rsi_period)

        signal = pd.Series(0, index=df.index)
        hold_counter = 0

        for i in range(1, len(df)):
            date = df.index[i]

            mom_val = mom_indicator.loc[date] if date in mom_indicator.index else 50
            is_mr_regime = mom_val < mom_threshold

            if hold_counter > 0:
                signal.iloc[i] = 1
                hold_counter -= 1
            elif is_mr_regime and rsi_val.iloc[i] < rsi_entry:
                signal.iloc[i] = 1
                hold_counter = hold_days

        signals[symbol] = signal

    return signals


def generate_signals_weighted_regime(
    data: dict[str, pd.DataFrame],
    mom_lookback: int = 252,
    rsi_period: int = 2,
    rsi_entry: float = 20,
    rsi_exit: float = 55,
) -> dict[str, pd.Series]:
    """
    Weight allocation based on momersion value.

    Higher Momersion = more weight to momentum signals
    Lower Momersion = more weight to MR signals
    """
    signals = {}

    spy_returns = data["SPY"]["Close"].pct_change()
    mom_indicator = momersion(spy_returns, lookback=mom_lookback)

    equity_assets = ["SPY", "QQQ"]
    defensive_assets = ["TLT", "GLD"]

    for symbol in data.keys():
        df = data[symbol]
        rsi_val = rsi(df["Close"], period=rsi_period)

        signal = pd.Series(0.0, index=df.index)
        in_position = False

        for i in range(1, len(df)):
            date = df.index[i]

            mom_val = mom_indicator.loc[date] if date in mom_indicator.index else 50
            mr_weight = (100 - mom_val) / 100  # Higher when Momersion is low

            if not in_position:
                # MR entry with regime-adjusted threshold
                adjusted_entry = rsi_entry + (1 - mr_weight) * 10

                if rsi_val.iloc[i] < adjusted_entry:
                    # Weight based on regime and asset type
                    if symbol in defensive_assets:
                        weight = mr_weight  # More defensive when MR
                    else:
                        weight = 1 - mr_weight  # More equity when momentum

                    signal.iloc[i] = weight
                    in_position = True
            else:
                if rsi_val.iloc[i] > rsi_exit:
                    signal.iloc[i] = 0
                    in_position = False
                else:
                    signal.iloc[i] = signal.iloc[i-1]

        signals[symbol] = signal

    return signals


def generate_signals_regime_filter(
    data: dict[str, pd.DataFrame],
    mom_lookback: int = 126,
    mom_threshold: float = 45,
    rsi_period: int = 2,
    rsi_entry: float = 15,
    rsi_exit: float = 60,
    equity_weight: float = 0.6,
    defensive_weight: float = 0.4,
) -> dict[str, pd.Series]:
    """
    Simple regime filter with fixed weights.

    In MR regime: Use MR signals on all assets
    In Mom regime: Stay in defensive assets only
    """
    signals = {}

    spy_returns = data["SPY"]["Close"].pct_change()
    mom_indicator = momersion(spy_returns, lookback=mom_lookback)

    equity_assets = ["SPY", "QQQ"]
    defensive_assets = ["TLT", "GLD"]

    for symbol in data.keys():
        df = data[symbol]
        rsi_val = rsi(df["Close"], period=rsi_period)

        signal = pd.Series(0, index=df.index)
        in_position = False

        for i in range(1, len(df)):
            date = df.index[i]

            mom_val = mom_indicator.loc[date] if date in mom_indicator.index else 50
            is_mr_regime = mom_val < mom_threshold

            is_equity = symbol in equity_assets

            if not in_position:
                # In MR regime: trade all assets
                # In Mom regime: only trade defensive
                can_trade = is_mr_regime or not is_equity

                if can_trade and rsi_val.iloc[i] < rsi_entry:
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


def run_optimization(train_data: dict[str, pd.DataFrame], verbose: bool = True) -> tuple[dict, Backtest]:
    """Run optimization to find best parameters."""
    best_result = None
    best_params = None
    best_score = -np.inf

    prices = {s: df["Close"] for s, df in train_data.items()}

    # Hypothesis 1: Momersion regime switching
    if verbose:
        console.print("[yellow]Testing Hypothesis 1: Momersion regime switching[/]")

    for mom_lookback in [126, 252]:
        for mom_threshold in [45, 50, 55]:
            for rsi_period in [2, 3]:
                for rsi_mr_entry in [10, 15, 20]:
                    for rsi_exit in [50, 55, 60]:
                        signals = generate_signals_momersion_regime(
                            train_data, mom_lookback, mom_threshold,
                            rsi_period, rsi_mr_entry, 70, rsi_exit
                        )
                        bt = run_multi_asset_backtest(prices, signals)

                        if bt.sharpe > best_score and bt.num_trades > 30:
                            best_score = bt.sharpe
                            best_params = {
                                "type": "momersion_regime",
                                "mom_lookback": mom_lookback,
                                "mom_threshold": mom_threshold,
                                "rsi_period": rsi_period,
                                "rsi_mr_entry": rsi_mr_entry,
                                "rsi_mom_entry": 70,
                                "rsi_exit": rsi_exit,
                            }
                            best_result = bt

    # Hypothesis 2: Asset rotation
    if verbose:
        console.print("[yellow]Testing Hypothesis 2: Asset rotation[/]")

    for mom_lookback in [126, 252]:
        for mom_threshold in [45, 50, 55]:
            for roc_period in [10, 20, 50]:
                signals = generate_signals_asset_rotation(
                    train_data, mom_lookback, mom_threshold, roc_period
                )
                bt = run_multi_asset_backtest(prices, signals)

                if bt.sharpe > best_score and bt.num_trades > 30:
                    best_score = bt.sharpe
                    best_params = {
                        "type": "asset_rotation",
                        "mom_lookback": mom_lookback,
                        "mom_threshold": mom_threshold,
                        "roc_period": roc_period,
                    }
                    best_result = bt

    # Hypothesis 3: Dual regime with hold
    if verbose:
        console.print("[yellow]Testing Hypothesis 3: Dual regime with fixed holding[/]")

    for mom_lookback in [126, 252]:
        for mom_threshold in [45, 50, 55]:
            for rsi_period in [2, 3]:
                for rsi_entry in [15, 20, 25]:
                    for hold_days in [2, 3, 4, 5]:
                        signals = generate_signals_dual_regime(
                            train_data, mom_lookback, mom_threshold,
                            rsi_period, rsi_entry, 60, hold_days
                        )
                        bt = run_multi_asset_backtest(prices, signals)

                        if bt.sharpe > best_score and bt.num_trades > 30:
                            best_score = bt.sharpe
                            best_params = {
                                "type": "dual_regime",
                                "mom_lookback": mom_lookback,
                                "mom_threshold": mom_threshold,
                                "rsi_period": rsi_period,
                                "rsi_entry": rsi_entry,
                                "rsi_exit": 60,
                                "hold_days": hold_days,
                            }
                            best_result = bt

    # Hypothesis 4: Weighted regime
    if verbose:
        console.print("[yellow]Testing Hypothesis 4: Weighted regime allocation[/]")

    for mom_lookback in [126, 252]:
        for rsi_period in [2, 3]:
            for rsi_entry in [15, 20, 25]:
                for rsi_exit in [50, 55, 60]:
                    signals = generate_signals_weighted_regime(
                        train_data, mom_lookback, rsi_period, rsi_entry, rsi_exit
                    )
                    bt = run_multi_asset_backtest(prices, signals)

                    if bt.sharpe > best_score and bt.num_trades > 30:
                        best_score = bt.sharpe
                        best_params = {
                            "type": "weighted_regime",
                            "mom_lookback": mom_lookback,
                            "rsi_period": rsi_period,
                            "rsi_entry": rsi_entry,
                            "rsi_exit": rsi_exit,
                        }
                        best_result = bt

    # Hypothesis 5: Regime filter
    if verbose:
        console.print("[yellow]Testing Hypothesis 5: Regime filter[/]")

    for mom_lookback in [126, 252]:
        for mom_threshold in [40, 45, 50]:
            for rsi_period in [2, 3]:
                for rsi_entry in [10, 15, 20]:
                    for rsi_exit in [55, 60, 65]:
                        signals = generate_signals_regime_filter(
                            train_data, mom_lookback, mom_threshold,
                            rsi_period, rsi_entry, rsi_exit
                        )
                        bt = run_multi_asset_backtest(prices, signals)

                        if bt.sharpe > best_score and bt.num_trades > 30:
                            best_score = bt.sharpe
                            best_params = {
                                "type": "regime_filter",
                                "mom_lookback": mom_lookback,
                                "mom_threshold": mom_threshold,
                                "rsi_period": rsi_period,
                                "rsi_entry": rsi_entry,
                                "rsi_exit": rsi_exit,
                            }
                            best_result = bt

    if verbose:
        console.print(f"[green]Best parameters:[/] {best_params}")
        console.print(f"[green]Best Sharpe:[/] {best_score:.3f}")

    return best_params, best_result


def apply_strategy(data: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.Series]:
    """Apply strategy with given parameters."""
    if params["type"] == "momersion_regime":
        return generate_signals_momersion_regime(
            data, params["mom_lookback"], params["mom_threshold"],
            params["rsi_period"], params["rsi_mr_entry"],
            params["rsi_mom_entry"], params["rsi_exit"]
        )
    elif params["type"] == "asset_rotation":
        return generate_signals_asset_rotation(
            data, params["mom_lookback"], params["mom_threshold"], params["roc_period"]
        )
    elif params["type"] == "dual_regime":
        return generate_signals_dual_regime(
            data, params["mom_lookback"], params["mom_threshold"],
            params["rsi_period"], params["rsi_entry"],
            params["rsi_exit"], params["hold_days"]
        )
    elif params["type"] == "weighted_regime":
        return generate_signals_weighted_regime(
            data, params["mom_lookback"], params["rsi_period"],
            params["rsi_entry"], params["rsi_exit"]
        )
    elif params["type"] == "regime_filter":
        return generate_signals_regime_filter(
            data, params["mom_lookback"], params["mom_threshold"],
            params["rsi_period"], params["rsi_entry"], params["rsi_exit"]
        )
    else:
        raise ValueError(f"Unknown strategy type: {params['type']}")


def main():
    """Main entry point."""
    console.print("[bold blue]MRMOM Strategy - Reverse Engineering[/]")
    console.print(f"Target: CAGR {TARGET['CAGR']:.1%}, MDD {TARGET['MDD']:.1%}, Sharpe {TARGET['Sharpe']:.2f}")
    console.print("-" * 60)

    # Load data
    console.print("\n[cyan]Loading data...[/]")
    spy = load_data("SPY")
    qqq = load_data("QQQ")
    tlt = load_data("TLT")
    gld = load_data("GLD")
    full_data = {"SPY": spy, "QQQ": qqq, "TLT": tlt, "GLD": gld}

    # Split
    train_data = {s: split_train_test(df)[0] for s, df in full_data.items()}
    test_data = {s: split_train_test(df)[1] for s, df in full_data.items()}

    console.print(f"Training period: {train_data['SPY'].index.min().date()} to {train_data['SPY'].index.max().date()}")
    console.print(f"Test period: {test_data['SPY'].index.min().date()} to {test_data['SPY'].index.max().date()}")

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
    ax.plot(full_bt.equity.index, full_bt.equity.values, label="MRMOM Strategy", color="blue")

    # Buy & hold comparison
    bh = (spy["Close"]/spy["Close"].iloc[0] + qqq["Close"]/qqq["Close"].iloc[0] +
          tlt["Close"]/tlt["Close"].iloc[0] + gld["Close"]/gld["Close"].iloc[0]) / 4
    ax.plot(bh.index, bh.values, label="Equal Weight B&H", color="gray", alpha=0.5)

    ax.set_title(f"MRMOM Strategy Equity Curve\nCAGR: {full_metrics['CAGR']:.1%}, Sharpe: {full_metrics['Sharpe']:.2f}, MaxDD: {full_metrics['Max DD']:.1%}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($1 initial)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    chart_path = output_dir / f"mrmom_equity_{TIMESTAMP}.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    console.print(f"\n[green]Saved equity curve:[/] {chart_path}")

    # Report
    report = f"""# MRMOM Strategy - Reverse Engineered

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

    report_path = output_dir / f"mrmom_report_{TIMESTAMP}.md"
    report_path.write_text(report)
    console.print(f"[green]Saved report:[/] {report_path}")

    return best_params, full_metrics


if __name__ == "__main__":
    params, metrics = main()
