# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "yfinance>=0.2.36",
#     "rich>=13.0",
#     "matplotlib>=3.7",
# ]
# ///
"""
Quick optimization with narrower parameter ranges based on research.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")
console = Console()

# Targets
TARGETS = {
    "MRETF": {"cagr": 5.2, "mdd": -9.3, "sharpe": 0.78, "beta": 0.14, "start": 2002},
    "MRMOM": {"cagr": 10.3, "mdd": -16.8, "sharpe": 1.15, "beta": 0.32, "start": 2003},
    "B2S2ETF": {"cagr": 9.1, "mdd": -30.6, "sharpe": 0.63, "beta": 0.60, "start": 2002},
    "ETFMR": {"cagr": 10.0, "mdd": -22.9, "sharpe": 0.82, "beta": 0.44, "start": 2003},
}


@dataclass
class Result:
    name: str
    params: dict
    cagr: float
    mdd: float
    sharpe: float
    beta: float
    trades: int
    win_rate: float
    exposure: float
    returns: pd.Series


def download_data(symbols: list[str], start: str = "2000-01-01") -> dict:
    """Download OHLCV data."""
    console.print(f"[cyan]Downloading {symbols}...[/cyan]")
    data = {}
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start, progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                data[symbol] = df
        except Exception:
            pass
    console.print(f"[green]Downloaded {len(data)} symbols[/green]")
    return data


def rsi(prices: pd.Series, period: int = 2) -> pd.Series:
    """RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def ibs(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Internal Bar Strength."""
    return (close - low) / (high - low + 1e-10)


def momersion(returns: pd.Series, lookback: int = 252) -> pd.Series:
    """Momersion indicator."""
    def calc(window):
        mom = mr = 0
        for i in range(1, len(window)):
            if window.iloc[i] * window.iloc[i-1] > 0:
                mom += 1
            elif window.iloc[i] * window.iloc[i-1] < 0:
                mr += 1
        total = mom + mr
        return 100 * mom / total if total > 0 else 50
    return returns.rolling(lookback).apply(calc, raw=False)


def calc_metrics(returns: pd.Series, benchmark: pd.Series, rf: float = 0.02) -> dict:
    """Calculate performance metrics."""
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if len(aligned) < 30:
        return {"cagr": 0, "mdd": 0, "sharpe": 0, "beta": 0}

    strat = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]

    total = (1 + strat).prod() - 1
    years = len(strat) / 252
    cagr = ((1 + total) ** (1 / years) - 1) * 100 if years > 0 else 0

    cum = (1 + strat).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100

    vol = strat.std() * np.sqrt(252)
    sharpe = (cagr / 100 - rf) / vol if vol > 0 else 0

    cov = strat.cov(bench)
    var = bench.var()
    beta = cov / var if var > 0 else 0

    return {"cagr": cagr, "mdd": mdd, "sharpe": sharpe, "beta": beta}


# ==============================================================================
# STRATEGIES - Best configurations found
# ==============================================================================

def strategy_mretf_ibs(data: dict, entry: float = 0.1, exit_: float = 0.9, max_hold: int = 15) -> pd.Series:
    """
    MRETF approximation using IBS.
    Target: CAGR 5.2%, Sharpe 0.78, Beta 0.14

    Key insight: Very low beta (0.14) suggests:
    - Very selective entries (low exposure)
    - Strict trend filter
    - Quick exits
    """
    df = data.get("SPY")
    if df is None:
        return pd.Series(dtype=float)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    ibs_val = ibs(high, low, close)
    sma_200 = close.rolling(200).mean()

    signals = pd.Series(0, index=close.index)
    position = 0
    entry_idx = 0

    for i in range(201, len(close)):
        trend_ok = close.iloc[i] > sma_200.iloc[i]

        if position == 0 and trend_ok:
            if ibs_val.iloc[i] < entry:
                position = 1
                entry_idx = i
        elif position == 1:
            days_held = i - entry_idx
            if ibs_val.iloc[i] > exit_ or days_held >= max_hold:
                position = 0

        signals.iloc[i] = position

    return signals


def strategy_etfmr_combo(data: dict, rsi_thresh: float = 25, ibs_thresh: float = 0.25) -> pd.Series:
    """
    ETFMR approximation - "based on two popular indicators"
    Target: CAGR 10.0%, Sharpe 0.82, Beta 0.44

    Uses RSI + IBS together for confirmation.
    Higher beta (0.44) means more exposure, no trend filter.
    """
    df = data.get("SPY")
    if df is None:
        return pd.Series(dtype=float)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    rsi_val = rsi(close, 2)
    ibs_val = ibs(high, low, close)
    sma_5 = close.rolling(5).mean()

    signals = pd.Series(0, index=close.index)
    position = 0

    for i in range(10, len(close)):
        if position == 0:
            # Enter when BOTH indicators are oversold
            if rsi_val.iloc[i] < rsi_thresh and ibs_val.iloc[i] < ibs_thresh:
                position = 1
        elif position == 1:
            # Exit on 5-day MA crossover
            if close.iloc[i] > sma_5.iloc[i]:
                position = 0

        signals.iloc[i] = position

    return signals


def strategy_mrmom(data: dict, mom_period: int = 252, rsi_entry: float = 15) -> pd.Series:
    """
    MRMOM approximation - regime switching
    Target: CAGR 10.3%, Sharpe 1.15, Beta 0.32

    Uses Momersion for regime detection:
    - < 50: Mean-reversion (buy RSI dips)
    - >= 50: Momentum (follow trends)
    """
    symbols = ["SPY", "QQQ", "TLT", "GLD"]
    available = [s for s in symbols if s in data]

    all_signals = []

    for symbol in available:
        df = data[symbol]
        close = df["Close"]
        returns = close.pct_change()

        mom = momersion(returns, mom_period)
        rsi_val = rsi(close, 2)
        momentum = close / close.shift(20) - 1
        sma_5 = close.rolling(5).mean()
        sma_200 = close.rolling(200).mean()

        signals = pd.Series(0.0, index=close.index)
        position = 0

        for i in range(mom_period + 1, len(close)):
            regime = mom.iloc[i]
            trend_ok = close.iloc[i] > sma_200.iloc[i]

            if pd.isna(regime):
                signals.iloc[i] = position
                continue

            # Mean-reversion regime
            if regime < 50:
                if position == 0 and trend_ok and rsi_val.iloc[i] < rsi_entry:
                    position = 1
                elif position == 1 and close.iloc[i] > sma_5.iloc[i]:
                    position = 0
            # Momentum regime
            else:
                if position == 0 and momentum.iloc[i] > 0.02:  # 2% momentum threshold
                    position = 1
                elif position == 1 and momentum.iloc[i] < -0.02:
                    position = 0

            signals.iloc[i] = position

        all_signals.append(signals)

    if not all_signals:
        return pd.Series(dtype=float)

    return pd.concat(all_signals, axis=1).mean(axis=1)


def strategy_b2s2(data: dict, down_days: int = 2, up_days: int = 1) -> pd.Series:
    """
    B2S2ETF approximation
    Target: CAGR 9.1%, Sharpe 0.63, Beta 0.60

    High beta (0.60) means high exposure - no trend filter.
    "No parameters, no filters" - keep it simple.
    """
    df = data.get("SPY")
    if df is None:
        return pd.Series(dtype=float)

    close = df["Close"]
    returns = close.pct_change()

    # Count consecutive days
    up_count = pd.Series(0, index=close.index)
    down_count = pd.Series(0, index=close.index)

    for i in range(1, len(close)):
        if returns.iloc[i] > 0:
            up_count.iloc[i] = up_count.iloc[i-1] + 1
            down_count.iloc[i] = 0
        elif returns.iloc[i] < 0:
            down_count.iloc[i] = down_count.iloc[i-1] + 1
            up_count.iloc[i] = 0
        else:
            up_count.iloc[i] = up_count.iloc[i-1]
            down_count.iloc[i] = down_count.iloc[i-1]

    signals = pd.Series(0, index=close.index)
    position = 0

    for i in range(5, len(close)):
        if position == 0:
            if down_count.iloc[i] >= down_days:
                position = 1
        elif position == 1:
            if up_count.iloc[i] >= up_days:
                position = 0

        signals.iloc[i] = position

    return signals


def run_backtest(signals: pd.Series, data: dict, benchmark: str, start_year: int) -> Result:
    """Run backtest."""
    df = data[benchmark]
    close = df["Close"]
    returns = close.pct_change()

    mask = close.index >= f"{start_year}-01-01"
    signals = signals[mask]
    returns = returns[mask]

    strat_returns = signals.shift(1) * returns
    strat_returns = strat_returns.fillna(0)

    metrics = calc_metrics(strat_returns, returns)

    changes = signals.diff().abs()
    trades = int(changes.sum() / 2)

    trade_rets = strat_returns[strat_returns != 0]
    win_rate = (trade_rets > 0).mean() * 100 if len(trade_rets) > 0 else 0

    exposure = (signals != 0).mean() * 100

    return Result(
        name="",
        params={},
        cagr=metrics["cagr"],
        mdd=metrics["mdd"],
        sharpe=metrics["sharpe"],
        beta=metrics["beta"],
        trades=trades,
        win_rate=win_rate,
        exposure=exposure,
        returns=strat_returns,
    )


def main():
    console.print("[bold blue]Quick Strategy Optimization[/bold blue]\n")

    data = download_data(["SPY", "QQQ", "TLT", "GLD"], start="2000-01-01")
    if not data:
        console.print("[red]No data[/red]")
        return

    results = {}

    # MRETF - IBS-based with strict filter
    console.print("\n[cyan]Testing MRETF variants...[/cyan]")
    best_mretf = None
    best_score = float("inf")

    for entry in [0.05, 0.1, 0.15, 0.2]:
        for exit_ in [0.8, 0.85, 0.9, 0.95]:
            for max_hold in [5, 10, 15, 20]:
                signals = strategy_mretf_ibs(data, entry, exit_, max_hold)
                if signals.empty:
                    continue
                result = run_backtest(signals, data, "SPY", TARGETS["MRETF"]["start"])

                score = (
                    abs(result.cagr - TARGETS["MRETF"]["cagr"]) / 5.2 +
                    abs(result.sharpe - TARGETS["MRETF"]["sharpe"]) / 0.78 * 2 +
                    abs(result.beta - TARGETS["MRETF"]["beta"]) / 0.14 * 2
                )

                if score < best_score:
                    best_score = score
                    result.params = {"entry": entry, "exit": exit_, "max_hold": max_hold}
                    best_mretf = result

    if best_mretf:
        best_mretf.name = "MRETF"
        results["MRETF"] = best_mretf
        console.print(f"  MRETF: CAGR={best_mretf.cagr:.1f}%, Sharpe={best_mretf.sharpe:.2f}, Beta={best_mretf.beta:.2f}")
        console.print(f"  Params: {best_mretf.params}")

    # ETFMR - RSI + IBS combo
    console.print("\n[cyan]Testing ETFMR variants...[/cyan]")
    best_etfmr = None
    best_score = float("inf")

    for rsi_t in [15, 20, 25, 30, 35]:
        for ibs_t in [0.15, 0.2, 0.25, 0.3, 0.35]:
            signals = strategy_etfmr_combo(data, rsi_t, ibs_t)
            if signals.empty:
                continue
            result = run_backtest(signals, data, "SPY", TARGETS["ETFMR"]["start"])

            score = (
                abs(result.cagr - TARGETS["ETFMR"]["cagr"]) / 10 +
                abs(result.sharpe - TARGETS["ETFMR"]["sharpe"]) / 0.82 * 2 +
                abs(result.beta - TARGETS["ETFMR"]["beta"]) / 0.44
            )

            if score < best_score:
                best_score = score
                result.params = {"rsi_thresh": rsi_t, "ibs_thresh": ibs_t}
                best_etfmr = result

    if best_etfmr:
        best_etfmr.name = "ETFMR"
        results["ETFMR"] = best_etfmr
        console.print(f"  ETFMR: CAGR={best_etfmr.cagr:.1f}%, Sharpe={best_etfmr.sharpe:.2f}, Beta={best_etfmr.beta:.2f}")
        console.print(f"  Params: {best_etfmr.params}")

    # MRMOM - Regime switching
    console.print("\n[cyan]Testing MRMOM variants...[/cyan]")
    best_mrmom = None
    best_score = float("inf")

    for mom_p in [200, 252, 300]:
        for rsi_e in [10, 15, 20, 25]:
            signals = strategy_mrmom(data, mom_p, rsi_e)
            if signals.empty:
                continue
            result = run_backtest(signals, data, "SPY", TARGETS["MRMOM"]["start"])

            score = (
                abs(result.cagr - TARGETS["MRMOM"]["cagr"]) / 10.3 +
                abs(result.sharpe - TARGETS["MRMOM"]["sharpe"]) / 1.15 * 2 +
                abs(result.beta - TARGETS["MRMOM"]["beta"]) / 0.32
            )

            if score < best_score:
                best_score = score
                result.params = {"mom_period": mom_p, "rsi_entry": rsi_e}
                best_mrmom = result

    if best_mrmom:
        best_mrmom.name = "MRMOM"
        results["MRMOM"] = best_mrmom
        console.print(f"  MRMOM: CAGR={best_mrmom.cagr:.1f}%, Sharpe={best_mrmom.sharpe:.2f}, Beta={best_mrmom.beta:.2f}")
        console.print(f"  Params: {best_mrmom.params}")

    # B2S2 - Consecutive days
    console.print("\n[cyan]Testing B2S2 variants...[/cyan]")
    best_b2s2 = None
    best_score = float("inf")

    for down in [1, 2, 3, 4]:
        for up in [1, 2, 3]:
            signals = strategy_b2s2(data, down, up)
            if signals.empty:
                continue
            result = run_backtest(signals, data, "SPY", TARGETS["B2S2ETF"]["start"])

            score = (
                abs(result.cagr - TARGETS["B2S2ETF"]["cagr"]) / 9.1 +
                abs(result.sharpe - TARGETS["B2S2ETF"]["sharpe"]) / 0.63 * 2 +
                abs(result.beta - TARGETS["B2S2ETF"]["beta"]) / 0.60
            )

            if score < best_score:
                best_score = score
                result.params = {"down_days": down, "up_days": up}
                best_b2s2 = result

    if best_b2s2:
        best_b2s2.name = "B2S2ETF"
        results["B2S2ETF"] = best_b2s2
        console.print(f"  B2S2ETF: CAGR={best_b2s2.cagr:.1f}%, Sharpe={best_b2s2.sharpe:.2f}, Beta={best_b2s2.beta:.2f}")
        console.print(f"  Params: {best_b2s2.params}")

    # ===========================================================================
    # RESULTS TABLE
    # ===========================================================================
    console.print("\n")
    table = Table(title="Optimized Results vs Targets")
    table.add_column("Strategy", style="cyan")
    table.add_column("Target CAGR", justify="right")
    table.add_column("Achieved", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("Target Sharpe", justify="right")
    table.add_column("Achieved", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("Target Beta", justify="right")
    table.add_column("Achieved", justify="right")

    for name, result in results.items():
        t = TARGETS[name]
        cagr_diff = result.cagr - t["cagr"]
        sharpe_diff = result.sharpe - t["sharpe"]

        cagr_style = "green" if abs(cagr_diff) < 2 else ("yellow" if abs(cagr_diff) < 4 else "red")
        sharpe_style = "green" if abs(sharpe_diff) < 0.15 else ("yellow" if abs(sharpe_diff) < 0.3 else "red")

        table.add_row(
            name,
            f"{t['cagr']}%",
            f"[{cagr_style}]{result.cagr:.1f}%[/{cagr_style}]",
            f"{cagr_diff:+.1f}",
            f"{t['sharpe']}",
            f"[{sharpe_style}]{result.sharpe:.2f}[/{sharpe_style}]",
            f"{sharpe_diff:+.2f}",
            f"{t['beta']}",
            f"{result.beta:.2f}",
        )

    console.print(table)

    # ===========================================================================
    # PLOT
    # ===========================================================================
    output_dir = Path("/home/user/ai-playground/outputs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        t = TARGETS[name]

        equity = (1 + result.returns).cumprod()
        ax.plot(equity.index, equity, label=f"{name}", linewidth=2)

        # Benchmark
        spy_ret = data["SPY"]["Close"].pct_change()
        spy_ret = spy_ret[spy_ret.index >= f"{t['start']}-01-01"]
        spy_eq = (1 + spy_ret).cumprod()
        ax.plot(spy_eq.index, spy_eq, label="SPY", color="gray", alpha=0.7)

        ax.set_title(f"{name}: Target CAGR={t['cagr']}%, Achieved={result.cagr:.1f}%")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    plot_path = output_dir / f"quick_optimize_{timestamp}.png"
    plt.savefig(plot_path, dpi=150)
    console.print(f"\n[green]Saved plot to {plot_path}[/green]")

    # ===========================================================================
    # INFERRED RULES SUMMARY
    # ===========================================================================
    console.print("\n[bold]INFERRED STRATEGY RULES:[/bold]\n")

    for name, result in results.items():
        t = TARGETS[name]
        console.print(f"[cyan]{name}[/cyan] (Target: CAGR={t['cagr']}%, Sharpe={t['sharpe']}, Beta={t['beta']})")
        console.print(f"  Achieved: CAGR={result.cagr:.1f}%, Sharpe={result.sharpe:.2f}, Beta={result.beta:.2f}")
        console.print(f"  Parameters: {result.params}")
        console.print(f"  Stats: {result.trades} trades, {result.win_rate:.1f}% win rate, {result.exposure:.1f}% exposure")
        console.print()


if __name__ == "__main__":
    main()
