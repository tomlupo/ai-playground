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
Final optimized strategy reverse engineering.
Uses vectorized operations for speed.
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


def momersion_fast(returns: pd.Series, lookback: int = 252) -> pd.Series:
    """
    Fast vectorized Momersion indicator.
    Momersion = 100 * (count of same-sign consecutive returns) / total
    """
    # Product of consecutive returns
    product = returns * returns.shift(1)

    # Momentum: same sign (product > 0)
    momentum = (product > 0).astype(int)

    # Mean-reversion: opposite sign (product < 0)
    meanrev = (product < 0).astype(int)

    # Rolling sum
    mom_count = momentum.rolling(lookback).sum()
    mr_count = meanrev.rolling(lookback).sum()

    total = mom_count + mr_count
    return 100 * mom_count / total.replace(0, np.nan)


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
# STRATEGIES
# ==============================================================================

def strategy_mretf_ibs(data: dict, entry: float = 0.1, exit_: float = 0.9, max_hold: int = 15) -> pd.Series:
    """MRETF using IBS with trend filter."""
    df = data.get("SPY")
    if df is None:
        return pd.Series(dtype=float)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    ibs_val = ibs(high, low, close)
    sma_200 = close.rolling(200).mean()
    trend_ok = close > sma_200

    signals = pd.Series(0, index=close.index)
    position = 0
    entry_idx = 0

    for i in range(201, len(close)):
        if position == 0 and trend_ok.iloc[i]:
            if ibs_val.iloc[i] < entry:
                position = 1
                entry_idx = i
        elif position == 1:
            if ibs_val.iloc[i] > exit_ or (i - entry_idx) >= max_hold:
                position = 0
        signals.iloc[i] = position

    return signals


def strategy_etfmr_combo(data: dict, rsi_thresh: float = 35, ibs_thresh: float = 0.15) -> pd.Series:
    """ETFMR using RSI + IBS combo."""
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
            if rsi_val.iloc[i] < rsi_thresh and ibs_val.iloc[i] < ibs_thresh:
                position = 1
        elif position == 1:
            if close.iloc[i] > sma_5.iloc[i]:
                position = 0
        signals.iloc[i] = position

    return signals


def strategy_mrmom(data: dict, mom_period: int = 252, rsi_entry: float = 15, mom_thresh: float = 0.02) -> pd.Series:
    """MRMOM regime switching using fast Momersion."""
    symbols = ["SPY", "QQQ", "TLT", "GLD"]
    available = [s for s in symbols if s in data]

    all_signals = []

    for symbol in available:
        df = data[symbol]
        close = df["Close"]
        returns = close.pct_change()

        mom = momersion_fast(returns, mom_period)
        rsi_val = rsi(close, 2)
        momentum = close.pct_change(20)
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

            # Mean-reversion regime (< 50)
            if regime < 50:
                if position == 0 and trend_ok and rsi_val.iloc[i] < rsi_entry:
                    position = 1
                elif position == 1 and close.iloc[i] > sma_5.iloc[i]:
                    position = 0
            # Momentum regime (>= 50)
            else:
                if position == 0 and momentum.iloc[i] > mom_thresh:
                    position = 1
                elif position == 1 and momentum.iloc[i] < -mom_thresh:
                    position = 0

            signals.iloc[i] = position

        all_signals.append(signals)

    if not all_signals:
        return pd.Series(dtype=float)

    return pd.concat(all_signals, axis=1).mean(axis=1)


def strategy_b2s2(data: dict, down_days: int = 2, up_days: int = 1) -> pd.Series:
    """B2S2 consecutive days strategy."""
    df = data.get("SPY")
    if df is None:
        return pd.Series(dtype=float)

    close = df["Close"]
    returns = close.pct_change()

    # Vectorized consecutive day counting
    down = (returns < 0).astype(int)
    up = (returns > 0).astype(int)

    # Use rolling sum for consecutive counts (approximation)
    down_count = down.rolling(down_days).sum()
    up_count = up.rolling(up_days).sum()

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
    console.print("[bold blue]Final Strategy Optimization[/bold blue]\n")

    data = download_data(["SPY", "QQQ", "TLT", "GLD"], start="2000-01-01")
    if not data:
        console.print("[red]No data[/red]")
        return

    results = {}

    # ===========================================================================
    # MRETF - IBS-based with strict filter
    # Target: CAGR 5.2%, Sharpe 0.78, Beta 0.14
    # ===========================================================================
    console.print("\n[cyan]Optimizing MRETF...[/cyan]")
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
        console.print(f"  Best: CAGR={best_mretf.cagr:.1f}% (target 5.2%), Sharpe={best_mretf.sharpe:.2f} (target 0.78)")
        console.print(f"  Params: {best_mretf.params}")

    # ===========================================================================
    # ETFMR - RSI + IBS combo
    # Target: CAGR 10.0%, Sharpe 0.82, Beta 0.44
    # ===========================================================================
    console.print("\n[cyan]Optimizing ETFMR...[/cyan]")
    best_etfmr = None
    best_score = float("inf")

    for rsi_t in [15, 20, 25, 30, 35, 40]:
        for ibs_t in [0.1, 0.15, 0.2, 0.25, 0.3]:
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
        console.print(f"  Best: CAGR={best_etfmr.cagr:.1f}% (target 10.0%), Sharpe={best_etfmr.sharpe:.2f} (target 0.82)")
        console.print(f"  Params: {best_etfmr.params}")

    # ===========================================================================
    # MRMOM - Regime switching
    # Target: CAGR 10.3%, Sharpe 1.15, Beta 0.32
    # ===========================================================================
    console.print("\n[cyan]Optimizing MRMOM (using fast Momersion)...[/cyan]")
    best_mrmom = None
    best_score = float("inf")

    for mom_p in [200, 252]:
        for rsi_e in [10, 15, 20]:
            for mom_t in [0.01, 0.02, 0.03]:
                signals = strategy_mrmom(data, mom_p, rsi_e, mom_t)
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
                    result.params = {"mom_period": mom_p, "rsi_entry": rsi_e, "mom_thresh": mom_t}
                    best_mrmom = result

    if best_mrmom:
        best_mrmom.name = "MRMOM"
        results["MRMOM"] = best_mrmom
        console.print(f"  Best: CAGR={best_mrmom.cagr:.1f}% (target 10.3%), Sharpe={best_mrmom.sharpe:.2f} (target 1.15)")
        console.print(f"  Params: {best_mrmom.params}")

    # ===========================================================================
    # B2S2ETF - Consecutive days
    # Target: CAGR 9.1%, Sharpe 0.63, Beta 0.60
    # ===========================================================================
    console.print("\n[cyan]Optimizing B2S2ETF...[/cyan]")
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
        console.print(f"  Best: CAGR={best_b2s2.cagr:.1f}% (target 9.1%), Sharpe={best_b2s2.sharpe:.2f} (target 0.63)")
        console.print(f"  Params: {best_b2s2.params}")

    # ===========================================================================
    # RESULTS TABLE
    # ===========================================================================
    console.print("\n")
    table = Table(title="FINAL RESULTS: Optimized vs Targets")
    table.add_column("Strategy", style="cyan")
    table.add_column("Target CAGR", justify="right")
    table.add_column("Achieved", justify="right")
    table.add_column("Target Sharpe", justify="right")
    table.add_column("Achieved", justify="right")
    table.add_column("Target Beta", justify="right")
    table.add_column("Achieved", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win%", justify="right")

    for name, result in results.items():
        t = TARGETS[name]
        cagr_diff = abs(result.cagr - t["cagr"])
        sharpe_diff = abs(result.sharpe - t["sharpe"])

        cagr_style = "green" if cagr_diff < 2 else ("yellow" if cagr_diff < 4 else "red")
        sharpe_style = "green" if sharpe_diff < 0.15 else ("yellow" if sharpe_diff < 0.3 else "red")

        table.add_row(
            name,
            f"{t['cagr']}%",
            f"[{cagr_style}]{result.cagr:.1f}%[/{cagr_style}]",
            f"{t['sharpe']}",
            f"[{sharpe_style}]{result.sharpe:.2f}[/{sharpe_style}]",
            f"{t['beta']}",
            f"{result.beta:.2f}",
            str(result.trades),
            f"{result.win_rate:.1f}%",
        )

    console.print(table)

    # ===========================================================================
    # SAVE OUTPUTS
    # ===========================================================================
    output_dir = Path("/home/user/ai-playground/outputs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Reverse-Engineered Strategies vs Targets", fontsize=14)

    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        t = TARGETS[name]

        equity = (1 + result.returns).cumprod()
        ax.plot(equity.index, equity, label=f"{name} (Sharpe={result.sharpe:.2f})", linewidth=2, color="blue")

        spy_ret = data["SPY"]["Close"].pct_change()
        spy_ret = spy_ret[spy_ret.index >= f"{t['start']}-01-01"]
        spy_eq = (1 + spy_ret).cumprod()
        ax.plot(spy_eq.index, spy_eq, label="SPY B&H", color="gray", alpha=0.7)

        ax.set_title(f"{name}: CAGR {result.cagr:.1f}% vs {t['cagr']}% target")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    plot_path = output_dir / f"final_optimization_{timestamp}.png"
    plt.savefig(plot_path, dpi=150)
    console.print(f"\n[green]Saved plot: {plot_path}[/green]")

    # ===========================================================================
    # SUMMARY OF INFERRED RULES
    # ===========================================================================
    console.print("\n[bold yellow]═══════════════════════════════════════════════════════════════[/bold yellow]")
    console.print("[bold]INFERRED STRATEGY RULES[/bold]")
    console.print("[bold yellow]═══════════════════════════════════════════════════════════════[/bold yellow]\n")

    rules = {
        "MRETF": """
[cyan]MRETF[/cyan] - IBS Mean-Reversion with Trend Filter
  Target: CAGR=5.2%, Sharpe=0.78, Beta=0.14

  INFERRED RULES:
  • Entry: IBS < {entry} (close near day's low)
  • Exit: IBS > {exit} OR held > {max_hold} days
  • Filter: Price > 200-day SMA (uptrend only)
  • Position: 100% when signal triggered

  WHY IT WORKS:
  - Low beta (0.14) = very selective, only trades in uptrends
  - Quick exits prevent drawdowns
  - IBS captures short-term mean-reversion effectively
""",
        "ETFMR": """
[cyan]ETFMR[/cyan] - "Two Popular Indicators" Strategy
  Target: CAGR=10.0%, Sharpe=0.82, Beta=0.44

  INFERRED RULES:
  • Entry: RSI(2) < {rsi_thresh} AND IBS < {ibs_thresh}
  • Exit: Price > 5-day SMA
  • No trend filter (higher exposure = higher beta)
  • Position: 100% when both conditions met

  WHY IT WORKS:
  - Dual confirmation reduces false signals
  - No filter allows more trades (higher CAGR potential)
  - 5-day SMA exit captures mean-reversion profits
""",
        "MRMOM": """
[cyan]MRMOM[/cyan] - Momersion Regime Switching
  Target: CAGR=10.3%, Sharpe=1.15, Beta=0.32

  INFERRED RULES:
  • Regime Detection: Momersion({mom_period})
    - If Momersion < 50: Mean-reversion regime
    - If Momersion >= 50: Momentum regime

  • Mean-Reversion Mode (Momersion < 50):
    - Entry: RSI(2) < {rsi_entry} AND Price > 200 SMA
    - Exit: Price > 5-day SMA

  • Momentum Mode (Momersion >= 50):
    - Entry: 20-day momentum > {mom_thresh}
    - Exit: 20-day momentum < -{mom_thresh}

  • Position: Equal weight across SPY, QQQ, TLT, GLD

  WHY IT WORKS:
  - Adapts to market regime (best of both worlds)
  - Diversification across asset classes reduces beta
  - Highest Sharpe due to regime awareness
""",
        "B2S2ETF": """
[cyan]B2S2ETF[/cyan] - Buy 2 Sell 2 (Consecutive Days)
  Target: CAGR=9.1%, Sharpe=0.63, Beta=0.60

  INFERRED RULES:
  • Entry: {down_days}+ consecutive down days
  • Exit: {up_days}+ consecutive up days
  • No filters (as stated: "no parameters, no filters")
  • Position: 100% on signal

  WHY IT WORKS:
  - Extreme simplicity avoids overfitting
  - High beta (0.60) = frequently invested
  - Captures basic mean-reversion tendency
  - "B2S2" name suggests Buy-2-Sell-2 pattern
""",
    }

    for name, result in results.items():
        rule_template = rules.get(name, "")
        if rule_template:
            formatted = rule_template.format(**result.params)
            console.print(formatted)

    # Save detailed markdown report
    report_path = output_dir / f"inferred_rules_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write("# PriceActionLab Strategy Reverse Engineering - Final Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Strategy | Target CAGR | Achieved | Target Sharpe | Achieved | Target Beta | Achieved |\n")
        f.write("|----------|-------------|----------|---------------|----------|-------------|----------|\n")
        for name, result in results.items():
            t = TARGETS[name]
            f.write(f"| {name} | {t['cagr']}% | {result.cagr:.1f}% | {t['sharpe']} | {result.sharpe:.2f} | {t['beta']} | {result.beta:.2f} |\n")

        f.write("\n## Inferred Parameters\n\n")
        for name, result in results.items():
            f.write(f"### {name}\n")
            f.write(f"```python\n{result.params}\n```\n\n")

    console.print(f"[green]Saved report: {report_path}[/green]")


if __name__ == "__main__":
    main()
