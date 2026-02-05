# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "yfinance>=0.2.36",
#     "rich>=13.0",
#     "scipy>=1.11",
#     "matplotlib>=3.7",
# ]
# ///
"""
PriceActionLab Strategy Reverse Engineering - Enhanced Version

Uses parameter optimization and additional indicators to better match target metrics.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.table import Table
from rich.progress import track

warnings.filterwarnings("ignore")
console = Console()

# Target metrics from PriceActionLab
TARGETS = {
    "MRETF": {"cagr": 5.2, "mdd": -9.3, "sharpe": 0.78, "corr": 0.41, "beta": 0.14, "start": 2002},
    "MRMOM": {"cagr": 10.3, "mdd": -16.8, "sharpe": 1.15, "corr": 0.67, "beta": 0.32, "start": 2003},
    "B2S2ETF": {"cagr": 9.1, "mdd": -30.6, "sharpe": 0.63, "corr": 0.76, "beta": 0.60, "start": 1993},
    "ETFMR": {"cagr": 10.0, "mdd": -22.9, "sharpe": 0.82, "corr": 0.68, "beta": 0.44, "start": 2003},
    "ETFSEAS": {"cagr": 7.3, "mdd": -13.4, "sharpe": 0.83, "corr": 0.22, "beta": 0.10, "start": 2005},
}


@dataclass
class Result:
    name: str
    params: dict
    cagr: float
    mdd: float
    sharpe: float
    corr: float
    beta: float
    trades: int
    win_rate: float
    exposure: float
    returns: pd.Series


def download_data(symbols: list[str], start: str = "2000-01-01") -> dict[str, pd.DataFrame]:
    """Download OHLCV data for symbols."""
    console.print(f"[cyan]Downloading data for {symbols}...[/cyan]")
    data = {}

    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start, progress=False, auto_adjust=True)
            if not df.empty:
                # Handle MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                data[symbol] = df
        except Exception as e:
            console.print(f"[yellow]Warning: {symbol}: {e}[/yellow]")

    console.print(f"[green]Downloaded data for {len(data)} symbols[/green]")
    return data


# ==============================================================================
# INDICATORS
# ==============================================================================

def rsi(prices: pd.Series, period: int = 2) -> pd.Series:
    """RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def cumulative_rsi(prices: pd.Series, rsi_period: int = 2, cum_period: int = 2) -> pd.Series:
    """Cumulative RSI - sum of RSI over cum_period days."""
    rsi_val = rsi(prices, rsi_period)
    return rsi_val.rolling(cum_period).sum()


def ibs(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Internal Bar Strength: (Close - Low) / (High - Low)."""
    return (close - low) / (high - low + 1e-10)


def momersion(returns: pd.Series, lookback: int = 250) -> pd.Series:
    """Momersion indicator for regime detection."""
    def calc(window):
        if len(window) < 2:
            return 50
        mom = mr = 0
        for i in range(1, len(window)):
            prod = window.iloc[i] * window.iloc[i-1]
            if prod > 0:
                mom += 1
            elif prod < 0:
                mr += 1
        total = mom + mr
        return 100 * mom / total if total > 0 else 50
    return returns.rolling(lookback).apply(calc, raw=False)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 5) -> pd.Series:
    """Average True Range."""
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ==============================================================================
# METRICS
# ==============================================================================

def calc_metrics(returns: pd.Series, benchmark: pd.Series, rf: float = 0.02) -> dict:
    """Calculate performance metrics."""
    # Align
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if len(aligned) < 30:
        return {"cagr": 0, "mdd": 0, "sharpe": 0, "corr": 0, "beta": 0}

    strat = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]

    # CAGR
    total = (1 + strat).prod() - 1
    years = len(strat) / 252
    cagr = ((1 + total) ** (1 / years) - 1) * 100 if years > 0 else 0

    # MDD
    cum = (1 + strat).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100

    # Sharpe
    vol = strat.std() * np.sqrt(252)
    sharpe = (cagr / 100 - rf) / vol if vol > 0 else 0

    # Correlation & Beta
    corr = strat.corr(bench)
    cov = strat.cov(bench)
    var = bench.var()
    beta = cov / var if var > 0 else 0

    return {"cagr": cagr, "mdd": mdd, "sharpe": sharpe, "corr": corr, "beta": beta}


# ==============================================================================
# STRATEGIES
# ==============================================================================

def strategy_rsi_mr(
    data: dict[str, pd.DataFrame],
    symbols: list[str],
    rsi_period: int = 2,
    entry_threshold: float = 10,
    exit_threshold: float = 70,
    use_trend_filter: bool = True,
    trend_period: int = 200,
    use_cum_rsi: bool = False,
    cum_period: int = 2,
    cum_threshold: float = 20,
) -> pd.Series:
    """RSI-based mean reversion strategy."""

    # Get first available symbol
    symbol = next((s for s in symbols if s in data), None)
    if not symbol:
        return pd.Series(dtype=float)

    df = data[symbol]
    close = df["Close"]

    # Calculate indicators
    if use_cum_rsi:
        indicator = cumulative_rsi(close, rsi_period, cum_period)
        entry_level = cum_threshold
    else:
        indicator = rsi(close, rsi_period)
        entry_level = entry_threshold

    sma_exit = close.rolling(5).mean()
    sma_trend = close.rolling(trend_period).mean()

    # Generate signals
    signals = pd.Series(0, index=close.index)
    position = 0

    for i in range(trend_period + 1, len(close)):
        # Trend filter
        trend_ok = True
        if use_trend_filter:
            trend_ok = close.iloc[i] > sma_trend.iloc[i]

        if position == 0 and trend_ok:
            if indicator.iloc[i] < entry_level:
                position = 1
        elif position == 1:
            # Exit on price > 5-day SMA or RSI > exit threshold
            if close.iloc[i] > sma_exit.iloc[i]:
                position = 0
            elif not use_cum_rsi and indicator.iloc[i] > exit_threshold:
                position = 0

        signals.iloc[i] = position

    return signals


def strategy_ibs(
    data: dict[str, pd.DataFrame],
    symbols: list[str],
    entry_threshold: float = 0.2,
    exit_threshold: float = 0.8,
    use_trend_filter: bool = True,
    trend_period: int = 200,
    max_hold_days: int = 10,
) -> pd.Series:
    """IBS (Internal Bar Strength) mean reversion strategy."""

    symbol = next((s for s in symbols if s in data), None)
    if not symbol:
        return pd.Series(dtype=float)

    df = data[symbol]
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    ibs_val = ibs(high, low, close)
    sma_trend = close.rolling(trend_period).mean()

    signals = pd.Series(0, index=close.index)
    position = 0
    entry_day = 0

    for i in range(trend_period + 1, len(close)):
        trend_ok = close.iloc[i] > sma_trend.iloc[i] if use_trend_filter else True

        if position == 0 and trend_ok:
            if ibs_val.iloc[i] < entry_threshold:
                position = 1
                entry_day = i
        elif position == 1:
            days_held = i - entry_day
            if ibs_val.iloc[i] > exit_threshold or days_held >= max_hold_days:
                position = 0

        signals.iloc[i] = position

    return signals


def strategy_rsi_ibs_combo(
    data: dict[str, pd.DataFrame],
    symbols: list[str],
    rsi_period: int = 2,
    rsi_threshold: float = 25,
    ibs_threshold: float = 0.3,
    use_trend_filter: bool = True,
    trend_period: int = 200,
) -> pd.Series:
    """Combined RSI + IBS strategy - requires both conditions."""

    symbol = next((s for s in symbols if s in data), None)
    if not symbol:
        return pd.Series(dtype=float)

    df = data[symbol]
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    rsi_val = rsi(close, rsi_period)
    ibs_val = ibs(high, low, close)
    sma_exit = close.rolling(5).mean()
    sma_trend = close.rolling(trend_period).mean()

    signals = pd.Series(0, index=close.index)
    position = 0

    for i in range(trend_period + 1, len(close)):
        trend_ok = close.iloc[i] > sma_trend.iloc[i] if use_trend_filter else True

        if position == 0 and trend_ok:
            # Enter when both RSI and IBS are oversold
            if rsi_val.iloc[i] < rsi_threshold and ibs_val.iloc[i] < ibs_threshold:
                position = 1
        elif position == 1:
            if close.iloc[i] > sma_exit.iloc[i]:
                position = 0

        signals.iloc[i] = position

    return signals


def strategy_momersion_regime(
    data: dict[str, pd.DataFrame],
    symbols: list[str],
    mom_period: int = 252,
    rsi_period: int = 2,
    rsi_entry: float = 15,
    momentum_period: int = 20,
    regime_threshold: float = 50,
) -> pd.Series:
    """Momersion-based regime switching strategy."""

    available = [s for s in symbols if s in data]
    if not available:
        return pd.Series(dtype=float)

    # Use equal weight across symbols
    all_signals = []

    for symbol in available:
        df = data[symbol]
        close = df["Close"]
        returns = close.pct_change()

        mom = momersion(returns, mom_period)
        rsi_val = rsi(close, rsi_period)
        momentum = close / close.shift(momentum_period) - 1
        sma_exit = close.rolling(5).mean()
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
            if regime < regime_threshold:
                if position == 0 and trend_ok and rsi_val.iloc[i] < rsi_entry:
                    position = 1
                elif position == 1 and close.iloc[i] > sma_exit.iloc[i]:
                    position = 0
            # Momentum regime
            else:
                if position == 0 and momentum.iloc[i] > 0:
                    position = 1
                elif position == 1 and momentum.iloc[i] < 0:
                    position = 0

            signals.iloc[i] = position

        all_signals.append(signals)

    # Average signals across symbols
    combined = pd.concat(all_signals, axis=1).mean(axis=1)
    return combined


def strategy_b2s2(
    data: dict[str, pd.DataFrame],
    symbols: list[str],
    down_days: int = 2,
    up_days: int = 2,
    use_trend_filter: bool = False,
    trend_period: int = 200,
) -> pd.Series:
    """B2S2 - Buy after N down days, sell after N up days."""

    symbol = next((s for s in symbols if s in data), None)
    if not symbol:
        return pd.Series(dtype=float)

    df = data[symbol]
    close = df["Close"]
    returns = close.pct_change()
    sma_trend = close.rolling(trend_period).mean()

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

    for i in range(max(down_days, up_days, trend_period) + 1, len(close)):
        trend_ok = close.iloc[i] > sma_trend.iloc[i] if use_trend_filter else True

        if position == 0 and trend_ok:
            if down_count.iloc[i] >= down_days:
                position = 1
        elif position == 1:
            if up_count.iloc[i] >= up_days:
                position = 0

        signals.iloc[i] = position

    return signals


def strategy_seasonality(
    data: dict[str, pd.DataFrame],
    symbols: list[str],
    eom_days: int = 4,
    tom_days: int = 3,
    use_friday_monday: bool = True,
) -> pd.Series:
    """Seasonality strategy - end of month, turn of month, day of week."""

    symbol = next((s for s in symbols if s in data), None)
    if not symbol:
        return pd.Series(dtype=float)

    df = data[symbol]
    close = df["Close"]

    signals = pd.Series(0.0, index=close.index)

    for i, date in enumerate(close.index):
        signal = 0

        # End of month effect - last N trading days
        if i < len(close) - 1:
            next_dates = close.index[close.index > date]
            if len(next_dates) > 0:
                days_to_eom = len(next_dates[next_dates.month == date.month])
                if days_to_eom < eom_days:
                    signal = 1

        # Turn of month - first N days of month
        trading_days_in_month = close.index[close.index.month == date.month]
        day_of_month = list(trading_days_in_month).index(date) if date in trading_days_in_month else -1
        if 0 <= day_of_month < tom_days:
            signal = 1

        # Friday/Monday effect
        if use_friday_monday and date.dayofweek in [0, 4]:  # Monday, Friday
            signal = 1

        signals.iloc[i] = signal

    return signals


# ==============================================================================
# GRID SEARCH
# ==============================================================================

def run_backtest(signals: pd.Series, data: dict, symbol: str, start_year: int) -> Result | None:
    """Run backtest and calculate metrics."""
    if symbol not in data:
        return None

    df = data[symbol]
    close = df["Close"]
    returns = close.pct_change()

    # Filter by start year
    mask = close.index >= f"{start_year}-01-01"
    signals = signals[mask]
    returns = returns[mask]

    # Strategy returns
    strat_returns = signals.shift(1) * returns
    strat_returns = strat_returns.fillna(0)

    # Metrics
    metrics = calc_metrics(strat_returns, returns)

    # Trade stats
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
        corr=metrics["corr"],
        beta=metrics["beta"],
        trades=trades,
        win_rate=win_rate,
        exposure=exposure,
        returns=strat_returns,
    )


def grid_search_rsi(
    data: dict,
    symbols: list[str],
    target: dict,
    benchmark: str = "SPY"
) -> list[tuple[dict, Result, float]]:
    """Grid search for RSI strategy parameters."""

    results = []

    param_grid = {
        "rsi_period": [2, 3, 4],
        "entry_threshold": [5, 10, 15, 20],
        "exit_threshold": [60, 70, 80],
        "use_trend_filter": [True, False],
        "trend_period": [100, 200],
    }

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))

    for combo in combos:
        params = dict(zip(keys, combo))

        signals = strategy_rsi_mr(data, symbols, **params)
        if signals.empty:
            continue

        result = run_backtest(signals, data, benchmark, target["start"])
        if result is None:
            continue

        # Score: minimize distance to targets
        score = (
            abs(result.cagr - target["cagr"]) / max(target["cagr"], 1) +
            abs(result.mdd - target["mdd"]) / max(abs(target["mdd"]), 1) +
            abs(result.sharpe - target["sharpe"]) / max(target["sharpe"], 0.1) * 2 +  # Weight Sharpe more
            abs(result.beta - target["beta"]) * 2
        )

        result.params = params
        results.append((params, result, score))

    results.sort(key=lambda x: x[2])
    return results[:10]


def grid_search_ibs(
    data: dict,
    symbols: list[str],
    target: dict,
    benchmark: str = "SPY"
) -> list[tuple[dict, Result, float]]:
    """Grid search for IBS strategy parameters."""

    results = []

    param_grid = {
        "entry_threshold": [0.1, 0.15, 0.2, 0.25, 0.3],
        "exit_threshold": [0.7, 0.8, 0.9, 0.95],
        "use_trend_filter": [True, False],
        "trend_period": [100, 200],
        "max_hold_days": [5, 10, 15],
    }

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))

    for combo in combos:
        params = dict(zip(keys, combo))

        signals = strategy_ibs(data, symbols, **params)
        if signals.empty:
            continue

        result = run_backtest(signals, data, benchmark, target["start"])
        if result is None:
            continue

        score = (
            abs(result.cagr - target["cagr"]) / max(target["cagr"], 1) +
            abs(result.mdd - target["mdd"]) / max(abs(target["mdd"]), 1) +
            abs(result.sharpe - target["sharpe"]) / max(target["sharpe"], 0.1) * 2 +
            abs(result.beta - target["beta"]) * 2
        )

        result.params = params
        results.append((params, result, score))

    results.sort(key=lambda x: x[2])
    return results[:10]


def grid_search_combo(
    data: dict,
    symbols: list[str],
    target: dict,
    benchmark: str = "SPY"
) -> list[tuple[dict, Result, float]]:
    """Grid search for RSI+IBS combo strategy."""

    results = []

    param_grid = {
        "rsi_period": [2, 3],
        "rsi_threshold": [15, 20, 25, 30],
        "ibs_threshold": [0.2, 0.3, 0.4],
        "use_trend_filter": [True, False],
        "trend_period": [100, 200],
    }

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))

    for combo in combos:
        params = dict(zip(keys, combo))

        signals = strategy_rsi_ibs_combo(data, symbols, **params)
        if signals.empty:
            continue

        result = run_backtest(signals, data, benchmark, target["start"])
        if result is None:
            continue

        score = (
            abs(result.cagr - target["cagr"]) / max(target["cagr"], 1) +
            abs(result.mdd - target["mdd"]) / max(abs(target["mdd"]), 1) +
            abs(result.sharpe - target["sharpe"]) / max(target["sharpe"], 0.1) * 2 +
            abs(result.beta - target["beta"]) * 2
        )

        result.params = params
        results.append((params, result, score))

    results.sort(key=lambda x: x[2])
    return results[:10]


def grid_search_momersion(
    data: dict,
    symbols: list[str],
    target: dict,
    benchmark: str = "SPY"
) -> list[tuple[dict, Result, float]]:
    """Grid search for Momersion regime strategy."""

    results = []

    param_grid = {
        "mom_period": [200, 252, 300],
        "rsi_period": [2, 3],
        "rsi_entry": [10, 15, 20],
        "momentum_period": [10, 20, 30],
        "regime_threshold": [45, 50, 55],
    }

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))

    for combo in combos:
        params = dict(zip(keys, combo))

        signals = strategy_momersion_regime(data, symbols, **params)
        if signals.empty:
            continue

        result = run_backtest(signals, data, benchmark, target["start"])
        if result is None:
            continue

        score = (
            abs(result.cagr - target["cagr"]) / max(target["cagr"], 1) +
            abs(result.mdd - target["mdd"]) / max(abs(target["mdd"]), 1) +
            abs(result.sharpe - target["sharpe"]) / max(target["sharpe"], 0.1) * 2 +
            abs(result.beta - target["beta"]) * 2
        )

        result.params = params
        results.append((params, result, score))

    results.sort(key=lambda x: x[2])
    return results[:10]


def grid_search_b2s2(
    data: dict,
    symbols: list[str],
    target: dict,
    benchmark: str = "SPY"
) -> list[tuple[dict, Result, float]]:
    """Grid search for B2S2 strategy."""

    results = []

    param_grid = {
        "down_days": [1, 2, 3, 4],
        "up_days": [1, 2, 3],
        "use_trend_filter": [True, False],
        "trend_period": [100, 200],
    }

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))

    for combo in combos:
        params = dict(zip(keys, combo))

        signals = strategy_b2s2(data, symbols, **params)
        if signals.empty:
            continue

        result = run_backtest(signals, data, benchmark, target["start"])
        if result is None:
            continue

        score = (
            abs(result.cagr - target["cagr"]) / max(target["cagr"], 1) +
            abs(result.mdd - target["mdd"]) / max(abs(target["mdd"]), 1) +
            abs(result.sharpe - target["sharpe"]) / max(target["sharpe"], 0.1) * 2 +
            abs(result.beta - target["beta"]) * 2
        )

        result.params = params
        results.append((params, result, score))

    results.sort(key=lambda x: x[2])
    return results[:10]


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    console.print("[bold blue]PriceActionLab Strategy Reverse Engineering - Enhanced[/bold blue]\n")

    # Download data
    symbols = ["SPY", "QQQ", "TLT", "GLD", "^VIX"]
    data = download_data(symbols, start="2000-01-01")

    if not data:
        console.print("[red]Failed to download data[/red]")
        return

    best_results = {}

    # =========================================================================
    # MRETF - Target: CAGR 5.2%, MDD -9.3%, Sharpe 0.78, Beta 0.14
    # =========================================================================
    console.print("\n[cyan]Optimizing MRETF (RSI mean-reversion)...[/cyan]")

    # Test multiple approaches
    rsi_results = grid_search_rsi(data, ["SPY", "QQQ", "TLT"], TARGETS["MRETF"])
    ibs_results = grid_search_ibs(data, ["SPY"], TARGETS["MRETF"])
    combo_results = grid_search_combo(data, ["SPY"], TARGETS["MRETF"])

    all_mretf = rsi_results + ibs_results + combo_results
    all_mretf.sort(key=lambda x: x[2])

    if all_mretf:
        best = all_mretf[0]
        best[1].name = "MRETF"
        best_results["MRETF"] = best[1]
        console.print(f"  Best: CAGR={best[1].cagr:.1f}%, Sharpe={best[1].sharpe:.2f}, Beta={best[1].beta:.2f}")
        console.print(f"  Params: {best[0]}")

    # =========================================================================
    # ETFMR - Target: CAGR 10.0%, MDD -22.9%, Sharpe 0.82, Beta 0.44
    # =========================================================================
    console.print("\n[cyan]Optimizing ETFMR (two indicators)...[/cyan]")

    rsi_results = grid_search_rsi(data, ["SPY", "QQQ"], TARGETS["ETFMR"])
    combo_results = grid_search_combo(data, ["SPY", "QQQ"], TARGETS["ETFMR"])

    all_etfmr = rsi_results + combo_results
    all_etfmr.sort(key=lambda x: x[2])

    if all_etfmr:
        best = all_etfmr[0]
        best[1].name = "ETFMR"
        best_results["ETFMR"] = best[1]
        console.print(f"  Best: CAGR={best[1].cagr:.1f}%, Sharpe={best[1].sharpe:.2f}, Beta={best[1].beta:.2f}")
        console.print(f"  Params: {best[0]}")

    # =========================================================================
    # MRMOM - Target: CAGR 10.3%, MDD -16.8%, Sharpe 1.15, Beta 0.32
    # =========================================================================
    console.print("\n[cyan]Optimizing MRMOM (regime switching)...[/cyan]")

    mom_results = grid_search_momersion(data, ["SPY", "QQQ", "TLT", "GLD"], TARGETS["MRMOM"])

    if mom_results:
        best = mom_results[0]
        best[1].name = "MRMOM"
        best_results["MRMOM"] = best[1]
        console.print(f"  Best: CAGR={best[1].cagr:.1f}%, Sharpe={best[1].sharpe:.2f}, Beta={best[1].beta:.2f}")
        console.print(f"  Params: {best[0]}")

    # =========================================================================
    # B2S2ETF - Target: CAGR 9.1%, MDD -30.6%, Sharpe 0.63, Beta 0.60
    # =========================================================================
    console.print("\n[cyan]Optimizing B2S2ETF (consecutive days)...[/cyan]")

    b2s2_results = grid_search_b2s2(data, ["SPY"], TARGETS["B2S2ETF"])

    if b2s2_results:
        best = b2s2_results[0]
        best[1].name = "B2S2ETF"
        best_results["B2S2ETF"] = best[1]
        console.print(f"  Best: CAGR={best[1].cagr:.1f}%, Sharpe={best[1].sharpe:.2f}, Beta={best[1].beta:.2f}")
        console.print(f"  Params: {best[0]}")

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    console.print("\n")

    table = Table(title="Optimized Results vs Targets")
    table.add_column("Strategy", style="cyan")
    table.add_column("Target CAGR", justify="right")
    table.add_column("Achieved", justify="right")
    table.add_column("Target Sharpe", justify="right")
    table.add_column("Achieved", justify="right")
    table.add_column("Target Beta", justify="right")
    table.add_column("Achieved", justify="right")
    table.add_column("Win%", justify="right")

    for name, result in best_results.items():
        target = TARGETS[name]

        cagr_diff = abs(result.cagr - target["cagr"])
        cagr_style = "green" if cagr_diff < 2 else ("yellow" if cagr_diff < 5 else "red")

        sharpe_diff = abs(result.sharpe - target["sharpe"])
        sharpe_style = "green" if sharpe_diff < 0.2 else ("yellow" if sharpe_diff < 0.4 else "red")

        table.add_row(
            name,
            f"{target['cagr']}%",
            f"[{cagr_style}]{result.cagr:.1f}%[/{cagr_style}]",
            f"{target['sharpe']}",
            f"[{sharpe_style}]{result.sharpe:.2f}[/{sharpe_style}]",
            f"{target['beta']}",
            f"{result.beta:.2f}",
            f"{result.win_rate:.1f}%",
        )

    console.print(table)

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    output_dir = Path("/home/user/ai-playground/outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot equity curves
    fig, ax = plt.subplots(figsize=(14, 8))

    for name, result in best_results.items():
        equity = (1 + result.returns).cumprod()
        ax.plot(equity.index, equity, label=f"{name} (Sharpe={result.sharpe:.2f})", linewidth=1.5)

    # Benchmark
    if "SPY" in data:
        spy_ret = data["SPY"]["Close"].pct_change()
        spy_ret = spy_ret[spy_ret.index >= "2003-01-01"]
        spy_eq = (1 + spy_ret).cumprod()
        ax.plot(spy_eq.index, spy_eq, label="SPY (Buy & Hold)", color="black", linestyle="--")

    ax.set_title("Optimized Strategy Equity Curves", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plot_path = output_dir / f"optimized_strategies_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    console.print(f"\n[green]Saved plot to {plot_path}[/green]")

    # Save detailed results
    report = []
    report.append("# PriceActionLab Strategy Reverse Engineering - Optimized Results\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for name, result in best_results.items():
        target = TARGETS[name]
        report.append(f"## {name}\n\n")
        report.append(f"**Target:** CAGR={target['cagr']}%, Sharpe={target['sharpe']}, Beta={target['beta']}\n\n")
        report.append(f"**Achieved:** CAGR={result.cagr:.1f}%, Sharpe={result.sharpe:.2f}, Beta={result.beta:.2f}\n\n")
        report.append(f"**Parameters:**\n```python\n{result.params}\n```\n\n")
        report.append(f"**Stats:** Trades={result.trades}, Win Rate={result.win_rate:.1f}%, Exposure={result.exposure:.1f}%\n\n")
        report.append("---\n\n")

    report_path = output_dir / f"optimized_results_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write("".join(report))

    console.print(f"[green]Saved report to {report_path}[/green]")

    # Print best parameters
    console.print("\n[bold]Best Parameters Found:[/bold]\n")
    for name, result in best_results.items():
        console.print(f"[cyan]{name}:[/cyan]")
        for k, v in result.params.items():
            console.print(f"  {k}: {v}")
        console.print()


if __name__ == "__main__":
    main()
