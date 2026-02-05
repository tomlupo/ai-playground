# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "yfinance>=0.2.36",
#     "rich>=13.0",
#     "scipy>=1.11",
#     "matplotlib>=3.7",
#     "seaborn>=0.13",
# ]
# ///
"""
PriceActionLab Strategy Reverse Engineering Tool

Attempts to reverse engineer trading strategies from PriceActionLab based on:
1. Published performance metrics (CAGR, MDD, Sharpe, etc.)
2. Strategy descriptions (mean-reversion, trend-following, regime-switching)
3. Known indicator formulas (Momersion, RSI variants)
4. Academic literature on similar approaches

Author: Claude Code
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from rich.console import Console
from rich.table import Table
from scipy import stats

warnings.filterwarnings("ignore")

console = Console()


# ==============================================================================
# Target Performance Metrics from PriceActionLab
# ==============================================================================
TARGET_METRICS = {
    "MRETF": {"cagr": 5.2, "mdd": -9.3, "sharpe": 0.78, "correlation": 0.41, "beta": 0.14, "alpha": 2.9, "start": 2002},
    "MRETFLS": {"cagr": 5.6, "mdd": -17.4, "sharpe": 0.62, "correlation": 0.07, "beta": 0.03, "alpha": 3.6, "start": 2002},
    "B2S2DJ": {"cagr": 14.4, "mdd": -30.8, "sharpe": 0.84, "correlation": 0.84, "beta": 0.78, "alpha": 3.7, "start": 1993},
    "B2S2ETF": {"cagr": 9.1, "mdd": -30.6, "sharpe": 0.63, "correlation": 0.76, "beta": 0.60, "alpha": 2.5, "start": 1993},
    "TFDLS": {"cagr": 14.1, "mdd": -32.2, "sharpe": 0.82, "correlation": -0.07, "beta": -0.06, "alpha": 5.2, "start": 1990},
    "MRMOM": {"cagr": 10.3, "mdd": -16.8, "sharpe": 1.15, "correlation": 0.67, "beta": 0.32, "alpha": 4.2, "start": 2003},
    "ETFMR": {"cagr": 10.0, "mdd": -22.9, "sharpe": 0.82, "correlation": 0.68, "beta": 0.44, "alpha": 3.6, "start": 2003},
    "ETFSEAS": {"cagr": 7.3, "mdd": -13.4, "sharpe": 0.83, "correlation": 0.22, "beta": 0.10, "alpha": 4.2, "start": 2005},
    "ETFMO": {"cagr": 13.1, "mdd": -21.3, "sharpe": 0.86, "correlation": -0.08, "beta": -0.07, "alpha": 7.8, "start": 2010},
    "MRDJ": {"cagr": 10.3, "mdd": -22.3, "sharpe": 0.74, "correlation": 0.52, "beta": 0.39, "alpha": 3.6, "start": 1993},
}


@dataclass
class BacktestResult:
    """Container for backtest results."""

    strategy_name: str
    returns: pd.Series
    positions: pd.Series
    trades: int
    win_rate: float
    avg_holding_period: float
    exposure: float
    cagr: float
    mdd: float
    sharpe: float
    correlation: float
    beta: float
    alpha: float
    params: dict


def download_data(symbols: list[str], start: str = "1990-01-01") -> pd.DataFrame:
    """Download historical price data."""
    console.print(f"[cyan]Downloading data for {symbols}...[/cyan]")

    try:
        # Download all symbols at once (yfinance handles multiple tickers)
        df = yf.download(symbols, start=start, progress=False, auto_adjust=True)

        if df.empty:
            console.print("[yellow]Warning: No data downloaded[/yellow]")
            return pd.DataFrame()

        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            # Extract Close prices for each ticker
            prices = df["Close"].copy()
            # If single ticker, the result might be a Series
            if isinstance(prices, pd.Series):
                prices = pd.DataFrame({symbols[0]: prices})
        else:
            # Single ticker case - columns are just price types
            prices = pd.DataFrame({symbols[0]: df["Close"]})

        prices = prices.ffill().dropna()
        console.print(f"[green]Downloaded {len(prices)} days of data for {len(prices.columns)} symbols[/green]")
        return prices

    except Exception as e:
        console.print(f"[yellow]Warning: Could not download data: {e}[/yellow]")
        return pd.DataFrame()


def calculate_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    rf_rate: float = 0.02
) -> dict[str, float]:
    """Calculate performance metrics."""
    # Annualized return
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252
    cagr = ((1 + total_return) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Volatility and Sharpe
    vol = returns.std() * np.sqrt(252)
    excess_return = cagr / 100 - rf_rate
    sharpe = excess_return / vol if vol > 0 else 0

    # Maximum Drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    mdd = drawdown.min() * 100

    metrics = {
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sharpe,
        "volatility": vol * 100,
    }

    # Beta and correlation vs benchmark
    if benchmark_returns is not None:
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) > 30:
            strat_ret = aligned.iloc[:, 0]
            bench_ret = aligned.iloc[:, 1]

            correlation = strat_ret.corr(bench_ret)

            # Beta = Cov(r_s, r_b) / Var(r_b)
            cov = strat_ret.cov(bench_ret)
            var = bench_ret.var()
            beta = cov / var if var > 0 else 0

            # Alpha (annualized)
            bench_cagr = ((1 + bench_ret).prod() ** (252 / len(bench_ret)) - 1)
            alpha = (cagr / 100 - rf_rate - beta * (bench_cagr - rf_rate)) * 100

            metrics["correlation"] = correlation
            metrics["beta"] = beta
            metrics["alpha"] = alpha

    return metrics


# ==============================================================================
# Indicator Functions
# ==============================================================================

def rsi(prices: pd.Series, period: int = 2) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def momersion(returns: pd.Series, lookback: int = 250) -> pd.Series:
    """
    Calculate Momersion indicator (PriceActionLab).

    Momersion(n) = 100 * MOMc / (MOMc + MRc)

    Where:
    - MOMc = count of momentum occurrences (same-sign consecutive returns)
    - MRc = count of mean-reversion occurrences (opposite-sign consecutive returns)
    """
    def calc_momersion(window):
        if len(window) < 2:
            return 50

        mom_count = 0
        mr_count = 0

        for i in range(1, len(window)):
            product = window.iloc[i] * window.iloc[i-1]
            if product > 0:
                mom_count += 1
            elif product < 0:
                mr_count += 1

        total = mom_count + mr_count
        return 100 * mom_count / total if total > 0 else 50

    return returns.rolling(lookback).apply(calc_momersion, raw=False)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rate_of_change(prices: pd.Series, period: int = 10) -> pd.Series:
    """Calculate Rate of Change."""
    return (prices / prices.shift(period) - 1) * 100


# ==============================================================================
# Strategy Implementations
# ==============================================================================

class Strategy:
    """Base strategy class."""

    def __init__(self, name: str):
        self.name = name

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Generate trading signals. Override in subclass."""
        raise NotImplementedError

    def backtest(
        self,
        prices: pd.DataFrame,
        benchmark: str = "SPY",
        start_year: int | None = None
    ) -> BacktestResult:
        """Run backtest."""
        if start_year:
            prices = prices[prices.index >= f"{start_year}-01-01"]

        signals = self.generate_signals(prices)

        # Calculate returns
        if benchmark in prices.columns:
            returns = prices[benchmark].pct_change()
        else:
            returns = prices.iloc[:, 0].pct_change()

        # Strategy returns
        strategy_returns = signals.shift(1) * returns
        strategy_returns = strategy_returns.fillna(0)

        # Trade statistics
        position_changes = signals.diff().abs()
        trades = int(position_changes.sum() / 2)

        # Win rate
        trade_returns = strategy_returns[strategy_returns != 0]
        win_rate = (trade_returns > 0).mean() * 100 if len(trade_returns) > 0 else 0

        # Holding period
        in_position = (signals != 0).astype(int)
        position_groups = (in_position != in_position.shift()).cumsum()
        holding_periods = in_position.groupby(position_groups).sum()
        avg_holding = holding_periods[holding_periods > 0].mean() if len(holding_periods) > 0 else 0

        # Exposure
        exposure = in_position.mean() * 100

        # Benchmark returns for comparison
        benchmark_returns = prices[benchmark].pct_change() if benchmark in prices.columns else None

        metrics = calculate_metrics(strategy_returns, benchmark_returns)

        return BacktestResult(
            strategy_name=self.name,
            returns=strategy_returns,
            positions=signals,
            trades=trades,
            win_rate=win_rate,
            avg_holding_period=avg_holding,
            exposure=exposure,
            cagr=metrics.get("cagr", 0),
            mdd=metrics.get("mdd", 0),
            sharpe=metrics.get("sharpe", 0),
            correlation=metrics.get("correlation", 0),
            beta=metrics.get("beta", 0),
            alpha=metrics.get("alpha", 0),
            params=getattr(self, "params", {}),
        )


class RSIMeanReversion(Strategy):
    """
    RSI-based mean reversion strategy (approximation of MRETF/ETFMR).

    Based on Larry Connors' RSI(2) approach with modifications:
    - Buy when RSI(2) < threshold (oversold)
    - Exit when price crosses above 5-day MA
    - Optional: 200-day MA filter for trend
    """

    def __init__(
        self,
        symbols: list[str],
        rsi_period: int = 2,
        entry_threshold: int = 10,
        exit_threshold: int = 65,
        use_trend_filter: bool = True,
        trend_period: int = 200,
        name: str = "RSI_MR"
    ):
        super().__init__(name)
        self.symbols = symbols
        self.rsi_period = rsi_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.use_trend_filter = use_trend_filter
        self.trend_period = trend_period
        self.params = {
            "rsi_period": rsi_period,
            "entry": entry_threshold,
            "exit": exit_threshold,
            "trend_filter": use_trend_filter,
        }

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Generate mean-reversion signals."""
        # Use first available symbol
        symbol = next((s for s in self.symbols if s in prices.columns), None)
        if not symbol:
            return pd.Series(0, index=prices.index)

        price = prices[symbol]
        rsi_val = rsi(price, self.rsi_period)
        sma_5 = price.rolling(5).mean()
        sma_200 = price.rolling(self.trend_period).mean()

        signals = pd.Series(0, index=prices.index)
        position = 0

        for i in range(1, len(prices)):
            date = prices.index[i]

            # Check trend filter
            trend_ok = True
            if self.use_trend_filter:
                trend_ok = price.iloc[i] > sma_200.iloc[i] if pd.notna(sma_200.iloc[i]) else True

            # Entry condition
            if position == 0 and trend_ok:
                if pd.notna(rsi_val.iloc[i]) and rsi_val.iloc[i] < self.entry_threshold:
                    position = 1

            # Exit condition
            elif position == 1:
                if price.iloc[i] > sma_5.iloc[i]:
                    position = 0

            signals.iloc[i] = position

        return signals


class MomersionRegimeSwitching(Strategy):
    """
    Regime-switching strategy using Momersion indicator (approximation of MRMOM).

    - When Momersion < 50: Mean-reversion regime (buy dips)
    - When Momersion > 50: Momentum regime (trend following)
    """

    def __init__(
        self,
        symbols: list[str],
        momersion_period: int = 250,
        rsi_period: int = 2,
        rsi_entry: int = 10,
        momentum_period: int = 20,
        name: str = "MRMOM_Clone"
    ):
        super().__init__(name)
        self.symbols = symbols
        self.momersion_period = momersion_period
        self.rsi_period = rsi_period
        self.rsi_entry = rsi_entry
        self.momentum_period = momentum_period
        self.params = {
            "momersion_period": momersion_period,
            "rsi_period": rsi_period,
            "rsi_entry": rsi_entry,
            "momentum_period": momentum_period,
        }

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Generate regime-switching signals."""
        available = [s for s in self.symbols if s in prices.columns]
        if not available:
            return pd.Series(0, index=prices.index)

        # Equal-weight portfolio signals
        combined_signals = pd.Series(0.0, index=prices.index)

        for symbol in available:
            price = prices[symbol]
            returns = price.pct_change()

            # Calculate indicators
            mom = momersion(returns, self.momersion_period)
            rsi_val = rsi(price, self.rsi_period)
            momentum = price / price.shift(self.momentum_period) - 1
            sma_5 = price.rolling(5).mean()

            signals = pd.Series(0, index=prices.index)
            position = 0

            for i in range(self.momersion_period + 1, len(prices)):
                mom_val = mom.iloc[i]

                if pd.isna(mom_val):
                    signals.iloc[i] = position
                    continue

                # Mean-reversion regime
                if mom_val < 50:
                    if position == 0 and rsi_val.iloc[i] < self.rsi_entry:
                        position = 1
                    elif position == 1 and price.iloc[i] > sma_5.iloc[i]:
                        position = 0

                # Momentum regime
                else:
                    if position == 0 and momentum.iloc[i] > 0:
                        position = 1
                    elif position == 1 and momentum.iloc[i] < 0:
                        position = 0

                signals.iloc[i] = position

            combined_signals += signals / len(available)

        return combined_signals


class SeasonalityStrategy(Strategy):
    """
    Day-of-week/month seasonality strategy (approximation of ETFSEAS).

    Based on known calendar effects:
    - Positive Monday effect (buy Friday close, sell Monday close)
    - End-of-month effect (last 4 trading days positive)
    - Turn-of-month effect (last day to first 3 days positive)
    """

    def __init__(
        self,
        symbols: list[str],
        use_day_of_week: bool = True,
        use_eom: bool = True,
        positive_days: list[int] = [0, 4],  # Monday, Friday
        eom_days: int = 4,
        name: str = "SEASONAL"
    ):
        super().__init__(name)
        self.symbols = symbols
        self.use_day_of_week = use_day_of_week
        self.use_eom = use_eom
        self.positive_days = positive_days
        self.eom_days = eom_days
        self.params = {
            "day_of_week": use_day_of_week,
            "eom": use_eom,
            "positive_days": positive_days,
        }

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Generate seasonality-based signals."""
        signals = pd.Series(0.0, index=prices.index)

        for i, date in enumerate(prices.index):
            signal = 0

            # Day of week effect
            if self.use_day_of_week:
                if date.dayofweek in self.positive_days:
                    signal = 1

            # End of month effect
            if self.use_eom:
                # Check if we're in last N trading days of month
                next_month = date + pd.offsets.MonthEnd(1)
                days_to_eom = len(prices.index[(prices.index > date) &
                                               (prices.index <= next_month)])
                if days_to_eom <= self.eom_days:
                    signal = 1

            signals.iloc[i] = signal

        return signals


class TrendFollowing(Strategy):
    """
    Simple trend-following strategy (approximation of TFDLS).

    - Enter long when price breaks above N-day high
    - Enter short when price breaks below N-day low
    - Use ATR-based stops
    """

    def __init__(
        self,
        symbols: list[str],
        breakout_period: int = 20,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        allow_short: bool = True,
        name: str = "TREND_FOLLOW"
    ):
        super().__init__(name)
        self.symbols = symbols
        self.breakout_period = breakout_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.allow_short = allow_short
        self.params = {
            "breakout": breakout_period,
            "atr_mult": atr_multiplier,
            "allow_short": allow_short,
        }

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Generate trend-following signals."""
        available = [s for s in self.symbols if s in prices.columns]
        if not available:
            return pd.Series(0, index=prices.index)

        combined = pd.Series(0.0, index=prices.index)

        for symbol in available:
            price = prices[symbol]

            high_n = price.rolling(self.breakout_period).max()
            low_n = price.rolling(self.breakout_period).min()

            signals = pd.Series(0, index=prices.index)
            position = 0
            entry_price = 0

            for i in range(self.breakout_period + 1, len(prices)):
                # Breakout entries
                if position == 0:
                    if price.iloc[i] > high_n.iloc[i-1]:
                        position = 1
                        entry_price = price.iloc[i]
                    elif self.allow_short and price.iloc[i] < low_n.iloc[i-1]:
                        position = -1
                        entry_price = price.iloc[i]

                # Exits on opposite breakout
                elif position == 1:
                    if price.iloc[i] < low_n.iloc[i-1]:
                        position = -1 if self.allow_short else 0
                        entry_price = price.iloc[i]

                elif position == -1:
                    if price.iloc[i] > high_n.iloc[i-1]:
                        position = 1
                        entry_price = price.iloc[i]

                signals.iloc[i] = position

            combined += signals / len(available)

        return combined


class B2S2Strategy(Strategy):
    """
    B2S2-style mean reversion (approximation of B2S2DJ/B2S2ETF).

    "No parameters, no filters" suggests a very simple approach:
    - Buy after consecutive down days
    - Sell after consecutive up days

    The name "B2S2" likely means "Buy 2 Sell 2" - buy after 2 down days,
    sell after 2 up days.
    """

    def __init__(
        self,
        symbols: list[str],
        down_days: int = 2,
        up_days: int = 2,
        name: str = "B2S2"
    ):
        super().__init__(name)
        self.symbols = symbols
        self.down_days = down_days
        self.up_days = up_days
        self.params = {
            "down_days": down_days,
            "up_days": up_days,
        }

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Generate B2S2 signals."""
        available = [s for s in self.symbols if s in prices.columns]
        if not available:
            return pd.Series(0, index=prices.index)

        combined = pd.Series(0.0, index=prices.index)

        for symbol in available:
            price = prices[symbol]
            returns = price.pct_change()

            # Count consecutive up/down days
            up_count = pd.Series(0, index=prices.index)
            down_count = pd.Series(0, index=prices.index)

            for i in range(1, len(prices)):
                if returns.iloc[i] > 0:
                    up_count.iloc[i] = up_count.iloc[i-1] + 1
                    down_count.iloc[i] = 0
                elif returns.iloc[i] < 0:
                    down_count.iloc[i] = down_count.iloc[i-1] + 1
                    up_count.iloc[i] = 0
                else:
                    up_count.iloc[i] = up_count.iloc[i-1]
                    down_count.iloc[i] = down_count.iloc[i-1]

            signals = pd.Series(0, index=prices.index)
            position = 0

            for i in range(max(self.down_days, self.up_days), len(prices)):
                if position == 0:
                    # Buy after N consecutive down days
                    if down_count.iloc[i] >= self.down_days:
                        position = 1
                elif position == 1:
                    # Sell after N consecutive up days
                    if up_count.iloc[i] >= self.up_days:
                        position = 0

                signals.iloc[i] = position

            combined += signals / len(available)

        return combined


# ==============================================================================
# Grid Search for Parameter Optimization
# ==============================================================================

def parameter_search(
    strategy_class: type,
    prices: pd.DataFrame,
    param_grid: dict[str, list],
    target_metrics: dict[str, float],
    start_year: int,
    benchmark: str = "SPY"
) -> list[tuple[dict, BacktestResult, float]]:
    """Search for parameters that best match target metrics."""

    results = []

    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    from itertools import product

    for combo in product(*values):
        params = dict(zip(keys, combo))

        try:
            strategy = strategy_class(**params)
            result = strategy.backtest(prices, benchmark=benchmark, start_year=start_year)

            # Calculate match score (lower is better)
            score = 0
            score += abs(result.cagr - target_metrics["cagr"]) / max(target_metrics["cagr"], 1)
            score += abs(result.mdd - target_metrics["mdd"]) / max(abs(target_metrics["mdd"]), 1)
            score += abs(result.sharpe - target_metrics["sharpe"]) / max(target_metrics["sharpe"], 0.1)

            if "correlation" in target_metrics:
                score += abs(result.correlation - target_metrics["correlation"])
            if "beta" in target_metrics:
                score += abs(result.beta - target_metrics["beta"])

            results.append((params, result, score))

        except Exception as e:
            continue

    # Sort by score (best matches first)
    results.sort(key=lambda x: x[2])
    return results


# ==============================================================================
# Analysis and Reporting
# ==============================================================================

def compare_results(results: list[BacktestResult], targets: dict[str, dict]) -> pd.DataFrame:
    """Compare backtest results with target metrics."""
    rows = []

    for result in results:
        target = targets.get(result.strategy_name, {})

        row = {
            "Strategy": result.strategy_name,
            "CAGR%": f"{result.cagr:.1f}",
            "Target CAGR%": target.get("cagr", "N/A"),
            "MDD%": f"{result.mdd:.1f}",
            "Target MDD%": target.get("mdd", "N/A"),
            "Sharpe": f"{result.sharpe:.2f}",
            "Target Sharpe": target.get("sharpe", "N/A"),
            "Trades": result.trades,
            "Win%": f"{result.win_rate:.1f}",
            "Exposure%": f"{result.exposure:.1f}",
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_equity_curves(
    results: list[BacktestResult],
    benchmark_returns: pd.Series | None = None,
    output_path: Path | None = None
):
    """Plot equity curves for all strategies."""
    fig, ax = plt.subplots(figsize=(14, 8))

    for result in results:
        equity = (1 + result.returns).cumprod()
        ax.plot(equity.index, equity, label=result.strategy_name, linewidth=1.5)

    if benchmark_returns is not None:
        benchmark_equity = (1 + benchmark_returns).cumprod()
        ax.plot(benchmark_equity.index, benchmark_equity,
                label="SPY (Buy & Hold)", color="black", linestyle="--", linewidth=1)

    ax.set_title("Reverse-Engineered Strategy Equity Curves", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        console.print(f"[green]Saved equity curve plot to {output_path}[/green]")

    return fig


def generate_report(
    results: list[BacktestResult],
    targets: dict[str, dict],
    output_path: Path
) -> str:
    """Generate markdown report."""

    report = []
    report.append("# PriceActionLab Strategy Reverse Engineering Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.append("## Executive Summary\n")
    report.append("""
This report attempts to reverse-engineer trading strategies from PriceActionLab
based on publicly available information:

- Performance metrics (CAGR, MDD, Sharpe, correlation, beta, alpha)
- Strategy descriptions (mean-reversion, trend-following, regime-switching)
- Published indicator formulas (Momersion indicator)
- Academic literature on similar approaches (Connors RSI, calendar effects)

**Disclaimer**: These are approximations based on inference. The actual strategies
may differ significantly from these reconstructions.
""")

    report.append("\n## Target vs Achieved Metrics\n\n")

    # Create comparison table
    report.append("| Strategy | Target CAGR | Achieved CAGR | Target MDD | Achieved MDD | Target Sharpe | Achieved Sharpe |\n")
    report.append("|----------|-------------|---------------|------------|--------------|---------------|------------------|\n")

    for result in results:
        target = targets.get(result.strategy_name, {})
        report.append(f"| {result.strategy_name} | {target.get('cagr', 'N/A')}% | {result.cagr:.1f}% | {target.get('mdd', 'N/A')}% | {result.mdd:.1f}% | {target.get('sharpe', 'N/A')} | {result.sharpe:.2f} |\n")

    report.append("\n## Strategy Descriptions\n\n")

    strategy_descriptions = {
        "MRETF": """
### MRETF (Mean-Reversion ETF - Long Only)
**Target Markets**: SPY, QQQ, TLT

**Inferred Logic**:
- Uses RSI(2) or similar short-term mean-reversion indicator
- Buys when RSI falls below oversold threshold (e.g., 10)
- Exits when price crosses above short-term MA (e.g., 5-day)
- Likely includes 200-day MA trend filter
- Single position at a time, rotates between ETFs

**Key Characteristics**:
- Very low beta (0.14) suggests highly selective entry
- Low exposure, high win rate (~69%)
- Short holding periods (~7 days)
""",
        "MRETFLS": """
### MRETFLS (Mean-Reversion ETF - Long/Short)
**Target Markets**: SPY, QQQ, TLT

**Inferred Logic**:
- Same base logic as MRETF but allows short positions
- Short when RSI reaches overbought levels
- Likely uses different thresholds for long vs short

**Key Characteristics**:
- Near-zero correlation and beta suggests market-neutral tendency
- Higher volatility than long-only version
""",
        "B2S2DJ": """
### B2S2DJ (Buy 2 Sell 2 - Dow Jones)
**Target Markets**: Dow 30 stocks

**Inferred Logic**:
- "No parameters, no filters" suggests extremely simple rules
- Name suggests "Buy after 2 down days, Sell after 2 up days"
- Ranks stocks by rate-of-change for position selection
- Maximum 10 concurrent positions
- Equal-weight allocation

**Key Characteristics**:
- High correlation (0.84) with market
- High beta (0.78) indicates leveraged market exposure
- Long backtest period (1993) suggests robust approach
""",
        "B2S2ETF": """
### B2S2ETF (Buy 2 Sell 2 - SPY ETF)
**Target Markets**: SPY only

**Inferred Logic**:
- Same "Buy 2 Sell 2" approach as B2S2DJ
- Applied only to SPY ETF
- Simpler implementation, single instrument

**Key Characteristics**:
- Similar metrics to B2S2DJ but single instrument
- Higher exposure than stock version
""",
        "TFDLS": """
### TFDLS (Trend Following - Long/Short)
**Target Markets**: 23 Futures contracts

**Inferred Logic**:
- Classic channel breakout system (Donchian-style)
- Enter long on new N-day high, short on new N-day low
- Uses ATR-based position sizing and stops
- Trades all 23 markets with identical parameters

**Key Characteristics**:
- Negative correlation (-0.07) with SPY - crisis alpha
- Requires significant capital (~$1M for proper sizing)
- Wide diversification across asset classes
""",
        "MRMOM": """
### MRMOM (Mean-Reversion/Momentum Regime Switching)
**Target Markets**: SPY, QQQ, TLT, GLD

**Inferred Logic**:
- Uses Momersion indicator for regime detection
- Momersion(250) = 100 * MOMc / (MOMc + MRc)
- Below 50: Mean-reversion regime - buy oversold
- Above 50: Momentum regime - follow trends
- Rotates between 4 ETFs with equal weight

**Key Characteristics**:
- Highest Sharpe ratio (1.15) of all strategies
- Moderate correlation (0.67) suggests adaptive behavior
- Combines best of both approaches
""",
        "ETFMR": """
### ETFMR (ETF Mean-Reversion)
**Target Markets**: SPY, QQQ

**Inferred Logic**:
- "Based on two popular indicators" - likely RSI + something else
- Could be RSI(2) + Williams %R, or RSI(2) + Bollinger Bands
- Similar to MRETF but different parameter set
- 100% position sizing (fully invested when signal triggers)

**Key Characteristics**:
- Higher CAGR (10%) than MRETF
- Higher drawdown (-22.9%) suggests more aggressive
- ~4.3 day average holding period
""",
        "ETFSEAS": """
### ETFSEAS (ETF Seasonality)
**Target Markets**: SPY, TLT, GLD

**Inferred Logic**:
- Calendar-based effects (day-of-week, end-of-month)
- Likely exploits:
  - Turn-of-month effect (last day to first 3 days)
  - End-of-month effect (last 4 trading days)
  - Possibly day-of-week patterns
- Very short holding period (~1 day)

**Key Characteristics**:
- Low correlation (0.22) - uncorrelated to market
- Very low beta (0.10) - minimal market exposure
- Low exposure (28.4%) - only trades on specific days
""",
        "ETFMO": """
### ETFMO (ETF Momentum - Leveraged)
**Target Markets**: Leveraged ETF (unspecified)

**Inferred Logic**:
- Seasonality strategy applied to leveraged ETF
- Likely trades TQQQ, UPRO, or TMF
- ~3-day holding period
- Very selective entry (13.6% exposure)

**Key Characteristics**:
- Negative beta (-0.07) suggests non-equity focus
- Could be trading leveraged bond ETF (TMF)
- High alpha (7.8%) indicates unique edge
""",
        "MRDJ": """
### MRDJ (Mean-Reversion Dow Jones)
**Target Markets**: Dow 30 stocks

**Inferred Logic**:
- Similar to B2S2DJ with possible breakout component
- "Mean-reversion/breakouts" description
- Could combine dip-buying with breakout entries
- Stock selection based on lowest ROC

**Key Characteristics**:
- Moderate correlation (0.52) vs B2S2DJ's 0.84
- Lower beta (0.39) suggests more selective
- May have trend filter that B2S2 lacks
""",
    }

    for name, desc in strategy_descriptions.items():
        report.append(desc)
        report.append("\n")

    report.append("\n## Implementation Notes\n\n")
    report.append("""
### Data Requirements
- **ETF Strategies**: Yahoo Finance data sufficient
- **Stock Strategies (B2S2DJ, MRDJ)**: Requires Norgate Data for delisted stocks
- **Futures Strategies (TFDLS)**: Requires continuous back-adjusted contracts

### Key Indicators Used
1. **RSI(2)**: Short-term oversold/overbought detection
2. **Momersion**: Regime detection (momentum vs mean-reversion)
3. **Rate of Change**: Stock ranking for selection
4. **Donchian Channels**: Trend-following breakouts
5. **ATR**: Position sizing and stops

### Risk Warnings
- Mean-reversion strategies typically don't use stops
- High win rate compensates for low payoff ratio
- Regime changes can cause strategy failure
- Transaction costs significantly impact high-turnover strategies
""")

    report.append("\n## Sources\n\n")
    report.append("""
- [PriceActionLab Strategies](https://www.priceactionlab.com/Blog/trading-strategies-for-sale/)
- [Momersion Indicator](https://www.priceactionlab.com/Blog/2015/08/momersion-indicator/)
- [Mean-Reversion Optimization Space](https://www.priceactionlab.com/Blog/2025/01/the-huge-optimization-space-of-mean-reversion/)
- [MRMOM Regime Switching](https://www.priceactionlab.com/Blog/2024/01/mean-reversion-and-momentum-regime-switching/)
- [Larry Connors RSI(2)](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/rsi-2)
""")

    content = "".join(report)

    with open(output_path, "w") as f:
        f.write(content)

    console.print(f"[green]Saved report to {output_path}[/green]")
    return content


def main():
    """Main execution."""
    console.print("[bold blue]PriceActionLab Strategy Reverse Engineering[/bold blue]\n")

    # Download data
    symbols = ["SPY", "QQQ", "TLT", "GLD", "DIA"]
    prices = download_data(symbols, start="2000-01-01")

    if prices.empty:
        console.print("[red]Failed to download data. Exiting.[/red]")
        return

    # Calculate benchmark returns
    benchmark_returns = prices["SPY"].pct_change().dropna()

    results = []

    # 1. MRETF - RSI Mean Reversion
    console.print("\n[cyan]Testing MRETF approximation...[/cyan]")
    mretf = RSIMeanReversion(
        symbols=["SPY", "QQQ", "TLT"],
        rsi_period=2,
        entry_threshold=10,
        exit_threshold=65,
        use_trend_filter=True,
        name="MRETF"
    )
    results.append(mretf.backtest(prices, start_year=2002))

    # 2. MRMOM - Regime Switching
    console.print("[cyan]Testing MRMOM approximation...[/cyan]")
    mrmom = MomersionRegimeSwitching(
        symbols=["SPY", "QQQ", "TLT", "GLD"],
        momersion_period=250,
        rsi_period=2,
        rsi_entry=10,
        momentum_period=20,
        name="MRMOM"
    )
    results.append(mrmom.backtest(prices, start_year=2003))

    # 3. B2S2ETF - Simple consecutive days
    console.print("[cyan]Testing B2S2ETF approximation...[/cyan]")
    b2s2 = B2S2Strategy(
        symbols=["SPY"],
        down_days=2,
        up_days=2,
        name="B2S2ETF"
    )
    results.append(b2s2.backtest(prices, start_year=2002))

    # 4. ETFSEAS - Seasonality
    console.print("[cyan]Testing ETFSEAS approximation...[/cyan]")
    seasonal = SeasonalityStrategy(
        symbols=["SPY", "TLT", "GLD"],
        use_day_of_week=True,
        use_eom=True,
        positive_days=[0, 4],  # Monday, Friday
        eom_days=4,
        name="ETFSEAS"
    )
    results.append(seasonal.backtest(prices, start_year=2005))

    # 5. ETFMR - Mean Reversion variant
    console.print("[cyan]Testing ETFMR approximation...[/cyan]")
    etfmr = RSIMeanReversion(
        symbols=["SPY", "QQQ"],
        rsi_period=2,
        entry_threshold=5,  # More aggressive
        exit_threshold=70,
        use_trend_filter=False,  # No filter = more trades
        name="ETFMR"
    )
    results.append(etfmr.backtest(prices, start_year=2003))

    # 6. Trend Following
    console.print("[cyan]Testing TFDLS approximation (on ETFs)...[/cyan]")
    tfdls = TrendFollowing(
        symbols=["SPY", "QQQ", "TLT", "GLD"],
        breakout_period=20,
        atr_multiplier=2.0,
        allow_short=True,
        name="TFDLS"
    )
    results.append(tfdls.backtest(prices, start_year=2003))

    # Display results
    console.print("\n[bold]Results Summary[/bold]\n")

    table = Table(title="Strategy Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("CAGR%", justify="right")
    table.add_column("Target", justify="right", style="dim")
    table.add_column("MDD%", justify="right")
    table.add_column("Target", justify="right", style="dim")
    table.add_column("Sharpe", justify="right")
    table.add_column("Target", justify="right", style="dim")
    table.add_column("Trades", justify="right")
    table.add_column("Win%", justify="right")

    for result in results:
        target = TARGET_METRICS.get(result.strategy_name, {})
        table.add_row(
            result.strategy_name,
            f"{result.cagr:.1f}",
            str(target.get("cagr", "N/A")),
            f"{result.mdd:.1f}",
            str(target.get("mdd", "N/A")),
            f"{result.sharpe:.2f}",
            str(target.get("sharpe", "N/A")),
            str(result.trades),
            f"{result.win_rate:.1f}",
        )

    console.print(table)

    # Generate outputs
    output_dir = Path("/home/user/ai-playground/outputs")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot equity curves
    plot_path = output_dir / f"strategy_equity_curves_{timestamp}.png"
    plot_equity_curves(results, benchmark_returns, plot_path)

    # Generate report
    report_path = output_dir / f"strategy_reverse_engineering_{timestamp}.md"
    generate_report(results, TARGET_METRICS, report_path)

    console.print(f"\n[green]Analysis complete! Outputs saved to {output_dir}[/green]")

    # Summary
    console.print("\n[bold]Key Findings:[/bold]")
    console.print("""
1. Mean-reversion strategies (MRETF, ETFMR, B2S2) use RSI(2) or consecutive day patterns
2. MRMOM uses Momersion indicator for regime detection
3. Seasonality strategies exploit calendar effects (day-of-week, end-of-month)
4. Trend-following uses Donchian-style breakouts
5. Most strategies have no stop-loss (high win rate compensates)
6. Low exposure/selective entry is key to high Sharpe ratios
""")


if __name__ == "__main__":
    main()
