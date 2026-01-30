# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "yfinance>=0.2.30",
#     "matplotlib>=3.7",
#     "seaborn>=0.12",
#     "scipy>=1.11",
#     "rich>=13.0",
# ]
# ///
"""
Replication of: "Interpretable Hypothesis-Driven Trading: A Rigorous
Walk-Forward Validation Framework for Market Microstructure Signals"

Gagan Deep, Akash Deep, William Lamptey (Texas Tech University, Dec 2025)
arXiv: 2512.12924v1

This script replicates the paper's core methodology:
  - 100 US equities (2015-2024) selected by GICS sector, volume, market cap
  - Market microstructure features: volume imbalance, volume ratio, price efficiency
  - 5 interpretable hypothesis types with confidence scores
  - RL agent with epsilon-greedy policy for trade selection
  - Walk-forward validation: 252-day train, 63-day test, 34 folds
  - Realistic execution costs (commissions, slippage, position limits)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from scipy import stats

warnings.filterwarnings("ignore")
console = Console()

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "replicate-research-analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Universe Selection ──────────────────────────────────────────────────────
# Paper: 100 US equities, 10 per GICS sector, selected by trading volume
# We use a representative universe matching paper criteria
UNIVERSE: dict[str, list[str]] = {
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HES"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DOW", "DD", "PPG"],
    "Industrials": ["UNP", "HON", "UPS", "CAT", "GE", "RTX", "DE", "BA", "LMT", "MMM"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG"],
    "Consumer Staples": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "KHC"],
    "Health Care": ["UNH", "JNJ", "LLY", "PFE", "ABT", "TMO", "ABBV", "MRK", "DHR", "BMY"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "C", "AXP", "USB"],
    "Information Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "INTC", "TXN"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC"],
}

ALL_TICKERS = [t for sector_tickers in UNIVERSE.values() for t in sector_tickers]

# ── Configuration ───────────────────────────────────────────────────────────
@dataclass
class Config:
    """Paper parameters (Section 3)."""
    start_date: str = "2015-01-02"
    end_date: str = "2024-10-31"
    train_window: int = 252        # W = 252 trading days
    test_window: int = 63          # H = 63 trading days (one quarter)
    step_size: int = 63            # Δ = 63 days
    commission: float = 1.0        # $1 fixed per trade
    slippage_bps: float = 5.0      # 5 basis points
    max_positions: int = 5
    max_position_pct: float = 0.20 # 20% per position
    initial_capital: float = 100_000.0
    epsilon_train: float = 0.7
    epsilon_test: float = 0.1
    vol_imbalance_window: int = 5
    vol_ratio_window: int = 20
    price_eff_window: int = 10
    vol_regime_threshold: float = 0.02


# ── Data Fetching ───────────────────────────────────────────────────────────
def _safe_float(val: Any) -> float:
    """Extract a scalar float from a value that may be a Series (duplicate index)."""
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    return float(val)


def fetch_market_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data from Yahoo Finance for all tickers."""
    console.print(f"[bold cyan]Fetching data for {len(tickers)} tickers...[/bold cyan]")
    data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    # Fetch in batches to avoid rate limits
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_str = " ".join(batch)
        try:
            df = yf.download(batch_str, start=start, end=end, group_by="ticker",
                             auto_adjust=True, progress=False, threads=True)
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        ticker_df = df.copy()
                    else:
                        ticker_df = df[ticker].copy()
                    ticker_df = ticker_df.dropna(subset=["Close"])
                    if len(ticker_df) > 500:
                        data[ticker] = ticker_df
                    else:
                        failed.append(ticker)
                except (KeyError, TypeError):
                    failed.append(ticker)
        except Exception as e:
            console.print(f"[red]Batch fetch error: {e}[/red]")
            failed.extend(batch)

    console.print(f"[green]Fetched {len(data)} tickers successfully[/green]")
    if failed:
        console.print(f"[yellow]Failed/insufficient data: {failed}[/yellow]")
    return data


# ── Feature Engineering ─────────────────────────────────────────────────────
class FeatureType(Enum):
    VOLUME_IMBALANCE = "volume_imbalance"
    VOLUME_RATIO = "volume_ratio"
    PRICE_EFFICIENCY = "price_efficiency"


def compute_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Compute market microstructure features per the paper (Section 4.1).

    Volume Imbalance: (Σ buy_vol - Σ sell_vol) / Σ total_vol over 5 days
    Volume Ratio: current_vol / 20-day avg volume
    Price Efficiency: |Σ returns| / Σ |returns| over 10 days
    """
    feat = pd.DataFrame(index=df.index)

    # Daily returns
    feat["return"] = df["Close"].pct_change()

    # Classify volume as buy/sell using close vs open direction
    feat["buy_volume"] = np.where(df["Close"] >= df["Open"], df["Volume"], 0)
    feat["sell_volume"] = np.where(df["Close"] < df["Open"], df["Volume"], 0)

    # Volume Imbalance (5-day rolling)
    buy_sum = feat["buy_volume"].rolling(cfg.vol_imbalance_window).sum()
    sell_sum = feat["sell_volume"].rolling(cfg.vol_imbalance_window).sum()
    total_sum = df["Volume"].rolling(cfg.vol_imbalance_window).sum()
    feat["volume_imbalance"] = (buy_sum - sell_sum) / total_sum.replace(0, np.nan)

    # Volume Ratio (current / 20-day average)
    avg_vol = df["Volume"].rolling(cfg.vol_ratio_window).mean()
    feat["volume_ratio"] = df["Volume"] / avg_vol.replace(0, np.nan)

    # Price Efficiency (10-day)
    ret = feat["return"]
    sum_returns = ret.rolling(cfg.price_eff_window).sum()
    sum_abs_returns = ret.abs().rolling(cfg.price_eff_window).sum()
    feat["price_efficiency"] = sum_returns.abs() / sum_abs_returns.replace(0, np.nan)

    # Additional derived features used by hypotheses
    feat["return_20d"] = df["Close"].pct_change(20)
    feat["return_5d"] = df["Close"].pct_change(5)
    feat["return_60d"] = df["Close"].pct_change(60)
    feat["volatility_20d"] = ret.rolling(20).std()
    feat["high_252d"] = df["High"].rolling(252).max()
    feat["close"] = df["Close"]
    feat["volume"] = df["Volume"]

    # Distance from 252-day high
    feat["dist_from_high"] = (df["Close"] - feat["high_252d"]) / feat["high_252d"]

    return feat.dropna()


# ── Hypothesis System ───────────────────────────────────────────────────────
class HypothesisType(Enum):
    INSTITUTIONAL_ACCUMULATION = "institutional_accumulation"
    FLOW_MOMENTUM = "flow_momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    RANGE_BOUND_VALUE = "range_bound_value"


@dataclass
class Hypothesis:
    """Trading hypothesis h = (s, a, θ, ℓ, c, x, r*, δ*)"""
    ticker: str                      # s: security identifier
    action: str                      # a: buy/sell
    hypothesis_type: HypothesisType  # θ
    explanation: str                  # ℓ: natural language
    confidence: float                # c ∈ [0,1]
    features: dict[str, float]       # x: feature vector
    target_return: float             # r*
    stop_loss: float                 # δ*
    date: pd.Timestamp | None = None


def generate_hypotheses(ticker: str, features: pd.DataFrame, date: pd.Timestamp,
                        cfg: Config) -> list[Hypothesis]:
    """Generate interpretable hypotheses for a given ticker and date.

    Paper Section 4.2: Five hypothesis types with specific conditions.
    """
    if date not in features.index:
        return []

    row = features.loc[date]
    hypotheses: list[Hypothesis] = []

    vi = row.get("volume_imbalance", 0)
    vr = row.get("volume_ratio", 0)
    pe = row.get("price_efficiency", 0)
    r20 = row.get("return_20d", 0)
    r5 = row.get("return_5d", 0)
    r60 = row.get("return_60d", 0)
    vol = row.get("volatility_20d", 0)
    dist_high = row.get("dist_from_high", -1)

    feat_dict = {
        "volume_imbalance": float(vi), "volume_ratio": float(vr),
        "price_efficiency": float(pe), "return_20d": float(r20),
        "return_5d": float(r5), "return_60d": float(r60),
        "volatility_20d": float(vol), "dist_from_high": float(dist_high),
    }

    # 1. Institutional Accumulation: VI>30%, VR>1.5, |R20|<10%
    if vi > 0.30 and vr > 1.5 and abs(r20) < 0.10:
        hypotheses.append(Hypothesis(
            ticker=ticker, action="buy",
            hypothesis_type=HypothesisType.INSTITUTIONAL_ACCUMULATION,
            explanation=(f"Institutional accumulation detected in {ticker}: "
                         f"volume imbalance {vi:.1%} with volume ratio {vr:.1f}x "
                         f"while price remains range-bound ({r20:.1%} over 20d)."),
            confidence=0.75, features=feat_dict,
            target_return=0.08, stop_loss=0.04, date=date,
        ))

    # 2. Flow Momentum: R20>10%, VI>20%, PE>50%
    if r20 > 0.10 and vi > 0.20 and pe > 0.50:
        hypotheses.append(Hypothesis(
            ticker=ticker, action="buy",
            hypothesis_type=HypothesisType.FLOW_MOMENTUM,
            explanation=(f"Flow momentum in {ticker}: strong 20d return {r20:.1%} "
                         f"with volume imbalance {vi:.1%} and high price efficiency {pe:.1%}."),
            confidence=0.70, features=feat_dict,
            target_return=0.10, stop_loss=0.05, date=date,
        ))

    # 3. Mean Reversion: oversold (R20<-10%), stable volatility (<2%)
    if r20 < -0.10 and vol < cfg.vol_regime_threshold and vi > 0:
        hypotheses.append(Hypothesis(
            ticker=ticker, action="buy",
            hypothesis_type=HypothesisType.MEAN_REVERSION,
            explanation=(f"Mean reversion opportunity in {ticker}: oversold {r20:.1%} "
                         f"in low-volatility regime (σ={vol:.3f}) with positive flow."),
            confidence=0.65, features=feat_dict,
            target_return=0.05, stop_loss=0.03, date=date,
        ))

    # 4. Breakout: near 252d high (>-5%), VR>1.3, positive momentum
    if dist_high > -0.05 and vr > 1.3 and r5 > 0 and r20 > 0:
        hypotheses.append(Hypothesis(
            ticker=ticker, action="buy",
            hypothesis_type=HypothesisType.BREAKOUT,
            explanation=(f"Breakout in {ticker}: within {abs(dist_high):.1%} of 252d high "
                         f"with volume expansion {vr:.1f}x and positive momentum."),
            confidence=0.68, features=feat_dict,
            target_return=0.07, stop_loss=0.04, date=date,
        ))

    # 5. Range-Bound Value: accumulation in stable market
    if abs(r60) < 0.05 and vi > 0.15 and vr > 1.0 and vol < cfg.vol_regime_threshold:
        hypotheses.append(Hypothesis(
            ticker=ticker, action="buy",
            hypothesis_type=HypothesisType.RANGE_BOUND_VALUE,
            explanation=(f"Range-bound accumulation in {ticker}: flat 60d return {r60:.1%} "
                         f"with accumulation (VI={vi:.1%}) in stable regime."),
            confidence=0.60, features=feat_dict,
            target_return=0.05, stop_loss=0.03, date=date,
        ))

    return hypotheses


# ── RL Agent ────────────────────────────────────────────────────────────────
@dataclass
class TradeRecord:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    hypothesis_type: str = ""
    confidence: float = 0.0
    target_return: float = 0.0
    stop_loss: float = 0.0
    pnl: float = 0.0
    return_pct: float = 0.0
    explanation: str = ""


@dataclass
class RLAgent:
    """
    Epsilon-greedy RL agent (Paper Section 4.3).

    Policy: π(h|A_t, ε) = explore with prob ε, else execute if win_rate > τ(c)
    Threshold: τ(c) = 0.45 + (1-c) × 0.10
    """
    epsilon: float = 0.7
    win_rates: dict[str, list[float]] = field(default_factory=dict)

    def _adaptive_threshold(self, confidence: float) -> float:
        """τ(c) = 0.45 + (1-c) × 0.10"""
        return 0.45 + (1 - confidence) * 0.10

    def _get_win_rate(self, h_type: str) -> float:
        results = self.win_rates.get(h_type, [])
        if len(results) < 3:
            return 0.50  # default prior
        return sum(1 for r in results if r > 0) / len(results)

    def should_execute(self, hypothesis: Hypothesis) -> bool:
        """Decide whether to execute a hypothesis."""
        if np.random.random() < self.epsilon:
            return True  # explore

        h_type = hypothesis.hypothesis_type.value
        win_rate = self._get_win_rate(h_type)
        threshold = self._adaptive_threshold(hypothesis.confidence)
        return win_rate > threshold

    def update(self, h_type: str, pnl: float) -> None:
        """Update win rate tracking after trade completes."""
        if h_type not in self.win_rates:
            self.win_rates[h_type] = []
        self.win_rates[h_type].append(pnl)


# ── Portfolio / Execution Engine ────────────────────────────────────────────
@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    hypothesis: Hypothesis
    max_hold_days: int = 63  # one quarter max hold


class Portfolio:
    """Portfolio with realistic execution costs and position limits."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cash = cfg.initial_capital
        self.positions: list[Position] = []
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []

    @property
    def num_positions(self) -> int:
        return len(self.positions)

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        slip = price * (self.cfg.slippage_bps / 10_000)
        return price + slip if is_buy else price - slip

    def total_equity(self, prices: dict[str, float]) -> float:
        eq = self.cash
        for pos in self.positions:
            px = prices.get(pos.ticker, pos.entry_price)
            eq += pos.shares * px
        return eq

    def open_position(self, hypothesis: Hypothesis, price: float, date: pd.Timestamp) -> bool:
        """Open a new position if constraints are met."""
        if self.num_positions >= self.cfg.max_positions:
            return False
        if any(p.ticker == hypothesis.ticker for p in self.positions):
            return False  # no duplicate positions

        alloc = self.cash * self.cfg.max_position_pct
        exec_price = self._apply_slippage(price, is_buy=True)
        shares = int(alloc / exec_price)
        if shares <= 0:
            return False

        cost = shares * exec_price + self.cfg.commission
        if cost > self.cash:
            return False

        self.cash -= cost
        self.positions.append(Position(
            ticker=hypothesis.ticker, entry_date=date,
            entry_price=exec_price, shares=shares,
            hypothesis=hypothesis,
        ))
        return True

    def close_position(self, pos: Position, price: float, date: pd.Timestamp) -> TradeRecord:
        """Close a position and record the trade."""
        exec_price = self._apply_slippage(price, is_buy=False)
        proceeds = pos.shares * exec_price - self.cfg.commission
        self.cash += proceeds

        pnl = proceeds - (pos.shares * pos.entry_price + self.cfg.commission)
        ret = (exec_price - pos.entry_price) / pos.entry_price

        trade = TradeRecord(
            ticker=pos.ticker, entry_date=pos.entry_date, exit_date=date,
            entry_price=pos.entry_price, exit_price=exec_price,
            hypothesis_type=pos.hypothesis.hypothesis_type.value,
            confidence=pos.hypothesis.confidence,
            target_return=pos.hypothesis.target_return,
            stop_loss=pos.hypothesis.stop_loss,
            pnl=pnl, return_pct=ret,
            explanation=pos.hypothesis.explanation,
        )
        self.trades.append(trade)
        self.positions = [p for p in self.positions if p.ticker != pos.ticker]
        return trade

    def check_exits(self, prices: dict[str, float], date: pd.Timestamp,
                    days_held: dict[str, int]) -> list[TradeRecord]:
        """Check stop-loss, target, and max hold exits."""
        closed: list[TradeRecord] = []
        for pos in list(self.positions):
            px = prices.get(pos.ticker)
            if px is None:
                continue
            ret = (px - pos.entry_price) / pos.entry_price
            held = days_held.get(pos.ticker, 0)

            # Stop-loss
            if ret < -pos.hypothesis.stop_loss:
                closed.append(self.close_position(pos, px, date))
            # Target hit
            elif ret > pos.hypothesis.target_return:
                closed.append(self.close_position(pos, px, date))
            # Max hold exceeded
            elif held >= pos.max_hold_days:
                closed.append(self.close_position(pos, px, date))

        return closed


# ── Walk-Forward Validation ─────────────────────────────────────────────────
@dataclass
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    quarterly_return: float
    num_trades: int
    win_rate: float
    sharpe: float
    max_drawdown: float
    hypothesis_counts: dict[str, int]
    trades: list[TradeRecord]
    equity_curve: list[tuple[pd.Timestamp, float]]


def run_fold(fold_id: int, all_features: dict[str, pd.DataFrame],
             all_data: dict[str, pd.DataFrame],
             train_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex,
             cfg: Config) -> FoldResult:
    """Run a single walk-forward fold."""
    # Training phase: build RL agent
    agent = RLAgent(epsilon=cfg.epsilon_train)

    # Quick training pass to calibrate agent win rates
    for ticker, features in all_features.items():
        price_data = all_data.get(ticker)
        if price_data is None:
            continue
        train_feat = features[features.index.isin(train_dates)]
        if len(train_feat) < 20:
            continue

        for date in train_feat.index[::5]:  # sample every 5 days for speed
            hypotheses = generate_hypotheses(ticker, features, date, cfg)
            for h in hypotheses:
                # Look forward to see outcome (allowed during training)
                loc = features.index.get_loc(date)
                future_idx = loc if isinstance(loc, int) else (loc.start if isinstance(loc, slice) else int(np.argmax(loc)))
                if future_idx + 20 >= len(features):
                    continue
                future_price = _safe_float(price_data.iloc[min(future_idx + 20, len(price_data) - 1)]["Close"])
                current_price = _safe_float(price_data.loc[date, "Close"]) if date in price_data.index else None
                if current_price is None or current_price == 0:
                    continue
                outcome = (future_price - current_price) / current_price
                agent.update(h.hypothesis_type.value, outcome)

    # Testing phase: strict out-of-sample
    agent.epsilon = cfg.epsilon_test
    portfolio = Portfolio(cfg)
    days_held: dict[str, int] = {}
    h_counts: dict[str, int] = {}

    for date in test_dates:
        # Get current prices
        current_prices: dict[str, float] = {}
        for ticker, df in all_data.items():
            if date in df.index:
                current_prices[ticker] = _safe_float(df.loc[date, "Close"])

        # Update days held
        for pos in portfolio.positions:
            days_held[pos.ticker] = days_held.get(pos.ticker, 0) + 1

        # Check exits
        portfolio.check_exits(current_prices, date, days_held)

        # Remove closed tickers from days_held
        open_tickers = {p.ticker for p in portfolio.positions}
        days_held = {k: v for k, v in days_held.items() if k in open_tickers}

        # Generate and evaluate new hypotheses
        all_hypotheses: list[tuple[Hypothesis, float]] = []
        for ticker, features in all_features.items():
            if date not in features.index:
                continue
            px = current_prices.get(ticker)
            if px is None:
                continue
            hypotheses = generate_hypotheses(ticker, features, date, cfg)
            for h in hypotheses:
                if agent.should_execute(h):
                    all_hypotheses.append((h, px))

        # Sort by confidence, take top opportunities
        all_hypotheses.sort(key=lambda x: x[0].confidence, reverse=True)
        for h, px in all_hypotheses[:cfg.max_positions - portfolio.num_positions]:
            if portfolio.open_position(h, px, date):
                h_type = h.hypothesis_type.value
                h_counts[h_type] = h_counts.get(h_type, 0) + 1

        # Record equity
        eq = portfolio.total_equity(current_prices)
        portfolio.equity_curve.append((date, eq))

    # Close remaining positions at end of test period
    last_date = test_dates[-1]
    for pos in list(portfolio.positions):
        if last_date in all_data.get(pos.ticker, pd.DataFrame()).index:
            px = _safe_float(all_data[pos.ticker].loc[last_date, "Close"])
            portfolio.close_position(pos, px, last_date)

    # Calculate fold metrics
    trades = portfolio.trades
    if not trades:
        return FoldResult(
            fold_id=fold_id, train_start=train_dates[0], train_end=train_dates[-1],
            test_start=test_dates[0], test_end=test_dates[-1],
            quarterly_return=0.0, num_trades=0, win_rate=0.0, sharpe=0.0,
            max_drawdown=0.0, hypothesis_counts=h_counts, trades=[],
            equity_curve=portfolio.equity_curve,
        )

    eq_values = [e[1] for e in portfolio.equity_curve]
    eq_returns = pd.Series(eq_values).pct_change().dropna()

    total_return = (eq_values[-1] / cfg.initial_capital - 1) if eq_values else 0
    win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades) if trades else 0

    # Sharpe (annualized from daily)
    if len(eq_returns) > 1 and eq_returns.std() > 0:
        sharpe = (eq_returns.mean() / eq_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    cum = pd.Series(eq_values)
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    return FoldResult(
        fold_id=fold_id, train_start=train_dates[0], train_end=train_dates[-1],
        test_start=test_dates[0], test_end=test_dates[-1],
        quarterly_return=total_return, num_trades=len(trades),
        win_rate=win_rate, sharpe=sharpe, max_drawdown=max_dd,
        hypothesis_counts=h_counts, trades=trades,
        equity_curve=portfolio.equity_curve,
    )


def walk_forward_validation(all_features: dict[str, pd.DataFrame],
                            all_data: dict[str, pd.DataFrame],
                            cfg: Config) -> list[FoldResult]:
    """Run full walk-forward validation across all folds."""
    # Get the union of all trading dates
    all_dates_set: set[pd.Timestamp] = set()
    for df in all_data.values():
        all_dates_set.update(df.index)
    all_dates = sorted(all_dates_set)
    all_dates_idx = pd.DatetimeIndex(all_dates)

    results: list[FoldResult] = []
    fold_id = 0
    start_idx = 0

    console.print(f"\n[bold cyan]Running walk-forward validation...[/bold cyan]")
    console.print(f"Total trading days: {len(all_dates)}, "
                  f"Train: {cfg.train_window}, Test: {cfg.test_window}, Step: {cfg.step_size}")

    while start_idx + cfg.train_window + cfg.test_window <= len(all_dates):
        train_start = start_idx
        train_end = start_idx + cfg.train_window
        test_start = train_end
        test_end = min(test_start + cfg.test_window, len(all_dates))

        train_dates = all_dates_idx[train_start:train_end]
        test_dates = all_dates_idx[test_start:test_end]

        if len(test_dates) < 10:
            break

        console.print(f"  Fold {fold_id + 1}: train [{train_dates[0].strftime('%Y-%m-%d')} → "
                       f"{train_dates[-1].strftime('%Y-%m-%d')}] | "
                       f"test [{test_dates[0].strftime('%Y-%m-%d')} → "
                       f"{test_dates[-1].strftime('%Y-%m-%d')}]")

        result = run_fold(fold_id, all_features, all_data, train_dates, test_dates, cfg)
        results.append(result)
        console.print(f"    → Return: {result.quarterly_return:.2%}, "
                       f"Trades: {result.num_trades}, Win: {result.win_rate:.1%}")

        fold_id += 1
        start_idx += cfg.step_size

    return results


# ── SPY Benchmark ───────────────────────────────────────────────────────────
def compute_spy_benchmark(all_data: dict[str, pd.DataFrame],
                          folds: list[FoldResult], cfg: Config) -> dict[str, Any]:
    """Compute SPY benchmark performance across the same test periods."""
    spy_data = all_data.get("SPY")
    if spy_data is None:
        try:
            spy_data = yf.download("SPY", start=cfg.start_date, end=cfg.end_date,
                                   auto_adjust=True, progress=False)
        except Exception:
            return {}

    if spy_data is None or spy_data.empty:
        return {}

    quarterly_returns: list[float] = []
    for fold in folds:
        start = fold.test_start
        end = fold.test_end
        spy_period = spy_data[(spy_data.index >= start) & (spy_data.index <= end)]
        if len(spy_period) >= 2:
            ret = (_safe_float(spy_period["Close"].iloc[-1]) / _safe_float(spy_period["Close"].iloc[0])) - 1
            quarterly_returns.append(float(ret))

    if not quarterly_returns:
        return {}

    returns_arr = np.array(quarterly_returns)
    ann_return = (1 + np.mean(returns_arr)) ** 4 - 1
    sharpe = (np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(4)) if np.std(returns_arr) > 0 else 0

    # Compute cumulative for max drawdown
    cum = np.cumprod(1 + returns_arr)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(dd.min())

    return {
        "mean_quarterly": float(np.mean(returns_arr)),
        "annualized": float(ann_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(np.mean(returns_arr > 0)),
        "quarterly_returns": quarterly_returns,
    }


# ── Statistical Tests ───────────────────────────────────────────────────────
def statistical_analysis(folds: list[FoldResult]) -> dict[str, Any]:
    """Paper Section 5.3: Statistical significance tests."""
    returns = [f.quarterly_return for f in folds]
    n = len(returns)
    if n < 2:
        return {}

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    # t-test: H0: mean return = 0
    t_stat, p_value = stats.ttest_1samp(returns, 0)

    # Bootstrap CI
    np.random.seed(42)
    boot_means: list[float] = []
    for _ in range(10_000):
        sample = np.random.choice(returns, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    # Effect size (Cohen's d)
    d = mean_ret / std_ret if std_ret > 0 else 0

    # Statistical power at observed effect size
    from scipy.stats import norm
    alpha = 0.05
    z_alpha = norm.ppf(1 - alpha / 2)
    power = 1 - norm.cdf(z_alpha - abs(d) * np.sqrt(n))

    # Required sample for 80% power
    z_beta = norm.ppf(0.80)
    n_required = int(np.ceil(((z_alpha + z_beta) / d) ** 2)) if d > 0 else float("inf")

    return {
        "n_folds": n,
        "mean_quarterly_return": float(mean_ret),
        "std_quarterly_return": float(std_ret),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "cohen_d": float(d),
        "statistical_power": float(power),
        "n_required_80pct_power": n_required,
    }


# ── Regime Analysis ─────────────────────────────────────────────────────────
def regime_analysis(folds: list[FoldResult], cfg: Config) -> dict[str, dict[str, Any]]:
    """Paper Table 3: regime-dependent performance."""
    regimes: dict[str, list[FoldResult]] = {
        "low_volatility_2015_2019": [],
        "high_volatility_2020_2024": [],
        "covid_crash_2020_q1q2": [],
        "bear_market_2022": [],
        "stabilization_2023_2024": [],
    }

    for fold in folds:
        year = fold.test_start.year
        month = fold.test_start.month

        if 2015 <= year <= 2019:
            regimes["low_volatility_2015_2019"].append(fold)
        if 2020 <= year <= 2024:
            regimes["high_volatility_2020_2024"].append(fold)
        if year == 2020 and month <= 6:
            regimes["covid_crash_2020_q1q2"].append(fold)
        if year == 2022:
            regimes["bear_market_2022"].append(fold)
        if 2023 <= year <= 2024:
            regimes["stabilization_2023_2024"].append(fold)

    results: dict[str, dict[str, Any]] = {}
    for regime_name, regime_folds in regimes.items():
        if not regime_folds:
            continue
        returns = [f.quarterly_return for f in regime_folds]
        mean_ret = float(np.mean(returns))
        win_rate = float(np.mean([r > 0 for r in returns]))
        std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.001
        sharpe = (mean_ret / std_ret * np.sqrt(4)) if std_ret > 0 else 0

        results[regime_name] = {
            "n_folds": len(regime_folds),
            "mean_quarterly_return": mean_ret,
            "win_rate": win_rate,
            "sharpe": float(sharpe),
        }

    return results


# ── Visualization ───────────────────────────────────────────────────────────
def create_visualizations(folds: list[FoldResult], spy_bench: dict[str, Any],
                          stat_results: dict[str, Any], regime_results: dict[str, dict[str, Any]],
                          cfg: Config) -> list[Path]:
    """Generate comprehensive analysis charts."""
    sns.set_theme(style="whitegrid", palette="muted")
    saved_files: list[Path] = []
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Cumulative equity curve across all folds
    fig, ax = plt.subplots(figsize=(14, 6))
    cumulative = [cfg.initial_capital]
    dates_list: list[pd.Timestamp] = []
    for fold in folds:
        if fold.equity_curve:
            fold_ret = fold.quarterly_return
            cumulative.append(cumulative[-1] * (1 + fold_ret))
            dates_list.append(fold.test_end)

    if dates_list:
        ax.plot(dates_list, cumulative[1:], "b-", linewidth=2, label="Strategy")

        # SPY benchmark cumulative
        if spy_bench and "quarterly_returns" in spy_bench:
            spy_cum = [cfg.initial_capital]
            for r in spy_bench["quarterly_returns"]:
                spy_cum.append(spy_cum[-1] * (1 + r))
            spy_dates = dates_list[:len(spy_cum) - 1]
            ax.plot(spy_dates, spy_cum[1:], "r--", linewidth=2, label="SPY", alpha=0.7)

    ax.set_title("Cumulative Equity: Strategy vs SPY Benchmark", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    path = OUTPUT_DIR / f"equity_curve_{ts}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_files.append(path)

    # 2. Quarterly returns bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    fold_returns = [f.quarterly_return for f in folds]
    fold_dates = [f.test_start for f in folds]
    colors = ["green" if r > 0 else "red" for r in fold_returns]
    ax.bar(range(len(fold_returns)), [r * 100 for r in fold_returns], color=colors, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("Quarterly Returns by Fold (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Fold Number")
    ax.set_ylabel("Return (%)")

    # Add date labels
    tick_positions = list(range(0, len(fold_dates), max(1, len(fold_dates) // 10)))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([fold_dates[i].strftime("%Y-Q%q" if hasattr(fold_dates[i], "quarter")
                        else "%Y-%m") for i in tick_positions], rotation=45, ha="right")
    fig.tight_layout()
    path = OUTPUT_DIR / f"quarterly_returns_{ts}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_files.append(path)

    # 3. Regime performance comparison
    if regime_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        regime_names = list(regime_results.keys())
        regime_labels = [n.replace("_", "\n") for n in regime_names]
        returns_by_regime = [regime_results[r]["mean_quarterly_return"] * 100 for r in regime_names]
        sharpes_by_regime = [regime_results[r]["sharpe"] for r in regime_names]

        colors_ret = ["green" if r > 0 else "red" for r in returns_by_regime]
        axes[0].barh(regime_labels, returns_by_regime, color=colors_ret, alpha=0.7)
        axes[0].set_xlabel("Mean Quarterly Return (%)")
        axes[0].set_title("Returns by Regime", fontweight="bold")
        axes[0].axvline(x=0, color="black", linewidth=0.5)

        colors_sh = ["green" if s > 0 else "red" for s in sharpes_by_regime]
        axes[1].barh(regime_labels, sharpes_by_regime, color=colors_sh, alpha=0.7)
        axes[1].set_xlabel("Sharpe Ratio")
        axes[1].set_title("Sharpe by Regime", fontweight="bold")
        axes[1].axvline(x=0, color="black", linewidth=0.5)

        fig.suptitle("Regime-Dependent Performance", fontsize=14, fontweight="bold")
        fig.tight_layout()
        path = OUTPUT_DIR / f"regime_analysis_{ts}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_files.append(path)

    # 4. Hypothesis type distribution and win rates
    all_trades = [t for f in folds for t in f.trades]
    if all_trades:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        h_types = {}
        for t in all_trades:
            ht = t.hypothesis_type
            if ht not in h_types:
                h_types[ht] = {"count": 0, "wins": 0, "returns": []}
            h_types[ht]["count"] += 1
            h_types[ht]["returns"].append(t.return_pct)
            if t.pnl > 0:
                h_types[ht]["wins"] += 1

        labels = [h.replace("_", "\n") for h in h_types.keys()]
        counts = [h_types[h]["count"] for h in h_types]
        win_rates = [h_types[h]["wins"] / h_types[h]["count"] * 100 for h in h_types]

        axes[0].bar(labels, counts, color="steelblue", alpha=0.7)
        axes[0].set_title("Trade Count by Hypothesis", fontweight="bold")
        axes[0].set_ylabel("Number of Trades")

        colors_wr = ["green" if w > 50 else "orange" if w > 40 else "red" for w in win_rates]
        axes[1].bar(labels, win_rates, color=colors_wr, alpha=0.7)
        axes[1].axhline(y=50, color="black", linestyle="--", alpha=0.5)
        axes[1].set_title("Win Rate by Hypothesis (%)", fontweight="bold")
        axes[1].set_ylabel("Win Rate (%)")

        fig.suptitle("Hypothesis Type Analysis", fontsize=14, fontweight="bold")
        fig.tight_layout()
        path = OUTPUT_DIR / f"hypothesis_analysis_{ts}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_files.append(path)

    # 5. Return distribution with bootstrap CI
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist([r * 100 for r in fold_returns], bins=15, color="steelblue",
            alpha=0.7, edgecolor="black")
    if stat_results:
        ci_low = stat_results.get("bootstrap_ci_low", 0) * 100
        ci_high = stat_results.get("bootstrap_ci_high", 0) * 100
        mean_r = stat_results.get("mean_quarterly_return", 0) * 100
        ax.axvline(mean_r, color="red", linestyle="-", linewidth=2, label=f"Mean: {mean_r:.2f}%")
        ax.axvline(ci_low, color="orange", linestyle="--", linewidth=1.5,
                   label=f"95% CI: [{ci_low:.2f}%, {ci_high:.2f}%]")
        ax.axvline(ci_high, color="orange", linestyle="--", linewidth=1.5)
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set_title("Distribution of Quarterly Returns", fontsize=14, fontweight="bold")
    ax.set_xlabel("Quarterly Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / f"return_distribution_{ts}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_files.append(path)

    # 6. Rolling Sharpe Ratio
    if len(fold_returns) >= 4:
        fig, ax = plt.subplots(figsize=(14, 5))
        rolling_sharpe: list[float] = []
        for i in range(3, len(fold_returns)):
            window = fold_returns[i - 3:i + 1]
            if np.std(window) > 0:
                rs = np.mean(window) / np.std(window) * np.sqrt(4)
            else:
                rs = 0
            rolling_sharpe.append(rs)

        roll_dates = fold_dates[3:]
        ax.plot(roll_dates, rolling_sharpe, "b-", linewidth=2)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax.fill_between(roll_dates, rolling_sharpe, 0,
                        where=[s > 0 for s in rolling_sharpe], alpha=0.3, color="green")
        ax.fill_between(roll_dates, rolling_sharpe, 0,
                        where=[s <= 0 for s in rolling_sharpe], alpha=0.3, color="red")
        ax.set_title("Rolling 4-Quarter Sharpe Ratio", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe Ratio")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        path = OUTPUT_DIR / f"rolling_sharpe_{ts}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_files.append(path)

    return saved_files


# ── Report Generation ───────────────────────────────────────────────────────
def generate_report(folds: list[FoldResult], spy_bench: dict[str, Any],
                    stat_results: dict[str, Any], regime_results: dict[str, dict[str, Any]],
                    chart_files: list[Path], cfg: Config) -> Path:
    """Generate comprehensive markdown report."""
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_trades = [t for f in folds for t in f.trades]
    fold_returns = [f.quarterly_return for f in folds]

    # Aggregate metrics
    ann_return = (1 + np.mean(fold_returns)) ** 4 - 1 if fold_returns else 0
    total_trades = len(all_trades)
    overall_wr = sum(1 for t in all_trades if t.pnl > 0) / total_trades if total_trades > 0 else 0
    max_dd = min(f.max_drawdown for f in folds) if folds else 0

    # Beta calculation
    if spy_bench and "quarterly_returns" in spy_bench:
        strategy_rets = np.array(fold_returns)
        spy_rets = np.array(spy_bench["quarterly_returns"][:len(fold_returns)])
        if len(spy_rets) == len(strategy_rets) and np.var(spy_rets) > 0:
            beta = float(np.cov(strategy_rets, spy_rets)[0, 1] / np.var(spy_rets))
            corr = float(np.corrcoef(strategy_rets, spy_rets)[0, 1])
        else:
            beta, corr = 0.0, 0.0
    else:
        beta, corr = 0.0, 0.0

    lines = [
        "# Replication: Interpretable Hypothesis-Driven Trading",
        "",
        "**Paper:** Deep, Deep & Lamptey (2025) - arXiv:2512.12924v1",
        f"**Replication Date:** {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Executive Summary",
        "",
        "This replicates the walk-forward validation framework for market microstructure",
        "signals across 100 US equities (2015-2024). The framework combines interpretable",
        "hypothesis-driven signal generation with reinforcement learning and strict",
        "out-of-sample testing.",
        "",
        "## Aggregate Out-of-Sample Performance",
        "",
        "| Metric | Strategy (Replication) | SPY Benchmark | Paper (Reported) |",
        "|--------|----------------------|---------------|------------------|",
        f"| Mean Quarterly Return | {np.mean(fold_returns):.2%} | "
        f"{spy_bench.get('mean_quarterly', 'N/A'):.2%} | 0.14% |" if spy_bench else
        f"| Mean Quarterly Return | {np.mean(fold_returns):.2%} | N/A | 0.14% |",
        f"| Annualized Return | {ann_return:.2%} | "
        f"{spy_bench.get('annualized', 'N/A'):.2%} | 0.55% |" if spy_bench else
        f"| Annualized Return | {ann_return:.2%} | N/A | 0.55% |",
        f"| Sharpe Ratio | {stat_results.get('mean_quarterly_return', 0) / stat_results.get('std_quarterly_return', 1) * 2:.2f} | "
        f"{spy_bench.get('sharpe', 'N/A'):.2f} | 0.33 |" if spy_bench and stat_results else
        f"| Sharpe Ratio | N/A | N/A | 0.33 |",
        f"| Maximum Drawdown | {max_dd:.2%} | "
        f"{spy_bench.get('max_drawdown', 'N/A'):.2%} | -2.76% |" if spy_bench else
        f"| Maximum Drawdown | {max_dd:.2%} | N/A | -2.76% |",
        f"| Beta | {beta:.3f} | 1.000 | 0.058 |",
        f"| Correlation w/ SPY | {corr:.2f} | 1.00 | 0.53 |",
        f"| Fold Win Rate | {np.mean([r > 0 for r in fold_returns]):.0%} | "
        f"{spy_bench.get('win_rate', 'N/A'):.0%} | 41% |" if spy_bench else
        f"| Fold Win Rate | {np.mean([r > 0 for r in fold_returns]):.0%} | N/A | 41% |",
        f"| Trade-Level Win Rate | {overall_wr:.1%} | — | 46.5% |",
        f"| Total Trades | {total_trades} | — | — |",
        f"| Total Folds | {len(folds)} | — | 34 |",
        "",
        "## Statistical Significance",
        "",
    ]

    if stat_results:
        lines.extend([
            "| Test | Value |",
            "|------|-------|",
            f"| t-statistic | {stat_results['t_statistic']:.2f} |",
            f"| p-value (two-sided) | {stat_results['p_value']:.4f} |",
            f"| Bootstrap 95% CI | [{stat_results['bootstrap_ci_low']:.2%}, {stat_results['bootstrap_ci_high']:.2%}] |",
            f"| Cohen's d | {stat_results['cohen_d']:.2f} |",
            f"| Statistical Power | {stat_results['statistical_power']:.1%} |",
            f"| N required (80% power) | {stat_results['n_required_80pct_power']} |",
            "",
        ])

    lines.extend(["## Regime-Dependent Performance", ""])
    if regime_results:
        lines.append("| Regime | Mean Quarterly Return | Win Rate | Sharpe | N Folds |")
        lines.append("|--------|----------------------|----------|--------|---------|")
        for regime, metrics in regime_results.items():
            label = regime.replace("_", " ").title()
            lines.append(
                f"| {label} | {metrics['mean_quarterly_return']:.2%} | "
                f"{metrics['win_rate']:.0%} | {metrics['sharpe']:.2f} | "
                f"{metrics['n_folds']} |"
            )
        lines.append("")

    # Hypothesis breakdown
    lines.extend(["## Hypothesis Type Analysis", ""])
    h_stats: dict[str, dict[str, Any]] = {}
    for t in all_trades:
        ht = t.hypothesis_type
        if ht not in h_stats:
            h_stats[ht] = {"count": 0, "wins": 0, "total_return": 0.0, "returns": []}
        h_stats[ht]["count"] += 1
        h_stats[ht]["total_return"] += t.return_pct
        h_stats[ht]["returns"].append(t.return_pct)
        if t.pnl > 0:
            h_stats[ht]["wins"] += 1

    if h_stats:
        lines.append("| Hypothesis | Trades | Win Rate | Mean Return | Sharpe |")
        lines.append("|------------|--------|----------|-------------|--------|")
        for ht, s in sorted(h_stats.items(), key=lambda x: x[1]["count"], reverse=True):
            wr = s["wins"] / s["count"]
            mr = np.mean(s["returns"])
            sr = (np.mean(s["returns"]) / np.std(s["returns"]) if np.std(s["returns"]) > 0
                  else 0)
            lines.append(f"| {ht.replace('_', ' ').title()} | {s['count']} | "
                         f"{wr:.1%} | {mr:.2%} | {sr:.2f} |")
        lines.append("")

    # Fold-by-fold results
    lines.extend(["## Fold-by-Fold Results", ""])
    lines.append("| Fold | Test Period | Return | Trades | Win Rate | Sharpe |")
    lines.append("|------|-------------|--------|--------|----------|--------|")
    for f in folds:
        period = f"{f.test_start.strftime('%Y-%m-%d')} → {f.test_end.strftime('%Y-%m-%d')}"
        lines.append(f"| {f.fold_id + 1} | {period} | {f.quarterly_return:.2%} | "
                     f"{f.num_trades} | {f.win_rate:.1%} | {f.sharpe:.2f} |")
    lines.append("")

    # Charts
    lines.extend(["## Generated Charts", ""])
    for chart in chart_files:
        lines.append(f"- `{chart.name}`")
    lines.append("")

    # Configuration
    lines.extend([
        "## Configuration",
        "",
        f"- Train window: {cfg.train_window} days",
        f"- Test window: {cfg.test_window} days",
        f"- Step size: {cfg.step_size} days",
        f"- Commission: ${cfg.commission}",
        f"- Slippage: {cfg.slippage_bps} bps",
        f"- Max positions: {cfg.max_positions}",
        f"- Max position size: {cfg.max_position_pct:.0%}",
        f"- Initial capital: ${cfg.initial_capital:,.0f}",
        f"- ε (train): {cfg.epsilon_train}",
        f"- ε (test): {cfg.epsilon_test}",
        "",
        "## Methodology Notes",
        "",
        "This replication follows the paper's methodology:",
        "1. **Information Set Discipline**: Features use only past data (no lookahead)",
        "2. **Walk-Forward Validation**: Rolling 252/63/63 day windows",
        "3. **Interpretability**: Every trade linked to a named hypothesis with explanation",
        "4. **Realistic Execution**: Commissions, slippage, position limits enforced",
        "5. **ε-greedy RL agent**: Adaptive threshold τ(c) = 0.45 + (1-c)×0.10",
        "",
        "### Differences from Paper",
        "- Universe selection: representative stocks matching GICS criteria (exact list may differ)",
        "- Volume classification: close-vs-open proxy (paper may use tick-level data)",
        "- Feature engineering: core 3 microstructure features implemented faithfully",
        "",
        f"*Report generated: {dt.datetime.now().isoformat()}*",
    ])

    report_path = OUTPUT_DIR / f"replication_report_{ts}.md"
    report_path.write_text("\n".join(lines))
    return report_path


# ── Main Entry Point ────────────────────────────────────────────────────────
def main() -> None:
    console.print("\n[bold magenta]═══════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]  Replication: Interpretable Hypothesis-Driven Trading  [/bold magenta]")
    console.print("[bold magenta]  Deep, Deep & Lamptey (2025) — arXiv:2512.12924v1      [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════════════[/bold magenta]\n")

    cfg = Config()
    np.random.seed(42)

    # Step 1: Fetch data
    console.rule("[bold]Step 1: Data Acquisition[/bold]")
    all_data = fetch_market_data(ALL_TICKERS, cfg.start_date, cfg.end_date)

    # Also fetch SPY for benchmark
    try:
        spy_df = yf.download("SPY", start=cfg.start_date, end=cfg.end_date,
                             auto_adjust=True, progress=False)
        if spy_df is not None and not spy_df.empty:
            all_data["SPY"] = spy_df
    except Exception:
        pass

    console.print(f"[green]Universe: {len(all_data)} securities loaded[/green]\n")

    # Step 2: Feature engineering
    console.rule("[bold]Step 2: Feature Engineering[/bold]")
    all_features: dict[str, pd.DataFrame] = {}
    for ticker, df in all_data.items():
        if ticker == "SPY":
            continue
        try:
            feat = compute_features(df, cfg)
            if len(feat) > cfg.train_window:
                all_features[ticker] = feat
        except Exception as e:
            console.print(f"[yellow]Feature error for {ticker}: {e}[/yellow]")

    console.print(f"[green]Features computed for {len(all_features)} tickers[/green]\n")

    # Step 3: Walk-forward validation
    console.rule("[bold]Step 3: Walk-Forward Validation[/bold]")
    folds = walk_forward_validation(all_features, all_data, cfg)
    console.print(f"\n[green]Completed {len(folds)} folds[/green]\n")

    # Step 4: SPY Benchmark
    console.rule("[bold]Step 4: SPY Benchmark[/bold]")
    spy_bench = compute_spy_benchmark(all_data, folds, cfg)
    if spy_bench:
        console.print(f"SPY annualized return: {spy_bench.get('annualized', 0):.2%}")
        console.print(f"SPY Sharpe: {spy_bench.get('sharpe', 0):.2f}")
    else:
        console.print("[yellow]SPY benchmark unavailable[/yellow]")
    console.print()

    # Step 5: Statistical analysis
    console.rule("[bold]Step 5: Statistical Analysis[/bold]")
    stat_results = statistical_analysis(folds)
    if stat_results:
        console.print(f"Mean quarterly return: {stat_results['mean_quarterly_return']:.2%}")
        console.print(f"t-statistic: {stat_results['t_statistic']:.2f} (p={stat_results['p_value']:.4f})")
        console.print(f"Bootstrap 95% CI: [{stat_results['bootstrap_ci_low']:.2%}, "
                       f"{stat_results['bootstrap_ci_high']:.2%}]")
        console.print(f"Cohen's d: {stat_results['cohen_d']:.2f}")
        console.print(f"Power: {stat_results['statistical_power']:.1%}")
    console.print()

    # Step 6: Regime analysis
    console.rule("[bold]Step 6: Regime Analysis[/bold]")
    regime_results = regime_analysis(folds, cfg)
    for regime, metrics in regime_results.items():
        label = regime.replace("_", " ").title()
        console.print(f"  {label}: return={metrics['mean_quarterly_return']:.2%}, "
                       f"Sharpe={metrics['sharpe']:.2f}")
    console.print()

    # Step 7: Visualization
    console.rule("[bold]Step 7: Generating Charts[/bold]")
    chart_files = create_visualizations(folds, spy_bench, stat_results, regime_results, cfg)
    for f in chart_files:
        console.print(f"  Saved: {f.name}")
    console.print()

    # Step 8: Report
    console.rule("[bold]Step 8: Report Generation[/bold]")
    report_path = generate_report(folds, spy_bench, stat_results, regime_results, chart_files, cfg)
    console.print(f"[bold green]Report saved: {report_path}[/bold green]\n")

    # Summary table
    console.rule("[bold]Summary: Replication vs Paper[/bold]")
    table = Table(title="Key Metrics Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Replication", style="green")
    table.add_column("Paper", style="yellow")

    fold_returns = [f.quarterly_return for f in folds]
    ann_return = (1 + np.mean(fold_returns)) ** 4 - 1 if fold_returns else 0
    all_trades = [t for f in folds for t in f.trades]
    overall_wr = sum(1 for t in all_trades if t.pnl > 0) / len(all_trades) if all_trades else 0
    max_dd = min(f.max_drawdown for f in folds) if folds else 0

    table.add_row("Mean Quarterly Return", f"{np.mean(fold_returns):.2%}", "0.14%")
    table.add_row("Annualized Return", f"{ann_return:.2%}", "0.55%")
    table.add_row("Max Drawdown", f"{max_dd:.2%}", "-2.76%")
    table.add_row("Trade Win Rate", f"{overall_wr:.1%}", "46.5%")
    table.add_row("Fold Win Rate", f"{np.mean([r > 0 for r in fold_returns]):.0%}", "41%")
    table.add_row("Total Folds", str(len(folds)), "34")
    table.add_row("Total Trades", str(len(all_trades)), "—")

    console.print(table)
    console.print(f"\n[bold]All outputs saved to: {OUTPUT_DIR}[/bold]")


if __name__ == "__main__":
    main()
