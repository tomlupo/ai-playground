"""
Performance analytics for trading systems.

Provides:
- Trade analysis and statistics
- Portfolio performance metrics
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Benchmark comparison
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
from collections import defaultdict
import math
import statistics

from trading.core.models import Trade, Position, OHLCV
from trading.core.enums import OrderSide


@dataclass
class TradeStats:
    """Statistics for a single trade or group of trades."""
    symbol: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    gross_profit: Decimal = Decimal("0")
    gross_loss: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")

    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")
    average_win: Decimal = Decimal("0")
    average_loss: Decimal = Decimal("0")

    avg_trade_duration: timedelta = field(default_factory=timedelta)
    avg_bars_in_trade: float = 0.0

    # Win rate and expectancy
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: Decimal = Decimal("0")

    # Risk metrics
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0


@dataclass
class DrawdownInfo:
    """Drawdown information."""
    start_date: datetime
    end_date: Optional[datetime]
    recovery_date: Optional[datetime]
    peak_value: Decimal
    trough_value: Decimal
    drawdown_pct: float
    duration_days: int
    recovery_days: Optional[int]


@dataclass
class RiskMetrics:
    """Risk-related performance metrics."""
    # Volatility
    daily_volatility: float = 0.0
    annualized_volatility: float = 0.0

    # Drawdown
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    avg_drawdown_pct: float = 0.0
    drawdown_count: int = 0

    # Value at Risk
    var_95: float = 0.0  # 95% VaR
    var_99: float = 0.0  # 99% VaR
    cvar_95: float = 0.0  # Conditional VaR (Expected Shortfall)

    # Tail risk
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Underwater metrics
    current_drawdown_pct: float = 0.0
    days_since_peak: int = 0


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    # Period
    start_date: datetime
    end_date: datetime
    trading_days: int

    # Returns
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    daily_return_avg: float = 0.0

    # Starting/ending values
    starting_equity: Decimal = Decimal("0")
    ending_equity: Decimal = Decimal("0")
    peak_equity: Decimal = Decimal("0")

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    trade_stats: Optional[TradeStats] = None
    stats_by_symbol: dict[str, TradeStats] = field(default_factory=dict)

    # Risk metrics
    risk_metrics: Optional[RiskMetrics] = None

    # Drawdowns
    drawdowns: list[DrawdownInfo] = field(default_factory=list)

    # Benchmark comparison
    benchmark_return_pct: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None

    # Additional metrics
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0
    pain_index: float = 0.0


class PerformanceAnalyzer:
    """
    Analyzes trading performance and generates reports.

    Usage:
        analyzer = PerformanceAnalyzer()

        # Add equity curve data
        analyzer.add_equity_point(datetime.now(), Decimal("10000"))

        # Add trades
        for trade in trades:
            analyzer.add_trade(trade)

        # Generate report
        report = analyzer.generate_report()
        print(f"Sharpe: {report.sharpe_ratio:.2f}")
        print(f"Max DD: {report.risk_metrics.max_drawdown_pct:.1%}")
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        trading_days_per_year: int = 252,
    ):
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year

        # Data storage
        self._equity_curve: list[tuple[datetime, Decimal]] = []
        self._trades: list[Trade] = []
        self._daily_returns: list[float] = []
        self._benchmark_returns: list[float] = []

    def add_equity_point(self, timestamp: datetime, equity: Decimal) -> None:
        """Add a point to the equity curve."""
        self._equity_curve.append((timestamp, equity))
        self._equity_curve.sort(key=lambda x: x[0])

        # Calculate daily return if we have previous point
        if len(self._equity_curve) >= 2:
            prev_equity = self._equity_curve[-2][1]
            if prev_equity > 0:
                daily_return = float((equity - prev_equity) / prev_equity)
                self._daily_returns.append(daily_return)

    def add_trade(self, trade: Trade) -> None:
        """Add a completed trade for analysis."""
        self._trades.append(trade)

    def add_trades(self, trades: list[Trade]) -> None:
        """Add multiple trades."""
        self._trades.extend(trades)

    def set_benchmark_returns(self, returns: list[float]) -> None:
        """Set benchmark returns for comparison."""
        self._benchmark_returns = returns

    def generate_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> PerformanceReport:
        """
        Generate comprehensive performance report.

        Args:
            start_date: Analysis start date (defaults to first data point)
            end_date: Analysis end date (defaults to last data point)

        Returns:
            PerformanceReport with all metrics
        """
        if not self._equity_curve:
            return PerformanceReport(
                start_date=start_date or datetime.utcnow(),
                end_date=end_date or datetime.utcnow(),
                trading_days=0,
            )

        # Determine date range
        start_date = start_date or self._equity_curve[0][0]
        end_date = end_date or self._equity_curve[-1][0]

        # Filter data to date range
        equity_in_range = [
            (ts, eq) for ts, eq in self._equity_curve
            if start_date <= ts <= end_date
        ]

        if not equity_in_range:
            return PerformanceReport(
                start_date=start_date,
                end_date=end_date,
                trading_days=0,
            )

        # Basic metrics
        starting_equity = equity_in_range[0][1]
        ending_equity = equity_in_range[-1][1]
        peak_equity = max(eq for _, eq in equity_in_range)

        trading_days = len(equity_in_range)

        # Returns
        total_return_pct = 0.0
        if starting_equity > 0:
            total_return_pct = float((ending_equity - starting_equity) / starting_equity)

        years = trading_days / self.trading_days_per_year
        annualized_return_pct = 0.0
        if years > 0 and total_return_pct > -1:
            annualized_return_pct = (1 + total_return_pct) ** (1 / years) - 1

        daily_return_avg = 0.0
        if self._daily_returns:
            daily_return_avg = statistics.mean(self._daily_returns)

        # Risk metrics
        risk_metrics = self._calculate_risk_metrics()

        # Risk-adjusted returns
        sharpe_ratio = self._calculate_sharpe_ratio()
        sortino_ratio = self._calculate_sortino_ratio()
        calmar_ratio = self._calculate_calmar_ratio(annualized_return_pct, risk_metrics)

        # Trade statistics
        trade_stats = self._calculate_trade_stats(self._trades)
        stats_by_symbol = self._calculate_stats_by_symbol()

        # Drawdowns
        drawdowns = self._calculate_drawdowns(equity_in_range)

        # Benchmark comparison
        alpha, beta, info_ratio, benchmark_return = self._calculate_benchmark_metrics()

        # Additional metrics
        recovery_factor = 0.0
        if risk_metrics.max_drawdown_pct > 0:
            recovery_factor = total_return_pct / risk_metrics.max_drawdown_pct

        ulcer_index = self._calculate_ulcer_index(equity_in_range)
        pain_index = self._calculate_pain_index(equity_in_range)

        return PerformanceReport(
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            daily_return_avg=daily_return_avg,
            starting_equity=starting_equity,
            ending_equity=ending_equity,
            peak_equity=peak_equity,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            trade_stats=trade_stats,
            stats_by_symbol=stats_by_symbol,
            risk_metrics=risk_metrics,
            drawdowns=drawdowns,
            benchmark_return_pct=benchmark_return,
            alpha=alpha,
            beta=beta,
            information_ratio=info_ratio,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
            pain_index=pain_index,
        )

    def _calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate risk-related metrics."""
        if not self._daily_returns:
            return RiskMetrics()

        returns = self._daily_returns

        # Volatility
        daily_vol = statistics.stdev(returns) if len(returns) > 1 else 0.0
        annualized_vol = daily_vol * math.sqrt(self.trading_days_per_year)

        # VaR calculations
        sorted_returns = sorted(returns)
        n = len(sorted_returns)

        var_95_idx = int(n * 0.05)
        var_99_idx = int(n * 0.01)

        var_95 = sorted_returns[var_95_idx] if var_95_idx < n else 0.0
        var_99 = sorted_returns[var_99_idx] if var_99_idx < n else 0.0

        # CVaR (Expected Shortfall)
        cvar_95 = 0.0
        if var_95_idx > 0:
            cvar_95 = statistics.mean(sorted_returns[:var_95_idx])

        # Skewness and Kurtosis
        skewness = self._calculate_skewness(returns)
        kurtosis = self._calculate_kurtosis(returns)

        # Drawdown metrics from equity curve
        max_dd_pct = 0.0
        max_dd_duration = 0
        drawdown_count = 0
        total_dd_pct = 0.0
        current_dd_pct = 0.0
        days_since_peak = 0

        if self._equity_curve:
            peak = self._equity_curve[0][1]
            in_drawdown = False
            dd_start_idx = 0

            for i, (_, equity) in enumerate(self._equity_curve):
                if equity > peak:
                    if in_drawdown:
                        duration = i - dd_start_idx
                        max_dd_duration = max(max_dd_duration, duration)
                        in_drawdown = False
                    peak = equity
                    days_since_peak = 0
                else:
                    dd_pct = float((peak - equity) / peak) if peak > 0 else 0.0
                    max_dd_pct = max(max_dd_pct, dd_pct)
                    days_since_peak = i - self._equity_curve.index(
                        next((ts, eq) for ts, eq in self._equity_curve if eq == peak)
                    )

                    if not in_drawdown:
                        in_drawdown = True
                        drawdown_count += 1
                        dd_start_idx = i

                    total_dd_pct += dd_pct

            # Current drawdown
            if self._equity_curve:
                last_equity = self._equity_curve[-1][1]
                if peak > 0:
                    current_dd_pct = float((peak - last_equity) / peak)

        avg_dd_pct = total_dd_pct / len(self._equity_curve) if self._equity_curve else 0.0

        return RiskMetrics(
            daily_volatility=daily_vol,
            annualized_volatility=annualized_vol,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_duration_days=max_dd_duration,
            avg_drawdown_pct=avg_dd_pct,
            drawdown_count=drawdown_count,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            current_drawdown_pct=current_dd_pct,
            days_since_peak=days_since_peak,
        )

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        if not self._daily_returns or len(self._daily_returns) < 2:
            return 0.0

        excess_returns = [
            r - self.risk_free_rate / self.trading_days_per_year
            for r in self._daily_returns
        ]

        avg_excess = statistics.mean(excess_returns)
        std_excess = statistics.stdev(excess_returns)

        if std_excess == 0:
            return 0.0

        # Annualized Sharpe
        return (avg_excess / std_excess) * math.sqrt(self.trading_days_per_year)

    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (using downside deviation)."""
        if not self._daily_returns or len(self._daily_returns) < 2:
            return 0.0

        target = self.risk_free_rate / self.trading_days_per_year
        excess_returns = [r - target for r in self._daily_returns]
        downside_returns = [min(0, r) for r in excess_returns]

        avg_excess = statistics.mean(excess_returns)

        # Downside deviation
        downside_sq_sum = sum(r ** 2 for r in downside_returns)
        downside_std = math.sqrt(downside_sq_sum / len(downside_returns))

        if downside_std == 0:
            return 0.0

        return (avg_excess / downside_std) * math.sqrt(self.trading_days_per_year)

    def _calculate_calmar_ratio(
        self,
        annualized_return: float,
        risk_metrics: RiskMetrics
    ) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        if risk_metrics.max_drawdown_pct == 0:
            return 0.0
        return annualized_return / risk_metrics.max_drawdown_pct

    def _calculate_trade_stats(self, trades: list[Trade]) -> TradeStats:
        """Calculate statistics for a set of trades."""
        if not trades:
            return TradeStats(symbol="ALL")

        stats = TradeStats(symbol="ALL")
        stats.total_trades = len(trades)

        pnls: list[Decimal] = []
        durations: list[timedelta] = []
        consecutive_wins = 0
        consecutive_losses = 0
        current_streak = 0
        last_was_win: Optional[bool] = None

        for trade in trades:
            pnl = trade.realized_pnl or Decimal("0")
            pnls.append(pnl)

            if pnl > 0:
                stats.winning_trades += 1
                stats.gross_profit += pnl
                stats.largest_win = max(stats.largest_win, pnl)

                if last_was_win is True:
                    current_streak += 1
                else:
                    current_streak = 1
                stats.max_consecutive_wins = max(
                    stats.max_consecutive_wins, current_streak
                )
                last_was_win = True

            elif pnl < 0:
                stats.losing_trades += 1
                stats.gross_loss += pnl
                stats.largest_loss = min(stats.largest_loss, pnl)

                if last_was_win is False:
                    current_streak += 1
                else:
                    current_streak = 1
                stats.max_consecutive_losses = max(
                    stats.max_consecutive_losses, current_streak
                )
                last_was_win = False
            else:
                stats.breakeven_trades += 1

            # Duration (if we have entry and exit times)
            if trade.entry_time and trade.exit_time:
                durations.append(trade.exit_time - trade.entry_time)

        # Net P&L
        stats.net_pnl = stats.gross_profit + stats.gross_loss

        # Averages
        if stats.winning_trades > 0:
            stats.average_win = stats.gross_profit / stats.winning_trades
        if stats.losing_trades > 0:
            stats.average_loss = stats.gross_loss / stats.losing_trades

        # Duration
        if durations:
            total_seconds = sum(d.total_seconds() for d in durations)
            stats.avg_trade_duration = timedelta(
                seconds=total_seconds / len(durations)
            )

        # Win rate
        if stats.total_trades > 0:
            stats.win_rate = stats.winning_trades / stats.total_trades

        # Profit factor
        if stats.gross_loss != 0:
            stats.profit_factor = abs(float(stats.gross_profit / stats.gross_loss))

        # Expectancy
        if stats.total_trades > 0:
            stats.expectancy = stats.net_pnl / stats.total_trades

        return stats

    def _calculate_stats_by_symbol(self) -> dict[str, TradeStats]:
        """Calculate trade statistics grouped by symbol."""
        trades_by_symbol: dict[str, list[Trade]] = defaultdict(list)

        for trade in self._trades:
            trades_by_symbol[trade.symbol].append(trade)

        return {
            symbol: self._calculate_trade_stats(symbol_trades)
            for symbol, symbol_trades in trades_by_symbol.items()
        }

    def _calculate_drawdowns(
        self,
        equity_curve: list[tuple[datetime, Decimal]]
    ) -> list[DrawdownInfo]:
        """Calculate all drawdown periods."""
        if not equity_curve:
            return []

        drawdowns = []
        peak = equity_curve[0][1]
        peak_date = equity_curve[0][0]
        in_drawdown = False
        dd_start: Optional[datetime] = None
        dd_trough = peak
        dd_trough_date = peak_date

        for ts, equity in equity_curve:
            if equity > peak:
                if in_drawdown:
                    # Drawdown ended - recovered
                    dd_pct = float((peak - dd_trough) / peak) if peak > 0 else 0.0
                    drawdowns.append(DrawdownInfo(
                        start_date=dd_start,
                        end_date=dd_trough_date,
                        recovery_date=ts,
                        peak_value=peak,
                        trough_value=dd_trough,
                        drawdown_pct=dd_pct,
                        duration_days=(dd_trough_date - dd_start).days if dd_start else 0,
                        recovery_days=(ts - dd_trough_date).days,
                    ))
                    in_drawdown = False

                peak = equity
                peak_date = ts
            else:
                if not in_drawdown:
                    in_drawdown = True
                    dd_start = peak_date
                    dd_trough = equity
                    dd_trough_date = ts
                elif equity < dd_trough:
                    dd_trough = equity
                    dd_trough_date = ts

        # Handle ongoing drawdown
        if in_drawdown and dd_start:
            dd_pct = float((peak - dd_trough) / peak) if peak > 0 else 0.0
            drawdowns.append(DrawdownInfo(
                start_date=dd_start,
                end_date=dd_trough_date,
                recovery_date=None,  # Still in drawdown
                peak_value=peak,
                trough_value=dd_trough,
                drawdown_pct=dd_pct,
                duration_days=(dd_trough_date - dd_start).days,
                recovery_days=None,
            ))

        return sorted(drawdowns, key=lambda d: d.drawdown_pct, reverse=True)

    def _calculate_benchmark_metrics(
        self
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Calculate alpha, beta, and information ratio vs benchmark."""
        if not self._benchmark_returns or len(self._benchmark_returns) != len(self._daily_returns):
            return None, None, None, None

        if len(self._daily_returns) < 2:
            return None, None, None, None

        strategy_returns = self._daily_returns
        benchmark_returns = self._benchmark_returns

        # Beta = Cov(strategy, benchmark) / Var(benchmark)
        cov = self._covariance(strategy_returns, benchmark_returns)
        var_benchmark = statistics.variance(benchmark_returns)

        beta = cov / var_benchmark if var_benchmark > 0 else 0.0

        # Alpha = mean(strategy) - beta * mean(benchmark)
        alpha = statistics.mean(strategy_returns) - beta * statistics.mean(benchmark_returns)
        alpha_annualized = alpha * self.trading_days_per_year

        # Information Ratio = (strategy - benchmark) / tracking_error
        active_returns = [s - b for s, b in zip(strategy_returns, benchmark_returns)]
        tracking_error = statistics.stdev(active_returns) if len(active_returns) > 1 else 0.0

        info_ratio = 0.0
        if tracking_error > 0:
            info_ratio = (statistics.mean(active_returns) / tracking_error) * math.sqrt(
                self.trading_days_per_year
            )

        # Benchmark total return
        benchmark_total = 1.0
        for r in benchmark_returns:
            benchmark_total *= (1 + r)
        benchmark_return_pct = benchmark_total - 1

        return alpha_annualized, beta, info_ratio, benchmark_return_pct

    def _calculate_ulcer_index(
        self,
        equity_curve: list[tuple[datetime, Decimal]]
    ) -> float:
        """
        Calculate Ulcer Index - measures depth and duration of drawdowns.
        Lower is better.
        """
        if not equity_curve:
            return 0.0

        peak = equity_curve[0][1]
        squared_drawdowns = []

        for _, equity in equity_curve:
            if equity > peak:
                peak = equity
            dd_pct = float((peak - equity) / peak) * 100 if peak > 0 else 0.0
            squared_drawdowns.append(dd_pct ** 2)

        if not squared_drawdowns:
            return 0.0

        return math.sqrt(statistics.mean(squared_drawdowns))

    def _calculate_pain_index(
        self,
        equity_curve: list[tuple[datetime, Decimal]]
    ) -> float:
        """
        Calculate Pain Index - average drawdown.
        """
        if not equity_curve:
            return 0.0

        peak = equity_curve[0][1]
        drawdowns = []

        for _, equity in equity_curve:
            if equity > peak:
                peak = equity
            dd_pct = float((peak - equity) / peak) if peak > 0 else 0.0
            drawdowns.append(dd_pct)

        return statistics.mean(drawdowns) if drawdowns else 0.0

    @staticmethod
    def _calculate_skewness(returns: list[float]) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0.0

        n = len(returns)
        mean = statistics.mean(returns)
        std = statistics.stdev(returns)

        if std == 0:
            return 0.0

        skew_sum = sum((r - mean) ** 3 for r in returns)
        return (n / ((n - 1) * (n - 2))) * (skew_sum / (std ** 3))

    @staticmethod
    def _calculate_kurtosis(returns: list[float]) -> float:
        """Calculate excess kurtosis of returns."""
        if len(returns) < 4:
            return 0.0

        n = len(returns)
        mean = statistics.mean(returns)
        std = statistics.stdev(returns)

        if std == 0:
            return 0.0

        kurt_sum = sum((r - mean) ** 4 for r in returns)
        kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * (kurt_sum / (std ** 4))
        excess_kurt = kurt - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

        return excess_kurt

    @staticmethod
    def _covariance(x: list[float], y: list[float]) -> float:
        """Calculate covariance between two series."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        return sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (len(x) - 1)

    def reset(self) -> None:
        """Clear all data."""
        self._equity_curve.clear()
        self._trades.clear()
        self._daily_returns.clear()
        self._benchmark_returns.clear()


def format_performance_report(report: PerformanceReport) -> str:
    """Format performance report as human-readable string."""
    lines = [
        "=" * 60,
        "PERFORMANCE REPORT",
        "=" * 60,
        f"Period: {report.start_date.date()} to {report.end_date.date()}",
        f"Trading Days: {report.trading_days}",
        "",
        "--- RETURNS ---",
        f"Total Return: {report.total_return_pct:.2%}",
        f"Annualized Return: {report.annualized_return_pct:.2%}",
        f"Daily Avg Return: {report.daily_return_avg:.4%}",
        "",
        f"Starting Equity: {report.starting_equity:,.2f}",
        f"Ending Equity: {report.ending_equity:,.2f}",
        f"Peak Equity: {report.peak_equity:,.2f}",
        "",
        "--- RISK-ADJUSTED METRICS ---",
        f"Sharpe Ratio: {report.sharpe_ratio:.2f}",
        f"Sortino Ratio: {report.sortino_ratio:.2f}",
        f"Calmar Ratio: {report.calmar_ratio:.2f}",
        f"Recovery Factor: {report.recovery_factor:.2f}",
        f"Ulcer Index: {report.ulcer_index:.2f}",
    ]

    if report.risk_metrics:
        rm = report.risk_metrics
        lines.extend([
            "",
            "--- RISK METRICS ---",
            f"Daily Volatility: {rm.daily_volatility:.4%}",
            f"Annualized Volatility: {rm.annualized_volatility:.2%}",
            f"Max Drawdown: {rm.max_drawdown_pct:.2%}",
            f"Max DD Duration: {rm.max_drawdown_duration_days} days",
            f"Current Drawdown: {rm.current_drawdown_pct:.2%}",
            f"Days Since Peak: {rm.days_since_peak}",
            f"VaR (95%): {rm.var_95:.4%}",
            f"CVaR (95%): {rm.cvar_95:.4%}",
            f"Skewness: {rm.skewness:.2f}",
            f"Kurtosis: {rm.kurtosis:.2f}",
        ])

    if report.trade_stats:
        ts = report.trade_stats
        lines.extend([
            "",
            "--- TRADE STATISTICS ---",
            f"Total Trades: {ts.total_trades}",
            f"Win/Loss/BE: {ts.winning_trades}/{ts.losing_trades}/{ts.breakeven_trades}",
            f"Win Rate: {ts.win_rate:.1%}",
            f"Profit Factor: {ts.profit_factor:.2f}",
            f"Net P&L: {ts.net_pnl:,.2f}",
            f"Avg Win: {ts.average_win:,.2f}",
            f"Avg Loss: {ts.average_loss:,.2f}",
            f"Largest Win: {ts.largest_win:,.2f}",
            f"Largest Loss: {ts.largest_loss:,.2f}",
            f"Expectancy: {ts.expectancy:,.2f}",
            f"Max Consecutive Wins: {ts.max_consecutive_wins}",
            f"Max Consecutive Losses: {ts.max_consecutive_losses}",
        ])

    if report.benchmark_return_pct is not None:
        lines.extend([
            "",
            "--- BENCHMARK COMPARISON ---",
            f"Benchmark Return: {report.benchmark_return_pct:.2%}",
            f"Alpha: {report.alpha:.4f}" if report.alpha else "",
            f"Beta: {report.beta:.2f}" if report.beta else "",
            f"Information Ratio: {report.information_ratio:.2f}" if report.information_ratio else "",
        ])

    if report.drawdowns:
        lines.extend([
            "",
            "--- TOP 5 DRAWDOWNS ---",
        ])
        for i, dd in enumerate(report.drawdowns[:5], 1):
            recovery_str = f"recovered {dd.recovery_date.date()}" if dd.recovery_date else "ongoing"
            lines.append(
                f"  {i}. {dd.drawdown_pct:.2%} ({dd.start_date.date()} - {dd.end_date.date() if dd.end_date else 'ongoing'}, {recovery_str})"
            )

    lines.append("=" * 60)

    return "\n".join(lines)
