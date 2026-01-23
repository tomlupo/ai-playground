"""
Advanced Backtesting module using quantstats.

Provides comprehensive performance analytics and reporting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any
import warnings

import quantstats as qs


class QuantStatsAnalyzer:
    """
    Analyze strategy performance using quantstats.

    Provides detailed metrics, visualizations, and reports.
    """

    def __init__(self, returns: pd.Series, benchmark: pd.Series | None = None):
        """
        Initialize analyzer.

        Args:
            returns: Strategy returns series (daily)
            benchmark: Optional benchmark returns
        """
        self.returns = returns
        self.benchmark = benchmark

        # Ensure proper index
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            self.returns.index = pd.to_datetime(self.returns.index)

        if self.benchmark is not None and not isinstance(self.benchmark.index, pd.DatetimeIndex):
            self.benchmark.index = pd.to_datetime(self.benchmark.index)

    def get_metrics(self) -> dict[str, float]:
        """
        Get comprehensive performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            metrics = {
                # Returns
                "total_return": qs.stats.comp(self.returns),
                "cagr": qs.stats.cagr(self.returns),

                # Risk
                "volatility": qs.stats.volatility(self.returns),
                "max_drawdown": qs.stats.max_drawdown(self.returns),
                "avg_drawdown": qs.stats.avg_loss(self.returns),

                # Risk-adjusted returns
                "sharpe": qs.stats.sharpe(self.returns),
                "sortino": qs.stats.sortino(self.returns),
                "calmar": qs.stats.calmar(self.returns),

                # Win/Loss
                "win_rate": qs.stats.win_rate(self.returns),
                "profit_factor": qs.stats.profit_factor(self.returns),
                "profit_ratio": qs.stats.profit_ratio(self.returns),
                "payoff_ratio": qs.stats.payoff_ratio(self.returns),

                # Other
                "skew": qs.stats.skew(self.returns),
                "kurtosis": qs.stats.kurtosis(self.returns),
                "var": qs.stats.var(self.returns),
                "cvar": qs.stats.cvar(self.returns),

                # Recovery
                "recovery_factor": qs.stats.recovery_factor(self.returns),
                "ulcer_index": qs.stats.ulcer_index(self.returns),
            }

            # Add benchmark comparison if available
            if self.benchmark is not None:
                metrics["alpha"] = qs.stats.greeks(self.returns, self.benchmark)["alpha"]
                metrics["beta"] = qs.stats.greeks(self.returns, self.benchmark)["beta"]
                metrics["information_ratio"] = qs.stats.information_ratio(
                    self.returns, self.benchmark
                )

        return metrics

    def get_monthly_returns(self) -> pd.DataFrame:
        """Get monthly returns table."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return qs.stats.monthly_returns(self.returns)

    def get_drawdown_details(self) -> pd.DataFrame:
        """Get drawdown periods details."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return qs.stats.drawdown_details(self.returns)

    def get_rolling_metrics(self, window: int = 252) -> pd.DataFrame:
        """
        Get rolling performance metrics.

        Args:
            window: Rolling window size (days)

        Returns:
            DataFrame with rolling metrics
        """
        rolling = pd.DataFrame(index=self.returns.index)

        rolling["return"] = self.returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1
        )
        rolling["volatility"] = self.returns.rolling(window).std() * np.sqrt(252)
        rolling["sharpe"] = (
            self.returns.rolling(window).mean() * 252
        ) / (self.returns.rolling(window).std() * np.sqrt(252))

        # Rolling max drawdown
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.rolling(window, min_periods=1).max()
        rolling["drawdown"] = (cumulative - rolling_max) / rolling_max

        return rolling.dropna()

    def generate_report(self, output: str | Path, title: str = "Strategy Report") -> None:
        """
        Generate full HTML report.

        Args:
            output: Output file path
            title: Report title
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qs.reports.html(
                self.returns,
                benchmark=self.benchmark,
                output=str(output),
                title=title,
            )
        print(f"Report saved to: {output}")

    def print_summary(self) -> None:
        """Print summary statistics."""
        metrics = self.get_metrics()

        print("\n" + "=" * 50)
        print(" PERFORMANCE SUMMARY")
        print("=" * 50)

        print("\nReturns:")
        print(f"  Total Return:    {metrics['total_return']:.2%}")
        print(f"  CAGR:            {metrics['cagr']:.2%}")

        print("\nRisk:")
        print(f"  Volatility:      {metrics['volatility']:.2%}")
        print(f"  Max Drawdown:    {metrics['max_drawdown']:.2%}")
        print(f"  VaR (95%):       {metrics['var']:.2%}")
        print(f"  CVaR (95%):      {metrics['cvar']:.2%}")

        print("\nRisk-Adjusted:")
        print(f"  Sharpe Ratio:    {metrics['sharpe']:.2f}")
        print(f"  Sortino Ratio:   {metrics['sortino']:.2f}")
        print(f"  Calmar Ratio:    {metrics['calmar']:.2f}")

        print("\nWin/Loss:")
        print(f"  Win Rate:        {metrics['win_rate']:.2%}")
        print(f"  Profit Factor:   {metrics['profit_factor']:.2f}")

        if "alpha" in metrics:
            print("\nBenchmark Comparison:")
            print(f"  Alpha:           {metrics['alpha']:.4f}")
            print(f"  Beta:            {metrics['beta']:.2f}")

        print("=" * 50)


class BacktestEngine:
    """
    Enhanced backtesting engine with quantstats integration.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run_signals(
        self,
        prices: pd.Series,
        signals: pd.Series,
        position_size: float = 1.0,
    ) -> dict[str, Any]:
        """
        Run backtest from trading signals.

        Args:
            prices: Price series
            signals: Signal series (1=long, -1=short, 0=flat)
            position_size: Fraction of capital to use

        Returns:
            Dictionary with results
        """
        # Align data
        aligned = pd.DataFrame({"price": prices, "signal": signals}).dropna()
        prices = aligned["price"]
        signals = aligned["signal"]

        # Track positions
        position = 0
        cash = self.initial_capital
        shares = 0

        equity = []
        positions_log = []

        for i, (date, price) in enumerate(prices.items()):
            signal = signals.loc[date]

            # Apply slippage
            buy_price = price * (1 + self.slippage)
            sell_price = price * (1 - self.slippage)

            # Position changes
            if signal == 1 and position <= 0:
                # Go long
                if position < 0:
                    # Close short
                    cash -= shares * buy_price * (1 + self.commission)
                    shares = 0
                # Open long
                invest = cash * position_size
                shares = int(invest / (buy_price * (1 + self.commission)))
                cash -= shares * buy_price * (1 + self.commission)
                position = 1
                positions_log.append(("long", date, buy_price, shares))

            elif signal == -1 and position >= 0:
                # Go short/flat
                if position > 0:
                    # Close long
                    cash += shares * sell_price * (1 - self.commission)
                    positions_log.append(("close", date, sell_price, shares))
                    shares = 0
                position = -1 if signal == -1 else 0

            elif signal == 0 and position != 0:
                # Go flat
                if position > 0:
                    cash += shares * sell_price * (1 - self.commission)
                elif position < 0:
                    cash -= shares * buy_price * (1 + self.commission)
                shares = 0
                position = 0

            # Calculate equity
            if position > 0:
                equity_value = cash + shares * price
            elif position < 0:
                equity_value = cash - shares * price
            else:
                equity_value = cash

            equity.append({"date": date, "equity": equity_value})

        # Create equity series
        equity_df = pd.DataFrame(equity).set_index("date")
        equity_series = equity_df["equity"]

        # Calculate returns
        returns = equity_series.pct_change().dropna()

        # Analyze with quantstats
        analyzer = QuantStatsAnalyzer(returns)
        metrics = analyzer.get_metrics()

        return {
            "equity": equity_series,
            "returns": returns,
            "metrics": metrics,
            "positions": positions_log,
            "analyzer": analyzer,
        }

    def run_weights(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame | pd.Series,
        rebalance_freq: str = "M",
    ) -> dict[str, Any]:
        """
        Run backtest with portfolio weights.

        Args:
            prices: DataFrame of asset prices
            weights: DataFrame or Series of weights
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q')

        Returns:
            Dictionary with results
        """
        returns = prices.pct_change().dropna()

        # Handle static weights
        if isinstance(weights, pd.Series):
            weights_df = pd.DataFrame(
                [weights.values] * len(returns),
                index=returns.index,
                columns=returns.columns,
            )
        else:
            weights_df = weights.reindex(returns.index, method="ffill")

        # Calculate portfolio returns
        port_returns = (returns * weights_df).sum(axis=1)

        # Apply transaction costs at rebalance dates
        if isinstance(weights, pd.Series):
            # Static weights - only initial cost
            port_returns.iloc[0] -= self.commission
        else:
            # Dynamic weights - cost at each rebalance
            rebalance_dates = port_returns.resample(rebalance_freq).last().index
            for date in rebalance_dates:
                if date in port_returns.index:
                    port_returns.loc[date] -= self.commission

        # Analyze
        analyzer = QuantStatsAnalyzer(port_returns)
        metrics = analyzer.get_metrics()

        # Calculate equity curve
        equity = self.initial_capital * (1 + port_returns).cumprod()

        return {
            "equity": equity,
            "returns": port_returns,
            "metrics": metrics,
            "weights": weights_df,
            "analyzer": analyzer,
        }


def analyze_returns(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    output_report: str | Path | None = None,
) -> dict[str, Any]:
    """
    Comprehensive return analysis using quantstats.

    Args:
        returns: Strategy returns series
        benchmark: Optional benchmark returns
        output_report: Optional path for HTML report

    Returns:
        Dictionary with analysis results
    """
    analyzer = QuantStatsAnalyzer(returns, benchmark)

    results = {
        "metrics": analyzer.get_metrics(),
        "monthly_returns": analyzer.get_monthly_returns(),
        "rolling_metrics": analyzer.get_rolling_metrics(),
        "analyzer": analyzer,
    }

    try:
        results["drawdown_details"] = analyzer.get_drawdown_details()
    except Exception:
        pass

    if output_report:
        analyzer.generate_report(output_report)

    return results
