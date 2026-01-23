"""
Vectorized Backtesting module (vectorbt-style).

Provides fast, vectorized backtesting using NumPy/Pandas operations.
Supports:
- Multiple symbols simultaneously
- Parameter optimization
- Signal generation from indicators
- Portfolio simulation
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Callable
from itertools import product


@dataclass
class VectorizedResult:
    """Container for vectorized backtest results."""

    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    drawdown: pd.Series
    trades: pd.DataFrame
    params: dict = field(default_factory=dict)


class VectorizedBacktester:
    """
    Fast vectorized backtesting engine.

    Uses NumPy/Pandas for efficient computation.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0,
        risk_free_rate: float = 0.02,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate

    def run_from_signals(
        self,
        prices: pd.Series,
        signals: pd.Series,
        position_size: float = 1.0,
    ) -> VectorizedResult:
        """
        Run backtest from pre-computed signals.

        Args:
            prices: Price series
            signals: Signal series (1=long, -1=short, 0=flat)
            position_size: Fraction of capital to use per trade

        Returns:
            VectorizedResult with all metrics
        """
        # Align data
        aligned = pd.DataFrame({"price": prices, "signal": signals}).dropna()
        prices = aligned["price"]
        signals = aligned["signal"]

        # Calculate position changes
        position_changes = signals.diff().fillna(signals)

        # Apply slippage
        trade_prices = prices.copy()
        trade_prices[position_changes > 0] *= 1 + self.slippage  # Buy higher
        trade_prices[position_changes < 0] *= 1 - self.slippage  # Sell lower

        # Calculate returns
        price_returns = prices.pct_change().fillna(0)

        # Strategy returns (position from previous day applied to today's return)
        positions = signals.shift(1).fillna(0)
        strategy_returns = positions * price_returns

        # Apply commission on trades
        trade_costs = position_changes.abs() * self.commission
        strategy_returns = strategy_returns - trade_costs

        # Build equity curve
        equity = self.initial_capital * (1 + strategy_returns).cumprod()

        # Calculate metrics
        return self._calculate_metrics(
            equity, strategy_returns, positions, signals, prices
        )

    def run_sma_crossover(
        self,
        prices: pd.Series,
        short_period: int = 20,
        long_period: int = 50,
    ) -> VectorizedResult:
        """
        Run SMA crossover strategy.

        Args:
            prices: Price series
            short_period: Short SMA period
            long_period: Long SMA period

        Returns:
            VectorizedResult
        """
        sma_short = prices.rolling(short_period).mean()
        sma_long = prices.rolling(long_period).mean()

        # Generate signals
        signals = pd.Series(0, index=prices.index)
        signals[sma_short > sma_long] = 1
        signals[sma_short <= sma_long] = 0

        result = self.run_from_signals(prices, signals)
        result.params = {"short_period": short_period, "long_period": long_period}
        return result

    def run_rsi_strategy(
        self,
        prices: pd.Series,
        period: int = 14,
        oversold: int = 30,
        overbought: int = 70,
    ) -> VectorizedResult:
        """
        Run RSI mean reversion strategy.

        Args:
            prices: Price series
            period: RSI period
            oversold: Oversold threshold
            overbought: Overbought threshold

        Returns:
            VectorizedResult
        """
        # Calculate RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        signals = pd.Series(0, index=prices.index)
        signals[rsi < oversold] = 1  # Buy when oversold
        signals[rsi > overbought] = -1  # Sell when overbought

        # Forward fill signals
        signals = signals.replace(0, np.nan).ffill().fillna(0)

        result = self.run_from_signals(prices, signals)
        result.params = {
            "period": period,
            "oversold": oversold,
            "overbought": overbought,
        }
        return result

    def run_bollinger_strategy(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> VectorizedResult:
        """
        Run Bollinger Bands mean reversion strategy.

        Args:
            prices: Price series
            period: BB period
            std_dev: Standard deviation multiplier

        Returns:
            VectorizedResult
        """
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std

        # Generate signals
        signals = pd.Series(0, index=prices.index)
        signals[prices < lower] = 1  # Buy below lower band
        signals[prices > upper] = -1  # Sell above upper band

        # Forward fill
        signals = signals.replace(0, np.nan).ffill().fillna(0)

        result = self.run_from_signals(prices, signals)
        result.params = {"period": period, "std_dev": std_dev}
        return result

    def run_momentum_strategy(
        self,
        prices: pd.Series,
        lookback: int = 20,
        threshold: float = 0.0,
    ) -> VectorizedResult:
        """
        Run momentum strategy.

        Args:
            prices: Price series
            lookback: Momentum lookback period
            threshold: Momentum threshold for entry

        Returns:
            VectorizedResult
        """
        momentum = prices.pct_change(lookback)

        signals = pd.Series(0, index=prices.index)
        signals[momentum > threshold] = 1
        signals[momentum <= -threshold] = -1

        result = self.run_from_signals(prices, signals)
        result.params = {"lookback": lookback, "threshold": threshold}
        return result

    def run_macd_strategy(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> VectorizedResult:
        """
        Run MACD crossover strategy.

        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            VectorizedResult
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()

        signals = pd.Series(0, index=prices.index)
        signals[macd_line > signal_line] = 1
        signals[macd_line <= signal_line] = 0

        result = self.run_from_signals(prices, signals)
        result.params = {"fast": fast, "slow": slow, "signal": signal}
        return result

    def _calculate_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        positions: pd.Series,
        signals: pd.Series,
        prices: pd.Series,
    ) -> VectorizedResult:
        """Calculate all performance metrics."""
        # Total return
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital

        # Annualized return
        days = (equity.index[-1] - equity.index[0]).days
        annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1

        # Volatility
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio
        excess_return = annual_return - self.risk_free_rate
        sharpe = excess_return / volatility if volatility > 0 else 0

        # Sortino ratio (downside volatility)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = excess_return / downside_vol if downside_vol > 0 else 0

        # Maximum drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate and profit factor
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        total_trades = (signals.diff().abs() > 0).sum()

        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        profit_factor = (
            winning_trades.sum() / abs(losing_trades.sum())
            if len(losing_trades) > 0 and losing_trades.sum() != 0
            else 0
        )

        # Trade log
        trade_dates = signals[signals.diff().abs() > 0].index
        trades_df = pd.DataFrame({
            "date": trade_dates,
            "signal": signals.loc[trade_dates],
            "price": prices.loc[trade_dates],
        }) if len(trade_dates) > 0 else pd.DataFrame()

        return VectorizedResult(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            equity_curve=equity,
            returns=returns,
            positions=positions,
            drawdown=drawdown,
            trades=trades_df,
        )


class ParameterOptimizer:
    """
    Optimize strategy parameters using vectorized backtesting.

    Supports grid search and random search.
    """

    def __init__(self, backtester: VectorizedBacktester | None = None):
        self.backtester = backtester or VectorizedBacktester()

    def grid_search(
        self,
        prices: pd.Series,
        strategy: Literal["sma", "rsi", "bollinger", "momentum", "macd"],
        param_grid: dict[str, list],
        metric: str = "sharpe_ratio",
    ) -> tuple[dict, pd.DataFrame]:
        """
        Grid search for optimal parameters.

        Args:
            prices: Price series
            strategy: Strategy name
            param_grid: Parameter grid {param_name: [values]}
            metric: Metric to optimize

        Returns:
            Tuple of (best params, all results DataFrame)
        """
        # Get strategy function
        strategy_funcs = {
            "sma": self.backtester.run_sma_crossover,
            "rsi": self.backtester.run_rsi_strategy,
            "bollinger": self.backtester.run_bollinger_strategy,
            "momentum": self.backtester.run_momentum_strategy,
            "macd": self.backtester.run_macd_strategy,
        }

        if strategy not in strategy_funcs:
            raise ValueError(f"Unknown strategy: {strategy}")

        func = strategy_funcs[strategy]

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        results = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            try:
                result = func(prices, **params)
                results.append({
                    **params,
                    "total_return": result.total_return,
                    "annual_return": result.annual_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "sortino_ratio": result.sortino_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "total_trades": result.total_trades,
                })
            except Exception:
                continue

        results_df = pd.DataFrame(results)

        if results_df.empty:
            return {}, results_df

        # Find best parameters
        best_idx = results_df[metric].idxmax()
        best_params = {
            name: results_df.loc[best_idx, name] for name in param_names
        }

        return best_params, results_df

    def walk_forward(
        self,
        prices: pd.Series,
        strategy: Literal["sma", "rsi", "bollinger", "momentum", "macd"],
        param_grid: dict[str, list],
        train_size: int = 252,
        test_size: int = 63,
        metric: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """
        Walk-forward optimization.

        Args:
            prices: Price series
            strategy: Strategy name
            param_grid: Parameter grid
            train_size: Training window size (days)
            test_size: Testing window size (days)
            metric: Optimization metric

        Returns:
            DataFrame with out-of-sample results
        """
        results = []
        start_idx = 0

        while start_idx + train_size + test_size <= len(prices):
            # Split data
            train_data = prices.iloc[start_idx : start_idx + train_size]
            test_data = prices.iloc[
                start_idx + train_size : start_idx + train_size + test_size
            ]

            # Optimize on training data
            best_params, _ = self.grid_search(
                train_data, strategy, param_grid, metric
            )

            if best_params:
                # Test on out-of-sample data
                strategy_funcs = {
                    "sma": self.backtester.run_sma_crossover,
                    "rsi": self.backtester.run_rsi_strategy,
                    "bollinger": self.backtester.run_bollinger_strategy,
                    "momentum": self.backtester.run_momentum_strategy,
                    "macd": self.backtester.run_macd_strategy,
                }
                func = strategy_funcs[strategy]
                test_result = func(test_data, **best_params)

                results.append({
                    "period_start": test_data.index[0],
                    "period_end": test_data.index[-1],
                    **best_params,
                    "oos_return": test_result.total_return,
                    "oos_sharpe": test_result.sharpe_ratio,
                    "oos_max_dd": test_result.max_drawdown,
                })

            start_idx += test_size

        return pd.DataFrame(results)


class MultiAssetBacktester:
    """
    Backtest strategies across multiple assets.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.backtester = VectorizedBacktester(
            initial_capital=initial_capital,
            commission=commission,
        )

    def run_portfolio(
        self,
        prices: pd.DataFrame,
        weights: pd.Series | dict,
        rebalance_freq: str = "M",
    ) -> VectorizedResult:
        """
        Run portfolio backtest with fixed weights.

        Args:
            prices: DataFrame of asset prices
            weights: Asset weights (Series or dict)
            rebalance_freq: Rebalancing frequency

        Returns:
            VectorizedResult
        """
        if isinstance(weights, dict):
            weights = pd.Series(weights)

        # Ensure weights sum to 1
        weights = weights / weights.sum()

        # Calculate returns
        returns = prices.pct_change().dropna()

        # Portfolio returns
        port_returns = (returns * weights).sum(axis=1)

        # Apply rebalancing costs
        rebalance_dates = returns.resample(rebalance_freq).last().index
        for date in rebalance_dates:
            if date in port_returns.index:
                port_returns.loc[date] -= self.commission * 2  # Round trip

        # Build equity curve
        equity = self.initial_capital * (1 + port_returns).cumprod()

        # Calculate metrics
        positions = pd.Series(1, index=returns.index)  # Always invested
        signals = positions.copy()

        return self.backtester._calculate_metrics(
            equity, port_returns, positions, signals, prices.iloc[:, 0]
        )

    def compare_assets(
        self,
        prices: pd.DataFrame,
        strategy: Literal["sma", "rsi", "bollinger", "momentum", "macd"] = "sma",
        **strategy_params,
    ) -> pd.DataFrame:
        """
        Run same strategy on multiple assets and compare.

        Args:
            prices: DataFrame of asset prices
            strategy: Strategy name
            **strategy_params: Strategy parameters

        Returns:
            DataFrame comparing results across assets
        """
        strategy_funcs = {
            "sma": self.backtester.run_sma_crossover,
            "rsi": self.backtester.run_rsi_strategy,
            "bollinger": self.backtester.run_bollinger_strategy,
            "momentum": self.backtester.run_momentum_strategy,
            "macd": self.backtester.run_macd_strategy,
        }

        func = strategy_funcs[strategy]
        results = []

        for col in prices.columns:
            try:
                result = func(prices[col], **strategy_params)
                results.append({
                    "symbol": col,
                    "total_return": result.total_return,
                    "annual_return": result.annual_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "sortino_ratio": result.sortino_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "total_trades": result.total_trades,
                })
            except Exception as e:
                print(f"Error processing {col}: {e}")

        return pd.DataFrame(results)
