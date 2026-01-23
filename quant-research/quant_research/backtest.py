"""
Backtesting module for strategy evaluation.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class BacktestResult:
    """Container for backtest results."""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    equity_curve: pd.Series
    trades: pd.DataFrame


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.

        Args:
            data: OHLCV DataFrame

        Returns:
            Series of signals: 1 (buy), -1 (sell), 0 (hold)
        """
        pass


class SMAcrossover(Strategy):
    """Simple Moving Average Crossover Strategy."""

    def __init__(self, short_period: int = 20, long_period: int = 50):
        self.short_period = short_period
        self.long_period = long_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on SMA crossover."""
        close = data["close"]

        short_sma = close.rolling(window=self.short_period).mean()
        long_sma = close.rolling(window=self.long_period).mean()

        signals = pd.Series(0, index=data.index)

        # Buy when short crosses above long
        signals[short_sma > long_sma] = 1
        # Sell when short crosses below long
        signals[short_sma <= long_sma] = -1

        return signals


class RSIMeanReversion(Strategy):
    """RSI Mean Reversion Strategy."""

    def __init__(
        self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0
    ):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on RSI levels."""
        close = data["close"]

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        signals = pd.Series(0, index=data.index)

        # Buy when oversold
        signals[rsi < self.oversold] = 1
        # Sell when overbought
        signals[rsi > self.overbought] = -1

        return signals


class Backtester:
    """Backtesting engine for evaluating trading strategies."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade
            risk_free_rate: Annual risk-free rate for Sharpe ratio
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.risk_free_rate = risk_free_rate

    def run(self, data: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        """
        Run backtest for a given strategy.

        Args:
            data: OHLCV DataFrame
            strategy: Trading strategy instance

        Returns:
            BacktestResult with performance metrics
        """
        signals = strategy.generate_signals(data)
        close = data["close"]

        # Track positions and trades
        positions = pd.Series(0.0, index=data.index, dtype=float)
        cash = pd.Series(float(self.initial_capital), index=data.index, dtype=float)
        holdings = pd.Series(0.0, index=data.index, dtype=float)
        trades_list = []

        position = 0
        current_cash = self.initial_capital
        shares = 0

        for i, (idx, signal) in enumerate(signals.items()):
            price = close.loc[idx]

            if i > 0:
                prev_idx = signals.index[i - 1]
                cash.loc[idx] = current_cash
                holdings.loc[idx] = shares * price
            else:
                cash.loc[idx] = current_cash
                holdings.loc[idx] = 0

            # Position changes
            if signal == 1 and position <= 0:  # Buy
                if position < 0:  # Close short first
                    current_cash -= shares * price * (1 + self.commission)
                    trades_list.append(
                        {"date": idx, "type": "cover", "price": price, "shares": shares}
                    )
                    shares = 0

                # Open long
                shares_to_buy = int(current_cash * 0.95 / (price * (1 + self.commission)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * price * (1 + self.commission)
                    current_cash -= cost
                    shares = shares_to_buy
                    position = 1
                    trades_list.append(
                        {
                            "date": idx,
                            "type": "buy",
                            "price": price,
                            "shares": shares_to_buy,
                        }
                    )

            elif signal == -1 and position >= 0:  # Sell
                if position > 0:  # Close long first
                    proceeds = shares * price * (1 - self.commission)
                    current_cash += proceeds
                    trades_list.append(
                        {"date": idx, "type": "sell", "price": price, "shares": shares}
                    )
                    shares = 0
                    position = -1

            positions.loc[idx] = position
            cash.loc[idx] = current_cash
            holdings.loc[idx] = shares * price

        # Calculate equity curve
        equity = cash + holdings

        # Performance metrics
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital

        # Annualized return
        days = (equity.index[-1] - equity.index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0.0

        # Volatility
        daily_returns = equity.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)

        # Sharpe ratio
        if volatility > 0:
            sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Win rate
        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
        if len(trades_df) >= 2:
            # Calculate wins
            buy_trades = trades_df[trades_df["type"] == "buy"]["price"].values
            sell_trades = trades_df[trades_df["type"] == "sell"]["price"].values
            min_len = min(len(buy_trades), len(sell_trades))
            if min_len > 0:
                wins = sum(sell_trades[:min_len] > buy_trades[:min_len])
                win_rate = wins / min_len
            else:
                win_rate = 0.0
        else:
            win_rate = 0.0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades_list),
            equity_curve=equity,
            trades=trades_df,
        )

    def compare_strategies(
        self, data: pd.DataFrame, strategies: dict[str, Strategy]
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Args:
            data: OHLCV DataFrame
            strategies: Dictionary of strategy name to Strategy instance

        Returns:
            DataFrame comparing strategy performance
        """
        results = {}
        for name, strategy in strategies.items():
            result = self.run(data, strategy)
            results[name] = {
                "Total Return": f"{result.total_return:.2%}",
                "Ann. Return": f"{result.annualized_return:.2%}",
                "Volatility": f"{result.volatility:.2%}",
                "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
                "Max Drawdown": f"{result.max_drawdown:.2%}",
                "Win Rate": f"{result.win_rate:.2%}",
                "Total Trades": result.total_trades,
            }

        return pd.DataFrame(results).T
