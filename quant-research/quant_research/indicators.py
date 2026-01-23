"""
Technical indicators module for quantitative analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional


class TechnicalIndicators:
    """Calculate technical indicators for financial time series."""

    @staticmethod
    def sma(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Simple Moving Average.

        Args:
            data: Price series
            period: Lookback period

        Returns:
            SMA series
        """
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Exponential Moving Average.

        Args:
            data: Price series
            period: Lookback period

        Returns:
            EMA series
        """
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.

        Args:
            data: Price series
            period: Lookback period

        Returns:
            RSI series (0-100)
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence.

        Args:
            data: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        fast_ema = data.ewm(span=fast_period, adjust=False).mean()
        slow_ema = data.ewm(span=slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        data: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.

        Args:
            data: Price series
            period: Moving average period
            std_dev: Number of standard deviations

        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Average True Range.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Lookback period

        Returns:
            ATR series
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    @staticmethod
    def returns(data: pd.Series, periods: int = 1) -> pd.Series:
        """
        Calculate percentage returns.

        Args:
            data: Price series
            periods: Number of periods for return calculation

        Returns:
            Returns series
        """
        return data.pct_change(periods=periods)

    @staticmethod
    def log_returns(data: pd.Series, periods: int = 1) -> pd.Series:
        """
        Calculate logarithmic returns.

        Args:
            data: Price series
            periods: Number of periods for return calculation

        Returns:
            Log returns series
        """
        return np.log(data / data.shift(periods))

    @staticmethod
    def volatility(data: pd.Series, period: int = 20, annualize: bool = True) -> pd.Series:
        """
        Calculate rolling volatility.

        Args:
            data: Price series
            period: Lookback period
            annualize: Whether to annualize (assumes 252 trading days)

        Returns:
            Volatility series
        """
        log_returns = np.log(data / data.shift(1))
        vol = log_returns.rolling(window=period).std()

        if annualize:
            vol = vol * np.sqrt(252)

        return vol

    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Price momentum.

        Args:
            data: Price series
            period: Lookback period

        Returns:
            Momentum series
        """
        return data - data.shift(period)

    @staticmethod
    def rate_of_change(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Rate of Change (ROC).

        Args:
            data: Price series
            period: Lookback period

        Returns:
            ROC series (percentage)
        """
        return ((data - data.shift(period)) / data.shift(period)) * 100
