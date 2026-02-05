"""
Standard Technical Indicators Library

Vectorized implementations for fast backtesting.
All functions return pd.Series with same index as input.
"""

import numpy as np
import pandas as pd


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Args:
        prices: Close prices
        period: Lookback period (default 14, use 2 for Connors-style)

    Returns:
        RSI values 0-100
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    # Wilder smoothing (EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def cumulative_rsi(prices: pd.Series, rsi_period: int = 2, cum_period: int = 2) -> pd.Series:
    """
    Cumulative RSI - sum of RSI over multiple days.

    Used for deeper oversold/overbought detection.
    """
    rsi_val = rsi(prices, rsi_period)
    return rsi_val.rolling(cum_period).sum()


def ibs(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Internal Bar Strength.

    IBS = (Close - Low) / (High - Low)

    Values:
        0.0 = Close at day's low (bearish close, bullish signal)
        1.0 = Close at day's high (bullish close, bearish signal)

    Mean-reversion: Buy when IBS < 0.2, sell when IBS > 0.8
    """
    return (close - low) / (high - low + 1e-10)


def momersion(returns: pd.Series, lookback: int = 252) -> pd.Series:
    """
    PriceActionLab Momersion Indicator (vectorized).

    Measures the ratio of momentum vs mean-reversion behavior.

    Formula:
        Momersion = 100 * MOMc / (MOMc + MRc)

        Where:
        - MOMc = count of same-sign consecutive returns (momentum)
        - MRc = count of opposite-sign consecutive returns (mean-reversion)

    Interpretation:
        > 50: Momentum-dominated regime
        < 50: Mean-reversion-dominated regime
        = 50: Mixed/transitioning regime

    Args:
        returns: Daily returns (not prices!)
        lookback: Window size (default 252 = 1 year)

    Source: https://www.priceactionlab.com/Blog/2015/08/momersion-indicator/
    """
    # Product of consecutive returns
    product = returns * returns.shift(1)

    # Momentum: same sign (product > 0)
    momentum = (product > 0).astype(float)

    # Mean-reversion: opposite sign (product < 0)
    meanrev = (product < 0).astype(float)

    # Rolling counts
    mom_count = momentum.rolling(lookback).sum()
    mr_count = meanrev.rolling(lookback).sum()

    total = mom_count + mr_count
    return 100 * mom_count / total.replace(0, np.nan)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range.

    Measures volatility as the average of true ranges.
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def atr_percent(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR as percentage of close price."""
    return atr(high, low, close, period) / close * 100


def consecutive_days(returns: pd.Series, direction: str = "down") -> pd.Series:
    """
    Count consecutive up or down days.

    Args:
        returns: Daily returns
        direction: "up" or "down"

    Returns:
        Series with count of consecutive days in that direction
    """
    if direction == "down":
        signal = (returns < 0).astype(int)
    else:
        signal = (returns > 0).astype(int)

    # Reset count when direction changes
    result = pd.Series(0, index=returns.index)

    for i in range(1, len(returns)):
        if signal.iloc[i] == 1:
            result.iloc[i] = result.iloc[i-1] + 1
        else:
            result.iloc[i] = 0

    return result


def rate_of_change(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Rate of Change (momentum indicator).

    ROC = (Price / Price[n]) - 1
    """
    return prices / prices.shift(period) - 1


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Williams %R.

    Similar to stochastic but inverted scale.
    Range: -100 to 0 (oversold < -80, overbought > -20)
    """
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest + 1e-10)


def bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Returns:
        (middle_band, upper_band, lower_band)
    """
    middle = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return middle, upper, lower


def bollinger_pct_b(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Bollinger %B - position within bands.

    %B = (Price - Lower) / (Upper - Lower)

    Values:
        > 1.0: Above upper band
        < 0.0: Below lower band
        0.5: At middle band
    """
    _, upper, lower = bollinger_bands(prices, period, num_std)
    return (prices - lower) / (upper - lower + 1e-10)


def donchian_channel(
    high: pd.Series,
    low: pd.Series,
    period: int = 20
) -> tuple[pd.Series, pd.Series]:
    """
    Donchian Channel (breakout indicator).

    Returns:
        (upper_channel, lower_channel)
    """
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    return upper, lower


def keltner_channel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Keltner Channel.

    Returns:
        (middle, upper, lower)
    """
    middle = close.ewm(span=ema_period).mean()
    atr_val = atr(high, low, close, atr_period)
    upper = middle + (atr_val * atr_mult)
    lower = middle - (atr_val * atr_mult)
    return middle, upper, lower


# ==============================================================================
# Regime Detection Helpers
# ==============================================================================

def regime_momersion(returns: pd.Series, lookback: int = 252, threshold: float = 50) -> pd.Series:
    """
    Detect market regime using Momersion.

    Returns:
        1 = Momentum regime (Momersion > threshold)
        0 = Mean-reversion regime (Momersion <= threshold)
    """
    mom = momersion(returns, lookback)
    return (mom > threshold).astype(int)


def regime_volatility(
    prices: pd.Series,
    short_period: int = 20,
    long_period: int = 100,
) -> pd.Series:
    """
    Detect volatility regime.

    Returns:
        1 = High volatility (short vol > long vol)
        0 = Low volatility
    """
    short_vol = prices.pct_change().rolling(short_period).std()
    long_vol = prices.pct_change().rolling(long_period).std()
    return (short_vol > long_vol).astype(int)


# ==============================================================================
# Performance Metrics
# ==============================================================================

def sharpe_ratio(returns: pd.Series, rf: float = 0.02, annualize: bool = True) -> float:
    """Calculate Sharpe ratio."""
    excess = returns.mean() - rf / 252
    vol = returns.std()
    if annualize:
        return (excess * 252) / (vol * np.sqrt(252)) if vol > 0 else 0
    return excess / vol if vol > 0 else 0


def max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown (as negative percentage)."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    return drawdown.min()


def cagr(returns: pd.Series) -> float:
    """Calculate Compound Annual Growth Rate."""
    total = (1 + returns).prod() - 1
    years = len(returns) / 252
    return ((1 + total) ** (1 / years) - 1) if years > 0 else 0


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Calculate beta vs benchmark."""
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if len(aligned) < 30:
        return 0
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].var()
    return cov / var if var > 0 else 0
