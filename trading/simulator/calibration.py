"""
Market model calibration from real feed observations.

Provides tools to:
- Collect and store market data observations
- Estimate model parameters from historical data
- Validate model fit against real data
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
import math
import statistics
from collections import deque

from trading.core.models import Tick, OHLCV
from trading.simulator.market_model import MarketModelConfig


@dataclass
class FeedObservation:
    """Single market data observation for calibration."""
    timestamp: datetime
    symbol: str

    # Price data
    bid: float
    ask: float
    mid: float = field(init=False)
    spread: float = field(init=False)

    # Volume (if available)
    volume: Optional[float] = None

    # Trade info (if available)
    last_price: Optional[float] = None
    last_size: Optional[float] = None

    def __post_init__(self):
        self.mid = (self.bid + self.ask) / 2
        self.spread = self.ask - self.bid

    @classmethod
    def from_tick(cls, tick: Tick) -> "FeedObservation":
        """Create observation from Tick."""
        return cls(
            timestamp=tick.timestamp,
            symbol=tick.symbol,
            bid=float(tick.bid),
            ask=float(tick.ask),
            volume=float(tick.volume) if tick.volume else None,
            last_price=float(tick.last_price) if tick.last_price else None,
            last_size=float(tick.last_size) if tick.last_size else None,
        )


@dataclass
class CalibrationResult:
    """Results from model calibration."""
    symbol: str
    calibration_time: datetime

    # Estimated parameters
    volatility: float           # Annualized volatility
    drift: float                # Annualized drift
    spread_pct: float           # Average spread as percentage
    spread_std: float           # Spread standard deviation

    # Volume statistics
    avg_volume: float
    volume_std: float

    # Jump detection
    has_jumps: bool
    jump_intensity: float       # Jumps per year
    jump_mean: float            # Mean jump size
    jump_std: float             # Jump size volatility

    # Mean reversion (if detected)
    mean_reverting: bool
    mean_reversion_speed: float
    mean_reversion_level: float

    # Calibration quality metrics
    num_observations: int
    observation_period_days: float
    r_squared: float            # Model fit quality

    # Recommended model
    recommended_model: str      # "gbm", "jump_diffusion", "ou"

    def to_model_config(self) -> MarketModelConfig:
        """Convert calibration results to model config."""
        return MarketModelConfig(
            volatility=self.volatility,
            drift=self.drift,
            spread_pct=self.spread_pct,
            spread_variability=self.spread_std / self.spread_pct if self.spread_pct > 0 else 0.5,
            jump_intensity=self.jump_intensity if self.has_jumps else 0.0,
            jump_mean=self.jump_mean if self.has_jumps else 0.0,
            jump_std=self.jump_std if self.has_jumps else 0.01,
            mean_reversion_speed=self.mean_reversion_speed if self.mean_reverting else 0.0,
            mean_reversion_level=self.mean_reversion_level if self.mean_reverting else 0.0,
            avg_volume_per_bar=self.avg_volume,
            volume_volatility=self.volume_std / self.avg_volume if self.avg_volume > 0 else 0.5,
        )


class MarketCalibrator:
    """
    Calibrates market models from real feed observations.

    Collects tick/bar data and estimates parameters for simulation.

    Usage:
        calibrator = MarketCalibrator(symbol="BTCUSDT")

        # Feed observations
        async for tick in exchange.stream_ticks():
            calibrator.add_observation(FeedObservation.from_tick(tick))

        # Or from historical bars
        for bar in await exchange.get_historical_bars(...):
            calibrator.add_bar(bar)

        # Calibrate
        result = calibrator.calibrate()
        config = result.to_model_config()
    """

    def __init__(
        self,
        symbol: str,
        max_observations: int = 100000,
        min_observations: int = 100,
    ):
        self.symbol = symbol
        self.max_observations = max_observations
        self.min_observations = min_observations

        # Store observations
        self._observations: deque[FeedObservation] = deque(maxlen=max_observations)
        self._bars: list[OHLCV] = []

        # Cached calculations
        self._returns: list[float] = []
        self._log_returns: list[float] = []

    def add_observation(self, obs: FeedObservation) -> None:
        """Add a tick observation."""
        if obs.symbol != self.symbol:
            return
        self._observations.append(obs)
        self._returns = []  # Invalidate cache

    def add_bar(self, bar: OHLCV) -> None:
        """Add an OHLCV bar observation."""
        if bar.symbol != self.symbol:
            return
        self._bars.append(bar)
        self._returns = []

    def add_bars(self, bars: list[OHLCV]) -> None:
        """Add multiple OHLCV bars."""
        for bar in bars:
            self.add_bar(bar)

    def clear(self) -> None:
        """Clear all observations."""
        self._observations.clear()
        self._bars.clear()
        self._returns = []
        self._log_returns = []

    def _compute_returns(self) -> tuple[list[float], list[float]]:
        """Compute returns from observations."""
        if self._returns:
            return self._returns, self._log_returns

        prices = []
        if self._bars:
            prices = [float(bar.close) for bar in self._bars]
        elif self._observations:
            prices = [obs.mid for obs in self._observations]

        if len(prices) < 2:
            return [], []

        self._returns = []
        self._log_returns = []

        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                ret = (prices[i] - prices[i - 1]) / prices[i - 1]
                self._returns.append(ret)

                log_ret = math.log(prices[i] / prices[i - 1])
                self._log_returns.append(log_ret)

        return self._returns, self._log_returns

    def _estimate_volatility(self, log_returns: list[float], periods_per_year: float) -> float:
        """Estimate annualized volatility from log returns."""
        if len(log_returns) < 2:
            return 0.20  # Default

        std = statistics.stdev(log_returns)
        return std * math.sqrt(periods_per_year)

    def _estimate_drift(self, log_returns: list[float], periods_per_year: float) -> float:
        """Estimate annualized drift from log returns."""
        if not log_returns:
            return 0.0

        mean_return = statistics.mean(log_returns)
        return mean_return * periods_per_year

    def _estimate_spread(self) -> tuple[float, float]:
        """Estimate spread statistics."""
        if not self._observations:
            return 0.001, 0.0005  # Default

        spreads = []
        spread_pcts = []

        for obs in self._observations:
            if obs.mid > 0:
                spread_pcts.append(obs.spread / obs.mid)
            spreads.append(obs.spread)

        if not spread_pcts:
            return 0.001, 0.0005

        avg_spread_pct = statistics.mean(spread_pcts)
        spread_std = statistics.stdev(spread_pcts) if len(spread_pcts) > 1 else avg_spread_pct * 0.5

        return avg_spread_pct, spread_std

    def _estimate_volume(self) -> tuple[float, float]:
        """Estimate volume statistics."""
        volumes = []

        if self._bars:
            volumes = [float(bar.volume) for bar in self._bars if bar.volume > 0]
        elif self._observations:
            volumes = [obs.volume for obs in self._observations if obs.volume]

        if not volumes:
            return 1000000.0, 500000.0  # Default

        avg_vol = statistics.mean(volumes)
        vol_std = statistics.stdev(volumes) if len(volumes) > 1 else avg_vol * 0.5

        return avg_vol, vol_std

    def _detect_jumps(self, log_returns: list[float], volatility: float) -> tuple[bool, float, float, float]:
        """
        Detect presence of jumps in the data.

        Uses excess kurtosis to detect fat tails indicative of jumps.

        Returns:
            (has_jumps, intensity, mean, std)
        """
        if len(log_returns) < 30:
            return False, 0.0, 0.0, 0.01

        # Calculate excess kurtosis
        n = len(log_returns)
        mean = statistics.mean(log_returns)
        std = statistics.stdev(log_returns)

        if std == 0:
            return False, 0.0, 0.0, 0.01

        # Fourth moment for kurtosis
        fourth_moment = sum((r - mean) ** 4 for r in log_returns) / n
        kurtosis = (fourth_moment / (std ** 4)) - 3  # Excess kurtosis

        # Normal distribution has kurtosis = 0
        # Kurtosis > 1 suggests fat tails (jumps)
        has_jumps = kurtosis > 1.0

        if not has_jumps:
            return False, 0.0, 0.0, 0.01

        # Estimate jump parameters
        # Identify outliers as potential jumps (>3 std)
        threshold = 3 * std
        jumps = [r for r in log_returns if abs(r - mean) > threshold]

        if not jumps:
            return False, 0.0, 0.0, 0.01

        # Jump intensity (annualized)
        periods_per_year = 252 if len(self._bars) > 0 else 365 * 24 * 3600  # Assume daily bars or tick data
        jump_intensity = (len(jumps) / n) * periods_per_year

        # Jump size distribution
        jump_mean = statistics.mean(jumps)
        jump_std = statistics.stdev(jumps) if len(jumps) > 1 else abs(jump_mean) * 0.5

        return True, jump_intensity, jump_mean, jump_std

    def _detect_mean_reversion(
        self,
        log_returns: list[float],
        volatility: float
    ) -> tuple[bool, float, float]:
        """
        Detect mean reversion using autocorrelation.

        Negative autocorrelation suggests mean reversion.

        Returns:
            (is_mean_reverting, speed, level)
        """
        if len(log_returns) < 30:
            return False, 0.0, 0.0

        # Calculate first-order autocorrelation
        n = len(log_returns)
        mean = statistics.mean(log_returns)

        numerator = sum(
            (log_returns[i] - mean) * (log_returns[i - 1] - mean)
            for i in range(1, n)
        )
        denominator = sum((r - mean) ** 2 for r in log_returns)

        if denominator == 0:
            return False, 0.0, 0.0

        autocorr = numerator / denominator

        # Negative autocorrelation indicates mean reversion
        is_mean_reverting = autocorr < -0.1

        if not is_mean_reverting:
            return False, 0.0, 0.0

        # Estimate OU parameters
        # Speed of mean reversion from autocorrelation
        # For OU: autocorr ≈ exp(-θ * dt)
        dt = 1.0 / 252  # Assume daily observations
        theta = -math.log(abs(autocorr) + 0.01) / dt if autocorr != 0 else 0

        # Mean reversion level (use recent average price)
        prices = []
        if self._bars:
            prices = [float(bar.close) for bar in self._bars[-100:]]
        elif self._observations:
            obs_list = list(self._observations)[-1000:]
            prices = [obs.mid for obs in obs_list]

        level = statistics.mean(prices) if prices else 100.0

        return True, theta, level

    def _calculate_r_squared(
        self,
        log_returns: list[float],
        volatility: float,
        drift: float
    ) -> float:
        """
        Calculate R-squared for model fit.

        Compares expected return distribution to actual.
        """
        if len(log_returns) < 10:
            return 0.0

        # Expected variance under GBM
        dt = 1.0 / 252  # Assume daily
        expected_var = volatility ** 2 * dt

        # Actual variance
        actual_var = statistics.variance(log_returns)

        if actual_var == 0:
            return 0.0

        # R-squared based on variance explanation
        r_squared = 1 - abs(expected_var - actual_var) / actual_var
        return max(0.0, min(1.0, r_squared))

    def calibrate(self) -> CalibrationResult:
        """
        Calibrate model parameters from collected observations.

        Returns:
            CalibrationResult with estimated parameters.

        Raises:
            ValueError: If insufficient observations.
        """
        total_obs = len(self._observations) + len(self._bars)

        if total_obs < self.min_observations:
            raise ValueError(
                f"Insufficient observations: {total_obs} < {self.min_observations}"
            )

        # Determine time scale
        if self._bars:
            periods_per_year = 252  # Assume daily bars
            start_time = self._bars[0].timestamp
            end_time = self._bars[-1].timestamp
        else:
            obs_list = list(self._observations)
            # Estimate from tick timestamps
            if len(obs_list) >= 2:
                time_diff = (obs_list[-1].timestamp - obs_list[0].timestamp).total_seconds()
                avg_interval = time_diff / (len(obs_list) - 1)
                periods_per_year = 365.25 * 24 * 3600 / avg_interval if avg_interval > 0 else 252
            else:
                periods_per_year = 252
            start_time = obs_list[0].timestamp if obs_list else datetime.utcnow()
            end_time = obs_list[-1].timestamp if obs_list else datetime.utcnow()

        # Compute returns
        returns, log_returns = self._compute_returns()

        if not log_returns:
            raise ValueError("Could not compute returns from observations")

        # Estimate basic parameters
        volatility = self._estimate_volatility(log_returns, periods_per_year)
        drift = self._estimate_drift(log_returns, periods_per_year)
        spread_pct, spread_std = self._estimate_spread()
        avg_volume, volume_std = self._estimate_volume()

        # Detect jumps
        has_jumps, jump_intensity, jump_mean, jump_std = self._detect_jumps(
            log_returns, volatility
        )

        # Detect mean reversion
        mean_reverting, mr_speed, mr_level = self._detect_mean_reversion(
            log_returns, volatility
        )

        # Calculate fit quality
        r_squared = self._calculate_r_squared(log_returns, volatility, drift)

        # Determine recommended model
        if mean_reverting and mr_speed > 1.0:
            recommended_model = "ou"
        elif has_jumps and jump_intensity > 5:
            recommended_model = "jump_diffusion"
        else:
            recommended_model = "gbm"

        # Calculate observation period
        period_days = (end_time - start_time).total_seconds() / 86400

        return CalibrationResult(
            symbol=self.symbol,
            calibration_time=datetime.utcnow(),
            volatility=volatility,
            drift=drift,
            spread_pct=spread_pct,
            spread_std=spread_std,
            avg_volume=avg_volume,
            volume_std=volume_std,
            has_jumps=has_jumps,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_std=jump_std,
            mean_reverting=mean_reverting,
            mean_reversion_speed=mr_speed,
            mean_reversion_level=mr_level,
            num_observations=total_obs,
            observation_period_days=period_days,
            r_squared=r_squared,
            recommended_model=recommended_model,
        )


class RealTimeCalibrator:
    """
    Continuously calibrates model parameters from streaming data.

    Uses exponential moving average for online parameter updates.
    """

    def __init__(
        self,
        symbol: str,
        half_life_seconds: float = 3600,  # 1 hour half-life
        min_samples: int = 100,
    ):
        self.symbol = symbol
        self.half_life = half_life_seconds
        self.min_samples = min_samples

        # Decay factor
        self._alpha = math.log(2) / half_life_seconds

        # Running statistics (exponentially weighted)
        self._last_price: Optional[float] = None
        self._last_time: Optional[datetime] = None
        self._ema_return: float = 0.0
        self._ema_return_sq: float = 0.0
        self._ema_spread: float = 0.001
        self._sample_count: int = 0

    def update(self, obs: FeedObservation) -> Optional[MarketModelConfig]:
        """
        Update calibration with new observation.

        Returns updated config if enough samples collected.
        """
        if obs.symbol != self.symbol:
            return None

        self._sample_count += 1

        # Update spread estimate
        if obs.mid > 0:
            spread_pct = obs.spread / obs.mid
            self._ema_spread = self._ema_spread * 0.99 + spread_pct * 0.01

        # Calculate return if we have previous price
        if self._last_price and self._last_time:
            dt = (obs.timestamp - self._last_time).total_seconds()

            if dt > 0 and self._last_price > 0:
                log_return = math.log(obs.mid / self._last_price)

                # Exponential decay weight
                weight = 1 - math.exp(-self._alpha * dt)

                # Update running statistics
                self._ema_return = (1 - weight) * self._ema_return + weight * log_return
                self._ema_return_sq = (1 - weight) * self._ema_return_sq + weight * (log_return ** 2)

        self._last_price = obs.mid
        self._last_time = obs.timestamp

        # Return config if enough samples
        if self._sample_count >= self.min_samples:
            return self._get_config()

        return None

    def _get_config(self) -> MarketModelConfig:
        """Get current estimated config."""
        # Variance from E[X^2] - E[X]^2
        var = self._ema_return_sq - self._ema_return ** 2
        var = max(0, var)  # Ensure non-negative

        # Annualize (assuming ~1 second updates)
        annualization = math.sqrt(365.25 * 24 * 3600)
        volatility = math.sqrt(var) * annualization
        drift = self._ema_return * 365.25 * 24 * 3600

        return MarketModelConfig(
            volatility=max(0.01, volatility),  # Minimum 1% volatility
            drift=drift,
            spread_pct=self._ema_spread,
        )

    @property
    def is_calibrated(self) -> bool:
        """Check if enough samples collected."""
        return self._sample_count >= self.min_samples
