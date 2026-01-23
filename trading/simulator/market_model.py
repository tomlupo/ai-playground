"""
Market dynamics models for price simulation.

Provides configurable models for simulating realistic market behavior:
- Geometric Brownian Motion (GBM) - Standard model for equities
- Jump Diffusion (Merton) - For assets with occasional large moves
- Ornstein-Uhlenbeck (OU) - Mean-reverting model for spreads/rates
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
import math
import random


@dataclass
class MarketModelConfig:
    """Configuration for market price dynamics."""
    # Base volatility (annualized, e.g., 0.20 = 20%)
    volatility: float = 0.20

    # Drift/trend (annualized, e.g., 0.05 = 5% expected return)
    drift: float = 0.0

    # Bid-ask spread as percentage of price
    spread_pct: float = 0.001  # 10 basis points

    # Spread variability (multiplier range)
    spread_variability: float = 0.5  # Spread can vary 0.5x to 1.5x

    # Tick frequency for price updates (seconds)
    tick_interval: float = 1.0

    # Jump parameters (for jump diffusion)
    jump_intensity: float = 0.0  # Average jumps per year (0 = no jumps)
    jump_mean: float = 0.0  # Mean jump size (log)
    jump_std: float = 0.01  # Jump size volatility

    # Mean reversion parameters (for OU process)
    mean_reversion_speed: float = 0.0  # 0 = no mean reversion
    mean_reversion_level: float = 0.0  # Long-term mean

    # Volume parameters
    avg_volume_per_bar: float = 1000000.0
    volume_volatility: float = 0.5

    # Order book depth parameters
    book_depth_levels: int = 10
    level_size_decay: float = 0.8  # Each level has 80% of previous

    # Slippage parameters
    base_slippage_bps: float = 1.0  # Base slippage in basis points
    impact_coefficient: float = 0.1  # Price impact per unit of volume

    # Latency simulation (milliseconds)
    min_latency_ms: float = 1.0
    max_latency_ms: float = 50.0

    # Market hours (optional - None = 24/7)
    market_open: Optional[str] = None  # "09:30"
    market_close: Optional[str] = None  # "16:00"
    timezone: str = "UTC"


class MarketModel(ABC):
    """Abstract base class for market price models."""

    def __init__(self, config: MarketModelConfig):
        self.config = config
        self._last_price: float = 100.0
        self._last_update: datetime = datetime.utcnow()
        self._random = random.Random()

    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._random.seed(seed)

    @abstractmethod
    def step(self, dt: float) -> float:
        """
        Advance the price process by dt seconds.

        Args:
            dt: Time step in seconds.

        Returns:
            New price after the step.
        """
        pass

    def set_price(self, price: float) -> None:
        """Set the current price."""
        self._last_price = price

    @property
    def price(self) -> float:
        """Current mid-price."""
        return self._last_price

    @property
    def bid(self) -> float:
        """Current bid price."""
        spread = self._last_price * self.config.spread_pct
        variability = 1.0 + (self._random.random() - 0.5) * self.config.spread_variability
        return self._last_price - (spread * variability / 2)

    @property
    def ask(self) -> float:
        """Current ask price."""
        spread = self._last_price * self.config.spread_pct
        variability = 1.0 + (self._random.random() - 0.5) * self.config.spread_variability
        return self._last_price + (spread * variability / 2)

    def get_fill_price(
        self,
        side: str,
        quantity: float,
        order_type: str = "MARKET"
    ) -> float:
        """
        Calculate fill price including slippage and market impact.

        Args:
            side: "BUY" or "SELL"
            quantity: Order quantity
            order_type: Order type (MARKET has more slippage)

        Returns:
            Execution price after slippage.
        """
        base_price = self.ask if side == "BUY" else self.bid

        # Calculate slippage
        slippage_bps = self.config.base_slippage_bps
        if order_type == "MARKET":
            slippage_bps *= 2  # Market orders have more slippage

        # Add market impact based on order size
        impact = self.config.impact_coefficient * math.sqrt(quantity / self.config.avg_volume_per_bar)
        total_slippage = (slippage_bps / 10000) + impact

        if side == "BUY":
            return base_price * (1 + total_slippage)
        else:
            return base_price * (1 - total_slippage)

    def generate_volume(self, dt: float) -> float:
        """Generate random volume for a time period."""
        # Scale average volume to time period (assuming avg is daily)
        scaled_avg = self.config.avg_volume_per_bar * (dt / 86400)

        # Log-normal volume distribution
        log_vol = math.log(scaled_avg) + self._random.gauss(0, self.config.volume_volatility)
        return math.exp(log_vol)

    def get_latency(self) -> float:
        """Get simulated network latency in seconds."""
        latency_ms = self._random.uniform(
            self.config.min_latency_ms,
            self.config.max_latency_ms
        )
        return latency_ms / 1000

    def is_market_open(self, timestamp: datetime) -> bool:
        """Check if market is open at given time."""
        if not self.config.market_open or not self.config.market_close:
            return True  # 24/7 market

        # Simple implementation - assumes same timezone
        time_str = timestamp.strftime("%H:%M")
        return self.config.market_open <= time_str < self.config.market_close


class GBMModel(MarketModel):
    """
    Geometric Brownian Motion model.

    Standard model for equity prices:
    dS = μS dt + σS dW

    Where:
    - μ is the drift (expected return)
    - σ is the volatility
    - dW is a Wiener process increment
    """

    def step(self, dt: float) -> float:
        """Advance price using GBM."""
        # Convert dt from seconds to years
        dt_years = dt / (365.25 * 24 * 3600)

        # Generate random normal increment
        dW = self._random.gauss(0, 1) * math.sqrt(dt_years)

        # GBM update
        drift_term = (self.config.drift - 0.5 * self.config.volatility ** 2) * dt_years
        diffusion_term = self.config.volatility * dW

        self._last_price = self._last_price * math.exp(drift_term + diffusion_term)
        self._last_update = datetime.utcnow()

        return self._last_price


class JumpDiffusionModel(MarketModel):
    """
    Merton Jump Diffusion model.

    Extends GBM with occasional jumps:
    dS = μS dt + σS dW + S dJ

    Where dJ is a compound Poisson process with:
    - Jump intensity λ
    - Log-normally distributed jump sizes
    """

    def step(self, dt: float) -> float:
        """Advance price using Jump Diffusion."""
        dt_years = dt / (365.25 * 24 * 3600)

        # GBM component
        dW = self._random.gauss(0, 1) * math.sqrt(dt_years)
        drift_term = (self.config.drift - 0.5 * self.config.volatility ** 2) * dt_years
        diffusion_term = self.config.volatility * dW

        # Jump component
        jump_term = 0.0
        if self.config.jump_intensity > 0:
            # Number of jumps in this interval (Poisson)
            expected_jumps = self.config.jump_intensity * dt_years
            num_jumps = self._random.poisson(expected_jumps) if expected_jumps < 10 else int(
                self._random.gauss(expected_jumps, math.sqrt(expected_jumps))
            )

            # Simulate each jump
            for _ in range(max(0, num_jumps)):
                jump_size = self._random.gauss(self.config.jump_mean, self.config.jump_std)
                jump_term += jump_size

        self._last_price = self._last_price * math.exp(drift_term + diffusion_term + jump_term)
        self._last_update = datetime.utcnow()

        return self._last_price


class OUModel(MarketModel):
    """
    Ornstein-Uhlenbeck mean-reverting model.

    Useful for spread trading, rates, or mean-reverting assets:
    dX = θ(μ - X) dt + σ dW

    Where:
    - θ is the mean reversion speed
    - μ is the long-term mean level
    - σ is the volatility
    """

    def step(self, dt: float) -> float:
        """Advance using OU process."""
        dt_years = dt / (365.25 * 24 * 3600)

        theta = self.config.mean_reversion_speed
        mu = self.config.mean_reversion_level or self._last_price
        sigma = self.config.volatility

        # OU exact solution for discrete time
        if theta > 0:
            mean = self._last_price * math.exp(-theta * dt_years) + \
                   mu * (1 - math.exp(-theta * dt_years))
            var = (sigma ** 2) / (2 * theta) * (1 - math.exp(-2 * theta * dt_years))
        else:
            # Fallback to random walk if no mean reversion
            mean = self._last_price
            var = sigma ** 2 * dt_years

        self._last_price = self._random.gauss(mean, math.sqrt(var))
        self._last_update = datetime.utcnow()

        # Ensure price stays positive
        self._last_price = max(self._last_price, 0.01)

        return self._last_price


class RegimeSwitchingModel(MarketModel):
    """
    Regime-switching model with multiple market states.

    Supports different volatility regimes (e.g., calm vs. volatile markets).
    """

    def __init__(self, config: MarketModelConfig):
        super().__init__(config)

        # Define regimes: (volatility_multiplier, drift_adjustment, duration_mean)
        self.regimes = [
            {"name": "calm", "vol_mult": 0.5, "drift_adj": 0.02, "duration": 30},
            {"name": "normal", "vol_mult": 1.0, "drift_adj": 0.0, "duration": 60},
            {"name": "volatile", "vol_mult": 2.0, "drift_adj": -0.05, "duration": 10},
        ]

        self.current_regime = 1  # Start in normal
        self.regime_time_remaining = self.regimes[1]["duration"] * 86400  # seconds

    def _maybe_switch_regime(self, dt: float) -> None:
        """Possibly switch to a new regime."""
        self.regime_time_remaining -= dt

        if self.regime_time_remaining <= 0:
            # Transition probabilities (simplified)
            probs = [0.3, 0.5, 0.2]  # Favor normal regime
            self.current_regime = self._random.choices(range(3), probs)[0]
            regime = self.regimes[self.current_regime]
            self.regime_time_remaining = regime["duration"] * 86400 * (0.5 + self._random.random())

    def step(self, dt: float) -> float:
        """Advance price with regime-dependent dynamics."""
        self._maybe_switch_regime(dt)

        regime = self.regimes[self.current_regime]
        dt_years = dt / (365.25 * 24 * 3600)

        # Adjusted parameters for current regime
        vol = self.config.volatility * regime["vol_mult"]
        drift = self.config.drift + regime["drift_adj"]

        # GBM step with regime parameters
        dW = self._random.gauss(0, 1) * math.sqrt(dt_years)
        drift_term = (drift - 0.5 * vol ** 2) * dt_years
        diffusion_term = vol * dW

        self._last_price = self._last_price * math.exp(drift_term + diffusion_term)
        self._last_update = datetime.utcnow()

        return self._last_price


@dataclass
class SimulatedOrderBook:
    """Simulated limit order book."""
    symbol: str
    model: MarketModel
    levels: int = 10

    def generate(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Generate order book levels.

        Returns:
            (bids, asks) where each is list of (price, quantity) tuples
        """
        mid = self.model.price
        spread = mid * self.model.config.spread_pct
        half_spread = spread / 2

        bids = []
        asks = []
        base_size = self.model.config.avg_volume_per_bar / 1000

        for i in range(self.levels):
            # Price levels with increasing distance from mid
            level_dist = half_spread * (1 + i * 0.5)
            bid_price = mid - level_dist
            ask_price = mid + level_dist

            # Size decreases with distance from mid
            decay = self.model.config.level_size_decay ** i
            size_noise = 0.5 + self.model._random.random()
            level_size = base_size * decay * size_noise

            bids.append((bid_price, level_size))
            asks.append((ask_price, level_size))

        return bids, asks
