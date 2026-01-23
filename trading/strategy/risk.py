"""
Risk management and position sizing.

Provides:
- Position sizing algorithms
- Risk metrics calculation
- Portfolio-level risk management
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
import math
import statistics

from trading.core.models import Position, Order, OHLCV, Trade
from trading.core.enums import OrderSide, PositionSide


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Position limits
    max_position_size: Decimal = Decimal("100000")  # Max notional per position
    max_position_pct: float = 0.10  # Max 10% of equity per position
    max_total_exposure: float = 1.0  # Max 100% total exposure

    # Loss limits
    max_loss_per_trade: Decimal = Decimal("1000")
    max_loss_per_trade_pct: float = 0.02  # 2% of equity
    daily_loss_limit: Decimal = Decimal("5000")
    daily_loss_limit_pct: float = 0.05  # 5% of equity

    # Drawdown limits
    max_drawdown_pct: float = 0.20  # 20% max drawdown
    trailing_stop_pct: float = 0.0  # 0 = disabled

    # Correlation limits
    max_correlated_positions: int = 3  # Max positions in correlated assets

    # Leverage
    max_leverage: float = 1.0
    margin_call_level: float = 0.50  # 50% margin level triggers warning

    # Volatility scaling
    volatility_target: float = 0.0  # 0 = disabled, otherwise annual vol target
    volatility_lookback_days: int = 20


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot."""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Exposure
    total_exposure: Decimal = Decimal("0")
    long_exposure: Decimal = Decimal("0")
    short_exposure: Decimal = Decimal("0")
    net_exposure: Decimal = Decimal("0")
    gross_exposure: Decimal = Decimal("0")

    # P&L
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl_today: Decimal = Decimal("0")
    daily_return_pct: float = 0.0

    # Risk measures
    current_drawdown: float = 0.0
    var_95: Decimal = Decimal("0")  # 95% Value at Risk
    expected_shortfall: Decimal = Decimal("0")  # Conditional VaR

    # Margin
    margin_used: Decimal = Decimal("0")
    margin_available: Decimal = Decimal("0")
    margin_level: float = 1.0

    # Limits
    daily_loss_remaining: Decimal = Decimal("0")
    position_limit_remaining: int = 0

    @property
    def leverage_ratio(self) -> float:
        """Current leverage ratio."""
        if self.margin_available + self.margin_used == 0:
            return 0.0
        return float(self.gross_exposure / (self.margin_available + self.margin_used))


class RiskManager:
    """
    Portfolio-level risk management.

    Monitors positions and enforces risk limits.

    Usage:
        config = RiskConfig(
            max_position_pct=0.05,  # 5% max per position
            daily_loss_limit_pct=0.02,  # 2% daily loss limit
        )
        risk_manager = RiskManager(config, initial_equity=100000)

        # Before placing order
        if risk_manager.can_open_position(symbol, size, price):
            # Place order
            pass

        # After fill
        risk_manager.record_fill(order, fill_price)

        # Check limits
        metrics = risk_manager.get_metrics()
        if metrics.daily_loss_remaining <= 0:
            # Stop trading for the day
            pass
    """

    def __init__(
        self,
        config: RiskConfig,
        initial_equity: Decimal = Decimal("100000"),
    ):
        self.config = config
        self._initial_equity = initial_equity
        self._current_equity = initial_equity
        self._peak_equity = initial_equity

        # Position tracking
        self._positions: dict[str, Position] = {}
        self._position_values: dict[str, Decimal] = {}

        # P&L tracking
        self._realized_pnl = Decimal("0")
        self._unrealized_pnl = Decimal("0")
        self._daily_pnl = Decimal("0")
        self._last_day: Optional[datetime] = None

        # Historical data for VaR
        self._returns_history: list[float] = []
        self._equity_history: list[tuple[datetime, Decimal]] = []

    def update_equity(self, equity: Decimal, timestamp: Optional[datetime] = None) -> None:
        """Update current equity value."""
        ts = timestamp or datetime.utcnow()

        # Track daily P&L
        if self._last_day and ts.date() > self._last_day.date():
            self._daily_pnl = Decimal("0")
        self._last_day = ts

        # Update equity curve
        if self._equity_history:
            last_equity = self._equity_history[-1][1]
            if last_equity > 0:
                daily_return = float((equity - last_equity) / last_equity)
                self._returns_history.append(daily_return)
                # Keep last 252 trading days
                self._returns_history = self._returns_history[-252:]

        self._equity_history.append((ts, equity))
        self._current_equity = equity

        # Update peak for drawdown
        if equity > self._peak_equity:
            self._peak_equity = equity

    def update_position(
        self,
        symbol: str,
        position: Optional[Position],
        current_price: Decimal,
    ) -> None:
        """Update position tracking."""
        if position and position.quantity > 0:
            self._positions[symbol] = position
            self._position_values[symbol] = position.quantity * current_price
        else:
            self._positions.pop(symbol, None)
            self._position_values.pop(symbol, None)

    def record_trade(self, pnl: Decimal) -> None:
        """Record a completed trade P&L."""
        self._realized_pnl += pnl
        self._daily_pnl += pnl

    # ==================== Risk Checks ====================

    def can_open_position(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        side: OrderSide,
    ) -> tuple[bool, str]:
        """
        Check if opening a position is within risk limits.

        Returns:
            (allowed, reason) tuple
        """
        notional = quantity * price

        # Check position size limit
        if notional > self.config.max_position_size:
            return False, f"Position size {notional} exceeds max {self.config.max_position_size}"

        # Check percentage limit
        max_by_pct = self._current_equity * Decimal(str(self.config.max_position_pct))
        if notional > max_by_pct:
            return False, f"Position size exceeds {self.config.max_position_pct*100}% of equity"

        # Check total exposure
        current_exposure = sum(self._position_values.values())
        new_exposure = current_exposure + notional
        max_exposure = self._current_equity * Decimal(str(self.config.max_total_exposure))
        if new_exposure > max_exposure:
            return False, "Total exposure would exceed limit"

        # Check daily loss limit
        if self.config.daily_loss_limit > 0:
            remaining = self.config.daily_loss_limit + self._daily_pnl
            if remaining <= 0:
                return False, "Daily loss limit reached"

        # Check drawdown
        current_drawdown = self._calculate_drawdown()
        if current_drawdown >= self.config.max_drawdown_pct:
            return False, f"Max drawdown {self.config.max_drawdown_pct*100}% reached"

        return True, ""

    def calculate_max_position_size(
        self,
        symbol: str,
        price: Decimal,
        stop_loss: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate maximum allowed position size.

        Args:
            symbol: Trading symbol
            price: Current price
            stop_loss: Optional stop loss price

        Returns:
            Maximum quantity allowed
        """
        # Start with max notional limit
        max_notional = min(
            self.config.max_position_size,
            self._current_equity * Decimal(str(self.config.max_position_pct)),
        )

        # Adjust for total exposure limit
        current_exposure = sum(self._position_values.values())
        exposure_headroom = self._current_equity * Decimal(str(self.config.max_total_exposure)) - current_exposure
        max_notional = min(max_notional, exposure_headroom)

        # If stop loss provided, size based on max loss per trade
        if stop_loss and stop_loss > 0:
            risk_per_unit = abs(price - stop_loss)
            max_loss = min(
                self.config.max_loss_per_trade,
                self._current_equity * Decimal(str(self.config.max_loss_per_trade_pct)),
            )
            if risk_per_unit > 0:
                risk_based_size = max_loss / risk_per_unit
                max_notional = min(max_notional, risk_based_size * price)

        # Convert to quantity
        if price > 0:
            return max_notional / price

        return Decimal("0")

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self._peak_equity == 0:
            return 0.0
        return float((self._peak_equity - self._current_equity) / self._peak_equity)

    def _calculate_var(self, confidence: float = 0.95) -> Decimal:
        """Calculate Value at Risk using historical simulation."""
        if len(self._returns_history) < 20:
            return Decimal("0")

        sorted_returns = sorted(self._returns_history)
        index = int((1 - confidence) * len(sorted_returns))
        var_return = sorted_returns[index]

        return self._current_equity * Decimal(str(abs(var_return)))

    def _calculate_expected_shortfall(self, confidence: float = 0.95) -> Decimal:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(self._returns_history) < 20:
            return Decimal("0")

        sorted_returns = sorted(self._returns_history)
        cutoff = int((1 - confidence) * len(sorted_returns))
        tail_returns = sorted_returns[:cutoff]

        if not tail_returns:
            return Decimal("0")

        avg_tail = statistics.mean(tail_returns)
        return self._current_equity * Decimal(str(abs(avg_tail)))

    def get_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics."""
        # Calculate exposures
        long_exposure = Decimal("0")
        short_exposure = Decimal("0")

        for symbol, position in self._positions.items():
            value = self._position_values.get(symbol, Decimal("0"))
            if position.is_long:
                long_exposure += value
            else:
                short_exposure += value

        gross = long_exposure + short_exposure
        net = long_exposure - short_exposure

        # Calculate unrealized P&L
        unrealized = sum(
            position.unrealized_pnl(self._position_values.get(symbol, Decimal("0")) / position.quantity)
            for symbol, position in self._positions.items()
            if position.quantity > 0
        )

        return RiskMetrics(
            timestamp=datetime.utcnow(),
            total_exposure=gross,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            net_exposure=net,
            gross_exposure=gross,
            unrealized_pnl=unrealized,
            realized_pnl_today=self._daily_pnl,
            current_drawdown=self._calculate_drawdown(),
            var_95=self._calculate_var(0.95),
            expected_shortfall=self._calculate_expected_shortfall(0.95),
            daily_loss_remaining=self.config.daily_loss_limit + self._daily_pnl if self.config.daily_loss_limit > 0 else Decimal("999999"),
            margin_used=gross / Decimal(str(self.config.max_leverage)),
            margin_available=self._current_equity - gross / Decimal(str(self.config.max_leverage)),
        )


# ==================== Position Sizing ====================

class PositionSizer(ABC):
    """Abstract base class for position sizing algorithms."""

    @abstractmethod
    def calculate_size(
        self,
        equity: Decimal,
        price: Decimal,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[Decimal] = None,
        avg_loss: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate position size.

        Args:
            equity: Current account equity
            price: Current asset price
            volatility: Asset volatility (optional)
            win_rate: Historical win rate (optional)
            avg_win: Average winning trade (optional)
            avg_loss: Average losing trade (optional)

        Returns:
            Quantity to trade
        """
        pass


class FixedSizer(PositionSizer):
    """Fixed position size regardless of account equity."""

    def __init__(self, quantity: Decimal):
        self.quantity = quantity

    def calculate_size(self, **kwargs) -> Decimal:
        return self.quantity


class FixedNotionalSizer(PositionSizer):
    """Fixed notional value per trade."""

    def __init__(self, notional: Decimal):
        self.notional = notional

    def calculate_size(
        self,
        equity: Decimal,
        price: Decimal,
        **kwargs,
    ) -> Decimal:
        if price == 0:
            return Decimal("0")
        return self.notional / price


class PercentEquitySizer(PositionSizer):
    """Size position as percentage of equity."""

    def __init__(self, percent: float = 0.02):
        """
        Args:
            percent: Percentage of equity per trade (e.g., 0.02 = 2%)
        """
        self.percent = percent

    def calculate_size(
        self,
        equity: Decimal,
        price: Decimal,
        **kwargs,
    ) -> Decimal:
        if price == 0:
            return Decimal("0")
        notional = equity * Decimal(str(self.percent))
        return notional / price


class RiskBasedSizer(PositionSizer):
    """
    Size position based on risk per trade.

    Uses stop loss distance to determine size such that
    hitting the stop loses a fixed percentage of equity.
    """

    def __init__(
        self,
        risk_per_trade: float = 0.01,  # 1% risk per trade
        default_stop_pct: float = 0.02,  # 2% default stop distance
    ):
        self.risk_per_trade = risk_per_trade
        self.default_stop_pct = default_stop_pct

    def calculate_size(
        self,
        equity: Decimal,
        price: Decimal,
        stop_price: Optional[Decimal] = None,
        **kwargs,
    ) -> Decimal:
        if price == 0:
            return Decimal("0")

        # Risk amount
        risk_amount = equity * Decimal(str(self.risk_per_trade))

        # Stop distance
        if stop_price:
            stop_distance = abs(price - stop_price)
        else:
            stop_distance = price * Decimal(str(self.default_stop_pct))

        if stop_distance == 0:
            return Decimal("0")

        # Size = Risk / Stop Distance
        return risk_amount / stop_distance


class VolatilitySizer(PositionSizer):
    """
    Size position inversely proportional to volatility.

    Targets a consistent risk per trade across different volatility regimes.
    """

    def __init__(
        self,
        target_volatility: float = 0.01,  # 1% daily target vol contribution
        base_size_pct: float = 0.05,  # 5% base size
        min_volatility: float = 0.005,  # Minimum vol to avoid huge positions
    ):
        self.target_volatility = target_volatility
        self.base_size_pct = base_size_pct
        self.min_volatility = min_volatility

    def calculate_size(
        self,
        equity: Decimal,
        price: Decimal,
        volatility: Optional[float] = None,
        **kwargs,
    ) -> Decimal:
        if price == 0:
            return Decimal("0")

        # Use provided volatility or assume base volatility
        vol = max(volatility or 0.02, self.min_volatility)

        # Scale size inversely with volatility
        vol_scalar = self.target_volatility / vol

        # Base notional
        base_notional = equity * Decimal(str(self.base_size_pct))

        # Adjusted notional
        adjusted_notional = base_notional * Decimal(str(vol_scalar))

        return adjusted_notional / price


class KellySizer(PositionSizer):
    """
    Kelly Criterion position sizing.

    Optimizes for maximum growth rate given win rate and average win/loss.

    Kelly fraction = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win

    In practice, use fractional Kelly (e.g., half-Kelly) to reduce variance.
    """

    def __init__(
        self,
        fraction: float = 0.5,  # Half-Kelly is common
        max_size_pct: float = 0.25,  # Cap at 25% of equity
        min_size_pct: float = 0.01,  # Minimum 1% of equity
    ):
        self.fraction = fraction
        self.max_size_pct = max_size_pct
        self.min_size_pct = min_size_pct

    def calculate_size(
        self,
        equity: Decimal,
        price: Decimal,
        win_rate: Optional[float] = None,
        avg_win: Optional[Decimal] = None,
        avg_loss: Optional[Decimal] = None,
        **kwargs,
    ) -> Decimal:
        if price == 0:
            return Decimal("0")

        # Default values if not provided
        win_rate = win_rate or 0.5
        avg_win = avg_win or Decimal("100")
        avg_loss = avg_loss or Decimal("100")

        if avg_win == 0:
            return equity * Decimal(str(self.min_size_pct)) / price

        # Kelly fraction
        edge = win_rate * float(avg_win) - (1 - win_rate) * float(avg_loss)
        kelly = edge / float(avg_win)

        # Apply fractional Kelly
        kelly_fraction = kelly * self.fraction

        # Clamp to limits
        kelly_fraction = max(self.min_size_pct, min(self.max_size_pct, kelly_fraction))

        # Convert to quantity
        notional = equity * Decimal(str(kelly_fraction))
        return notional / price


class ATRSizer(PositionSizer):
    """
    Size position based on Average True Range (ATR).

    Uses ATR to normalize position sizes across different instruments.
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,  # Stop at 2x ATR
        risk_per_trade: float = 0.01,  # 1% risk
    ):
        self.atr_multiplier = atr_multiplier
        self.risk_per_trade = risk_per_trade

    def calculate_size(
        self,
        equity: Decimal,
        price: Decimal,
        atr: Optional[Decimal] = None,
        **kwargs,
    ) -> Decimal:
        if price == 0:
            return Decimal("0")

        # Default ATR if not provided (assume 2% of price)
        atr = atr or (price * Decimal("0.02"))

        # Stop distance
        stop_distance = atr * Decimal(str(self.atr_multiplier))

        if stop_distance == 0:
            return Decimal("0")

        # Risk amount
        risk_amount = equity * Decimal(str(self.risk_per_trade))

        # Size = Risk / Stop Distance
        return risk_amount / stop_distance
