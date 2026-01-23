"""
Strategy base class for implementing trading strategies.

Provides a framework for:
- Signal generation
- Order management
- Position tracking
- Performance monitoring
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Optional, Callable, Any

from trading.core.models import Order, Position, Tick, OHLCV, Trade
from trading.core.enums import OrderSide, OrderType, OrderStatus
from trading.exchanges.base import Exchange


class StrategyState(Enum):
    """Strategy lifecycle state."""
    INITIALIZED = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    ERROR = auto()


class Signal(Enum):
    """Trading signal."""
    LONG = auto()       # Open/increase long position
    SHORT = auto()      # Open/increase short position
    EXIT_LONG = auto()  # Close long position
    EXIT_SHORT = auto() # Close short position
    FLAT = auto()       # Close all positions
    HOLD = auto()       # No action


@dataclass
class StrategyConfig:
    """Base configuration for strategies."""
    # Identification
    name: str = "BaseStrategy"
    version: str = "1.0.0"

    # Trading parameters
    symbols: list[str] = field(default_factory=list)
    max_positions: int = 5
    max_position_size: Decimal = Decimal("1000000")

    # Risk parameters
    max_drawdown_pct: float = 0.20  # 20% max drawdown
    daily_loss_limit: Decimal = Decimal("0")  # 0 = disabled
    position_timeout_hours: float = 0  # 0 = no timeout

    # Execution
    order_type: OrderType = OrderType.MARKET
    slippage_tolerance_pct: float = 0.01

    # Logging
    log_signals: bool = True
    log_orders: bool = True
    log_fills: bool = True


@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""
    # Returns
    total_return: Decimal = Decimal("0")
    total_return_pct: float = 0.0
    daily_returns: list[float] = field(default_factory=list)

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    profit_factor: float = 0.0
    avg_trade_duration_hours: float = 0.0

    # Exposure
    time_in_market_pct: float = 0.0
    max_concurrent_positions: int = 0

    @property
    def expectancy(self) -> float:
        """Expected value per trade."""
        if self.total_trades == 0:
            return 0.0
        return float(self.total_return) / self.total_trades


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Subclasses must implement:
    - on_tick(tick): Process incoming tick data
    - on_bar(bar): Process incoming bar data
    - generate_signal(symbol): Generate trading signal

    Optional overrides:
    - on_order_filled(order, fill): Handle order fills
    - on_position_opened(position): Handle new positions
    - on_position_closed(position, pnl): Handle closed positions

    Usage:
        class MyStrategy(Strategy):
            def __init__(self, exchange, config):
                super().__init__(exchange, config)
                self.fast_ma = []
                self.slow_ma = []

            def on_bar(self, bar: OHLCV) -> None:
                # Update indicators
                self.fast_ma.append(float(bar.close))
                self.slow_ma.append(float(bar.close))

                # Generate and execute signal
                signal = self.generate_signal(bar.symbol)
                self.execute_signal(bar.symbol, signal)

            def generate_signal(self, symbol: str) -> Signal:
                if fast_avg > slow_avg:
                    return Signal.LONG
                elif fast_avg < slow_avg:
                    return Signal.SHORT
                return Signal.HOLD

        strategy = MyStrategy(exchange, config)
        await strategy.start()
    """

    def __init__(self, exchange: Exchange, config: Optional[StrategyConfig] = None):
        self.exchange = exchange
        self.config = config or StrategyConfig()

        self._state = StrategyState.INITIALIZED
        self._start_time: Optional[datetime] = None
        self._equity_curve: list[tuple[datetime, Decimal]] = []
        self._trades: list[Trade] = []
        self._pending_orders: dict[str, Order] = {}

        # Performance tracking
        self.metrics = StrategyMetrics()
        self._peak_equity = Decimal("0")
        self._daily_pnl = Decimal("0")
        self._last_day: Optional[datetime] = None

        # Callbacks
        self._signal_callbacks: list[Callable[[str, Signal], None]] = []

    @property
    def state(self) -> StrategyState:
        """Current strategy state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if strategy is actively running."""
        return self._state == StrategyState.RUNNING

    # ==================== Lifecycle ====================

    async def start(self) -> None:
        """Start the strategy."""
        if self._state == StrategyState.RUNNING:
            return

        self._state = StrategyState.RUNNING
        self._start_time = datetime.utcnow()

        # Subscribe to market data
        if self.config.symbols:
            await self.exchange.subscribe_ticks(
                self.config.symbols,
                self._handle_tick,
            )

        # Initialize
        await self.on_start()

    async def stop(self) -> None:
        """Stop the strategy."""
        self._state = StrategyState.STOPPED

        # Unsubscribe from market data
        await self.exchange.unsubscribe_all()

        # Cancel pending orders
        for order_id in list(self._pending_orders.keys()):
            await self.exchange.cancel_order(order_id)

        await self.on_stop()

    async def pause(self) -> None:
        """Pause the strategy (stops new signals but keeps positions)."""
        self._state = StrategyState.PAUSED
        await self.on_pause()

    async def resume(self) -> None:
        """Resume a paused strategy."""
        if self._state == StrategyState.PAUSED:
            self._state = StrategyState.RUNNING
            await self.on_resume()

    # ==================== Event Handlers ====================

    def _handle_tick(self, tick: Tick) -> None:
        """Internal tick handler."""
        if self._state != StrategyState.RUNNING:
            return

        try:
            self.on_tick(tick)
        except Exception as e:
            self._state = StrategyState.ERROR
            self.on_error(e)

    def _handle_bar(self, bar: OHLCV) -> None:
        """Internal bar handler."""
        if self._state != StrategyState.RUNNING:
            return

        try:
            self.on_bar(bar)
        except Exception as e:
            self._state = StrategyState.ERROR
            self.on_error(e)

    # ==================== Abstract Methods ====================

    @abstractmethod
    def on_tick(self, tick: Tick) -> None:
        """
        Process incoming tick data.

        Called for each tick when subscribed to tick data.

        Args:
            tick: Market tick data
        """
        pass

    @abstractmethod
    def on_bar(self, bar: OHLCV) -> None:
        """
        Process incoming bar data.

        Called for each bar when subscribed to bar data.

        Args:
            bar: OHLCV bar data
        """
        pass

    @abstractmethod
    def generate_signal(self, symbol: str) -> Signal:
        """
        Generate trading signal for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Trading signal
        """
        pass

    # ==================== Optional Hooks ====================

    async def on_start(self) -> None:
        """Called when strategy starts. Override for initialization."""
        pass

    async def on_stop(self) -> None:
        """Called when strategy stops. Override for cleanup."""
        pass

    async def on_pause(self) -> None:
        """Called when strategy is paused."""
        pass

    async def on_resume(self) -> None:
        """Called when strategy resumes."""
        pass

    def on_order_submitted(self, order: Order) -> None:
        """Called when an order is submitted."""
        if self.config.log_orders:
            print(f"[{self.config.name}] Order submitted: {order.symbol} {order.side.name} {order.quantity}")

    def on_order_filled(self, order: Order) -> None:
        """Called when an order is filled."""
        if self.config.log_fills:
            print(f"[{self.config.name}] Order filled: {order.symbol} @ {order.average_fill_price}")

    def on_order_cancelled(self, order: Order) -> None:
        """Called when an order is cancelled."""
        pass

    def on_order_rejected(self, order: Order, reason: str) -> None:
        """Called when an order is rejected."""
        print(f"[{self.config.name}] Order rejected: {reason}")

    def on_position_opened(self, position: Position) -> None:
        """Called when a new position is opened."""
        pass

    def on_position_closed(self, position: Position, realized_pnl: Decimal) -> None:
        """Called when a position is closed."""
        self._update_trade_stats(position, realized_pnl)

    def on_error(self, error: Exception) -> None:
        """Called when an error occurs."""
        print(f"[{self.config.name}] Error: {error}")

    # ==================== Signal Execution ====================

    async def execute_signal(
        self,
        symbol: str,
        signal: Signal,
        quantity: Optional[Decimal] = None,
    ) -> Optional[Order]:
        """
        Execute a trading signal.

        Args:
            symbol: Trading symbol
            signal: Signal to execute
            quantity: Order quantity (uses position sizer if None)

        Returns:
            Submitted order or None
        """
        if signal == Signal.HOLD:
            return None

        # Get current position
        position = await self.exchange.get_position(symbol)

        # Log signal
        if self.config.log_signals:
            print(f"[{self.config.name}] Signal: {symbol} {signal.name}")

        # Notify callbacks
        for callback in self._signal_callbacks:
            callback(symbol, signal)

        # Determine order parameters
        order_side = None
        order_qty = quantity or Decimal("1")

        if signal == Signal.LONG:
            if not position or not position.is_long:
                order_side = OrderSide.BUY
        elif signal == Signal.SHORT:
            if not position or not position.is_short:
                order_side = OrderSide.SELL
        elif signal == Signal.EXIT_LONG:
            if position and position.is_long:
                order_side = OrderSide.SELL
                order_qty = position.quantity
        elif signal == Signal.EXIT_SHORT:
            if position and position.is_short:
                order_side = OrderSide.BUY
                order_qty = position.quantity
        elif signal == Signal.FLAT:
            if position and position.quantity > 0:
                order_side = OrderSide.SELL if position.is_long else OrderSide.BUY
                order_qty = position.quantity

        if not order_side:
            return None

        # Create and submit order
        order = Order(
            symbol=symbol,
            side=order_side,
            quantity=order_qty,
            order_type=self.config.order_type,
            strategy_id=self.config.name,
        )

        try:
            result = await self.exchange.submit_order(order)
            self._pending_orders[order.order_id] = order
            self.on_order_submitted(order)

            if result.status == OrderStatus.FILLED:
                self.on_order_filled(result)
                del self._pending_orders[order.order_id]

            return result

        except Exception as e:
            self.on_order_rejected(order, str(e))
            return None

    # ==================== Position Management ====================

    async def close_all_positions(self) -> list[Order]:
        """Close all open positions."""
        orders = []
        positions = await self.exchange.get_positions()

        for position in positions:
            if position.symbol in self.config.symbols:
                order = await self.execute_signal(position.symbol, Signal.FLAT)
                if order:
                    orders.append(order)

        return orders

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""
        return await self.exchange.get_position(symbol)

    async def get_positions(self) -> list[Position]:
        """Get all positions managed by this strategy."""
        all_positions = await self.exchange.get_positions()
        return [p for p in all_positions if p.symbol in self.config.symbols]

    # ==================== Performance Tracking ====================

    def _update_trade_stats(self, position: Position, pnl: Decimal) -> None:
        """Update trade statistics after a position closes."""
        self.metrics.total_trades += 1
        self.metrics.total_return += pnl

        if pnl > 0:
            self.metrics.winning_trades += 1
            # Update average win
            total_wins = self.metrics.avg_win * (self.metrics.winning_trades - 1) + pnl
            self.metrics.avg_win = total_wins / self.metrics.winning_trades
        else:
            self.metrics.losing_trades += 1
            # Update average loss
            total_losses = self.metrics.avg_loss * (self.metrics.losing_trades - 1) + abs(pnl)
            self.metrics.avg_loss = total_losses / self.metrics.losing_trades

        # Win rate
        self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades

        # Profit factor
        if self.metrics.avg_loss > 0 and self.metrics.losing_trades > 0:
            total_profit = self.metrics.avg_win * self.metrics.winning_trades
            total_loss = self.metrics.avg_loss * self.metrics.losing_trades
            self.metrics.profit_factor = float(total_profit / total_loss)

    def update_equity(self, equity: Decimal, timestamp: datetime) -> None:
        """Update equity curve and drawdown."""
        self._equity_curve.append((timestamp, equity))

        # Update peak and drawdown
        if equity > self._peak_equity:
            self._peak_equity = equity
        else:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            self.metrics.current_drawdown = float(drawdown)
            self.metrics.max_drawdown = max(self.metrics.max_drawdown, float(drawdown))

        # Check drawdown limit
        if self.metrics.current_drawdown >= self.config.max_drawdown_pct:
            self._state = StrategyState.PAUSED
            print(f"[{self.config.name}] Max drawdown reached, strategy paused")

        # Daily P&L tracking
        if self._last_day and timestamp.date() > self._last_day.date():
            # New day - check daily loss limit
            if self.config.daily_loss_limit > 0 and self._daily_pnl < -self.config.daily_loss_limit:
                self._state = StrategyState.PAUSED
                print(f"[{self.config.name}] Daily loss limit reached")
            self._daily_pnl = Decimal("0")

        self._last_day = timestamp

    def get_metrics(self) -> StrategyMetrics:
        """Get current performance metrics."""
        return self.metrics

    def add_signal_callback(self, callback: Callable[[str, Signal], None]) -> None:
        """Add callback for signal notifications."""
        self._signal_callbacks.append(callback)


class SimpleStrategy(Strategy):
    """
    Simple strategy implementation for quick prototyping.

    Allows defining logic via callbacks instead of subclassing.
    """

    def __init__(
        self,
        exchange: Exchange,
        config: Optional[StrategyConfig] = None,
        on_tick_callback: Optional[Callable[["SimpleStrategy", Tick], None]] = None,
        on_bar_callback: Optional[Callable[["SimpleStrategy", OHLCV], None]] = None,
        signal_callback: Optional[Callable[["SimpleStrategy", str], Signal]] = None,
    ):
        super().__init__(exchange, config)
        self._on_tick_callback = on_tick_callback
        self._on_bar_callback = on_bar_callback
        self._signal_callback = signal_callback

    def on_tick(self, tick: Tick) -> None:
        if self._on_tick_callback:
            self._on_tick_callback(self, tick)

    def on_bar(self, bar: OHLCV) -> None:
        if self._on_bar_callback:
            self._on_bar_callback(self, bar)

    def generate_signal(self, symbol: str) -> Signal:
        if self._signal_callback:
            return self._signal_callback(self, symbol)
        return Signal.HOLD
