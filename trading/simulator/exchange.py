"""
Simulated exchange for paper trading.

Provides a fully functional exchange implementation using market models
for price simulation, supporting:
- Realistic order execution with slippage
- Position and balance tracking
- Market data generation
- Configurable market dynamics
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Callable, Any
from collections import defaultdict
from uuid import uuid4

from trading.core.models import (
    Order,
    Position,
    Tick,
    OHLCV,
    OrderBook,
    OrderBookLevel,
    Balance,
    Instrument,
    Fill,
)
from trading.core.enums import (
    OrderType,
    OrderSide,
    OrderStatus,
    PositionSide,
    TimeInForce,
    ExchangeType,
    AssetClass,
    DataResolution,
    FillType,
)
from trading.core.events import (
    EventType,
    OrderEvent,
    TickEvent,
    BarEvent,
    PositionEvent,
)
from trading.exchanges.base import (
    Exchange,
    ExchangeConfig,
    ExchangeError,
    OrderRejectedError,
    InsufficientFundsError,
)
from trading.simulator.market_model import (
    MarketModel,
    MarketModelConfig,
    GBMModel,
    JumpDiffusionModel,
    OUModel,
    SimulatedOrderBook,
)
from trading.simulator.calibration import CalibrationResult


@dataclass
class SimulatorConfig(ExchangeConfig):
    """Configuration for the simulated exchange."""
    # Initial account balance
    initial_balance: dict[str, float] = field(default_factory=lambda: {"USD": 100000.0})

    # Default market model settings
    default_model_config: MarketModelConfig = field(default_factory=MarketModelConfig)

    # Price simulation
    tick_interval_ms: int = 100  # Tick generation interval
    enable_price_simulation: bool = True

    # Order execution settings
    fill_latency_ms: float = 50.0  # Order fill latency
    partial_fill_probability: float = 0.1  # Chance of partial fills
    rejection_probability: float = 0.01  # Random rejection rate

    # Commission settings
    commission_rate: float = 0.001  # 0.1% per trade
    min_commission: float = 1.0  # Minimum commission

    # Margin settings
    default_leverage: float = 1.0
    maintenance_margin: float = 0.25  # 25% maintenance margin

    # Market hours (None = 24/7)
    market_open_hour: Optional[int] = None
    market_close_hour: Optional[int] = None

    # Random seed for reproducibility
    random_seed: Optional[int] = None


class SimulatedExchange(Exchange):
    """
    Fully simulated exchange for paper trading and backtesting.

    Features:
    - Configurable market dynamics (GBM, jump diffusion, mean reversion)
    - Realistic order execution with slippage and latency
    - Full position and balance tracking
    - Commission and margin handling
    - Calibration from real market data

    Usage:
        # Basic setup
        config = SimulatorConfig(
            initial_balance={"USD": 100000, "BTC": 1.0},
        )
        exchange = SimulatedExchange(config)

        # Add instrument with custom market model
        exchange.add_instrument(
            symbol="BTCUSDT",
            initial_price=50000.0,
            model_config=MarketModelConfig(
                volatility=0.60,  # 60% annual volatility
                spread_pct=0.0005,  # 5 bps spread
            ),
        )

        # Or calibrate from real data
        calibrator = MarketCalibrator("BTCUSDT")
        # ... add observations ...
        result = calibrator.calibrate()
        exchange.add_instrument_from_calibration("BTCUSDT", 50000.0, result)

        await exchange.connect()

        # Place orders
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )
        filled_order = await exchange.submit_order(order)
    """

    def __init__(self, config: SimulatorConfig):
        super().__init__(config)
        self.config: SimulatorConfig = config

        # Market models for each instrument
        self._models: dict[str, MarketModel] = {}
        self._order_books: dict[str, SimulatedOrderBook] = {}

        # Account state
        self._balances: dict[str, Balance] = {}
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._pending_orders: dict[str, Order] = {}  # Orders waiting to be filled
        self._trades: list[dict] = []

        # Simulation state
        self._simulation_time: datetime = datetime.utcnow()
        self._tick_task: Optional[asyncio.Task] = None
        self._running = False

        # Callbacks
        self._tick_callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._bar_callbacks: dict[str, list[Callable]] = defaultdict(list)

        # Initialize balances
        for currency, amount in config.initial_balance.items():
            self._balances[currency] = Balance(
                currency=currency,
                total=Decimal(str(amount)),
                available=Decimal(str(amount)),
                locked=Decimal("0"),
            )

    @property
    def exchange_type(self) -> ExchangeType:
        return ExchangeType.SIMULATED

    @property
    def name(self) -> str:
        return "Simulated Exchange"

    @property
    def simulation_time(self) -> datetime:
        """Current simulation time."""
        return self._simulation_time

    # ==================== Instrument Setup ====================

    def add_instrument(
        self,
        symbol: str,
        initial_price: float,
        model_config: Optional[MarketModelConfig] = None,
        model_type: str = "gbm",
        asset_class: AssetClass = AssetClass.CRYPTO,
        base_currency: Optional[str] = None,
        quote_currency: str = "USD",
        lot_size: float = 0.001,
        tick_size: float = 0.01,
    ) -> Instrument:
        """
        Add a tradeable instrument with market model.

        Args:
            symbol: Trading symbol
            initial_price: Starting price
            model_config: Market model configuration
            model_type: "gbm", "jump_diffusion", or "ou"
            asset_class: Asset classification
            base_currency: Base currency (default: derived from symbol)
            quote_currency: Quote currency
            lot_size: Minimum quantity increment
            tick_size: Minimum price increment

        Returns:
            Created Instrument
        """
        config = model_config or self.config.default_model_config

        # Create market model
        model_classes = {
            "gbm": GBMModel,
            "jump_diffusion": JumpDiffusionModel,
            "ou": OUModel,
        }
        model_class = model_classes.get(model_type, GBMModel)
        model = model_class(config)
        model.set_price(initial_price)

        if self.config.random_seed:
            model.seed(self.config.random_seed + hash(symbol))

        self._models[symbol] = model

        # Create order book
        self._order_books[symbol] = SimulatedOrderBook(symbol, model)

        # Create instrument
        instrument = Instrument(
            symbol=symbol,
            base_currency=base_currency or symbol.replace(quote_currency, "").replace("_", ""),
            quote_currency=quote_currency,
            asset_class=asset_class,
            exchange="simulated",
            lot_size=Decimal(str(lot_size)),
            tick_size=Decimal(str(tick_size)),
        )
        self._instruments[symbol] = instrument

        return instrument

    def add_instrument_from_calibration(
        self,
        symbol: str,
        initial_price: float,
        calibration: CalibrationResult,
        asset_class: AssetClass = AssetClass.CRYPTO,
    ) -> Instrument:
        """
        Add instrument using calibration results.

        Args:
            symbol: Trading symbol
            initial_price: Starting price
            calibration: Calibration result from MarketCalibrator
            asset_class: Asset classification

        Returns:
            Created Instrument
        """
        config = calibration.to_model_config()
        return self.add_instrument(
            symbol=symbol,
            initial_price=initial_price,
            model_config=config,
            model_type=calibration.recommended_model,
            asset_class=asset_class,
        )

    # ==================== Connection ====================

    async def connect(self) -> bool:
        """Start the simulation."""
        if self._connected:
            return True

        self._connected = True
        self._running = True
        self._simulation_time = datetime.utcnow()

        # Start tick generation
        if self.config.enable_price_simulation:
            self._tick_task = asyncio.create_task(self._tick_loop())

        self.events.emit(OrderEvent(
            event_type=EventType.CONNECTED,
            source=self.name,
        ))

        return True

    async def disconnect(self) -> None:
        """Stop the simulation."""
        self._running = False

        if self._tick_task:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
            self._tick_task = None

        self._connected = False

        self.events.emit(OrderEvent(
            event_type=EventType.DISCONNECTED,
            source=self.name,
        ))

    async def _tick_loop(self) -> None:
        """Generate price ticks at regular intervals."""
        interval = self.config.tick_interval_ms / 1000

        while self._running:
            try:
                # Advance all market models
                for symbol, model in self._models.items():
                    model.step(interval)

                    # Generate tick
                    tick = Tick(
                        symbol=symbol,
                        timestamp=self._simulation_time,
                        bid=Decimal(str(model.bid)),
                        ask=Decimal(str(model.ask)),
                        bid_size=Decimal("1000"),
                        ask_size=Decimal("1000"),
                        last_price=Decimal(str(model.price)),
                    )

                    # Notify callbacks
                    for callback in self._tick_callbacks.get(symbol, []):
                        try:
                            callback(tick)
                        except Exception:
                            pass

                    self.events.emit(TickEvent(tick=tick, source=self.name))

                    # Check pending orders for fills
                    await self._process_pending_orders(symbol, tick)

                # Advance simulation time
                self._simulation_time += timedelta(milliseconds=self.config.tick_interval_ms)

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in tick loop: {e}")

    async def _process_pending_orders(self, symbol: str, tick: Tick) -> None:
        """Check and fill pending limit/stop orders."""
        orders_to_remove = []

        for order_id, order in list(self._pending_orders.items()):
            if order.symbol != symbol:
                continue

            should_fill = False
            fill_price = None

            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and tick.ask <= order.limit_price:
                    should_fill = True
                    fill_price = float(order.limit_price)
                elif order.side == OrderSide.SELL and tick.bid >= order.limit_price:
                    should_fill = True
                    fill_price = float(order.limit_price)

            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and tick.ask >= order.stop_price:
                    should_fill = True
                    fill_price = float(tick.ask)
                elif order.side == OrderSide.SELL and tick.bid <= order.stop_price:
                    should_fill = True
                    fill_price = float(tick.bid)

            if should_fill and fill_price:
                await self._execute_fill(order, fill_price)
                orders_to_remove.append(order_id)

        for order_id in orders_to_remove:
            del self._pending_orders[order_id]

    # ==================== Account ====================

    async def get_balances(self) -> list[Balance]:
        """Get all account balances."""
        return list(self._balances.values())

    async def get_balance(self, currency: str) -> Optional[Balance]:
        """Get balance for specific currency."""
        return self._balances.get(currency)

    def _update_balance(self, currency: str, delta: Decimal, lock: bool = False) -> None:
        """Update balance for a currency."""
        if currency not in self._balances:
            self._balances[currency] = Balance(
                currency=currency,
                total=Decimal("0"),
                available=Decimal("0"),
            )

        balance = self._balances[currency]

        if lock:
            balance.locked += delta
            balance.available -= delta
        else:
            balance.total += delta
            balance.available += delta

    # ==================== Instruments ====================

    async def get_instruments(self) -> list[Instrument]:
        """Get all configured instruments."""
        return list(self._instruments.values())

    async def get_instrument(self, symbol: str) -> Optional[Instrument]:
        """Get instrument details."""
        return self._instruments.get(symbol)

    # ==================== Orders ====================

    async def submit_order(self, order: Order) -> Order:
        """Submit an order for execution."""
        # Validate instrument
        if order.symbol not in self._instruments:
            raise OrderRejectedError(order, f"Unknown instrument: {order.symbol}")

        # Random rejection simulation
        import random
        if random.random() < self.config.rejection_probability:
            order.status = OrderStatus.REJECTED
            self.events.emit(OrderEvent(
                event_type=EventType.ORDER_REJECTED,
                order=order,
                reason="Random rejection (simulated)",
                source=self.name,
            ))
            return order

        model = self._models[order.symbol]
        instrument = self._instruments[order.symbol]

        # Check available balance
        quote_currency = instrument.quote_currency
        base_currency = instrument.base_currency

        if order.side == OrderSide.BUY:
            required = order.quantity * Decimal(str(model.ask))
            balance = self._balances.get(quote_currency)
            if not balance or balance.available < required:
                raise InsufficientFundsError(required, balance.available if balance else Decimal("0"), quote_currency)

            # Lock funds
            self._update_balance(quote_currency, required, lock=True)
        else:
            # Check we have the asset to sell
            position = self._positions.get(order.symbol)
            if not order.reduce_only:
                balance = self._balances.get(base_currency)
                if not balance or balance.available < order.quantity:
                    if not position or position.quantity < order.quantity:
                        raise InsufficientFundsError(order.quantity, balance.available if balance else Decimal("0"), base_currency)

        # Set order metadata
        order.exchange_order_id = str(uuid4())[:8]
        order.submitted_at = self._simulation_time
        order.status = OrderStatus.ACCEPTED

        self._orders[order.order_id] = order
        self._orders[order.exchange_order_id] = order

        self.events.emit(OrderEvent(
            event_type=EventType.ORDER_SUBMITTED,
            order=order,
            source=self.name,
        ))

        # Execute immediately for market orders
        if order.order_type == OrderType.MARKET:
            # Simulate latency
            await asyncio.sleep(self.config.fill_latency_ms / 1000)

            fill_price = model.get_fill_price(
                "BUY" if order.side == OrderSide.BUY else "SELL",
                float(order.quantity),
                "MARKET",
            )
            await self._execute_fill(order, fill_price)
        else:
            # Add to pending orders
            self._pending_orders[order.order_id] = order

        return order

    async def _execute_fill(self, order: Order, fill_price: float) -> None:
        """Execute order fill and update positions/balances."""
        instrument = self._instruments[order.symbol]
        quote_currency = instrument.quote_currency
        base_currency = instrument.base_currency

        # Calculate commission
        notional = float(order.quantity) * fill_price
        commission = max(
            notional * self.config.commission_rate,
            self.config.min_commission
        )

        # Create fill
        fill = Fill(
            fill_id=str(uuid4())[:8],
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=Decimal(str(fill_price)),
            commission=Decimal(str(commission)),
            timestamp=self._simulation_time,
        )

        order.add_fill(fill)
        order.status = OrderStatus.FILLED
        order.filled_at = self._simulation_time

        # Update balances
        if order.side == OrderSide.BUY:
            # Unlock and deduct quote currency
            cost = order.quantity * Decimal(str(fill_price)) + Decimal(str(commission))
            self._update_balance(quote_currency, -cost, lock=False)
            if quote_currency in self._balances:
                # Release any excess locked funds
                locked = self._balances[quote_currency].locked
                if locked > 0:
                    self._balances[quote_currency].locked = Decimal("0")
                    self._balances[quote_currency].available += locked - cost

            # Add base currency
            self._update_balance(base_currency, order.quantity)
        else:
            # Deduct base currency
            self._update_balance(base_currency, -order.quantity)

            # Add quote currency (minus commission)
            proceeds = order.quantity * Decimal(str(fill_price)) - Decimal(str(commission))
            self._update_balance(quote_currency, proceeds)

        # Update position
        await self._update_position(order, fill)

        # Emit events
        self.events.emit(OrderEvent(
            event_type=EventType.ORDER_FILLED,
            order=order,
            fill=fill,
            source=self.name,
        ))

    async def _update_position(self, order: Order, fill: Fill) -> None:
        """Update position after a fill."""
        symbol = order.symbol
        position = self._positions.get(symbol)

        if not position:
            # Create new position
            position = Position(
                symbol=symbol,
                side=PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT,
                quantity=fill.quantity,
                average_entry_price=fill.price,
                opened_at=self._simulation_time,
            )
            self._positions[symbol] = position

            self.events.emit(PositionEvent(
                event_type=EventType.POSITION_OPENED,
                position=position,
                source=self.name,
            ))
        else:
            # Update existing position
            prev_qty = position.quantity
            realized = position.update_from_fill(fill)

            if position.quantity == 0:
                # Position closed
                del self._positions[symbol]
                self.events.emit(PositionEvent(
                    event_type=EventType.POSITION_CLOSED,
                    position=position,
                    previous_quantity=prev_qty,
                    realized_pnl=realized,
                    source=self.name,
                ))
            else:
                self.events.emit(PositionEvent(
                    event_type=EventType.POSITION_UPDATED,
                    position=position,
                    previous_quantity=prev_qty,
                    realized_pnl=realized,
                    source=self.name,
                ))

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        order = self._orders.get(order_id) or self._pending_orders.get(order_id)

        if not order or not order.is_active:
            return False

        # Release locked funds
        if order.side == OrderSide.BUY and order.symbol in self._instruments:
            instrument = self._instruments[order.symbol]
            model = self._models[order.symbol]
            locked = order.remaining_quantity * Decimal(str(model.ask))
            self._update_balance(instrument.quote_currency, -locked, lock=True)

        order.status = OrderStatus.CANCELLED
        order.cancelled_at = self._simulation_time

        if order_id in self._pending_orders:
            del self._pending_orders[order_id]

        self.events.emit(OrderEvent(
            event_type=EventType.ORDER_CANCELLED,
            order=order,
            source=self.name,
        ))

        return True

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all pending orders."""
        cancelled = 0

        for order_id in list(self._pending_orders.keys()):
            order = self._pending_orders[order_id]
            if symbol is None or order.symbol == symbol:
                if await self.cancel_order(order_id):
                    cancelled += 1

        return cancelled

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Get all pending orders."""
        orders = []
        for order in self._pending_orders.values():
            if symbol is None or order.symbol == symbol:
                orders.append(order)
        return orders

    # ==================== Positions ====================

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self._positions.values())

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self._positions.get(symbol)

    # ==================== Market Data ====================

    async def get_ticker(self, symbol: str) -> Optional[Tick]:
        """Get current ticker for symbol."""
        model = self._models.get(symbol)
        if not model:
            return None

        return Tick(
            symbol=symbol,
            timestamp=self._simulation_time,
            bid=Decimal(str(model.bid)),
            ask=Decimal(str(model.ask)),
            bid_size=Decimal("1000"),
            ask_size=Decimal("1000"),
            last_price=Decimal(str(model.price)),
        )

    async def get_order_book(self, symbol: str, depth: int = 10) -> Optional[OrderBook]:
        """Get order book for symbol."""
        order_book = self._order_books.get(symbol)
        if not order_book:
            return None

        order_book.levels = depth
        bids, asks = order_book.generate()

        return OrderBook(
            symbol=symbol,
            timestamp=self._simulation_time,
            bids=[
                OrderBookLevel(Decimal(str(p)), Decimal(str(q)))
                for p, q in bids
            ],
            asks=[
                OrderBookLevel(Decimal(str(p)), Decimal(str(q)))
                for p, q in asks
            ],
        )

    async def get_historical_bars(
        self,
        symbol: str,
        resolution: DataResolution,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[OHLCV]:
        """
        Generate historical bars using the market model.

        Note: This generates simulated history, not real data.
        """
        model = self._models.get(symbol)
        if not model:
            return []

        end_time = end or self._simulation_time
        interval_seconds = resolution.to_seconds() or 3600

        bars = []
        current = start
        original_price = model.price

        while current < end_time and len(bars) < limit:
            # Simulate price evolution for this bar
            open_price = model.price

            # Generate multiple ticks for OHLC
            high = open_price
            low = open_price
            volume = model.generate_volume(interval_seconds)

            for _ in range(10):  # 10 steps per bar
                model.step(interval_seconds / 10)
                high = max(high, model.price)
                low = min(low, model.price)

            close_price = model.price

            bars.append(OHLCV(
                symbol=symbol,
                timestamp=current,
                open=Decimal(str(open_price)),
                high=Decimal(str(high)),
                low=Decimal(str(low)),
                close=Decimal(str(close_price)),
                volume=Decimal(str(volume)),
            ))

            current += timedelta(seconds=interval_seconds)

        # Restore original price
        model.set_price(original_price)

        return bars

    # ==================== Streaming ====================

    async def subscribe_ticks(
        self,
        symbols: list[str],
        callback: Callable[[Tick], None]
    ) -> None:
        """Subscribe to tick updates."""
        for symbol in symbols:
            if symbol in self._models:
                self._tick_callbacks[symbol].append(callback)

    async def subscribe_bars(
        self,
        symbols: list[str],
        resolution: DataResolution,
        callback: Callable[[OHLCV], None]
    ) -> None:
        """Subscribe to bar updates."""
        for symbol in symbols:
            key = f"{symbol}_{resolution.name}"
            self._bar_callbacks[key].append(callback)

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all data streams."""
        self._tick_callbacks.clear()
        self._bar_callbacks.clear()

    # ==================== Simulation Control ====================

    def advance_time(self, seconds: float) -> None:
        """
        Manually advance simulation time.

        Useful for backtesting without real-time delays.
        """
        steps = int(seconds / (self.config.tick_interval_ms / 1000))

        for _ in range(steps):
            for symbol, model in self._models.items():
                model.step(self.config.tick_interval_ms / 1000)

            self._simulation_time += timedelta(milliseconds=self.config.tick_interval_ms)

    def reset(self) -> None:
        """Reset simulation to initial state."""
        # Reset balances
        self._balances.clear()
        for currency, amount in self.config.initial_balance.items():
            self._balances[currency] = Balance(
                currency=currency,
                total=Decimal(str(amount)),
                available=Decimal(str(amount)),
                locked=Decimal("0"),
            )

        # Clear positions and orders
        self._positions.clear()
        self._orders.clear()
        self._pending_orders.clear()
        self._trades.clear()

        # Reset time
        self._simulation_time = datetime.utcnow()

    def get_equity(self) -> Decimal:
        """
        Calculate total account equity.

        Sum of all balances plus unrealized P&L on positions.
        """
        equity = Decimal("0")

        # Add cash balances (convert to USD equivalent)
        for balance in self._balances.values():
            if balance.currency == "USD":
                equity += balance.total
            else:
                # Try to find price for conversion
                for symbol, model in self._models.items():
                    if balance.currency in symbol and "USD" in symbol:
                        equity += balance.total * Decimal(str(model.price))
                        break

        # Add unrealized P&L
        for position in self._positions.values():
            model = self._models.get(position.symbol)
            if model:
                equity += position.unrealized_pnl(Decimal(str(model.price)))

        return equity
