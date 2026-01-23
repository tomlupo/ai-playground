"""Abstract base class for exchange implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from typing import Optional, Callable, AsyncIterator

from trading.core.models import (
    Order,
    Position,
    Trade,
    Tick,
    OHLCV,
    OrderBook,
    Balance,
    Instrument,
    Fill,
)
from trading.core.enums import (
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    ExchangeType,
    DataResolution,
)
from trading.core.events import EventEmitter, AsyncEventEmitter


@dataclass
class ExchangeConfig:
    """Base configuration for exchange connections."""
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    sandbox: bool = False

    # Connection settings
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0

    # Rate limiting
    rate_limit_per_second: float = 10.0

    # Additional settings
    extra: dict = field(default_factory=dict)


class Exchange(ABC):
    """
    Abstract base class for exchange implementations.

    Provides a unified interface for:
    - Order management (submit, cancel, modify)
    - Position tracking
    - Market data (ticks, bars, order book)
    - Account information (balances)

    Implementations:
    - BinanceExchange: Cryptocurrency trading
    - InteractiveBrokersExchange: Equity trading
    - OandaExchange: Forex/Futures trading
    - SimulatedExchange: Paper trading with market simulation
    """

    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.events = EventEmitter()
        self._connected = False
        self._instruments: dict[str, Instrument] = {}

    @property
    @abstractmethod
    def exchange_type(self) -> ExchangeType:
        """Return the exchange type identifier."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable exchange name."""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if connected to exchange."""
        return self._connected

    # ==================== Connection Management ====================

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the exchange.

        Returns:
            True if connection successful, False otherwise.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        pass

    # ==================== Instrument Information ====================

    @abstractmethod
    async def get_instruments(self) -> list[Instrument]:
        """
        Fetch all available instruments.

        Returns:
            List of tradeable instruments.
        """
        pass

    @abstractmethod
    async def get_instrument(self, symbol: str) -> Optional[Instrument]:
        """
        Fetch instrument details for a specific symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT", "AAPL", "EUR_USD")

        Returns:
            Instrument details or None if not found.
        """
        pass

    # ==================== Account Information ====================

    @abstractmethod
    async def get_balances(self) -> list[Balance]:
        """
        Fetch account balances for all currencies/assets.

        Returns:
            List of Balance objects.
        """
        pass

    @abstractmethod
    async def get_balance(self, currency: str) -> Optional[Balance]:
        """
        Fetch balance for a specific currency.

        Args:
            currency: Currency code (e.g., "USD", "BTC")

        Returns:
            Balance object or None.
        """
        pass

    # ==================== Order Management ====================

    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """
        Submit an order to the exchange.

        Args:
            order: Order to submit.

        Returns:
            Updated order with exchange order ID and status.

        Raises:
            OrderRejectedError: If order is rejected.
            InsufficientFundsError: If insufficient balance.
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Internal order ID or exchange order ID.

        Returns:
            True if cancellation successful.
        """
        pass

    @abstractmethod
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders, optionally for a specific symbol.

        Args:
            symbol: Optional symbol to filter orders.

        Returns:
            Number of orders cancelled.
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Fetch order status and details.

        Args:
            order_id: Order identifier.

        Returns:
            Order object or None if not found.
        """
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """
        Fetch all open orders.

        Args:
            symbol: Optional symbol filter.

        Returns:
            List of open orders.
        """
        pass

    async def modify_order(
        self,
        order_id: str,
        quantity: Optional[Decimal] = None,
        limit_price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
    ) -> Order:
        """
        Modify an existing order.

        Default implementation cancels and replaces.
        Override for exchanges that support native modification.

        Args:
            order_id: Order to modify.
            quantity: New quantity (optional).
            limit_price: New limit price (optional).
            stop_price: New stop price (optional).

        Returns:
            New or modified order.
        """
        order = await self.get_order(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")

        await self.cancel_order(order_id)

        # Create new order with modified parameters
        new_order = Order(
            symbol=order.symbol,
            side=order.side,
            quantity=quantity or order.quantity,
            order_type=order.order_type,
            limit_price=limit_price or order.limit_price,
            stop_price=stop_price or order.stop_price,
            time_in_force=order.time_in_force,
            reduce_only=order.reduce_only,
            post_only=order.post_only,
            strategy_id=order.strategy_id,
            tags=order.tags,
        )

        return await self.submit_order(new_order)

    # ==================== Position Management ====================

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """
        Fetch all open positions.

        Returns:
            List of Position objects.
        """
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Fetch position for a specific symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            Position object or None if no position.
        """
        pass

    async def close_position(
        self,
        symbol: str,
        quantity: Optional[Decimal] = None
    ) -> Order:
        """
        Close an open position.

        Args:
            symbol: Position symbol.
            quantity: Optional partial close quantity.

        Returns:
            Closing order.
        """
        position = await self.get_position(symbol)
        if not position or position.quantity == 0:
            raise ValueError(f"No open position for {symbol}")

        close_qty = quantity or position.quantity
        close_side = OrderSide.SELL if position.is_long else OrderSide.BUY

        close_order = Order(
            symbol=symbol,
            side=close_side,
            quantity=close_qty,
            order_type=OrderType.MARKET,
            reduce_only=True,
        )

        return await self.submit_order(close_order)

    # ==================== Market Data ====================

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Optional[Tick]:
        """
        Fetch current ticker/quote for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            Current Tick or None.
        """
        pass

    @abstractmethod
    async def get_order_book(
        self,
        symbol: str,
        depth: int = 10
    ) -> Optional[OrderBook]:
        """
        Fetch order book depth.

        Args:
            symbol: Trading symbol.
            depth: Number of levels to fetch.

        Returns:
            OrderBook or None.
        """
        pass

    @abstractmethod
    async def get_historical_bars(
        self,
        symbol: str,
        resolution: DataResolution,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[OHLCV]:
        """
        Fetch historical OHLCV bars.

        Args:
            symbol: Trading symbol.
            resolution: Bar timeframe.
            start: Start datetime.
            end: End datetime (default: now).
            limit: Maximum bars to fetch.

        Returns:
            List of OHLCV bars.
        """
        pass

    # ==================== Streaming Data ====================

    @abstractmethod
    async def subscribe_ticks(
        self,
        symbols: list[str],
        callback: Callable[[Tick], None]
    ) -> None:
        """
        Subscribe to real-time tick data.

        Args:
            symbols: List of symbols to subscribe.
            callback: Function to call on each tick.
        """
        pass

    @abstractmethod
    async def subscribe_bars(
        self,
        symbols: list[str],
        resolution: DataResolution,
        callback: Callable[[OHLCV], None]
    ) -> None:
        """
        Subscribe to real-time bar data.

        Args:
            symbols: List of symbols.
            resolution: Bar timeframe.
            callback: Function to call on each bar.
        """
        pass

    @abstractmethod
    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all data streams."""
        pass

    # ==================== Utility Methods ====================

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for this exchange.

        Override in subclasses for exchange-specific formatting.

        Args:
            symbol: Input symbol.

        Returns:
            Normalized symbol string.
        """
        return symbol.upper()

    def round_price(self, symbol: str, price: Decimal) -> Decimal:
        """
        Round price to valid tick size for symbol.

        Args:
            symbol: Trading symbol.
            price: Price to round.

        Returns:
            Rounded price.
        """
        instrument = self._instruments.get(symbol)
        if instrument:
            tick_size = instrument.tick_size
            return (price / tick_size).quantize(Decimal("1")) * tick_size
        return price

    def round_quantity(self, symbol: str, quantity: Decimal) -> Decimal:
        """
        Round quantity to valid lot size for symbol.

        Args:
            symbol: Trading symbol.
            quantity: Quantity to round.

        Returns:
            Rounded quantity.
        """
        instrument = self._instruments.get(symbol)
        if instrument:
            lot_size = instrument.lot_size
            return (quantity / lot_size).quantize(Decimal("1")) * lot_size
        return quantity


class ExchangeError(Exception):
    """Base exception for exchange errors."""
    pass


class ConnectionError(ExchangeError):
    """Connection to exchange failed."""
    pass


class OrderRejectedError(ExchangeError):
    """Order was rejected by exchange."""
    def __init__(self, order: Order, reason: str):
        self.order = order
        self.reason = reason
        super().__init__(f"Order rejected: {reason}")


class InsufficientFundsError(ExchangeError):
    """Insufficient funds for order."""
    def __init__(self, required: Decimal, available: Decimal, currency: str):
        self.required = required
        self.available = available
        self.currency = currency
        super().__init__(
            f"Insufficient {currency}: required {required}, available {available}"
        )


class RateLimitError(ExchangeError):
    """Rate limit exceeded."""
    def __init__(self, retry_after: Optional[float] = None):
        self.retry_after = retry_after
        msg = "Rate limit exceeded"
        if retry_after:
            msg += f", retry after {retry_after}s"
        super().__init__(msg)
