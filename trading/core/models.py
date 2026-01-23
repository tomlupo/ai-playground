"""Core domain models for the trading framework."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import uuid4

from trading.core.enums import (
    OrderType,
    OrderSide,
    OrderStatus,
    PositionSide,
    TimeInForce,
    AssetClass,
    FillType,
)


@dataclass
class Instrument:
    """
    Tradeable instrument definition.

    Represents any tradeable asset across different exchanges/brokers.
    """
    symbol: str                          # Trading symbol (e.g., "BTCUSDT", "AAPL", "EUR_USD")
    base_currency: str                   # Base asset (e.g., "BTC", "AAPL", "EUR")
    quote_currency: str                  # Quote asset (e.g., "USDT", "USD", "USD")
    asset_class: AssetClass
    exchange: str                        # Exchange identifier

    # Contract specifications
    lot_size: Decimal = Decimal("1")     # Minimum tradeable quantity
    tick_size: Decimal = Decimal("0.01") # Minimum price increment
    min_notional: Decimal = Decimal("0") # Minimum order value
    max_quantity: Optional[Decimal] = None

    # Margin/leverage (for derivatives)
    margin_required: Decimal = Decimal("1.0")  # 1.0 = no leverage
    max_leverage: Decimal = Decimal("1.0")

    # Trading hours (for equities)
    trading_hours: Optional[str] = None  # e.g., "09:30-16:00 EST"

    # Contract details (for futures)
    contract_size: Decimal = Decimal("1")
    expiry_date: Optional[datetime] = None

    def __post_init__(self):
        # Ensure Decimal types
        if not isinstance(self.lot_size, Decimal):
            self.lot_size = Decimal(str(self.lot_size))
        if not isinstance(self.tick_size, Decimal):
            self.tick_size = Decimal(str(self.tick_size))
        if not isinstance(self.min_notional, Decimal):
            self.min_notional = Decimal(str(self.min_notional))


@dataclass
class Tick:
    """
    Single price tick/quote.

    Represents a point-in-time market data update.
    """
    symbol: str
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    bid_size: Decimal = Decimal("0")
    ask_size: Decimal = Decimal("0")
    last_price: Optional[Decimal] = None
    last_size: Optional[Decimal] = None
    volume: Optional[Decimal] = None

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid-price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> Decimal:
        """Calculate spread in basis points."""
        if self.mid_price == 0:
            return Decimal("0")
        return (self.spread / self.mid_price) * 10000


@dataclass
class OHLCV:
    """
    OHLCV bar/candle data.

    Represents aggregated price data over a time period.
    """
    symbol: str
    timestamp: datetime          # Bar open time
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    trades: int = 0              # Number of trades in bar
    vwap: Optional[Decimal] = None  # Volume-weighted average price

    @property
    def range(self) -> Decimal:
        """Price range (high - low)."""
        return self.high - self.low

    @property
    def body(self) -> Decimal:
        """Candle body size (close - open)."""
        return self.close - self.open

    @property
    def is_bullish(self) -> bool:
        """True if close > open."""
        return self.close > self.open


@dataclass
class OrderBookLevel:
    """Single level in the order book."""
    price: Decimal
    quantity: Decimal
    order_count: int = 1


@dataclass
class OrderBook:
    """
    Level 2 order book data.

    Contains bid/ask depth information.
    """
    symbol: str
    timestamp: datetime
    bids: list[OrderBookLevel] = field(default_factory=list)  # Sorted desc by price
    asks: list[OrderBookLevel] = field(default_factory=list)  # Sorted asc by price

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[Decimal]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[Decimal]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    def get_depth_at_price(self, price: Decimal, side: OrderSide) -> Decimal:
        """Get total quantity available at or better than price."""
        total = Decimal("0")
        levels = self.bids if side == OrderSide.BUY else self.asks

        for level in levels:
            if side == OrderSide.BUY and level.price >= price:
                total += level.quantity
            elif side == OrderSide.SELL and level.price <= price:
                total += level.quantity

        return total


@dataclass
class Fill:
    """Individual order fill/execution."""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    commission: Decimal
    timestamp: datetime
    fill_type: FillType = FillType.COMPLETE
    liquidity: str = "TAKER"  # MAKER or TAKER


@dataclass
class Order:
    """
    Trading order representation.

    Supports all common order types across different exchanges.
    """
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET

    # Price parameters (depending on order type)
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    trailing_delta: Optional[Decimal] = None  # For trailing stops

    # Order policies
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False        # Only reduce position
    post_only: bool = False          # Only maker orders

    # Identifiers
    order_id: str = field(default_factory=lambda: str(uuid4()))
    client_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None

    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Optional[Decimal] = None
    fills: list[Fill] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    # Metadata
    strategy_id: Optional[str] = None
    tags: dict = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> Decimal:
        """Unfilled quantity."""
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        """True if order can still be filled."""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
        )

    @property
    def is_complete(self) -> bool:
        """True if order is in terminal state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    def add_fill(self, fill: Fill) -> None:
        """Record a fill for this order."""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity

        # Update average fill price
        total_value = sum(f.price * f.quantity for f in self.fills)
        total_qty = sum(f.quantity for f in self.fills)
        self.average_fill_price = total_value / total_qty if total_qty > 0 else None

        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = fill.timestamp
        else:
            self.status = OrderStatus.PARTIALLY_FILLED


@dataclass
class Position:
    """
    Open trading position.

    Tracks quantity, cost basis, and P&L for a single instrument.
    """
    symbol: str
    side: PositionSide
    quantity: Decimal
    average_entry_price: Decimal

    # P&L tracking
    realized_pnl: Decimal = Decimal("0")
    commission_paid: Decimal = Decimal("0")

    # Position management
    opened_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    # For margin/derivatives
    leverage: Decimal = Decimal("1")
    liquidation_price: Optional[Decimal] = None
    margin_used: Decimal = Decimal("0")

    # Metadata
    strategy_id: Optional[str] = None
    tags: dict = field(default_factory=dict)

    @property
    def notional_value(self) -> Decimal:
        """Position value at entry price."""
        return self.quantity * self.average_entry_price

    @property
    def is_long(self) -> bool:
        return self.side == PositionSide.LONG

    @property
    def is_short(self) -> bool:
        return self.side == PositionSide.SHORT

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L at given price."""
        if self.side == PositionSide.LONG:
            return (current_price - self.average_entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            return (self.average_entry_price - current_price) * self.quantity
        return Decimal("0")

    def unrealized_pnl_pct(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L as percentage."""
        if self.average_entry_price == 0:
            return Decimal("0")
        return (self.unrealized_pnl(current_price) / self.notional_value) * 100

    def update_from_fill(self, fill: Fill) -> Optional[Decimal]:
        """
        Update position from a fill.

        Returns realized P&L if position was reduced, None otherwise.
        """
        realized = None
        self.commission_paid += fill.commission
        self.last_updated = fill.timestamp

        # Determine if increasing or reducing position
        is_increasing = (
            (self.side == PositionSide.LONG and fill.side == OrderSide.BUY) or
            (self.side == PositionSide.SHORT and fill.side == OrderSide.SELL)
        )

        if is_increasing:
            # Add to position - update average entry
            total_cost = self.quantity * self.average_entry_price + fill.quantity * fill.price
            self.quantity += fill.quantity
            self.average_entry_price = total_cost / self.quantity
        else:
            # Reduce position - realize P&L
            reduce_qty = min(fill.quantity, self.quantity)
            if self.side == PositionSide.LONG:
                realized = (fill.price - self.average_entry_price) * reduce_qty
            else:
                realized = (self.average_entry_price - fill.price) * reduce_qty

            self.realized_pnl += realized
            self.quantity -= reduce_qty

            # Check if position flipped
            remaining = fill.quantity - reduce_qty
            if remaining > 0:
                # Position flipped to other side
                self.side = PositionSide.LONG if fill.side == OrderSide.BUY else PositionSide.SHORT
                self.quantity = remaining
                self.average_entry_price = fill.price
            elif self.quantity == 0:
                self.side = PositionSide.FLAT

        return realized


@dataclass
class Balance:
    """Account balance for a single currency/asset."""
    currency: str
    total: Decimal
    available: Decimal              # Free to use
    locked: Decimal = Decimal("0")  # In open orders

    @property
    def used(self) -> Decimal:
        """Total minus available."""
        return self.total - self.available


@dataclass
class Trade:
    """
    Completed round-trip trade.

    Aggregates entry and exit information for P&L reporting.
    """
    trade_id: str
    symbol: str
    side: PositionSide

    # Entry
    entry_price: Decimal
    entry_quantity: Decimal
    entry_time: datetime
    entry_orders: list[str] = field(default_factory=list)

    # Exit
    exit_price: Optional[Decimal] = None
    exit_quantity: Decimal = Decimal("0")
    exit_time: Optional[datetime] = None
    exit_orders: list[str] = field(default_factory=list)

    # P&L
    realized_pnl: Decimal = Decimal("0")
    commission_total: Decimal = Decimal("0")

    # Metadata
    strategy_id: Optional[str] = None
    tags: dict = field(default_factory=dict)

    @property
    def is_closed(self) -> bool:
        return self.exit_quantity >= self.entry_quantity

    @property
    def duration(self) -> Optional[float]:
        """Trade duration in seconds."""
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds()
        return None

    @property
    def net_pnl(self) -> Decimal:
        """P&L after commissions."""
        return self.realized_pnl - self.commission_total

    @property
    def return_pct(self) -> Decimal:
        """Return as percentage."""
        entry_value = self.entry_price * self.entry_quantity
        if entry_value == 0:
            return Decimal("0")
        return (self.net_pnl / entry_value) * 100
