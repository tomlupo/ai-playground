"""Interactive Brokers exchange implementation for equity trading."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Callable, Any
from collections import defaultdict

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


@dataclass
class IBConfig(ExchangeConfig):
    """Interactive Brokers specific configuration."""
    # TWS/Gateway connection
    host: str = "127.0.0.1"
    port: int = 7497  # 7497 for TWS paper, 7496 for TWS live, 4001/4002 for Gateway
    client_id: int = 1

    # Account
    account_id: str = ""  # Leave empty to use first account

    # Paper trading
    paper_trading: bool = True

    # Market data
    market_data_type: int = 3  # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen

    # Timeouts
    connection_timeout: float = 30.0
    order_timeout: float = 60.0


class InteractiveBrokersExchange(Exchange):
    """
    Interactive Brokers exchange adapter for equity trading.

    Requires TWS or IB Gateway to be running. Uses the IB API for communication.

    Note: This implementation provides the interface and mock behavior.
    For production use, integrate with ib_insync or ibapi library.

    Usage:
        config = IBConfig(
            host="127.0.0.1",
            port=7497,  # TWS paper trading port
            client_id=1,
            paper_trading=True,
        )
        exchange = InteractiveBrokersExchange(config)
        await exchange.connect()

        # Place a market order for Apple stock
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET,
        )
        result = await exchange.submit_order(order)
    """

    # IB Order type mapping
    ORDER_TYPE_MAP = {
        OrderType.MARKET: "MKT",
        OrderType.LIMIT: "LMT",
        OrderType.STOP: "STP",
        OrderType.STOP_LIMIT: "STP LMT",
        OrderType.TRAILING_STOP: "TRAIL",
    }

    # IB Time in force mapping
    TIF_MAP = {
        TimeInForce.GTC: "GTC",
        TimeInForce.IOC: "IOC",
        TimeInForce.FOK: "FOK",
        TimeInForce.DAY: "DAY",
        TimeInForce.OPG: "OPG",
        TimeInForce.CLS: "CLS",
    }

    # Data resolution mapping
    RESOLUTION_MAP = {
        DataResolution.SECOND_1: "1 secs",
        DataResolution.SECOND_5: "5 secs",
        DataResolution.SECOND_15: "15 secs",
        DataResolution.SECOND_30: "30 secs",
        DataResolution.MINUTE_1: "1 min",
        DataResolution.MINUTE_5: "5 mins",
        DataResolution.MINUTE_15: "15 mins",
        DataResolution.MINUTE_30: "30 mins",
        DataResolution.HOUR_1: "1 hour",
        DataResolution.HOUR_4: "4 hours",
        DataResolution.DAY_1: "1 day",
        DataResolution.WEEK_1: "1 week",
        DataResolution.MONTH_1: "1 month",
    }

    def __init__(self, config: IBConfig):
        super().__init__(config)
        self.config: IBConfig = config

        # IB API state
        self._ib = None  # ib_insync.IB instance
        self._next_order_id = 1
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._balances: dict[str, Balance] = {}

        # Subscription tracking
        self._tick_subscriptions: dict[str, Any] = {}
        self._bar_subscriptions: dict[str, Any] = {}
        self._tick_callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._bar_callbacks: dict[str, list[Callable]] = defaultdict(list)

    @property
    def exchange_type(self) -> ExchangeType:
        return ExchangeType.INTERACTIVE_BROKERS

    @property
    def name(self) -> str:
        return "Interactive Brokers"

    # ==================== Connection ====================

    async def connect(self) -> bool:
        """
        Connect to TWS/IB Gateway.

        Requires ib_insync library: pip install ib_insync
        """
        try:
            # Try to import ib_insync
            try:
                from ib_insync import IB, util
                util.startLoop()  # Needed for asyncio integration
            except ImportError:
                # Fallback to mock mode
                print("ib_insync not installed, running in mock mode")
                self._connected = True
                self._setup_mock_data()
                return True

            self._ib = IB()
            await asyncio.wait_for(
                asyncio.to_thread(
                    self._ib.connect,
                    self.config.host,
                    self.config.port,
                    clientId=self.config.client_id,
                    readonly=False,
                ),
                timeout=self.config.connection_timeout,
            )

            # Set market data type
            self._ib.reqMarketDataType(self.config.market_data_type)

            # Get account info
            if not self.config.account_id:
                accounts = self._ib.managedAccounts()
                if accounts:
                    self.config.account_id = accounts[0]

            self._connected = True

            # Register callbacks
            self._ib.orderStatusEvent += self._on_order_status
            self._ib.execDetailsEvent += self._on_execution
            self._ib.positionEvent += self._on_position

            self.events.emit(OrderEvent(
                event_type=EventType.CONNECTED,
                source=self.name,
            ))

            return True

        except asyncio.TimeoutError:
            raise ExchangeError("Connection to IB timed out")
        except Exception as e:
            raise ExchangeError(f"Failed to connect to IB: {e}")

    def _setup_mock_data(self) -> None:
        """Setup mock data for testing without IB connection."""
        self._balances["USD"] = Balance(
            currency="USD",
            total=Decimal("100000"),
            available=Decimal("100000"),
        )

    async def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        await self.unsubscribe_all()

        if self._ib:
            self._ib.disconnect()
            self._ib = None

        self._connected = False

        self.events.emit(OrderEvent(
            event_type=EventType.DISCONNECTED,
            source=self.name,
        ))

    # ==================== IB Callbacks ====================

    def _on_order_status(self, trade: Any) -> None:
        """Handle order status updates from IB."""
        order_id = str(trade.order.orderId)
        if order_id in self._orders:
            order = self._orders[order_id]
            status = trade.orderStatus.status

            status_map = {
                "PendingSubmit": OrderStatus.PENDING,
                "PreSubmitted": OrderStatus.SUBMITTED,
                "Submitted": OrderStatus.ACCEPTED,
                "Filled": OrderStatus.FILLED,
                "Cancelled": OrderStatus.CANCELLED,
                "ApiCancelled": OrderStatus.CANCELLED,
            }

            order.status = status_map.get(status, OrderStatus.ACCEPTED)
            order.filled_quantity = Decimal(str(trade.orderStatus.filled))

            if trade.orderStatus.avgFillPrice:
                order.average_fill_price = Decimal(str(trade.orderStatus.avgFillPrice))

            event_type = {
                OrderStatus.FILLED: EventType.ORDER_FILLED,
                OrderStatus.CANCELLED: EventType.ORDER_CANCELLED,
                OrderStatus.PARTIALLY_FILLED: EventType.ORDER_PARTIALLY_FILLED,
            }.get(order.status, EventType.ORDER_ACCEPTED)

            self.events.emit(OrderEvent(
                event_type=event_type,
                order=order,
                source=self.name,
            ))

    def _on_execution(self, trade: Any, fill: Any) -> None:
        """Handle execution/fill events from IB."""
        order_id = str(trade.order.orderId)
        if order_id in self._orders:
            order = self._orders[order_id]

            fill_obj = Fill(
                fill_id=str(fill.execution.execId),
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=Decimal(str(fill.execution.shares)),
                price=Decimal(str(fill.execution.price)),
                commission=Decimal(str(fill.commissionReport.commission)) if fill.commissionReport else Decimal("0"),
                timestamp=datetime.now(),
            )

            order.add_fill(fill_obj)

    def _on_position(self, position: Any) -> None:
        """Handle position updates from IB."""
        symbol = position.contract.symbol
        qty = Decimal(str(position.position))

        if qty == 0:
            if symbol in self._positions:
                del self._positions[symbol]
            return

        pos = Position(
            symbol=symbol,
            side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
            quantity=abs(qty),
            average_entry_price=Decimal(str(position.avgCost)) / abs(qty) if qty != 0 else Decimal("0"),
        )
        self._positions[symbol] = pos

        self.events.emit(PositionEvent(
            event_type=EventType.POSITION_UPDATED,
            position=pos,
            source=self.name,
        ))

    # ==================== Instruments ====================

    async def get_instruments(self) -> list[Instrument]:
        """Get available instruments."""
        # IB doesn't have a simple "list all" endpoint
        # Return cached instruments
        return list(self._instruments.values())

    async def get_instrument(self, symbol: str) -> Optional[Instrument]:
        """Get instrument details."""
        if symbol in self._instruments:
            return self._instruments[symbol]

        # Create a basic equity instrument
        instrument = Instrument(
            symbol=symbol,
            base_currency=symbol,
            quote_currency="USD",
            asset_class=AssetClass.EQUITY,
            exchange="SMART",
            lot_size=Decimal("1"),
            tick_size=Decimal("0.01"),
        )
        self._instruments[symbol] = instrument
        return instrument

    # ==================== Account ====================

    async def get_balances(self) -> list[Balance]:
        """Get account balances."""
        if not self._ib:
            return list(self._balances.values())

        try:
            account_values = self._ib.accountValues(self.config.account_id)

            balances = {}
            for av in account_values:
                if av.tag == "TotalCashBalance" and av.currency != "BASE":
                    balances[av.currency] = Balance(
                        currency=av.currency,
                        total=Decimal(av.value),
                        available=Decimal(av.value),
                    )

            return list(balances.values())

        except Exception as e:
            raise ExchangeError(f"Failed to get balances: {e}")

    async def get_balance(self, currency: str) -> Optional[Balance]:
        """Get balance for specific currency."""
        balances = await self.get_balances()
        for b in balances:
            if b.currency == currency:
                return b
        return None

    # ==================== Orders ====================

    async def submit_order(self, order: Order) -> Order:
        """Submit order to IB."""
        if not self._ib:
            # Mock mode
            return await self._submit_mock_order(order)

        try:
            from ib_insync import Stock, LimitOrder, MarketOrder, StopOrder

            # Create contract
            contract = Stock(order.symbol, "SMART", "USD")

            # Create IB order
            action = "BUY" if order.side == OrderSide.BUY else "SELL"
            quantity = float(order.quantity)

            if order.order_type == OrderType.MARKET:
                ib_order = MarketOrder(action, quantity)
            elif order.order_type == OrderType.LIMIT:
                ib_order = LimitOrder(action, quantity, float(order.limit_price))
            elif order.order_type == OrderType.STOP:
                ib_order = StopOrder(action, quantity, float(order.stop_price))
            else:
                raise ValueError(f"Unsupported order type: {order.order_type}")

            # Set time in force
            ib_order.tif = self.TIF_MAP.get(order.time_in_force, "DAY")

            # Place order
            trade = self._ib.placeOrder(contract, ib_order)

            order.exchange_order_id = str(trade.order.orderId)
            order.submitted_at = datetime.utcnow()
            order.status = OrderStatus.SUBMITTED

            self._orders[order.order_id] = order
            self._orders[order.exchange_order_id] = order

            self.events.emit(OrderEvent(
                event_type=EventType.ORDER_SUBMITTED,
                order=order,
                source=self.name,
            ))

            return order

        except Exception as e:
            raise ExchangeError(f"Failed to submit order: {e}")

    async def _submit_mock_order(self, order: Order) -> Order:
        """Submit order in mock mode."""
        order.exchange_order_id = str(self._next_order_id)
        self._next_order_id += 1
        order.submitted_at = datetime.utcnow()
        order.status = OrderStatus.ACCEPTED

        self._orders[order.order_id] = order
        self._orders[order.exchange_order_id] = order

        # Simulate immediate fill for market orders
        if order.order_type == OrderType.MARKET:
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = Decimal("150.00")  # Mock price
            order.filled_at = datetime.utcnow()

        self.events.emit(OrderEvent(
            event_type=EventType.ORDER_SUBMITTED,
            order=order,
            source=self.name,
        ))

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        order = self._orders.get(order_id)
        if not order:
            return False

        if not self._ib:
            # Mock mode
            order.status = OrderStatus.CANCELLED
            order.cancelled_at = datetime.utcnow()
            return True

        try:
            # Find the trade object
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == order.exchange_order_id:
                    self._ib.cancelOrder(trade.order)
                    return True
            return False
        except Exception as e:
            raise ExchangeError(f"Failed to cancel order: {e}")

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders."""
        cancelled = 0

        if not self._ib:
            # Mock mode
            for order in list(self._orders.values()):
                if order.is_active and (symbol is None or order.symbol == symbol):
                    order.status = OrderStatus.CANCELLED
                    cancelled += 1
            return cancelled

        try:
            for trade in self._ib.openTrades():
                if symbol is None or trade.contract.symbol == symbol:
                    self._ib.cancelOrder(trade.order)
                    cancelled += 1
            return cancelled
        except Exception as e:
            raise ExchangeError(f"Failed to cancel orders: {e}")

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Get all open orders."""
        if not self._ib:
            # Mock mode
            return [
                o for o in self._orders.values()
                if o.is_active and (symbol is None or o.symbol == symbol)
            ]

        orders = []
        for trade in self._ib.openTrades():
            if symbol is None or trade.contract.symbol == symbol:
                order_id = str(trade.order.orderId)
                if order_id in self._orders:
                    orders.append(self._orders[order_id])

        return orders

    # ==================== Positions ====================

    async def get_positions(self) -> list[Position]:
        """Get all positions."""
        if not self._ib:
            return list(self._positions.values())

        try:
            positions = []
            for pos in self._ib.positions(self.config.account_id):
                qty = Decimal(str(pos.position))
                if qty != 0:
                    position = Position(
                        symbol=pos.contract.symbol,
                        side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
                        quantity=abs(qty),
                        average_entry_price=Decimal(str(pos.avgCost)) / abs(qty),
                    )
                    positions.append(position)
            return positions
        except Exception as e:
            raise ExchangeError(f"Failed to get positions: {e}")

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        positions = await self.get_positions()
        for p in positions:
            if p.symbol == symbol:
                return p
        return None

    # ==================== Market Data ====================

    async def get_ticker(self, symbol: str) -> Optional[Tick]:
        """Get current ticker."""
        if not self._ib:
            # Mock data
            return Tick(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                bid=Decimal("149.95"),
                ask=Decimal("150.05"),
                bid_size=Decimal("100"),
                ask_size=Decimal("100"),
                last_price=Decimal("150.00"),
            )

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)

            ticker = self._ib.reqMktData(contract, snapshot=True)
            await asyncio.sleep(2)  # Wait for data

            return Tick(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                bid=Decimal(str(ticker.bid)) if ticker.bid else Decimal("0"),
                ask=Decimal(str(ticker.ask)) if ticker.ask else Decimal("0"),
                bid_size=Decimal(str(ticker.bidSize)) if ticker.bidSize else Decimal("0"),
                ask_size=Decimal(str(ticker.askSize)) if ticker.askSize else Decimal("0"),
                last_price=Decimal(str(ticker.last)) if ticker.last else None,
                last_size=Decimal(str(ticker.lastSize)) if ticker.lastSize else None,
            )

        except Exception as e:
            raise ExchangeError(f"Failed to get ticker: {e}")

    async def get_order_book(self, symbol: str, depth: int = 10) -> Optional[OrderBook]:
        """Get order book."""
        if not self._ib:
            # Mock order book
            mid = Decimal("150.00")
            return OrderBook(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                bids=[
                    OrderBookLevel(mid - Decimal(str(i * 0.01)), Decimal("100"))
                    for i in range(1, depth + 1)
                ],
                asks=[
                    OrderBookLevel(mid + Decimal(str(i * 0.01)), Decimal("100"))
                    for i in range(1, depth + 1)
                ],
            )

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)

            ticker = self._ib.reqMktDepth(contract, numRows=depth)
            await asyncio.sleep(2)

            return OrderBook(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                bids=[
                    OrderBookLevel(
                        price=Decimal(str(d.price)),
                        quantity=Decimal(str(d.size)),
                    )
                    for d in ticker.domBids[:depth]
                ],
                asks=[
                    OrderBookLevel(
                        price=Decimal(str(d.price)),
                        quantity=Decimal(str(d.size)),
                    )
                    for d in ticker.domAsks[:depth]
                ],
            )

        except Exception as e:
            raise ExchangeError(f"Failed to get order book: {e}")

    async def get_historical_bars(
        self,
        symbol: str,
        resolution: DataResolution,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[OHLCV]:
        """Get historical OHLCV data."""
        if not self._ib:
            # Generate mock historical data
            bars = []
            current = start
            end_time = end or datetime.utcnow()
            interval = timedelta(seconds=resolution.to_seconds()) or timedelta(hours=1)

            price = Decimal("150.00")
            while current <= end_time and len(bars) < limit:
                change = Decimal(str(0.5 - (hash(str(current)) % 100) / 100))
                bars.append(OHLCV(
                    symbol=symbol,
                    timestamp=current,
                    open=price,
                    high=price + abs(change),
                    low=price - abs(change),
                    close=price + change,
                    volume=Decimal("10000"),
                ))
                price = price + change
                current += interval

            return bars

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)

            bar_size = self.RESOLUTION_MAP.get(resolution, "1 hour")
            duration = self._calculate_duration(start, end)

            ib_bars = self._ib.reqHistoricalData(
                contract,
                endDateTime=end or datetime.utcnow(),
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
            )

            return [
                OHLCV(
                    symbol=symbol,
                    timestamp=bar.date,
                    open=Decimal(str(bar.open)),
                    high=Decimal(str(bar.high)),
                    low=Decimal(str(bar.low)),
                    close=Decimal(str(bar.close)),
                    volume=Decimal(str(bar.volume)),
                )
                for bar in ib_bars[:limit]
            ]

        except Exception as e:
            raise ExchangeError(f"Failed to get historical bars: {e}")

    def _calculate_duration(self, start: datetime, end: Optional[datetime]) -> str:
        """Calculate IB duration string."""
        end_time = end or datetime.utcnow()
        delta = end_time - start

        if delta.days > 365:
            return f"{delta.days // 365} Y"
        elif delta.days > 30:
            return f"{delta.days // 30} M"
        elif delta.days > 0:
            return f"{delta.days} D"
        else:
            return f"{int(delta.total_seconds())} S"

    # ==================== Streaming ====================

    async def subscribe_ticks(
        self,
        symbols: list[str],
        callback: Callable[[Tick], None]
    ) -> None:
        """Subscribe to tick data."""
        for symbol in symbols:
            self._tick_callbacks[symbol].append(callback)

            if symbol not in self._tick_subscriptions:
                task = asyncio.create_task(self._stream_ticks(symbol))
                self._tick_subscriptions[symbol] = task

    async def _stream_ticks(self, symbol: str) -> None:
        """Stream tick data for symbol."""
        while symbol in self._tick_callbacks:
            try:
                tick = await self.get_ticker(symbol)
                if tick:
                    for callback in self._tick_callbacks[symbol]:
                        callback(tick)
                    self.events.emit(TickEvent(tick=tick, source=self.name))

                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(5.0)

    async def subscribe_bars(
        self,
        symbols: list[str],
        resolution: DataResolution,
        callback: Callable[[OHLCV], None]
    ) -> None:
        """Subscribe to bar data."""
        for symbol in symbols:
            key = f"{symbol}_{resolution.name}"
            self._bar_callbacks[key].append(callback)

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all data streams."""
        for task in self._tick_subscriptions.values():
            task.cancel()
        self._tick_subscriptions.clear()
        self._tick_callbacks.clear()
        self._bar_callbacks.clear()
