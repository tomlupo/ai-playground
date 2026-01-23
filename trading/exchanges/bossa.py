"""
Bossa (DM BOŚ) exchange implementation for Polish market (GPW/WSE).

Uses bossaAPI which communicates via XML over sockets with bossaNOL3 platform.

Note: Requires bossaNOL3 platform to be running locally.
"""

import asyncio
import socket
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, Callable, Any
from collections import defaultdict
import re

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
)
from trading.core.events import (
    EventType,
    OrderEvent,
    TickEvent,
)
from trading.exchanges.base import (
    Exchange,
    ExchangeConfig,
    ExchangeError,
    OrderRejectedError,
    InsufficientFundsError,
)


@dataclass
class BossaConfig(ExchangeConfig):
    """Bossa (bossaNOL3) specific configuration."""
    # Connection settings (read from registry when NOL3 connects)
    host: str = "127.0.0.1"
    port: int = 0  # 0 = auto-detect from registry

    # Account
    account_id: str = ""

    # Timeouts
    connection_timeout: float = 30.0
    read_timeout: float = 10.0

    # Registry path for port auto-detection (Windows)
    registry_path: str = r"SOFTWARE\COMARCH\NOL3\Settings"


# WSE/GPW instrument definitions
WSE_INDICES = ["WIG20", "WIG", "mWIG40", "sWIG80", "WIG30"]
WSE_BLUE_CHIPS = [
    "PKO", "PKN", "PZU", "PEO", "KGH", "CDR", "DNP", "SPL", "LPP", "CCC",
    "ALE", "OPL", "JSW", "PGE", "TPE", "MBK", "CPS", "11B", "KRU", "PCO",
]


class BossaExchange(Exchange):
    """
    Bossa (DM BOŚ) exchange adapter for Polish market trading.

    Connects to bossaNOL3 platform via socket-based bossaAPI.
    Supports GPW (Warsaw Stock Exchange) instruments.

    Prerequisites:
    - bossaNOL3 platform must be installed and running
    - Active account with Dom Maklerski BOŚ

    Usage:
        config = BossaConfig(
            account_id="your_account",
        )
        exchange = BossaExchange(config)
        await exchange.connect()

        # Trade WSE stocks
        order = Order(
            symbol="PKO",  # PKO Bank Polski
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("45.50"),
        )
        result = await exchange.submit_order(order)
    """

    # Order type mapping to bossaAPI
    ORDER_TYPE_MAP = {
        OrderType.MARKET: "PKC",  # Po Każdej Cenie
        OrderType.LIMIT: "LIMIT",
        OrderType.STOP: "STOP",
        OrderType.STOP_LIMIT: "STOP_LIMIT",
    }

    ORDER_SIDE_MAP = {
        OrderSide.BUY: "K",  # Kupno
        OrderSide.SELL: "S",  # Sprzedaż
    }

    # Time in force mapping
    TIF_MAP = {
        TimeInForce.DAY: "D",  # Dzień
        TimeInForce.GTC: "WDD",  # Ważne Do Dnia
        TimeInForce.IOC: "WIA",  # Wykonaj I Anuluj
        TimeInForce.FOK: "WUL",  # Wykonaj Lub Anuluj
    }

    def __init__(self, config: BossaConfig):
        super().__init__(config)
        self.config: BossaConfig = config

        # Socket connection
        self._socket: Optional[socket.socket] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

        # State
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._balances: dict[str, Balance] = {}
        self._next_order_id = 1

        # Subscriptions
        self._tick_callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._stream_task: Optional[asyncio.Task] = None

        # Message buffer
        self._message_buffer = ""

    @property
    def exchange_type(self) -> ExchangeType:
        return ExchangeType.INTERACTIVE_BROKERS  # Using IB as placeholder

    @property
    def name(self) -> str:
        return "Bossa (GPW)"

    # ==================== Connection ====================

    async def connect(self) -> bool:
        """Connect to bossaNOL3 via socket."""
        try:
            # Auto-detect port if not specified
            port = self.config.port
            if port == 0:
                port = self._detect_port()
                if port == 0:
                    raise ExchangeError(
                        "Could not auto-detect bossaNOL3 port. "
                        "Make sure NOL3 is running."
                    )

            # Connect via asyncio streams
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.config.host, port),
                timeout=self.config.connection_timeout,
            )

            self._connected = True

            # Start message handler
            self._stream_task = asyncio.create_task(self._handle_messages())

            # Login and get account info
            await self._login()
            await self._load_account_info()

            self.events.emit(OrderEvent(
                event_type=EventType.CONNECTED,
                source=self.name,
            ))

            return True

        except asyncio.TimeoutError:
            raise ExchangeError("Connection to bossaNOL3 timed out")
        except ConnectionRefusedError:
            raise ExchangeError(
                "Connection refused. Make sure bossaNOL3 is running."
            )
        except Exception as e:
            raise ExchangeError(f"Failed to connect to bossaNOL3: {e}")

    def _detect_port(self) -> int:
        """Auto-detect bossaNOL3 port from Windows registry."""
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                self.config.registry_path,
            )
            port, _ = winreg.QueryValueEx(key, "Port")
            winreg.CloseKey(key)
            return int(port)
        except Exception:
            # Fallback to common port
            return 8223  # Default bossaAPI port

    async def disconnect(self) -> None:
        """Disconnect from bossaNOL3."""
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()

        self._reader = None
        self._writer = None
        self._connected = False

        self.events.emit(OrderEvent(
            event_type=EventType.DISCONNECTED,
            source=self.name,
        ))

    # ==================== Message Handling ====================

    async def _send_message(self, xml_msg: str) -> None:
        """Send XML message to bossaNOL3."""
        if not self._writer:
            raise ExchangeError("Not connected")

        # Ensure proper encoding and termination
        message = xml_msg.encode('utf-8') + b'\x00'
        self._writer.write(message)
        await self._writer.drain()

    async def _receive_message(self) -> Optional[str]:
        """Receive XML message from bossaNOL3."""
        if not self._reader:
            return None

        try:
            # Read until null terminator
            data = await asyncio.wait_for(
                self._reader.readuntil(b'\x00'),
                timeout=self.config.read_timeout,
            )
            return data[:-1].decode('utf-8')
        except asyncio.TimeoutError:
            return None

    async def _handle_messages(self) -> None:
        """Background task to handle incoming messages."""
        while self._connected:
            try:
                if not self._reader:
                    break

                # Read data
                data = await self._reader.read(4096)
                if not data:
                    break

                self._message_buffer += data.decode('utf-8')

                # Process complete messages (null-terminated)
                while '\x00' in self._message_buffer:
                    msg, self._message_buffer = self._message_buffer.split('\x00', 1)
                    if msg:
                        await self._process_message(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Message handler error: {e}")
                await asyncio.sleep(1.0)

    async def _process_message(self, xml_str: str) -> None:
        """Process incoming XML message."""
        try:
            root = ET.fromstring(xml_str)
            msg_type = root.tag

            if msg_type == "QuotationsUpdate":
                self._handle_quote_update(root)
            elif msg_type == "OrderStatusUpdate":
                self._handle_order_update(root)
            elif msg_type == "TradeConfirmation":
                self._handle_trade_confirmation(root)
            elif msg_type == "AccountUpdate":
                self._handle_account_update(root)

        except ET.ParseError as e:
            print(f"XML parse error: {e}")

    def _handle_quote_update(self, root: ET.Element) -> None:
        """Handle real-time quote update."""
        symbol = root.findtext("Symbol", "")
        bid = root.findtext("Bid", "0")
        ask = root.findtext("Ask", "0")
        last = root.findtext("Last", "0")
        volume = root.findtext("Volume", "0")

        tick = Tick(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=Decimal(bid),
            ask=Decimal(ask),
            last_price=Decimal(last) if last != "0" else None,
            volume=Decimal(volume) if volume != "0" else None,
        )

        # Emit to callbacks
        for callback in self._tick_callbacks.get(symbol, []):
            try:
                callback(tick)
            except Exception as e:
                print(f"Tick callback error: {e}")

        self.events.emit(TickEvent(tick=tick, source=self.name))

    def _handle_order_update(self, root: ET.Element) -> None:
        """Handle order status update."""
        order_id = root.findtext("OrderId", "")
        status_str = root.findtext("Status", "")

        if order_id in self._orders:
            order = self._orders[order_id]

            status_map = {
                "PENDING": OrderStatus.PENDING,
                "ACCEPTED": OrderStatus.ACCEPTED,
                "PARTIAL": OrderStatus.PARTIALLY_FILLED,
                "FILLED": OrderStatus.FILLED,
                "CANCELLED": OrderStatus.CANCELLED,
                "REJECTED": OrderStatus.REJECTED,
            }
            order.status = status_map.get(status_str, OrderStatus.PENDING)

            filled = root.findtext("FilledQty", "0")
            order.filled_quantity = Decimal(filled)

            avg_price = root.findtext("AvgPrice", "0")
            if avg_price != "0":
                order.average_fill_price = Decimal(avg_price)

            self.events.emit(OrderEvent(
                event_type=EventType.ORDER_FILLED if order.status == OrderStatus.FILLED
                else EventType.ORDER_ACCEPTED,
                order=order,
                source=self.name,
            ))

    def _handle_trade_confirmation(self, root: ET.Element) -> None:
        """Handle trade execution confirmation."""
        order_id = root.findtext("OrderId", "")
        exec_price = root.findtext("Price", "0")
        exec_qty = root.findtext("Quantity", "0")

        if order_id in self._orders:
            order = self._orders[order_id]

            fill = Fill(
                fill_id=root.findtext("TradeId", ""),
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=Decimal(exec_qty),
                price=Decimal(exec_price),
                commission=Decimal(root.findtext("Commission", "0")),
                timestamp=datetime.now(),
            )
            order.add_fill(fill)

    def _handle_account_update(self, root: ET.Element) -> None:
        """Handle account information update."""
        cash = root.findtext("Cash", "0")
        portfolio_value = root.findtext("PortfolioValue", "0")

        self._balances["PLN"] = Balance(
            currency="PLN",
            total=Decimal(portfolio_value),
            available=Decimal(cash),
            locked=Decimal(portfolio_value) - Decimal(cash),
        )

    # ==================== Authentication ====================

    async def _login(self) -> None:
        """Login to bossaAPI."""
        login_msg = f"""<?xml version="1.0" encoding="utf-8"?>
<Login>
    <Account>{self.config.account_id}</Account>
    <Version>10.0</Version>
</Login>"""
        await self._send_message(login_msg)

    async def _load_account_info(self) -> None:
        """Request account information."""
        msg = """<?xml version="1.0" encoding="utf-8"?>
<GetAccountInfo/>"""
        await self._send_message(msg)

    # ==================== Instruments ====================

    async def get_instruments(self) -> list[Instrument]:
        """Get available instruments."""
        if not self._instruments:
            # Request instrument list
            msg = """<?xml version="1.0" encoding="utf-8"?>
<GetInstruments/>"""
            await self._send_message(msg)

            # Wait for response (simplified - would need proper async handling)
            await asyncio.sleep(1.0)

        return list(self._instruments.values())

    async def get_instrument(self, symbol: str) -> Optional[Instrument]:
        """Get instrument details."""
        if symbol not in self._instruments:
            # Create basic instrument
            instrument = Instrument(
                symbol=symbol,
                base_currency=symbol,
                quote_currency="PLN",
                asset_class=AssetClass.EQUITY,
                exchange="GPW",
                lot_size=Decimal("1"),
                tick_size=Decimal("0.01"),
            )
            self._instruments[symbol] = instrument

        return self._instruments.get(symbol)

    # ==================== Account ====================

    async def get_balances(self) -> list[Balance]:
        """Get account balances."""
        msg = """<?xml version="1.0" encoding="utf-8"?>
<GetAccountInfo/>"""
        await self._send_message(msg)

        await asyncio.sleep(0.5)  # Wait for response

        return list(self._balances.values())

    async def get_balance(self, currency: str) -> Optional[Balance]:
        """Get balance for currency."""
        await self.get_balances()
        return self._balances.get(currency)

    # ==================== Orders ====================

    async def submit_order(self, order: Order) -> Order:
        """Submit order to GPW via bossaAPI."""
        order.exchange_order_id = str(self._next_order_id)
        self._next_order_id += 1

        # Build order XML
        order_type = self.ORDER_TYPE_MAP.get(order.order_type, "LIMIT")
        side = self.ORDER_SIDE_MAP.get(order.side, "K")
        tif = self.TIF_MAP.get(order.time_in_force, "D")

        price_element = ""
        if order.limit_price:
            price_element = f"<Price>{order.limit_price}</Price>"
        if order.stop_price:
            price_element += f"<StopPrice>{order.stop_price}</StopPrice>"

        msg = f"""<?xml version="1.0" encoding="utf-8"?>
<NewOrder>
    <Symbol>{order.symbol}</Symbol>
    <Side>{side}</Side>
    <Quantity>{int(order.quantity)}</Quantity>
    <OrderType>{order_type}</OrderType>
    <TimeInForce>{tif}</TimeInForce>
    {price_element}
    <ClientOrderId>{order.order_id}</ClientOrderId>
</NewOrder>"""

        try:
            await self._send_message(msg)

            order.submitted_at = datetime.now()
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

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        order = self._orders.get(order_id)
        if not order:
            return False

        msg = f"""<?xml version="1.0" encoding="utf-8"?>
<CancelOrder>
    <OrderId>{order.exchange_order_id}</OrderId>
</CancelOrder>"""

        try:
            await self._send_message(msg)

            order.status = OrderStatus.CANCELLED
            order.cancelled_at = datetime.now()

            self.events.emit(OrderEvent(
                event_type=EventType.ORDER_CANCELLED,
                order=order,
                source=self.name,
            ))

            return True

        except Exception as e:
            raise ExchangeError(f"Failed to cancel order: {e}")

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders."""
        cancelled = 0
        for order in list(self._orders.values()):
            if order.is_active:
                if symbol is None or order.symbol == symbol:
                    if await self.cancel_order(order.order_id):
                        cancelled += 1
        return cancelled

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Get all open orders."""
        return [
            o for o in self._orders.values()
            if o.is_active and (symbol is None or o.symbol == symbol)
        ]

    # ==================== Positions ====================

    async def get_positions(self) -> list[Position]:
        """Get all positions."""
        msg = """<?xml version="1.0" encoding="utf-8"?>
<GetPositions/>"""
        await self._send_message(msg)

        await asyncio.sleep(0.5)

        return list(self._positions.values())

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        await self.get_positions()
        return self._positions.get(symbol)

    # ==================== Market Data ====================

    async def get_ticker(self, symbol: str) -> Optional[Tick]:
        """Get current quote for symbol."""
        msg = f"""<?xml version="1.0" encoding="utf-8"?>
<GetQuote>
    <Symbol>{symbol}</Symbol>
</GetQuote>"""
        await self._send_message(msg)

        # Would need proper async response handling
        await asyncio.sleep(0.5)

        # Return mock for now
        return Tick(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=Decimal("0"),
            ask=Decimal("0"),
        )

    async def get_order_book(self, symbol: str, depth: int = 10) -> Optional[OrderBook]:
        """Get order book for symbol."""
        msg = f"""<?xml version="1.0" encoding="utf-8"?>
<GetOrderBook>
    <Symbol>{symbol}</Symbol>
    <Depth>{depth}</Depth>
</GetOrderBook>"""
        await self._send_message(msg)

        await asyncio.sleep(0.5)

        return None  # Would parse response

    async def get_historical_bars(
        self,
        symbol: str,
        resolution: DataResolution,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[OHLCV]:
        """Get historical OHLCV data."""
        # bossaAPI has limited historical data
        # For comprehensive history, use alternative data sources
        return []

    # ==================== Streaming ====================

    async def subscribe_ticks(
        self,
        symbols: list[str],
        callback: Callable[[Tick], None]
    ) -> None:
        """Subscribe to real-time quotes."""
        for symbol in symbols:
            self._tick_callbacks[symbol].append(callback)

            # Send subscription request
            msg = f"""<?xml version="1.0" encoding="utf-8"?>
<Subscribe>
    <Symbol>{symbol}</Symbol>
    <Type>QUOTE</Type>
</Subscribe>"""
            await self._send_message(msg)

    async def subscribe_bars(
        self,
        symbols: list[str],
        resolution: DataResolution,
        callback: Callable[[OHLCV], None]
    ) -> None:
        """Subscribe to bar data (aggregated from ticks)."""
        pass  # Would implement tick aggregation

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all data."""
        msg = """<?xml version="1.0" encoding="utf-8"?>
<UnsubscribeAll/>"""
        await self._send_message(msg)
        self._tick_callbacks.clear()
