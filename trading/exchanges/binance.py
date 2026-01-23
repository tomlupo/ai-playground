"""Binance exchange implementation for cryptocurrency trading."""

import asyncio
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, Callable
from urllib.parse import urlencode

import httpx

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
)
from trading.exchanges.base import (
    Exchange,
    ExchangeConfig,
    ExchangeError,
    OrderRejectedError,
    InsufficientFundsError,
    RateLimitError,
)


@dataclass
class BinanceConfig(ExchangeConfig):
    """Binance-specific configuration."""
    # API endpoints
    base_url: str = "https://api.binance.com"
    testnet_url: str = "https://testnet.binance.vision"
    futures_url: str = "https://fapi.binance.com"
    futures_testnet_url: str = "https://testnet.binancefuture.com"

    # WebSocket endpoints
    ws_url: str = "wss://stream.binance.com:9443/ws"
    ws_testnet_url: str = "wss://testnet.binance.vision/ws"

    # Trading mode
    use_futures: bool = False
    hedge_mode: bool = False  # For futures: separate long/short positions

    # Default parameters
    recv_window: int = 5000


class BinanceExchange(Exchange):
    """
    Binance exchange adapter for cryptocurrency trading.

    Supports both Spot and Futures markets.

    Usage:
        config = BinanceConfig(
            api_key="your_api_key",
            api_secret="your_api_secret",
            testnet=True,  # Use testnet for paper trading
        )
        exchange = BinanceExchange(config)
        await exchange.connect()

        # Place a market order
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.001"),
            order_type=OrderType.MARKET,
        )
        result = await exchange.submit_order(order)
    """

    # Mapping internal enums to Binance API values
    ORDER_TYPE_MAP = {
        OrderType.MARKET: "MARKET",
        OrderType.LIMIT: "LIMIT",
        OrderType.STOP: "STOP_MARKET",
        OrderType.STOP_LIMIT: "STOP",
        OrderType.TAKE_PROFIT: "TAKE_PROFIT_MARKET",
        OrderType.TAKE_PROFIT_LIMIT: "TAKE_PROFIT",
        OrderType.TRAILING_STOP: "TRAILING_STOP_MARKET",
    }

    ORDER_SIDE_MAP = {
        OrderSide.BUY: "BUY",
        OrderSide.SELL: "SELL",
    }

    TIME_IN_FORCE_MAP = {
        TimeInForce.GTC: "GTC",
        TimeInForce.IOC: "IOC",
        TimeInForce.FOK: "FOK",
    }

    RESOLUTION_MAP = {
        DataResolution.MINUTE_1: "1m",
        DataResolution.MINUTE_5: "5m",
        DataResolution.MINUTE_15: "15m",
        DataResolution.MINUTE_30: "30m",
        DataResolution.HOUR_1: "1h",
        DataResolution.HOUR_4: "4h",
        DataResolution.DAY_1: "1d",
        DataResolution.WEEK_1: "1w",
        DataResolution.MONTH_1: "1M",
    }

    def __init__(self, config: BinanceConfig):
        super().__init__(config)
        self.config: BinanceConfig = config
        self._client: Optional[httpx.AsyncClient] = None
        self._ws_client = None
        self._ws_subscriptions: dict[str, asyncio.Task] = {}
        self._tick_callbacks: dict[str, list[Callable]] = {}
        self._bar_callbacks: dict[str, list[Callable]] = {}

    @property
    def exchange_type(self) -> ExchangeType:
        return ExchangeType.BINANCE_FUTURES if self.config.use_futures else ExchangeType.BINANCE

    @property
    def name(self) -> str:
        return "Binance Futures" if self.config.use_futures else "Binance Spot"

    @property
    def base_url(self) -> str:
        if self.config.use_futures:
            return self.config.futures_testnet_url if self.config.testnet else self.config.futures_url
        return self.config.testnet_url if self.config.testnet else self.config.base_url

    # ==================== Connection ====================

    async def connect(self) -> bool:
        """Connect to Binance API."""
        try:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.config.timeout,
            )

            # Test connection
            response = await self._client.get("/api/v3/ping")
            if response.status_code == 200:
                self._connected = True

                # Load exchange info
                await self._load_exchange_info()

                self.events.emit(OrderEvent(
                    event_type=EventType.CONNECTED,
                    source=self.name,
                ))
                return True

            return False

        except Exception as e:
            raise ExchangeError(f"Failed to connect to Binance: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Binance."""
        await self.unsubscribe_all()

        if self._client:
            await self._client.aclose()
            self._client = None

        self._connected = False
        self.events.emit(OrderEvent(
            event_type=EventType.DISCONNECTED,
            source=self.name,
        ))

    async def _load_exchange_info(self) -> None:
        """Load exchange trading rules and instruments."""
        endpoint = "/fapi/v1/exchangeInfo" if self.config.use_futures else "/api/v3/exchangeInfo"
        response = await self._client.get(endpoint)
        data = response.json()

        for symbol_info in data.get("symbols", []):
            instrument = self._parse_instrument(symbol_info)
            self._instruments[instrument.symbol] = instrument

    def _parse_instrument(self, data: dict) -> Instrument:
        """Parse Binance symbol info to Instrument."""
        filters = {f["filterType"]: f for f in data.get("filters", [])}

        lot_filter = filters.get("LOT_SIZE", {})
        price_filter = filters.get("PRICE_FILTER", {})
        notional_filter = filters.get("MIN_NOTIONAL", filters.get("NOTIONAL", {}))

        return Instrument(
            symbol=data["symbol"],
            base_currency=data["baseAsset"],
            quote_currency=data["quoteAsset"],
            asset_class=AssetClass.CRYPTO,
            exchange="binance",
            lot_size=Decimal(lot_filter.get("stepSize", "0.00000001")),
            tick_size=Decimal(price_filter.get("tickSize", "0.01")),
            min_notional=Decimal(notional_filter.get("minNotional", "0")),
            max_quantity=Decimal(lot_filter.get("maxQty", "9999999")),
        )

    # ==================== Authentication ====================

    def _sign_request(self, params: dict) -> dict:
        """Add signature to request parameters."""
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self.config.recv_window

        query_string = urlencode(params)
        signature = hmac.new(
            self.config.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()

        params["signature"] = signature
        return params

    def _get_headers(self) -> dict:
        """Get request headers with API key."""
        return {"X-MBX-APIKEY": self.config.api_key}

    # ==================== Account Information ====================

    async def get_balances(self) -> list[Balance]:
        """Fetch all account balances."""
        if self.config.use_futures:
            response = await self._client.get(
                "/fapi/v2/balance",
                params=self._sign_request({}),
                headers=self._get_headers(),
            )
            data = response.json()
            return [
                Balance(
                    currency=item["asset"],
                    total=Decimal(item["balance"]),
                    available=Decimal(item["availableBalance"]),
                    locked=Decimal(item["balance"]) - Decimal(item["availableBalance"]),
                )
                for item in data
            ]
        else:
            response = await self._client.get(
                "/api/v3/account",
                params=self._sign_request({}),
                headers=self._get_headers(),
            )
            data = response.json()
            return [
                Balance(
                    currency=item["asset"],
                    total=Decimal(item["free"]) + Decimal(item["locked"]),
                    available=Decimal(item["free"]),
                    locked=Decimal(item["locked"]),
                )
                for item in data.get("balances", [])
                if Decimal(item["free"]) > 0 or Decimal(item["locked"]) > 0
            ]

    async def get_balance(self, currency: str) -> Optional[Balance]:
        """Fetch balance for specific currency."""
        balances = await self.get_balances()
        for balance in balances:
            if balance.currency == currency:
                return balance
        return None

    # ==================== Instruments ====================

    async def get_instruments(self) -> list[Instrument]:
        """Get all available trading instruments."""
        if not self._instruments:
            await self._load_exchange_info()
        return list(self._instruments.values())

    async def get_instrument(self, symbol: str) -> Optional[Instrument]:
        """Get instrument details for symbol."""
        if not self._instruments:
            await self._load_exchange_info()
        return self._instruments.get(symbol)

    # ==================== Order Management ====================

    async def submit_order(self, order: Order) -> Order:
        """Submit order to Binance."""
        endpoint = "/fapi/v1/order" if self.config.use_futures else "/api/v3/order"

        params = {
            "symbol": order.symbol,
            "side": self.ORDER_SIDE_MAP[order.side],
            "type": self.ORDER_TYPE_MAP[order.order_type],
            "quantity": str(order.quantity),
        }

        # Add optional parameters based on order type
        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT):
            params["price"] = str(order.limit_price)
            params["timeInForce"] = self.TIME_IN_FORCE_MAP.get(order.time_in_force, "GTC")

        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT):
            params["stopPrice"] = str(order.stop_price)

        if order.client_order_id:
            params["newClientOrderId"] = order.client_order_id

        if self.config.use_futures:
            if order.reduce_only:
                params["reduceOnly"] = "true"
            if order.post_only:
                params["timeInForce"] = "GTX"

        try:
            response = await self._client.post(
                endpoint,
                params=self._sign_request(params),
                headers=self._get_headers(),
            )

            if response.status_code == 429:
                raise RateLimitError()

            data = response.json()

            if "code" in data:
                # Error response
                error_code = data["code"]
                error_msg = data.get("msg", "Unknown error")

                if error_code == -2010:  # Insufficient balance
                    raise InsufficientFundsError(
                        required=order.quantity * (order.limit_price or Decimal("0")),
                        available=Decimal("0"),
                        currency=order.symbol.replace("USDT", "").replace("BTC", ""),
                    )

                raise OrderRejectedError(order, error_msg)

            # Update order with exchange response
            order.exchange_order_id = str(data["orderId"])
            order.client_order_id = data.get("clientOrderId")
            order.submitted_at = datetime.utcnow()
            order.status = self._parse_order_status(data["status"])

            if data.get("executedQty"):
                order.filled_quantity = Decimal(data["executedQty"])

            if data.get("avgPrice") and Decimal(data["avgPrice"]) > 0:
                order.average_fill_price = Decimal(data["avgPrice"])

            # Emit event
            self.events.emit(OrderEvent(
                event_type=EventType.ORDER_SUBMITTED,
                order=order,
                source=self.name,
            ))

            return order

        except (OrderRejectedError, InsufficientFundsError, RateLimitError):
            raise
        except Exception as e:
            raise ExchangeError(f"Failed to submit order: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        endpoint = "/fapi/v1/order" if self.config.use_futures else "/api/v3/order"

        # Try to find the order to get the symbol
        order = await self.get_order(order_id)
        if not order:
            return False

        params = {"symbol": order.symbol, "orderId": int(order.exchange_order_id)}

        try:
            response = await self._client.delete(
                endpoint,
                params=self._sign_request(params),
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                order.status = OrderStatus.CANCELLED
                order.cancelled_at = datetime.utcnow()

                self.events.emit(OrderEvent(
                    event_type=EventType.ORDER_CANCELLED,
                    order=order,
                    source=self.name,
                ))
                return True

            return False

        except Exception as e:
            raise ExchangeError(f"Failed to cancel order: {e}")

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders."""
        if symbol:
            endpoint = "/fapi/v1/allOpenOrders" if self.config.use_futures else "/api/v3/openOrders"
            params = {"symbol": symbol}
            response = await self._client.delete(
                endpoint,
                params=self._sign_request(params),
                headers=self._get_headers(),
            )
            return 1 if response.status_code == 200 else 0
        else:
            # Cancel for all symbols
            open_orders = await self.get_open_orders()
            cancelled = 0
            for order in open_orders:
                if await self.cancel_order(order.order_id):
                    cancelled += 1
            return cancelled

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        # This is a simplified implementation
        # In production, we'd need to track orders locally or query by symbol
        open_orders = await self.get_open_orders()
        for order in open_orders:
            if order.order_id == order_id or order.exchange_order_id == order_id:
                return order
        return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Get all open orders."""
        endpoint = "/fapi/v1/openOrders" if self.config.use_futures else "/api/v3/openOrders"
        params = {}
        if symbol:
            params["symbol"] = symbol

        response = await self._client.get(
            endpoint,
            params=self._sign_request(params),
            headers=self._get_headers(),
        )

        data = response.json()
        if isinstance(data, dict) and "code" in data:
            return []

        orders = []
        for item in data:
            order = self._parse_order(item)
            orders.append(order)

        return orders

    def _parse_order(self, data: dict) -> Order:
        """Parse Binance order response to Order object."""
        side_map = {"BUY": OrderSide.BUY, "SELL": OrderSide.SELL}
        type_map = {v: k for k, v in self.ORDER_TYPE_MAP.items()}

        order = Order(
            symbol=data["symbol"],
            side=side_map[data["side"]],
            quantity=Decimal(data["origQty"]),
            order_type=type_map.get(data["type"], OrderType.MARKET),
            exchange_order_id=str(data["orderId"]),
            client_order_id=data.get("clientOrderId"),
            status=self._parse_order_status(data["status"]),
            filled_quantity=Decimal(data.get("executedQty", "0")),
        )

        if data.get("price") and Decimal(data["price"]) > 0:
            order.limit_price = Decimal(data["price"])
        if data.get("stopPrice") and Decimal(data["stopPrice"]) > 0:
            order.stop_price = Decimal(data["stopPrice"])
        if data.get("avgPrice") and Decimal(data["avgPrice"]) > 0:
            order.average_fill_price = Decimal(data["avgPrice"])

        return order

    def _parse_order_status(self, status: str) -> OrderStatus:
        """Parse Binance order status string."""
        status_map = {
            "NEW": OrderStatus.ACCEPTED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return status_map.get(status, OrderStatus.PENDING)

    # ==================== Positions ====================

    async def get_positions(self) -> list[Position]:
        """Get all open positions (futures only)."""
        if not self.config.use_futures:
            # Spot doesn't have positions in the traditional sense
            # Return balances as pseudo-positions
            return []

        response = await self._client.get(
            "/fapi/v2/positionRisk",
            params=self._sign_request({}),
            headers=self._get_headers(),
        )

        data = response.json()
        positions = []

        for item in data:
            qty = Decimal(item["positionAmt"])
            if qty != 0:
                position = Position(
                    symbol=item["symbol"],
                    side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
                    quantity=abs(qty),
                    average_entry_price=Decimal(item["entryPrice"]),
                    realized_pnl=Decimal(item.get("realizedPnl", "0")),
                    leverage=Decimal(item.get("leverage", "1")),
                    liquidation_price=Decimal(item["liquidationPrice"]) if item.get("liquidationPrice") else None,
                )
                positions.append(position)

        return positions

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    # ==================== Market Data ====================

    async def get_ticker(self, symbol: str) -> Optional[Tick]:
        """Get current ticker for symbol."""
        endpoint = "/fapi/v1/ticker/bookTicker" if self.config.use_futures else "/api/v3/ticker/bookTicker"

        response = await self._client.get(endpoint, params={"symbol": symbol})
        data = response.json()

        if "code" in data:
            return None

        return Tick(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bid=Decimal(data["bidPrice"]),
            ask=Decimal(data["askPrice"]),
            bid_size=Decimal(data["bidQty"]),
            ask_size=Decimal(data["askQty"]),
        )

    async def get_order_book(self, symbol: str, depth: int = 10) -> Optional[OrderBook]:
        """Get order book for symbol."""
        endpoint = "/fapi/v1/depth" if self.config.use_futures else "/api/v3/depth"

        response = await self._client.get(
            endpoint,
            params={"symbol": symbol, "limit": depth}
        )
        data = response.json()

        if "code" in data:
            return None

        return OrderBook(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=[
                OrderBookLevel(
                    price=Decimal(level[0]),
                    quantity=Decimal(level[1]),
                )
                for level in data["bids"]
            ],
            asks=[
                OrderBookLevel(
                    price=Decimal(level[0]),
                    quantity=Decimal(level[1]),
                )
                for level in data["asks"]
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
        """Get historical OHLCV data."""
        endpoint = "/fapi/v1/klines" if self.config.use_futures else "/api/v3/klines"

        interval = self.RESOLUTION_MAP.get(resolution, "1h")
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(start.timestamp() * 1000),
            "limit": min(limit, 1000),
        }
        if end:
            params["endTime"] = int(end.timestamp() * 1000)

        response = await self._client.get(endpoint, params=params)
        data = response.json()

        if isinstance(data, dict) and "code" in data:
            return []

        bars = []
        for item in data:
            bar = OHLCV(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(item[0] / 1000),
                open=Decimal(item[1]),
                high=Decimal(item[2]),
                low=Decimal(item[3]),
                close=Decimal(item[4]),
                volume=Decimal(item[5]),
                trades=int(item[8]),
            )
            bars.append(bar)

        return bars

    # ==================== Streaming ====================

    async def subscribe_ticks(
        self,
        symbols: list[str],
        callback: Callable[[Tick], None]
    ) -> None:
        """Subscribe to tick stream via WebSocket."""
        # Store callbacks
        for symbol in symbols:
            if symbol not in self._tick_callbacks:
                self._tick_callbacks[symbol] = []
            self._tick_callbacks[symbol].append(callback)

        # Start WebSocket connection if not already running
        # This is a simplified implementation - production would use proper WS
        for symbol in symbols:
            if symbol not in self._ws_subscriptions:
                task = asyncio.create_task(self._poll_ticker(symbol))
                self._ws_subscriptions[symbol] = task

    async def _poll_ticker(self, symbol: str) -> None:
        """Poll ticker data (fallback for WebSocket)."""
        while symbol in self._tick_callbacks:
            try:
                tick = await self.get_ticker(symbol)
                if tick:
                    for callback in self._tick_callbacks.get(symbol, []):
                        callback(tick)

                    self.events.emit(TickEvent(tick=tick, source=self.name))

                await asyncio.sleep(1.0)  # Poll interval
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
        """Subscribe to bar stream."""
        for symbol in symbols:
            key = f"{symbol}_{resolution.name}"
            if key not in self._bar_callbacks:
                self._bar_callbacks[key] = []
            self._bar_callbacks[key].append(callback)

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all streams."""
        for task in self._ws_subscriptions.values():
            task.cancel()
        self._ws_subscriptions.clear()
        self._tick_callbacks.clear()
        self._bar_callbacks.clear()
