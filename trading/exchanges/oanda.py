"""OANDA exchange implementation for forex and CFD trading."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Callable, Any
from collections import defaultdict
import json

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
    PositionEvent,
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
class OandaConfig(ExchangeConfig):
    """OANDA specific configuration."""
    # API endpoints
    api_url: str = "https://api-fxpractice.oanda.com"
    stream_url: str = "https://stream-fxpractice.oanda.com"

    # Live endpoints (use these for real trading)
    live_api_url: str = "https://api-fxtrade.oanda.com"
    live_stream_url: str = "https://stream-fxtrade.oanda.com"

    # Account
    account_id: str = ""

    # Practice mode (paper trading)
    practice: bool = True

    # Date format
    datetime_format: str = "RFC3339"


class OandaExchange(Exchange):
    """
    OANDA exchange adapter for forex and CFD trading.

    Supports forex pairs, indices, commodities, and bonds via CFDs.

    Usage:
        config = OandaConfig(
            api_key="your-api-token",
            account_id="your-account-id",
            practice=True,  # Use practice account
        )
        exchange = OandaExchange(config)
        await exchange.connect()

        # Place a market order on EUR/USD
        order = Order(
            symbol="EUR_USD",
            side=OrderSide.BUY,
            quantity=Decimal("10000"),  # 0.1 lot
            order_type=OrderType.MARKET,
        )
        result = await exchange.submit_order(order)
    """

    # OANDA instrument categories
    FOREX_PAIRS = [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD",
        "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY", "EUR_CHF", "AUD_JPY",
    ]

    # Resolution mapping to OANDA granularity
    RESOLUTION_MAP = {
        DataResolution.SECOND_5: "S5",
        DataResolution.SECOND_15: "S15",
        DataResolution.SECOND_30: "S30",
        DataResolution.MINUTE_1: "M1",
        DataResolution.MINUTE_5: "M5",
        DataResolution.MINUTE_15: "M15",
        DataResolution.MINUTE_30: "M30",
        DataResolution.HOUR_1: "H1",
        DataResolution.HOUR_4: "H4",
        DataResolution.DAY_1: "D",
        DataResolution.WEEK_1: "W",
        DataResolution.MONTH_1: "M",
    }

    def __init__(self, config: OandaConfig):
        super().__init__(config)
        self.config: OandaConfig = config
        self._client: Optional[httpx.AsyncClient] = None
        self._stream_client: Optional[httpx.AsyncClient] = None

        # Local state
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}

        # Subscriptions
        self._tick_subscriptions: dict[str, asyncio.Task] = {}
        self._tick_callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._bar_callbacks: dict[str, list[Callable]] = defaultdict(list)

    @property
    def exchange_type(self) -> ExchangeType:
        return ExchangeType.OANDA

    @property
    def name(self) -> str:
        return "OANDA"

    @property
    def base_url(self) -> str:
        return self.config.api_url if self.config.practice else self.config.live_api_url

    @property
    def stream_url(self) -> str:
        return self.config.stream_url if self.config.practice else self.config.live_stream_url

    def _get_headers(self) -> dict:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": self.config.datetime_format,
        }

    # ==================== Connection ====================

    async def connect(self) -> bool:
        """Connect to OANDA API."""
        try:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._get_headers(),
                timeout=self.config.timeout,
            )

            self._stream_client = httpx.AsyncClient(
                base_url=self.stream_url,
                headers=self._get_headers(),
                timeout=None,  # Streaming connection
            )

            # Test connection by fetching account
            response = await self._client.get(
                f"/v3/accounts/{self.config.account_id}"
            )

            if response.status_code == 200:
                self._connected = True

                # Load instruments
                await self._load_instruments()

                self.events.emit(OrderEvent(
                    event_type=EventType.CONNECTED,
                    source=self.name,
                ))
                return True

            elif response.status_code == 401:
                raise ExchangeError("Invalid API key")
            else:
                data = response.json()
                raise ExchangeError(f"Connection failed: {data.get('errorMessage', 'Unknown error')}")

        except ExchangeError:
            raise
        except Exception as e:
            raise ExchangeError(f"Failed to connect to OANDA: {e}")

    async def disconnect(self) -> None:
        """Disconnect from OANDA."""
        await self.unsubscribe_all()

        if self._client:
            await self._client.aclose()
            self._client = None

        if self._stream_client:
            await self._stream_client.aclose()
            self._stream_client = None

        self._connected = False

        self.events.emit(OrderEvent(
            event_type=EventType.DISCONNECTED,
            source=self.name,
        ))

    async def _load_instruments(self) -> None:
        """Load tradeable instruments."""
        response = await self._client.get(
            f"/v3/accounts/{self.config.account_id}/instruments"
        )

        if response.status_code != 200:
            return

        data = response.json()

        for inst in data.get("instruments", []):
            instrument = Instrument(
                symbol=inst["name"],
                base_currency=inst["name"].split("_")[0],
                quote_currency=inst["name"].split("_")[1] if "_" in inst["name"] else "USD",
                asset_class=self._get_asset_class(inst["type"]),
                exchange="oanda",
                lot_size=Decimal(str(inst.get("minimumTradeSize", "1"))),
                tick_size=Decimal("0.00001") if "JPY" not in inst["name"] else Decimal("0.001"),
                max_quantity=Decimal(str(inst.get("maximumOrderUnits", "10000000"))),
                margin_required=Decimal(str(inst.get("marginRate", "0.02"))),
            )
            self._instruments[instrument.symbol] = instrument

    def _get_asset_class(self, instrument_type: str) -> AssetClass:
        """Map OANDA instrument type to AssetClass."""
        type_map = {
            "CURRENCY": AssetClass.FOREX,
            "CFD": AssetClass.FUTURES,  # Using FUTURES for CFDs
            "METAL": AssetClass.FUTURES,
        }
        return type_map.get(instrument_type, AssetClass.FOREX)

    # ==================== Account ====================

    async def get_balances(self) -> list[Balance]:
        """Get account balance."""
        response = await self._client.get(
            f"/v3/accounts/{self.config.account_id}/summary"
        )

        if response.status_code != 200:
            raise ExchangeError("Failed to fetch account summary")

        data = response.json()
        account = data.get("account", {})

        # OANDA accounts are denominated in a single currency
        currency = account.get("currency", "USD")
        balance = Decimal(str(account.get("balance", "0")))
        margin_available = Decimal(str(account.get("marginAvailable", "0")))
        margin_used = Decimal(str(account.get("marginUsed", "0")))

        return [
            Balance(
                currency=currency,
                total=balance,
                available=margin_available,
                locked=margin_used,
            )
        ]

    async def get_balance(self, currency: str) -> Optional[Balance]:
        """Get balance for currency."""
        balances = await self.get_balances()
        for b in balances:
            if b.currency == currency:
                return b
        return None

    # ==================== Instruments ====================

    async def get_instruments(self) -> list[Instrument]:
        """Get all tradeable instruments."""
        if not self._instruments:
            await self._load_instruments()
        return list(self._instruments.values())

    async def get_instrument(self, symbol: str) -> Optional[Instrument]:
        """Get instrument details."""
        if not self._instruments:
            await self._load_instruments()
        return self._instruments.get(symbol)

    # ==================== Orders ====================

    async def submit_order(self, order: Order) -> Order:
        """Submit order to OANDA."""
        # Build order request
        units = int(order.quantity)
        if order.side == OrderSide.SELL:
            units = -units

        order_request: dict[str, Any] = {
            "order": {
                "instrument": order.symbol,
                "units": str(units),
                "timeInForce": self._map_tif(order.time_in_force),
                "positionFill": "DEFAULT",
            }
        }

        # Set order type
        if order.order_type == OrderType.MARKET:
            order_request["order"]["type"] = "MARKET"
        elif order.order_type == OrderType.LIMIT:
            order_request["order"]["type"] = "LIMIT"
            order_request["order"]["price"] = str(order.limit_price)
        elif order.order_type == OrderType.STOP:
            order_request["order"]["type"] = "STOP"
            order_request["order"]["price"] = str(order.stop_price)
        elif order.order_type == OrderType.STOP_LIMIT:
            order_request["order"]["type"] = "STOP"
            order_request["order"]["price"] = str(order.stop_price)
            order_request["order"]["priceBound"] = str(order.limit_price)

        if order.client_order_id:
            order_request["order"]["clientExtensions"] = {
                "id": order.client_order_id
            }

        try:
            response = await self._client.post(
                f"/v3/accounts/{self.config.account_id}/orders",
                json=order_request,
            )

            data = response.json()

            if response.status_code == 201:
                # Order accepted
                if "orderFillTransaction" in data:
                    # Market order filled immediately
                    fill_txn = data["orderFillTransaction"]
                    order.exchange_order_id = fill_txn["orderID"]
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = abs(Decimal(str(fill_txn["units"])))
                    order.average_fill_price = Decimal(str(fill_txn["price"]))
                    order.filled_at = datetime.fromisoformat(
                        fill_txn["time"].replace("Z", "+00:00")
                    )

                    # Record fill
                    fill = Fill(
                        fill_id=fill_txn["id"],
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.filled_quantity,
                        price=order.average_fill_price,
                        commission=Decimal(str(fill_txn.get("commission", "0"))),
                        timestamp=order.filled_at,
                    )
                    order.add_fill(fill)

                elif "orderCreateTransaction" in data:
                    # Pending order created
                    create_txn = data["orderCreateTransaction"]
                    order.exchange_order_id = create_txn["id"]
                    order.status = OrderStatus.ACCEPTED
                    order.submitted_at = datetime.fromisoformat(
                        create_txn["time"].replace("Z", "+00:00")
                    )

                self._orders[order.order_id] = order
                self._orders[order.exchange_order_id] = order

                self.events.emit(OrderEvent(
                    event_type=EventType.ORDER_SUBMITTED,
                    order=order,
                    source=self.name,
                ))

                return order

            elif response.status_code == 400:
                error_msg = data.get("errorMessage", "Order rejected")
                raise OrderRejectedError(order, error_msg)

            elif response.status_code == 404:
                raise ExchangeError(f"Instrument not found: {order.symbol}")

            else:
                raise ExchangeError(f"Order failed: {data}")

        except (OrderRejectedError, ExchangeError):
            raise
        except Exception as e:
            raise ExchangeError(f"Failed to submit order: {e}")

    def _map_tif(self, tif: TimeInForce) -> str:
        """Map TimeInForce to OANDA value."""
        tif_map = {
            TimeInForce.GTC: "GTC",
            TimeInForce.IOC: "IOC",
            TimeInForce.FOK: "FOK",
            TimeInForce.GTD: "GTD",
        }
        return tif_map.get(tif, "GTC")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        order = self._orders.get(order_id)
        if not order:
            return False

        try:
            response = await self._client.put(
                f"/v3/accounts/{self.config.account_id}/orders/{order.exchange_order_id}/cancel"
            )

            if response.status_code == 200:
                order.status = OrderStatus.CANCELLED
                order.cancelled_at = datetime.now(timezone.utc)

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
        """Cancel all pending orders."""
        open_orders = await self.get_open_orders(symbol)
        cancelled = 0

        for order in open_orders:
            if await self.cancel_order(order.order_id):
                cancelled += 1

        return cancelled

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Get all pending orders."""
        params = {}
        if symbol:
            params["instrument"] = symbol

        response = await self._client.get(
            f"/v3/accounts/{self.config.account_id}/pendingOrders",
            params=params,
        )

        if response.status_code != 200:
            return []

        data = response.json()
        orders = []

        for order_data in data.get("orders", []):
            order = self._parse_order(order_data)
            self._orders[order.order_id] = order
            self._orders[order.exchange_order_id] = order
            orders.append(order)

        return orders

    def _parse_order(self, data: dict) -> Order:
        """Parse OANDA order response."""
        units = int(data.get("units", "0"))
        side = OrderSide.BUY if units > 0 else OrderSide.SELL

        type_map = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP": OrderType.STOP,
        }

        order = Order(
            symbol=data["instrument"],
            side=side,
            quantity=Decimal(str(abs(units))),
            order_type=type_map.get(data["type"], OrderType.MARKET),
            exchange_order_id=data["id"],
            status=self._parse_status(data.get("state", "PENDING")),
        )

        if "price" in data:
            if data["type"] == "STOP":
                order.stop_price = Decimal(data["price"])
            else:
                order.limit_price = Decimal(data["price"])

        return order

    def _parse_status(self, state: str) -> OrderStatus:
        """Parse OANDA order state."""
        status_map = {
            "PENDING": OrderStatus.ACCEPTED,
            "FILLED": OrderStatus.FILLED,
            "TRIGGERED": OrderStatus.ACCEPTED,
            "CANCELLED": OrderStatus.CANCELLED,
        }
        return status_map.get(state, OrderStatus.PENDING)

    # ==================== Positions ====================

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        response = await self._client.get(
            f"/v3/accounts/{self.config.account_id}/openPositions"
        )

        if response.status_code != 200:
            return []

        data = response.json()
        positions = []

        for pos_data in data.get("positions", []):
            # OANDA tracks long and short separately
            long_units = int(pos_data.get("long", {}).get("units", "0"))
            short_units = int(pos_data.get("short", {}).get("units", "0"))

            if long_units > 0:
                position = Position(
                    symbol=pos_data["instrument"],
                    side=PositionSide.LONG,
                    quantity=Decimal(str(long_units)),
                    average_entry_price=Decimal(pos_data["long"]["averagePrice"]),
                    realized_pnl=Decimal(str(pos_data["long"].get("pl", "0"))),
                )
                positions.append(position)
                self._positions[pos_data["instrument"]] = position

            if short_units != 0:
                position = Position(
                    symbol=pos_data["instrument"],
                    side=PositionSide.SHORT,
                    quantity=Decimal(str(abs(short_units))),
                    average_entry_price=Decimal(pos_data["short"]["averagePrice"]),
                    realized_pnl=Decimal(str(pos_data["short"].get("pl", "0"))),
                )
                positions.append(position)
                self._positions[f"{pos_data['instrument']}_short"] = position

        return positions

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        positions = await self.get_positions()
        for p in positions:
            if p.symbol == symbol:
                return p
        return None

    # ==================== Market Data ====================

    async def get_ticker(self, symbol: str) -> Optional[Tick]:
        """Get current price for symbol."""
        response = await self._client.get(
            f"/v3/accounts/{self.config.account_id}/pricing",
            params={"instruments": symbol},
        )

        if response.status_code != 200:
            return None

        data = response.json()
        prices = data.get("prices", [])

        if not prices:
            return None

        price = prices[0]

        # Get best bid/ask
        bids = price.get("bids", [])
        asks = price.get("asks", [])

        return Tick(
            symbol=symbol,
            timestamp=datetime.fromisoformat(
                price["time"].replace("Z", "+00:00")
            ),
            bid=Decimal(bids[0]["price"]) if bids else Decimal("0"),
            ask=Decimal(asks[0]["price"]) if asks else Decimal("0"),
            bid_size=Decimal(str(bids[0].get("liquidity", 0))) if bids else Decimal("0"),
            ask_size=Decimal(str(asks[0].get("liquidity", 0))) if asks else Decimal("0"),
        )

    async def get_order_book(self, symbol: str, depth: int = 10) -> Optional[OrderBook]:
        """Get order book (limited availability on OANDA)."""
        response = await self._client.get(
            f"/v3/instruments/{symbol}/orderBook"
        )

        if response.status_code != 200:
            # Fall back to pricing endpoint for bid/ask
            tick = await self.get_ticker(symbol)
            if tick:
                return OrderBook(
                    symbol=symbol,
                    timestamp=tick.timestamp,
                    bids=[OrderBookLevel(tick.bid, tick.bid_size)],
                    asks=[OrderBookLevel(tick.ask, tick.ask_size)],
                )
            return None

        data = response.json()
        book = data.get("orderBook", {})
        buckets = book.get("buckets", [])

        bids = []
        asks = []

        for bucket in buckets[:depth * 2]:
            price = Decimal(bucket["price"])
            long_pct = Decimal(bucket.get("longCountPercent", "0"))
            short_pct = Decimal(bucket.get("shortCountPercent", "0"))

            if long_pct > 0:
                bids.append(OrderBookLevel(price, long_pct * 1000))
            if short_pct > 0:
                asks.append(OrderBookLevel(price, short_pct * 1000))

        return OrderBook(
            symbol=symbol,
            timestamp=datetime.fromisoformat(
                book.get("time", datetime.now(timezone.utc).isoformat()).replace("Z", "+00:00")
            ),
            bids=sorted(bids, key=lambda x: x.price, reverse=True)[:depth],
            asks=sorted(asks, key=lambda x: x.price)[:depth],
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
        granularity = self.RESOLUTION_MAP.get(resolution, "H1")

        params = {
            "granularity": granularity,
            "from": start.isoformat(),
            "count": min(limit, 5000),
        }

        if end:
            params["to"] = end.isoformat()

        response = await self._client.get(
            f"/v3/instruments/{symbol}/candles",
            params=params,
        )

        if response.status_code != 200:
            return []

        data = response.json()
        bars = []

        for candle in data.get("candles", []):
            if not candle.get("complete", True):
                continue

            mid = candle.get("mid", {})
            bars.append(OHLCV(
                symbol=symbol,
                timestamp=datetime.fromisoformat(
                    candle["time"].replace("Z", "+00:00")
                ),
                open=Decimal(mid.get("o", "0")),
                high=Decimal(mid.get("h", "0")),
                low=Decimal(mid.get("l", "0")),
                close=Decimal(mid.get("c", "0")),
                volume=Decimal(str(candle.get("volume", 0))),
            ))

        return bars

    # ==================== Streaming ====================

    async def subscribe_ticks(
        self,
        symbols: list[str],
        callback: Callable[[Tick], None]
    ) -> None:
        """Subscribe to streaming prices."""
        for symbol in symbols:
            self._tick_callbacks[symbol].append(callback)

        # Start streaming if not already running
        instruments = ",".join(symbols)
        if instruments not in self._tick_subscriptions:
            task = asyncio.create_task(self._stream_prices(symbols))
            self._tick_subscriptions[instruments] = task

    async def _stream_prices(self, symbols: list[str]) -> None:
        """Stream prices from OANDA."""
        instruments = ",".join(symbols)

        while any(s in self._tick_callbacks for s in symbols):
            try:
                async with self._stream_client.stream(
                    "GET",
                    f"/v3/accounts/{self.config.account_id}/pricing/stream",
                    params={"instruments": instruments},
                ) as response:
                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        try:
                            data = json.loads(line)

                            if data.get("type") == "PRICE":
                                tick = self._parse_price_tick(data)
                                if tick:
                                    for cb in self._tick_callbacks.get(tick.symbol, []):
                                        cb(tick)
                                    self.events.emit(TickEvent(tick=tick, source=self.name))

                        except json.JSONDecodeError:
                            continue

            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(5.0)

    def _parse_price_tick(self, data: dict) -> Optional[Tick]:
        """Parse streaming price to Tick."""
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        if not bids or not asks:
            return None

        return Tick(
            symbol=data["instrument"],
            timestamp=datetime.fromisoformat(
                data["time"].replace("Z", "+00:00")
            ),
            bid=Decimal(bids[0]["price"]),
            ask=Decimal(asks[0]["price"]),
            bid_size=Decimal(str(bids[0].get("liquidity", 0))),
            ask_size=Decimal(str(asks[0].get("liquidity", 0))),
        )

    async def subscribe_bars(
        self,
        symbols: list[str],
        resolution: DataResolution,
        callback: Callable[[OHLCV], None]
    ) -> None:
        """Subscribe to bar data (aggregated from ticks)."""
        for symbol in symbols:
            key = f"{symbol}_{resolution.name}"
            self._bar_callbacks[key].append(callback)

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all streams."""
        for task in self._tick_subscriptions.values():
            task.cancel()
        self._tick_subscriptions.clear()
        self._tick_callbacks.clear()
        self._bar_callbacks.clear()
