"""
Live data feed handling.

Provides:
- Real-time tick streaming from exchanges
- Automatic bar aggregation
- Reconnection handling
- Multi-exchange feed management
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, Callable, Any
from collections import defaultdict

from trading.core.models import Tick, OHLCV, OrderBook
from trading.core.enums import DataResolution
from trading.exchanges.base import Exchange
from trading.feeds.base import (
    DataFeed,
    FeedConfig,
    FeedStatus,
    TickAggregator,
    TickBuffer,
    BarBuffer,
)


class LiveFeed(DataFeed):
    """
    Live data feed from an exchange.

    Streams real-time market data and aggregates into bars.

    Usage:
        feed = LiveFeed(exchange, FeedConfig(
            symbols=["BTCUSDT", "ETHUSDT"],
            bar_resolutions=[DataResolution.MINUTE_1, DataResolution.MINUTE_5],
        ))

        # Register callbacks
        feed.on_tick("BTCUSDT", lambda tick: print(f"Tick: {tick.mid_price}"))
        feed.on_bar("BTCUSDT", lambda bar: print(f"Bar: {bar.close}"))

        # Connect and subscribe
        await feed.connect()
        await feed.subscribe(["BTCUSDT"])

        # Or use async iterator
        async for tick in feed.stream_ticks("BTCUSDT"):
            process_tick(tick)
    """

    def __init__(
        self,
        exchange: Exchange,
        config: Optional[FeedConfig] = None,
    ):
        super().__init__(config or FeedConfig())
        self.exchange = exchange

        # Aggregators for each symbol
        self._aggregators: dict[str, TickAggregator] = {}

        # Buffers
        self._tick_buffers: dict[str, TickBuffer] = defaultdict(
            lambda: TickBuffer(self.config.tick_buffer_size)
        )
        self._bar_buffer = BarBuffer(self.config.bar_buffer_size)

        # Latest data cache
        self._latest_ticks: dict[str, Tick] = {}
        self._latest_bars: dict[str, dict[DataResolution, OHLCV]] = defaultdict(dict)

        # Subscription state
        self._subscribed_symbols: set[str] = set()
        self._stream_tasks: dict[str, asyncio.Task] = {}

        # Reconnection
        self._reconnect_task: Optional[asyncio.Task] = None
        self._should_reconnect = True

    async def connect(self) -> bool:
        """Connect to exchange and start data feed."""
        self._set_status(FeedStatus.CONNECTING)

        try:
            if not self.exchange.is_connected:
                connected = await self.exchange.connect()
                if not connected:
                    self._set_status(FeedStatus.ERROR)
                    return False

            self._set_status(FeedStatus.CONNECTED)
            self._should_reconnect = True

            # Subscribe to configured symbols
            if self.config.symbols:
                await self.subscribe(self.config.symbols)

            return True

        except Exception as e:
            print(f"Feed connection error: {e}")
            self._set_status(FeedStatus.ERROR)
            return False

    async def disconnect(self) -> None:
        """Disconnect from data feed."""
        self._should_reconnect = False

        # Cancel all stream tasks
        for task in self._stream_tasks.values():
            task.cancel()
        self._stream_tasks.clear()

        # Cancel reconnect task
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        # Flush aggregators
        for aggregator in self._aggregators.values():
            bars = aggregator.flush()
            for bar in bars:
                self._emit_bar(bar)

        self._set_status(FeedStatus.DISCONNECTED)

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to market data for symbols."""
        for symbol in symbols:
            if symbol in self._subscribed_symbols:
                continue

            # Create aggregator if needed
            if symbol not in self._aggregators:
                self._aggregators[symbol] = TickAggregator(
                    symbol=symbol,
                    resolutions=self.config.bar_resolutions,
                    on_bar=lambda bar: self._handle_bar(bar),
                )

            # Start streaming
            if self.config.subscribe_ticks:
                task = asyncio.create_task(self._stream_symbol(symbol))
                self._stream_tasks[symbol] = task

            self._subscribed_symbols.add(symbol)

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols."""
        for symbol in symbols:
            if symbol not in self._subscribed_symbols:
                continue

            # Cancel stream task
            if symbol in self._stream_tasks:
                self._stream_tasks[symbol].cancel()
                del self._stream_tasks[symbol]

            # Flush aggregator
            if symbol in self._aggregators:
                bars = self._aggregators[symbol].flush()
                for bar in bars:
                    self._emit_bar(bar)

            self._subscribed_symbols.discard(symbol)

    async def _stream_symbol(self, symbol: str) -> None:
        """Stream ticks for a symbol."""
        while self._should_reconnect and symbol in self._subscribed_symbols:
            try:
                # Use exchange's tick subscription
                await self.exchange.subscribe_ticks(
                    [symbol],
                    lambda tick: self._handle_tick(tick),
                )

                # Keep task alive while subscribed
                while symbol in self._subscribed_symbols:
                    await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Stream error for {symbol}: {e}")
                if self._should_reconnect:
                    await self._handle_reconnect()

    def _handle_tick(self, tick: Tick) -> None:
        """Process incoming tick."""
        # Update cache
        self._latest_ticks[tick.symbol] = tick

        # Buffer tick
        if self.config.store_ticks:
            self._tick_buffers[tick.symbol].add(tick)

        # Aggregate into bars
        if tick.symbol in self._aggregators:
            completed_bars = self._aggregators[tick.symbol].process_tick(tick)
            for bar in completed_bars:
                self._handle_bar(bar)

        # Emit to callbacks
        self._emit_tick(tick)

    def _handle_bar(self, bar: OHLCV) -> None:
        """Process completed bar."""
        # Determine resolution from bar duration
        # (In practice, the aggregator would tag the bar with resolution)
        for resolution in self.config.bar_resolutions:
            self._latest_bars[bar.symbol][resolution] = bar

            if self.config.store_bars:
                self._bar_buffer.add(bar, resolution)

        self._emit_bar(bar)

    async def _handle_reconnect(self) -> None:
        """Handle reconnection with backoff."""
        if not self._should_reconnect:
            return

        self._set_status(FeedStatus.RECONNECTING)

        for attempt in range(self.config.reconnect_attempts):
            delay = self.config.reconnect_delay_seconds * (2 ** attempt)
            print(f"Reconnecting in {delay}s (attempt {attempt + 1})...")

            await asyncio.sleep(delay)

            try:
                if await self.connect():
                    # Resubscribe to symbols
                    symbols = list(self._subscribed_symbols)
                    self._subscribed_symbols.clear()
                    await self.subscribe(symbols)
                    return

            except Exception as e:
                print(f"Reconnect attempt failed: {e}")

        self._set_status(FeedStatus.ERROR)
        print("Max reconnect attempts reached")

    # ==================== Data Access ====================

    async def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        """Get most recent tick for symbol."""
        # Check cache first
        if symbol in self._latest_ticks:
            return self._latest_ticks[symbol]

        # Fetch from exchange
        return await self.exchange.get_ticker(symbol)

    async def get_latest_bar(
        self,
        symbol: str,
        resolution: DataResolution
    ) -> Optional[OHLCV]:
        """Get most recent bar for symbol and resolution."""
        # Check cache
        if symbol in self._latest_bars:
            if resolution in self._latest_bars[symbol]:
                return self._latest_bars[symbol][resolution]

        # Check buffer
        bars = self._bar_buffer.get_latest(resolution, 1)
        if bars:
            return bars[0]

        return None

    def get_recent_ticks(
        self,
        symbol: str,
        count: int = 100
    ) -> list[Tick]:
        """Get recent ticks from buffer."""
        return self._tick_buffers[symbol].get_latest(count)

    def get_recent_bars(
        self,
        symbol: str,
        resolution: DataResolution,
        count: int = 100
    ) -> list[OHLCV]:
        """Get recent bars from buffer."""
        return self._bar_buffer.get_latest(resolution, count)

    # ==================== Streaming Iterators ====================

    async def stream_ticks(
        self,
        symbol: str,
        timeout: Optional[float] = None
    ):
        """
        Async iterator for tick data.

        Args:
            symbol: Symbol to stream
            timeout: Optional timeout between ticks

        Yields:
            Tick objects
        """
        queue: asyncio.Queue[Tick] = asyncio.Queue()

        def callback(tick: Tick):
            if tick.symbol == symbol:
                queue.put_nowait(tick)

        self.on_tick(symbol, callback)

        try:
            while self.is_connected:
                try:
                    tick = await asyncio.wait_for(queue.get(), timeout=timeout)
                    yield tick
                except asyncio.TimeoutError:
                    if timeout:
                        continue
                    break
        finally:
            # Remove callback
            if symbol in self._tick_callbacks:
                self._tick_callbacks[symbol].remove(callback)

    async def stream_bars(
        self,
        symbol: str,
        resolution: DataResolution = DataResolution.MINUTE_1,
        timeout: Optional[float] = None
    ):
        """
        Async iterator for bar data.

        Args:
            symbol: Symbol to stream
            resolution: Bar resolution
            timeout: Optional timeout between bars

        Yields:
            OHLCV objects
        """
        queue: asyncio.Queue[OHLCV] = asyncio.Queue()

        def callback(bar: OHLCV):
            if bar.symbol == symbol:
                queue.put_nowait(bar)

        self.on_bar(symbol, callback)

        try:
            while self.is_connected:
                try:
                    bar = await asyncio.wait_for(queue.get(), timeout=timeout)
                    yield bar
                except asyncio.TimeoutError:
                    if timeout:
                        continue
                    break
        finally:
            if symbol in self._bar_callbacks:
                self._bar_callbacks[symbol].remove(callback)


class LiveFeedManager:
    """
    Manages multiple live feeds from different exchanges.

    Provides unified interface for multi-exchange data.

    Usage:
        manager = LiveFeedManager()

        # Add exchanges
        manager.add_feed("binance", binance_exchange)
        manager.add_feed("ib", ib_exchange)

        # Subscribe across exchanges
        await manager.subscribe({
            "binance": ["BTCUSDT", "ETHUSDT"],
            "ib": ["AAPL", "MSFT"],
        })

        # Get data from any exchange
        tick = await manager.get_latest_tick("BTCUSDT")
    """

    def __init__(self):
        self._feeds: dict[str, LiveFeed] = {}
        self._symbol_to_feed: dict[str, str] = {}  # symbol -> feed_name

        # Global callbacks
        self._tick_callbacks: list[Callable[[str, Tick], None]] = []
        self._bar_callbacks: list[Callable[[str, OHLCV], None]] = []

    def add_feed(
        self,
        name: str,
        exchange: Exchange,
        config: Optional[FeedConfig] = None,
    ) -> LiveFeed:
        """
        Add a feed for an exchange.

        Args:
            name: Feed identifier
            exchange: Exchange instance
            config: Feed configuration

        Returns:
            Created LiveFeed
        """
        feed = LiveFeed(exchange, config)
        self._feeds[name] = feed

        # Wire up callbacks to global handlers
        feed.on_tick("*", lambda tick: self._handle_tick(name, tick))
        feed.on_bar("*", lambda bar: self._handle_bar(name, bar))

        return feed

    def get_feed(self, name: str) -> Optional[LiveFeed]:
        """Get feed by name."""
        return self._feeds.get(name)

    async def connect_all(self) -> dict[str, bool]:
        """Connect all feeds."""
        results = {}
        for name, feed in self._feeds.items():
            results[name] = await feed.connect()
        return results

    async def disconnect_all(self) -> None:
        """Disconnect all feeds."""
        for feed in self._feeds.values():
            await feed.disconnect()

    async def subscribe(
        self,
        subscriptions: dict[str, list[str]]
    ) -> None:
        """
        Subscribe to symbols on multiple exchanges.

        Args:
            subscriptions: Dict mapping feed name to symbol list
        """
        for feed_name, symbols in subscriptions.items():
            feed = self._feeds.get(feed_name)
            if feed:
                await feed.subscribe(symbols)
                for symbol in symbols:
                    self._symbol_to_feed[symbol] = feed_name

    def _handle_tick(self, feed_name: str, tick: Tick) -> None:
        """Handle tick from any feed."""
        for callback in self._tick_callbacks:
            try:
                callback(feed_name, tick)
            except Exception as e:
                print(f"Error in global tick callback: {e}")

    def _handle_bar(self, feed_name: str, bar: OHLCV) -> None:
        """Handle bar from any feed."""
        for callback in self._bar_callbacks:
            try:
                callback(feed_name, bar)
            except Exception as e:
                print(f"Error in global bar callback: {e}")

    def on_tick(self, callback: Callable[[str, Tick], None]) -> None:
        """Register global tick callback (receives feed_name, tick)."""
        self._tick_callbacks.append(callback)

    def on_bar(self, callback: Callable[[str, OHLCV], None]) -> None:
        """Register global bar callback (receives feed_name, bar)."""
        self._bar_callbacks.append(callback)

    async def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        """Get latest tick for symbol from appropriate feed."""
        feed_name = self._symbol_to_feed.get(symbol)
        if feed_name and feed_name in self._feeds:
            return await self._feeds[feed_name].get_latest_tick(symbol)

        # Search all feeds
        for feed in self._feeds.values():
            tick = await feed.get_latest_tick(symbol)
            if tick:
                return tick

        return None

    async def get_latest_bar(
        self,
        symbol: str,
        resolution: DataResolution
    ) -> Optional[OHLCV]:
        """Get latest bar for symbol from appropriate feed."""
        feed_name = self._symbol_to_feed.get(symbol)
        if feed_name and feed_name in self._feeds:
            return await self._feeds[feed_name].get_latest_bar(symbol, resolution)

        # Search all feeds
        for feed in self._feeds.values():
            bar = await feed.get_latest_bar(symbol, resolution)
            if bar:
                return bar

        return None

    @property
    def is_connected(self) -> bool:
        """Check if any feed is connected."""
        return any(feed.is_connected for feed in self._feeds.values())

    @property
    def all_connected(self) -> bool:
        """Check if all feeds are connected."""
        return all(feed.is_connected for feed in self._feeds.values())
