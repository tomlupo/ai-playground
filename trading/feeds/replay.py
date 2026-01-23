"""
Data replay for backtesting.

Provides:
- Historical data playback at configurable speeds
- Event-driven backtesting support
- Multiple data source support
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Optional, Callable, Iterator, AsyncIterator
import heapq

from trading.core.models import Tick, OHLCV
from trading.core.enums import DataResolution
from trading.feeds.base import DataFeed, FeedConfig, FeedStatus, TickAggregator
from trading.feeds.history import HistoryStore


class ReplaySpeed(Enum):
    """Replay speed modes."""
    REALTIME = auto()     # 1x speed
    FAST = auto()         # As fast as possible
    FIXED_RATE = auto()   # Fixed events per second


@dataclass
class ReplayConfig:
    """Configuration for data replay."""
    # Time range
    start: datetime = field(default_factory=datetime.utcnow)
    end: Optional[datetime] = None

    # Speed control
    speed_mode: ReplaySpeed = ReplaySpeed.FAST
    speed_multiplier: float = 1.0  # For REALTIME mode
    events_per_second: float = 1000.0  # For FIXED_RATE mode

    # Data settings
    use_ticks: bool = False  # Replay ticks (slower) or bars
    bar_resolution: DataResolution = DataResolution.MINUTE_1

    # Preload settings
    preload_bars: int = 100  # Bars to preload before start
    chunk_size: int = 10000  # Data chunk size for loading


class DataReplayer:
    """
    Replays historical data for backtesting.

    Supports:
    - Tick-level and bar-level replay
    - Variable speed playback
    - Multiple symbols
    - Event callbacks

    Usage:
        store = SQLiteHistoryStore("./data/history.db")
        replayer = DataReplayer(store, ReplayConfig(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            speed_mode=ReplaySpeed.FAST,
        ))

        # Register callbacks
        replayer.on_bar(lambda bar: strategy.process(bar))

        # Run replay
        await replayer.run(["BTCUSDT", "ETHUSDT"])

        # Or use iterator
        async for event in replayer.iterate(["BTCUSDT"]):
            process(event)
    """

    def __init__(
        self,
        store: HistoryStore,
        config: Optional[ReplayConfig] = None,
    ):
        self.store = store
        self.config = config or ReplayConfig()

        # Callbacks
        self._tick_callbacks: list[Callable[[Tick], None]] = []
        self._bar_callbacks: list[Callable[[OHLCV], None]] = []

        # State
        self._running = False
        self._paused = False
        self._current_time: Optional[datetime] = None
        self._events_processed = 0

    @property
    def current_time(self) -> Optional[datetime]:
        """Current simulation time."""
        return self._current_time

    @property
    def is_running(self) -> bool:
        return self._running

    def on_tick(self, callback: Callable[[Tick], None]) -> None:
        """Register tick callback."""
        self._tick_callbacks.append(callback)

    def on_bar(self, callback: Callable[[OHLCV], None]) -> None:
        """Register bar callback."""
        self._bar_callbacks.append(callback)

    async def run(self, symbols: list[str]) -> None:
        """
        Run replay for specified symbols.

        Args:
            symbols: List of symbols to replay
        """
        self._running = True
        self._paused = False
        self._events_processed = 0

        try:
            async for event in self.iterate(symbols):
                if not self._running:
                    break

                # Wait if paused
                while self._paused:
                    await asyncio.sleep(0.1)

                # Emit event
                if isinstance(event, Tick):
                    for callback in self._tick_callbacks:
                        callback(event)
                elif isinstance(event, OHLCV):
                    for callback in self._bar_callbacks:
                        callback(event)

                self._events_processed += 1

        finally:
            self._running = False

    async def iterate(
        self,
        symbols: list[str]
    ) -> AsyncIterator[Tick | OHLCV]:
        """
        Async iterator over replay events.

        Args:
            symbols: Symbols to iterate

        Yields:
            Tick or OHLCV events in chronological order
        """
        if self.config.use_ticks:
            async for tick in self._iterate_ticks(symbols):
                yield tick
        else:
            async for bar in self._iterate_bars(symbols):
                yield bar

    async def _iterate_ticks(
        self,
        symbols: list[str]
    ) -> AsyncIterator[Tick]:
        """Iterate over ticks from multiple symbols."""
        # Use a heap to merge ticks from multiple symbols
        heap: list[tuple[datetime, str, Tick]] = []
        iterators: dict[str, Iterator[Tick]] = {}

        # Initialize iterators for each symbol
        for symbol in symbols:
            ticks = self.store.get_ticks(
                symbol,
                self.config.start,
                self.config.end,
            )
            if ticks:
                it = iter(ticks)
                iterators[symbol] = it
                try:
                    tick = next(it)
                    heapq.heappush(heap, (tick.timestamp, symbol, tick))
                except StopIteration:
                    pass

        last_time: Optional[datetime] = None

        while heap:
            timestamp, symbol, tick = heapq.heappop(heap)

            # Speed control
            await self._apply_speed_delay(timestamp, last_time)
            last_time = timestamp
            self._current_time = timestamp

            yield tick

            # Get next tick from same symbol
            if symbol in iterators:
                try:
                    next_tick = next(iterators[symbol])
                    heapq.heappush(heap, (next_tick.timestamp, symbol, next_tick))
                except StopIteration:
                    del iterators[symbol]

    async def _iterate_bars(
        self,
        symbols: list[str]
    ) -> AsyncIterator[OHLCV]:
        """Iterate over bars from multiple symbols."""
        heap: list[tuple[datetime, str, OHLCV]] = []
        iterators: dict[str, Iterator[OHLCV]] = {}

        # Initialize iterators
        for symbol in symbols:
            bars = self.store.get_bars(
                symbol,
                self.config.bar_resolution,
                self.config.start,
                self.config.end,
            )
            if bars:
                it = iter(bars)
                iterators[symbol] = it
                try:
                    bar = next(it)
                    heapq.heappush(heap, (bar.timestamp, symbol, bar))
                except StopIteration:
                    pass

        last_time: Optional[datetime] = None

        while heap:
            timestamp, symbol, bar = heapq.heappop(heap)

            # Speed control
            await self._apply_speed_delay(timestamp, last_time)
            last_time = timestamp
            self._current_time = timestamp

            yield bar

            # Get next bar from same symbol
            if symbol in iterators:
                try:
                    next_bar = next(iterators[symbol])
                    heapq.heappush(heap, (next_bar.timestamp, symbol, next_bar))
                except StopIteration:
                    del iterators[symbol]

    async def _apply_speed_delay(
        self,
        current: datetime,
        previous: Optional[datetime]
    ) -> None:
        """Apply delay based on speed mode."""
        if self.config.speed_mode == ReplaySpeed.FAST:
            # Yield control but don't wait
            if self._events_processed % 100 == 0:
                await asyncio.sleep(0)
            return

        if previous is None:
            return

        if self.config.speed_mode == ReplaySpeed.REALTIME:
            # Calculate real time difference
            delta = (current - previous).total_seconds()
            delay = delta / self.config.speed_multiplier
            if delay > 0:
                await asyncio.sleep(delay)

        elif self.config.speed_mode == ReplaySpeed.FIXED_RATE:
            # Fixed delay between events
            delay = 1.0 / self.config.events_per_second
            await asyncio.sleep(delay)

    def pause(self) -> None:
        """Pause replay."""
        self._paused = True

    def resume(self) -> None:
        """Resume replay."""
        self._paused = False

    def stop(self) -> None:
        """Stop replay."""
        self._running = False


class BacktestFeed(DataFeed):
    """
    Data feed that replays historical data.

    Implements DataFeed interface for backtesting compatibility.
    Can be used as drop-in replacement for LiveFeed.

    Usage:
        store = SQLiteHistoryStore("./data/history.db")
        feed = BacktestFeed(store, FeedConfig(
            symbols=["BTCUSDT"],
            bar_resolutions=[DataResolution.MINUTE_1],
        ))

        feed.set_time_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
        )

        feed.on_bar("BTCUSDT", lambda bar: strategy.process(bar))

        await feed.connect()
        await feed.run()
    """

    def __init__(
        self,
        store: HistoryStore,
        config: Optional[FeedConfig] = None,
    ):
        super().__init__(config or FeedConfig())
        self.store = store

        # Replay settings
        self._start: Optional[datetime] = None
        self._end: Optional[datetime] = None
        self._replay_speed = ReplaySpeed.FAST
        self._speed_multiplier = 1.0

        # State
        self._current_time: Optional[datetime] = None
        self._replayer: Optional[DataReplayer] = None
        self._run_task: Optional[asyncio.Task] = None

        # Data cache
        self._preloaded_bars: dict[str, list[OHLCV]] = {}
        self._current_bar_index: dict[str, int] = {}

    def set_time_range(
        self,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> None:
        """Set replay time range."""
        self._start = start
        self._end = end

    def set_speed(
        self,
        mode: ReplaySpeed,
        multiplier: float = 1.0
    ) -> None:
        """Set replay speed."""
        self._replay_speed = mode
        self._speed_multiplier = multiplier

    @property
    def current_time(self) -> Optional[datetime]:
        """Current simulation time."""
        return self._current_time

    async def connect(self) -> bool:
        """Initialize backtest feed."""
        if not self._start:
            # Use available data range
            if self.config.symbols:
                start, end = self.store.get_time_range(
                    self.config.symbols[0],
                    self.config.bar_resolutions[0] if self.config.bar_resolutions else None,
                )
                self._start = start or datetime.utcnow() - timedelta(days=30)
                self._end = end

        # Preload data
        for symbol in self.config.symbols:
            for resolution in self.config.bar_resolutions:
                bars = self.store.get_bars(
                    symbol,
                    resolution,
                    self._start,
                    self._end,
                )
                key = f"{symbol}_{resolution.name}"
                self._preloaded_bars[key] = bars
                self._current_bar_index[key] = 0

        self._set_status(FeedStatus.CONNECTED)
        self._current_time = self._start

        return True

    async def disconnect(self) -> None:
        """Stop backtest feed."""
        if self._run_task:
            self._run_task.cancel()
            self._run_task = None

        self._set_status(FeedStatus.DISCONNECTED)

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols (preload their data)."""
        for symbol in symbols:
            if symbol not in self.config.symbols:
                self.config.symbols.append(symbol)

                # Load data
                for resolution in self.config.bar_resolutions:
                    bars = self.store.get_bars(
                        symbol,
                        resolution,
                        self._start,
                        self._end,
                    )
                    key = f"{symbol}_{resolution.name}"
                    self._preloaded_bars[key] = bars
                    self._current_bar_index[key] = 0

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols."""
        for symbol in symbols:
            if symbol in self.config.symbols:
                self.config.symbols.remove(symbol)

    async def run(self) -> None:
        """
        Run the backtest replay.

        Emits events to registered callbacks.
        """
        self._set_status(FeedStatus.CONNECTED)

        replay_config = ReplayConfig(
            start=self._start,
            end=self._end,
            speed_mode=self._replay_speed,
            speed_multiplier=self._speed_multiplier,
            use_ticks=self.config.subscribe_ticks,
            bar_resolution=self.config.bar_resolutions[0] if self.config.bar_resolutions else DataResolution.MINUTE_1,
        )

        self._replayer = DataReplayer(self.store, replay_config)

        # Wire callbacks
        self._replayer.on_tick(lambda tick: self._handle_tick(tick))
        self._replayer.on_bar(lambda bar: self._handle_bar(bar))

        await self._replayer.run(self.config.symbols)

    def _handle_tick(self, tick: Tick) -> None:
        """Handle replayed tick."""
        self._current_time = tick.timestamp
        self._emit_tick(tick)

    def _handle_bar(self, bar: OHLCV) -> None:
        """Handle replayed bar."""
        self._current_time = bar.timestamp
        self._emit_bar(bar)

    async def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        """Get latest tick up to current time."""
        if not self._current_time:
            return None

        ticks = self.store.get_ticks(
            symbol,
            self._start or datetime.min,
            self._current_time,
            limit=1,
        )
        return ticks[-1] if ticks else None

    async def get_latest_bar(
        self,
        symbol: str,
        resolution: DataResolution
    ) -> Optional[OHLCV]:
        """Get latest bar up to current time."""
        key = f"{symbol}_{resolution.name}"
        bars = self._preloaded_bars.get(key, [])
        index = self._current_bar_index.get(key, 0)

        if bars and index > 0:
            return bars[index - 1]

        return None

    def get_historical_bars(
        self,
        symbol: str,
        resolution: DataResolution,
        count: int = 100,
    ) -> list[OHLCV]:
        """
        Get historical bars up to current time.

        Useful for calculating indicators during backtest.

        Args:
            symbol: Trading symbol
            resolution: Bar resolution
            count: Number of bars to retrieve

        Returns:
            List of bars (most recent last)
        """
        key = f"{symbol}_{resolution.name}"
        bars = self._preloaded_bars.get(key, [])
        index = self._current_bar_index.get(key, 0)

        # Get bars up to current position
        start_idx = max(0, index - count)
        return bars[start_idx:index]

    def advance(self) -> Optional[OHLCV]:
        """
        Manually advance to next bar (for step-by-step backtesting).

        Returns:
            Next bar or None if end reached
        """
        if not self.config.symbols or not self.config.bar_resolutions:
            return None

        symbol = self.config.symbols[0]
        resolution = self.config.bar_resolutions[0]
        key = f"{symbol}_{resolution.name}"

        bars = self._preloaded_bars.get(key, [])
        index = self._current_bar_index.get(key, 0)

        if index >= len(bars):
            return None

        bar = bars[index]
        self._current_bar_index[key] = index + 1
        self._current_time = bar.timestamp

        self._emit_bar(bar)

        return bar

    def reset(self) -> None:
        """Reset to beginning of backtest period."""
        for key in self._current_bar_index:
            self._current_bar_index[key] = 0
        self._current_time = self._start


class BarDataFrame:
    """
    DataFrame-like interface for bar data.

    Provides convenient access to historical bars for indicator calculation.

    Usage:
        df = BarDataFrame(bars)

        # Access columns
        closes = df.close  # numpy array
        highs = df.high

        # Calculate SMA
        sma = df.close.rolling(20).mean()

        # Get last N values
        last_10 = df.tail(10)
    """

    def __init__(self, bars: list[OHLCV]):
        self._bars = bars

        # Lazy-loaded arrays
        self._timestamps: Optional[list] = None
        self._opens: Optional[list] = None
        self._highs: Optional[list] = None
        self._lows: Optional[list] = None
        self._closes: Optional[list] = None
        self._volumes: Optional[list] = None

    def __len__(self) -> int:
        return len(self._bars)

    @property
    def timestamp(self) -> list[datetime]:
        if self._timestamps is None:
            self._timestamps = [b.timestamp for b in self._bars]
        return self._timestamps

    @property
    def open(self) -> list[float]:
        if self._opens is None:
            self._opens = [float(b.open) for b in self._bars]
        return self._opens

    @property
    def high(self) -> list[float]:
        if self._highs is None:
            self._highs = [float(b.high) for b in self._bars]
        return self._highs

    @property
    def low(self) -> list[float]:
        if self._lows is None:
            self._lows = [float(b.low) for b in self._bars]
        return self._lows

    @property
    def close(self) -> list[float]:
        if self._closes is None:
            self._closes = [float(b.close) for b in self._bars]
        return self._closes

    @property
    def volume(self) -> list[float]:
        if self._volumes is None:
            self._volumes = [float(b.volume) for b in self._bars]
        return self._volumes

    def tail(self, n: int) -> "BarDataFrame":
        """Get last n bars."""
        return BarDataFrame(self._bars[-n:])

    def head(self, n: int) -> "BarDataFrame":
        """Get first n bars."""
        return BarDataFrame(self._bars[:n])

    def slice(self, start: int, end: int) -> "BarDataFrame":
        """Get slice of bars."""
        return BarDataFrame(self._bars[start:end])

    @property
    def last(self) -> Optional[OHLCV]:
        """Get last bar."""
        return self._bars[-1] if self._bars else None

    @property
    def first(self) -> Optional[OHLCV]:
        """Get first bar."""
        return self._bars[0] if self._bars else None

    def sma(self, period: int, field: str = "close") -> list[Optional[float]]:
        """Calculate Simple Moving Average."""
        data = getattr(self, field)
        result = [None] * len(data)

        for i in range(period - 1, len(data)):
            window = data[i - period + 1:i + 1]
            result[i] = sum(window) / period

        return result

    def ema(self, period: int, field: str = "close") -> list[Optional[float]]:
        """Calculate Exponential Moving Average."""
        data = getattr(self, field)
        result = [None] * len(data)

        if len(data) < period:
            return result

        # Start with SMA
        result[period - 1] = sum(data[:period]) / period

        # EMA multiplier
        mult = 2 / (period + 1)

        for i in range(period, len(data)):
            result[i] = (data[i] - result[i - 1]) * mult + result[i - 1]

        return result

    def atr(self, period: int = 14) -> list[Optional[float]]:
        """Calculate Average True Range."""
        if len(self._bars) < 2:
            return [None] * len(self._bars)

        tr_values = [None]

        for i in range(1, len(self._bars)):
            high = float(self._bars[i].high)
            low = float(self._bars[i].low)
            prev_close = float(self._bars[i - 1].close)

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            tr_values.append(tr)

        # Calculate ATR as EMA of TR
        result = [None] * len(self._bars)
        if len(tr_values) < period + 1:
            return result

        # Initial ATR
        result[period] = sum(tr_values[1:period + 1]) / period

        # Smoothed ATR
        for i in range(period + 1, len(tr_values)):
            result[i] = (result[i - 1] * (period - 1) + tr_values[i]) / period

        return result
