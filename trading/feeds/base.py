"""
Base classes for data feeds.

Provides abstractions for:
- Data feed interface
- Tick-to-bar aggregation
- Feed status management
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Optional, Callable, AsyncIterator
from collections import defaultdict
import asyncio

from trading.core.models import Tick, OHLCV, OrderBook
from trading.core.enums import DataResolution


class FeedStatus(Enum):
    """Data feed connection status."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    ERROR = auto()


@dataclass
class FeedConfig:
    """Configuration for data feeds."""
    # Symbols to subscribe
    symbols: list[str] = field(default_factory=list)

    # Data types to subscribe
    subscribe_ticks: bool = True
    subscribe_bars: bool = True
    subscribe_orderbook: bool = False

    # Bar aggregation settings
    bar_resolutions: list[DataResolution] = field(
        default_factory=lambda: [DataResolution.MINUTE_1]
    )

    # Connection settings
    reconnect_attempts: int = 5
    reconnect_delay_seconds: float = 5.0
    heartbeat_interval_seconds: float = 30.0

    # Buffering
    tick_buffer_size: int = 10000
    bar_buffer_size: int = 1000

    # History settings
    store_ticks: bool = False  # Ticks can be large
    store_bars: bool = True
    history_days: int = 30  # Days of history to maintain


class DataFeed(ABC):
    """
    Abstract base class for data feeds.

    Provides interface for:
    - Subscribing to market data
    - Receiving ticks and bars
    - Managing feed lifecycle
    """

    def __init__(self, config: FeedConfig):
        self.config = config
        self._status = FeedStatus.DISCONNECTED
        self._tick_callbacks: dict[str, list[Callable[[Tick], None]]] = defaultdict(list)
        self._bar_callbacks: dict[str, list[Callable[[OHLCV], None]]] = defaultdict(list)
        self._orderbook_callbacks: dict[str, list[Callable[[OrderBook], None]]] = defaultdict(list)
        self._status_callbacks: list[Callable[[FeedStatus], None]] = []

    @property
    def status(self) -> FeedStatus:
        """Current feed status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Check if feed is connected."""
        return self._status == FeedStatus.CONNECTED

    # ==================== Lifecycle ====================

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the data feed.

        Returns:
            True if connected successfully.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data feed."""
        pass

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None:
        """
        Subscribe to symbols.

        Args:
            symbols: List of symbols to subscribe.
        """
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: list[str]) -> None:
        """
        Unsubscribe from symbols.

        Args:
            symbols: List of symbols to unsubscribe.
        """
        pass

    # ==================== Callbacks ====================

    def on_tick(self, symbol: str, callback: Callable[[Tick], None]) -> None:
        """Register tick callback for symbol."""
        self._tick_callbacks[symbol].append(callback)

    def on_bar(self, symbol: str, callback: Callable[[OHLCV], None]) -> None:
        """Register bar callback for symbol."""
        self._bar_callbacks[symbol].append(callback)

    def on_orderbook(self, symbol: str, callback: Callable[[OrderBook], None]) -> None:
        """Register order book callback for symbol."""
        self._orderbook_callbacks[symbol].append(callback)

    def on_status_change(self, callback: Callable[[FeedStatus], None]) -> None:
        """Register status change callback."""
        self._status_callbacks.append(callback)

    def _emit_tick(self, tick: Tick) -> None:
        """Emit tick to registered callbacks."""
        for callback in self._tick_callbacks.get(tick.symbol, []):
            try:
                callback(tick)
            except Exception as e:
                print(f"Error in tick callback: {e}")

        # Also emit to wildcard subscribers
        for callback in self._tick_callbacks.get("*", []):
            try:
                callback(tick)
            except Exception as e:
                print(f"Error in tick callback: {e}")

    def _emit_bar(self, bar: OHLCV) -> None:
        """Emit bar to registered callbacks."""
        for callback in self._bar_callbacks.get(bar.symbol, []):
            try:
                callback(bar)
            except Exception as e:
                print(f"Error in bar callback: {e}")

        for callback in self._bar_callbacks.get("*", []):
            try:
                callback(bar)
            except Exception as e:
                print(f"Error in bar callback: {e}")

    def _emit_orderbook(self, orderbook: OrderBook) -> None:
        """Emit order book to registered callbacks."""
        for callback in self._orderbook_callbacks.get(orderbook.symbol, []):
            try:
                callback(orderbook)
            except Exception as e:
                print(f"Error in orderbook callback: {e}")

    def _set_status(self, status: FeedStatus) -> None:
        """Update status and notify callbacks."""
        if status != self._status:
            self._status = status
            for callback in self._status_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    print(f"Error in status callback: {e}")

    # ==================== Data Access ====================

    @abstractmethod
    async def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        """Get the most recent tick for a symbol."""
        pass

    @abstractmethod
    async def get_latest_bar(
        self,
        symbol: str,
        resolution: DataResolution
    ) -> Optional[OHLCV]:
        """Get the most recent bar for a symbol and resolution."""
        pass


class TickAggregator:
    """
    Aggregates ticks into OHLCV bars.

    Supports multiple resolutions simultaneously.
    Handles bar completion and emission.
    """

    def __init__(
        self,
        symbol: str,
        resolutions: list[DataResolution],
        on_bar: Optional[Callable[[OHLCV], None]] = None,
    ):
        self.symbol = symbol
        self.resolutions = resolutions
        self.on_bar = on_bar

        # Current bar state for each resolution
        self._current_bars: dict[DataResolution, dict] = {}
        self._bar_start_times: dict[DataResolution, datetime] = {}

        for resolution in resolutions:
            self._reset_bar(resolution)

    def _reset_bar(self, resolution: DataResolution, start_time: Optional[datetime] = None) -> None:
        """Reset bar state for a resolution."""
        self._current_bars[resolution] = {
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "volume": Decimal("0"),
            "trades": 0,
            "vwap_numerator": Decimal("0"),
        }
        self._bar_start_times[resolution] = start_time or datetime.utcnow()

    def _get_bar_start(self, timestamp: datetime, resolution: DataResolution) -> datetime:
        """Calculate bar start time for a timestamp."""
        seconds = resolution.to_seconds()
        if seconds == 0:
            return timestamp

        # Align to bar boundary
        epoch = timestamp.timestamp()
        bar_epoch = (epoch // seconds) * seconds
        return datetime.utcfromtimestamp(bar_epoch)

    def _should_close_bar(
        self,
        timestamp: datetime,
        resolution: DataResolution
    ) -> bool:
        """Check if current bar should be closed."""
        if resolution not in self._bar_start_times:
            return False

        bar_start = self._bar_start_times[resolution]
        seconds = resolution.to_seconds()

        if seconds == 0:
            return False

        return timestamp >= bar_start + timedelta(seconds=seconds)

    def process_tick(self, tick: Tick) -> list[OHLCV]:
        """
        Process a tick and update bars.

        Args:
            tick: Incoming tick data.

        Returns:
            List of completed bars (may be empty).
        """
        completed_bars = []
        price = tick.last_price or tick.mid_price
        size = tick.last_size or Decimal("0")

        for resolution in self.resolutions:
            # Check if bar should be closed
            if self._should_close_bar(tick.timestamp, resolution):
                bar = self._complete_bar(resolution)
                if bar:
                    completed_bars.append(bar)
                    if self.on_bar:
                        self.on_bar(bar)

                # Start new bar aligned to boundary
                bar_start = self._get_bar_start(tick.timestamp, resolution)
                self._reset_bar(resolution, bar_start)

            # Update current bar
            bar_data = self._current_bars[resolution]

            if bar_data["open"] is None:
                bar_data["open"] = price
                self._bar_start_times[resolution] = self._get_bar_start(
                    tick.timestamp, resolution
                )

            bar_data["high"] = max(bar_data["high"] or price, price)
            bar_data["low"] = min(bar_data["low"] or price, price)
            bar_data["close"] = price
            bar_data["volume"] += size
            bar_data["trades"] += 1
            bar_data["vwap_numerator"] += price * size

        return completed_bars

    def _complete_bar(self, resolution: DataResolution) -> Optional[OHLCV]:
        """Complete and return current bar."""
        bar_data = self._current_bars[resolution]

        if bar_data["open"] is None:
            return None

        # Calculate VWAP
        vwap = None
        if bar_data["volume"] > 0:
            vwap = bar_data["vwap_numerator"] / bar_data["volume"]

        return OHLCV(
            symbol=self.symbol,
            timestamp=self._bar_start_times[resolution],
            open=bar_data["open"],
            high=bar_data["high"],
            low=bar_data["low"],
            close=bar_data["close"],
            volume=bar_data["volume"],
            trades=bar_data["trades"],
            vwap=vwap,
        )

    def flush(self) -> list[OHLCV]:
        """
        Force complete all current bars.

        Useful when feed disconnects or at end of session.

        Returns:
            List of completed bars.
        """
        bars = []
        for resolution in self.resolutions:
            bar = self._complete_bar(resolution)
            if bar:
                bars.append(bar)
                if self.on_bar:
                    self.on_bar(bar)
            self._reset_bar(resolution)
        return bars

    def get_current_bar(self, resolution: DataResolution) -> Optional[OHLCV]:
        """Get current incomplete bar for a resolution."""
        return self._complete_bar(resolution)


class TickBuffer:
    """
    Circular buffer for tick data.

    Provides efficient storage and retrieval of recent ticks.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer: list[Tick] = []
        self._index = 0

    def add(self, tick: Tick) -> None:
        """Add a tick to the buffer."""
        if len(self._buffer) < self.max_size:
            self._buffer.append(tick)
        else:
            self._buffer[self._index] = tick
        self._index = (self._index + 1) % self.max_size

    def get_latest(self, count: int = 1) -> list[Tick]:
        """Get the most recent ticks."""
        if not self._buffer:
            return []

        count = min(count, len(self._buffer))
        result = []

        for i in range(count):
            idx = (self._index - 1 - i) % len(self._buffer)
            result.append(self._buffer[idx])

        return result

    def get_since(self, timestamp: datetime) -> list[Tick]:
        """Get all ticks since a timestamp."""
        result = []
        for tick in self._buffer:
            if tick.timestamp >= timestamp:
                result.append(tick)
        return sorted(result, key=lambda t: t.timestamp)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._index = 0

    def __len__(self) -> int:
        return len(self._buffer)


class BarBuffer:
    """
    Buffer for OHLCV bars organized by resolution.
    """

    def __init__(self, max_bars_per_resolution: int = 1000):
        self.max_size = max_bars_per_resolution
        self._bars: dict[DataResolution, list[OHLCV]] = defaultdict(list)

    def add(self, bar: OHLCV, resolution: DataResolution) -> None:
        """Add a bar to the buffer."""
        bars = self._bars[resolution]
        bars.append(bar)

        # Trim if needed
        if len(bars) > self.max_size:
            self._bars[resolution] = bars[-self.max_size:]

    def get_latest(
        self,
        resolution: DataResolution,
        count: int = 1
    ) -> list[OHLCV]:
        """Get the most recent bars for a resolution."""
        bars = self._bars.get(resolution, [])
        return bars[-count:] if bars else []

    def get_range(
        self,
        resolution: DataResolution,
        start: datetime,
        end: Optional[datetime] = None
    ) -> list[OHLCV]:
        """Get bars in a time range."""
        bars = self._bars.get(resolution, [])
        end = end or datetime.utcnow()

        return [
            bar for bar in bars
            if start <= bar.timestamp <= end
        ]

    def clear(self, resolution: Optional[DataResolution] = None) -> None:
        """Clear bars, optionally for specific resolution."""
        if resolution:
            self._bars[resolution].clear()
        else:
            self._bars.clear()
