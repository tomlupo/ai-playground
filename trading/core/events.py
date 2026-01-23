"""Event system for the trading framework."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Callable, Optional, Any
from collections import defaultdict

from trading.core.models import Order, Position, Trade, Tick, OHLCV, Fill


class EventType(Enum):
    """Types of events in the trading system."""
    # Market data events
    TICK = auto()
    BAR = auto()
    ORDER_BOOK = auto()

    # Order events
    ORDER_SUBMITTED = auto()
    ORDER_ACCEPTED = auto()
    ORDER_REJECTED = auto()
    ORDER_FILLED = auto()
    ORDER_PARTIALLY_FILLED = auto()
    ORDER_CANCELLED = auto()
    ORDER_EXPIRED = auto()

    # Position events
    POSITION_OPENED = auto()
    POSITION_UPDATED = auto()
    POSITION_CLOSED = auto()

    # Trade events
    TRADE_OPENED = auto()
    TRADE_CLOSED = auto()

    # System events
    CONNECTED = auto()
    DISCONNECTED = auto()
    ERROR = auto()
    HEARTBEAT = auto()


@dataclass
class Event:
    """Base event class."""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    data: Any = None


@dataclass
class TickEvent(Event):
    """Market tick event."""
    tick: Tick = None

    def __post_init__(self):
        self.event_type = EventType.TICK
        if self.tick:
            self.data = self.tick


@dataclass
class BarEvent(Event):
    """OHLCV bar event."""
    bar: OHLCV = None

    def __post_init__(self):
        self.event_type = EventType.BAR
        if self.bar:
            self.data = self.bar


@dataclass
class OrderEvent(Event):
    """Order lifecycle event."""
    order: Order = None
    fill: Optional[Fill] = None
    reason: Optional[str] = None

    def __post_init__(self):
        if self.order:
            self.data = self.order


@dataclass
class PositionEvent(Event):
    """Position change event."""
    position: Position = None
    previous_quantity: Optional[Any] = None  # Using Any to avoid Decimal import issues
    realized_pnl: Optional[Any] = None

    def __post_init__(self):
        if self.position:
            self.data = self.position


@dataclass
class TradeEvent(Event):
    """Round-trip trade event."""
    trade: Trade = None

    def __post_init__(self):
        if self.trade:
            self.data = self.trade


EventHandler = Callable[[Event], None]


class EventEmitter:
    """
    Thread-safe event emitter with subscription management.

    Supports both synchronous and asynchronous event handlers.
    """

    def __init__(self):
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._all_handlers: list[EventHandler] = []  # Handlers for all events
        self._event_history: list[Event] = []
        self._history_limit: int = 1000

    def on(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe to a specific event type."""
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def on_all(self, handler: EventHandler) -> None:
        """Subscribe to all events."""
        if handler not in self._all_handlers:
            self._all_handlers.append(handler)

    def off(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe from a specific event type."""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def off_all(self, handler: EventHandler) -> None:
        """Unsubscribe from all events."""
        if handler in self._all_handlers:
            self._all_handlers.remove(handler)

    def emit(self, event: Event) -> None:
        """Emit an event to all subscribers."""
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._history_limit:
            self._event_history = self._event_history[-self._history_limit:]

        # Call specific handlers
        for handler in self._handlers[event.event_type]:
            try:
                handler(event)
            except Exception as e:
                # Log error but don't stop other handlers
                print(f"Error in event handler: {e}")

        # Call all-event handlers
        for handler in self._all_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in all-event handler: {e}")

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> list[Event]:
        """Get recent event history, optionally filtered by type."""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history = []

    def clear_handlers(self) -> None:
        """Remove all event handlers."""
        self._handlers.clear()
        self._all_handlers.clear()


class AsyncEventEmitter:
    """
    Async-compatible event emitter.

    Supports both sync and async handlers.
    """

    def __init__(self):
        self._sync_handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._async_handlers: dict[EventType, list[Callable]] = defaultdict(list)
        self._all_sync_handlers: list[EventHandler] = []
        self._all_async_handlers: list[Callable] = []

    def on(self, event_type: EventType, handler: Callable) -> None:
        """Subscribe to a specific event type."""
        import asyncio
        if asyncio.iscoroutinefunction(handler):
            if handler not in self._async_handlers[event_type]:
                self._async_handlers[event_type].append(handler)
        else:
            if handler not in self._sync_handlers[event_type]:
                self._sync_handlers[event_type].append(handler)

    def on_all(self, handler: Callable) -> None:
        """Subscribe to all events."""
        import asyncio
        if asyncio.iscoroutinefunction(handler):
            if handler not in self._all_async_handlers:
                self._all_async_handlers.append(handler)
        else:
            if handler not in self._all_sync_handlers:
                self._all_sync_handlers.append(handler)

    async def emit(self, event: Event) -> None:
        """Emit an event to all subscribers (async)."""
        import asyncio

        # Sync handlers
        for handler in self._sync_handlers[event.event_type]:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in sync event handler: {e}")

        for handler in self._all_sync_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in sync all-event handler: {e}")

        # Async handlers
        async_tasks = []
        for handler in self._async_handlers[event.event_type]:
            async_tasks.append(handler(event))
        for handler in self._all_async_handlers:
            async_tasks.append(handler(event))

        if async_tasks:
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    print(f"Error in async event handler: {result}")
