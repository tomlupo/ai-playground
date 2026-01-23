"""Enumerations for the trading framework."""

from enum import Enum, auto


class OrderType(Enum):
    """Order type classification."""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()
    TRAILING_STOP = auto()
    TAKE_PROFIT = auto()
    TAKE_PROFIT_LIMIT = auto()


class OrderSide(Enum):
    """Order side (direction)."""
    BUY = auto()
    SELL = auto()


class OrderStatus(Enum):
    """Order lifecycle status."""
    PENDING = auto()      # Created but not yet submitted
    SUBMITTED = auto()    # Sent to exchange
    ACCEPTED = auto()     # Acknowledged by exchange
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


class PositionSide(Enum):
    """Position direction."""
    LONG = auto()
    SHORT = auto()
    FLAT = auto()


class TimeInForce(Enum):
    """Order time-in-force policies."""
    GTC = auto()  # Good Till Cancelled
    IOC = auto()  # Immediate Or Cancel
    FOK = auto()  # Fill Or Kill
    GTD = auto()  # Good Till Date
    DAY = auto()  # Day order (expires at market close)
    OPG = auto()  # At the Opening
    CLS = auto()  # At the Close


class AssetClass(Enum):
    """Asset class classification."""
    CRYPTO = auto()
    EQUITY = auto()
    FUTURES = auto()
    FOREX = auto()
    OPTIONS = auto()
    BOND = auto()
    ETF = auto()
    INDEX = auto()


class FillType(Enum):
    """How an order was filled."""
    COMPLETE = auto()
    PARTIAL = auto()


class ExchangeType(Enum):
    """Exchange/broker type."""
    BINANCE = auto()
    BINANCE_FUTURES = auto()
    INTERACTIVE_BROKERS = auto()
    OANDA = auto()
    SIMULATED = auto()


class MarketStatus(Enum):
    """Market trading status."""
    PRE_MARKET = auto()
    OPEN = auto()
    POST_MARKET = auto()
    CLOSED = auto()
    HALTED = auto()


class DataResolution(Enum):
    """Data bar resolution/timeframe."""
    TICK = auto()
    SECOND_1 = auto()
    SECOND_5 = auto()
    SECOND_15 = auto()
    SECOND_30 = auto()
    MINUTE_1 = auto()
    MINUTE_5 = auto()
    MINUTE_15 = auto()
    MINUTE_30 = auto()
    HOUR_1 = auto()
    HOUR_4 = auto()
    DAY_1 = auto()
    WEEK_1 = auto()
    MONTH_1 = auto()

    def to_seconds(self) -> int:
        """Convert resolution to seconds."""
        mapping = {
            DataResolution.TICK: 0,
            DataResolution.SECOND_1: 1,
            DataResolution.SECOND_5: 5,
            DataResolution.SECOND_15: 15,
            DataResolution.SECOND_30: 30,
            DataResolution.MINUTE_1: 60,
            DataResolution.MINUTE_5: 300,
            DataResolution.MINUTE_15: 900,
            DataResolution.MINUTE_30: 1800,
            DataResolution.HOUR_1: 3600,
            DataResolution.HOUR_4: 14400,
            DataResolution.DAY_1: 86400,
            DataResolution.WEEK_1: 604800,
            DataResolution.MONTH_1: 2592000,
        }
        return mapping[self]
