"""
Generic Trading Bot Framework

A modular, exchange-agnostic trading framework supporting:
- Crypto (Binance)
- Equity (Interactive Brokers)
- Futures/Forex (OANDA)
- Paper trading with calibrated market simulation
- Live data feeds with history storage
- Historical data replay for backtesting

Usage:
    from trading import TradingEngine, BinanceExchange, SimulatedExchange
    from trading.strategy import Strategy
    from trading.config import TradingConfig
    from trading.feeds import LiveFeed, HistoryStore, BacktestFeed
"""

from trading.core.models import (
    Order,
    Position,
    Trade,
    OHLCV,
    Tick,
    OrderBook,
    Balance,
    Instrument,
)
from trading.core.enums import (
    OrderType,
    OrderSide,
    OrderStatus,
    PositionSide,
    TimeInForce,
    AssetClass,
    DataResolution,
)
from trading.engine import TradingEngine
from trading.exchanges import (
    Exchange,
    BinanceExchange,
    InteractiveBrokersExchange,
    OandaExchange,
)
from trading.simulator import SimulatedExchange, MarketSimulator
from trading.strategy import Strategy, RiskManager
from trading.config import TradingConfig
from trading.feeds import (
    LiveFeed,
    LiveFeedManager,
    HistoryStore,
    SQLiteHistoryStore,
    DataReplayer,
    BacktestFeed,
)

__version__ = "0.1.0"
__all__ = [
    # Core models
    "Order",
    "Position",
    "Trade",
    "OHLCV",
    "Tick",
    "OrderBook",
    "Balance",
    "Instrument",
    # Enums
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "PositionSide",
    "TimeInForce",
    "AssetClass",
    "DataResolution",
    # Engine
    "TradingEngine",
    # Exchanges
    "Exchange",
    "BinanceExchange",
    "InteractiveBrokersExchange",
    "OandaExchange",
    # Simulator
    "SimulatedExchange",
    "MarketSimulator",
    # Strategy
    "Strategy",
    "RiskManager",
    # Config
    "TradingConfig",
    # Feeds
    "LiveFeed",
    "LiveFeedManager",
    "HistoryStore",
    "SQLiteHistoryStore",
    "DataReplayer",
    "BacktestFeed",
]
