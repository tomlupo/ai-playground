"""
Generic Trading Bot Framework

A modular, exchange-agnostic trading framework supporting:
- Crypto (Binance)
- Equity (Interactive Brokers)
- Futures/Forex (OANDA)
- Paper trading with calibrated market simulation

Usage:
    from trading import TradingEngine, BinanceExchange, SimulatedExchange
    from trading.strategy import Strategy
    from trading.config import TradingConfig
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
]
