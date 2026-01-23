"""Exchange adapters for different brokers/platforms."""

from trading.exchanges.base import Exchange, ExchangeConfig
from trading.exchanges.binance import BinanceExchange, BinanceConfig
from trading.exchanges.interactive_brokers import InteractiveBrokersExchange, IBConfig
from trading.exchanges.oanda import OandaExchange, OandaConfig
from trading.exchanges.bossa import BossaExchange, BossaConfig

__all__ = [
    "Exchange",
    "ExchangeConfig",
    "BinanceExchange",
    "BinanceConfig",
    "InteractiveBrokersExchange",
    "IBConfig",
    "OandaExchange",
    "OandaConfig",
    "BossaExchange",
    "BossaConfig",
]
