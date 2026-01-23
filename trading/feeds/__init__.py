"""
Data feeds module for live streaming and historical data management.

Provides:
- Live data feed handling with aggregation
- Historical data storage and retrieval
- Data replay for backtesting
- Multiple storage backends (SQLite, CSV, Parquet)
"""

from trading.feeds.base import (
    DataFeed,
    FeedConfig,
    FeedStatus,
    TickAggregator,
)
from trading.feeds.live import (
    LiveFeed,
    LiveFeedManager,
)
from trading.feeds.history import (
    HistoryStore,
    SQLiteHistoryStore,
    CSVHistoryStore,
    ParquetHistoryStore,
)
from trading.feeds.replay import (
    DataReplayer,
    ReplayConfig,
    BacktestFeed,
)

__all__ = [
    # Base
    "DataFeed",
    "FeedConfig",
    "FeedStatus",
    "TickAggregator",
    # Live
    "LiveFeed",
    "LiveFeedManager",
    # History
    "HistoryStore",
    "SQLiteHistoryStore",
    "CSVHistoryStore",
    "ParquetHistoryStore",
    # Replay
    "DataReplayer",
    "ReplayConfig",
    "BacktestFeed",
]
