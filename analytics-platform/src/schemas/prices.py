"""Price and market data schema definitions."""

import pyarrow as pa
from .base import BaseSchema, SchemaField


class PriceSchema(BaseSchema):
    """
    Schema for OHLCV price bars.

    Standard candlestick/bar data for any timeframe.
    """

    symbol = SchemaField(
        name="symbol",
        dtype=pa.string(),
        nullable=False,
        description="Trading symbol/ticker",
    )
    exchange = SchemaField(
        name="exchange",
        dtype=pa.string(),
        nullable=True,
        description="Exchange or data source",
    )

    # OHLCV
    open = SchemaField(
        name="open",
        dtype=pa.float64(),
        nullable=False,
        description="Opening price",
    )
    high = SchemaField(
        name="high",
        dtype=pa.float64(),
        nullable=False,
        description="High price",
    )
    low = SchemaField(
        name="low",
        dtype=pa.float64(),
        nullable=False,
        description="Low price",
    )
    close = SchemaField(
        name="close",
        dtype=pa.float64(),
        nullable=False,
        description="Closing price",
    )
    volume = SchemaField(
        name="volume",
        dtype=pa.float64(),
        nullable=True,
        description="Trading volume",
    )
    vwap = SchemaField(
        name="vwap",
        dtype=pa.float64(),
        nullable=True,
        description="Volume-weighted average price",
    )
    trades = SchemaField(
        name="trades",
        dtype=pa.int64(),
        nullable=True,
        description="Number of trades in bar",
    )

    # Timestamps
    timestamp = SchemaField(
        name="timestamp",
        dtype=pa.timestamp("us"),
        nullable=False,
        description="Bar start timestamp",
    )
    timeframe = SchemaField(
        name="timeframe",
        dtype=pa.string(),
        nullable=True,
        description="Bar timeframe: 1m, 5m, 1h, 1d, etc.",
    )

    # Derived
    returns = SchemaField(
        name="returns",
        dtype=pa.float64(),
        nullable=True,
        description="Period returns (close/prev_close - 1)",
    )

    # Partitioning
    date = SchemaField(
        name="date",
        dtype=pa.date32(),
        nullable=False,
        description="Bar date (partition key)",
    )


class TickSchema(BaseSchema):
    """
    Schema for tick data.

    Individual trade ticks from market data feeds.
    """

    symbol = SchemaField(
        name="symbol",
        dtype=pa.string(),
        nullable=False,
        description="Trading symbol",
    )
    exchange = SchemaField(
        name="exchange",
        dtype=pa.string(),
        nullable=True,
        description="Exchange",
    )

    price = SchemaField(
        name="price",
        dtype=pa.float64(),
        nullable=False,
        description="Trade price",
    )
    size = SchemaField(
        name="size",
        dtype=pa.float64(),
        nullable=False,
        description="Trade size",
    )
    side = SchemaField(
        name="side",
        dtype=pa.string(),
        nullable=True,
        description="Aggressor side",
    )

    timestamp = SchemaField(
        name="timestamp",
        dtype=pa.timestamp("us"),
        nullable=False,
        description="Tick timestamp",
    )
    sequence = SchemaField(
        name="sequence",
        dtype=pa.int64(),
        nullable=True,
        description="Sequence number",
    )

    date = SchemaField(
        name="date",
        dtype=pa.date32(),
        nullable=False,
        description="Tick date (partition key)",
    )


class QuoteSchema(BaseSchema):
    """
    Schema for quote/BBO data.

    Best bid/offer snapshots.
    """

    symbol = SchemaField(
        name="symbol",
        dtype=pa.string(),
        nullable=False,
        description="Trading symbol",
    )
    exchange = SchemaField(
        name="exchange",
        dtype=pa.string(),
        nullable=True,
        description="Exchange",
    )

    bid_price = SchemaField(
        name="bid_price",
        dtype=pa.float64(),
        nullable=True,
        description="Best bid price",
    )
    bid_size = SchemaField(
        name="bid_size",
        dtype=pa.float64(),
        nullable=True,
        description="Best bid size",
    )
    ask_price = SchemaField(
        name="ask_price",
        dtype=pa.float64(),
        nullable=True,
        description="Best ask price",
    )
    ask_size = SchemaField(
        name="ask_size",
        dtype=pa.float64(),
        nullable=True,
        description="Best ask size",
    )
    mid_price = SchemaField(
        name="mid_price",
        dtype=pa.float64(),
        nullable=True,
        description="Mid price",
    )
    spread = SchemaField(
        name="spread",
        dtype=pa.float64(),
        nullable=True,
        description="Bid-ask spread",
    )

    timestamp = SchemaField(
        name="timestamp",
        dtype=pa.timestamp("us"),
        nullable=False,
        description="Quote timestamp",
    )

    date = SchemaField(
        name="date",
        dtype=pa.date32(),
        nullable=False,
        description="Quote date (partition key)",
    )


class ReferenceDataSchema(BaseSchema):
    """
    Schema for instrument reference data.

    Static data about tradeable instruments.
    """

    symbol = SchemaField(
        name="symbol",
        dtype=pa.string(),
        nullable=False,
        description="Trading symbol",
    )
    name = SchemaField(
        name="name",
        dtype=pa.string(),
        nullable=True,
        description="Instrument name",
    )
    asset_class = SchemaField(
        name="asset_class",
        dtype=pa.string(),
        nullable=True,
        description="Asset class",
    )
    exchange = SchemaField(
        name="exchange",
        dtype=pa.string(),
        nullable=True,
        description="Primary exchange",
    )
    currency = SchemaField(
        name="currency",
        dtype=pa.string(),
        nullable=True,
        description="Quote currency",
    )

    # Contract specs
    tick_size = SchemaField(
        name="tick_size",
        dtype=pa.float64(),
        nullable=True,
        description="Minimum price increment",
    )
    lot_size = SchemaField(
        name="lot_size",
        dtype=pa.float64(),
        nullable=True,
        description="Minimum trade size",
    )
    contract_size = SchemaField(
        name="contract_size",
        dtype=pa.float64(),
        nullable=True,
        description="Contract multiplier",
    )

    # Status
    tradeable = SchemaField(
        name="tradeable",
        dtype=pa.bool_(),
        nullable=True,
        description="Is currently tradeable",
    )

    # Metadata
    updated_at = SchemaField(
        name="updated_at",
        dtype=pa.timestamp("us"),
        nullable=True,
        description="Last update timestamp",
    )
