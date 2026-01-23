"""Trade schema definitions."""

import pyarrow as pa
from .base import BaseSchema, SchemaField


class TradeSchema(BaseSchema):
    """
    Schema for trade records.

    Represents executed trades from various sources.
    """

    # Identifiers
    trade_id = SchemaField(
        name="trade_id",
        dtype=pa.string(),
        nullable=False,
        description="Unique trade identifier",
    )
    order_id = SchemaField(
        name="order_id",
        dtype=pa.string(),
        nullable=True,
        description="Parent order identifier",
    )
    external_id = SchemaField(
        name="external_id",
        dtype=pa.string(),
        nullable=True,
        description="External system trade ID",
    )

    # Instrument
    symbol = SchemaField(
        name="symbol",
        dtype=pa.string(),
        nullable=False,
        description="Trading symbol/ticker",
    )
    asset_class = SchemaField(
        name="asset_class",
        dtype=pa.string(),
        nullable=True,
        description="Asset class (equity, fx, crypto, etc.)",
    )
    exchange = SchemaField(
        name="exchange",
        dtype=pa.string(),
        nullable=True,
        description="Exchange or venue",
    )

    # Trade details
    side = SchemaField(
        name="side",
        dtype=pa.string(),
        nullable=False,
        description="Trade side: buy or sell",
    )
    quantity = SchemaField(
        name="quantity",
        dtype=pa.float64(),
        nullable=False,
        description="Trade quantity (positive)",
    )
    price = SchemaField(
        name="price",
        dtype=pa.float64(),
        nullable=False,
        description="Execution price",
    )
    notional = SchemaField(
        name="notional",
        dtype=pa.float64(),
        nullable=True,
        description="Trade notional value (quantity * price)",
    )
    currency = SchemaField(
        name="currency",
        dtype=pa.string(),
        nullable=True,
        description="Settlement currency",
    )

    # Costs
    commission = SchemaField(
        name="commission",
        dtype=pa.float64(),
        nullable=True,
        description="Commission/fee amount",
    )
    slippage = SchemaField(
        name="slippage",
        dtype=pa.float64(),
        nullable=True,
        description="Slippage vs reference price",
    )

    # Timestamps
    trade_time = SchemaField(
        name="trade_time",
        dtype=pa.timestamp("us"),
        nullable=False,
        description="Trade execution timestamp",
    )
    settle_date = SchemaField(
        name="settle_date",
        dtype=pa.date32(),
        nullable=True,
        description="Settlement date",
    )

    # Metadata
    account = SchemaField(
        name="account",
        dtype=pa.string(),
        nullable=True,
        description="Trading account",
    )
    strategy = SchemaField(
        name="strategy",
        dtype=pa.string(),
        nullable=True,
        description="Strategy identifier",
    )
    source = SchemaField(
        name="source",
        dtype=pa.string(),
        nullable=True,
        description="Data source",
    )

    # Partitioning
    date = SchemaField(
        name="date",
        dtype=pa.date32(),
        nullable=False,
        description="Trade date (partition key)",
    )


class OrderSchema(BaseSchema):
    """
    Schema for order records.

    Represents orders submitted to trading systems.
    """

    order_id = SchemaField(
        name="order_id",
        dtype=pa.string(),
        nullable=False,
        description="Unique order identifier",
    )
    parent_order_id = SchemaField(
        name="parent_order_id",
        dtype=pa.string(),
        nullable=True,
        description="Parent order ID for child orders",
    )
    external_id = SchemaField(
        name="external_id",
        dtype=pa.string(),
        nullable=True,
        description="External system order ID",
    )

    # Instrument
    symbol = SchemaField(
        name="symbol",
        dtype=pa.string(),
        nullable=False,
        description="Trading symbol/ticker",
    )

    # Order details
    side = SchemaField(
        name="side",
        dtype=pa.string(),
        nullable=False,
        description="Order side: buy or sell",
    )
    order_type = SchemaField(
        name="order_type",
        dtype=pa.string(),
        nullable=False,
        description="Order type: market, limit, stop, etc.",
    )
    quantity = SchemaField(
        name="quantity",
        dtype=pa.float64(),
        nullable=False,
        description="Order quantity",
    )
    limit_price = SchemaField(
        name="limit_price",
        dtype=pa.float64(),
        nullable=True,
        description="Limit price (for limit orders)",
    )
    stop_price = SchemaField(
        name="stop_price",
        dtype=pa.float64(),
        nullable=True,
        description="Stop price (for stop orders)",
    )
    time_in_force = SchemaField(
        name="time_in_force",
        dtype=pa.string(),
        nullable=True,
        description="Time in force: day, gtc, ioc, fok",
    )

    # Status
    status = SchemaField(
        name="status",
        dtype=pa.string(),
        nullable=False,
        description="Order status: pending, filled, partial, cancelled",
    )
    filled_qty = SchemaField(
        name="filled_qty",
        dtype=pa.float64(),
        nullable=True,
        description="Filled quantity",
    )
    avg_fill_price = SchemaField(
        name="avg_fill_price",
        dtype=pa.float64(),
        nullable=True,
        description="Average fill price",
    )

    # Timestamps
    created_at = SchemaField(
        name="created_at",
        dtype=pa.timestamp("us"),
        nullable=False,
        description="Order creation timestamp",
    )
    updated_at = SchemaField(
        name="updated_at",
        dtype=pa.timestamp("us"),
        nullable=True,
        description="Last update timestamp",
    )

    # Metadata
    account = SchemaField(
        name="account",
        dtype=pa.string(),
        nullable=True,
        description="Trading account",
    )
    strategy = SchemaField(
        name="strategy",
        dtype=pa.string(),
        nullable=True,
        description="Strategy identifier",
    )

    # Partitioning
    date = SchemaField(
        name="date",
        dtype=pa.date32(),
        nullable=False,
        description="Order date (partition key)",
    )


class FillSchema(BaseSchema):
    """
    Schema for fill/execution records.

    Individual fills that make up trades.
    """

    fill_id = SchemaField(
        name="fill_id",
        dtype=pa.string(),
        nullable=False,
        description="Unique fill identifier",
    )
    order_id = SchemaField(
        name="order_id",
        dtype=pa.string(),
        nullable=False,
        description="Parent order ID",
    )
    trade_id = SchemaField(
        name="trade_id",
        dtype=pa.string(),
        nullable=True,
        description="Aggregated trade ID",
    )

    symbol = SchemaField(
        name="symbol",
        dtype=pa.string(),
        nullable=False,
        description="Trading symbol",
    )
    side = SchemaField(
        name="side",
        dtype=pa.string(),
        nullable=False,
        description="Fill side",
    )
    quantity = SchemaField(
        name="quantity",
        dtype=pa.float64(),
        nullable=False,
        description="Fill quantity",
    )
    price = SchemaField(
        name="price",
        dtype=pa.float64(),
        nullable=False,
        description="Fill price",
    )

    fill_time = SchemaField(
        name="fill_time",
        dtype=pa.timestamp("us"),
        nullable=False,
        description="Fill timestamp",
    )
    exchange = SchemaField(
        name="exchange",
        dtype=pa.string(),
        nullable=True,
        description="Execution venue",
    )
    liquidity = SchemaField(
        name="liquidity",
        dtype=pa.string(),
        nullable=True,
        description="Liquidity indicator: maker, taker",
    )

    date = SchemaField(
        name="date",
        dtype=pa.date32(),
        nullable=False,
        description="Fill date (partition key)",
    )
