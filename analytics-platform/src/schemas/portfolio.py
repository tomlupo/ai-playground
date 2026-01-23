"""Portfolio and P&L schema definitions."""

import pyarrow as pa
from .base import BaseSchema, SchemaField


class PositionSchema(BaseSchema):
    """
    Schema for position snapshots.

    Point-in-time position records.
    """

    # Identifiers
    position_id = SchemaField(
        name="position_id",
        dtype=pa.string(),
        nullable=True,
        description="Unique position identifier",
    )
    symbol = SchemaField(
        name="symbol",
        dtype=pa.string(),
        nullable=False,
        description="Trading symbol",
    )
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

    # Position details
    quantity = SchemaField(
        name="quantity",
        dtype=pa.float64(),
        nullable=False,
        description="Position quantity (+ long, - short)",
    )
    avg_cost = SchemaField(
        name="avg_cost",
        dtype=pa.float64(),
        nullable=True,
        description="Average cost basis",
    )
    market_price = SchemaField(
        name="market_price",
        dtype=pa.float64(),
        nullable=True,
        description="Current market price",
    )
    market_value = SchemaField(
        name="market_value",
        dtype=pa.float64(),
        nullable=True,
        description="Current market value",
    )
    cost_basis = SchemaField(
        name="cost_basis",
        dtype=pa.float64(),
        nullable=True,
        description="Total cost basis",
    )

    # P&L
    unrealized_pnl = SchemaField(
        name="unrealized_pnl",
        dtype=pa.float64(),
        nullable=True,
        description="Unrealized P&L",
    )
    realized_pnl = SchemaField(
        name="realized_pnl",
        dtype=pa.float64(),
        nullable=True,
        description="Realized P&L (cumulative)",
    )

    # Risk
    notional = SchemaField(
        name="notional",
        dtype=pa.float64(),
        nullable=True,
        description="Notional exposure",
    )
    weight = SchemaField(
        name="weight",
        dtype=pa.float64(),
        nullable=True,
        description="Portfolio weight",
    )

    # Timestamps
    as_of = SchemaField(
        name="as_of",
        dtype=pa.timestamp("us"),
        nullable=False,
        description="Position snapshot timestamp",
    )
    opened_at = SchemaField(
        name="opened_at",
        dtype=pa.timestamp("us"),
        nullable=True,
        description="Position open timestamp",
    )

    # Partitioning
    date = SchemaField(
        name="date",
        dtype=pa.date32(),
        nullable=False,
        description="Position date (partition key)",
    )


class PnLSchema(BaseSchema):
    """
    Schema for P&L records.

    Daily/periodic P&L by various dimensions.
    """

    # Dimensions
    date = SchemaField(
        name="date",
        dtype=pa.date32(),
        nullable=False,
        description="P&L date",
    )
    symbol = SchemaField(
        name="symbol",
        dtype=pa.string(),
        nullable=True,
        description="Symbol (null for portfolio level)",
    )
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
    asset_class = SchemaField(
        name="asset_class",
        dtype=pa.string(),
        nullable=True,
        description="Asset class",
    )

    # P&L breakdown
    gross_pnl = SchemaField(
        name="gross_pnl",
        dtype=pa.float64(),
        nullable=False,
        description="Gross P&L (before costs)",
    )
    trading_pnl = SchemaField(
        name="trading_pnl",
        dtype=pa.float64(),
        nullable=True,
        description="P&L from trading activity",
    )
    position_pnl = SchemaField(
        name="position_pnl",
        dtype=pa.float64(),
        nullable=True,
        description="P&L from position mark-to-market",
    )

    # Costs
    commissions = SchemaField(
        name="commissions",
        dtype=pa.float64(),
        nullable=True,
        description="Commission costs",
    )
    financing = SchemaField(
        name="financing",
        dtype=pa.float64(),
        nullable=True,
        description="Financing/borrow costs",
    )
    slippage = SchemaField(
        name="slippage",
        dtype=pa.float64(),
        nullable=True,
        description="Estimated slippage cost",
    )

    # Net P&L
    net_pnl = SchemaField(
        name="net_pnl",
        dtype=pa.float64(),
        nullable=False,
        description="Net P&L (after costs)",
    )

    # Cumulative
    cumulative_pnl = SchemaField(
        name="cumulative_pnl",
        dtype=pa.float64(),
        nullable=True,
        description="Cumulative P&L",
    )

    # Returns
    returns = SchemaField(
        name="returns",
        dtype=pa.float64(),
        nullable=True,
        description="Daily returns",
    )
    cumulative_returns = SchemaField(
        name="cumulative_returns",
        dtype=pa.float64(),
        nullable=True,
        description="Cumulative returns",
    )


class PortfolioSnapshotSchema(BaseSchema):
    """
    Schema for portfolio-level snapshots.

    Aggregate portfolio metrics over time.
    """

    date = SchemaField(
        name="date",
        dtype=pa.date32(),
        nullable=False,
        description="Snapshot date",
    )
    account = SchemaField(
        name="account",
        dtype=pa.string(),
        nullable=True,
        description="Trading account",
    )
    as_of = SchemaField(
        name="as_of",
        dtype=pa.timestamp("us"),
        nullable=False,
        description="Snapshot timestamp",
    )

    # NAV
    nav = SchemaField(
        name="nav",
        dtype=pa.float64(),
        nullable=False,
        description="Net asset value",
    )
    cash = SchemaField(
        name="cash",
        dtype=pa.float64(),
        nullable=True,
        description="Cash balance",
    )
    securities_value = SchemaField(
        name="securities_value",
        dtype=pa.float64(),
        nullable=True,
        description="Securities market value",
    )

    # Exposure
    gross_exposure = SchemaField(
        name="gross_exposure",
        dtype=pa.float64(),
        nullable=True,
        description="Gross exposure (|long| + |short|)",
    )
    net_exposure = SchemaField(
        name="net_exposure",
        dtype=pa.float64(),
        nullable=True,
        description="Net exposure (long - short)",
    )
    long_exposure = SchemaField(
        name="long_exposure",
        dtype=pa.float64(),
        nullable=True,
        description="Long exposure",
    )
    short_exposure = SchemaField(
        name="short_exposure",
        dtype=pa.float64(),
        nullable=True,
        description="Short exposure",
    )

    # Counts
    num_positions = SchemaField(
        name="num_positions",
        dtype=pa.int64(),
        nullable=True,
        description="Number of positions",
    )
    num_long = SchemaField(
        name="num_long",
        dtype=pa.int64(),
        nullable=True,
        description="Number of long positions",
    )
    num_short = SchemaField(
        name="num_short",
        dtype=pa.int64(),
        nullable=True,
        description="Number of short positions",
    )

    # P&L (daily)
    daily_pnl = SchemaField(
        name="daily_pnl",
        dtype=pa.float64(),
        nullable=True,
        description="Daily P&L",
    )
    daily_returns = SchemaField(
        name="daily_returns",
        dtype=pa.float64(),
        nullable=True,
        description="Daily returns",
    )

    # Risk metrics
    volatility = SchemaField(
        name="volatility",
        dtype=pa.float64(),
        nullable=True,
        description="Rolling volatility",
    )
    sharpe = SchemaField(
        name="sharpe",
        dtype=pa.float64(),
        nullable=True,
        description="Rolling Sharpe ratio",
    )
    max_drawdown = SchemaField(
        name="max_drawdown",
        dtype=pa.float64(),
        nullable=True,
        description="Maximum drawdown",
    )


class AttributionSchema(BaseSchema):
    """
    Schema for P&L attribution.

    Factor-based P&L decomposition.
    """

    date = SchemaField(
        name="date",
        dtype=pa.date32(),
        nullable=False,
        description="Attribution date",
    )
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

    # Total
    total_pnl = SchemaField(
        name="total_pnl",
        dtype=pa.float64(),
        nullable=False,
        description="Total P&L",
    )

    # Factor contributions
    market_pnl = SchemaField(
        name="market_pnl",
        dtype=pa.float64(),
        nullable=True,
        description="Market factor contribution",
    )
    sector_pnl = SchemaField(
        name="sector_pnl",
        dtype=pa.float64(),
        nullable=True,
        description="Sector factor contribution",
    )
    style_pnl = SchemaField(
        name="style_pnl",
        dtype=pa.float64(),
        nullable=True,
        description="Style factor contribution",
    )
    alpha_pnl = SchemaField(
        name="alpha_pnl",
        dtype=pa.float64(),
        nullable=True,
        description="Alpha/residual P&L",
    )

    # Timing
    timing_pnl = SchemaField(
        name="timing_pnl",
        dtype=pa.float64(),
        nullable=True,
        description="Timing contribution",
    )
    selection_pnl = SchemaField(
        name="selection_pnl",
        dtype=pa.float64(),
        nullable=True,
        description="Selection contribution",
    )
