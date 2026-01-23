-- Silver layer views
-- Curated views with business logic applied

-- Curated trades with derived fields
CREATE OR REPLACE VIEW v_trades AS
SELECT
    trade_id,
    order_id,
    symbol,
    side,
    quantity,
    price,
    COALESCE(notional, quantity * price) AS notional,
    commission,
    slippage,
    trade_time,
    date,
    account,
    strategy,
    asset_class,
    exchange,
    -- Derived fields
    CASE WHEN side = 'buy' THEN quantity ELSE -quantity END AS signed_quantity,
    CASE WHEN side = 'buy' THEN notional ELSE -notional END AS signed_notional
FROM v_trades_raw;

-- Curated prices with returns
CREATE OR REPLACE VIEW v_prices AS
SELECT
    symbol,
    timestamp,
    date,
    open,
    high,
    low,
    close,
    volume,
    vwap,
    timeframe,
    -- Calculate returns
    (close / LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp)) - 1 AS returns,
    -- Calculate intraday range
    (high - low) / NULLIF(open, 0) AS intraday_range,
    -- Volume-weighted metrics
    close * volume AS dollar_volume
FROM v_prices_raw;

-- Daily OHLCV aggregation
CREATE OR REPLACE VIEW v_daily_prices AS
SELECT
    symbol,
    date,
    FIRST(open) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close) AS close,
    SUM(volume) AS volume,
    SUM(close * volume) / NULLIF(SUM(volume), 0) AS vwap
FROM v_prices
GROUP BY symbol, date;

-- Order fill aggregation
CREATE OR REPLACE VIEW v_order_summary AS
SELECT
    o.order_id,
    o.symbol,
    o.side,
    o.order_type,
    o.quantity AS ordered_qty,
    o.limit_price,
    o.status,
    o.created_at,
    COUNT(f.fill_id) AS num_fills,
    SUM(f.quantity) AS filled_qty,
    SUM(f.quantity * f.price) / NULLIF(SUM(f.quantity), 0) AS avg_fill_price,
    MAX(f.fill_time) AS last_fill_time
FROM v_orders_raw o
LEFT JOIN v_fills_raw f ON o.order_id = f.order_id
GROUP BY o.order_id, o.symbol, o.side, o.order_type, o.quantity, o.limit_price, o.status, o.created_at;
