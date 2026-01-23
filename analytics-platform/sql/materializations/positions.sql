-- Position calculation from trades
-- Materializes current positions based on trade history

-- Step 1: Calculate position changes from trades
CREATE OR REPLACE TABLE _position_changes AS
SELECT
    symbol,
    account,
    strategy,
    date,
    SUM(CASE WHEN side = 'buy' THEN quantity ELSE -quantity END) AS position_delta,
    SUM(CASE WHEN side = 'buy' THEN quantity * price ELSE -quantity * price END) AS cost_delta,
    COUNT(*) AS num_trades
FROM v_trades
GROUP BY symbol, account, strategy, date;

-- Step 2: Calculate cumulative positions with running totals
CREATE OR REPLACE TABLE positions AS
SELECT
    symbol,
    account,
    strategy,
    date,
    SUM(position_delta) OVER w AS quantity,
    SUM(cost_delta) OVER w AS cost_basis,
    SUM(cost_delta) OVER w / NULLIF(SUM(position_delta) OVER w, 0) AS avg_cost,
    NOW() AS as_of
FROM _position_changes
WINDOW w AS (PARTITION BY symbol, account, strategy ORDER BY date ROWS UNBOUNDED PRECEDING);

-- Step 3: Get latest position snapshot
CREATE OR REPLACE TABLE positions_current AS
SELECT DISTINCT ON (symbol, account, strategy)
    symbol,
    account,
    strategy,
    quantity,
    cost_basis,
    avg_cost,
    date AS last_trade_date,
    as_of
FROM positions
WHERE quantity != 0
ORDER BY symbol, account, strategy, date DESC;

-- Cleanup
DROP TABLE IF EXISTS _position_changes;
