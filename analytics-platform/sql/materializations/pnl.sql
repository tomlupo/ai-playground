-- P&L calculation
-- Calculates daily and cumulative P&L from trades and positions

-- Step 1: Daily trading P&L (realized)
CREATE OR REPLACE TABLE _daily_trading_pnl AS
SELECT
    date,
    symbol,
    account,
    strategy,
    SUM(CASE
        WHEN side = 'sell' THEN quantity * price
        ELSE -quantity * price
    END) AS trading_pnl,
    SUM(COALESCE(commission, 0)) AS commissions,
    SUM(COALESCE(slippage, 0)) AS slippage_cost
FROM v_trades
GROUP BY date, symbol, account, strategy;

-- Step 2: Daily position P&L (mark-to-market)
CREATE OR REPLACE TABLE _daily_position_pnl AS
WITH position_eod AS (
    SELECT
        p.symbol,
        p.account,
        p.strategy,
        p.date,
        p.quantity,
        p.avg_cost,
        pr.close AS eod_price,
        LAG(pr.close) OVER (PARTITION BY p.symbol, p.account, p.strategy ORDER BY p.date) AS prev_close
    FROM positions p
    LEFT JOIN v_daily_prices pr ON p.symbol = pr.symbol AND p.date = pr.date
)
SELECT
    date,
    symbol,
    account,
    strategy,
    quantity * (eod_price - COALESCE(prev_close, avg_cost)) AS position_pnl
FROM position_eod
WHERE quantity != 0;

-- Step 3: Combine into daily P&L
CREATE OR REPLACE TABLE daily_pnl AS
SELECT
    COALESCE(t.date, p.date) AS date,
    COALESCE(t.symbol, p.symbol) AS symbol,
    COALESCE(t.account, p.account) AS account,
    COALESCE(t.strategy, p.strategy) AS strategy,
    COALESCE(t.trading_pnl, 0) AS trading_pnl,
    COALESCE(p.position_pnl, 0) AS position_pnl,
    COALESCE(t.trading_pnl, 0) + COALESCE(p.position_pnl, 0) AS gross_pnl,
    COALESCE(t.commissions, 0) AS commissions,
    COALESCE(t.slippage_cost, 0) AS slippage,
    0.0 AS financing,  -- Placeholder for financing costs
    COALESCE(t.trading_pnl, 0) + COALESCE(p.position_pnl, 0) - COALESCE(t.commissions, 0) - COALESCE(t.slippage_cost, 0) AS net_pnl
FROM _daily_trading_pnl t
FULL OUTER JOIN _daily_position_pnl p
    ON t.date = p.date
    AND t.symbol = p.symbol
    AND t.account = p.account
    AND t.strategy = p.strategy;

-- Step 4: Calculate cumulative P&L
CREATE OR REPLACE TABLE pnl_cumulative AS
SELECT
    date,
    symbol,
    account,
    strategy,
    trading_pnl,
    position_pnl,
    gross_pnl,
    net_pnl,
    commissions,
    slippage,
    financing,
    SUM(net_pnl) OVER (PARTITION BY symbol, account, strategy ORDER BY date) AS cumulative_pnl
FROM daily_pnl;

-- Step 5: Portfolio-level daily P&L
CREATE OR REPLACE TABLE portfolio_daily_pnl AS
SELECT
    date,
    account,
    SUM(gross_pnl) AS gross_pnl,
    SUM(net_pnl) AS net_pnl,
    SUM(commissions) AS commissions,
    SUM(slippage) AS slippage,
    SUM(financing) AS financing,
    COUNT(DISTINCT symbol) AS num_symbols
FROM daily_pnl
GROUP BY date, account;

-- Cleanup
DROP TABLE IF EXISTS _daily_trading_pnl;
DROP TABLE IF EXISTS _daily_position_pnl;
