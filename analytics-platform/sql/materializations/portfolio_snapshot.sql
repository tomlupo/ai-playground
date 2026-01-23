-- Portfolio snapshot materialization
-- Calculates portfolio-level metrics at each point in time

-- Step 1: Position values with market prices
CREATE OR REPLACE TABLE _position_values AS
SELECT
    p.symbol,
    p.account,
    p.strategy,
    p.date,
    p.quantity,
    p.avg_cost,
    p.cost_basis,
    pr.close AS market_price,
    p.quantity * pr.close AS market_value,
    p.quantity * pr.close - p.cost_basis AS unrealized_pnl
FROM positions p
LEFT JOIN v_daily_prices pr ON p.symbol = pr.symbol AND p.date = pr.date;

-- Step 2: Account-level aggregation
CREATE OR REPLACE TABLE portfolio_snapshot AS
SELECT
    date,
    account,
    NOW() AS as_of,

    -- NAV components (assuming some initial cash)
    SUM(market_value) AS securities_value,
    SUM(market_value) AS nav,  -- Simplified: no cash tracking

    -- Exposure metrics
    SUM(CASE WHEN quantity > 0 THEN market_value ELSE 0 END) AS long_exposure,
    SUM(CASE WHEN quantity < 0 THEN ABS(market_value) ELSE 0 END) AS short_exposure,
    SUM(ABS(market_value)) AS gross_exposure,
    SUM(market_value) AS net_exposure,

    -- Position counts
    COUNT(DISTINCT symbol) AS num_positions,
    COUNT(DISTINCT CASE WHEN quantity > 0 THEN symbol END) AS num_long,
    COUNT(DISTINCT CASE WHEN quantity < 0 THEN symbol END) AS num_short,

    -- P&L
    SUM(unrealized_pnl) AS total_unrealized_pnl

FROM _position_values
WHERE quantity != 0
GROUP BY date, account;

-- Step 3: Add daily P&L and returns
CREATE OR REPLACE TABLE portfolio_snapshot_with_returns AS
SELECT
    ps.*,
    pnl.net_pnl AS daily_pnl,
    pnl.net_pnl / NULLIF(LAG(ps.nav) OVER (PARTITION BY ps.account ORDER BY ps.date), 0) AS daily_returns
FROM portfolio_snapshot ps
LEFT JOIN portfolio_daily_pnl pnl ON ps.date = pnl.date AND ps.account = pnl.account;

-- Step 4: Calculate risk metrics (rolling)
CREATE OR REPLACE TABLE portfolio_risk_metrics AS
SELECT
    date,
    account,
    nav,
    daily_returns,

    -- Rolling metrics (21-day = ~1 month)
    STDDEV(daily_returns) OVER (PARTITION BY account ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) * SQRT(252) AS volatility_ann,
    AVG(daily_returns) OVER (PARTITION BY account ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) * 252 AS return_ann,

    -- Sharpe ratio (assuming 0 risk-free rate)
    (AVG(daily_returns) OVER (PARTITION BY account ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) * 252) /
    NULLIF(STDDEV(daily_returns) OVER (PARTITION BY account ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) * SQRT(252), 0) AS sharpe_ratio,

    -- Drawdown
    nav / NULLIF(MAX(nav) OVER (PARTITION BY account ORDER BY date ROWS UNBOUNDED PRECEDING), 0) - 1 AS drawdown,
    MIN(nav / NULLIF(MAX(nav) OVER (PARTITION BY account ORDER BY date ROWS UNBOUNDED PRECEDING), 0) - 1)
        OVER (PARTITION BY account ORDER BY date ROWS UNBOUNDED PRECEDING) AS max_drawdown

FROM portfolio_snapshot_with_returns;

-- Cleanup
DROP TABLE IF EXISTS _position_values;
