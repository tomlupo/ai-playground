-- Feature engineering for quantitative analysis
-- Generates technical indicators and features from price data

-- Step 1: Price features with technical indicators
CREATE OR REPLACE TABLE price_features AS
SELECT
    symbol,
    date,
    timestamp,
    open,
    high,
    low,
    close,
    volume,

    -- Returns at various horizons
    (close / LAG(close, 1) OVER w) - 1 AS returns_1d,
    (close / LAG(close, 5) OVER w) - 1 AS returns_5d,
    (close / LAG(close, 21) OVER w) - 1 AS returns_21d,

    -- Moving averages
    AVG(close) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS sma_5,
    AVG(close) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_20,
    AVG(close) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS sma_50,

    -- Volatility (rolling std of returns)
    STDDEV(close / LAG(close, 1) OVER w - 1) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS volatility_20d,

    -- Volume features
    AVG(volume) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS avg_volume_20d,
    volume / NULLIF(AVG(volume) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) AS volume_ratio,

    -- Price momentum
    close / NULLIF(AVG(close) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) AS price_to_sma20,

    -- Range and spread
    (high - low) / NULLIF(open, 0) AS intraday_range,
    (close - open) / NULLIF(open, 0) AS close_to_open,

    -- Rolling high/low
    MAX(high) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 51 PRECEDING AND CURRENT ROW) AS high_52w,
    MIN(low) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 51 PRECEDING AND CURRENT ROW) AS low_52w

FROM v_prices
WINDOW w AS (PARTITION BY symbol ORDER BY timestamp);

-- Step 2: Cross-sectional features (relative to universe)
CREATE OR REPLACE TABLE cross_sectional_features AS
WITH daily_stats AS (
    SELECT
        date,
        AVG(returns_1d) AS market_return,
        STDDEV(returns_1d) AS market_vol,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY volume_ratio) AS median_volume_ratio
    FROM price_features
    GROUP BY date
)
SELECT
    p.symbol,
    p.date,
    p.returns_1d,
    p.volatility_20d,
    p.volume_ratio,

    -- Relative to market
    p.returns_1d - s.market_return AS excess_return,
    p.returns_1d / NULLIF(s.market_vol, 0) AS z_score_return,

    -- Percentile ranks
    PERCENT_RANK() OVER (PARTITION BY p.date ORDER BY p.returns_1d) AS return_percentile,
    PERCENT_RANK() OVER (PARTITION BY p.date ORDER BY p.volatility_20d) AS vol_percentile,
    PERCENT_RANK() OVER (PARTITION BY p.date ORDER BY p.volume_ratio) AS volume_percentile

FROM price_features p
LEFT JOIN daily_stats s ON p.date = s.date;

-- Step 3: Lagged features for ML (avoid lookahead)
CREATE OR REPLACE TABLE ml_features AS
SELECT
    symbol,
    date,

    -- Target: forward returns (what we're predicting)
    LEAD(returns_1d, 1) OVER w AS target_1d,
    LEAD(returns_1d, 5) OVER w AS target_5d,

    -- Features: all lagged by 1 day to avoid lookahead
    LAG(returns_1d, 1) OVER w AS lag1_return,
    LAG(returns_5d, 1) OVER w AS lag1_return_5d,
    LAG(returns_21d, 1) OVER w AS lag1_return_21d,
    LAG(volatility_20d, 1) OVER w AS lag1_volatility,
    LAG(volume_ratio, 1) OVER w AS lag1_volume_ratio,
    LAG(price_to_sma20, 1) OVER w AS lag1_price_to_sma,
    LAG(intraday_range, 1) OVER w AS lag1_range

FROM price_features
WINDOW w AS (PARTITION BY symbol ORDER BY date);
