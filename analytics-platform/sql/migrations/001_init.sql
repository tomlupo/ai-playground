-- Initial DuckDB schema setup
-- Run this to initialize the warehouse

-- Create dimension tables for reference data
CREATE TABLE IF NOT EXISTS dim_symbols (
    symbol VARCHAR PRIMARY KEY,
    name VARCHAR,
    asset_class VARCHAR,
    exchange VARCHAR,
    currency VARCHAR,
    tick_size DOUBLE,
    lot_size DOUBLE,
    tradeable BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create dimension table for accounts
CREATE TABLE IF NOT EXISTS dim_accounts (
    account VARCHAR PRIMARY KEY,
    name VARCHAR,
    type VARCHAR,  -- 'live', 'paper', 'backtest'
    currency VARCHAR DEFAULT 'USD',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create dimension table for strategies
CREATE TABLE IF NOT EXISTS dim_strategies (
    strategy VARCHAR PRIMARY KEY,
    name VARCHAR,
    description VARCHAR,
    asset_class VARCHAR,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create calendar table for date dimensions
CREATE TABLE IF NOT EXISTS dim_calendar AS
SELECT
    date::DATE AS date,
    EXTRACT(YEAR FROM date) AS year,
    EXTRACT(MONTH FROM date) AS month,
    EXTRACT(DAY FROM date) AS day,
    EXTRACT(DOW FROM date) AS day_of_week,
    EXTRACT(WEEK FROM date) AS week_of_year,
    EXTRACT(QUARTER FROM date) AS quarter,
    CASE WHEN EXTRACT(DOW FROM date) IN (0, 6) THEN FALSE ELSE TRUE END AS is_weekday
FROM generate_series('2020-01-01'::DATE, '2030-12-31'::DATE, INTERVAL '1 day') AS date;

-- Index on date
CREATE INDEX IF NOT EXISTS idx_calendar_date ON dim_calendar(date);
