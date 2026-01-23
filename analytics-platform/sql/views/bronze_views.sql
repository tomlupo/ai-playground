-- Bronze layer views
-- Create views over raw Parquet data for easy querying

-- Trades view over Parquet
CREATE OR REPLACE VIEW v_trades_raw AS
SELECT * FROM read_parquet('{{data_path}}/bronze/trades/**/*.parquet', hive_partitioning=true);

-- Prices view over Parquet
CREATE OR REPLACE VIEW v_prices_raw AS
SELECT * FROM read_parquet('{{data_path}}/bronze/prices/**/*.parquet', hive_partitioning=true);

-- Orders view over Parquet
CREATE OR REPLACE VIEW v_orders_raw AS
SELECT * FROM read_parquet('{{data_path}}/bronze/orders/**/*.parquet', hive_partitioning=true);

-- Fills view over Parquet
CREATE OR REPLACE VIEW v_fills_raw AS
SELECT * FROM read_parquet('{{data_path}}/bronze/fills/**/*.parquet', hive_partitioning=true);
