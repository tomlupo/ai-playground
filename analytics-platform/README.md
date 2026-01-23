# Local Analytical Data Platform

A hybrid analytics stack using **Parquet**, **DuckDB**, and **PostgreSQL** for flexible, fast, and reproducible analytics.

## Architecture

```
          ┌────────────┐
          │ PostgreSQL │  (source / integration)
          └──────┬─────┘
                 │ (duckdb postgres extension)
                 ▼
┌────────────┐  ┌──────────────────┐  ┌───────────────┐
│  Parquet   │◄─►│   DuckDB (file)  │◄─►│   Python API  │
│ (data lake)│   │  analytics core │   │  pipelines    │
└────────────┘  └──────────────────┘  └───────────────┘
```

**Design Principle:**
- Parquet is the **truth** (immutable, append-only storage)
- DuckDB is the **brain** (fast analytical queries)
- PostgreSQL is the **nervous system** (transactional integration)

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Initialize
make init

# Check status
make status
```

## Project Structure

```
analytics-platform/
├── data/                    # Data lake (Parquet)
│   ├── bronze/              # Raw data
│   │   ├── trades/
│   │   ├── prices/
│   │   └── orders/
│   ├── silver/              # Curated data
│   │   ├── positions/
│   │   └── features/
│   └── gold/                # Aggregated/exported
│       ├── pnl/
│       └── reports/
├── warehouse/
│   └── warehouse.duckdb     # DuckDB database
├── sql/
│   ├── views/               # View definitions
│   ├── materializations/    # Table materializations
│   └── migrations/          # Schema migrations
├── src/
│   ├── backends/            # Storage backends
│   ├── pipelines/           # ETL pipelines
│   ├── schemas/             # Data schemas
│   └── cli.py               # CLI entry point
└── tests/
```

## Usage

### 1. Ingest Data

```python
from src.pipelines import IngestionPipeline
from src.schemas import TradeSchema
from src.config import PlatformConfig

config = PlatformConfig.default()
pipeline = IngestionPipeline(
    name="trades",
    config=config,
    target_table="trades",
    schema=TradeSchema,
    partition_cols=["date"]
)

# From PyArrow table
result = pipeline.ingest_arrow(data)

# From pandas DataFrame
result = pipeline.ingest_dataframe(df)

# From CSV
result = pipeline.ingest_from_csv("path/to/trades.csv")
```

### 2. Query with DuckDB

```python
from src.backends import DuckDBBackend
from src.config import DuckDBConfig

config = DuckDBConfig()
db = DuckDBBackend(config)

# Query Parquet directly
result = db.read_parquet("data/bronze/trades/**/*.parquet")

# Create view over Parquet
db.create_parquet_view("v_trades", "data/bronze/trades/**/*.parquet")

# Run SQL
df = db.query_df("""
    SELECT symbol, SUM(quantity) as total
    FROM v_trades
    GROUP BY symbol
""")
```

### 3. Transform Data

```python
from src.pipelines import TransformationPipeline

transform = TransformationPipeline("curate", config)

# Create views
transform.create_parquet_view("trades", "v_trades_raw")

# SQL transformation -> DuckDB table
transform.transform_sql(
    query="SELECT * FROM v_trades_raw WHERE quantity > 0",
    target_table="trades_curated",
    target_type="duckdb"
)

# SQL transformation -> Parquet
transform.transform_sql(
    query="SELECT symbol, date, SUM(quantity) as total FROM v_trades_raw GROUP BY 1, 2",
    target_table="daily_volume",
    target_type="parquet",
    layer="silver",
    partition_cols=["date"]
)
```

### 4. Export Data

```python
from src.pipelines import ExportPipeline

export = ExportPipeline("export", config)

# Export to Parquet (gold layer)
export.export_to_parquet(
    query="SELECT * FROM pnl_summary",
    target_table="daily_pnl",
    layer="gold"
)

# Export to PostgreSQL
export.export_to_postgres_direct(
    query="SELECT * FROM positions_current",
    target_table="positions"
)
```

## CLI Commands

```bash
# Initialize platform
analytics init

# Run SQL script
analytics run sql/views/bronze_views.sql -v data_path=data

# Execute query
analytics query "SELECT * FROM trades LIMIT 10"

# Show status
analytics status

# List tables
analytics tables

# Describe table
analytics describe trades

# Vacuum database
analytics vacuum
```

## Make Targets

```bash
make init              # Initialize platform
make views             # Create all views
make materialize       # Run all materializations
make refresh           # Full refresh (views + materialize)
make export            # Export to gold layer
make status            # Show platform status
make shell             # Interactive DuckDB shell
make test              # Run tests
```

## Data Flow

```
1. INGESTION
   Market/system data → Parquet (bronze, partitioned)
   External DBs → DuckDB via Postgres scan

2. TRANSFORMATION
   DuckDB reads Parquet + Postgres
   Builds curated tables: positions, pnl, features

3. SERVING
   Fast analytics from DuckDB
   Export to Parquet (archive) or PostgreSQL (APIs)
```

## Schemas

Pre-defined schemas for common financial data:

- **TradeSchema** - Trade records
- **OrderSchema** - Order records
- **FillSchema** - Fill/execution records
- **PriceSchema** - OHLCV price bars
- **TickSchema** - Tick data
- **QuoteSchema** - Quote/BBO data
- **PositionSchema** - Position snapshots
- **PnLSchema** - P&L records
- **PortfolioSnapshotSchema** - Portfolio metrics

## Configuration

```python
from src.config import PlatformConfig

# Default config
config = PlatformConfig.default()

# From file
config = PlatformConfig.from_file("config/platform.json")

# Environment variables
# POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
```

## Example Use Cases

### Quant Research
- Raw ticks in Parquet (bronze)
- Features in DuckDB (silver)
- Results exported to Parquet snapshots (gold)

### Trading System
- Orders & fills in PostgreSQL
- PnL and risk in DuckDB
- History archived to Parquet

### Personal Data Lake
- Raw data in Parquet
- Analytics in DuckDB
- Apps connect to PostgreSQL

## Requirements

- Python 3.10+
- DuckDB 1.0+
- PyArrow 15.0+
- PostgreSQL (optional)

## License

MIT
