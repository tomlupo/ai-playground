# Analytics Platform Makefile
# Local analytical data stack orchestration

.PHONY: all init clean test lint format help
.PHONY: views materialize refresh export
.PHONY: ingest-trades ingest-prices

# Configuration
PYTHON := python
DATA_PATH := data
SQL_PATH := sql
WAREHOUSE := warehouse/warehouse.duckdb

# Default target
all: help

#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

## Initialize the platform (create directories, setup DuckDB)
init:
	@echo "Initializing Analytics Platform..."
	@mkdir -p $(DATA_PATH)/bronze/{trades,prices,orders,fills}
	@mkdir -p $(DATA_PATH)/silver/{positions,features}
	@mkdir -p $(DATA_PATH)/gold/{pnl,reports}
	@mkdir -p warehouse
	@mkdir -p config
	$(PYTHON) -c "from src.cli import main; main(['init'])"
	@echo "Done!"

## Install dependencies
install:
	pip install -e ".[all]"

## Install dev dependencies only
install-dev:
	pip install -e ".[dev]"

## Clean generated files
clean:
	rm -rf $(WAREHOUSE)
	rm -rf $(DATA_PATH)/silver/*
	rm -rf $(DATA_PATH)/gold/*
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

## Full reset (clean + bronze data)
reset: clean
	rm -rf $(DATA_PATH)/bronze/*
	@echo "Full reset complete. Run 'make init' to reinitialize."

#------------------------------------------------------------------------------
# Views and Materializations
#------------------------------------------------------------------------------

## Create bronze layer views over Parquet
views-bronze:
	@echo "Creating bronze views..."
	$(PYTHON) -m src.cli run $(SQL_PATH)/views/bronze_views.sql -v data_path=$(DATA_PATH)

## Create silver layer views
views-silver:
	@echo "Creating silver views..."
	$(PYTHON) -m src.cli run $(SQL_PATH)/views/silver_views.sql -v data_path=$(DATA_PATH)

## Create all views
views: views-bronze views-silver

## Materialize positions
materialize-positions:
	@echo "Materializing positions..."
	$(PYTHON) -m src.cli run $(SQL_PATH)/materializations/positions.sql

## Materialize P&L
materialize-pnl:
	@echo "Materializing P&L..."
	$(PYTHON) -m src.cli run $(SQL_PATH)/materializations/pnl.sql

## Materialize features
materialize-features:
	@echo "Materializing features..."
	$(PYTHON) -m src.cli run $(SQL_PATH)/materializations/features.sql

## Materialize portfolio snapshot
materialize-portfolio:
	@echo "Materializing portfolio snapshot..."
	$(PYTHON) -m src.cli run $(SQL_PATH)/materializations/portfolio_snapshot.sql

## Run all materializations
materialize: materialize-positions materialize-pnl materialize-features materialize-portfolio

## Full refresh (views + materializations)
refresh: views materialize
	@echo "Full refresh complete!"

#------------------------------------------------------------------------------
# Data Pipeline Shortcuts
#------------------------------------------------------------------------------

## Example: Ingest trades from CSV
ingest-trades:
	@echo "Ingesting trades..."
	$(PYTHON) -c "\
from src.pipelines import IngestionPipeline; \
from src.schemas import TradeSchema; \
from src.config import PlatformConfig; \
config = PlatformConfig.default(); \
pipeline = IngestionPipeline('trades', config, 'trades', TradeSchema, ['date']); \
result = pipeline.ingest_from_csv('input/trades.csv'); \
print(result.to_dict())"

## Example: Ingest prices from CSV
ingest-prices:
	@echo "Ingesting prices..."
	$(PYTHON) -c "\
from src.pipelines import IngestionPipeline; \
from src.schemas import PriceSchema; \
from src.config import PlatformConfig; \
config = PlatformConfig.default(); \
pipeline = IngestionPipeline('prices', config, 'prices', PriceSchema, ['date']); \
result = pipeline.ingest_from_csv('input/prices.csv'); \
print(result.to_dict())"

#------------------------------------------------------------------------------
# Export
#------------------------------------------------------------------------------

## Export P&L to gold layer Parquet
export-pnl:
	@echo "Exporting P&L to gold..."
	$(PYTHON) -c "\
from src.pipelines import ExportPipeline; \
from src.config import PlatformConfig; \
config = PlatformConfig.default(); \
pipeline = ExportPipeline('export-pnl', config); \
result = pipeline.export_to_parquet('SELECT * FROM pnl_cumulative', 'pnl', partition_cols=['date']); \
print(result.to_dict())"

## Export positions to gold layer
export-positions:
	@echo "Exporting positions to gold..."
	$(PYTHON) -c "\
from src.pipelines import ExportPipeline; \
from src.config import PlatformConfig; \
config = PlatformConfig.default(); \
pipeline = ExportPipeline('export-pos', config); \
result = pipeline.export_to_parquet('SELECT * FROM positions_current', 'positions_snapshot'); \
print(result.to_dict())"

## Export all to gold
export: export-pnl export-positions

#------------------------------------------------------------------------------
# Development
#------------------------------------------------------------------------------

## Run tests
test:
	pytest tests/ -v

## Run tests with coverage
test-cov:
	pytest tests/ --cov=src --cov-report=term-missing

## Lint code
lint:
	ruff check src/ tests/

## Format code
format:
	ruff format src/ tests/

## Type check
typecheck:
	mypy src/

## Run all checks
check: lint typecheck test

#------------------------------------------------------------------------------
# Interactive
#------------------------------------------------------------------------------

## Start interactive DuckDB shell
shell:
	duckdb $(WAREHOUSE)

## Query helper (usage: make query SQL="SELECT * FROM trades LIMIT 10")
query:
	$(PYTHON) -m src.cli query "$(SQL)"

## Show platform status
status:
	$(PYTHON) -m src.cli status

## List all tables
tables:
	$(PYTHON) -m src.cli tables

## Vacuum database
vacuum:
	$(PYTHON) -m src.cli vacuum

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

## Show this help
help:
	@echo "Analytics Platform - Local Analytical Data Stack"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'
	@echo ""
	@echo "Workflow:"
	@echo "  1. make init          - Initialize platform"
	@echo "  2. make ingest-*      - Ingest data to bronze"
	@echo "  3. make views         - Create views over bronze"
	@echo "  4. make materialize   - Build silver/gold tables"
	@echo "  5. make export        - Export to gold Parquet"
	@echo ""
	@echo "Quick commands:"
	@echo "  make refresh          - Full refresh (views + materialize)"
	@echo "  make status           - Show platform status"
	@echo "  make shell            - Interactive DuckDB shell"
