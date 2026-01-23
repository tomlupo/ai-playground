"""Tests for pipeline modules."""

import tempfile
from datetime import date, datetime
from pathlib import Path

import pyarrow as pa
import pytest

from src.pipelines import (
    IngestionPipeline,
    TransformationPipeline,
    ExportPipeline,
    PipelineStatus,
)
from src.config import PlatformConfig, ParquetConfig, DuckDBConfig
from src.schemas import TradeSchema


@pytest.fixture
def temp_config():
    """Create a temporary configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config = PlatformConfig(
            parquet=ParquetConfig(base_path=tmpdir / "data"),
            duckdb=DuckDBConfig(database_path=tmpdir / "warehouse.duckdb"),
        )
        yield config


@pytest.fixture
def sample_trades():
    """Create sample trade data."""
    return pa.table({
        "trade_id": ["T001", "T002", "T003"],
        "symbol": ["AAPL", "GOOG", "AAPL"],
        "side": ["buy", "buy", "sell"],
        "quantity": [100.0, 50.0, 50.0],
        "price": [150.0, 2800.0, 155.0],
        "trade_time": [datetime.now()] * 3,
        "date": [date.today()] * 3,
    })


class TestIngestionPipeline:
    """Tests for IngestionPipeline."""

    def test_ingest_arrow(self, temp_config, sample_trades):
        """Test ingesting PyArrow table."""
        pipeline = IngestionPipeline(
            name="test_ingest",
            config=temp_config,
            target_table="trades",
            partition_cols=["date"],
        )

        result = pipeline.ingest_arrow(sample_trades, validate=False)

        assert result.status == PipelineStatus.SUCCESS
        assert result.rows_written == 3

    def test_ingest_with_validation(self, temp_config, sample_trades):
        """Test ingestion with schema validation."""
        pipeline = IngestionPipeline(
            name="test_ingest",
            config=temp_config,
            target_table="trades",
            schema=TradeSchema,
            partition_cols=["date"],
        )

        result = pipeline.ingest_arrow(sample_trades, validate=True)

        assert result.status == PipelineStatus.SUCCESS

    def test_ingest_invalid_data(self, temp_config):
        """Test ingestion with invalid data."""
        pipeline = IngestionPipeline(
            name="test_ingest",
            config=temp_config,
            target_table="trades",
            schema=TradeSchema,
            partition_cols=["date"],
        )

        # Missing required columns
        invalid_data = pa.table({
            "symbol": ["AAPL"],
            "price": [150.0],
        })

        result = pipeline.ingest_arrow(invalid_data, validate=True)

        assert result.status == PipelineStatus.FAILED
        assert "validation" in result.error.lower()


class TestTransformationPipeline:
    """Tests for TransformationPipeline."""

    def test_transform_sql_to_table(self, temp_config, sample_trades):
        """Test SQL transformation to DuckDB table."""
        # First ingest some data
        ingest = IngestionPipeline(
            name="ingest",
            config=temp_config,
            target_table="trades",
            partition_cols=["date"],
        )
        ingest.ingest_arrow(sample_trades, validate=False)

        # Create transformation pipeline
        transform = TransformationPipeline(
            name="transform",
            config=temp_config,
        )

        # Create view over Parquet
        transform.create_parquet_view("trades", "v_trades")

        # Transform
        result = transform.transform_sql(
            query="SELECT symbol, SUM(quantity) as total_qty FROM v_trades GROUP BY symbol",
            target_table="trades_summary",
            target_type="duckdb",
        )

        assert result.status == PipelineStatus.SUCCESS
        assert result.rows_written == 2  # AAPL and GOOG

        transform.close()

    def test_transform_sql_to_view(self, temp_config, sample_trades):
        """Test SQL transformation to view."""
        ingest = IngestionPipeline(
            name="ingest",
            config=temp_config,
            target_table="trades",
            partition_cols=["date"],
        )
        ingest.ingest_arrow(sample_trades, validate=False)

        transform = TransformationPipeline(
            name="transform",
            config=temp_config,
        )

        transform.create_parquet_view("trades", "v_trades")

        result = transform.transform_sql(
            query="SELECT * FROM v_trades WHERE symbol = 'AAPL'",
            target_table="v_aapl_trades",
            target_type="view",
        )

        assert result.status == PipelineStatus.SUCCESS
        assert result.metadata["type"] == "view"

        transform.close()


class TestExportPipeline:
    """Tests for ExportPipeline."""

    def test_export_to_parquet(self, temp_config, sample_trades):
        """Test exporting to Parquet gold layer."""
        # Setup: ingest and transform
        ingest = IngestionPipeline(
            name="ingest",
            config=temp_config,
            target_table="trades",
            partition_cols=["date"],
        )
        ingest.ingest_arrow(sample_trades, validate=False)

        transform = TransformationPipeline(
            name="transform",
            config=temp_config,
        )
        transform.create_parquet_view("trades", "v_trades")
        transform.transform_sql(
            query="SELECT symbol, SUM(quantity) as total FROM v_trades GROUP BY symbol",
            target_table="summary",
            target_type="duckdb",
        )

        # Export
        export = ExportPipeline(
            name="export",
            config=temp_config,
        )

        result = export.export_to_parquet(
            query="summary",
            target_table="trade_summary",
            layer="gold",
        )

        assert result.status == PipelineStatus.SUCCESS
        assert result.rows_written == 2

        transform.close()
        export.close()


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_duration_calculation(self):
        """Test duration calculation."""
        from src.pipelines.base import PipelineResult

        result = PipelineResult(
            status=PipelineStatus.SUCCESS,
            started_at=datetime(2024, 1, 1, 12, 0, 0),
            completed_at=datetime(2024, 1, 1, 12, 0, 10),
        )

        assert result.duration_seconds == 10.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        from src.pipelines.base import PipelineResult

        result = PipelineResult(
            status=PipelineStatus.SUCCESS,
            started_at=datetime.now(),
            rows_written=100,
        )

        d = result.to_dict()

        assert d["status"] == "success"
        assert d["rows_written"] == 100
