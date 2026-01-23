"""Tests for storage backends."""

import tempfile
from datetime import date, datetime
from pathlib import Path

import pyarrow as pa
import pytest

from src.backends import ParquetBackend, DuckDBBackend
from src.config import ParquetConfig, DuckDBConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data():
    """Create sample PyArrow table."""
    return pa.table({
        "symbol": ["AAPL", "GOOG", "MSFT"],
        "price": [150.0, 2800.0, 350.0],
        "volume": [1000, 500, 750],
        "date": [date(2024, 1, 1)] * 3,
    })


class TestParquetBackend:
    """Tests for ParquetBackend."""

    def test_write_and_read(self, temp_dir, sample_data):
        """Test basic write and read operations."""
        config = ParquetConfig(base_path=temp_dir / "data")
        backend = ParquetBackend(config)

        # Write data
        backend.write("test_table", sample_data, partition_cols=["date"])

        # Read data
        result = backend.read("test_table")

        assert len(result) == 3
        assert "symbol" in result.column_names
        assert "price" in result.column_names

    def test_list_tables(self, temp_dir, sample_data):
        """Test listing tables."""
        config = ParquetConfig(base_path=temp_dir / "data")
        backend = ParquetBackend(config)

        backend.write("table1", sample_data)
        backend.write("table2", sample_data)

        tables = backend.list_tables()
        assert "table1" in tables
        assert "table2" in tables

    def test_table_exists(self, temp_dir, sample_data):
        """Test checking table existence."""
        config = ParquetConfig(base_path=temp_dir / "data")
        backend = ParquetBackend(config)

        assert not backend.table_exists("test_table")

        backend.write("test_table", sample_data)

        assert backend.table_exists("test_table")

    def test_get_schema(self, temp_dir, sample_data):
        """Test getting table schema."""
        config = ParquetConfig(base_path=temp_dir / "data")
        backend = ParquetBackend(config)

        backend.write("test_table", sample_data)
        schema = backend.get_schema("test_table")

        assert "symbol" in schema.names
        assert "price" in schema.names

    def test_read_with_filters(self, temp_dir):
        """Test reading with filters."""
        config = ParquetConfig(base_path=temp_dir / "data")
        backend = ParquetBackend(config)

        data = pa.table({
            "symbol": ["AAPL", "GOOG", "MSFT", "AAPL"],
            "price": [150.0, 2800.0, 350.0, 155.0],
            "date": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 2)],
        })

        backend.write("test_table", data, partition_cols=["date"])

        # Filter by symbol
        result = backend.read("test_table", filters=[("symbol", "==", "AAPL")])
        assert len(result) == 2

    def test_layers(self, temp_dir, sample_data):
        """Test writing to different layers."""
        config = ParquetConfig(base_path=temp_dir / "data")
        backend = ParquetBackend(config)

        backend.write("bronze_table", sample_data, layer="bronze")
        backend.write("silver_table", sample_data, layer="silver")
        backend.write("gold_table", sample_data, layer="gold")

        assert backend.table_exists("bronze_table", layer="bronze")
        assert backend.table_exists("silver_table", layer="silver")
        assert backend.table_exists("gold_table", layer="gold")


class TestDuckDBBackend:
    """Tests for DuckDBBackend."""

    def test_write_and_read(self, temp_dir, sample_data):
        """Test basic write and read operations."""
        config = DuckDBConfig(database_path=temp_dir / "test.duckdb")
        backend = DuckDBBackend(config)

        backend.write("test_table", sample_data)
        result = backend.read("test_table")

        assert len(result) == 3
        backend.close()

    def test_execute_query(self, temp_dir, sample_data):
        """Test executing SQL queries."""
        config = DuckDBConfig(database_path=temp_dir / "test.duckdb")
        backend = DuckDBBackend(config)

        backend.write("test_table", sample_data)
        result = backend.query("SELECT symbol, price FROM test_table WHERE price > 200")

        assert len(result) == 2  # GOOG and MSFT
        backend.close()

    def test_create_view(self, temp_dir, sample_data):
        """Test creating views."""
        config = DuckDBConfig(database_path=temp_dir / "test.duckdb")
        backend = DuckDBBackend(config)

        backend.write("test_table", sample_data)
        backend.create_view("test_view", "SELECT * FROM test_table WHERE price > 200")

        views = backend.list_views()
        assert "test_view" in views

        result = backend.query("SELECT * FROM test_view")
        assert len(result) == 2
        backend.close()

    def test_materialize(self, temp_dir, sample_data):
        """Test materializing query results."""
        config = DuckDBConfig(database_path=temp_dir / "test.duckdb")
        backend = DuckDBBackend(config)

        backend.write("source_table", sample_data)
        rows = backend.materialize(
            "materialized_table",
            "SELECT symbol, price * 2 AS doubled_price FROM source_table"
        )

        assert rows == 3
        assert "materialized_table" in backend.list_tables()
        backend.close()

    def test_read_parquet(self, temp_dir, sample_data):
        """Test reading Parquet files directly."""
        # Write Parquet file
        parquet_path = temp_dir / "data.parquet"
        import pyarrow.parquet as pq
        pq.write_table(sample_data, parquet_path)

        # Read via DuckDB
        config = DuckDBConfig(database_path=temp_dir / "test.duckdb")
        backend = DuckDBBackend(config)

        result = backend.read_parquet(str(parquet_path))
        assert len(result) == 3
        backend.close()

    def test_transaction(self, temp_dir, sample_data):
        """Test transaction handling."""
        config = DuckDBConfig(database_path=temp_dir / "test.duckdb")
        backend = DuckDBBackend(config)

        backend.write("test_table", sample_data)

        # Successful transaction
        with backend.transaction():
            backend.execute("UPDATE test_table SET price = price + 10 WHERE symbol = 'AAPL'")

        result = backend.query("SELECT price FROM test_table WHERE symbol = 'AAPL'")
        assert result["price"][0].as_py() == 160.0

        backend.close()


class TestBackendIntegration:
    """Integration tests between backends."""

    def test_parquet_to_duckdb(self, temp_dir, sample_data):
        """Test reading Parquet data through DuckDB."""
        # Write to Parquet
        parquet_config = ParquetConfig(base_path=temp_dir / "data")
        parquet = ParquetBackend(parquet_config)
        parquet.write("trades", sample_data, partition_cols=["date"])

        # Read via DuckDB
        duckdb_config = DuckDBConfig(database_path=temp_dir / "warehouse.duckdb")
        duckdb = DuckDBBackend(duckdb_config)

        parquet_path = parquet.get_duckdb_path("trades")
        result = duckdb.read_parquet(parquet_path)

        assert len(result) == 3
        duckdb.close()
