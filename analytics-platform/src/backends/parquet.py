"""
Parquet storage backend.

Bronze / Raw layer - immutable, append-only fact storage.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Union
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from .base import StorageBackend
from ..config import ParquetConfig


class ParquetBackend(StorageBackend):
    """
    Parquet storage backend for the data lake.

    Supports partitioned storage with automatic partition discovery.
    """

    def __init__(self, config: ParquetConfig):
        self.config = config
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create data directories if they don't exist."""
        for path in [
            self.config.bronze_path,
            self.config.silver_path,
            self.config.gold_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _get_table_path(self, table_name: str, layer: str = "bronze") -> Path:
        """Get the path for a table based on layer."""
        layer_path = getattr(self.config, f"{layer}_path")
        return layer_path / table_name

    def write(
        self,
        table_name: str,
        data: pa.Table,
        partition_cols: Optional[list[str]] = None,
        mode: str = "append",
        layer: str = "bronze",
    ) -> None:
        """
        Write data to Parquet.

        Args:
            table_name: Name of the dataset
            data: PyArrow table to write
            partition_cols: Columns to partition by (e.g., ['date'])
            mode: 'append' or 'overwrite'
            layer: 'bronze', 'silver', or 'gold'
        """
        table_path = self._get_table_path(table_name, layer)

        if mode == "overwrite" and table_path.exists():
            import shutil

            shutil.rmtree(table_path)

        table_path.mkdir(parents=True, exist_ok=True)

        if partition_cols:
            pq.write_to_dataset(
                data,
                root_path=str(table_path),
                partition_cols=partition_cols,
                compression=self.config.compression,
                existing_data_behavior="overwrite_or_ignore",
            )
        else:
            # Generate unique filename for append
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = table_path / f"part-{timestamp}.parquet"
            pq.write_table(
                data,
                file_path,
                compression=self.config.compression,
                row_group_size=self.config.row_group_size,
            )

    def read(
        self,
        table_name: str,
        columns: Optional[list[str]] = None,
        filters: Optional[list[tuple]] = None,
        layer: str = "bronze",
    ) -> pa.Table:
        """
        Read data from Parquet with optional predicate pushdown.

        Args:
            table_name: Name of the dataset
            columns: Columns to read
            filters: PyArrow filters e.g., [('date', '>=', '2024-01-01')]
            layer: 'bronze', 'silver', or 'gold'

        Returns:
            PyArrow Table
        """
        table_path = self._get_table_path(table_name, layer)

        if not table_path.exists():
            raise FileNotFoundError(f"Table not found: {table_name} in {layer}")

        dataset = ds.dataset(
            table_path,
            format="parquet",
            partitioning="hive",
        )

        return dataset.to_table(columns=columns, filter=self._build_filter(filters))

    def _build_filter(
        self, filters: Optional[list[tuple]]
    ) -> Optional[ds.Expression]:
        """Build PyArrow filter expression from list of tuples."""
        if not filters:
            return None

        expressions = []
        for col, op, val in filters:
            field = ds.field(col)
            if op == "==":
                expressions.append(field == val)
            elif op == "!=":
                expressions.append(field != val)
            elif op == ">":
                expressions.append(field > val)
            elif op == ">=":
                expressions.append(field >= val)
            elif op == "<":
                expressions.append(field < val)
            elif op == "<=":
                expressions.append(field <= val)
            elif op == "in":
                expressions.append(field.isin(val))

        if len(expressions) == 1:
            return expressions[0]

        result = expressions[0]
        for expr in expressions[1:]:
            result = result & expr
        return result

    def read_partitions(
        self,
        table_name: str,
        partitions: dict[str, Union[str, list[str]]],
        columns: Optional[list[str]] = None,
        layer: str = "bronze",
    ) -> pa.Table:
        """
        Read specific partitions.

        Args:
            table_name: Name of the dataset
            partitions: Dict of partition column to value(s)
                        e.g., {'date': '2024-01-01'} or {'date': ['2024-01-01', '2024-01-02']}
            columns: Columns to read
            layer: Storage layer

        Returns:
            PyArrow Table
        """
        filters = []
        for col, val in partitions.items():
            if isinstance(val, list):
                filters.append((col, "in", val))
            else:
                filters.append((col, "==", val))

        return self.read(table_name, columns=columns, filters=filters, layer=layer)

    def execute(self, query: str, params: Optional[dict] = None) -> Any:
        """
        Execute is not directly supported on Parquet.
        Use DuckDB to query Parquet files with SQL.
        """
        raise NotImplementedError(
            "Direct SQL queries not supported on Parquet. "
            "Use DuckDB with read_parquet() instead."
        )

    def list_tables(self, layer: str = "bronze") -> list[str]:
        """List all tables in a layer."""
        layer_path = getattr(self.config, f"{layer}_path")
        if not layer_path.exists():
            return []
        return [d.name for d in layer_path.iterdir() if d.is_dir()]

    def table_exists(self, table_name: str, layer: str = "bronze") -> bool:
        """Check if a table exists."""
        table_path = self._get_table_path(table_name, layer)
        return table_path.exists() and any(table_path.iterdir())

    def get_schema(self, table_name: str, layer: str = "bronze") -> pa.Schema:
        """Get the schema of a table."""
        table_path = self._get_table_path(table_name, layer)

        if not table_path.exists():
            raise FileNotFoundError(f"Table not found: {table_name}")

        dataset = ds.dataset(table_path, format="parquet", partitioning="hive")
        return dataset.schema

    def get_partitions(self, table_name: str, layer: str = "bronze") -> list[dict]:
        """Get list of partitions for a table."""
        table_path = self._get_table_path(table_name, layer)

        if not table_path.exists():
            return []

        dataset = ds.dataset(table_path, format="parquet", partitioning="hive")
        partitions = []

        for fragment in dataset.get_fragments():
            part_dict = {}
            for expr in fragment.partition_expression.operands if hasattr(
                fragment.partition_expression, 'operands'
            ) else []:
                if hasattr(expr, 'operands') and len(expr.operands) == 2:
                    col = str(expr.operands[0])
                    val = expr.operands[1].as_py() if hasattr(expr.operands[1], 'as_py') else str(expr.operands[1])
                    part_dict[col] = val
            if part_dict:
                partitions.append(part_dict)

        return partitions

    def get_row_count(self, table_name: str, layer: str = "bronze") -> int:
        """Get total row count for a table."""
        table_path = self._get_table_path(table_name, layer)

        if not table_path.exists():
            return 0

        dataset = ds.dataset(table_path, format="parquet", partitioning="hive")
        return dataset.count_rows()

    def get_duckdb_path(self, table_name: str, layer: str = "bronze") -> str:
        """
        Get the glob pattern for reading this table from DuckDB.

        Returns:
            String like 'data/bronze/trades/**/*.parquet'
        """
        table_path = self._get_table_path(table_name, layer)
        return f"{table_path}/**/*.parquet"
