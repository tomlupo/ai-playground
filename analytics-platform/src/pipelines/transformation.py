"""
Transformation pipelines.

Transform Bronze data into Silver/Gold using DuckDB.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union
import pyarrow as pa

from .base import BasePipeline, PipelineResult, PipelineStatus
from ..backends import ParquetBackend, DuckDBBackend
from ..config import PlatformConfig
from ..schemas.base import BaseSchema


class TransformationPipeline(BasePipeline):
    """
    Pipeline for transforming data using DuckDB.

    Reads from Bronze (Parquet), transforms with SQL, writes to Silver/Gold.
    """

    def __init__(
        self,
        name: str,
        config: PlatformConfig,
    ):
        super().__init__(name)
        self.config = config
        self.parquet = ParquetBackend(config.parquet)
        self.duckdb = DuckDBBackend(config.duckdb)

    def build(self) -> "TransformationPipeline":
        """Build default transformation steps."""
        return self

    def transform_sql(
        self,
        query: str,
        target_table: str,
        target_type: str = "duckdb",  # 'duckdb', 'parquet', or 'view'
        partition_cols: Optional[list[str]] = None,
        layer: str = "silver",
        replace: bool = True,
    ) -> PipelineResult:
        """
        Execute a SQL transformation.

        Args:
            query: SQL query (can reference Parquet via read_parquet())
            target_table: Name of the output table
            target_type: 'duckdb' (materialize), 'parquet' (export), or 'view'
            partition_cols: Columns to partition by (for Parquet)
            layer: Target layer for Parquet output
            replace: Whether to replace existing table/view

        Returns:
            PipelineResult
        """
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            if target_type == "view":
                # Create view
                self.duckdb.create_view(target_table, query, replace=replace)
                result.status = PipelineStatus.SUCCESS
                result.metadata["type"] = "view"

            elif target_type == "duckdb":
                # Materialize as DuckDB table
                rows = self.duckdb.materialize(target_table, query, replace=replace)
                result.status = PipelineStatus.SUCCESS
                result.rows_written = rows
                result.metadata["type"] = "table"

            elif target_type == "parquet":
                # Execute and write to Parquet
                data = self.duckdb.query(query)
                result.rows_processed = len(data)

                self.parquet.write(
                    table_name=target_table,
                    data=data,
                    partition_cols=partition_cols,
                    mode="overwrite" if replace else "append",
                    layer=layer,
                )
                result.status = PipelineStatus.SUCCESS
                result.rows_written = len(data)
                result.metadata["type"] = "parquet"
                result.metadata["layer"] = layer

            else:
                raise ValueError(f"Unknown target_type: {target_type}")

            result.metadata["target"] = target_table

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    def transform_from_file(
        self,
        sql_file: Union[str, Path],
        target_table: str,
        target_type: str = "duckdb",
        partition_cols: Optional[list[str]] = None,
        layer: str = "silver",
        replace: bool = True,
        variables: Optional[dict[str, str]] = None,
    ) -> PipelineResult:
        """
        Execute transformation from SQL file.

        Args:
            sql_file: Path to SQL file
            target_table: Name of the output table
            target_type: 'duckdb', 'parquet', or 'view'
            partition_cols: Columns to partition by
            layer: Target layer for Parquet
            replace: Whether to replace existing
            variables: Variables to substitute in SQL ({{var}} syntax)

        Returns:
            PipelineResult
        """
        sql_path = Path(sql_file)
        query = sql_path.read_text()

        # Substitute variables
        if variables:
            for key, value in variables.items():
                query = query.replace(f"{{{{{key}}}}}", value)

        return self.transform_sql(
            query=query,
            target_table=target_table,
            target_type=target_type,
            partition_cols=partition_cols,
            layer=layer,
            replace=replace,
        )

    def create_parquet_view(
        self,
        source_table: str,
        view_name: str,
        layer: str = "bronze",
    ) -> PipelineResult:
        """
        Create a DuckDB view over Parquet files.

        Args:
            source_table: Name of the Parquet table
            view_name: Name of the view to create
            layer: Source layer

        Returns:
            PipelineResult
        """
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            parquet_path = self.parquet.get_duckdb_path(source_table, layer)
            self.duckdb.create_parquet_view(view_name, parquet_path)

            result.status = PipelineStatus.SUCCESS
            result.metadata["source"] = parquet_path
            result.metadata["view"] = view_name

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    def materialize_incremental(
        self,
        query: str,
        target_table: str,
        merge_keys: list[str],
    ) -> PipelineResult:
        """
        Incremental materialization using MERGE/upsert logic.

        Args:
            query: SQL query for new/updated data
            target_table: Target table name
            merge_keys: Columns to use for matching existing rows

        Returns:
            PipelineResult
        """
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            # Check if table exists
            if not self.duckdb.table_exists(target_table):
                # First run - just create the table
                rows = self.duckdb.materialize(target_table, query)
                result.rows_written = rows
            else:
                # Incremental update
                staging_table = f"_staging_{target_table}"

                # Create staging table with new data
                self.duckdb.materialize(staging_table, query)

                # Build merge query
                key_conditions = " AND ".join(
                    f"t.{k} = s.{k}" for k in merge_keys
                )

                # Get all columns
                schema = self.duckdb.get_schema(staging_table)
                all_cols = [f.name for f in schema]
                non_key_cols = [c for c in all_cols if c not in merge_keys]

                # Update existing rows
                if non_key_cols:
                    update_sets = ", ".join(f"{c} = s.{c}" for c in non_key_cols)
                    self.duckdb.execute(
                        f"UPDATE {target_table} t SET {update_sets} "
                        f"FROM {staging_table} s WHERE {key_conditions}"
                    )

                # Insert new rows
                self.duckdb.execute(
                    f"INSERT INTO {target_table} "
                    f"SELECT s.* FROM {staging_table} s "
                    f"WHERE NOT EXISTS ("
                    f"SELECT 1 FROM {target_table} t WHERE {key_conditions})"
                )

                # Get counts
                new_count = self.duckdb.execute(
                    f"SELECT COUNT(*) FROM {staging_table}"
                ).fetchone()[0]
                result.rows_processed = new_count

                # Cleanup
                self.duckdb.execute(f"DROP TABLE {staging_table}")

            result.status = PipelineStatus.SUCCESS
            result.metadata["target"] = target_table

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    def run_sql_script(
        self,
        sql_file: Union[str, Path],
        variables: Optional[dict[str, str]] = None,
    ) -> PipelineResult:
        """
        Run a SQL script with multiple statements.

        Args:
            sql_file: Path to SQL file
            variables: Variables to substitute

        Returns:
            PipelineResult
        """
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            sql_path = Path(sql_file)
            script = sql_path.read_text()

            # Substitute variables
            if variables:
                for key, value in variables.items():
                    script = script.replace(f"{{{{{key}}}}}", value)

            # Split and execute statements
            statements = [s.strip() for s in script.split(";") if s.strip()]
            for stmt in statements:
                self.duckdb.execute(stmt)

            result.status = PipelineStatus.SUCCESS
            result.metadata["statements"] = len(statements)
            result.metadata["script"] = sql_path.name

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    def close(self) -> None:
        """Close connections."""
        self.duckdb.close()
