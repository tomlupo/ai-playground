"""
Ingestion pipelines.

Move data from external sources to Parquet (Bronze layer).
"""

from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union
import pyarrow as pa

from .base import BasePipeline, PipelineResult, PipelineStatus
from ..backends import ParquetBackend, DuckDBBackend, PostgresBackend
from ..config import PlatformConfig
from ..schemas.base import BaseSchema


class IngestionPipeline(BasePipeline):
    """
    Pipeline for ingesting data into the Bronze layer (Parquet).

    Supports various data sources:
    - In-memory data (pandas, PyArrow)
    - PostgreSQL tables
    - CSV/JSON files
    - Custom data fetchers
    """

    def __init__(
        self,
        name: str,
        config: PlatformConfig,
        target_table: str,
        schema: Optional[type[BaseSchema]] = None,
        partition_cols: Optional[list[str]] = None,
    ):
        super().__init__(name)
        self.config = config
        self.target_table = target_table
        self.schema = schema
        self.partition_cols = partition_cols or ["date"]
        self.parquet = ParquetBackend(config.parquet)

    def build(self) -> "IngestionPipeline":
        """Build default ingestion steps."""
        # Default steps can be overridden
        return self

    def ingest_arrow(
        self,
        data: pa.Table,
        validate: bool = True,
        layer: str = "bronze",
    ) -> PipelineResult:
        """
        Ingest PyArrow table directly.

        Args:
            data: PyArrow table to ingest
            validate: Whether to validate against schema
            layer: Target layer (bronze, silver, gold)

        Returns:
            PipelineResult
        """
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            # Validate schema
            if validate and self.schema:
                errors = self.schema.validate(data)
                if errors:
                    raise ValueError(f"Schema validation failed: {errors}")

            # Write to Parquet
            self.parquet.write(
                table_name=self.target_table,
                data=data,
                partition_cols=self.partition_cols,
                layer=layer,
            )

            result.status = PipelineStatus.SUCCESS
            result.rows_written = len(data)
            result.metadata["target"] = f"{layer}/{self.target_table}"

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    def ingest_dataframe(
        self,
        df,  # pandas or polars DataFrame
        validate: bool = True,
        layer: str = "bronze",
    ) -> PipelineResult:
        """Ingest from pandas or polars DataFrame."""
        # Convert to PyArrow
        if hasattr(df, "to_arrow"):
            # Polars
            data = df.to_arrow()
        else:
            # Pandas
            data = pa.Table.from_pandas(df)

        return self.ingest_arrow(data, validate=validate, layer=layer)

    def ingest_from_postgres(
        self,
        source_table: str,
        query: Optional[str] = None,
        postgres_config: Optional[Any] = None,
        validate: bool = True,
        layer: str = "bronze",
    ) -> PipelineResult:
        """
        Ingest from PostgreSQL table.

        Args:
            source_table: Source table name (or used in default query)
            query: Custom SQL query (overrides source_table)
            postgres_config: PostgreSQL config (uses platform config if None)
            validate: Whether to validate schema
            layer: Target layer

        Returns:
            PipelineResult
        """
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            pg_config = postgres_config or self.config.postgres
            postgres = PostgresBackend(pg_config)

            # Read from PostgreSQL
            if query:
                data = postgres.query_arrow(query)
            else:
                data = postgres.read(source_table)

            result.rows_processed = len(data)

            # Validate and write
            if validate and self.schema:
                errors = self.schema.validate(data)
                if errors:
                    raise ValueError(f"Schema validation failed: {errors}")

            self.parquet.write(
                table_name=self.target_table,
                data=data,
                partition_cols=self.partition_cols,
                layer=layer,
            )

            result.status = PipelineStatus.SUCCESS
            result.rows_written = len(data)
            result.metadata["source"] = source_table
            result.metadata["target"] = f"{layer}/{self.target_table}"

            postgres.close()

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    def ingest_from_csv(
        self,
        path: Union[str, Path],
        validate: bool = True,
        layer: str = "bronze",
        **csv_options,
    ) -> PipelineResult:
        """
        Ingest from CSV file(s).

        Args:
            path: Path to CSV file or glob pattern
            validate: Whether to validate schema
            layer: Target layer
            **csv_options: Options passed to PyArrow CSV reader

        Returns:
            PipelineResult
        """
        import pyarrow.csv as pv

        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            path = Path(path)

            if path.is_file():
                data = pv.read_csv(path, **csv_options)
            else:
                # Glob pattern
                tables = []
                for file in Path(path.parent).glob(path.name):
                    tables.append(pv.read_csv(file, **csv_options))
                data = pa.concat_tables(tables)

            result.rows_processed = len(data)

            # Validate and write
            if validate and self.schema:
                errors = self.schema.validate(data)
                if errors:
                    raise ValueError(f"Schema validation failed: {errors}")

            self.parquet.write(
                table_name=self.target_table,
                data=data,
                partition_cols=self.partition_cols,
                layer=layer,
            )

            result.status = PipelineStatus.SUCCESS
            result.rows_written = len(data)
            result.metadata["source"] = str(path)
            result.metadata["target"] = f"{layer}/{self.target_table}"

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    def ingest_from_fetcher(
        self,
        fetcher: Callable[..., pa.Table],
        validate: bool = True,
        layer: str = "bronze",
        **fetcher_kwargs,
    ) -> PipelineResult:
        """
        Ingest using a custom data fetcher function.

        Args:
            fetcher: Callable that returns PyArrow Table
            validate: Whether to validate schema
            layer: Target layer
            **fetcher_kwargs: Arguments passed to fetcher

        Returns:
            PipelineResult
        """
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            # Fetch data
            data = fetcher(**fetcher_kwargs)
            result.rows_processed = len(data)

            # Validate and write
            if validate and self.schema:
                errors = self.schema.validate(data)
                if errors:
                    raise ValueError(f"Schema validation failed: {errors}")

            self.parquet.write(
                table_name=self.target_table,
                data=data,
                partition_cols=self.partition_cols,
                layer=layer,
            )

            result.status = PipelineStatus.SUCCESS
            result.rows_written = len(data)
            result.metadata["target"] = f"{layer}/{self.target_table}"

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result


class IncrementalIngestion(IngestionPipeline):
    """
    Incremental ingestion with watermark tracking.

    Tracks the last processed timestamp/ID to enable incremental updates.
    """

    def __init__(
        self,
        name: str,
        config: PlatformConfig,
        target_table: str,
        watermark_column: str,
        schema: Optional[type[BaseSchema]] = None,
        partition_cols: Optional[list[str]] = None,
    ):
        super().__init__(name, config, target_table, schema, partition_cols)
        self.watermark_column = watermark_column
        self._watermark_file = config.parquet.base_path / ".watermarks" / f"{target_table}.txt"

    def get_watermark(self) -> Optional[str]:
        """Get the current watermark value."""
        if self._watermark_file.exists():
            return self._watermark_file.read_text().strip()
        return None

    def set_watermark(self, value: str) -> None:
        """Set the watermark value."""
        self._watermark_file.parent.mkdir(parents=True, exist_ok=True)
        self._watermark_file.write_text(str(value))

    def ingest_incremental_from_postgres(
        self,
        source_table: str,
        postgres_config: Optional[Any] = None,
        validate: bool = True,
        layer: str = "bronze",
    ) -> PipelineResult:
        """
        Incremental ingestion from PostgreSQL.

        Only fetches rows where watermark_column > last watermark.
        """
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            pg_config = postgres_config or self.config.postgres
            postgres = PostgresBackend(pg_config)

            # Build incremental query
            watermark = self.get_watermark()
            if watermark:
                query = (
                    f'SELECT * FROM "{source_table}" '
                    f"WHERE {self.watermark_column} > '{watermark}' "
                    f"ORDER BY {self.watermark_column}"
                )
            else:
                query = f'SELECT * FROM "{source_table}" ORDER BY {self.watermark_column}'

            data = postgres.query_arrow(query)
            result.rows_processed = len(data)

            if len(data) == 0:
                result.status = PipelineStatus.SUCCESS
                result.metadata["message"] = "No new data"
                result.completed_at = datetime.now()
                return result

            # Validate and write
            if validate and self.schema:
                errors = self.schema.validate(data)
                if errors:
                    raise ValueError(f"Schema validation failed: {errors}")

            self.parquet.write(
                table_name=self.target_table,
                data=data,
                partition_cols=self.partition_cols,
                layer=layer,
            )

            # Update watermark
            new_watermark = data.column(self.watermark_column)[-1].as_py()
            self.set_watermark(str(new_watermark))

            result.status = PipelineStatus.SUCCESS
            result.rows_written = len(data)
            result.metadata["source"] = source_table
            result.metadata["target"] = f"{layer}/{self.target_table}"
            result.metadata["watermark"] = str(new_watermark)

            postgres.close()

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result
