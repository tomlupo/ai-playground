"""
Export pipelines.

Export data from DuckDB to Parquet and/or PostgreSQL.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from .base import BasePipeline, PipelineResult, PipelineStatus
from ..backends import ParquetBackend, DuckDBBackend, PostgresBackend
from ..config import PlatformConfig


class ExportPipeline(BasePipeline):
    """
    Pipeline for exporting data from DuckDB.

    Exports to Parquet (archive/cloud) or PostgreSQL (dashboards/APIs).
    """

    def __init__(
        self,
        name: str,
        config: PlatformConfig,
    ):
        super().__init__(name)
        self.config = config
        self.parquet = ParquetBackend(config.parquet)
        self.duckdb = DuckDBBackend(config.duckdb, config.postgres)

    def build(self) -> "ExportPipeline":
        """Build default export steps."""
        return self

    def export_to_parquet(
        self,
        query: str,
        target_table: str,
        layer: str = "gold",
        partition_cols: Optional[list[str]] = None,
        mode: str = "overwrite",
    ) -> PipelineResult:
        """
        Export query results to Parquet.

        Args:
            query: SQL query or table name
            target_table: Target table name in Parquet
            layer: Target layer (typically 'gold')
            partition_cols: Columns to partition by
            mode: 'overwrite' or 'append'

        Returns:
            PipelineResult
        """
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            # If query is just a table name, make it a SELECT
            if " " not in query.strip():
                query = f"SELECT * FROM {query}"

            data = self.duckdb.query(query)
            result.rows_processed = len(data)

            self.parquet.write(
                table_name=target_table,
                data=data,
                partition_cols=partition_cols,
                mode=mode,
                layer=layer,
            )

            result.status = PipelineStatus.SUCCESS
            result.rows_written = len(data)
            result.metadata["target"] = f"{layer}/{target_table}"

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    def export_to_postgres(
        self,
        query: str,
        target_table: str,
        schema: str = "public",
        mode: str = "overwrite",
    ) -> PipelineResult:
        """
        Export query results to PostgreSQL.

        Args:
            query: SQL query or table name
            target_table: Target table name in PostgreSQL
            schema: PostgreSQL schema
            mode: 'overwrite' or 'append'

        Returns:
            PipelineResult
        """
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            # Attach PostgreSQL if not already attached
            try:
                self.duckdb.attach_postgres("pg_export")
            except Exception:
                # Already attached or connection error
                pass

            # If query is just a table name, make it a SELECT
            if " " not in query.strip():
                query = f"SELECT * FROM {query}"

            # Export using DuckDB's postgres extension
            self.duckdb.export_to_postgres(
                query=query,
                table_name=target_table,
                pg_alias="pg_export",
                schema=schema,
                mode=mode,
            )

            # Get row count
            count = self.duckdb.execute(
                f"SELECT COUNT(*) FROM pg_export.{schema}.{target_table}"
            ).fetchone()[0]

            result.status = PipelineStatus.SUCCESS
            result.rows_written = count
            result.metadata["target"] = f"postgres:{schema}.{target_table}"

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    def export_to_postgres_direct(
        self,
        query: str,
        target_table: str,
        schema: str = "public",
        mode: str = "overwrite",
    ) -> PipelineResult:
        """
        Export to PostgreSQL using direct connection (without DuckDB postgres ext).

        Useful when DuckDB postgres extension is not available.
        """
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            postgres = PostgresBackend(self.config.postgres)

            # Execute query and get data
            if " " not in query.strip():
                query = f"SELECT * FROM {query}"

            data = self.duckdb.query(query)
            result.rows_processed = len(data)

            # Write to PostgreSQL
            postgres.write(
                table_name=target_table,
                data=data,
                mode=mode,
            )

            result.status = PipelineStatus.SUCCESS
            result.rows_written = len(data)
            result.metadata["target"] = f"postgres:{schema}.{target_table}"

            postgres.close()

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    def export_snapshot(
        self,
        tables: list[str],
        snapshot_name: str,
        layer: str = "gold",
    ) -> PipelineResult:
        """
        Export multiple tables as a snapshot.

        Creates a timestamped snapshot directory with all specified tables.

        Args:
            tables: List of table names to export
            snapshot_name: Name for the snapshot
            layer: Target layer

        Returns:
            PipelineResult
        """
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_dir = f"{snapshot_name}_{timestamp}"

            total_rows = 0
            for table in tables:
                data = self.duckdb.read(table)

                # Export each table
                table_path = f"{snapshot_dir}/{table}"
                self.parquet.write(
                    table_name=table_path,
                    data=data,
                    mode="overwrite",
                    layer=layer,
                )
                total_rows += len(data)

            result.status = PipelineStatus.SUCCESS
            result.rows_written = total_rows
            result.metadata["snapshot"] = snapshot_dir
            result.metadata["tables"] = tables

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    def archive_table(
        self,
        table_name: str,
        archive_name: Optional[str] = None,
        delete_after: bool = False,
    ) -> PipelineResult:
        """
        Archive a DuckDB table to Parquet gold layer.

        Args:
            table_name: Table to archive
            archive_name: Archive name (defaults to table_name_YYYYMMDD)
            delete_after: Whether to drop the DuckDB table after archiving

        Returns:
            PipelineResult
        """
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            if archive_name is None:
                timestamp = datetime.now().strftime("%Y%m%d")
                archive_name = f"{table_name}_{timestamp}"

            data = self.duckdb.read(table_name)
            result.rows_processed = len(data)

            self.parquet.write(
                table_name=f"archive/{archive_name}",
                data=data,
                mode="overwrite",
                layer="gold",
            )

            if delete_after:
                self.duckdb.execute(f"DROP TABLE {table_name}")
                result.metadata["deleted"] = True

            result.status = PipelineStatus.SUCCESS
            result.rows_written = len(data)
            result.metadata["archive"] = f"gold/archive/{archive_name}"

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    def close(self) -> None:
        """Close connections."""
        self.duckdb.close()


class SyncPipeline(ExportPipeline):
    """
    Bidirectional sync between DuckDB and PostgreSQL.
    """

    def sync_to_postgres(
        self,
        table_name: str,
        conflict_columns: list[str],
        update_columns: Optional[list[str]] = None,
        schema: str = "public",
    ) -> PipelineResult:
        """
        Sync a DuckDB table to PostgreSQL with upsert logic.

        Args:
            table_name: Table to sync
            conflict_columns: Columns that define uniqueness
            update_columns: Columns to update on conflict
            schema: PostgreSQL schema

        Returns:
            PipelineResult
        """
        result = PipelineResult(
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            postgres = PostgresBackend(self.config.postgres)

            # Get data from DuckDB
            data = self.duckdb.read(table_name)
            result.rows_processed = len(data)

            # Upsert to PostgreSQL
            affected = postgres.upsert(
                table_name=table_name,
                data=data,
                conflict_columns=conflict_columns,
                update_columns=update_columns,
            )

            result.status = PipelineStatus.SUCCESS
            result.rows_written = affected
            result.metadata["target"] = f"postgres:{schema}.{table_name}"

            postgres.close()

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result
