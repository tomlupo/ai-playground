"""
DuckDB storage backend.

Silver / Gold layer - the analytical brain.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional, Union
import pyarrow as pa

try:
    import duckdb
except ImportError:
    raise ImportError("duckdb is required. Install with: pip install duckdb")

from .base import StorageBackend
from ..config import DuckDBConfig, PostgresConfig


class DuckDBBackend(StorageBackend):
    """
    DuckDB analytical engine.

    The query brain that can read from Parquet, PostgreSQL, and its own tables.
    """

    def __init__(self, config: DuckDBConfig, postgres_config: Optional[PostgresConfig] = None):
        self.config = config
        self.postgres_config = postgres_config
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

        # Ensure warehouse directory exists
        self.config.database_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create DuckDB connection."""
        if self._conn is None:
            self._conn = self._create_connection()
        return self._conn

    def _create_connection(self) -> duckdb.DuckDBPyConnection:
        """Create and configure DuckDB connection."""
        conn = duckdb.connect(
            str(self.config.database_path),
            read_only=self.config.read_only,
        )

        # Configure performance settings
        if self.config.threads:
            conn.execute(f"SET threads = {self.config.threads}")

        if self.config.memory_limit:
            conn.execute(f"SET memory_limit = '{self.config.memory_limit}'")

        # Enable progress bar for long queries
        conn.execute("SET enable_progress_bar = true")

        return conn

    def attach_postgres(self, alias: str = "pg") -> None:
        """
        Attach PostgreSQL database.

        After attaching, you can query PostgreSQL tables as:
            SELECT * FROM pg.public.table_name
        """
        if self.postgres_config is None:
            raise ValueError("PostgreSQL config not provided")

        # Install and load postgres extension
        self.conn.execute("INSTALL postgres")
        self.conn.execute("LOAD postgres")

        # Attach the database
        conn_str = self.postgres_config.get_connection_string()
        self.conn.execute(f"ATTACH '{conn_str}' AS {alias} (TYPE postgres)")

    def detach(self, alias: str) -> None:
        """Detach an attached database."""
        self.conn.execute(f"DETACH {alias}")

    def write(
        self,
        table_name: str,
        data: pa.Table,
        partition_cols: Optional[list[str]] = None,
        mode: str = "append",
    ) -> None:
        """
        Write data to DuckDB table.

        Args:
            table_name: Name of the table
            data: PyArrow table to write
            partition_cols: Not used for DuckDB tables
            mode: 'append' or 'overwrite'
        """
        if mode == "overwrite":
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM data")
        else:
            if self.table_exists(table_name):
                self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM data")
            else:
                self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM data")

    def read(
        self,
        table_name: str,
        columns: Optional[list[str]] = None,
        filters: Optional[list[tuple]] = None,
    ) -> pa.Table:
        """
        Read data from DuckDB table.

        Args:
            table_name: Name of the table
            columns: Columns to read
            filters: SQL WHERE clause conditions as tuples

        Returns:
            PyArrow Table
        """
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM {table_name}"

        if filters:
            where_clauses = []
            for col, op, val in filters:
                if isinstance(val, str):
                    where_clauses.append(f"{col} {op} '{val}'")
                elif isinstance(val, (list, tuple)):
                    vals = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in val)
                    where_clauses.append(f"{col} IN ({vals})")
                else:
                    where_clauses.append(f"{col} {op} {val}")
            query += " WHERE " + " AND ".join(where_clauses)

        return self.conn.execute(query).arrow()

    def execute(self, query: str, params: Optional[dict] = None) -> duckdb.DuckDBPyRelation:
        """
        Execute a SQL query.

        Args:
            query: SQL query string
            params: Optional parameters for parameterized queries

        Returns:
            DuckDB relation (lazy evaluation)
        """
        if params:
            return self.conn.execute(query, params)
        return self.conn.execute(query)

    def query(self, query: str, params: Optional[dict] = None) -> pa.Table:
        """Execute query and return as PyArrow Table."""
        return self.execute(query, params).arrow()

    def query_df(self, query: str, params: Optional[dict] = None):
        """Execute query and return as pandas DataFrame."""
        return self.execute(query, params).df()

    def query_polars(self, query: str, params: Optional[dict] = None):
        """Execute query and return as Polars DataFrame."""
        return self.execute(query, params).pl()

    def read_parquet(
        self,
        path: str,
        columns: Optional[list[str]] = None,
        filters: Optional[str] = None,
    ) -> pa.Table:
        """
        Read Parquet files directly via DuckDB.

        Args:
            path: Glob pattern for Parquet files
            columns: Columns to read
            filters: SQL WHERE clause

        Returns:
            PyArrow Table
        """
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM read_parquet('{path}', hive_partitioning=true)"

        if filters:
            query += f" WHERE {filters}"

        return self.conn.execute(query).arrow()

    def create_view(
        self,
        view_name: str,
        query: str,
        replace: bool = True,
    ) -> None:
        """
        Create a view.

        Args:
            view_name: Name of the view
            query: SQL query for the view
            replace: Whether to replace existing view
        """
        create = "CREATE OR REPLACE VIEW" if replace else "CREATE VIEW"
        self.conn.execute(f"{create} {view_name} AS {query}")

    def create_parquet_view(
        self,
        view_name: str,
        parquet_path: str,
        replace: bool = True,
    ) -> None:
        """
        Create a view over Parquet files.

        Args:
            view_name: Name of the view
            parquet_path: Glob pattern for Parquet files
            replace: Whether to replace existing view
        """
        query = f"SELECT * FROM read_parquet('{parquet_path}', hive_partitioning=true)"
        self.create_view(view_name, query, replace)

    def materialize(
        self,
        table_name: str,
        query: str,
        replace: bool = True,
    ) -> int:
        """
        Materialize a query result as a table.

        Args:
            table_name: Name for the materialized table
            query: SQL query to materialize
            replace: Whether to replace existing table

        Returns:
            Number of rows in the materialized table
        """
        if replace:
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        self.conn.execute(f"CREATE TABLE {table_name} AS {query}")

        result = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return result[0] if result else 0

    def export_parquet(
        self,
        query: str,
        path: str,
        partition_by: Optional[list[str]] = None,
    ) -> None:
        """
        Export query results to Parquet.

        Args:
            query: SQL query to export
            path: Output path
            partition_by: Columns to partition by
        """
        if partition_by:
            partition_clause = ", ".join(partition_by)
            self.conn.execute(
                f"COPY ({query}) TO '{path}' "
                f"(FORMAT PARQUET, PARTITION_BY ({partition_clause}))"
            )
        else:
            self.conn.execute(f"COPY ({query}) TO '{path}' (FORMAT PARQUET)")

    def export_to_postgres(
        self,
        query: str,
        table_name: str,
        pg_alias: str = "pg",
        schema: str = "public",
        mode: str = "overwrite",
    ) -> None:
        """
        Export query results to PostgreSQL.

        Args:
            query: SQL query to export
            table_name: Target table name in PostgreSQL
            pg_alias: Alias of attached PostgreSQL database
            schema: PostgreSQL schema
            mode: 'overwrite' or 'append'
        """
        full_table = f"{pg_alias}.{schema}.{table_name}"

        if mode == "overwrite":
            self.conn.execute(f"DROP TABLE IF EXISTS {full_table}")
            self.conn.execute(f"CREATE TABLE {full_table} AS {query}")
        else:
            self.conn.execute(f"INSERT INTO {full_table} {query}")

    def list_tables(self) -> list[str]:
        """List all tables in the database."""
        result = self.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'BASE TABLE'"
        ).fetchall()
        return [row[0] for row in result]

    def list_views(self) -> list[str]:
        """List all views in the database."""
        result = self.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'VIEW'"
        ).fetchall()
        return [row[0] for row in result]

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        result = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            f"WHERE table_name = '{table_name}'"
        ).fetchone()
        return result[0] > 0 if result else False

    def get_schema(self, table_name: str) -> pa.Schema:
        """Get the schema of a table."""
        result = self.conn.execute(f"SELECT * FROM {table_name} LIMIT 0").arrow()
        return result.schema

    def describe(self, table_name: str) -> list[dict]:
        """Get detailed column information for a table."""
        result = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
        return [
            {
                "column_name": row[0],
                "column_type": row[1],
                "null": row[2],
                "key": row[3],
                "default": row[4],
                "extra": row[5],
            }
            for row in result
        ]

    def vacuum(self) -> None:
        """Vacuum the database to reclaim space."""
        self.conn.execute("VACUUM")

    def checkpoint(self) -> None:
        """Force a checkpoint to persist changes."""
        self.conn.execute("CHECKPOINT")

    @contextmanager
    def transaction(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Context manager for transactions."""
        self.conn.execute("BEGIN TRANSACTION")
        try:
            yield self.conn
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()
