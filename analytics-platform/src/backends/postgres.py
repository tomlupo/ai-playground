"""
PostgreSQL storage backend.

Source / Sync layer - transactional source and integration layer.
"""

from contextlib import contextmanager
from typing import Any, Generator, Optional
import pyarrow as pa

try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extras import execute_values, RealDictCursor
except ImportError:
    psycopg2 = None

from .base import StorageBackend
from ..config import PostgresConfig


class PostgresBackend(StorageBackend):
    """
    PostgreSQL backend for transactional data and external integrations.

    Best used via DuckDB's postgres extension for analytics.
    Direct access provided for transactional operations.
    """

    def __init__(self, config: PostgresConfig):
        if psycopg2 is None:
            raise ImportError("psycopg2 is required. Install with: pip install psycopg2-binary")

        self.config = config
        self._conn = None

    @property
    def conn(self):
        """Get or create PostgreSQL connection."""
        if self._conn is None or self._conn.closed:
            self._conn = self._create_connection()
        return self._conn

    def _create_connection(self):
        """Create PostgreSQL connection."""
        return psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
        )

    def write(
        self,
        table_name: str,
        data: pa.Table,
        partition_cols: Optional[list[str]] = None,
        mode: str = "append",
    ) -> None:
        """
        Write data to PostgreSQL table.

        Args:
            table_name: Name of the table
            data: PyArrow table to write
            partition_cols: Not used for PostgreSQL
            mode: 'append' or 'overwrite'
        """
        df = data.to_pandas()
        columns = list(df.columns)

        with self.conn.cursor() as cur:
            if mode == "overwrite":
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                        sql.Identifier(self.config.schema),
                        sql.Identifier(table_name),
                    )
                )
                self._create_table_from_arrow(cur, table_name, data.schema)

            elif mode == "append" and not self.table_exists(table_name):
                self._create_table_from_arrow(cur, table_name, data.schema)

            # Bulk insert
            values = [tuple(row) for row in df.itertuples(index=False, name=None)]
            insert_query = sql.SQL("INSERT INTO {}.{} ({}) VALUES %s").format(
                sql.Identifier(self.config.schema),
                sql.Identifier(table_name),
                sql.SQL(", ").join(map(sql.Identifier, columns)),
            )
            execute_values(cur, insert_query, values)

        self.conn.commit()

    def _create_table_from_arrow(self, cursor, table_name: str, schema: pa.Schema) -> None:
        """Create table from PyArrow schema."""
        type_mapping = {
            pa.int8(): "SMALLINT",
            pa.int16(): "SMALLINT",
            pa.int32(): "INTEGER",
            pa.int64(): "BIGINT",
            pa.float32(): "REAL",
            pa.float64(): "DOUBLE PRECISION",
            pa.string(): "TEXT",
            pa.large_string(): "TEXT",
            pa.bool_(): "BOOLEAN",
            pa.date32(): "DATE",
            pa.date64(): "DATE",
        }

        columns = []
        for field in schema:
            pg_type = type_mapping.get(field.type, "TEXT")

            # Handle timestamp types
            if pa.types.is_timestamp(field.type):
                if field.type.tz:
                    pg_type = "TIMESTAMPTZ"
                else:
                    pg_type = "TIMESTAMP"
            elif pa.types.is_decimal(field.type):
                pg_type = f"DECIMAL({field.type.precision}, {field.type.scale})"

            columns.append(f'"{field.name}" {pg_type}')

        create_sql = f'CREATE TABLE {self.config.schema}."{table_name}" ({", ".join(columns)})'
        cursor.execute(create_sql)

    def read(
        self,
        table_name: str,
        columns: Optional[list[str]] = None,
        filters: Optional[list[tuple]] = None,
    ) -> pa.Table:
        """
        Read data from PostgreSQL table.

        Args:
            table_name: Name of the table
            columns: Columns to read
            filters: WHERE clause conditions as tuples

        Returns:
            PyArrow Table
        """
        cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        query = f'SELECT {cols} FROM {self.config.schema}."{table_name}"'

        if filters:
            where_clauses = []
            for col, op, val in filters:
                if isinstance(val, str):
                    where_clauses.append(f'"{col}" {op} \'{val}\'')
                elif isinstance(val, (list, tuple)):
                    vals = ", ".join(
                        f"'{v}'" if isinstance(v, str) else str(v) for v in val
                    )
                    where_clauses.append(f'"{col}" IN ({vals})')
                else:
                    where_clauses.append(f'"{col}" {op} {val}')
            query += " WHERE " + " AND ".join(where_clauses)

        return self.query_arrow(query)

    def execute(self, query: str, params: Optional[dict] = None) -> Any:
        """Execute a query."""
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description:
                return cur.fetchall()
            self.conn.commit()
            return cur.rowcount

    def query_arrow(self, query: str, params: Optional[dict] = None) -> pa.Table:
        """Execute query and return as PyArrow Table."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            if not rows:
                return pa.table({})

            # Convert to columnar format
            columns = {key: [] for key in rows[0].keys()}
            for row in rows:
                for key, value in row.items():
                    columns[key].append(value)

            return pa.table(columns)

    def query_df(self, query: str, params: Optional[dict] = None):
        """Execute query and return as pandas DataFrame."""
        return self.query_arrow(query, params).to_pandas()

    def list_tables(self) -> list[str]:
        """List all tables in the schema."""
        result = self.execute(
            "SELECT table_name FROM information_schema.tables "
            f"WHERE table_schema = '{self.config.schema}' "
            "AND table_type = 'BASE TABLE'"
        )
        return [row[0] for row in result] if result else []

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        result = self.execute(
            "SELECT EXISTS ("
            "SELECT FROM information_schema.tables "
            f"WHERE table_schema = '{self.config.schema}' "
            f"AND table_name = '{table_name}'"
            ")"
        )
        return result[0][0] if result else False

    def get_schema(self, table_name: str) -> pa.Schema:
        """Get the schema of a table."""
        # Read zero rows to get schema
        query = f'SELECT * FROM {self.config.schema}."{table_name}" LIMIT 0'
        return self.query_arrow(query).schema

    def describe(self, table_name: str) -> list[dict]:
        """Get detailed column information."""
        result = self.execute(
            "SELECT column_name, data_type, is_nullable, column_default "
            "FROM information_schema.columns "
            f"WHERE table_schema = '{self.config.schema}' "
            f"AND table_name = '{table_name}' "
            "ORDER BY ordinal_position"
        )
        return [
            {
                "column_name": row[0],
                "data_type": row[1],
                "nullable": row[2] == "YES",
                "default": row[3],
            }
            for row in result
        ] if result else []

    def create_index(
        self,
        table_name: str,
        columns: list[str],
        index_name: Optional[str] = None,
        unique: bool = False,
    ) -> None:
        """Create an index on a table."""
        if index_name is None:
            index_name = f"idx_{table_name}_{'_'.join(columns)}"

        unique_clause = "UNIQUE" if unique else ""
        cols = ", ".join(f'"{c}"' for c in columns)

        self.execute(
            f'CREATE {unique_clause} INDEX IF NOT EXISTS "{index_name}" '
            f'ON {self.config.schema}."{table_name}" ({cols})'
        )
        self.conn.commit()

    def upsert(
        self,
        table_name: str,
        data: pa.Table,
        conflict_columns: list[str],
        update_columns: Optional[list[str]] = None,
    ) -> int:
        """
        Insert or update data based on conflict columns.

        Args:
            table_name: Target table
            data: Data to upsert
            conflict_columns: Columns that define uniqueness
            update_columns: Columns to update on conflict (None = all non-conflict)

        Returns:
            Number of affected rows
        """
        df = data.to_pandas()
        all_columns = list(df.columns)

        if update_columns is None:
            update_columns = [c for c in all_columns if c not in conflict_columns]

        conflict_cols = ", ".join(f'"{c}"' for c in conflict_columns)
        update_set = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_columns)

        with self.conn.cursor() as cur:
            values = [tuple(row) for row in df.itertuples(index=False, name=None)]
            insert_query = sql.SQL(
                "INSERT INTO {}.{} ({}) VALUES %s "
                "ON CONFLICT ({}) DO UPDATE SET {}"
            ).format(
                sql.Identifier(self.config.schema),
                sql.Identifier(table_name),
                sql.SQL(", ").join(map(sql.Identifier, all_columns)),
                sql.SQL(conflict_cols),
                sql.SQL(update_set),
            )
            execute_values(cur, insert_query, values)
            affected = cur.rowcount

        self.conn.commit()
        return affected

    @contextmanager
    def transaction(self) -> Generator[Any, None, None]:
        """Context manager for transactions."""
        try:
            yield self.conn.cursor()
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def close(self) -> None:
        """Close the PostgreSQL connection."""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()
