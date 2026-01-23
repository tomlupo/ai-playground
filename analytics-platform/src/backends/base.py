"""Base backend interface."""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional
import pyarrow as pa


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def write(
        self,
        table_name: str,
        data: pa.Table,
        partition_cols: Optional[list[str]] = None,
        mode: str = "append",
    ) -> None:
        """
        Write data to storage.

        Args:
            table_name: Name of the table/dataset
            data: PyArrow table to write
            partition_cols: Columns to partition by
            mode: 'append' or 'overwrite'
        """
        pass

    @abstractmethod
    def read(
        self,
        table_name: str,
        columns: Optional[list[str]] = None,
        filters: Optional[list[tuple]] = None,
    ) -> pa.Table:
        """
        Read data from storage.

        Args:
            table_name: Name of the table/dataset
            columns: Columns to read (None for all)
            filters: PyArrow filters for predicate pushdown

        Returns:
            PyArrow Table
        """
        pass

    @abstractmethod
    def execute(self, query: str, params: Optional[dict] = None) -> Any:
        """Execute a query against the backend."""
        pass

    @abstractmethod
    def list_tables(self) -> list[str]:
        """List all available tables."""
        pass

    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        pass

    @abstractmethod
    def get_schema(self, table_name: str) -> pa.Schema:
        """Get the schema of a table."""
        pass

    def close(self) -> None:
        """Close any open connections."""
        pass

    def __enter__(self) -> "StorageBackend":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
