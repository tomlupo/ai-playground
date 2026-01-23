"""
Configuration for the analytics platform.

Supports environment variables and config files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class ParquetConfig:
    """Configuration for Parquet storage."""

    base_path: Path = field(default_factory=lambda: Path("data"))
    compression: str = "zstd"
    row_group_size: int = 100_000

    @property
    def bronze_path(self) -> Path:
        return self.base_path / "bronze"

    @property
    def silver_path(self) -> Path:
        return self.base_path / "silver"

    @property
    def gold_path(self) -> Path:
        return self.base_path / "gold"


@dataclass
class DuckDBConfig:
    """Configuration for DuckDB."""

    database_path: Path = field(default_factory=lambda: Path("warehouse/warehouse.duckdb"))
    read_only: bool = False
    threads: Optional[int] = None
    memory_limit: Optional[str] = None  # e.g., "4GB"

    def get_connection_string(self) -> str:
        return str(self.database_path)


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL."""

    host: str = "localhost"
    port: int = 5432
    database: str = "analytics"
    user: str = "postgres"
    password: str = ""
    schema: str = "public"

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        """Create config from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "analytics"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            schema=os.getenv("POSTGRES_SCHEMA", "public"),
        )

    def get_connection_string(self) -> str:
        """Get connection string for DuckDB postgres extension."""
        return f"host={self.host} port={self.port} dbname={self.database} user={self.user} password={self.password}"

    def get_sqlalchemy_url(self) -> str:
        """Get SQLAlchemy connection URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class PlatformConfig:
    """Main platform configuration."""

    parquet: ParquetConfig = field(default_factory=ParquetConfig)
    duckdb: DuckDBConfig = field(default_factory=DuckDBConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)

    @classmethod
    def from_file(cls, path: Path) -> "PlatformConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            parquet=ParquetConfig(**data.get("parquet", {})),
            duckdb=DuckDBConfig(**data.get("duckdb", {})),
            postgres=PostgresConfig(**data.get("postgres", {})),
        )

    @classmethod
    def default(cls, base_path: Optional[Path] = None) -> "PlatformConfig":
        """Create default configuration."""
        if base_path is None:
            base_path = Path.cwd()

        return cls(
            parquet=ParquetConfig(base_path=base_path / "data"),
            duckdb=DuckDBConfig(database_path=base_path / "warehouse" / "warehouse.duckdb"),
            postgres=PostgresConfig.from_env(),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "parquet": {
                "base_path": str(self.parquet.base_path),
                "compression": self.parquet.compression,
                "row_group_size": self.parquet.row_group_size,
            },
            "duckdb": {
                "database_path": str(self.duckdb.database_path),
                "read_only": self.duckdb.read_only,
                "threads": self.duckdb.threads,
                "memory_limit": self.duckdb.memory_limit,
            },
            "postgres": {
                "host": self.postgres.host,
                "port": self.postgres.port,
                "database": self.postgres.database,
                "user": self.postgres.user,
                "schema": self.postgres.schema,
            },
        }

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
