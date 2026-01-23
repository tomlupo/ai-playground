"""
CLI entry point for the analytics platform.

Usage:
    analytics init          Initialize the platform
    analytics run <script>  Run a SQL script
    analytics query <sql>   Execute a SQL query
    analytics status        Show platform status
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from .config import PlatformConfig
from .backends import DuckDBBackend, ParquetBackend


console = Console()


def get_config(config_path: Optional[str] = None) -> PlatformConfig:
    """Load or create configuration."""
    if config_path:
        return PlatformConfig.from_file(Path(config_path))
    return PlatformConfig.default(Path.cwd())


@click.group()
@click.option("--config", "-c", help="Path to config file")
@click.pass_context
def main(ctx, config):
    """Analytics Platform CLI - Local analytical data stack."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = get_config(config)


@main.command()
@click.pass_context
def init(ctx):
    """Initialize the analytics platform."""
    config = ctx.obj["config"]

    console.print("[bold blue]Initializing Analytics Platform...[/bold blue]")

    # Create directories
    dirs = [
        config.parquet.bronze_path,
        config.parquet.silver_path,
        config.parquet.gold_path,
        config.duckdb.database_path.parent,
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        console.print(f"  Created: {d}")

    # Initialize DuckDB
    duckdb = DuckDBBackend(config.duckdb)

    # Run init migration if exists
    init_sql = Path.cwd() / "sql" / "migrations" / "001_init.sql"
    if init_sql.exists():
        script = init_sql.read_text()
        for stmt in script.split(";"):
            stmt = stmt.strip()
            if stmt:
                duckdb.execute(stmt)
        console.print(f"  Executed: {init_sql}")

    duckdb.close()

    # Save config
    config_file = Path.cwd() / "config" / "platform.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config.save(config_file)
    console.print(f"  Config saved: {config_file}")

    console.print("[bold green]Platform initialized![/bold green]")


@main.command()
@click.argument("sql_file", type=click.Path(exists=True))
@click.option("--vars", "-v", multiple=True, help="Variables (key=value)")
@click.pass_context
def run(ctx, sql_file, vars):
    """Run a SQL script."""
    config = ctx.obj["config"]

    # Parse variables
    variables = {}
    for v in vars:
        key, value = v.split("=", 1)
        variables[key] = value

    # Add default variables
    variables.setdefault("data_path", str(config.parquet.base_path))

    # Read and process script
    script = Path(sql_file).read_text()
    for key, value in variables.items():
        script = script.replace(f"{{{{{key}}}}}", value)

    # Execute
    duckdb = DuckDBBackend(config.duckdb)

    console.print(f"[bold blue]Running: {sql_file}[/bold blue]")

    statements = [s.strip() for s in script.split(";") if s.strip()]
    for i, stmt in enumerate(statements, 1):
        try:
            duckdb.execute(stmt)
            console.print(f"  [{i}/{len(statements)}] OK")
        except Exception as e:
            console.print(f"  [{i}/{len(statements)}] [red]ERROR: {e}[/red]")
            duckdb.close()
            sys.exit(1)

    duckdb.close()
    console.print("[bold green]Script completed![/bold green]")


@main.command()
@click.argument("sql")
@click.option("--format", "-f", type=click.Choice(["table", "csv", "json"]), default="table")
@click.option("--limit", "-l", default=100, help="Row limit")
@click.pass_context
def query(ctx, sql, format, limit):
    """Execute a SQL query."""
    config = ctx.obj["config"]
    duckdb = DuckDBBackend(config.duckdb)

    # Add limit if not present
    if "limit" not in sql.lower():
        sql = f"{sql} LIMIT {limit}"

    try:
        result = duckdb.query_df(sql)

        if format == "table":
            table = Table(show_header=True, header_style="bold")
            for col in result.columns:
                table.add_column(col)
            for _, row in result.head(limit).iterrows():
                table.add_row(*[str(v) for v in row])
            console.print(table)
            console.print(f"\n[dim]{len(result)} rows[/dim]")

        elif format == "csv":
            print(result.to_csv(index=False))

        elif format == "json":
            print(result.to_json(orient="records", indent=2))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    finally:
        duckdb.close()


@main.command()
@click.pass_context
def status(ctx):
    """Show platform status."""
    config = ctx.obj["config"]

    console.print("[bold blue]Analytics Platform Status[/bold blue]\n")

    # DuckDB status
    console.print("[bold]DuckDB:[/bold]")
    if config.duckdb.database_path.exists():
        size_mb = config.duckdb.database_path.stat().st_size / (1024 * 1024)
        console.print(f"  Database: {config.duckdb.database_path} ({size_mb:.2f} MB)")

        duckdb = DuckDBBackend(config.duckdb)
        tables = duckdb.list_tables()
        views = duckdb.list_views()
        console.print(f"  Tables: {len(tables)}")
        console.print(f"  Views: {len(views)}")
        duckdb.close()
    else:
        console.print("  [yellow]Not initialized[/yellow]")

    # Parquet status
    console.print("\n[bold]Parquet Data Lake:[/bold]")
    parquet = ParquetBackend(config.parquet)

    for layer in ["bronze", "silver", "gold"]:
        tables = parquet.list_tables(layer)
        layer_path = getattr(config.parquet, f"{layer}_path")
        if tables:
            total_files = sum(
                1 for _ in layer_path.rglob("*.parquet")
            )
            console.print(f"  {layer.capitalize()}: {len(tables)} tables, {total_files} files")
        else:
            console.print(f"  {layer.capitalize()}: [dim]empty[/dim]")

    # PostgreSQL status
    console.print("\n[bold]PostgreSQL:[/bold]")
    console.print(f"  Host: {config.postgres.host}:{config.postgres.port}")
    console.print(f"  Database: {config.postgres.database}")


@main.command()
@click.pass_context
def tables(ctx):
    """List all tables and views."""
    config = ctx.obj["config"]
    duckdb = DuckDBBackend(config.duckdb)

    # DuckDB tables
    console.print("[bold]DuckDB Tables:[/bold]")
    for table in duckdb.list_tables():
        schema = duckdb.get_schema(table)
        console.print(f"  {table} ({len(schema)} columns)")

    # DuckDB views
    console.print("\n[bold]DuckDB Views:[/bold]")
    for view in duckdb.list_views():
        console.print(f"  {view}")

    # Parquet tables
    parquet = ParquetBackend(config.parquet)
    console.print("\n[bold]Parquet Tables:[/bold]")
    for layer in ["bronze", "silver", "gold"]:
        tables = parquet.list_tables(layer)
        if tables:
            console.print(f"  [{layer}]")
            for t in tables:
                rows = parquet.get_row_count(t, layer)
                console.print(f"    {t}: {rows:,} rows")

    duckdb.close()


@main.command()
@click.argument("table_name")
@click.pass_context
def describe(ctx, table_name):
    """Describe a table schema."""
    config = ctx.obj["config"]
    duckdb = DuckDBBackend(config.duckdb)

    try:
        info = duckdb.describe(table_name)

        table = Table(title=f"Table: {table_name}", show_header=True)
        table.add_column("Column")
        table.add_column("Type")
        table.add_column("Nullable")

        for col in info:
            table.add_row(col["column_name"], col["column_type"], col["null"])

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        duckdb.close()


@main.command()
@click.pass_context
def vacuum(ctx):
    """Vacuum the DuckDB database."""
    config = ctx.obj["config"]
    duckdb = DuckDBBackend(config.duckdb)

    before_size = config.duckdb.database_path.stat().st_size

    console.print("Running VACUUM...")
    duckdb.vacuum()
    duckdb.checkpoint()
    duckdb.close()

    after_size = config.duckdb.database_path.stat().st_size
    saved = (before_size - after_size) / (1024 * 1024)

    console.print(f"[green]Done! Saved {saved:.2f} MB[/green]")


if __name__ == "__main__":
    main()
