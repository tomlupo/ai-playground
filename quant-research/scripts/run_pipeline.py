#!/usr/bin/env python3
"""
CLI script to run quant research pipelines.

Usage:
    uv run python scripts/run_pipeline.py --config config/tech_momentum.yaml
    uv run python scripts/run_pipeline.py -c config/mean_reversion.yaml --quiet
    uv run python scripts/run_pipeline.py --list
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_research.pipeline import PipelineConfig, ResearchPipeline, run_pipeline


console = Console()


@click.group(invoke_without_command=True)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to pipeline configuration YAML file"
)
@click.option(
    "--list", "-l", "list_configs",
    is_flag=True,
    help="List available configuration files"
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Suppress verbose output during pipeline execution"
)
@click.pass_context
def cli(ctx, config, list_configs, quiet):
    """
    Quant Research Pipeline Runner

    Run quantitative research pipelines with YAML configurations.

    Examples:

        # Run a specific config
        uv run python scripts/run_pipeline.py -c config/tech_momentum.yaml

        # List available configs
        uv run python scripts/run_pipeline.py --list

        # Run quietly (less output)
        uv run python scripts/run_pipeline.py -c config/mean_reversion.yaml -q
    """
    if list_configs:
        list_available_configs()
        return

    if config:
        run_config(config, verbose=not quiet)
    elif ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def list_available_configs():
    """List all available configuration files."""
    config_dir = Path(__file__).parent.parent / "config"

    if not config_dir.exists():
        console.print("[red]Config directory not found![/red]")
        return

    configs = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

    if not configs:
        console.print("[yellow]No configuration files found in config/[/yellow]")
        return

    table = Table(title="Available Pipeline Configurations")
    table.add_column("File", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description", style="white")

    for config_path in sorted(configs):
        try:
            cfg = PipelineConfig.from_yaml(config_path)
            table.add_row(
                config_path.name,
                cfg.name,
                cfg.description[:50] + "..." if len(cfg.description) > 50 else cfg.description
            )
        except Exception as e:
            table.add_row(config_path.name, "[red]Error[/red]", str(e)[:50])

    console.print(table)
    console.print("\n[dim]Run with: uv run python scripts/run_pipeline.py -c config/<filename>[/dim]")


def run_config(config_path: str, verbose: bool = True):
    """Run a pipeline with the given configuration."""
    console.print(Panel(
        f"[bold blue]Loading configuration:[/bold blue] {config_path}",
        title="Quant Research Pipeline"
    ))

    try:
        config = PipelineConfig.from_yaml(config_path)

        console.print(f"\n[bold]{config.name}[/bold]")
        console.print(f"[dim]{config.description}[/dim]\n")

        # Show configuration summary
        table = Table(title="Configuration Summary", show_header=False)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Symbols", ", ".join(config.data.get("symbols", [])))
        table.add_row("Period", config.data.get("period", "1y"))
        table.add_row("Strategy", config.backtest.get("strategy", "N/A"))
        table.add_row("Initial Capital", f"${config.portfolio.get('initial_capital', 100000):,}")

        console.print(table)
        console.print()

        # Run pipeline
        import warnings
        warnings.filterwarnings("ignore")

        pipeline = ResearchPipeline(config, verbose=verbose)
        result = pipeline.run()
        pipeline.print_summary()

        console.print("\n[green]Pipeline completed successfully![/green]")

        return result

    except FileNotFoundError:
        console.print(f"[red]Configuration file not found: {config_path}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error running pipeline: {e}[/red]")
        raise


@cli.command()
@click.argument("config_name")
@click.option("--quiet", "-q", is_flag=True, help="Suppress verbose output")
def run(config_name, quiet):
    """
    Run a pipeline by config name (without path).

    Example: uv run python scripts/run_pipeline.py run tech_momentum
    """
    config_dir = Path(__file__).parent.parent / "config"

    # Try different extensions
    for ext in [".yaml", ".yml", ""]:
        config_path = config_dir / f"{config_name}{ext}"
        if config_path.exists():
            run_config(str(config_path), verbose=not quiet)
            return

    console.print(f"[red]Configuration not found: {config_name}[/red]")
    console.print("[dim]Use --list to see available configurations[/dim]")
    sys.exit(1)


@cli.command()
def all():
    """Run all available pipeline configurations."""
    config_dir = Path(__file__).parent.parent / "config"
    configs = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

    console.print(Panel(
        f"[bold]Running {len(configs)} pipeline configurations[/bold]",
        title="Batch Pipeline Execution"
    ))

    for i, config_path in enumerate(sorted(configs), 1):
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold]Pipeline {i}/{len(configs)}: {config_path.name}[/bold]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

        try:
            run_config(str(config_path), verbose=True)
        except Exception as e:
            console.print(f"[red]Failed: {e}[/red]")
            continue

    console.print("\n[green]All pipelines completed![/green]")


if __name__ == "__main__":
    cli()
