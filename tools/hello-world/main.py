#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.25",
#     "rich>=13.0",
# ]
# ///
"""
Hello World Tool

A simple example tool demonstrating the ai-playground pattern.
"""

from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def main() -> None:
    """Main entry point."""
    console.print("[bold green]🧪 AI Playground - Hello World Tool[/]\n")

    # Example: Create a nice table
    table = Table(title="Sample Data")
    table.add_column("Item", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Timestamp", datetime.now().isoformat())
    table.add_row("Python", "3.11+")
    table.add_row("Package Manager", "uv")
    table.add_row("Status", "✅ Working!")

    console.print(table)

    # Example: Save output
    output_dir = Path(__file__).parent.parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"hello-world_{timestamp}.md"

    output_path.write_text(f"""# Hello World Output

Generated: {datetime.now().isoformat()}

## Status

Everything is working! 🎉

## Next Steps

1. Create your own tool in `tools/your-tool-name/`
2. Use PEP 723 inline metadata for dependencies
3. Run with `uv run tools/your-tool-name/main.py`
4. Share via `gh gist create ...`
""")

    console.print(f"\n[dim]Output saved to:[/] {output_path}")
    console.print("[bold green]Done![/]")


if __name__ == "__main__":
    main()
