---
name: create-tool
description: Create a new Python tool in the tools/ directory with proper structure, dependencies, and documentation
---

# Create Tool Skill

Creates a new self-contained Python tool in `tools/{name}/`.

## Steps

1. **Create directory**: `tools/{tool-name}/`

2. **Create main.py** with this template:

```python
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # Add tool-specific dependencies here
# ]
# ///
"""
{Tool Name}

{Brief description of what this tool does}
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def main() -> None:
    """Main entry point."""
    console.print("[bold green]Starting {tool-name}...[/]")
    
    # TODO: Implement tool logic
    
    # Save outputs
    output_dir = Path(__file__).parent.parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_path = output_dir / f"{tool-name}_{timestamp}.png"
    
    console.print("[bold green]Done![/]")


if __name__ == "__main__":
    main()
```

3. **Create README.md**:

```markdown
# {Tool Name}

## Purpose

{What problem does this solve?}

## Usage

```bash
uv run tools/{tool-name}/main.py
```

## Output

{What does it produce?}

## Notes

{Any observations, learnings, or TODO items}
```

4. **Run the tool** to verify it works:

```bash
uv run tools/{tool-name}/main.py
```

5. **If requested**, create a gist:

```bash
gh gist create tools/{tool-name}/main.py tools/{tool-name}/README.md --public -d "{description}"
```

## Naming Convention

- Use lowercase with hyphens: `portfolio-analyzer`, `data-fetcher`
- Be descriptive but concise
- Prefix with domain if needed: `fin-returns-calc`, `ml-feature-gen`
