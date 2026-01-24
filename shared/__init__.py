"""
Shared utilities for ai-playground tools.

Add reusable functions here that multiple tools can import.
"""

from datetime import datetime
from pathlib import Path


def get_output_path(name: str, extension: str = "png") -> Path:
    """Generate timestamped output path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir / f"{name}_{timestamp}.{extension}"


def get_data_path(filename: str) -> Path:
    """Get path to data file."""
    return Path(__file__).parent.parent / "data" / filename
