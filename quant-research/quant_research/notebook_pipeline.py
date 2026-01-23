"""
Notebook Pipeline using Papermill.

Execute Jupyter notebooks as part of research pipelines.
Supports:
- Parameterized notebook execution
- Batch processing
- Output collection
- Report generation
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any
import shutil

import papermill as pm
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell


class NotebookPipeline:
    """
    Execute notebooks as pipeline stages using papermill.
    """

    def __init__(
        self,
        notebooks_dir: str | Path = "notebooks",
        output_dir: str | Path = "outputs",
    ):
        self.notebooks_dir = Path(notebooks_dir)
        self.output_dir = Path(output_dir)
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        notebook_path: str | Path,
        parameters: dict | None = None,
        output_path: str | Path | None = None,
        kernel_name: str = "python3",
    ) -> Path:
        """
        Execute a notebook with parameters.

        Args:
            notebook_path: Path to input notebook
            parameters: Parameters to inject
            output_path: Path for output notebook
            kernel_name: Jupyter kernel name

        Returns:
            Path to executed notebook
        """
        notebook_path = Path(notebook_path)
        parameters = parameters or {}

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = (
                self.output_dir
                / f"{notebook_path.stem}_{timestamp}.ipynb"
            )
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        pm.execute_notebook(
            str(notebook_path),
            str(output_path),
            parameters=parameters,
            kernel_name=kernel_name,
        )

        return output_path

    def execute_batch(
        self,
        notebook_path: str | Path,
        parameter_sets: list[dict],
        parallel: bool = False,
    ) -> list[Path]:
        """
        Execute notebook with multiple parameter sets.

        Args:
            notebook_path: Path to input notebook
            parameter_sets: List of parameter dictionaries
            parallel: Run in parallel (not implemented yet)

        Returns:
            List of output paths
        """
        outputs = []
        for i, params in enumerate(parameter_sets):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = (
                self.output_dir
                / f"{Path(notebook_path).stem}_batch{i}_{timestamp}.ipynb"
            )
            try:
                result = self.execute(notebook_path, params, output_path)
                outputs.append(result)
            except Exception as e:
                print(f"Error executing batch {i}: {e}")
        return outputs

    def create_research_notebook(
        self,
        name: str,
        symbols: list[str],
        analysis_type: str = "full",
    ) -> Path:
        """
        Create a research notebook template.

        Args:
            name: Notebook name
            symbols: Symbols to analyze
            analysis_type: Type of analysis

        Returns:
            Path to created notebook
        """
        nb = new_notebook()

        # Title
        nb.cells.append(new_markdown_cell(f"# {name}\n\nGenerated: {datetime.now()}"))

        # Parameters cell (tagged for papermill)
        params_cell = new_code_cell(
            f"""# Parameters
symbols = {symbols}
period = "1y"
analysis_type = "{analysis_type}"
"""
        )
        params_cell.metadata["tags"] = ["parameters"]
        nb.cells.append(params_cell)

        # Imports
        nb.cells.append(
            new_code_cell(
                """import warnings
warnings.filterwarnings('ignore')

from quant_research import (
    DataFetcher,
    TechnicalIndicators,
    PortfolioAnalyzer,
    run_ml_analysis,
)
from quant_research.backtest_vectorized import VectorizedBacktester, ParameterOptimizer
import pandas as pd
import numpy as np
"""
            )
        )

        # Data fetching
        nb.cells.append(new_markdown_cell("## Data Fetching"))
        nb.cells.append(
            new_code_cell(
                """fetcher = DataFetcher()
prices = fetcher.get_combined_close_prices(symbols, period=period)
print(f"Loaded data for {len(symbols)} symbols, {len(prices)} days")
prices.head()
"""
            )
        )

        # Technical Analysis
        nb.cells.append(new_markdown_cell("## Technical Analysis"))
        nb.cells.append(
            new_code_cell(
                """ti = TechnicalIndicators()
symbol = symbols[0]
data = fetcher.get_stock_data(symbol, period=period)

indicators = {
    'SMA 20': ti.sma(data['close'], 20),
    'SMA 50': ti.sma(data['close'], 50),
    'RSI': ti.rsi(data['close'], 14),
}

indicators_df = pd.DataFrame(indicators)
indicators_df.tail(10)
"""
            )
        )

        # Backtesting
        nb.cells.append(new_markdown_cell("## Backtesting"))
        nb.cells.append(
            new_code_cell(
                """backtester = VectorizedBacktester()

# SMA Crossover
result = backtester.run_sma_crossover(data['close'], short_period=20, long_period=50)

print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
"""
            )
        )

        # Parameter Optimization
        nb.cells.append(new_markdown_cell("## Parameter Optimization"))
        nb.cells.append(
            new_code_cell(
                """optimizer = ParameterOptimizer(backtester)

param_grid = {
    'short_period': [10, 20, 30],
    'long_period': [50, 100, 200],
}

best_params, results = optimizer.grid_search(
    data['close'],
    strategy='sma',
    param_grid=param_grid,
)

print(f"Best Parameters: {best_params}")
results.sort_values('sharpe_ratio', ascending=False).head()
"""
            )
        )

        # Portfolio Analysis
        if len(symbols) > 1:
            nb.cells.append(new_markdown_cell("## Portfolio Analysis"))
            nb.cells.append(
                new_code_cell(
                    """analyzer = PortfolioAnalyzer(prices)

stats = analyzer.calculate_statistics()
print("Asset Statistics:")
display(stats)

corr = analyzer.correlation_matrix()
print("\\nCorrelation Matrix:")
display(corr)
"""
                )
            )

        # ML Analysis
        nb.cells.append(new_markdown_cell("## ML Analysis"))
        nb.cells.append(
            new_code_cell(
                """if analysis_type == 'full':
    ml_results = run_ml_analysis(data, task='classification', horizon=5, verbose=True)
    print(f"\\nBest Model: {ml_results['best_model']}")
"""
            )
        )

        # Summary
        nb.cells.append(new_markdown_cell("## Summary"))
        nb.cells.append(
            new_code_cell(
                """summary = {
    'symbols': symbols,
    'period': period,
    'best_strategy_params': best_params if 'best_params' in dir() else None,
    'sharpe_ratio': result.sharpe_ratio,
}
summary
"""
            )
        )

        # Save notebook
        notebook_path = self.notebooks_dir / f"{name}.ipynb"
        with open(notebook_path, "w") as f:
            nbformat.write(nb, f)

        return notebook_path


class NotebookTemplates:
    """
    Pre-built notebook templates for common analyses.
    """

    @staticmethod
    def strategy_comparison(output_dir: Path) -> Path:
        """Create strategy comparison notebook."""
        nb = new_notebook()

        nb.cells.append(new_markdown_cell("# Strategy Comparison Analysis"))

        params_cell = new_code_cell(
            """# Parameters
symbol = "AAPL"
period = "2y"
strategies = ["sma", "rsi", "bollinger", "macd"]
"""
        )
        params_cell.metadata["tags"] = ["parameters"]
        nb.cells.append(params_cell)

        nb.cells.append(
            new_code_cell(
                """import warnings
warnings.filterwarnings('ignore')

from quant_research import DataFetcher
from quant_research.backtest_vectorized import VectorizedBacktester
import pandas as pd

fetcher = DataFetcher()
data = fetcher.get_stock_data(symbol, period=period)
backtester = VectorizedBacktester()

results = []
for strategy in strategies:
    if strategy == 'sma':
        result = backtester.run_sma_crossover(data['close'])
    elif strategy == 'rsi':
        result = backtester.run_rsi_strategy(data['close'])
    elif strategy == 'bollinger':
        result = backtester.run_bollinger_strategy(data['close'])
    elif strategy == 'macd':
        result = backtester.run_macd_strategy(data['close'])

    results.append({
        'strategy': strategy,
        'return': result.total_return,
        'sharpe': result.sharpe_ratio,
        'max_dd': result.max_drawdown,
    })

comparison = pd.DataFrame(results)
print(comparison.to_string(index=False))
"""
            )
        )

        path = output_dir / "strategy_comparison.ipynb"
        with open(path, "w") as f:
            nbformat.write(nb, f)
        return path

    @staticmethod
    def portfolio_optimization(output_dir: Path) -> Path:
        """Create portfolio optimization notebook."""
        nb = new_notebook()

        nb.cells.append(new_markdown_cell("# Portfolio Optimization Analysis"))

        params_cell = new_code_cell(
            """# Parameters
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
period = "2y"
risk_free_rate = 0.05
"""
        )
        params_cell.metadata["tags"] = ["parameters"]
        nb.cells.append(params_cell)

        nb.cells.append(
            new_code_cell(
                """import warnings
warnings.filterwarnings('ignore')

from quant_research import DataFetcher, AdvancedPortfolioOptimizer
import pandas as pd

fetcher = DataFetcher()
prices = fetcher.get_combined_close_prices(symbols, period=period)

optimizer = AdvancedPortfolioOptimizer(prices, risk_free_rate=risk_free_rate)

print("Comparing optimization strategies...")
comparison = optimizer.compare_strategies()
print(comparison.to_string(index=False))
"""
            )
        )

        path = output_dir / "portfolio_optimization.ipynb"
        with open(path, "w") as f:
            nbformat.write(nb, f)
        return path

    @staticmethod
    def ml_prediction(output_dir: Path) -> Path:
        """Create ML prediction notebook."""
        nb = new_notebook()

        nb.cells.append(new_markdown_cell("# ML Price Direction Prediction"))

        params_cell = new_code_cell(
            """# Parameters
symbol = "AAPL"
period = "3y"
horizon = 5
task = "classification"
"""
        )
        params_cell.metadata["tags"] = ["parameters"]
        nb.cells.append(params_cell)

        nb.cells.append(
            new_code_cell(
                """import warnings
warnings.filterwarnings('ignore')

from quant_research import DataFetcher, run_ml_analysis
import pandas as pd

fetcher = DataFetcher()
data = fetcher.get_stock_data(symbol, period=period)

print(f"Running ML analysis for {symbol}...")
results = run_ml_analysis(data, task=task, horizon=horizon, verbose=True)

print(f"\\nBest Model: {results['best_model']}")
print(f"Metrics: {results['best_result'].metrics}")
"""
            )
        )

        path = output_dir / "ml_prediction.ipynb"
        with open(path, "w") as f:
            nbformat.write(nb, f)
        return path


def create_all_templates(output_dir: str | Path = "notebooks") -> list[Path]:
    """Create all notebook templates."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = NotebookTemplates()
    paths = [
        templates.strategy_comparison(output_dir),
        templates.portfolio_optimization(output_dir),
        templates.ml_prediction(output_dir),
    ]

    print(f"Created {len(paths)} notebook templates in {output_dir}")
    return paths


def run_notebook_pipeline(
    notebook_path: str | Path,
    parameters: dict | None = None,
    output_dir: str | Path = "outputs",
) -> Path:
    """
    Convenience function to run a notebook.

    Args:
        notebook_path: Path to notebook
        parameters: Parameters to inject
        output_dir: Output directory

    Returns:
        Path to executed notebook
    """
    pipeline = NotebookPipeline(output_dir=output_dir)
    return pipeline.execute(notebook_path, parameters)
