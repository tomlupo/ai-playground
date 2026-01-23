"""
Quant Research - A Python quantitative finance research toolkit.

Following the Compound Engineering philosophy:
Plan → Work → Review → Compound → Repeat
"""

from quant_research.data import DataFetcher
from quant_research.indicators import TechnicalIndicators
from quant_research.backtest import Backtester, Strategy, SMAcrossover, RSIMeanReversion
from quant_research.portfolio import PortfolioAnalyzer
from quant_research.pipeline import PipelineConfig, ResearchPipeline, PipelineResult, run_pipeline

__version__ = "0.1.0"
__all__ = [
    "DataFetcher",
    "TechnicalIndicators",
    "Backtester",
    "Strategy",
    "SMAcrossover",
    "RSIMeanReversion",
    "PortfolioAnalyzer",
    "PipelineConfig",
    "ResearchPipeline",
    "PipelineResult",
    "run_pipeline",
]
