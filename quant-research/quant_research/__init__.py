"""
Quant Research - A Python quantitative finance research toolkit.

Following the Compound Engineering philosophy:
Plan → Work → Review → Compound → Repeat

Modules:
- data: Market data fetching (Yahoo Finance)
- indicators: Technical indicators (SMA, EMA, RSI, MACD, etc.)
- backtest: Basic backtesting framework
- backtest_advanced: Advanced backtesting with quantstats
- portfolio: Basic portfolio analysis
- portfolio_advanced: Advanced optimization with riskfolio-lib
- ml_pipeline: Machine learning for finance (scikit-learn)
- reporting: HTML report generation (qreporting-style)
- pipeline: Research workflow orchestration
"""

# Core modules
from quant_research.data import DataFetcher
from quant_research.indicators import TechnicalIndicators
from quant_research.backtest import Backtester, Strategy, SMAcrossover, RSIMeanReversion
from quant_research.portfolio import PortfolioAnalyzer
from quant_research.pipeline import PipelineConfig, ResearchPipeline, PipelineResult, run_pipeline

# Advanced modules
from quant_research.ml_pipeline import (
    MLPipeline,
    FeatureEngineer,
    MLResult,
    run_ml_analysis,
)
from quant_research.portfolio_advanced import AdvancedPortfolioOptimizer
from quant_research.backtest_advanced import (
    QuantStatsAnalyzer,
    BacktestEngine,
    analyze_returns,
)
from quant_research.reporting import (
    ReportBuilder,
    QuantReportGenerator,
    generate_report,
)

__version__ = "0.2.0"
__all__ = [
    # Core
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
    # ML
    "MLPipeline",
    "FeatureEngineer",
    "MLResult",
    "run_ml_analysis",
    # Advanced Portfolio
    "AdvancedPortfolioOptimizer",
    # Advanced Backtest
    "QuantStatsAnalyzer",
    "BacktestEngine",
    "analyze_returns",
    # Reporting
    "ReportBuilder",
    "QuantReportGenerator",
    "generate_report",
]
