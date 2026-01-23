"""
Quant Research - A Python quantitative finance research toolkit.

Following the Compound Engineering philosophy:
Plan → Work → Review → Compound → Repeat

Modules:
- data: Market data fetching (Yahoo Finance)
- indicators: Technical indicators (SMA, EMA, RSI, MACD, etc.)
- backtest: Basic backtesting framework
- backtest_advanced: Advanced backtesting with quantstats
- backtest_vectorized: Fast vectorized backtesting (vectorbt-style)
- portfolio: Basic portfolio analysis
- portfolio_advanced: Advanced optimization with riskfolio-lib
- ml_pipeline: Machine learning for finance (scikit-learn)
- reporting: HTML report generation (qreporting-style)
- pipeline: Research workflow orchestration
- caching: Disk and memory caching for expensive operations
- notebook_pipeline: Jupyter notebook execution with papermill
"""

# Core modules
from quant_research.data import DataFetcher
from quant_research.indicators import TechnicalIndicators
from quant_research.backtest import Backtester, Strategy, SMAcrossover, RSIMeanReversion
from quant_research.portfolio import PortfolioAnalyzer
from quant_research.pipeline import (
    PipelineConfig,
    ResearchPipeline,
    CachedResearchPipeline,
    PipelineResult,
    run_pipeline,
)

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
from quant_research.backtest_vectorized import (
    VectorizedBacktester,
    VectorizedResult,
    ParameterOptimizer,
    MultiAssetBacktester,
)
from quant_research.reporting import (
    ReportBuilder,
    QuantReportGenerator,
    generate_report,
)
from quant_research.caching import (
    CacheManager,
    DataCache,
    IndicatorCache,
    ModelCache,
    cached,
    cache_dataframe,
    clear_all_caches,
    get_cache_info,
)
from quant_research.notebook_pipeline import (
    NotebookPipeline,
    NotebookTemplates,
    create_all_templates,
    run_notebook_pipeline,
)

__version__ = "0.3.0"
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
    "CachedResearchPipeline",
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
    # Vectorized Backtest
    "VectorizedBacktester",
    "VectorizedResult",
    "ParameterOptimizer",
    "MultiAssetBacktester",
    # Reporting
    "ReportBuilder",
    "QuantReportGenerator",
    "generate_report",
    # Caching
    "CacheManager",
    "DataCache",
    "IndicatorCache",
    "ModelCache",
    "cached",
    "cache_dataframe",
    "clear_all_caches",
    "get_cache_info",
    # Notebook Pipeline
    "NotebookPipeline",
    "NotebookTemplates",
    "create_all_templates",
    "run_notebook_pipeline",
]
