# Quant Research Toolkit

A comprehensive Python quantitative finance research toolkit built with modern best practices and powered by industry-standard libraries.

## Features

### Core Capabilities
- **Data Fetching**: Yahoo Finance integration with caching
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and more
- **Backtesting**: Strategy backtesting with performance metrics
- **Portfolio Analysis**: Optimization, risk metrics, efficient frontier

### Advanced Features
- **Machine Learning**: Return prediction and direction classification with scikit-learn
- **Advanced Portfolio Optimization**: Risk parity, HRP, CVaR optimization via riskfolio-lib
- **Performance Analytics**: Comprehensive metrics with quantstats
- **HTML Reporting**: Interactive reports with Plotly visualizations
- **Vectorized Backtesting**: Fast numpy-based backtesting with parameter optimization
- **Caching**: Disk and memory caching for expensive operations
- **Notebook Pipelines**: Parameterized notebook execution with papermill

### Pipeline System
- YAML-based configuration
- CLI interface for running research flows
- Automated workflow: Config → Data → Indicators → Portfolio → Backtest → Analysis
- Built-in caching for faster re-runs

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd quant-research

# Install with uv
uv sync

# Or install dependencies manually
uv add pandas numpy yfinance scipy matplotlib scikit-learn quantstats riskfolio-lib plotly
```

## Quick Start

### Basic Usage

```python
from quant_research import (
    DataFetcher,
    TechnicalIndicators,
    Backtester,
    SMAcrossover,
    PortfolioAnalyzer,
)

# Fetch data
fetcher = DataFetcher()
data = fetcher.get_stock_data("AAPL", period="1y")

# Calculate indicators
ti = TechnicalIndicators()
sma_20 = ti.sma(data["close"], period=20)
rsi = ti.rsi(data["close"], period=14)

# Run backtest
backtester = Backtester(initial_capital=100000)
strategy = SMAcrossover(short_period=20, long_period=50)
result = backtester.run(data, strategy)

print(f"Return: {result.total_return:.2%}")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
```

### Machine Learning Pipeline

```python
from quant_research import run_ml_analysis, DataFetcher

# Fetch data
fetcher = DataFetcher()
data = fetcher.get_stock_data("AAPL", period="2y")

# Run ML analysis (direction prediction)
results = run_ml_analysis(
    data,
    task="classification",
    horizon=5,  # 5-day prediction
    verbose=True,
)

# Get best model
print(f"Best Model: {results['best_model']}")
print(f"Accuracy: {results['best_result'].metrics['accuracy']:.2%}")

# View feature importance
for feat, imp in results['pipeline'].get_top_features(results['best_result'], 5):
    print(f"  {feat}: {imp:.4f}")
```

### Advanced Portfolio Optimization

```python
from quant_research import DataFetcher, AdvancedPortfolioOptimizer

# Fetch multi-asset data
fetcher = DataFetcher()
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
prices = fetcher.get_combined_close_prices(symbols, period="2y")

# Initialize optimizer
optimizer = AdvancedPortfolioOptimizer(prices, risk_free_rate=0.05)

# Mean-Variance Optimization
weights, metrics = optimizer.optimize_mean_variance(objective="Sharpe")
print(f"Max Sharpe Portfolio: {metrics['sharpe_ratio']:.2f}")

# Risk Parity
weights_rp, metrics_rp = optimizer.optimize_risk_parity()
print(f"Risk Parity Sharpe: {metrics_rp['sharpe_ratio']:.2f}")

# Hierarchical Risk Parity
weights_hrp, metrics_hrp = optimizer.optimize_hrp()
print(f"HRP Sharpe: {metrics_hrp['sharpe_ratio']:.2f}")

# Compare all strategies
comparison = optimizer.compare_strategies()
print(comparison)
```

### Performance Analytics with QuantStats

```python
from quant_research import analyze_returns, DataFetcher

# Get returns
fetcher = DataFetcher()
data = fetcher.get_stock_data("AAPL", period="1y")
returns = data["close"].pct_change().dropna()

# Comprehensive analysis
results = analyze_returns(
    returns,
    output_report="aapl_report.html",  # Generate HTML report
)

# View metrics
print(f"Sharpe: {results['metrics']['sharpe']:.2f}")
print(f"Sortino: {results['metrics']['sortino']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
```

### HTML Report Generation

```python
from quant_research import ReportBuilder, QuantReportGenerator
import pandas as pd

# Method 1: Manual report building
builder = ReportBuilder("My Research Report")

builder.add_section("Overview")
builder.add_content("Overview", "Summary", {"Total Return": 0.15, "Sharpe": 1.2})
builder.add_content("Overview", "Statistics", my_dataframe)

builder.save("report.html")

# Method 2: From pipeline results
generator = QuantReportGenerator("Pipeline Report")
generator.generate_from_pipeline(pipeline_result)
generator.save("pipeline_report.html")
```

### Vectorized Backtesting (Fast)

```python
from quant_research import VectorizedBacktester, ParameterOptimizer

# Fast vectorized backtester
backtester = VectorizedBacktester()

# Run different strategies
result = backtester.run_sma_crossover(prices, short_period=20, long_period=50)
result = backtester.run_rsi_strategy(prices, period=14, oversold=30, overbought=70)
result = backtester.run_bollinger_strategy(prices, period=20, std_dev=2)
result = backtester.run_macd_strategy(prices, fast=12, slow=26, signal=9)

print(f"Return: {result.total_return:.2%}")
print(f"Sharpe: {result.sharpe_ratio:.2f}")

# Parameter optimization with grid search
optimizer = ParameterOptimizer(backtester)
best_params, results = optimizer.grid_search(
    prices,
    strategy="sma",
    param_grid={
        "short_period": [10, 20, 30],
        "long_period": [50, 100, 200],
    },
)
print(f"Best params: {best_params}")

# Walk-forward optimization
wf_results = optimizer.walk_forward(
    prices,
    strategy="sma",
    param_grid={"short_period": [10, 20], "long_period": [50, 100]},
    train_size=252,
    test_size=63,
)
```

### Caching for Performance

```python
from quant_research import (
    CacheManager,
    DataCache,
    cached,
    clear_all_caches,
    get_cache_info,
)

# Use the @cached decorator
@cached(ttl=3600, key_prefix="my_analysis")
def expensive_analysis(data):
    # This result will be cached for 1 hour
    return complex_computation(data)

# Data caching for market data
data_cache = DataCache()
data_cache.set("AAPL", df)
cached_data = data_cache.get("AAPL")

# Cache management
info = get_cache_info()
print(f"Cache size: {info['total_size_mb']:.2f} MB")
clear_all_caches()  # Clear all caches
```

### Notebook Pipelines with Papermill

```python
from quant_research import (
    NotebookPipeline,
    create_all_templates,
    run_notebook_pipeline,
)

# Create notebook templates
create_all_templates("notebooks/")

# Execute notebook with parameters
output = run_notebook_pipeline(
    "notebooks/strategy_comparison.ipynb",
    parameters={
        "symbol": "AAPL",
        "period": "2y",
        "strategies": ["sma", "rsi", "macd"],
    },
)

# Batch execution
pipeline = NotebookPipeline()
outputs = pipeline.execute_batch(
    "notebooks/ml_prediction.ipynb",
    parameter_sets=[
        {"symbol": "AAPL", "horizon": 5},
        {"symbol": "MSFT", "horizon": 5},
        {"symbol": "GOOGL", "horizon": 5},
    ],
)
```

## Pipeline System

### Running Pipelines via CLI

```bash
# List available configurations
uv run python scripts/run_pipeline.py --list

# Run a specific config
uv run python scripts/run_pipeline.py -c config/tech_momentum.yaml

# Run with quiet mode
uv run python scripts/run_pipeline.py -c config/mean_reversion.yaml -q

# Run by config name
uv run python scripts/run_pipeline.py run tech_momentum

# Run all configs
uv run python scripts/run_pipeline.py all
```

### Configuration Structure

```yaml
name: "Strategy Name"
description: "Strategy description"

data:
  symbols:
    - AAPL
    - MSFT
  period: "1y"
  interval: "1d"

indicators:
  sma:
    - period: 20
    - period: 50
  rsi:
    period: 14
  macd:
    fast: 12
    slow: 26
    signal: 9

portfolio:
  initial_capital: 100000
  risk_free_rate: 0.05
  optimization:
    - max_sharpe
    - min_variance

backtest:
  strategy: sma_crossover
  params:
    short_period: 20
    long_period: 50
  commission: 0.001

analysis:
  metrics:
    - total_return
    - sharpe_ratio
    - max_drawdown
  export:
    format: "console"
```

### Available Configs

| Config | Strategy | Description |
|--------|----------|-------------|
| `tech_momentum.yaml` | SMA Crossover | Momentum on large-cap tech stocks |
| `mean_reversion.yaml` | RSI Mean Reversion | Mean reversion on blue-chip stocks |
| `portfolio_optimization.yaml` | Multi-Asset | Diversified portfolio with ETFs |
| `parameter_optimization.yaml` | Grid Search | Strategy parameter optimization |
| `sector_rotation.yaml` | Sector Rotation | Momentum-based sector ETF rotation |
| `multi_strategy.yaml` | Ensemble | Multiple strategies with voting |
| `ml_prediction.yaml` | ML Prediction | Machine learning direction prediction |
| `notebook_analysis.yaml` | Notebook Pipeline | Execute parameterized notebooks |

## Module Reference

### Core Modules

| Module | Description | Key Classes |
|--------|-------------|-------------|
| `data` | Data fetching | `DataFetcher` |
| `indicators` | Technical indicators | `TechnicalIndicators` |
| `backtest` | Basic backtesting | `Backtester`, `Strategy` |
| `portfolio` | Portfolio analysis | `PortfolioAnalyzer` |
| `pipeline` | Workflow orchestration | `ResearchPipeline` |

### Advanced Modules

| Module | Library | Description |
|--------|---------|-------------|
| `ml_pipeline` | scikit-learn | ML for return/direction prediction |
| `portfolio_advanced` | riskfolio-lib | Advanced optimization (HRP, CVaR, etc.) |
| `backtest_advanced` | quantstats | Comprehensive performance analytics |
| `backtest_vectorized` | numpy | Fast vectorized backtesting & optimization |
| `reporting` | plotly | Interactive HTML reports |
| `caching` | diskcache/joblib | Disk and memory caching |
| `notebook_pipeline` | papermill | Parameterized notebook execution |

## Technical Indicators

| Indicator | Method | Parameters |
|-----------|--------|------------|
| SMA | `sma()` | period |
| EMA | `ema()` | period |
| RSI | `rsi()` | period |
| MACD | `macd()` | fast, slow, signal |
| Bollinger Bands | `bollinger_bands()` | period, std_dev |
| ATR | `atr()` | period |
| Volatility | `volatility()` | period |
| Momentum | `momentum()` | period |
| ROC | `rate_of_change()` | period |

## ML Models

### Regression (Return Prediction)
- Ridge, Lasso, ElasticNet
- Random Forest Regressor
- Gradient Boosting Regressor
- SVR

### Classification (Direction Prediction)
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- SVC

### Feature Engineering
Automatically generates 40+ features including:
- Returns at multiple horizons
- Momentum indicators
- Moving average ratios
- RSI, MACD, Bollinger Band position
- Volume features
- Volatility measures

## Dependencies

### Core
- pandas, numpy - Data manipulation
- yfinance - Market data
- scipy - Optimization

### Advanced
- scikit-learn - Machine learning
- riskfolio-lib - Portfolio optimization
- quantstats - Performance analytics
- plotly - Visualizations

### Performance
- joblib - Parallel processing & caching
- diskcache - Disk-based caching

### Notebooks
- papermill - Parameterized notebook execution
- nbformat - Notebook manipulation
- ipykernel - Jupyter kernel

### CLI
- click - Command line interface
- rich - Terminal formatting
- pyyaml - Configuration files

## Project Structure

```
quant-research/
├── config/                      # Pipeline configurations
│   ├── tech_momentum.yaml       # Tech stock momentum strategy
│   ├── mean_reversion.yaml      # RSI mean reversion strategy
│   ├── portfolio_optimization.yaml  # Multi-asset optimization
│   ├── parameter_optimization.yaml  # Grid search optimization
│   ├── sector_rotation.yaml     # Sector ETF rotation
│   ├── multi_strategy.yaml      # Ensemble strategies
│   ├── ml_prediction.yaml       # ML direction prediction
│   └── notebook_analysis.yaml   # Notebook pipeline
├── notebooks/                   # Jupyter notebook templates
├── scripts/
│   └── run_pipeline.py          # CLI interface
├── quant_research/
│   ├── __init__.py              # Package exports
│   ├── data.py                  # Data fetching
│   ├── indicators.py            # Technical indicators
│   ├── backtest.py              # Basic backtesting
│   ├── backtest_advanced.py     # QuantStats integration
│   ├── backtest_vectorized.py   # Fast vectorized backtesting
│   ├── portfolio.py             # Basic portfolio
│   ├── portfolio_advanced.py    # Riskfolio integration
│   ├── ml_pipeline.py           # ML models
│   ├── pipeline.py              # Research workflow (with caching)
│   ├── reporting.py             # HTML reports
│   ├── caching.py               # Disk/memory caching
│   └── notebook_pipeline.py     # Papermill integration
├── main.py                      # Example script
├── pyproject.toml               # Project config
└── README.md
```

## License

MIT License

## Contributing

Contributions welcome! Please follow the Compound Engineering philosophy:
1. **Plan** (80%): Design thoroughly before coding
2. **Work** (20%): Execute the plan
3. **Review**: Validate results
4. **Compound**: Document learnings for future use
