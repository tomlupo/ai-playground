"""
Pipeline module for orchestrating quant research workflows.

Flow: Config -> Data -> Indicators -> Portfolio -> Backtest -> Analysis
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
import numpy as np

from quant_research.data import DataFetcher
from quant_research.indicators import TechnicalIndicators
from quant_research.portfolio import PortfolioAnalyzer
from quant_research.backtest import Backtester, SMAcrossover, RSIMeanReversion


@dataclass
class PipelineConfig:
    """Configuration container for pipeline execution."""

    name: str
    description: str
    data: dict
    indicators: dict
    portfolio: dict
    backtest: dict
    analysis: dict

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)

        return cls(
            name=config.get("name", "Unnamed Pipeline"),
            description=config.get("description", ""),
            data=config.get("data", {}),
            indicators=config.get("indicators", {}),
            portfolio=config.get("portfolio", {}),
            backtest=config.get("backtest", {}),
            analysis=config.get("analysis", {}),
        )


@dataclass
class PipelineResult:
    """Container for pipeline execution results."""

    config: PipelineConfig
    price_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    combined_prices: pd.DataFrame | None = None
    indicators_data: dict[str, dict[str, pd.Series]] = field(default_factory=dict)
    portfolio_stats: pd.DataFrame | None = None
    portfolio_correlation: pd.DataFrame | None = None
    optimized_portfolios: dict[str, dict] = field(default_factory=dict)
    backtest_results: dict[str, Any] = field(default_factory=dict)
    analysis_summary: dict[str, Any] = field(default_factory=dict)


class ResearchPipeline:
    """
    Orchestrates the complete quant research workflow.

    Stages:
    1. Config: Load and validate configuration
    2. Data: Fetch price data for all symbols
    3. Indicators: Calculate technical indicators
    4. Portfolio: Analyze and optimize portfolio
    5. Backtest: Run strategy backtests
    6. Analysis: Generate summary and metrics
    """

    def __init__(self, config: PipelineConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.result = PipelineResult(config=config)

        # Initialize components
        self.fetcher = DataFetcher()
        self.indicators = TechnicalIndicators()

    def log(self, message: str, stage: str = "") -> None:
        """Log pipeline progress."""
        if self.verbose:
            prefix = f"[{stage}] " if stage else ""
            print(f"{prefix}{message}")

    def run(self) -> PipelineResult:
        """Execute the complete pipeline."""
        self.log(f"Starting pipeline: {self.config.name}")
        self.log(f"Description: {self.config.description}\n")

        self._stage_data()
        self._stage_indicators()
        self._stage_portfolio()
        self._stage_backtest()
        self._stage_analysis()

        self.log("\nPipeline execution complete!")
        return self.result

    def _stage_data(self) -> None:
        """Stage 1: Fetch price data."""
        self.log("Fetching price data...", "DATA")

        symbols = self.config.data.get("symbols", [])
        period = self.config.data.get("period", "1y")

        for symbol in symbols:
            self.log(f"  Fetching {symbol}...", "DATA")
            data = self.fetcher.get_stock_data(symbol, period=period)
            if data is not None and not data.empty:
                self.result.price_data[symbol] = data
                self.log(f"  {symbol}: {len(data)} records", "DATA")

        # Combined close prices for portfolio analysis
        if self.result.price_data:
            self.result.combined_prices = self.fetcher.get_combined_close_prices(
                list(self.result.price_data.keys()), period=period
            )
            self.log(
                f"Combined data shape: {self.result.combined_prices.shape}", "DATA"
            )

    def _stage_indicators(self) -> None:
        """Stage 2: Calculate technical indicators."""
        self.log("\nCalculating technical indicators...", "INDICATORS")

        ind_config = self.config.indicators

        for symbol, data in self.result.price_data.items():
            self.log(f"  Processing {symbol}...", "INDICATORS")
            close = data["close"]
            high = data["high"]
            low = data["low"]

            symbol_indicators: dict[str, pd.Series] = {}

            # SMA
            if "sma" in ind_config:
                for sma_conf in ind_config["sma"]:
                    period = sma_conf.get("period", 20)
                    symbol_indicators[f"sma_{period}"] = self.indicators.sma(
                        close, period
                    )

            # EMA
            if "ema" in ind_config:
                for ema_conf in ind_config["ema"]:
                    period = ema_conf.get("period", 20)
                    symbol_indicators[f"ema_{period}"] = self.indicators.ema(
                        close, period
                    )

            # RSI
            if "rsi" in ind_config:
                period = ind_config["rsi"].get("period", 14)
                symbol_indicators["rsi"] = self.indicators.rsi(close, period)

            # MACD
            if "macd" in ind_config:
                macd_line, signal_line, histogram = self.indicators.macd(
                    close,
                    fast_period=ind_config["macd"].get("fast", 12),
                    slow_period=ind_config["macd"].get("slow", 26),
                    signal_period=ind_config["macd"].get("signal", 9),
                )
                symbol_indicators["macd"] = macd_line
                symbol_indicators["macd_signal"] = signal_line
                symbol_indicators["macd_histogram"] = histogram

            # Bollinger Bands
            if "bollinger" in ind_config:
                upper, middle, lower = self.indicators.bollinger_bands(
                    close,
                    period=ind_config["bollinger"].get("period", 20),
                    std_dev=ind_config["bollinger"].get("std_dev", 2),
                )
                symbol_indicators["bb_upper"] = upper
                symbol_indicators["bb_middle"] = middle
                symbol_indicators["bb_lower"] = lower

            # ATR
            if "atr" in ind_config:
                period = ind_config["atr"].get("period", 14)
                symbol_indicators["atr"] = self.indicators.atr(high, low, close, period)

            # Volatility
            if "volatility" in ind_config:
                period = ind_config["volatility"].get("period", 20)
                symbol_indicators["volatility"] = self.indicators.volatility(
                    close, period
                )

            self.result.indicators_data[symbol] = symbol_indicators
            self.log(
                f"  {symbol}: {len(symbol_indicators)} indicators calculated",
                "INDICATORS",
            )

    def _stage_portfolio(self) -> None:
        """Stage 3: Portfolio analysis and optimization."""
        self.log("\nAnalyzing portfolio...", "PORTFOLIO")

        if self.result.combined_prices is None or self.result.combined_prices.empty:
            self.log("  No price data available for portfolio analysis", "PORTFOLIO")
            return

        port_config = self.config.portfolio
        risk_free_rate = port_config.get("risk_free_rate", 0.05)

        analyzer = PortfolioAnalyzer(
            self.result.combined_prices, risk_free_rate=risk_free_rate
        )

        # Asset statistics
        self.result.portfolio_stats = analyzer.calculate_statistics()
        self.log("  Calculated asset statistics", "PORTFOLIO")

        # Correlation matrix
        self.result.portfolio_correlation = analyzer.correlation_matrix()
        self.log("  Calculated correlation matrix", "PORTFOLIO")

        # Portfolio optimizations
        optimizations = port_config.get("optimization", ["max_sharpe"])

        for opt_type in optimizations:
            if opt_type == "max_sharpe":
                weights, metrics = analyzer.optimize_sharpe()
                self.result.optimized_portfolios["max_sharpe"] = {
                    "weights": weights,
                    "metrics": metrics,
                }
                self.log(
                    f"  Max Sharpe: Return={metrics['return']:.2%}, "
                    f"Vol={metrics['volatility']:.2%}, Sharpe={metrics['sharpe_ratio']:.2f}",
                    "PORTFOLIO",
                )

            elif opt_type == "min_variance":
                weights, metrics = analyzer.optimize_min_variance()
                self.result.optimized_portfolios["min_variance"] = {
                    "weights": weights,
                    "metrics": metrics,
                }
                self.log(
                    f"  Min Variance: Return={metrics['return']:.2%}, "
                    f"Vol={metrics['volatility']:.2%}, Sharpe={metrics['sharpe_ratio']:.2f}",
                    "PORTFOLIO",
                )

            elif opt_type == "equal_weight":
                metrics = analyzer.equal_weight_portfolio()
                n_assets = len(self.result.combined_prices.columns)
                weights = np.array([1 / n_assets] * n_assets)
                self.result.optimized_portfolios["equal_weight"] = {
                    "weights": weights,
                    "metrics": metrics,
                }
                self.log(
                    f"  Equal Weight: Return={metrics['return']:.2%}, "
                    f"Vol={metrics['volatility']:.2%}, Sharpe={metrics['sharpe_ratio']:.2f}",
                    "PORTFOLIO",
                )

        # Risk metrics
        if self.result.optimized_portfolios:
            best_weights = list(self.result.optimized_portfolios.values())[0]["weights"]
            var_conf = self.config.analysis.get("risk_metrics", {}).get(
                "var_confidence", 0.95
            )
            cvar_conf = self.config.analysis.get("risk_metrics", {}).get(
                "cvar_confidence", 0.95
            )

            var = analyzer.var(best_weights, confidence=var_conf)
            cvar = analyzer.cvar(best_weights, confidence=cvar_conf)

            self.result.analysis_summary["var"] = var
            self.result.analysis_summary["cvar"] = cvar
            self.log(
                f"  Risk: VaR({var_conf:.0%})={var:.2%}, CVaR({cvar_conf:.0%})={cvar:.2%}",
                "PORTFOLIO",
            )

    def _stage_backtest(self) -> None:
        """Stage 4: Run strategy backtests."""
        self.log("\nRunning backtests...", "BACKTEST")

        bt_config = self.config.backtest
        strategy_name = bt_config.get("strategy", "sma_crossover")
        params = bt_config.get("params", {})
        commission = bt_config.get("commission", 0.001)
        initial_capital = self.config.portfolio.get("initial_capital", 100000)

        backtester = Backtester(initial_capital=initial_capital, commission=commission)

        # Select strategy
        if strategy_name == "sma_crossover":
            strategy = SMAcrossover(
                short_period=params.get("short_period", 20),
                long_period=params.get("long_period", 50),
            )
        elif strategy_name == "rsi_mean_reversion":
            strategy = RSIMeanReversion(
                period=params.get("period", 14),
                oversold=params.get("oversold", 30),
                overbought=params.get("overbought", 70),
            )
        elif strategy_name == "buy_and_hold":
            # Simple buy and hold - no active trading
            strategy = SMAcrossover(short_period=1, long_period=2)  # Always long
        else:
            self.log(f"  Unknown strategy: {strategy_name}", "BACKTEST")
            return

        # Run backtest on each symbol
        for symbol, data in self.result.price_data.items():
            self.log(f"  Backtesting {symbol} with {strategy_name}...", "BACKTEST")
            result = backtester.run(data, strategy)
            self.result.backtest_results[symbol] = result
            self.log(
                f"  {symbol}: Return={result.total_return:.2%}, "
                f"Sharpe={result.sharpe_ratio:.2f}, MaxDD={result.max_drawdown:.2%}",
                "BACKTEST",
            )

    def _stage_analysis(self) -> None:
        """Stage 5: Generate analysis summary."""
        self.log("\nGenerating analysis summary...", "ANALYSIS")

        # Aggregate backtest results
        if self.result.backtest_results:
            returns = []
            sharpes = []
            drawdowns = []

            for symbol, result in self.result.backtest_results.items():
                returns.append(result.total_return)
                sharpes.append(result.sharpe_ratio)
                drawdowns.append(result.max_drawdown)

            self.result.analysis_summary["backtest"] = {
                "avg_return": np.mean(returns),
                "avg_sharpe": np.mean(sharpes),
                "avg_max_drawdown": np.mean(drawdowns),
                "best_symbol": max(
                    self.result.backtest_results.items(), key=lambda x: x[1].sharpe_ratio
                )[0],
                "worst_symbol": min(
                    self.result.backtest_results.items(), key=lambda x: x[1].sharpe_ratio
                )[0],
            }

        # Portfolio summary
        if self.result.optimized_portfolios:
            best_portfolio = max(
                self.result.optimized_portfolios.items(),
                key=lambda x: x[1]["metrics"]["sharpe_ratio"],
            )
            self.result.analysis_summary["best_portfolio"] = {
                "type": best_portfolio[0],
                "sharpe": best_portfolio[1]["metrics"]["sharpe_ratio"],
                "return": best_portfolio[1]["metrics"]["return"],
                "volatility": best_portfolio[1]["metrics"]["volatility"],
            }

        self.log("  Analysis summary generated", "ANALYSIS")

    def print_summary(self) -> None:
        """Print a formatted summary of results."""
        print("\n" + "=" * 70)
        print(f" PIPELINE RESULTS: {self.config.name}")
        print("=" * 70)

        # Data summary
        print(f"\n{'─'*70}")
        print(" DATA SUMMARY")
        print(f"{'─'*70}")
        print(f"  Symbols analyzed: {len(self.result.price_data)}")
        if self.result.combined_prices is not None:
            print(f"  Date range: {self.result.combined_prices.index[0].date()} to "
                  f"{self.result.combined_prices.index[-1].date()}")
            print(f"  Trading days: {len(self.result.combined_prices)}")

        # Portfolio statistics
        if self.result.portfolio_stats is not None:
            print(f"\n{'─'*70}")
            print(" ASSET STATISTICS")
            print(f"{'─'*70}")
            print(self.result.portfolio_stats.round(4).to_string())

        # Correlation
        if (
            self.result.portfolio_correlation is not None
            and self.config.analysis.get("correlation_analysis", False)
        ):
            print(f"\n{'─'*70}")
            print(" CORRELATION MATRIX")
            print(f"{'─'*70}")
            print(self.result.portfolio_correlation.round(3).to_string())

        # Optimized portfolios
        if self.result.optimized_portfolios:
            print(f"\n{'─'*70}")
            print(" PORTFOLIO OPTIMIZATION")
            print(f"{'─'*70}")
            for name, data in self.result.optimized_portfolios.items():
                metrics = data["metrics"]
                print(f"\n  {name.upper().replace('_', ' ')}:")
                print(f"    Expected Return: {metrics['return']:.2%}")
                print(f"    Volatility:      {metrics['volatility']:.2%}")
                print(f"    Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
                print("    Weights:")
                for asset, weight in metrics["weights"].items():
                    if weight > 0.01:
                        print(f"      {asset}: {weight:.2%}")

        # Backtest results
        if self.result.backtest_results:
            print(f"\n{'─'*70}")
            print(" BACKTEST RESULTS")
            print(f"{'─'*70}")
            print(
                f"  Strategy: {self.config.backtest.get('strategy', 'unknown')}"
            )
            print()

            # Table header
            print(
                f"  {'Symbol':<10} {'Return':>12} {'Ann. Return':>12} "
                f"{'Volatility':>12} {'Sharpe':>10} {'Max DD':>12} {'Trades':>8}"
            )
            print(f"  {'-'*76}")

            for symbol, result in self.result.backtest_results.items():
                print(
                    f"  {symbol:<10} {result.total_return:>11.2%} "
                    f"{result.annualized_return:>11.2%} "
                    f"{result.volatility:>11.2%} "
                    f"{result.sharpe_ratio:>10.2f} "
                    f"{result.max_drawdown:>11.2%} "
                    f"{result.total_trades:>8}"
                )

        # Risk metrics
        if "var" in self.result.analysis_summary:
            print(f"\n{'─'*70}")
            print(" RISK METRICS")
            print(f"{'─'*70}")
            print(f"  Value at Risk (95%):       {self.result.analysis_summary['var']:.2%} daily")
            print(f"  Conditional VaR (95%):     {self.result.analysis_summary['cvar']:.2%} daily")

        # Summary
        if "backtest" in self.result.analysis_summary:
            print(f"\n{'─'*70}")
            print(" SUMMARY")
            print(f"{'─'*70}")
            bt_summary = self.result.analysis_summary["backtest"]
            print(f"  Average Return:       {bt_summary['avg_return']:.2%}")
            print(f"  Average Sharpe:       {bt_summary['avg_sharpe']:.2f}")
            print(f"  Average Max Drawdown: {bt_summary['avg_max_drawdown']:.2%}")
            print(f"  Best Performer:       {bt_summary['best_symbol']}")
            print(f"  Worst Performer:      {bt_summary['worst_symbol']}")

        if "best_portfolio" in self.result.analysis_summary:
            bp = self.result.analysis_summary["best_portfolio"]
            print(f"\n  Best Portfolio: {bp['type'].replace('_', ' ').title()}")
            print(f"    Sharpe Ratio: {bp['sharpe']:.2f}")

        print("\n" + "=" * 70)


def run_pipeline(config_path: str | Path, verbose: bool = True) -> PipelineResult:
    """
    Convenience function to run a pipeline from a config file.

    Args:
        config_path: Path to YAML configuration file
        verbose: Whether to print progress

    Returns:
        PipelineResult with all results
    """
    config = PipelineConfig.from_yaml(config_path)
    pipeline = ResearchPipeline(config, verbose=verbose)
    result = pipeline.run()
    pipeline.print_summary()
    return result
