#!/usr/bin/env python3
"""
Quant Research Example - Demonstrating the toolkit capabilities.

Following the Compound Engineering philosophy:
- Thorough planning and systematic execution
- Each step builds on the previous
- Knowledge is codified for future use
"""

import warnings
warnings.filterwarnings("ignore")

from quant_research import (
    DataFetcher,
    TechnicalIndicators,
    Backtester,
    SMAcrossover,
    RSIMeanReversion,
    PortfolioAnalyzer,
)


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def main():
    # =========================================================
    # 1. DATA FETCHING
    # =========================================================
    section("1. DATA FETCHING")

    fetcher = DataFetcher()

    # Fetch single stock data
    print("Fetching AAPL data (1 year)...")
    aapl_data = fetcher.get_stock_data("AAPL", period="1y")
    print(f"  Retrieved {len(aapl_data)} trading days")
    print(f"  Date range: {aapl_data.index[0].date()} to {aapl_data.index[-1].date()}")
    print(f"  Latest close: ${aapl_data['close'].iloc[-1]:.2f}")

    # Fetch multiple stocks for portfolio analysis
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    print(f"\nFetching portfolio data for: {symbols}")
    prices = fetcher.get_combined_close_prices(symbols, period="1y")
    print(f"  Combined data shape: {prices.shape}")

    # =========================================================
    # 2. TECHNICAL INDICATORS
    # =========================================================
    section("2. TECHNICAL INDICATORS")

    close = aapl_data["close"]
    ti = TechnicalIndicators()

    # Calculate various indicators
    sma_20 = ti.sma(close, period=20)
    ema_20 = ti.ema(close, period=20)
    rsi = ti.rsi(close, period=14)
    macd_line, signal_line, histogram = ti.macd(close)
    upper_bb, middle_bb, lower_bb = ti.bollinger_bands(close)
    volatility = ti.volatility(close, period=20)

    print("AAPL Technical Indicators (Latest Values):")
    print(f"  Close Price:      ${close.iloc[-1]:.2f}")
    print(f"  SMA (20):         ${sma_20.iloc[-1]:.2f}")
    print(f"  EMA (20):         ${ema_20.iloc[-1]:.2f}")
    print(f"  RSI (14):         {rsi.iloc[-1]:.2f}")
    print(f"  MACD:             {macd_line.iloc[-1]:.4f}")
    print(f"  MACD Signal:      {signal_line.iloc[-1]:.4f}")
    print(f"  Bollinger Upper:  ${upper_bb.iloc[-1]:.2f}")
    print(f"  Bollinger Lower:  ${lower_bb.iloc[-1]:.2f}")
    print(f"  Volatility (Ann): {volatility.iloc[-1]:.2%}")

    # RSI analysis
    print(f"\nRSI Analysis:")
    print(f"  Current RSI: {rsi.iloc[-1]:.2f}")
    if rsi.iloc[-1] < 30:
        print("  Signal: OVERSOLD - Potential buy opportunity")
    elif rsi.iloc[-1] > 70:
        print("  Signal: OVERBOUGHT - Potential sell opportunity")
    else:
        print("  Signal: NEUTRAL")

    # =========================================================
    # 3. BACKTESTING STRATEGIES
    # =========================================================
    section("3. BACKTESTING STRATEGIES")

    backtester = Backtester(initial_capital=100000, commission=0.001)

    # Strategy 1: SMA Crossover
    print("Strategy 1: SMA Crossover (20/50)")
    sma_strategy = SMAcrossover(short_period=20, long_period=50)
    sma_result = backtester.run(aapl_data, sma_strategy)

    print(f"  Total Return:     {sma_result.total_return:.2%}")
    print(f"  Annual Return:    {sma_result.annualized_return:.2%}")
    print(f"  Volatility:       {sma_result.volatility:.2%}")
    print(f"  Sharpe Ratio:     {sma_result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:     {sma_result.max_drawdown:.2%}")
    print(f"  Win Rate:         {sma_result.win_rate:.2%}")
    print(f"  Total Trades:     {sma_result.total_trades}")

    # Strategy 2: RSI Mean Reversion
    print("\nStrategy 2: RSI Mean Reversion (14, 30/70)")
    rsi_strategy = RSIMeanReversion(period=14, oversold=30, overbought=70)
    rsi_result = backtester.run(aapl_data, rsi_strategy)

    print(f"  Total Return:     {rsi_result.total_return:.2%}")
    print(f"  Annual Return:    {rsi_result.annualized_return:.2%}")
    print(f"  Volatility:       {rsi_result.volatility:.2%}")
    print(f"  Sharpe Ratio:     {rsi_result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:     {rsi_result.max_drawdown:.2%}")
    print(f"  Win Rate:         {rsi_result.win_rate:.2%}")
    print(f"  Total Trades:     {rsi_result.total_trades}")

    # Compare strategies
    print("\nStrategy Comparison:")
    comparison = backtester.compare_strategies(
        aapl_data,
        {"SMA Crossover": sma_strategy, "RSI Mean Reversion": rsi_strategy},
    )
    print(comparison.to_string())

    # =========================================================
    # 4. PORTFOLIO ANALYSIS
    # =========================================================
    section("4. PORTFOLIO ANALYSIS")

    analyzer = PortfolioAnalyzer(prices, risk_free_rate=0.05)

    # Individual asset statistics
    print("Individual Asset Statistics:")
    stats = analyzer.calculate_statistics()
    print(stats.round(4).to_string())

    # Correlation matrix
    print("\nCorrelation Matrix:")
    corr = analyzer.correlation_matrix()
    print(corr.round(3).to_string())

    # Equal weight portfolio
    print("\nEqual Weight Portfolio:")
    eq_portfolio = analyzer.equal_weight_portfolio()
    print(f"  Expected Return: {eq_portfolio['return']:.2%}")
    print(f"  Volatility:      {eq_portfolio['volatility']:.2%}")
    print(f"  Sharpe Ratio:    {eq_portfolio['sharpe_ratio']:.2f}")

    # Optimize for maximum Sharpe ratio
    print("\nMaximum Sharpe Ratio Portfolio:")
    opt_weights, opt_metrics = analyzer.optimize_sharpe()
    print(f"  Expected Return: {opt_metrics['return']:.2%}")
    print(f"  Volatility:      {opt_metrics['volatility']:.2%}")
    print(f"  Sharpe Ratio:    {opt_metrics['sharpe_ratio']:.2f}")
    print("  Optimal Weights:")
    for asset, weight in opt_metrics["weights"].items():
        if weight > 0.01:  # Only show significant weights
            print(f"    {asset}: {weight:.2%}")

    # Minimum variance portfolio
    print("\nMinimum Variance Portfolio:")
    mv_weights, mv_metrics = analyzer.optimize_min_variance()
    print(f"  Expected Return: {mv_metrics['return']:.2%}")
    print(f"  Volatility:      {mv_metrics['volatility']:.2%}")
    print(f"  Sharpe Ratio:    {mv_metrics['sharpe_ratio']:.2f}")
    print("  Optimal Weights:")
    for asset, weight in mv_metrics["weights"].items():
        if weight > 0.01:
            print(f"    {asset}: {weight:.2%}")

    # Risk metrics
    print("\nRisk Metrics (Equal Weight Portfolio):")
    eq_weights = [0.25, 0.25, 0.25, 0.25]
    var_95 = analyzer.var(eq_weights, confidence=0.95)
    cvar_95 = analyzer.cvar(eq_weights, confidence=0.95)
    print(f"  VaR (95%):  {var_95:.2%} daily")
    print(f"  CVaR (95%): {cvar_95:.2%} daily")

    # =========================================================
    # 5. SUMMARY
    # =========================================================
    section("5. RESEARCH SUMMARY")

    print("Key Findings:")
    print(f"  - Analyzed {len(symbols)} stocks over {len(prices)} trading days")
    print(f"  - Best performing strategy: ", end="")
    if sma_result.sharpe_ratio > rsi_result.sharpe_ratio:
        print(f"SMA Crossover (Sharpe: {sma_result.sharpe_ratio:.2f})")
    else:
        print(f"RSI Mean Reversion (Sharpe: {rsi_result.sharpe_ratio:.2f})")

    best_weight_asset = max(opt_metrics["weights"].items(), key=lambda x: x[1])
    print(f"  - Highest optimal weight: {best_weight_asset[0]} ({best_weight_asset[1]:.2%})")

    print("\nQuant Research toolkit demonstration complete!")
    print("All modules working correctly.")


if __name__ == "__main__":
    main()
