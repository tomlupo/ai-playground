#!/usr/bin/env python3
"""
Test script for BeGlobal strategy implementation.

Tests the full BeGlobal investment methodology:
- Risk profiles
- Dual momentum strategy
- Core-satellite allocation
- Corridor rebalancing
- Volatility targeting
"""

import sys
sys.path.insert(0, ".")

from quant_research import (
    DataFetcher,
    RiskProfile,
    BeGlobalPortfolio,
    DualMomentumStrategy,
    RelativeStrengthStrategy,
    TrendFollowingStrategy,
    CorridorRebalancer,
    VolatilityTargeting,
    create_beglobal_portfolio,
    ASSET_CLASSES,
    RISK_PROFILE_ALLOCATIONS,
)
import pandas as pd
import numpy as np

def test_asset_classes():
    """Test asset class definitions."""
    print("\n=== Testing Asset Classes ===")
    print(f"Total asset classes: {len(ASSET_CLASSES)}")

    categories = {}
    for name, asset in ASSET_CLASSES.items():
        cat = asset.category
        categories[cat] = categories.get(cat, 0) + 1
        print(f"  {name}: {asset.etf_ticker} ({cat})")

    print(f"\nBy category: {categories}")
    assert len(ASSET_CLASSES) == 14, "Expected 14 asset classes"
    print("✓ Asset classes test passed")


def test_risk_profiles():
    """Test risk profile allocations."""
    print("\n=== Testing Risk Profiles ===")

    for profile in RiskProfile:
        allocation = RISK_PROFILE_ALLOCATIONS[profile]
        total = sum(allocation.values())
        print(f"\n{profile.value}:")
        for asset, weight in sorted(allocation.items(), key=lambda x: -x[1]):
            if weight > 0:
                print(f"  {asset}: {weight:.1%}")
        print(f"  Total: {total:.1%}")
        assert abs(total - 1.0) < 0.01, f"Allocation for {profile} should sum to 1.0"

    print("\n✓ Risk profiles test passed")


def test_dual_momentum():
    """Test dual momentum strategy."""
    print("\n=== Testing Dual Momentum Strategy ===")

    # Create synthetic price data
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", "2024-01-01", freq="D")

    # Asset with positive trend
    trending_up = pd.Series(
        100 * np.cumprod(1 + np.random.normal(0.0003, 0.015, len(dates))),
        index=dates
    )

    # Asset with negative trend
    trending_down = pd.Series(
        100 * np.cumprod(1 + np.random.normal(-0.0003, 0.015, len(dates))),
        index=dates
    )

    # Neutral asset
    neutral = pd.Series(
        100 * np.cumprod(1 + np.random.normal(0, 0.01, len(dates))),
        index=dates
    )

    prices = {
        "trending_up": trending_up,
        "trending_down": trending_down,
        "neutral": neutral,
    }

    strategy = DualMomentumStrategy(lookback_period=252)
    signals = strategy.generate_signals(prices)

    print(f"Signals generated for {len(signals)} assets:")
    for asset, signal in signals.items():
        print(f"  {asset}:")
        print(f"    Absolute momentum: {signal.absolute_momentum:.2%}")
        print(f"    Relative momentum: {signal.relative_momentum:.2%}")
        print(f"    Combined signal: {signal.combined_signal}")
        print(f"    Score: {signal.score:.4f}")

    print("\n✓ Dual momentum test passed")


def test_relative_strength():
    """Test relative strength strategy."""
    print("\n=== Testing Relative Strength Strategy ===")

    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")

    # Create 5 assets with different momentum
    prices = {}
    for i, momentum in enumerate([0.0005, 0.0003, 0.0001, -0.0001, -0.0003]):
        prices[f"asset_{i}"] = pd.Series(
            100 * np.cumprod(1 + np.random.normal(momentum, 0.012, len(dates))),
            index=dates
        )

    strategy = RelativeStrengthStrategy(top_n=2)
    rankings = strategy.rank_assets(prices)

    print("Rankings:")
    print(rankings.to_string())

    base_weights = {f"asset_{i}": 0.2 for i in range(5)}
    adjusted = strategy.generate_weights(prices, base_weights)

    print("\nAdjusted weights:")
    for asset, weight in sorted(adjusted.items(), key=lambda x: -x[1]):
        print(f"  {asset}: {weight:.2%}")

    print("\n✓ Relative strength test passed")


def test_corridor_rebalancing():
    """Test corridor rebalancing logic."""
    print("\n=== Testing Corridor Rebalancing ===")

    rebalancer = CorridorRebalancer(threshold=0.025)

    target = {"us_stocks": 0.40, "bonds": 0.40, "gold": 0.20}

    # Within corridor - no rebalance
    current_ok = {"us_stocks": 0.42, "bonds": 0.39, "gold": 0.19}
    needs_rebalance = rebalancer.check_rebalance_needed(current_ok, target)
    print(f"Within corridor: needs_rebalance = {needs_rebalance}")
    assert not needs_rebalance, "Should not need rebalancing within corridor"

    # Outside corridor - needs rebalance
    current_drift = {"us_stocks": 0.50, "bonds": 0.35, "gold": 0.15}
    needs_rebalance = rebalancer.check_rebalance_needed(current_drift, target)
    print(f"Outside corridor: needs_rebalance = {needs_rebalance}")
    assert needs_rebalance, "Should need rebalancing outside corridor"

    actions = rebalancer.generate_rebalance_actions(current_drift, target)
    print("\nRebalance actions:")
    for action in actions:
        print(f"  {action.asset}: {action.action} {action.amount_pct:.2%} "
              f"({action.current_weight:.2%} -> {action.target_weight:.2%})")

    print("\n✓ Corridor rebalancing test passed")


def test_volatility_targeting():
    """Test volatility targeting."""
    print("\n=== Testing Volatility Targeting ===")

    vol_target = VolatilityTargeting(target_volatility=0.10)

    np.random.seed(42)

    # Normal volatility
    normal_returns = pd.Series(np.random.normal(0.0005, 0.006, 100))  # ~9.5% vol
    scalar = vol_target.get_volatility_scalar(normal_returns)
    print(f"Normal vol regime: scalar = {scalar:.2f}")

    # High volatility
    high_vol_returns = pd.Series(np.random.normal(0, 0.02, 100))  # ~32% vol
    scalar = vol_target.get_volatility_scalar(high_vol_returns)
    print(f"High vol regime: scalar = {scalar:.2f}")
    assert scalar < 1.0, "Should reduce exposure in high vol"

    # Low volatility
    low_vol_returns = pd.Series(np.random.normal(0.0003, 0.002, 100))  # ~3% vol
    scalar = vol_target.get_volatility_scalar(low_vol_returns)
    print(f"Low vol regime: scalar = {scalar:.2f}")
    assert scalar > 1.0, "Should increase exposure in low vol"

    print("\n✓ Volatility targeting test passed")


def test_beglobal_portfolio():
    """Test complete BeGlobal portfolio."""
    print("\n=== Testing BeGlobal Portfolio ===")

    # Create portfolio
    portfolio = create_beglobal_portfolio(risk_profile="mixed", core_weight=0.7)

    print(f"Risk profile: {portfolio.risk_profile.value}")
    print(f"Core weight: {portfolio.core_weight:.0%}")
    print(f"Satellite weight: {portfolio.satellite_weight:.0%}")

    print("\nBase allocation:")
    for asset, weight in sorted(portfolio.base_allocation.items(), key=lambda x: -x[1]):
        print(f"  {asset}: {weight:.1%}")

    # Get ETF tickers
    tickers = portfolio.get_etf_tickers()
    print(f"\nETF tickers needed: {tickers}")

    print("\n✓ BeGlobal portfolio creation test passed")


def test_beglobal_backtest():
    """Test BeGlobal backtest with real data."""
    print("\n=== Testing BeGlobal Backtest (with real data) ===")

    # Fetch real ETF data
    fetcher = DataFetcher()

    # Use a subset of ETFs for faster testing
    test_symbols = ["SPY", "SHY", "GLD", "IEF", "SHV"]

    print(f"Fetching data for: {test_symbols}")
    data_dict = fetcher.get_multiple_stocks(test_symbols, period="2y")

    # Extract close prices from each symbol's DataFrame
    prices = pd.DataFrame({
        symbol: df["close"] for symbol, df in data_dict.items()
    })
    prices = prices.dropna()

    print(f"Data shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

    # Create portfolio with matching allocation
    portfolio = BeGlobalPortfolio(
        risk_profile=RiskProfile.MIXED,
        core_weight=0.7,
        satellite_weight=0.3,
    )

    # Run backtest
    print("\nRunning backtest...")
    results = portfolio.run_backtest(prices, initial_capital=100000)

    print("\n=== Backtest Results ===")
    print(f"Final portfolio value: ${results['equity'].iloc[-1]:,.2f}")
    print(f"Total return: {results['metrics']['total_return']:.2%}")
    print(f"Annualized return: {results['metrics']['annual_return']:.2%}")
    print(f"Volatility: {results['metrics']['volatility']:.2%}")
    print(f"Sharpe ratio: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"Max drawdown: {results['metrics']['max_drawdown']:.2%}")
    print(f"Number of rebalances: {results['metrics']['rebalance_count']}")

    print("\nFinal weights:")
    for asset, weight in sorted(results["final_weights"].items(), key=lambda x: -x[1]):
        if weight > 0.01:
            print(f"  {asset}: {weight:.2%}")

    print("\n✓ BeGlobal backtest test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("BeGlobal Strategy Implementation Tests")
    print("=" * 60)

    try:
        test_asset_classes()
        test_risk_profiles()
        test_dual_momentum()
        test_relative_strength()
        test_corridor_rebalancing()
        test_volatility_targeting()
        test_beglobal_portfolio()
        test_beglobal_backtest()

        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
