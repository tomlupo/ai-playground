#!/usr/bin/env python3
"""
Market Behavior Simulator - Example Usage

Demonstrates stress testing and forecasting capabilities:
1. Multi-asset Monte Carlo simulation with correlated returns
2. Correlation structure analysis (static, rolling, DCC)
3. Historical and hypothetical stress scenarios
4. Multi-term expected returns forecasting
5. Risk metrics (VaR, CVaR, drawdown analysis)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from market_simulator import (
    MarketSimulator,
    GBM,
    JumpDiffusion,
    GARCH,
    MeanReversion,
    CorrelationEngine,
    StressTestEngine,
    ReturnForecaster,
    SimulationPlotter
)
from market_simulator.core.simulation import SimulationConfig
from market_simulator.core.models import GBMParams, JumpDiffusionParams, GARCHParams
from market_simulator.stress_testing.scenarios import HistoricalScenario, StressScenario
from market_simulator.forecasting.returns import Horizon


def generate_sample_data(n_days: int = 756, n_assets: int = 4, seed: int = 42) -> pd.DataFrame:
    """Generate sample historical returns for demonstration."""
    np.random.seed(seed)

    # Asset characteristics (annual)
    asset_params = {
        'US_Equity': {'mu': 0.10, 'sigma': 0.18},
        'Intl_Equity': {'mu': 0.08, 'sigma': 0.22},
        'Bonds': {'mu': 0.04, 'sigma': 0.06},
        'Commodities': {'mu': 0.05, 'sigma': 0.25}
    }

    # Correlation matrix
    corr = np.array([
        [1.00, 0.75, -0.10, 0.30],
        [0.75, 1.00, -0.05, 0.35],
        [-0.10, -0.05, 1.00, 0.05],
        [0.30, 0.35, 0.05, 1.00]
    ])

    L = np.linalg.cholesky(corr)

    # Generate returns
    Z = np.random.randn(n_days, n_assets)
    Z_correlated = Z @ L.T

    returns = {}
    for i, (name, params) in enumerate(asset_params.items()):
        daily_mu = params['mu'] / 252
        daily_sigma = params['sigma'] / np.sqrt(252)
        returns[name] = daily_mu + daily_sigma * Z_correlated[:, i]

    return pd.DataFrame(returns)


def demo_monte_carlo_simulation():
    """Demonstrate Monte Carlo simulation with correlated assets."""
    print("\n" + "="*60)
    print("1. MONTE CARLO SIMULATION WITH CORRELATED ASSETS")
    print("="*60)

    # Create simulator with different models for each asset
    simulator = MarketSimulator()

    # Add assets with different dynamics
    simulator.add_asset(
        'US_Equity',
        GBM(GBMParams(mu=0.10, sigma=0.18)),
        initial_price=100
    )
    simulator.add_asset(
        'Intl_Equity',
        JumpDiffusion(JumpDiffusionParams(mu=0.08, sigma=0.20, jump_intensity=0.1, jump_mean=-0.03, jump_std=0.05)),
        initial_price=100
    )
    simulator.add_asset(
        'Bonds',
        GBM(GBMParams(mu=0.04, sigma=0.06)),
        initial_price=100
    )
    simulator.add_asset(
        'Commodities',
        GBM(GBMParams(mu=0.05, sigma=0.28)),
        initial_price=100
    )

    # Set correlation matrix
    corr = np.array([
        [1.00, 0.75, -0.15, 0.30],
        [0.75, 1.00, -0.10, 0.40],
        [-0.15, -0.10, 1.00, 0.00],
        [0.30, 0.40, 0.00, 1.00]
    ])
    simulator.set_correlation_matrix(corr)

    # Run simulation
    config = SimulationConfig(
        n_paths=10000,
        n_steps=252,  # 1 year
        random_state=42
    )

    result = simulator.simulate(config)

    # Print statistics
    print("\nSimulation Statistics (1-year horizon):")
    print("-" * 50)
    stats = result.get_path_statistics()
    print(stats[['mean_return', 'std_return', 'var_95', 'cvar_95', 'sharpe_ratio']].to_string())

    # Terminal return distribution
    terminal = result.get_terminal_returns()
    print("\n\nTerminal Return Percentiles:")
    print("-" * 50)
    for asset in terminal.columns:
        p5, p50, p95 = np.percentile(terminal[asset], [5, 50, 95])
        print(f"{asset}: 5th={p5:+.1%}, Median={p50:+.1%}, 95th={p95:+.1%}")

    return result


def demo_correlation_analysis():
    """Demonstrate correlation structure analysis."""
    print("\n" + "="*60)
    print("2. CORRELATION STRUCTURE ANALYSIS")
    print("="*60)

    # Generate sample data
    returns = generate_sample_data(n_days=756)  # 3 years

    # Create correlation engine
    engine = CorrelationEngine(returns)

    # Static correlation
    static = engine.static_correlation()
    print("\nStatic (Sample) Correlation Matrix:")
    print("-" * 50)
    print(static.to_dataframe().round(3).to_string())

    # EWMA correlation
    ewma = engine.ewma_correlation(lambda_param=0.94)
    print("\nCurrent EWMA Correlation (λ=0.94):")
    print("-" * 50)
    print(pd.DataFrame(ewma.current_correlation,
                      index=returns.columns,
                      columns=returns.columns).round(3).to_string())

    # DCC correlation
    print("\nFitting DCC model...")
    a_opt, b_opt, dcc = engine.fit_dcc()
    print(f"Optimal DCC parameters: α={a_opt:.4f}, β={b_opt:.4f}")
    print("Current DCC Correlation:")
    print(pd.DataFrame(dcc.current_correlation,
                      index=returns.columns,
                      columns=returns.columns).round(3).to_string())

    # Ledoit-Wolf shrinkage
    lw = engine.ledoit_wolf_shrinkage()
    print(f"\nLedoit-Wolf Shrinkage Correlation (intensity in method: {lw.method}):")
    print("-" * 50)
    print(pd.DataFrame(lw.current_correlation,
                      index=returns.columns,
                      columns=returns.columns).round(3).to_string())

    # Stressed correlation
    stressed = engine.correlation_stress(stress_factor=1.5)
    print("\nStressed Correlation (1.5x off-diagonal):")
    print("-" * 50)
    print(pd.DataFrame(stressed,
                      index=returns.columns,
                      columns=returns.columns).round(3).to_string())

    return engine


def demo_stress_testing():
    """Demonstrate stress testing capabilities."""
    print("\n" + "="*60)
    print("3. STRESS TESTING")
    print("="*60)

    # Generate sample data
    returns = generate_sample_data(n_days=504)  # 2 years

    # Portfolio weights
    weights = {
        'US_Equity': 0.40,
        'Intl_Equity': 0.20,
        'Bonds': 0.30,
        'Commodities': 0.10
    }

    # Create stress test engine
    engine = StressTestEngine(returns=returns, weights=weights)

    # Run historical scenarios
    print("\nHistorical Stress Scenarios:")
    print("-" * 50)

    scenarios = [
        HistoricalScenario.FINANCIAL_CRISIS_2008,
        HistoricalScenario.COVID_CRASH_2020,
        HistoricalScenario.DOT_COM_BUST_2000,
        HistoricalScenario.VOLMAGEDDON_2018
    ]

    results = []
    for scenario in scenarios:
        result = engine.run_historical_scenario(scenario, n_simulations=5000)
        results.append({
            'Scenario': result.scenario.name,
            'Portfolio Impact': f"{result.portfolio_impact:+.1%}",
            'VaR 95%': f"{result.var_95:+.1%}",
            'CVaR 95%': f"{result.cvar_95:+.1%}",
            'Max Drawdown': f"{result.max_drawdown:.1%}"
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Custom stress scenario
    print("\n\nCustom Stress Scenario - Rate Shock:")
    print("-" * 50)

    rate_shock = StressScenario(
        name="Interest Rate Shock",
        description="Rapid 200bp rate increase",
        equity_shock=-0.15,
        bond_shock=-0.10,
        volatility_multiplier=2.0,
        correlation_stress=1.3,
        duration_days=60,
        custom_shocks={'Commodities': -0.05}
    )

    custom_result = engine.run_custom_scenario(rate_shock, n_simulations=5000)
    print(f"Portfolio Impact: {custom_result.portfolio_impact:+.1%}")
    print(f"VaR 95%: {custom_result.var_95:+.1%}")
    print(f"CVaR 95%: {custom_result.cvar_95:+.1%}")
    print("\nPer-Asset Impacts:")
    for asset, impact in custom_result.asset_impacts.items():
        print(f"  {asset}: {impact:+.1%}")

    # Sensitivity analysis
    print("\n\nEquity Shock Sensitivity Analysis:")
    print("-" * 50)

    sensitivity = engine.sensitivity_analysis(
        shock_variable='equity',
        shock_range=np.linspace(-0.40, 0, 9),
        n_simulations=2000
    )
    print(sensitivity[['shock_level', 'portfolio_impact', 'var_95', 'max_drawdown']].to_string(index=False))

    # VaR and CVaR
    print("\n\nRisk Metrics (Historical Method):")
    print("-" * 50)

    var = engine.var_calculation(confidence_levels=[0.95, 0.99], horizon_days=10)
    cvar = engine.cvar_calculation(confidence_levels=[0.95, 0.99], horizon_days=10)

    for level in [0.95, 0.99]:
        print(f"{int(level*100)}% confidence (10-day):")
        print(f"  VaR: {var[level]:+.2%}")
        print(f"  CVaR: {cvar[level]:+.2%}")

    return engine


def demo_return_forecasting():
    """Demonstrate multi-term return forecasting."""
    print("\n" + "="*60)
    print("4. MULTI-TERM RETURN FORECASTING")
    print("="*60)

    # Generate sample data
    returns = generate_sample_data(n_days=756)  # 3 years

    # Create forecaster
    forecaster = ReturnForecaster(returns, risk_free_rate=0.05)

    # Multi-horizon forecast for US Equity
    print("\nMulti-Horizon Forecasts (US_Equity) - Historical Method:")
    print("-" * 50)

    forecast = forecaster.forecast_multi_horizon(
        asset='US_Equity',
        horizons=[Horizon.WEEKLY, Horizon.MONTHLY, Horizon.QUARTERLY, Horizon.ANNUAL],
        method='historical'
    )

    term_structure = forecast.term_structure
    print(term_structure[['horizon_name', 'expected_return', 'volatility',
                          'annualized_return', 'sharpe_ratio',
                          'ci_95_lower', 'ci_95_upper']].to_string(index=False))

    # Different forecasting methods comparison
    print("\n\nForecasting Methods Comparison (1-year horizon):")
    print("-" * 50)

    methods = ['historical', 'bootstrap', 'parametric', 'garch']

    for method in methods:
        try:
            result = forecaster.forecast_single_horizon('US_Equity', Horizon.ANNUAL, method=method)
            ci = result.confidence_intervals.get(0.95, (0, 0))
            print(f"{method.capitalize():12s}: Expected={result.expected_return:+.2%}, "
                  f"Vol={result.volatility:.2%}, 95% CI=[{ci[0]:+.2%}, {ci[1]:+.2%}]")
        except Exception as e:
            print(f"{method.capitalize():12s}: Error - {str(e)[:40]}")

    # Mean reversion forecast
    print("\n\nMean Reversion Forecast (US_Equity, 1-year):")
    print("-" * 50)

    mr_forecast = forecaster.expected_return_with_mean_reversion(
        asset='US_Equity',
        horizon=252,
        long_term_return=0.08,
        mean_reversion_speed=0.3
    )

    print(f"Expected Return: {mr_forecast.expected_return:+.2%}")
    print(f"Volatility: {mr_forecast.volatility:.2%}")
    print(f"Current (recent): {mr_forecast.distribution_params['current_return']:+.2%}")
    print(f"Long-term mean: {mr_forecast.distribution_params['long_term_return']:+.2%}")

    # Bayesian shrinkage forecast
    print("\n\nBayesian Shrinkage Forecast (US_Equity, 1-year):")
    print("-" * 50)

    bayes_forecast = forecaster.bayesian_shrinkage_forecast(
        asset='US_Equity',
        horizon=252,
        prior_return=0.07,
        prior_weight=0.4
    )

    print(f"Historical estimate: {bayes_forecast.distribution_params['historical_return']:+.2%}")
    print(f"Prior: {bayes_forecast.distribution_params['prior_return']:+.2%}")
    print(f"Shrunk estimate: {bayes_forecast.distribution_params['shrunk_return']:+.2%}")
    print(f"Expected (horizon): {bayes_forecast.expected_return:+.2%}")

    # Portfolio forecast
    print("\n\nCorrelation-Adjusted Portfolio Forecast:")
    print("-" * 50)

    weights = {'US_Equity': 0.4, 'Intl_Equity': 0.2, 'Bonds': 0.3, 'Commodities': 0.1}
    port_forecast = forecaster.correlation_adjusted_forecast(
        weights=weights,
        horizon=252,
        method='historical'
    )

    print(f"Portfolio Expected Return (1Y): {port_forecast.expected_return:+.2%}")
    print(f"Portfolio Volatility (1Y): {port_forecast.volatility:.2%}")
    ci = port_forecast.confidence_intervals[0.95]
    print(f"95% Confidence Interval: [{ci[0]:+.2%}, {ci[1]:+.2%}]")

    return forecaster


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n" + "="*60)
    print("5. VISUALIZATION (Generating Plots)")
    print("="*60)

    # Generate data and run simulation
    returns = generate_sample_data(n_days=756)

    # Create plotter
    plotter = SimulationPlotter(figsize=(12, 8))

    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # 1. Simulation paths
    simulator = MarketSimulator()
    simulator.add_asset('Test', GBM(GBMParams(mu=0.10, sigma=0.20)), 100)
    result = simulator.simulate(SimulationConfig(n_paths=1000, n_steps=252, random_state=42))

    fig1 = plotter.plot_simulation_paths(
        result.prices[0],
        n_paths_to_show=100,
        title="Monte Carlo Simulation Paths",
        save_path=str(output_dir / "simulation_paths.png")
    )
    print(f"Saved: {output_dir / 'simulation_paths.png'}")

    # 2. Return distribution
    terminal_returns = result.prices[0, :, -1] / result.prices[0, :, 0] - 1
    fig2 = plotter.plot_return_distribution(
        terminal_returns,
        title="Terminal Return Distribution",
        save_path=str(output_dir / "return_distribution.png")
    )
    print(f"Saved: {output_dir / 'return_distribution.png'}")

    # 3. Correlation matrix
    engine = CorrelationEngine(returns)
    corr = engine.static_correlation().current_correlation
    fig3 = plotter.plot_correlation_matrix(
        corr,
        asset_names=list(returns.columns),
        title="Asset Correlation Matrix",
        save_path=str(output_dir / "correlation_matrix.png")
    )
    print(f"Saved: {output_dir / 'correlation_matrix.png'}")

    # 4. Correlation dynamics
    rolling = engine.rolling_correlation(window=60)
    if rolling.correlation_history is not None:
        fig4 = plotter.plot_correlation_dynamics(
            rolling.correlation_history,
            asset_names=list(returns.columns),
            title="Rolling Correlation Dynamics (60-day window)",
            save_path=str(output_dir / "correlation_dynamics.png")
        )
        print(f"Saved: {output_dir / 'correlation_dynamics.png'}")

    # 5. Fan chart
    forecaster = ReturnForecaster(returns)
    fan_data = forecaster.generate_fan_chart_data(
        asset='US_Equity',
        max_horizon=252,
        step=5
    )
    fig5 = plotter.plot_fan_chart(
        fan_data,
        title="Return Forecast Fan Chart - US Equity",
        save_path=str(output_dir / "fan_chart.png")
    )
    print(f"Saved: {output_dir / 'fan_chart.png'}")

    # 6. Risk metrics
    stress_engine = StressTestEngine(
        returns=returns,
        weights={'US_Equity': 0.4, 'Intl_Equity': 0.2, 'Bonds': 0.3, 'Commodities': 0.1}
    )
    var = stress_engine.var_calculation([0.90, 0.95, 0.99])
    cvar = stress_engine.cvar_calculation([0.90, 0.95, 0.99])
    fig6 = plotter.plot_risk_metrics(
        var, cvar,
        title="Value at Risk Analysis",
        save_path=str(output_dir / "risk_metrics.png")
    )
    print(f"Saved: {output_dir / 'risk_metrics.png'}")

    plt.close('all')
    print(f"\nAll visualizations saved to: {output_dir.absolute()}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("  MARKET BEHAVIOR SIMULATOR")
    print("  Stress Testing & Forecasting Framework")
    print("="*60)

    # Run demonstrations
    result = demo_monte_carlo_simulation()
    engine = demo_correlation_analysis()
    stress_engine = demo_stress_testing()
    forecaster = demo_return_forecasting()
    demo_visualization()

    print("\n" + "="*60)
    print("  DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nThe market simulator provides:")
    print("  - Multi-asset Monte Carlo simulation with correlation")
    print("  - Multiple stochastic models (GBM, Jump Diffusion, GARCH)")
    print("  - Dynamic correlation estimation (EWMA, DCC)")
    print("  - Historical and custom stress scenarios")
    print("  - Multi-horizon return forecasting")
    print("  - Risk metrics (VaR, CVaR, drawdowns)")
    print("  - Comprehensive visualization tools")
    print("\nCheck the 'output' directory for generated plots.")


if __name__ == "__main__":
    main()
