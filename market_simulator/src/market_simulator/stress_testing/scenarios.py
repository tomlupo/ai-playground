"""
Stress Testing Scenarios and Risk Analysis

Implements:
- Historical stress scenarios (2008, 2020 COVID, etc.)
- Hypothetical stress scenarios
- Sensitivity analysis
- VaR and CVaR calculations
- Drawdown analysis
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum


class HistoricalScenario(Enum):
    """Pre-defined historical stress scenarios."""
    FINANCIAL_CRISIS_2008 = "2008_financial_crisis"
    COVID_CRASH_2020 = "2020_covid_crash"
    DOT_COM_BUST_2000 = "2000_dotcom_bust"
    BLACK_MONDAY_1987 = "1987_black_monday"
    EURO_CRISIS_2011 = "2011_euro_crisis"
    TAPER_TANTRUM_2013 = "2013_taper_tantrum"
    FLASH_CRASH_2010 = "2010_flash_crash"
    VOLMAGEDDON_2018 = "2018_volmageddon"


@dataclass
class StressScenario:
    """Definition of a stress scenario."""
    name: str
    description: str
    equity_shock: float  # Percentage shock to equities
    bond_shock: float  # Percentage shock to bonds
    volatility_multiplier: float  # Multiplier for volatility
    correlation_stress: float  # Stress factor for correlations
    duration_days: int  # Duration of the scenario
    recovery_days: int = 0  # Optional recovery period
    custom_shocks: dict = field(default_factory=dict)  # Asset-specific shocks


# Pre-defined historical scenarios
HISTORICAL_SCENARIOS = {
    HistoricalScenario.FINANCIAL_CRISIS_2008: StressScenario(
        name="2008 Financial Crisis",
        description="Lehman Brothers collapse and global financial crisis",
        equity_shock=-0.55,  # S&P 500 peak-to-trough
        bond_shock=0.10,  # Flight to quality
        volatility_multiplier=4.0,  # VIX spiked to 80
        correlation_stress=1.8,  # Correlations increased sharply
        duration_days=350,
        recovery_days=700
    ),
    HistoricalScenario.COVID_CRASH_2020: StressScenario(
        name="COVID-19 Crash",
        description="Pandemic-induced market crash",
        equity_shock=-0.34,  # S&P 500 drop
        bond_shock=0.05,  # Treasury rally
        volatility_multiplier=5.0,  # Fastest VIX spike ever
        correlation_stress=2.0,
        duration_days=23,  # Very rapid decline
        recovery_days=140  # V-shaped recovery
    ),
    HistoricalScenario.DOT_COM_BUST_2000: StressScenario(
        name="Dot-Com Bust",
        description="Technology bubble burst",
        equity_shock=-0.49,  # S&P 500 decline
        bond_shock=0.15,  # Bond rally
        volatility_multiplier=2.0,
        correlation_stress=1.3,
        duration_days=650,
        recovery_days=1500
    ),
    HistoricalScenario.BLACK_MONDAY_1987: StressScenario(
        name="Black Monday 1987",
        description="Single-day market crash",
        equity_shock=-0.22,  # Single day drop
        bond_shock=0.03,
        volatility_multiplier=6.0,
        correlation_stress=2.5,
        duration_days=1,
        recovery_days=400
    ),
    HistoricalScenario.EURO_CRISIS_2011: StressScenario(
        name="European Debt Crisis",
        description="Sovereign debt crisis in Europe",
        equity_shock=-0.20,
        bond_shock=-0.05,  # Even bonds sold off initially
        volatility_multiplier=2.5,
        correlation_stress=1.5,
        duration_days=180,
        recovery_days=300
    ),
    HistoricalScenario.TAPER_TANTRUM_2013: StressScenario(
        name="Taper Tantrum",
        description="Fed tapering announcement shock",
        equity_shock=-0.06,
        bond_shock=-0.08,  # Rates spiked
        volatility_multiplier=1.5,
        correlation_stress=1.2,
        duration_days=60,
        recovery_days=90
    ),
    HistoricalScenario.FLASH_CRASH_2010: StressScenario(
        name="Flash Crash 2010",
        description="Rapid intraday market crash",
        equity_shock=-0.09,  # Intraday decline
        bond_shock=0.02,
        volatility_multiplier=3.0,
        correlation_stress=1.8,
        duration_days=1,
        recovery_days=1
    ),
    HistoricalScenario.VOLMAGEDDON_2018: StressScenario(
        name="Volmageddon",
        description="February 2018 volatility spike",
        equity_shock=-0.10,
        bond_shock=-0.02,
        volatility_multiplier=4.0,
        correlation_stress=1.6,
        duration_days=10,
        recovery_days=60
    )
}


@dataclass
class StressTestResult:
    """Results from stress testing."""
    scenario: StressScenario
    portfolio_impact: float  # Total portfolio return under stress
    asset_impacts: dict[str, float]  # Per-asset impacts
    var_95: float  # 95% VaR under stress
    var_99: float  # 99% VaR under stress
    cvar_95: float  # 95% CVaR (Expected Shortfall)
    max_drawdown: float
    time_to_recovery: Optional[int] = None
    stressed_prices: Optional[np.ndarray] = None
    stressed_returns: Optional[np.ndarray] = None


class StressTestEngine:
    """
    Engine for stress testing portfolios and market scenarios.

    Supports both historical and hypothetical stress scenarios.
    """

    def __init__(
        self,
        returns: Optional[pd.DataFrame] = None,
        weights: Optional[dict[str, float]] = None
    ):
        """
        Initialize stress test engine.

        Args:
            returns: Historical returns data for calibration
            weights: Portfolio weights by asset
        """
        self.returns = returns
        self.weights = weights or {}
        self.asset_names = list(returns.columns) if returns is not None else []

    def run_historical_scenario(
        self,
        scenario: Union[HistoricalScenario, str],
        n_simulations: int = 1000,
        random_state: Optional[int] = None
    ) -> StressTestResult:
        """
        Run a pre-defined historical stress scenario.

        Args:
            scenario: Historical scenario to run
            n_simulations: Number of Monte Carlo simulations
            random_state: Random seed

        Returns:
            StressTestResult with impact analysis
        """
        if isinstance(scenario, str):
            scenario = HistoricalScenario(scenario)

        stress_def = HISTORICAL_SCENARIOS[scenario]
        return self.run_custom_scenario(stress_def, n_simulations, random_state)

    def run_custom_scenario(
        self,
        scenario: StressScenario,
        n_simulations: int = 1000,
        random_state: Optional[int] = None
    ) -> StressTestResult:
        """
        Run a custom stress scenario.

        Args:
            scenario: Custom stress scenario definition
            n_simulations: Number of Monte Carlo simulations
            random_state: Random seed

        Returns:
            StressTestResult with impact analysis
        """
        rng = np.random.default_rng(random_state)

        n_assets = len(self.asset_names)
        n_steps = scenario.duration_days

        # Calculate base parameters from historical data
        if self.returns is not None:
            base_means = self.returns.mean().values
            base_stds = self.returns.std().values
            base_corr = self.returns.corr().values
        else:
            base_means = np.zeros(n_assets)
            base_stds = np.ones(n_assets) * 0.01
            base_corr = np.eye(n_assets)

        # Apply stress to parameters
        stressed_stds = base_stds * scenario.volatility_multiplier

        # Stress correlations (increase toward 1)
        stressed_corr = self._stress_correlation_matrix(
            base_corr,
            scenario.correlation_stress
        )

        # Generate Cholesky decomposition
        L = np.linalg.cholesky(stressed_corr)

        # Simulate stressed returns
        Z = rng.standard_normal((n_simulations, n_steps, n_assets))

        stressed_returns = np.zeros((n_simulations, n_steps, n_assets))

        for t in range(n_steps):
            correlated_shocks = Z[:, t, :] @ L.T
            stressed_returns[:, t, :] = stressed_stds * correlated_shocks

        # Apply directional shocks
        for i, asset in enumerate(self.asset_names):
            if asset in scenario.custom_shocks:
                shock = scenario.custom_shocks[asset]
            elif 'equity' in asset.lower() or 'stock' in asset.lower():
                shock = scenario.equity_shock
            elif 'bond' in asset.lower() or 'fixed' in asset.lower():
                shock = scenario.bond_shock
            else:
                shock = scenario.equity_shock  # Default to equity shock

            # Distribute shock over duration
            daily_shock = shock / n_steps
            stressed_returns[:, :, i] += daily_shock

        # Calculate portfolio-level impacts
        weights_arr = np.array([self.weights.get(a, 1/n_assets) for a in self.asset_names])

        # Portfolio returns per simulation
        portfolio_returns = (stressed_returns * weights_arr).sum(axis=2)

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1) - 1

        # Terminal returns
        terminal_returns = cumulative_returns[:, -1]

        # Calculate risk metrics
        var_95 = np.percentile(terminal_returns, 5)
        var_99 = np.percentile(terminal_returns, 1)
        cvar_95 = np.mean(terminal_returns[terminal_returns <= var_95])

        # Maximum drawdown
        max_dd = self._calculate_max_drawdown(cumulative_returns)

        # Per-asset impacts
        asset_impacts = {}
        for i, asset in enumerate(self.asset_names):
            asset_returns = stressed_returns[:, :, i]
            asset_cum = np.cumprod(1 + asset_returns, axis=1) - 1
            asset_impacts[asset] = np.mean(asset_cum[:, -1])

        # Mean portfolio impact
        portfolio_impact = np.mean(terminal_returns)

        return StressTestResult(
            scenario=scenario,
            portfolio_impact=portfolio_impact,
            asset_impacts=asset_impacts,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            max_drawdown=np.mean(max_dd),
            stressed_returns=stressed_returns
        )

    def sensitivity_analysis(
        self,
        shock_variable: str,
        shock_range: np.ndarray,
        n_simulations: int = 1000,
        random_state: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis by varying a shock parameter.

        Args:
            shock_variable: Variable to shock ('equity', 'volatility', 'correlation')
            shock_range: Range of shock values to test
            n_simulations: Number of simulations per shock level
            random_state: Random seed

        Returns:
            DataFrame with sensitivity results
        """
        results = []

        for shock_level in shock_range:
            # Create scenario with this shock level
            if shock_variable == 'equity':
                scenario = StressScenario(
                    name=f"Equity shock {shock_level:.1%}",
                    description="Equity sensitivity test",
                    equity_shock=shock_level,
                    bond_shock=0.0,
                    volatility_multiplier=1.0,
                    correlation_stress=1.0,
                    duration_days=21
                )
            elif shock_variable == 'volatility':
                scenario = StressScenario(
                    name=f"Volatility shock {shock_level:.1f}x",
                    description="Volatility sensitivity test",
                    equity_shock=0.0,
                    bond_shock=0.0,
                    volatility_multiplier=shock_level,
                    correlation_stress=1.0,
                    duration_days=21
                )
            elif shock_variable == 'correlation':
                scenario = StressScenario(
                    name=f"Correlation stress {shock_level:.1f}x",
                    description="Correlation sensitivity test",
                    equity_shock=0.0,
                    bond_shock=0.0,
                    volatility_multiplier=1.0,
                    correlation_stress=shock_level,
                    duration_days=21
                )
            else:
                raise ValueError(f"Unknown shock variable: {shock_variable}")

            result = self.run_custom_scenario(scenario, n_simulations, random_state)

            results.append({
                'shock_level': shock_level,
                'portfolio_impact': result.portfolio_impact,
                'var_95': result.var_95,
                'var_99': result.var_99,
                'cvar_95': result.cvar_95,
                'max_drawdown': result.max_drawdown
            })

        return pd.DataFrame(results)

    def reverse_stress_test(
        self,
        target_loss: float,
        shock_variable: str = 'equity',
        tolerance: float = 0.01,
        max_iterations: int = 50,
        n_simulations: int = 1000
    ) -> tuple[float, StressTestResult]:
        """
        Find shock level required to achieve target loss.

        Args:
            target_loss: Target portfolio loss (negative)
            shock_variable: Variable to shock
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            n_simulations: Simulations per iteration

        Returns:
            Tuple of (required_shock_level, StressTestResult)
        """
        # Binary search for required shock
        if shock_variable == 'equity':
            low, high = -0.8, 0.0
        elif shock_variable == 'volatility':
            low, high = 1.0, 10.0
        else:
            low, high = 1.0, 5.0

        for _ in range(max_iterations):
            mid = (low + high) / 2

            if shock_variable == 'equity':
                scenario = StressScenario(
                    name="Reverse stress",
                    description="Reverse stress test",
                    equity_shock=mid,
                    bond_shock=0.0,
                    volatility_multiplier=1.5,
                    correlation_stress=1.3,
                    duration_days=21
                )
            elif shock_variable == 'volatility':
                scenario = StressScenario(
                    name="Reverse stress",
                    description="Reverse stress test",
                    equity_shock=-0.1,
                    bond_shock=0.0,
                    volatility_multiplier=mid,
                    correlation_stress=1.3,
                    duration_days=21
                )
            else:
                scenario = StressScenario(
                    name="Reverse stress",
                    description="Reverse stress test",
                    equity_shock=-0.1,
                    bond_shock=0.0,
                    volatility_multiplier=1.5,
                    correlation_stress=mid,
                    duration_days=21
                )

            result = self.run_custom_scenario(scenario, n_simulations)
            current_loss = result.portfolio_impact

            if abs(current_loss - target_loss) < tolerance:
                return mid, result

            if current_loss > target_loss:
                # Need more shock
                if shock_variable == 'equity':
                    high = mid
                else:
                    low = mid
            else:
                if shock_variable == 'equity':
                    low = mid
                else:
                    high = mid

        return mid, result

    def var_calculation(
        self,
        confidence_levels: list[float] = [0.95, 0.99],
        horizon_days: int = 1,
        method: str = 'historical',
        n_simulations: int = 10000
    ) -> dict[float, float]:
        """
        Calculate Value at Risk.

        Args:
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
            horizon_days: Time horizon in days
            method: 'historical', 'parametric', or 'monte_carlo'
            n_simulations: Number of simulations for MC method

        Returns:
            Dictionary mapping confidence level to VaR
        """
        if self.returns is None:
            raise ValueError("Historical returns required for VaR calculation")

        weights_arr = np.array([self.weights.get(a, 1/len(self.asset_names))
                                for a in self.asset_names])

        # Portfolio returns
        portfolio_returns = (self.returns.values * weights_arr).sum(axis=1)

        # Scale to horizon
        if horizon_days > 1:
            # Use rolling sum for multi-day returns
            scaled_returns = pd.Series(portfolio_returns).rolling(horizon_days).sum().dropna().values
        else:
            scaled_returns = portfolio_returns

        var_results = {}

        for level in confidence_levels:
            alpha = 1 - level

            if method == 'historical':
                var = np.percentile(scaled_returns, alpha * 100)

            elif method == 'parametric':
                from scipy.stats import norm
                mu = np.mean(scaled_returns)
                sigma = np.std(scaled_returns)
                var = mu + sigma * norm.ppf(alpha)

            elif method == 'monte_carlo':
                rng = np.random.default_rng(42)
                mu = np.mean(scaled_returns)
                sigma = np.std(scaled_returns)
                simulated = rng.normal(mu, sigma, n_simulations)
                var = np.percentile(simulated, alpha * 100)

            else:
                raise ValueError(f"Unknown method: {method}")

            var_results[level] = var

        return var_results

    def cvar_calculation(
        self,
        confidence_levels: list[float] = [0.95, 0.99],
        horizon_days: int = 1
    ) -> dict[float, float]:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            confidence_levels: List of confidence levels
            horizon_days: Time horizon in days

        Returns:
            Dictionary mapping confidence level to CVaR
        """
        if self.returns is None:
            raise ValueError("Historical returns required")

        weights_arr = np.array([self.weights.get(a, 1/len(self.asset_names))
                                for a in self.asset_names])

        portfolio_returns = (self.returns.values * weights_arr).sum(axis=1)

        if horizon_days > 1:
            scaled_returns = pd.Series(portfolio_returns).rolling(horizon_days).sum().dropna().values
        else:
            scaled_returns = portfolio_returns

        cvar_results = {}

        for level in confidence_levels:
            alpha = 1 - level
            var = np.percentile(scaled_returns, alpha * 100)
            cvar = np.mean(scaled_returns[scaled_returns <= var])
            cvar_results[level] = cvar

        return cvar_results

    def _stress_correlation_matrix(
        self,
        corr: np.ndarray,
        stress_factor: float
    ) -> np.ndarray:
        """Apply stress to correlation matrix."""
        n = len(corr)
        stressed = corr.copy()

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Push correlations toward 1 (or -1 for negatives)
                    sign = np.sign(corr[i, j])
                    magnitude = min(abs(corr[i, j]) * stress_factor, 0.999)
                    stressed[i, j] = sign * magnitude

        # Ensure positive semi-definiteness
        eigvals, eigvecs = np.linalg.eigh(stressed)
        eigvals = np.maximum(eigvals, 1e-8)
        stressed = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Normalize back to correlation
        d = np.sqrt(np.diag(stressed))
        stressed = stressed / np.outer(d, d)
        np.fill_diagonal(stressed, 1.0)

        return stressed

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> np.ndarray:
        """Calculate maximum drawdown for each simulation path."""
        # cumulative_returns shape: (n_simulations, n_steps)
        wealth = 1 + cumulative_returns
        running_max = np.maximum.accumulate(wealth, axis=1)
        drawdowns = (wealth - running_max) / running_max
        return np.min(drawdowns, axis=1)

    def compare_scenarios(
        self,
        scenarios: list[Union[HistoricalScenario, StressScenario]],
        n_simulations: int = 1000
    ) -> pd.DataFrame:
        """
        Compare multiple stress scenarios.

        Args:
            scenarios: List of scenarios to compare
            n_simulations: Simulations per scenario

        Returns:
            DataFrame comparing scenario impacts
        """
        results = []

        for scenario in scenarios:
            if isinstance(scenario, HistoricalScenario):
                result = self.run_historical_scenario(scenario, n_simulations)
            else:
                result = self.run_custom_scenario(scenario, n_simulations)

            results.append({
                'scenario': result.scenario.name,
                'portfolio_impact': result.portfolio_impact,
                'var_95': result.var_95,
                'var_99': result.var_99,
                'cvar_95': result.cvar_95,
                'max_drawdown': result.max_drawdown,
                'duration_days': result.scenario.duration_days
            })

        return pd.DataFrame(results).set_index('scenario')
