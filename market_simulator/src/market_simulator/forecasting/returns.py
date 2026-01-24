"""
Multi-Term Expected Returns Forecasting

Implements:
- Multi-horizon return forecasts (daily to yearly)
- Confidence intervals and fan charts
- Mean reversion adjustments
- Factor-based forecasts
- Bayesian shrinkage estimators
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from dataclasses import dataclass, field
from scipy import stats
from enum import Enum


class Horizon(Enum):
    """Standard forecast horizons."""
    DAILY = 1
    WEEKLY = 5
    MONTHLY = 21
    QUARTERLY = 63
    SEMI_ANNUAL = 126
    ANNUAL = 252
    TWO_YEAR = 504
    FIVE_YEAR = 1260
    TEN_YEAR = 2520


@dataclass
class ForecastResult:
    """Container for return forecast results."""
    asset: str
    horizon: int
    horizon_name: str
    expected_return: float  # Point estimate
    volatility: float  # Forecast volatility
    confidence_intervals: dict[float, tuple[float, float]]  # level -> (lower, upper)
    percentiles: dict[int, float]  # percentile -> value
    distribution_params: dict  # Parameters of fitted distribution
    method: str  # Method used for forecast

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'asset': self.asset,
            'horizon': self.horizon,
            'horizon_name': self.horizon_name,
            'expected_return': self.expected_return,
            'volatility': self.volatility,
            'ci_95_lower': self.confidence_intervals.get(0.95, (None, None))[0],
            'ci_95_upper': self.confidence_intervals.get(0.95, (None, None))[1],
            'ci_99_lower': self.confidence_intervals.get(0.99, (None, None))[0],
            'ci_99_upper': self.confidence_intervals.get(0.99, (None, None))[1],
            'method': self.method
        }


@dataclass
class MultiHorizonForecast:
    """Container for multi-horizon forecasts."""
    asset: str
    forecasts: dict[int, ForecastResult]  # horizon -> ForecastResult
    term_structure: pd.DataFrame  # Summary across horizons

    def get_horizon(self, horizon: Union[int, Horizon]) -> ForecastResult:
        """Get forecast for specific horizon."""
        if isinstance(horizon, Horizon):
            horizon = horizon.value
        return self.forecasts[horizon]

    def plot_term_structure(self):
        """Return data for plotting term structure."""
        return self.term_structure


class ReturnForecaster:
    """
    Multi-term expected returns forecaster.

    Provides forecasts across multiple time horizons with
    confidence intervals and various estimation methods.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.04
    ):
        """
        Initialize forecaster with historical data.

        Args:
            returns: DataFrame of asset returns (daily)
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.asset_names = list(returns.columns)
        self.risk_free_rate = risk_free_rate
        self._daily_rf = risk_free_rate / 252

    def forecast_single_horizon(
        self,
        asset: str,
        horizon: Union[int, Horizon],
        method: str = 'historical',
        confidence_levels: list[float] = [0.90, 0.95, 0.99],
        n_simulations: int = 10000
    ) -> ForecastResult:
        """
        Forecast returns for a single horizon.

        Args:
            asset: Asset name
            horizon: Forecast horizon (days or Horizon enum)
            method: 'historical', 'bootstrap', 'parametric', 'garch'
            confidence_levels: Confidence levels for intervals
            n_simulations: Number of simulations for bootstrap

        Returns:
            ForecastResult with forecast details
        """
        if isinstance(horizon, Horizon):
            horizon_name = horizon.name
            horizon = horizon.value
        else:
            # Map integer horizon to name
            horizon_name = self._horizon_to_name(horizon)

        daily_returns = self.returns[asset].values

        if method == 'historical':
            result = self._historical_forecast(
                daily_returns, horizon, confidence_levels
            )
        elif method == 'bootstrap':
            result = self._bootstrap_forecast(
                daily_returns, horizon, confidence_levels, n_simulations
            )
        elif method == 'parametric':
            result = self._parametric_forecast(
                daily_returns, horizon, confidence_levels
            )
        elif method == 'garch':
            result = self._garch_forecast(
                daily_returns, horizon, confidence_levels, n_simulations
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        return ForecastResult(
            asset=asset,
            horizon=horizon,
            horizon_name=horizon_name,
            expected_return=result['expected_return'],
            volatility=result['volatility'],
            confidence_intervals=result['confidence_intervals'],
            percentiles=result['percentiles'],
            distribution_params=result['distribution_params'],
            method=method
        )

    def forecast_multi_horizon(
        self,
        asset: str,
        horizons: list[Union[int, Horizon]] = None,
        method: str = 'historical',
        **kwargs
    ) -> MultiHorizonForecast:
        """
        Forecast returns across multiple horizons.

        Args:
            asset: Asset name
            horizons: List of horizons (defaults to all standard horizons)
            method: Forecasting method
            **kwargs: Additional arguments for forecast method

        Returns:
            MultiHorizonForecast with all horizons
        """
        if horizons is None:
            horizons = list(Horizon)

        forecasts = {}
        for h in horizons:
            if isinstance(h, Horizon):
                horizon_days = h.value
            else:
                horizon_days = h

            forecast = self.forecast_single_horizon(asset, h, method, **kwargs)
            forecasts[horizon_days] = forecast

        # Create term structure summary
        term_data = []
        for h, f in sorted(forecasts.items()):
            term_data.append({
                'horizon_days': h,
                'horizon_name': f.horizon_name,
                'expected_return': f.expected_return,
                'annualized_return': f.expected_return * (252 / h),
                'volatility': f.volatility,
                'annualized_volatility': f.volatility * np.sqrt(252 / h),
                'sharpe_ratio': (f.expected_return - self._daily_rf * h) / f.volatility if f.volatility > 0 else 0,
                'ci_95_lower': f.confidence_intervals.get(0.95, (0, 0))[0],
                'ci_95_upper': f.confidence_intervals.get(0.95, (0, 0))[1]
            })

        term_structure = pd.DataFrame(term_data)

        return MultiHorizonForecast(
            asset=asset,
            forecasts=forecasts,
            term_structure=term_structure
        )

    def forecast_all_assets(
        self,
        horizons: list[Union[int, Horizon]] = None,
        method: str = 'historical',
        **kwargs
    ) -> dict[str, MultiHorizonForecast]:
        """
        Forecast returns for all assets across multiple horizons.

        Args:
            horizons: List of horizons
            method: Forecasting method
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping asset names to MultiHorizonForecast
        """
        results = {}
        for asset in self.asset_names:
            results[asset] = self.forecast_multi_horizon(
                asset, horizons, method, **kwargs
            )
        return results

    def expected_return_with_mean_reversion(
        self,
        asset: str,
        horizon: int,
        long_term_return: float,
        mean_reversion_speed: float = 0.2
    ) -> ForecastResult:
        """
        Forecast with mean reversion toward long-term equilibrium.

        Uses Ornstein-Uhlenbeck process for mean-reverting returns.

        Args:
            asset: Asset name
            horizon: Forecast horizon (days)
            long_term_return: Long-term expected annual return
            mean_reversion_speed: Speed of mean reversion (annual)

        Returns:
            ForecastResult with mean-reverting forecast
        """
        daily_returns = self.returns[asset].values
        current_return = np.mean(daily_returns[-21:]) * 252  # Recent annualized

        # Mean reversion forecast
        theta = mean_reversion_speed
        t = horizon / 252  # Convert to years

        # Expected return: current + (long_term - current) * (1 - exp(-theta * t))
        expected_annual = current_return + (long_term_return - current_return) * (1 - np.exp(-theta * t))
        expected_horizon = expected_annual * (horizon / 252)

        # Volatility with mean reversion
        daily_vol = np.std(daily_returns)
        # Long-term variance of OU process
        long_term_var = (daily_vol ** 2) / (2 * theta / 252)
        horizon_vol = np.sqrt(long_term_var * (1 - np.exp(-2 * theta * t)))

        # Confidence intervals
        confidence_intervals = {}
        for level in [0.90, 0.95, 0.99]:
            z = stats.norm.ppf((1 + level) / 2)
            lower = expected_horizon - z * horizon_vol
            upper = expected_horizon + z * horizon_vol
            confidence_intervals[level] = (lower, upper)

        return ForecastResult(
            asset=asset,
            horizon=horizon,
            horizon_name=self._horizon_to_name(horizon),
            expected_return=expected_horizon,
            volatility=horizon_vol,
            confidence_intervals=confidence_intervals,
            percentiles={5: expected_horizon - 1.645 * horizon_vol,
                        50: expected_horizon,
                        95: expected_horizon + 1.645 * horizon_vol},
            distribution_params={
                'current_return': current_return,
                'long_term_return': long_term_return,
                'mean_reversion_speed': mean_reversion_speed
            },
            method='mean_reversion'
        )

    def bayesian_shrinkage_forecast(
        self,
        asset: str,
        horizon: int,
        prior_return: float = 0.06,
        prior_weight: float = 0.5
    ) -> ForecastResult:
        """
        Bayesian shrinkage estimator combining historical and prior.

        Args:
            asset: Asset name
            horizon: Forecast horizon (days)
            prior_return: Prior annual expected return
            prior_weight: Weight on prior vs historical

        Returns:
            ForecastResult with shrinkage estimate
        """
        daily_returns = self.returns[asset].values
        n = len(daily_returns)

        # Historical estimates (annualized)
        hist_mean = np.mean(daily_returns) * 252
        hist_var = np.var(daily_returns) * 252

        # Shrinkage estimator
        shrunk_return = prior_weight * prior_return + (1 - prior_weight) * hist_mean

        # Convert to horizon
        expected_horizon = shrunk_return * (horizon / 252)

        # Volatility
        daily_vol = np.std(daily_returns)
        horizon_vol = daily_vol * np.sqrt(horizon)

        # Confidence intervals
        confidence_intervals = {}
        for level in [0.90, 0.95, 0.99]:
            z = stats.norm.ppf((1 + level) / 2)
            lower = expected_horizon - z * horizon_vol
            upper = expected_horizon + z * horizon_vol
            confidence_intervals[level] = (lower, upper)

        return ForecastResult(
            asset=asset,
            horizon=horizon,
            horizon_name=self._horizon_to_name(horizon),
            expected_return=expected_horizon,
            volatility=horizon_vol,
            confidence_intervals=confidence_intervals,
            percentiles={5: expected_horizon - 1.645 * horizon_vol,
                        50: expected_horizon,
                        95: expected_horizon + 1.645 * horizon_vol},
            distribution_params={
                'historical_return': hist_mean,
                'prior_return': prior_return,
                'prior_weight': prior_weight,
                'shrunk_return': shrunk_return
            },
            method='bayesian_shrinkage'
        )

    def generate_fan_chart_data(
        self,
        asset: str,
        max_horizon: int = 252,
        step: int = 5,
        percentiles: list[int] = [5, 10, 25, 50, 75, 90, 95],
        n_simulations: int = 10000
    ) -> pd.DataFrame:
        """
        Generate data for fan chart visualization.

        Args:
            asset: Asset name
            max_horizon: Maximum forecast horizon
            step: Step size between forecasts
            percentiles: Percentiles to compute
            n_simulations: Number of simulations

        Returns:
            DataFrame with percentile paths
        """
        daily_returns = self.returns[asset].values
        mu = np.mean(daily_returns)
        sigma = np.std(daily_returns)

        horizons = range(1, max_horizon + 1, step)
        rng = np.random.default_rng(42)

        # Simulate paths
        simulated = np.zeros((n_simulations, len(list(horizons))))

        for i, h in enumerate(horizons):
            # Simulate cumulative returns
            random_returns = rng.normal(mu, sigma, (n_simulations, h))
            cumulative = np.sum(random_returns, axis=1)
            simulated[:, i] = cumulative

        # Calculate percentiles
        data = {'horizon': list(horizons)}
        for p in percentiles:
            data[f'p{p}'] = np.percentile(simulated, p, axis=0)

        return pd.DataFrame(data)

    def correlation_adjusted_forecast(
        self,
        weights: dict[str, float],
        horizon: int,
        method: str = 'historical'
    ) -> ForecastResult:
        """
        Forecast portfolio returns accounting for correlations.

        Args:
            weights: Portfolio weights by asset
            horizon: Forecast horizon
            method: Forecasting method

        Returns:
            ForecastResult for portfolio
        """
        # Get individual forecasts
        forecasts = {}
        for asset in weights.keys():
            forecasts[asset] = self.forecast_single_horizon(asset, horizon, method)

        # Weight array
        weight_arr = np.array([weights[a] for a in weights.keys()])

        # Expected returns
        expected_arr = np.array([forecasts[a].expected_return for a in weights.keys()])
        portfolio_expected = np.sum(weight_arr * expected_arr)

        # Covariance matrix at horizon
        horizon_returns = self.returns[list(weights.keys())].rolling(horizon).sum().dropna()
        cov_matrix = horizon_returns.cov().values

        # Portfolio volatility
        portfolio_var = weight_arr @ cov_matrix @ weight_arr
        portfolio_vol = np.sqrt(portfolio_var)

        # Confidence intervals
        confidence_intervals = {}
        for level in [0.90, 0.95, 0.99]:
            z = stats.norm.ppf((1 + level) / 2)
            lower = portfolio_expected - z * portfolio_vol
            upper = portfolio_expected + z * portfolio_vol
            confidence_intervals[level] = (lower, upper)

        return ForecastResult(
            asset='Portfolio',
            horizon=horizon,
            horizon_name=self._horizon_to_name(horizon),
            expected_return=portfolio_expected,
            volatility=portfolio_vol,
            confidence_intervals=confidence_intervals,
            percentiles={5: portfolio_expected - 1.645 * portfolio_vol,
                        50: portfolio_expected,
                        95: portfolio_expected + 1.645 * portfolio_vol},
            distribution_params={'weights': weights},
            method=f'portfolio_{method}'
        )

    def _historical_forecast(
        self,
        returns: np.ndarray,
        horizon: int,
        confidence_levels: list[float]
    ) -> dict:
        """Historical simulation forecast."""
        # Calculate horizon returns by rolling sum
        n = len(returns)
        if horizon > 1:
            horizon_returns = np.array([
                np.sum(returns[i:i+horizon])
                for i in range(n - horizon + 1)
            ])
        else:
            horizon_returns = returns

        expected = np.mean(horizon_returns)
        volatility = np.std(horizon_returns)

        # Confidence intervals from empirical distribution
        confidence_intervals = {}
        for level in confidence_levels:
            alpha = (1 - level) / 2
            lower = np.percentile(horizon_returns, alpha * 100)
            upper = np.percentile(horizon_returns, (1 - alpha) * 100)
            confidence_intervals[level] = (lower, upper)

        # Percentiles
        percentiles = {
            p: np.percentile(horizon_returns, p)
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        return {
            'expected_return': expected,
            'volatility': volatility,
            'confidence_intervals': confidence_intervals,
            'percentiles': percentiles,
            'distribution_params': {
                'n_observations': len(horizon_returns),
                'skewness': stats.skew(horizon_returns),
                'kurtosis': stats.kurtosis(horizon_returns)
            }
        }

    def _bootstrap_forecast(
        self,
        returns: np.ndarray,
        horizon: int,
        confidence_levels: list[float],
        n_simulations: int
    ) -> dict:
        """Bootstrap simulation forecast."""
        rng = np.random.default_rng(42)

        # Block bootstrap
        block_size = min(21, horizon)  # Monthly blocks
        n_blocks = (horizon + block_size - 1) // block_size

        simulated_returns = np.zeros(n_simulations)

        for i in range(n_simulations):
            path = []
            for _ in range(n_blocks):
                # Random starting point
                start = rng.integers(0, len(returns) - block_size)
                path.extend(returns[start:start + block_size])

            simulated_returns[i] = np.sum(path[:horizon])

        expected = np.mean(simulated_returns)
        volatility = np.std(simulated_returns)

        confidence_intervals = {}
        for level in confidence_levels:
            alpha = (1 - level) / 2
            lower = np.percentile(simulated_returns, alpha * 100)
            upper = np.percentile(simulated_returns, (1 - alpha) * 100)
            confidence_intervals[level] = (lower, upper)

        percentiles = {
            p: np.percentile(simulated_returns, p)
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        return {
            'expected_return': expected,
            'volatility': volatility,
            'confidence_intervals': confidence_intervals,
            'percentiles': percentiles,
            'distribution_params': {
                'n_simulations': n_simulations,
                'block_size': block_size
            }
        }

    def _parametric_forecast(
        self,
        returns: np.ndarray,
        horizon: int,
        confidence_levels: list[float]
    ) -> dict:
        """Parametric (normal) forecast."""
        daily_mu = np.mean(returns)
        daily_sigma = np.std(returns)

        # Scale to horizon
        expected = daily_mu * horizon
        volatility = daily_sigma * np.sqrt(horizon)

        confidence_intervals = {}
        for level in confidence_levels:
            z = stats.norm.ppf((1 + level) / 2)
            lower = expected - z * volatility
            upper = expected + z * volatility
            confidence_intervals[level] = (lower, upper)

        percentiles = {
            p: expected + stats.norm.ppf(p / 100) * volatility
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        return {
            'expected_return': expected,
            'volatility': volatility,
            'confidence_intervals': confidence_intervals,
            'percentiles': percentiles,
            'distribution_params': {
                'daily_mu': daily_mu,
                'daily_sigma': daily_sigma,
                'distribution': 'normal'
            }
        }

    def _garch_forecast(
        self,
        returns: np.ndarray,
        horizon: int,
        confidence_levels: list[float],
        n_simulations: int
    ) -> dict:
        """GARCH-based forecast with time-varying volatility."""
        from arch import arch_model

        # Fit GARCH model
        model = arch_model(returns * 100, vol='Garch', p=1, q=1)
        result = model.fit(disp='off')

        # Multi-step forecast
        forecasts = result.forecast(horizon=horizon, method='simulation',
                                   simulations=n_simulations)

        # Mean return
        mean_returns = result.params['mu'] / 100

        # Simulated variance paths
        simulated_variance = forecasts.variance.values[-1, :]  # Last observation forecast

        # Generate paths
        rng = np.random.default_rng(42)
        expected_vol = np.sqrt(np.mean(simulated_variance)) / 100

        # Scale to horizon
        expected = mean_returns * horizon
        volatility = expected_vol * np.sqrt(horizon)

        confidence_intervals = {}
        for level in confidence_levels:
            z = stats.norm.ppf((1 + level) / 2)
            lower = expected - z * volatility
            upper = expected + z * volatility
            confidence_intervals[level] = (lower, upper)

        percentiles = {
            p: expected + stats.norm.ppf(p / 100) * volatility
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        return {
            'expected_return': expected,
            'volatility': volatility,
            'confidence_intervals': confidence_intervals,
            'percentiles': percentiles,
            'distribution_params': {
                'omega': result.params.get('omega', 0),
                'alpha': result.params.get('alpha[1]', 0),
                'beta': result.params.get('beta[1]', 0)
            }
        }

    def _horizon_to_name(self, horizon: int) -> str:
        """Convert horizon days to name."""
        for h in Horizon:
            if h.value == horizon:
                return h.name

        # Approximate name
        if horizon <= 5:
            return f'{horizon}D'
        elif horizon <= 21:
            return f'{horizon // 5}W'
        elif horizon <= 63:
            return f'{horizon // 21}M'
        elif horizon <= 252:
            return f'{horizon // 63}Q'
        else:
            return f'{horizon // 252}Y'
