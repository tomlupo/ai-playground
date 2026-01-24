"""
Market Simulation Engine

Orchestrates multi-asset simulations with correlated returns.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from dataclasses import dataclass, field
from .models import BaseModel, GBM, GBMParams


@dataclass
class SimulationConfig:
    """Configuration for market simulation."""
    n_paths: int = 10000  # Number of Monte Carlo paths
    n_steps: int = 252  # Number of time steps (1 year daily)
    dt: float = 1/252  # Time step size (daily)
    random_state: Optional[int] = None  # Random seed


@dataclass
class SimulationResult:
    """Results container for market simulation."""
    prices: np.ndarray  # Shape: (n_assets, n_paths, n_steps + 1)
    returns: np.ndarray  # Shape: (n_assets, n_paths, n_steps)
    asset_names: list[str]
    config: SimulationConfig
    correlation_matrix: Optional[np.ndarray] = None

    def get_terminal_prices(self) -> pd.DataFrame:
        """Get distribution of terminal prices."""
        data = {
            name: self.prices[i, :, -1]
            for i, name in enumerate(self.asset_names)
        }
        return pd.DataFrame(data)

    def get_terminal_returns(self) -> pd.DataFrame:
        """Get distribution of total returns."""
        total_returns = self.prices[:, :, -1] / self.prices[:, :, 0] - 1
        data = {
            name: total_returns[i]
            for i, name in enumerate(self.asset_names)
        }
        return pd.DataFrame(data)

    def get_path_statistics(self) -> pd.DataFrame:
        """Compute statistics across all paths for each asset."""
        stats = []
        for i, name in enumerate(self.asset_names):
            terminal = self.prices[i, :, -1] / self.prices[i, :, 0] - 1
            path_returns = self.returns[i]

            stats.append({
                'asset': name,
                'mean_return': np.mean(terminal),
                'median_return': np.median(terminal),
                'std_return': np.std(terminal),
                'skewness': self._skewness(terminal),
                'kurtosis': self._kurtosis(terminal),
                'var_95': np.percentile(terminal, 5),
                'var_99': np.percentile(terminal, 1),
                'cvar_95': np.mean(terminal[terminal <= np.percentile(terminal, 5)]),
                'max_drawdown_mean': np.mean(self._max_drawdowns(self.prices[i])),
                'sharpe_ratio': np.mean(terminal) / np.std(terminal) if np.std(terminal) > 0 else 0
            })

        return pd.DataFrame(stats).set_index('asset')

    def _skewness(self, x: np.ndarray) -> float:
        """Calculate skewness."""
        n = len(x)
        m = np.mean(x)
        s = np.std(x)
        if s == 0:
            return 0
        return np.sum(((x - m) / s) ** 3) / n

    def _kurtosis(self, x: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        n = len(x)
        m = np.mean(x)
        s = np.std(x)
        if s == 0:
            return 0
        return np.sum(((x - m) / s) ** 4) / n - 3

    def _max_drawdowns(self, prices: np.ndarray) -> np.ndarray:
        """Calculate maximum drawdown for each path."""
        # prices shape: (n_paths, n_steps + 1)
        running_max = np.maximum.accumulate(prices, axis=1)
        drawdowns = (prices - running_max) / running_max
        return np.min(drawdowns, axis=1)

    def get_percentile_paths(
        self,
        asset: str,
        percentiles: list[float] = [5, 25, 50, 75, 95]
    ) -> pd.DataFrame:
        """Get percentile bands across time for an asset."""
        idx = self.asset_names.index(asset)
        prices = self.prices[idx]

        data = {}
        for p in percentiles:
            data[f'p{p}'] = np.percentile(prices, p, axis=0)

        return pd.DataFrame(data)


class MarketSimulator:
    """
    Multi-asset market simulator with correlated returns.

    Supports various market models and correlation structures.
    """

    def __init__(
        self,
        models: dict[str, BaseModel] = None,
        initial_prices: dict[str, float] = None,
        correlation_matrix: np.ndarray = None
    ):
        """
        Initialize the market simulator.

        Args:
            models: Dictionary mapping asset names to model instances
            initial_prices: Dictionary mapping asset names to initial prices
            correlation_matrix: Correlation matrix for asset returns
        """
        self.models = models or {}
        self.initial_prices = initial_prices or {}
        self.correlation_matrix = correlation_matrix
        self._cholesky = None

        if correlation_matrix is not None:
            self._cholesky = np.linalg.cholesky(correlation_matrix)

    def add_asset(
        self,
        name: str,
        model: BaseModel,
        initial_price: float = 100.0
    ) -> "MarketSimulator":
        """Add an asset to the simulator."""
        self.models[name] = model
        self.initial_prices[name] = initial_price
        return self

    def set_correlation_matrix(self, corr_matrix: np.ndarray) -> "MarketSimulator":
        """Set the correlation matrix for asset returns."""
        n_assets = len(self.models)
        if corr_matrix.shape != (n_assets, n_assets):
            raise ValueError(f"Correlation matrix must be {n_assets}x{n_assets}")

        self.correlation_matrix = corr_matrix
        self._cholesky = np.linalg.cholesky(corr_matrix)
        return self

    def simulate(self, config: SimulationConfig = None) -> SimulationResult:
        """
        Run the market simulation.

        Args:
            config: Simulation configuration

        Returns:
            SimulationResult with all paths and statistics
        """
        config = config or SimulationConfig()
        rng = np.random.default_rng(config.random_state)

        asset_names = list(self.models.keys())
        n_assets = len(asset_names)

        # Generate correlated random numbers
        Z = rng.standard_normal((n_assets, config.n_paths, config.n_steps))

        if self._cholesky is not None:
            # Apply correlation structure
            for t in range(config.n_steps):
                Z[:, :, t] = self._cholesky @ Z[:, :, t]

        # Simulate each asset
        all_prices = np.zeros((n_assets, config.n_paths, config.n_steps + 1))
        all_returns = np.zeros((n_assets, config.n_paths, config.n_steps))

        for i, name in enumerate(asset_names):
            model = self.models[name]
            S0 = self.initial_prices.get(name, 100.0)

            # Use the correlated random numbers
            params = model.params
            dt = config.dt

            if hasattr(params, 'mu') and hasattr(params, 'sigma'):
                # Standard diffusion model
                drift = (params.mu - 0.5 * params.sigma**2) * dt
                diffusion = params.sigma * np.sqrt(dt)

                log_returns = drift + diffusion * Z[i]

                # Handle jumps if applicable
                if hasattr(params, 'jump_intensity'):
                    lam = params.jump_intensity
                    jump_mean = params.jump_mean
                    jump_std = params.jump_std

                    N = rng.poisson(lam * dt, (config.n_paths, config.n_steps))
                    J = rng.normal(jump_mean, jump_std, (config.n_paths, config.n_steps))
                    log_returns += N * J

                # Build price paths
                log_prices = np.zeros((config.n_paths, config.n_steps + 1))
                log_prices[:, 0] = np.log(S0)
                log_prices[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)

                all_prices[i] = np.exp(log_prices)
                all_returns[i] = np.exp(log_returns) - 1
            else:
                # Fallback to model's own simulation
                all_prices[i] = model.simulate(S0, config.n_steps, config.n_paths, config.random_state)
                all_returns[i] = np.diff(all_prices[i], axis=1) / all_prices[i, :, :-1]

        return SimulationResult(
            prices=all_prices,
            returns=all_returns,
            asset_names=asset_names,
            config=config,
            correlation_matrix=self.correlation_matrix
        )

    @classmethod
    def from_historical_data(
        cls,
        prices: pd.DataFrame,
        model_type: str = 'gbm',
        lookback_days: int = 252
    ) -> "MarketSimulator":
        """
        Create a simulator calibrated to historical data.

        Args:
            prices: DataFrame with asset prices (columns are assets)
            model_type: Type of model to fit ('gbm', 'garch')
            lookback_days: Number of days to use for calibration

        Returns:
            Configured MarketSimulator instance
        """
        # Calculate returns
        returns = prices.pct_change().dropna().tail(lookback_days)

        # Fit models
        models = {}
        initial_prices = {}

        for col in prices.columns:
            asset_returns = returns[col].values

            if model_type == 'gbm':
                from .models import GBM
                models[col] = GBM.fit(asset_returns)
            elif model_type == 'garch':
                from .models import GARCH
                models[col] = GARCH.fit(asset_returns)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            initial_prices[col] = prices[col].iloc[-1]

        # Calculate correlation matrix
        corr_matrix = returns.corr().values

        return cls(
            models=models,
            initial_prices=initial_prices,
            correlation_matrix=corr_matrix
        )


def generate_correlated_returns(
    n_assets: int,
    n_steps: int,
    n_paths: int,
    means: np.ndarray,
    stds: np.ndarray,
    correlation_matrix: np.ndarray,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate correlated multivariate returns.

    Args:
        n_assets: Number of assets
        n_steps: Number of time steps
        n_paths: Number of paths
        means: Mean returns for each asset
        stds: Standard deviations for each asset
        correlation_matrix: Correlation matrix
        random_state: Random seed

    Returns:
        Array of shape (n_assets, n_paths, n_steps) with correlated returns
    """
    rng = np.random.default_rng(random_state)

    # Cholesky decomposition
    L = np.linalg.cholesky(correlation_matrix)

    # Generate independent standard normal
    Z = rng.standard_normal((n_assets, n_paths, n_steps))

    # Apply correlation
    for t in range(n_steps):
        Z[:, :, t] = L @ Z[:, :, t]

    # Scale by mean and std
    returns = np.zeros_like(Z)
    for i in range(n_assets):
        returns[i] = means[i] + stds[i] * Z[i]

    return returns
