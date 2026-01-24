"""
Correlation Structures for Multi-Asset Simulation

Implements various correlation estimation methods:
- Static (sample) correlation
- Rolling window correlation
- Exponentially weighted correlation
- DCC (Dynamic Conditional Correlation) from Engle (2002)
- Shrinkage estimators (Ledoit-Wolf)
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy import stats


@dataclass
class CorrelationResult:
    """Container for correlation analysis results."""
    current_correlation: np.ndarray
    correlation_history: Optional[np.ndarray] = None
    asset_names: Optional[list[str]] = None
    method: str = "static"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert current correlation to DataFrame."""
        names = self.asset_names or [f"Asset_{i}" for i in range(len(self.current_correlation))]
        return pd.DataFrame(
            self.current_correlation,
            index=names,
            columns=names
        )

    def get_correlation_at(self, t: int) -> np.ndarray:
        """Get correlation matrix at time t."""
        if self.correlation_history is None:
            return self.current_correlation
        return self.correlation_history[t]


class CorrelationEngine:
    """
    Engine for computing and forecasting correlation structures.

    Supports multiple estimation methods and can generate
    time-varying correlation matrices for simulation.
    """

    def __init__(self, returns: pd.DataFrame):
        """
        Initialize with historical returns data.

        Args:
            returns: DataFrame of asset returns (each column is an asset)
        """
        self.returns = returns
        self.asset_names = list(returns.columns)
        self.n_assets = len(self.asset_names)

    def static_correlation(self) -> CorrelationResult:
        """
        Compute static (sample) correlation matrix.

        Returns:
            CorrelationResult with sample correlation
        """
        corr = self.returns.corr().values
        return CorrelationResult(
            current_correlation=corr,
            asset_names=self.asset_names,
            method="static"
        )

    def rolling_correlation(
        self,
        window: int = 60,
        min_periods: int = 30
    ) -> CorrelationResult:
        """
        Compute rolling window correlation.

        Args:
            window: Rolling window size
            min_periods: Minimum periods for valid correlation

        Returns:
            CorrelationResult with time series of correlations
        """
        n_obs = len(self.returns)
        n_valid = n_obs - window + 1

        # Store correlation history
        corr_history = np.zeros((n_valid, self.n_assets, self.n_assets))

        for t in range(n_valid):
            window_data = self.returns.iloc[t:t + window]
            corr_history[t] = window_data.corr().values

        # Current correlation is the last one
        current = corr_history[-1]

        return CorrelationResult(
            current_correlation=current,
            correlation_history=corr_history,
            asset_names=self.asset_names,
            method=f"rolling_{window}"
        )

    def ewma_correlation(
        self,
        lambda_param: float = 0.94,
        min_periods: int = 30
    ) -> CorrelationResult:
        """
        Compute Exponentially Weighted Moving Average correlation (RiskMetrics style).

        Args:
            lambda_param: Decay factor (higher = slower decay)
            min_periods: Minimum periods before computing

        Returns:
            CorrelationResult with EWMA correlation
        """
        returns_arr = self.returns.values
        n_obs, n_assets = returns_arr.shape

        # Initialize covariance matrix
        cov = np.cov(returns_arr[:min_periods].T)

        # Store history
        corr_history = np.zeros((n_obs - min_periods, n_assets, n_assets))

        for t in range(min_periods, n_obs):
            r = returns_arr[t - 1]
            outer = np.outer(r, r)

            # EWMA update
            cov = lambda_param * cov + (1 - lambda_param) * outer

            # Convert to correlation
            std = np.sqrt(np.diag(cov))
            corr = cov / np.outer(std, std)
            np.fill_diagonal(corr, 1.0)

            corr_history[t - min_periods] = corr

        return CorrelationResult(
            current_correlation=corr_history[-1],
            correlation_history=corr_history,
            asset_names=self.asset_names,
            method=f"ewma_{lambda_param}"
        )

    def dcc_correlation(
        self,
        a: float = 0.05,
        b: float = 0.93
    ) -> CorrelationResult:
        """
        Compute DCC (Dynamic Conditional Correlation) model.

        Implements Engle (2002) DCC-GARCH:
        Q_t = (1 - a - b) * Q_bar + a * epsilon_{t-1} * epsilon_{t-1}' + b * Q_{t-1}
        R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}

        Args:
            a: DCC alpha parameter (innovation)
            b: DCC beta parameter (persistence)

        Returns:
            CorrelationResult with DCC correlations
        """
        returns_arr = self.returns.values
        n_obs, n_assets = returns_arr.shape

        # Step 1: Estimate univariate GARCH for each asset
        from arch import arch_model

        standardized = np.zeros_like(returns_arr)
        conditional_vols = np.zeros_like(returns_arr)

        for i in range(n_assets):
            model = arch_model(returns_arr[:, i] * 100, vol='Garch', p=1, q=1)
            result = model.fit(disp='off')
            conditional_vols[:, i] = result.conditional_volatility / 100
            standardized[:, i] = returns_arr[:, i] / (conditional_vols[:, i] + 1e-8)

        # Step 2: Compute Q_bar (unconditional correlation of standardized residuals)
        Q_bar = np.corrcoef(standardized.T)

        # Step 3: DCC recursion
        Q = Q_bar.copy()
        corr_history = np.zeros((n_obs, n_assets, n_assets))

        for t in range(n_obs):
            eps = standardized[t]
            outer = np.outer(eps, eps)

            # DCC update
            Q = (1 - a - b) * Q_bar + a * outer + b * Q

            # Normalize to correlation matrix
            Q_diag_sqrt = np.sqrt(np.diag(Q))
            R = Q / np.outer(Q_diag_sqrt, Q_diag_sqrt)
            np.fill_diagonal(R, 1.0)

            corr_history[t] = R

        return CorrelationResult(
            current_correlation=corr_history[-1],
            correlation_history=corr_history,
            asset_names=self.asset_names,
            method=f"dcc_a{a}_b{b}"
        )

    def fit_dcc(self) -> tuple[float, float, CorrelationResult]:
        """
        Fit DCC parameters using maximum likelihood estimation.

        Returns:
            Tuple of (optimal_a, optimal_b, CorrelationResult)
        """
        returns_arr = self.returns.values
        n_obs, n_assets = returns_arr.shape

        # Estimate univariate GARCH first
        from arch import arch_model

        standardized = np.zeros_like(returns_arr)

        for i in range(n_assets):
            model = arch_model(returns_arr[:, i] * 100, vol='Garch', p=1, q=1)
            result = model.fit(disp='off')
            cond_vol = result.conditional_volatility / 100
            standardized[:, i] = returns_arr[:, i] / (cond_vol + 1e-8)

        Q_bar = np.corrcoef(standardized.T)

        def neg_log_likelihood(params):
            a, b = params
            if a < 0 or b < 0 or a + b >= 1:
                return 1e10

            Q = Q_bar.copy()
            ll = 0

            for t in range(n_obs):
                eps = standardized[t]
                outer = np.outer(eps, eps)

                Q = (1 - a - b) * Q_bar + a * outer + b * Q

                Q_diag_sqrt = np.sqrt(np.diag(Q))
                R = Q / np.outer(Q_diag_sqrt, Q_diag_sqrt)
                np.fill_diagonal(R, 1.0)

                # Log-likelihood contribution
                try:
                    ll += -0.5 * (np.log(np.linalg.det(R)) + eps @ np.linalg.solve(R, eps))
                except np.linalg.LinAlgError:
                    return 1e10

            return -ll

        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0=[0.05, 0.90],
            bounds=[(0.001, 0.3), (0.5, 0.99)],
            method='L-BFGS-B'
        )

        a_opt, b_opt = result.x

        # Get final correlation result
        corr_result = self.dcc_correlation(a_opt, b_opt)

        return a_opt, b_opt, corr_result

    def ledoit_wolf_shrinkage(self) -> CorrelationResult:
        """
        Compute Ledoit-Wolf shrinkage correlation estimator.

        Shrinks sample correlation toward identity matrix for improved stability.

        Returns:
            CorrelationResult with shrunk correlation
        """
        returns_arr = self.returns.values
        n_obs, n_assets = returns_arr.shape

        # Standardize returns
        std_returns = (returns_arr - returns_arr.mean(axis=0)) / returns_arr.std(axis=0)

        # Sample correlation
        sample_corr = np.corrcoef(std_returns.T)

        # Target: Identity matrix
        target = np.eye(n_assets)

        # Compute shrinkage intensity (Ledoit-Wolf formula)
        X = std_returns

        # Compute pi (sum of variances of off-diagonal elements)
        pi_sum = 0
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    wij = X[:, i] * X[:, j]
                    pi_sum += np.var(wij)

        pi = pi_sum

        # Compute gamma (distance between target and sample)
        gamma = np.sum((sample_corr - target) ** 2)

        # Optimal shrinkage intensity
        kappa = (pi) / (gamma * n_obs)
        shrinkage = max(0, min(1, kappa))

        # Shrunk correlation
        shrunk_corr = shrinkage * target + (1 - shrinkage) * sample_corr

        return CorrelationResult(
            current_correlation=shrunk_corr,
            asset_names=self.asset_names,
            method=f"ledoit_wolf_{shrinkage:.3f}"
        )

    def correlation_stress(
        self,
        stress_factor: float = 1.5,
        floor: float = 0.0
    ) -> np.ndarray:
        """
        Generate stressed correlation matrix.

        Increases correlations toward 1 (common in market stress).

        Args:
            stress_factor: Multiplier for correlations (>1 increases)
            floor: Minimum correlation after stress

        Returns:
            Stressed correlation matrix
        """
        base_corr = self.static_correlation().current_correlation

        # Stress the off-diagonal elements
        stressed = base_corr.copy()
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                if i != j:
                    # Increase correlation, bounded by 1
                    stressed[i, j] = np.clip(
                        stress_factor * base_corr[i, j],
                        floor,
                        0.999
                    )

        # Ensure positive semi-definiteness
        stressed = self._nearest_positive_definite(stressed)

        return stressed

    def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        """Find nearest positive definite matrix using Higham's algorithm."""
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if self._is_positive_definite(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self._is_positive_definite(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3

    def _is_positive_definite(self, A: np.ndarray) -> bool:
        """Check if matrix is positive definite."""
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False

    def forecast_correlation(
        self,
        n_steps: int = 21,
        method: str = 'ewma',
        **kwargs
    ) -> np.ndarray:
        """
        Forecast future correlation matrices.

        Args:
            n_steps: Number of steps to forecast
            method: Forecasting method ('ewma', 'dcc', 'static')
            **kwargs: Additional parameters for the method

        Returns:
            Array of shape (n_steps, n_assets, n_assets) with forecasted correlations
        """
        if method == 'static':
            # Static forecast: same correlation for all periods
            current = self.static_correlation().current_correlation
            return np.tile(current, (n_steps, 1, 1))

        elif method == 'ewma':
            # EWMA converges to unconditional correlation
            ewma_result = self.ewma_correlation(**kwargs)
            current = ewma_result.current_correlation
            unconditional = self.static_correlation().current_correlation

            lambda_param = kwargs.get('lambda_param', 0.94)

            forecasts = np.zeros((n_steps, self.n_assets, self.n_assets))
            for t in range(n_steps):
                # Decay toward unconditional
                weight = lambda_param ** t
                forecasts[t] = weight * current + (1 - weight) * unconditional

            return forecasts

        elif method == 'dcc':
            # DCC converges to Q_bar
            a, b, dcc_result = self.fit_dcc()
            current = dcc_result.current_correlation
            unconditional = self.static_correlation().current_correlation

            forecasts = np.zeros((n_steps, self.n_assets, self.n_assets))
            for t in range(n_steps):
                # DCC mean reversion
                weight = b ** t
                forecasts[t] = weight * current + (1 - weight) * unconditional

            return forecasts

        else:
            raise ValueError(f"Unknown method: {method}")

    def correlation_regime_detection(
        self,
        window: int = 60,
        threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        Detect correlation regime changes.

        Args:
            window: Rolling window size
            threshold: Threshold for regime change detection

        Returns:
            DataFrame with regime indicators
        """
        rolling = self.rolling_correlation(window=window)

        # Calculate average correlation over time
        avg_corr = []
        for t in range(len(rolling.correlation_history)):
            corr = rolling.correlation_history[t]
            # Average of off-diagonal elements
            mask = ~np.eye(self.n_assets, dtype=bool)
            avg_corr.append(np.mean(corr[mask]))

        avg_corr = np.array(avg_corr)

        # Detect regime changes
        regime_changes = np.abs(np.diff(avg_corr)) > threshold

        return pd.DataFrame({
            'average_correlation': avg_corr,
            'regime_change': np.concatenate([[False], regime_changes])
        })


def generate_random_correlation_matrix(
    n: int,
    eigenvalue_concentration: float = 0.8,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate a random valid correlation matrix.

    Args:
        n: Number of assets
        eigenvalue_concentration: Fraction of variance in first eigenvalue
        random_state: Random seed

    Returns:
        Random positive definite correlation matrix
    """
    rng = np.random.default_rng(random_state)

    # Generate random orthogonal matrix
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)

    # Generate eigenvalues with specified concentration
    eigenvalues = np.zeros(n)
    eigenvalues[0] = eigenvalue_concentration * n
    remaining = (1 - eigenvalue_concentration) * n
    eigenvalues[1:] = remaining / (n - 1)

    # Construct correlation matrix
    corr = Q @ np.diag(eigenvalues) @ Q.T

    # Normalize to correlation matrix
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)

    return corr
