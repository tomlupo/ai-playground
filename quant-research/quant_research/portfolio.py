"""
Portfolio analysis module for multi-asset research.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Optional


class PortfolioAnalyzer:
    """Analyze and optimize investment portfolios."""

    def __init__(self, prices: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize portfolio analyzer.

        Args:
            prices: DataFrame of asset prices (columns are assets)
            risk_free_rate: Annual risk-free rate
        """
        self.prices = prices
        self.risk_free_rate = risk_free_rate
        self.returns = prices.pct_change().dropna()
        self.log_returns = np.log(prices / prices.shift(1)).dropna()

    def calculate_statistics(self) -> pd.DataFrame:
        """
        Calculate basic statistics for each asset.

        Returns:
            DataFrame with statistics for each asset
        """
        stats = pd.DataFrame()

        # Annualized return
        stats["Ann. Return"] = self.returns.mean() * 252

        # Annualized volatility
        stats["Ann. Volatility"] = self.returns.std() * np.sqrt(252)

        # Sharpe ratio
        stats["Sharpe Ratio"] = (
            stats["Ann. Return"] - self.risk_free_rate
        ) / stats["Ann. Volatility"]

        # Max drawdown for each asset
        max_dd = {}
        for col in self.prices.columns:
            rolling_max = self.prices[col].cummax()
            drawdown = (self.prices[col] - rolling_max) / rolling_max
            max_dd[col] = drawdown.min()
        stats["Max Drawdown"] = pd.Series(max_dd)

        # Skewness and Kurtosis
        stats["Skewness"] = self.returns.skew()
        stats["Kurtosis"] = self.returns.kurtosis()

        return stats

    def correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix of returns.

        Returns:
            Correlation matrix
        """
        return self.returns.corr()

    def covariance_matrix(self, annualize: bool = True) -> pd.DataFrame:
        """
        Calculate covariance matrix.

        Args:
            annualize: Whether to annualize

        Returns:
            Covariance matrix
        """
        cov = self.returns.cov()
        if annualize:
            cov = cov * 252
        return cov

    def portfolio_performance(
        self, weights: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Calculate portfolio performance for given weights.

        Args:
            weights: Array of asset weights

        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        """
        weights = np.array(weights)

        # Portfolio return
        port_return = np.sum(self.returns.mean() * weights) * 252

        # Portfolio volatility
        port_vol = np.sqrt(
            np.dot(weights.T, np.dot(self.covariance_matrix(), weights))
        )

        # Sharpe ratio
        sharpe = (port_return - self.risk_free_rate) / port_vol

        return port_return, port_vol, sharpe

    def optimize_sharpe(self) -> tuple[np.ndarray, dict]:
        """
        Find the portfolio with maximum Sharpe ratio.

        Returns:
            Tuple of (optimal weights, performance metrics)
        """
        n_assets = len(self.returns.columns)

        def neg_sharpe(weights):
            return -self.portfolio_performance(weights)[2]

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1 / n_assets] * n_assets)

        result = minimize(
            neg_sharpe,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimal_weights = result.x
        ret, vol, sharpe = self.portfolio_performance(optimal_weights)

        return optimal_weights, {
            "return": ret,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "weights": dict(zip(self.returns.columns, optimal_weights)),
        }

    def optimize_min_variance(self) -> tuple[np.ndarray, dict]:
        """
        Find the minimum variance portfolio.

        Returns:
            Tuple of (optimal weights, performance metrics)
        """
        n_assets = len(self.returns.columns)

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.covariance_matrix(), weights))

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1 / n_assets] * n_assets)

        result = minimize(
            portfolio_variance,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimal_weights = result.x
        ret, vol, sharpe = self.portfolio_performance(optimal_weights)

        return optimal_weights, {
            "return": ret,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "weights": dict(zip(self.returns.columns, optimal_weights)),
        }

    def efficient_frontier(
        self, n_points: int = 50
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """
        Calculate the efficient frontier.

        Args:
            n_points: Number of points on the frontier

        Returns:
            Tuple of (returns array, volatilities array, weights list)
        """
        n_assets = len(self.returns.columns)

        # Get range of possible returns
        min_ret = self.returns.mean().min() * 252
        max_ret = self.returns.mean().max() * 252
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier_vol = []
        frontier_weights = []

        for target_ret in target_returns:

            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(self.covariance_matrix(), weights))

            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {
                    "type": "eq",
                    "fun": lambda x, r=target_ret: np.sum(self.returns.mean() * x) * 252
                    - r,
                },
            ]
            bounds = tuple((0, 1) for _ in range(n_assets))
            initial_weights = np.array([1 / n_assets] * n_assets)

            result = minimize(
                portfolio_variance,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                frontier_vol.append(np.sqrt(result.fun))
                frontier_weights.append(result.x)
            else:
                frontier_vol.append(np.nan)
                frontier_weights.append(np.full(n_assets, np.nan))

        return target_returns, np.array(frontier_vol), frontier_weights

    def equal_weight_portfolio(self) -> dict:
        """
        Calculate equal-weight portfolio performance.

        Returns:
            Performance metrics
        """
        n_assets = len(self.returns.columns)
        weights = np.array([1 / n_assets] * n_assets)
        ret, vol, sharpe = self.portfolio_performance(weights)

        return {
            "return": ret,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "weights": dict(zip(self.returns.columns, weights)),
        }

    def var(
        self, weights: np.ndarray, confidence: float = 0.95, method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            weights: Portfolio weights
            confidence: Confidence level
            method: 'historical' or 'parametric'

        Returns:
            VaR as a positive number (loss)
        """
        portfolio_returns = (self.returns * weights).sum(axis=1)

        if method == "historical":
            var = -np.percentile(portfolio_returns, (1 - confidence) * 100)
        else:  # parametric
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            from scipy.stats import norm

            var = -(mean + std * norm.ppf(1 - confidence))

        return var

    def cvar(self, weights: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            weights: Portfolio weights
            confidence: Confidence level

        Returns:
            CVaR as a positive number (expected loss beyond VaR)
        """
        portfolio_returns = (self.returns * weights).sum(axis=1)
        var = self.var(weights, confidence, method="historical")
        cvar = -portfolio_returns[portfolio_returns <= -var].mean()
        return cvar
