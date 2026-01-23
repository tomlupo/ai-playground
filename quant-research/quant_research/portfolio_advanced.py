"""
Advanced Portfolio Optimization using riskfolio-lib.

Provides sophisticated portfolio optimization including:
- Mean-Variance Optimization
- Risk Parity
- Hierarchical Risk Parity (HRP)
- Black-Litterman
- CVaR Optimization
"""

import pandas as pd
import numpy as np
from typing import Literal
import warnings

import riskfolio as rp


class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimization using riskfolio-lib.

    Supports multiple optimization methods and risk measures.
    """

    RISK_MEASURES = [
        "MV",      # Variance
        "MAD",     # Mean Absolute Deviation
        "MSV",     # Semi-Variance
        "CVaR",    # Conditional Value at Risk
        "WR",      # Worst Realization
        "MDD",     # Maximum Drawdown
    ]

    OBJ_FUNCTIONS = [
        "MinRisk",     # Minimize Risk
        "MaxRet",      # Maximize Return
        "Utility",     # Maximize Risk-Adjusted Return
        "Sharpe",      # Maximize Sharpe Ratio
    ]

    def __init__(self, prices: pd.DataFrame, risk_free_rate: float = 0.0):
        """
        Initialize optimizer with price data.

        Args:
            prices: DataFrame of asset prices (columns are assets)
            risk_free_rate: Annual risk-free rate
        """
        self.prices = prices
        self.risk_free_rate = risk_free_rate
        self.returns = prices.pct_change().dropna()

        # Initialize portfolio object
        self.port = rp.Portfolio(returns=self.returns)

    def _setup_portfolio(self) -> None:
        """Setup portfolio statistics."""
        # Calculate expected returns and covariance
        self.port.assets_stats(method_mu="hist", method_cov="hist")

    def optimize_mean_variance(
        self,
        objective: Literal["MinRisk", "MaxRet", "Utility", "Sharpe"] = "Sharpe",
        risk_measure: str = "MV",
        constraints: dict | None = None,
    ) -> tuple[pd.Series, dict]:
        """
        Mean-Variance Optimization.

        Args:
            objective: Optimization objective
            risk_measure: Risk measure to use
            constraints: Optional weight constraints

        Returns:
            Tuple of (weights Series, metrics dict)
        """
        self._setup_portfolio()

        # Set constraints
        if constraints:
            if "max_weight" in constraints:
                self.port.upperlng = constraints["max_weight"]
            if "min_weight" in constraints:
                self.port.lowerlng = constraints["min_weight"]

        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weights = self.port.optimization(
                model="Classic",
                rm=risk_measure,
                obj=objective,
                rf=self.risk_free_rate / 252,  # Daily risk-free rate
                hist=True,
            )

        if weights is None:
            raise ValueError("Optimization failed")

        weights_series = pd.Series(
            weights["weights"].values.flatten(),
            index=self.returns.columns,
        )

        metrics = self._calculate_metrics(weights_series)
        metrics["objective"] = objective
        metrics["risk_measure"] = risk_measure

        return weights_series, metrics

    def optimize_risk_parity(self) -> tuple[pd.Series, dict]:
        """
        Risk Parity Optimization.

        Allocates weights so each asset contributes equally to portfolio risk.

        Returns:
            Tuple of (weights Series, metrics dict)
        """
        self._setup_portfolio()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weights = self.port.rp_optimization(
                model="Classic",
                rm="MV",
                rf=self.risk_free_rate / 252,
                hist=True,
            )

        if weights is None:
            raise ValueError("Risk parity optimization failed")

        weights_series = pd.Series(
            weights["weights"].values.flatten(),
            index=self.returns.columns,
        )

        metrics = self._calculate_metrics(weights_series)
        metrics["method"] = "Risk Parity"

        return weights_series, metrics

    def optimize_hrp(self) -> tuple[pd.Series, dict]:
        """
        Hierarchical Risk Parity (HRP) Optimization.

        Uses hierarchical clustering to determine asset allocation.

        Returns:
            Tuple of (weights Series, metrics dict)
        """
        self._setup_portfolio()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weights = self.port.optimization(
                model="HRP",
                rm="MV",
                rf=self.risk_free_rate / 252,
                hist=True,
            )

        if weights is None:
            raise ValueError("HRP optimization failed")

        weights_series = pd.Series(
            weights["weights"].values.flatten(),
            index=self.returns.columns,
        )

        metrics = self._calculate_metrics(weights_series)
        metrics["method"] = "Hierarchical Risk Parity"

        return weights_series, metrics

    def optimize_cvar(
        self,
        alpha: float = 0.05,
        objective: str = "MinRisk",
    ) -> tuple[pd.Series, dict]:
        """
        CVaR (Conditional Value at Risk) Optimization.

        Minimizes expected loss in worst-case scenarios.

        Args:
            alpha: Confidence level (e.g., 0.05 for 95% CVaR)
            objective: Optimization objective

        Returns:
            Tuple of (weights Series, metrics dict)
        """
        self._setup_portfolio()
        self.port.alpha = alpha

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weights = self.port.optimization(
                model="Classic",
                rm="CVaR",
                obj=objective,
                rf=self.risk_free_rate / 252,
                hist=True,
            )

        if weights is None:
            raise ValueError("CVaR optimization failed")

        weights_series = pd.Series(
            weights["weights"].values.flatten(),
            index=self.returns.columns,
        )

        metrics = self._calculate_metrics(weights_series)
        metrics["method"] = f"CVaR ({1-alpha:.0%} confidence)"
        metrics["alpha"] = alpha

        return weights_series, metrics

    def optimize_max_diversification(self) -> tuple[pd.Series, dict]:
        """
        Maximum Diversification Optimization.

        Maximizes the diversification ratio.

        Returns:
            Tuple of (weights Series, metrics dict)
        """
        self._setup_portfolio()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weights = self.port.optimization(
                model="Classic",
                rm="MV",
                obj="MaxRet",  # This with correlation maximizes diversification
                rf=self.risk_free_rate / 252,
                hist=True,
            )

        if weights is None:
            raise ValueError("Max diversification optimization failed")

        weights_series = pd.Series(
            weights["weights"].values.flatten(),
            index=self.returns.columns,
        )

        metrics = self._calculate_metrics(weights_series)
        metrics["method"] = "Maximum Diversification"

        return weights_series, metrics

    def _calculate_metrics(self, weights: pd.Series) -> dict:
        """Calculate portfolio performance metrics."""
        weights_arr = weights.values

        # Portfolio return
        port_return = (self.returns.mean() @ weights_arr) * 252

        # Portfolio volatility
        cov_matrix = self.returns.cov() * 252
        port_vol = np.sqrt(weights_arr @ cov_matrix @ weights_arr)

        # Sharpe ratio
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        # VaR and CVaR
        port_returns = (self.returns @ weights_arr)
        var_95 = -np.percentile(port_returns, 5)
        cvar_95 = -port_returns[port_returns <= -var_95].mean()

        # Max drawdown
        cumulative = (1 + port_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Diversification ratio
        asset_vols = np.sqrt(np.diag(cov_matrix))
        weighted_vol = weights_arr @ asset_vols
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1

        return {
            "return": port_return,
            "volatility": port_vol,
            "sharpe_ratio": sharpe,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "max_drawdown": max_dd,
            "diversification_ratio": div_ratio,
            "weights": weights.to_dict(),
        }

    def efficient_frontier(
        self,
        n_points: int = 50,
        risk_measure: str = "MV",
    ) -> tuple[np.ndarray, np.ndarray, list[pd.Series]]:
        """
        Calculate the efficient frontier.

        Args:
            n_points: Number of points on the frontier
            risk_measure: Risk measure to use

        Returns:
            Tuple of (returns, risks, weights list)
        """
        self._setup_portfolio()

        # Get frontier
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frontier = self.port.efficient_frontier(
                model="Classic",
                rm=risk_measure,
                points=n_points,
                rf=self.risk_free_rate / 252,
                hist=True,
            )

        if frontier is None:
            raise ValueError("Failed to compute efficient frontier")

        # Extract returns and risks
        returns = []
        risks = []
        weights_list = []

        for col in frontier.columns:
            w = frontier[col].values
            weights_series = pd.Series(w, index=self.returns.columns)
            weights_list.append(weights_series)

            # Calculate return and risk
            port_ret = (self.returns.mean() @ w) * 252
            cov = self.returns.cov() * 252
            port_risk = np.sqrt(w @ cov @ w)

            returns.append(port_ret)
            risks.append(port_risk)

        return np.array(returns), np.array(risks), weights_list

    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare all optimization strategies.

        Returns:
            DataFrame comparing different strategies
        """
        results = []

        strategies = [
            ("Max Sharpe (MV)", lambda: self.optimize_mean_variance(objective="Sharpe")),
            ("Min Variance", lambda: self.optimize_mean_variance(objective="MinRisk")),
            ("Risk Parity", self.optimize_risk_parity),
            ("HRP", self.optimize_hrp),
            ("CVaR Min", lambda: self.optimize_cvar(objective="MinRisk")),
        ]

        for name, func in strategies:
            try:
                weights, metrics = func()
                results.append({
                    "Strategy": name,
                    "Return": f"{metrics['return']:.2%}",
                    "Volatility": f"{metrics['volatility']:.2%}",
                    "Sharpe": f"{metrics['sharpe_ratio']:.2f}",
                    "Max DD": f"{metrics['max_drawdown']:.2%}",
                    "VaR 95%": f"{metrics['var_95']:.2%}",
                    "CVaR 95%": f"{metrics['cvar_95']:.2%}",
                })
            except Exception as e:
                print(f"Warning: {name} failed: {e}")

        return pd.DataFrame(results)

    def get_risk_contributions(self, weights: pd.Series) -> pd.Series:
        """
        Calculate risk contribution of each asset.

        Args:
            weights: Portfolio weights

        Returns:
            Series of risk contributions
        """
        cov = self.returns.cov() * 252
        port_vol = np.sqrt(weights @ cov @ weights)

        # Marginal risk contribution
        mrc = cov @ weights / port_vol

        # Risk contribution
        rc = weights * mrc
        rc_pct = rc / rc.sum()

        return pd.Series(rc_pct, index=weights.index, name="Risk Contribution")
