"""
BeGlobal Investment Methodology Implementation.

Replicates the methodology from https://beglobal.pl/metodologia-inwestycyjna

Features:
- Core-Satellite approach (passive + active)
- 13 asset classes across fixed income, equities, and alternatives
- 5 risk profiles (Safe to Profit+)
- Dual Momentum, Relative Strength, and Trend Following strategies
- Corridor rebalancing (±2.5% threshold)
- Conditional volatility targeting
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Literal
from enum import Enum


class RiskProfile(Enum):
    """BeGlobal risk profiles with minimum investment periods."""

    SAFE = "safe"  # Very Low risk, 3 months
    BOND_PLUS = "bond_plus"  # Low risk, 2 years
    MIXED = "mixed"  # Medium risk, 3 years
    PROFIT = "profit"  # High risk, 4 years
    PROFIT_PLUS = "profit_plus"  # Very High risk, 5 years


@dataclass
class AssetClass:
    """Asset class definition."""

    name: str
    category: Literal["fixed_income", "equity", "alternative"]
    etf_ticker: str
    description: str = ""


# 13 Primary Asset Classes with representative ETFs
ASSET_CLASSES = {
    # Fixed Income
    "money_market": AssetClass(
        "Money Market", "fixed_income", "SHV", "0-1 year maturity"
    ),
    "us_treasury_short": AssetClass(
        "US Treasury Short", "fixed_income", "SHY", "1-3 year US Treasuries"
    ),
    "us_treasury_medium": AssetClass(
        "US Treasury Medium", "fixed_income", "IEF", "7-10 year US Treasuries"
    ),
    "us_treasury_long": AssetClass(
        "US Treasury Long", "fixed_income", "TLT", "20+ year US Treasuries"
    ),
    "developed_sovereigns": AssetClass(
        "Developed Sovereigns ex-US", "fixed_income", "BWX", "International govt bonds"
    ),
    "corporate_ig": AssetClass(
        "Corporate Investment Grade", "fixed_income", "LQD", "Investment grade corporate"
    ),
    "corporate_hy": AssetClass(
        "Corporate High Yield", "fixed_income", "HYG", "High yield corporate"
    ),
    "em_bonds": AssetClass(
        "Emerging Market Bonds", "fixed_income", "EMB", "EM sovereign bonds"
    ),
    # Equities
    "us_stocks": AssetClass("US Stocks", "equity", "SPY", "S&P 500"),
    "developed_stocks": AssetClass(
        "Developed Markets ex-US", "equity", "EFA", "MSCI EAFE"
    ),
    "em_stocks": AssetClass("Emerging Market Stocks", "equity", "EEM", "MSCI EM"),
    # Alternatives
    "real_estate": AssetClass("Real Estate", "alternative", "VNQ", "US REITs"),
    "commodities": AssetClass("Commodities", "alternative", "DJP", "Commodity index"),
    "gold": AssetClass("Gold", "alternative", "GLD", "Gold bullion"),
}


# Risk Profile Allocations (passive/core component)
RISK_PROFILE_ALLOCATIONS = {
    RiskProfile.SAFE: {
        "money_market": 0.60,
        "us_treasury_short": 0.30,
        "us_treasury_medium": 0.10,
        "us_stocks": 0.00,
        "gold": 0.00,
    },
    RiskProfile.BOND_PLUS: {
        "money_market": 0.30,
        "us_treasury_short": 0.35,
        "us_treasury_medium": 0.20,
        "corporate_ig": 0.10,
        "us_stocks": 0.05,
        "gold": 0.00,
    },
    RiskProfile.MIXED: {
        "money_market": 0.20,
        "us_treasury_short": 0.35,
        "us_treasury_medium": 0.15,
        "us_stocks": 0.20,
        "developed_stocks": 0.05,
        "gold": 0.05,
    },
    RiskProfile.PROFIT: {
        "money_market": 0.10,
        "us_treasury_short": 0.20,
        "us_treasury_medium": 0.10,
        "us_stocks": 0.35,
        "developed_stocks": 0.10,
        "em_stocks": 0.05,
        "real_estate": 0.05,
        "gold": 0.05,
    },
    RiskProfile.PROFIT_PLUS: {
        "money_market": 0.05,
        "us_treasury_short": 0.10,
        "us_stocks": 0.45,
        "developed_stocks": 0.15,
        "em_stocks": 0.10,
        "real_estate": 0.05,
        "commodities": 0.05,
        "gold": 0.05,
    },
}


@dataclass
class MomentumSignal:
    """Momentum signal container."""

    asset: str
    absolute_momentum: float
    relative_momentum: float
    combined_signal: Literal["long", "short", "neutral"]
    score: float


@dataclass
class RebalanceAction:
    """Rebalancing action."""

    asset: str
    current_weight: float
    target_weight: float
    action: Literal["buy", "sell", "hold"]
    amount_pct: float


class DualMomentumStrategy:
    """
    Dual Momentum Strategy (Antonacci-style).

    Combines:
    1. Absolute momentum (trend following)
    2. Relative momentum (cross-sectional)
    """

    def __init__(
        self,
        lookback_period: int = 252,  # 12 months
        short_lookback: int = 21,  # 1 month for regime
        risk_free_asset: str = "money_market",
    ):
        self.lookback_period = lookback_period
        self.short_lookback = short_lookback
        self.risk_free_asset = risk_free_asset

    def calculate_momentum(self, prices: pd.Series, period: int) -> float:
        """Calculate momentum as total return over period."""
        if len(prices) < period:
            return 0.0
        return (prices.iloc[-1] / prices.iloc[-period]) - 1

    def calculate_absolute_momentum(
        self, prices: pd.Series, risk_free_return: float = 0.0
    ) -> bool:
        """
        Absolute momentum: Is the asset trending up vs risk-free?

        Returns True if momentum > risk-free return.
        """
        momentum = self.calculate_momentum(prices, self.lookback_period)
        return momentum > risk_free_return

    def calculate_relative_momentum(
        self, prices_dict: dict[str, pd.Series]
    ) -> pd.Series:
        """
        Relative momentum: Rank assets by momentum.

        Returns Series with momentum scores.
        """
        scores = {}
        for asset, prices in prices_dict.items():
            scores[asset] = self.calculate_momentum(prices, self.lookback_period)
        return pd.Series(scores).sort_values(ascending=False)

    def generate_signals(
        self,
        prices_dict: dict[str, pd.Series],
        risk_free_prices: pd.Series | None = None,
    ) -> dict[str, MomentumSignal]:
        """
        Generate dual momentum signals for all assets.

        Args:
            prices_dict: Dict of asset prices
            risk_free_prices: Risk-free asset prices for benchmark

        Returns:
            Dict of MomentumSignal for each asset
        """
        # Calculate risk-free return
        if risk_free_prices is not None and len(risk_free_prices) >= self.lookback_period:
            rf_return = self.calculate_momentum(risk_free_prices, self.lookback_period)
        else:
            rf_return = 0.02 / 252 * self.lookback_period  # Approximate

        # Relative momentum scores
        rel_scores = self.calculate_relative_momentum(prices_dict)
        rel_rank = rel_scores.rank(ascending=False, pct=True)

        signals = {}
        for asset, prices in prices_dict.items():
            # Absolute momentum
            abs_mom = self.calculate_momentum(prices, self.lookback_period)
            abs_positive = abs_mom > rf_return

            # Relative momentum
            rel_mom = rel_scores[asset]
            rel_percentile = rel_rank[asset]

            # Combined signal
            if abs_positive and rel_percentile >= 0.5:
                signal = "long"
                score = (abs_mom + rel_mom) / 2
            elif not abs_positive:
                signal = "neutral"  # Move to risk-free
                score = 0.0
            else:
                signal = "short" if rel_percentile < 0.25 else "neutral"
                score = rel_mom

            signals[asset] = MomentumSignal(
                asset=asset,
                absolute_momentum=abs_mom,
                relative_momentum=rel_mom,
                combined_signal=signal,
                score=score,
            )

        return signals


class RelativeStrengthStrategy:
    """
    Relative Strength Strategy.

    Over/underweight market segments based on comparative momentum.
    """

    def __init__(
        self,
        lookback_periods: list[int] | None = None,
        top_n: int = 3,
    ):
        self.lookback_periods = lookback_periods or [21, 63, 126, 252]
        self.top_n = top_n

    def calculate_composite_momentum(self, prices: pd.Series) -> float:
        """Calculate composite momentum across multiple timeframes."""
        scores = []
        for period in self.lookback_periods:
            if len(prices) >= period:
                ret = (prices.iloc[-1] / prices.iloc[-period]) - 1
                scores.append(ret)
        return np.mean(scores) if scores else 0.0

    def rank_assets(self, prices_dict: dict[str, pd.Series]) -> pd.DataFrame:
        """Rank assets by composite momentum."""
        scores = {}
        for asset, prices in prices_dict.items():
            scores[asset] = self.calculate_composite_momentum(prices)

        df = pd.DataFrame({
            "asset": list(scores.keys()),
            "momentum": list(scores.values()),
        })
        df["rank"] = df["momentum"].rank(ascending=False)
        df["percentile"] = df["momentum"].rank(ascending=False, pct=True)
        return df.sort_values("rank")

    def generate_weights(
        self,
        prices_dict: dict[str, pd.Series],
        base_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Generate adjusted weights based on relative strength.

        Args:
            prices_dict: Asset prices
            base_weights: Base/neutral weights

        Returns:
            Adjusted weights
        """
        rankings = self.rank_assets(prices_dict)
        top_assets = set(rankings.head(self.top_n)["asset"])
        bottom_assets = set(rankings.tail(self.top_n)["asset"])

        adjusted = {}
        total_adjustment = 0.0

        for asset, base_weight in base_weights.items():
            if asset in top_assets:
                # Overweight top performers
                adjustment = base_weight * 0.25
                adjusted[asset] = base_weight + adjustment
                total_adjustment += adjustment
            elif asset in bottom_assets:
                # Underweight bottom performers
                adjustment = base_weight * 0.25
                adjusted[asset] = max(0, base_weight - adjustment)
                total_adjustment -= adjustment
            else:
                adjusted[asset] = base_weight

        # Normalize to sum to 1
        total = sum(adjusted.values())
        return {k: v / total for k, v in adjusted.items()}


class TrendFollowingStrategy:
    """
    Trend Following Strategy.

    Buy assets with positive trends, avoid declining assets.
    """

    def __init__(
        self,
        short_ma: int = 50,
        long_ma: int = 200,
        atr_period: int = 14,
    ):
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.atr_period = atr_period

    def calculate_trend(self, prices: pd.Series) -> Literal["up", "down", "neutral"]:
        """Determine trend based on moving average crossover."""
        if len(prices) < self.long_ma:
            return "neutral"

        sma_short = prices.rolling(self.short_ma).mean().iloc[-1]
        sma_long = prices.rolling(self.long_ma).mean().iloc[-1]
        current = prices.iloc[-1]

        if current > sma_short > sma_long:
            return "up"
        elif current < sma_short < sma_long:
            return "down"
        else:
            return "neutral"

    def generate_signals(
        self, prices_dict: dict[str, pd.Series]
    ) -> dict[str, Literal["long", "short", "flat"]]:
        """Generate trend following signals."""
        signals = {}
        for asset, prices in prices_dict.items():
            trend = self.calculate_trend(prices)
            if trend == "up":
                signals[asset] = "long"
            elif trend == "down":
                signals[asset] = "flat"
            else:
                signals[asset] = "flat"
        return signals


class CorridorRebalancer:
    """
    Corridor-based Rebalancing.

    Rebalance when weights deviate beyond threshold (±2.5% default).
    """

    def __init__(self, threshold: float = 0.025):
        self.threshold = threshold

    def check_rebalance_needed(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> bool:
        """Check if any position needs rebalancing."""
        for asset in target_weights:
            current = current_weights.get(asset, 0.0)
            target = target_weights[asset]
            if abs(current - target) > self.threshold:
                return True
        return False

    def generate_rebalance_actions(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> list[RebalanceAction]:
        """Generate specific rebalancing actions."""
        actions = []

        for asset in set(current_weights.keys()) | set(target_weights.keys()):
            current = current_weights.get(asset, 0.0)
            target = target_weights.get(asset, 0.0)
            diff = target - current

            if abs(diff) > self.threshold:
                action = "buy" if diff > 0 else "sell"
                actions.append(RebalanceAction(
                    asset=asset,
                    current_weight=current,
                    target_weight=target,
                    action=action,
                    amount_pct=abs(diff),
                ))
            else:
                actions.append(RebalanceAction(
                    asset=asset,
                    current_weight=current,
                    target_weight=target,
                    action="hold",
                    amount_pct=0.0,
                ))

        return sorted(actions, key=lambda x: abs(x.amount_pct), reverse=True)


class VolatilityTargeting:
    """
    Conditional Volatility Targeting.

    Adjust exposure only during extreme volatility regimes.
    """

    def __init__(
        self,
        target_volatility: float = 0.10,  # 10% annualized
        lookback: int = 20,
        high_vol_threshold: float = 1.5,  # 1.5x normal vol triggers adjustment
        low_vol_threshold: float = 0.5,
    ):
        self.target_volatility = target_volatility
        self.lookback = lookback
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold

    def calculate_realized_volatility(self, returns: pd.Series) -> float:
        """Calculate realized volatility."""
        if len(returns) < self.lookback:
            return self.target_volatility
        return returns.tail(self.lookback).std() * np.sqrt(252)

    def get_volatility_scalar(self, returns: pd.Series) -> float:
        """
        Get position scalar based on volatility regime.

        Returns 1.0 for normal vol, <1.0 for high vol, >1.0 for low vol.
        """
        realized_vol = self.calculate_realized_volatility(returns)
        vol_ratio = realized_vol / self.target_volatility

        if vol_ratio > self.high_vol_threshold:
            # High volatility regime - reduce exposure
            return self.target_volatility / realized_vol
        elif vol_ratio < self.low_vol_threshold:
            # Low volatility regime - can increase slightly
            return min(1.2, self.target_volatility / realized_vol)
        else:
            # Normal regime - no adjustment
            return 1.0


class BeGlobalPortfolio:
    """
    Complete BeGlobal Portfolio Implementation.

    Combines all components:
    - Risk profile allocation
    - Dual momentum satellite
    - Corridor rebalancing
    - Volatility targeting
    """

    def __init__(
        self,
        risk_profile: RiskProfile = RiskProfile.MIXED,
        core_weight: float = 0.7,  # 70% passive core
        satellite_weight: float = 0.3,  # 30% active satellite
        rebalance_threshold: float = 0.025,
        target_volatility: float = 0.10,
    ):
        self.risk_profile = risk_profile
        self.core_weight = core_weight
        self.satellite_weight = satellite_weight

        # Get base allocation for risk profile
        self.base_allocation = RISK_PROFILE_ALLOCATIONS[risk_profile]

        # Initialize strategies
        self.dual_momentum = DualMomentumStrategy()
        self.relative_strength = RelativeStrengthStrategy()
        self.trend_following = TrendFollowingStrategy()
        self.rebalancer = CorridorRebalancer(threshold=rebalance_threshold)
        self.vol_targeting = VolatilityTargeting(target_volatility=target_volatility)

    def get_etf_tickers(self) -> list[str]:
        """Get list of ETF tickers for the portfolio."""
        tickers = []
        for asset in self.base_allocation.keys():
            if asset in ASSET_CLASSES:
                tickers.append(ASSET_CLASSES[asset].etf_ticker)
        return tickers

    def calculate_satellite_weights(
        self,
        prices_dict: dict[str, pd.Series],
    ) -> dict[str, float]:
        """Calculate active satellite weights using momentum strategies."""
        # Get momentum signals
        signals = self.dual_momentum.generate_signals(prices_dict)

        # Calculate weights based on signals
        long_assets = [s.asset for s in signals.values() if s.combined_signal == "long"]

        if not long_assets:
            # If no long signals, go to risk-free
            return {"money_market": 1.0}

        # Equal weight among long assets
        weight_per_asset = 1.0 / len(long_assets)
        weights = {asset: weight_per_asset for asset in long_assets}

        return weights

    def calculate_portfolio_weights(
        self,
        prices_dict: dict[str, pd.Series],
        portfolio_returns: pd.Series | None = None,
    ) -> dict[str, float]:
        """
        Calculate combined portfolio weights.

        Args:
            prices_dict: Asset prices
            portfolio_returns: Historical portfolio returns for vol targeting

        Returns:
            Combined weights
        """
        # Core (passive) allocation
        core_weights = {
            k: v * self.core_weight for k, v in self.base_allocation.items()
        }

        # Satellite (active) allocation
        satellite_raw = self.calculate_satellite_weights(prices_dict)
        satellite_weights = {
            k: v * self.satellite_weight for k, v in satellite_raw.items()
        }

        # Combine
        combined = {}
        all_assets = set(core_weights.keys()) | set(satellite_weights.keys())
        for asset in all_assets:
            combined[asset] = core_weights.get(asset, 0) + satellite_weights.get(asset, 0)

        # Apply volatility targeting if returns available
        if portfolio_returns is not None and len(portfolio_returns) > 20:
            vol_scalar = self.vol_targeting.get_volatility_scalar(portfolio_returns)
            if vol_scalar != 1.0:
                # Scale risky assets, keep safe assets
                safe_assets = ["money_market", "us_treasury_short"]
                for asset in combined:
                    if asset not in safe_assets:
                        combined[asset] *= vol_scalar
                # Add remainder to money market
                total = sum(combined.values())
                if total < 1.0:
                    combined["money_market"] = combined.get("money_market", 0) + (1.0 - total)

        # Normalize
        total = sum(combined.values())
        return {k: v / total for k, v in combined.items()}

    def run_backtest(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 100000,
        rebalance_check_freq: str = "D",  # Daily monitoring
    ) -> dict:
        """
        Run backtest of the BeGlobal strategy.

        Args:
            prices: DataFrame with asset prices (columns are assets)
            initial_capital: Starting capital
            rebalance_check_freq: How often to check rebalancing

        Returns:
            Backtest results dict
        """
        # Map ETF tickers to asset names
        ticker_to_asset = {
            ASSET_CLASSES[a].etf_ticker: a
            for a in self.base_allocation.keys()
            if a in ASSET_CLASSES
        }

        # Ensure we have the right columns
        available_assets = {}
        for col in prices.columns:
            if col in ticker_to_asset:
                available_assets[ticker_to_asset[col]] = prices[col]
            elif col in ASSET_CLASSES:
                available_assets[col] = prices[col]

        if not available_assets:
            raise ValueError("No matching assets found in price data")

        # Initialize tracking
        portfolio_value = [initial_capital]
        weights_history = []
        rebalance_dates = []

        returns = pd.DataFrame({k: v.pct_change() for k, v in available_assets.items()})
        returns = returns.dropna()

        current_weights = self.calculate_portfolio_weights(available_assets)
        weights_history.append(current_weights.copy())

        # Track position values
        positions = {a: initial_capital * w for a, w in current_weights.items()}

        for i, date in enumerate(returns.index[1:], 1):
            # Update position values with returns
            daily_returns = returns.loc[date]
            for asset in positions:
                if asset in daily_returns.index:
                    positions[asset] *= (1 + daily_returns[asset])

            total_value = sum(positions.values())
            portfolio_value.append(total_value)

            # Calculate current weights
            current_weights = {a: v / total_value for a, v in positions.items()}

            # Check rebalancing (daily as per BeGlobal)
            prices_to_date = {a: p.loc[:date] for a, p in available_assets.items()}
            port_returns = pd.Series(portfolio_value).pct_change().dropna()

            target_weights = self.calculate_portfolio_weights(
                prices_to_date, port_returns
            )

            if self.rebalancer.check_rebalance_needed(current_weights, target_weights):
                # Rebalance
                rebalance_dates.append(date)
                positions = {a: total_value * w for a, w in target_weights.items()}
                current_weights = target_weights.copy()

            weights_history.append(current_weights.copy())

        # Calculate metrics
        equity = pd.Series(portfolio_value, index=returns.index)
        port_returns = equity.pct_change().dropna()

        total_return = (equity.iloc[-1] - initial_capital) / initial_capital
        days = (equity.index[-1] - equity.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        volatility = port_returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0

        # Max drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = drawdown.min()

        return {
            "equity": equity,
            "returns": port_returns,
            "weights_history": weights_history,
            "rebalance_dates": rebalance_dates,
            "metrics": {
                "total_return": total_return,
                "annual_return": annual_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "rebalance_count": len(rebalance_dates),
            },
            "final_weights": current_weights,
        }


def create_beglobal_portfolio(
    risk_profile: str = "mixed",
    core_weight: float = 0.7,
) -> BeGlobalPortfolio:
    """
    Factory function to create BeGlobal portfolio.

    Args:
        risk_profile: One of "safe", "bond_plus", "mixed", "profit", "profit_plus"
        core_weight: Weight of passive core (default 70%)

    Returns:
        Configured BeGlobalPortfolio
    """
    profile_map = {
        "safe": RiskProfile.SAFE,
        "bond_plus": RiskProfile.BOND_PLUS,
        "mixed": RiskProfile.MIXED,
        "profit": RiskProfile.PROFIT,
        "profit_plus": RiskProfile.PROFIT_PLUS,
    }

    profile = profile_map.get(risk_profile.lower(), RiskProfile.MIXED)

    return BeGlobalPortfolio(
        risk_profile=profile,
        core_weight=core_weight,
        satellite_weight=1.0 - core_weight,
    )
