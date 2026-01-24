"""
Market Behavior Simulator for Stress Testing and Forecasting

A comprehensive framework for:
- Multi-term expected returns modeling
- Correlation structures (static, rolling, DCC)
- Stress testing with historical and hypothetical scenarios
- Monte Carlo simulation with various market models
"""

from .core.simulation import MarketSimulator
from .core.models import GBM, JumpDiffusion, MeanReversion, GARCH
from .correlation.structures import CorrelationEngine
from .stress_testing.scenarios import StressTestEngine
from .forecasting.returns import ReturnForecaster
from .visualization.plots import SimulationPlotter

__version__ = "0.1.0"
__all__ = [
    "MarketSimulator",
    "GBM",
    "JumpDiffusion",
    "MeanReversion",
    "GARCH",
    "CorrelationEngine",
    "StressTestEngine",
    "ReturnForecaster",
    "SimulationPlotter",
]
