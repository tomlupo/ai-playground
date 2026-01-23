"""
Analytics module for reconciliation and performance analysis.

Provides:
- Account reconciliation (local vs broker records)
- Discrepancy detection and monitoring
- Performance metrics calculation
- Risk analytics
"""

from trading.analytics.reconciliation import (
    AccountReconciler,
    ReconciliationResult,
    Discrepancy,
    DiscrepancyType,
)
from trading.analytics.performance import (
    PerformanceAnalyzer,
    PerformanceReport,
    TradeStats,
    DrawdownInfo,
    RiskMetrics,
    format_performance_report,
)

__all__ = [
    # Reconciliation
    "AccountReconciler",
    "ReconciliationResult",
    "Discrepancy",
    "DiscrepancyType",
    # Performance
    "PerformanceAnalyzer",
    "PerformanceReport",
    "TradeStats",
    "DrawdownInfo",
    "RiskMetrics",
    "format_performance_report",
]
