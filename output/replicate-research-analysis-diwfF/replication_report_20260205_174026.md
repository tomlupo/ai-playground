# Replication: Interpretable Hypothesis-Driven Trading

**Paper:** Deep, Deep & Lamptey (2025) - arXiv:2512.12924v1
**Replication Date:** 2026-02-05 17:40

## Executive Summary

This replicates the walk-forward validation framework for market microstructure
signals across 100 US equities (2015-2024). The framework combines interpretable
hypothesis-driven signal generation with reinforcement learning and strict
out-of-sample testing.

## Aggregate Out-of-Sample Performance

| Metric | Strategy (Replication) | SPY Benchmark | Paper (Reported) |
|--------|----------------------|---------------|------------------|
| Mean Quarterly Return | 0.37% | 3.24% | 0.14% |
| Annualized Return | 1.51% | 13.62% | 0.55% |
| Sharpe Ratio | 0.44 | 0.79 | 0.33 |
| Maximum Drawdown | -3.46% | -22.86% | -2.76% |
| Beta | 0.093 | 1.000 | 0.058 |
| Correlation w/ SPY | 0.44 | 1.00 | 0.53 |
| Fold Win Rate | 71% | 74% | 41% |
| Trade-Level Win Rate | 49.2% | — | 46.5% |
| Total Trades | 258 | — | — |
| Total Folds | 35 | — | 34 |

## Statistical Significance

| Test | Value |
|------|-------|
| t-statistic | 1.29 |
| p-value (two-sided) | 0.2054 |
| Bootstrap 95% CI | [-0.20%, 0.93%] |
| Cohen's d | 0.22 |
| Statistical Power | 25.2% |
| N required (80% power) | 165 |

## Regime-Dependent Performance

| Regime | Mean Quarterly Return | Win Rate | Sharpe | N Folds |
|--------|----------------------|----------|--------|---------|
| Low Volatility 2015 2019 | 0.04% | 62% | 0.05 | 16 |
| High Volatility 2020 2024 | 0.66% | 79% | 0.70 | 19 |
| Covid Crash 2020 Q1Q2 | -0.95% | 50% | -0.66 | 2 |
| Bear Market 2022 | -0.13% | 50% | -0.10 | 4 |
| Stabilization 2023 2024 | 0.55% | 86% | 1.15 | 7 |

## Hypothesis Type Analysis

| Hypothesis | Trades | Win Rate | Mean Return | Sharpe |
|------------|--------|----------|-------------|--------|
| Flow Momentum | 73 | 53.4% | 2.10% | 0.30 |
| Institutional Accumulation | 58 | 41.4% | -0.17% | -0.03 |
| Breakout | 46 | 54.3% | 1.33% | 0.27 |
| Range Bound Value | 43 | 48.8% | 0.53% | 0.11 |
| Mean Reversion | 38 | 47.4% | 0.20% | 0.05 |

## Fold-by-Fold Results

| Fold | Test Period | Return | Trades | Win Rate | Sharpe |
|------|-------------|--------|--------|----------|--------|
| 1 | 2016-01-04 → 2016-04-04 | -0.26% | 11 | 63.6% | -0.36 |
| 2 | 2016-04-05 → 2016-07-01 | 0.71% | 9 | 44.4% | 0.82 |
| 3 | 2016-07-05 → 2016-09-30 | -0.11% | 5 | 40.0% | -0.12 |
| 4 | 2016-10-03 → 2016-12-30 | 2.51% | 6 | 50.0% | 3.17 |
| 5 | 2017-01-03 → 2017-04-03 | 0.28% | 11 | 63.6% | 0.40 |
| 6 | 2017-04-04 → 2017-07-03 | 1.41% | 6 | 66.7% | 2.42 |
| 7 | 2017-07-05 → 2017-10-02 | 0.18% | 8 | 50.0% | 0.29 |
| 8 | 2017-10-03 → 2018-01-02 | 0.46% | 9 | 66.7% | 0.78 |
| 9 | 2018-01-03 → 2018-04-04 | 0.98% | 2 | 100.0% | 0.88 |
| 10 | 2018-04-05 → 2018-07-03 | -0.53% | 9 | 44.4% | -0.58 |
| 11 | 2018-07-05 → 2018-10-02 | 0.37% | 9 | 44.4% | 0.53 |
| 12 | 2018-10-03 → 2019-01-03 | -3.18% | 11 | 36.4% | -3.09 |
| 13 | 2019-01-04 → 2019-04-04 | 1.52% | 8 | 62.5% | 2.06 |
| 14 | 2019-04-05 → 2019-07-05 | -1.51% | 6 | 33.3% | -1.44 |
| 15 | 2019-07-08 → 2019-10-03 | -2.82% | 6 | 16.7% | -3.13 |
| 16 | 2019-10-04 → 2020-01-03 | 0.56% | 5 | 60.0% | 0.66 |
| 17 | 2020-01-06 → 2020-04-03 | -3.00% | 8 | 12.5% | -5.33 |
| 18 | 2020-04-06 → 2020-07-06 | 1.10% | 4 | 50.0% | 1.26 |
| 19 | 2020-07-07 → 2020-10-02 | 2.26% | 4 | 100.0% | 2.50 |
| 20 | 2020-10-05 → 2021-01-04 | 0.56% | 3 | 66.7% | 0.53 |
| 21 | 2021-01-05 → 2021-04-06 | 0.86% | 7 | 57.1% | 0.66 |
| 22 | 2021-04-07 → 2021-07-06 | 0.25% | 8 | 62.5% | 0.31 |
| 23 | 2021-07-07 → 2021-10-04 | 2.46% | 5 | 40.0% | 2.38 |
| 24 | 2021-10-05 → 2022-01-03 | 4.72% | 13 | 61.5% | 3.07 |
| 25 | 2022-01-04 → 2022-04-04 | 1.27% | 7 | 42.9% | 1.27 |
| 26 | 2022-04-05 → 2022-07-06 | 2.48% | 5 | 60.0% | 2.31 |
| 27 | 2022-07-07 → 2022-10-04 | -3.21% | 4 | 25.0% | -2.68 |
| 28 | 2022-10-05 → 2023-01-04 | -1.07% | 5 | 60.0% | -0.89 |
| 29 | 2023-01-05 → 2023-04-05 | 1.12% | 6 | 16.7% | 0.99 |
| 30 | 2023-04-06 → 2023-07-07 | 0.35% | 12 | 41.7% | 0.43 |
| 31 | 2023-07-10 → 2023-10-05 | 0.03% | 10 | 30.0% | 0.07 |
| 32 | 2023-10-06 → 2024-01-05 | 1.72% | 14 | 50.0% | 1.82 |
| 33 | 2024-01-08 → 2024-04-08 | 0.38% | 12 | 50.0% | 0.45 |
| 34 | 2024-04-09 → 2024-07-09 | -1.13% | 7 | 28.6% | -1.02 |
| 35 | 2024-07-10 → 2024-10-07 | 1.37% | 3 | 100.0% | 1.20 |

## Generated Charts

- `equity_curve_20260205_174022.png`
- `quarterly_returns_20260205_174022.png`
- `regime_analysis_20260205_174022.png`
- `hypothesis_analysis_20260205_174022.png`
- `return_distribution_20260205_174022.png`
- `rolling_sharpe_20260205_174022.png`

## Configuration

- Train window: 252 days
- Test window: 63 days
- Step size: 63 days
- Commission: $1.0
- Slippage: 5.0 bps
- Max positions: 5
- Max position size: 20%
- Initial capital: $100,000
- ε (train): 0.7
- ε (test): 0.1

## Methodology Notes

This replication follows the paper's methodology:
1. **Information Set Discipline**: Features use only past data (no lookahead)
2. **Walk-Forward Validation**: Rolling 252/63/63 day windows
3. **Interpretability**: Every trade linked to a named hypothesis with explanation
4. **Realistic Execution**: Commissions, slippage, position limits enforced
5. **ε-greedy RL agent**: Adaptive threshold τ(c) = 0.45 + (1-c)×0.10

### Differences from Paper
- Universe selection: representative stocks matching GICS criteria (exact list may differ)
- Volume classification: close-vs-open proxy (paper may use tick-level data)
- Feature engineering: core 3 microstructure features implemented faithfully

*Report generated: 2026-02-05T17:40:26.482907*