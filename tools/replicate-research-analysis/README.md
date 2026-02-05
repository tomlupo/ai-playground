# Replicate Research Analysis

Replication of **"Interpretable Hypothesis-Driven Trading: A Rigorous Walk-Forward Validation Framework for Market Microstructure Signals"** by Deep, Deep & Lamptey (Texas Tech, Dec 2025).

**Paper:** [arXiv:2512.12924v1](https://arxiv.org/html/2512.12924v1)

## What It Does

Replicates the paper's core methodology:

1. **Universe**: 100 US equities (10 per GICS sector), 2015-2024
2. **Features**: Volume imbalance, volume ratio, price efficiency
3. **Hypotheses**: 5 interpretable types (institutional accumulation, flow momentum, mean reversion, breakout, range-bound value)
4. **RL Agent**: ε-greedy with adaptive threshold τ(c) = 0.45 + (1-c)×0.10
5. **Validation**: Walk-forward with 252d train / 63d test / 63d step
6. **Execution**: Realistic costs ($1 commission, 5bps slippage, position limits)

## Usage

```bash
uv run tools/replicate-research-analysis/main.py
```

## Outputs

- `outputs/replicate-research-analysis/replication_report_*.md` — Full report
- `outputs/replicate-research-analysis/equity_curve_*.png` — Cumulative equity
- `outputs/replicate-research-analysis/quarterly_returns_*.png` — Per-fold returns
- `outputs/replicate-research-analysis/regime_analysis_*.png` — Regime comparison
- `outputs/replicate-research-analysis/hypothesis_analysis_*.png` — Hypothesis breakdown
- `outputs/replicate-research-analysis/return_distribution_*.png` — Return distribution with CI
- `outputs/replicate-research-analysis/rolling_sharpe_*.png` — Rolling Sharpe ratio

## Paper's Key Findings (for comparison)

| Metric | Paper Result |
|--------|-------------|
| Annualized Return | 0.55% |
| Sharpe Ratio | 0.33 |
| Max Drawdown | -2.76% |
| Beta | 0.058 |
| Trade Win Rate | 46.5% |
| Fold Win Rate | 41% |
| t-stat | 0.96 (p=0.34) |
