# PAL Strategies - Reverse Engineered Trading Strategies

Attempt to reverse-engineer trading strategies from Price Action Lab (PAL) based on their published performance metrics.

## Strategies

| Strategy | Type | Assets | Target CAGR | Target DD | Target Sharpe |
|----------|------|--------|-------------|-----------|---------------|
| ETFMR | Mean-reversion | SPY, QQQ | 10.0% | -22.9% | 0.82 |
| B2S2ETF | Mean-reversion | SPY | 9.1% | -30.6% | 0.63 |
| MRMOM | Regime switching | SPY, QQQ, TLT, GLD | 10.3% | -16.8% | 1.15 |

## Usage

```bash
# Run individual strategy with optimization
uv run python tools/pal-strategies/etfmr.py --optimize
uv run python tools/pal-strategies/b2s2etf.py --optimize
uv run python tools/pal-strategies/mrmom.py --optimize

# Run all strategies and generate comparison report
uv run python tools/pal-strategies/run_all.py
```

## Results Summary

| Strategy | Achieved CAGR | Achieved DD | Achieved Sharpe | Match |
|----------|---------------|-------------|-----------------|-------|
| ETFMR | 7.3% | -21.6% | 0.49 | ✗ |
| B2S2ETF | 6.5% | -24.7% | 0.36 | ✗ |
| MRMOM | 7.4% | -17.2% | 0.68 | ✗ |

Simple mean-reversion strategies achieve ~40-70% of PAL's published Sharpe ratios.

## Strategy Details

### ETFMR
RSI-2 mean-reversion with IBS filter on SPY and QQQ.
- Entry: RSI < 20 AND IBS < 0.2
- Exit: RSI > 55

### B2S2ETF
"Buy 2 Sell 2" pattern on SPY.
- Entry: 2 consecutive down days AND RSI < 20
- Exit: After 5 days

### MRMOM
Regime switching using Momersion indicator.
- MR regime (Momersion < 55): RSI mean-reversion
- MOM regime: Trend-following (price > 200 SMA)
- TLT/GLD for risk-off allocation

## Key Learnings

1. Simple mean-reversion generates consistent returns but low Sharpe
2. PAL likely uses proprietary indicator ensembles
3. Volatility-adjusted position sizing would improve results
4. The gap suggests additional undisclosed features

## Files

- `base.py` - Shared backtest infrastructure
- `etfmr.py` - ETFMR strategy
- `b2s2etf.py` - B2S2ETF strategy
- `mrmom.py` - MRMOM strategy
- `run_all.py` - Comparison report generator
