# Reverse Engineer PAL Strategies

Run with:
```bash
claude -p "$(cat docs/prompts/reverse-engineer-pal-strategies.md)" --dangerously-skip-permissions
```

---

## Execution Mode: AUTONOMOUS

## Skills to Invoke

1. **Before writing signal generation code:**
   - `/context7 pandas vectorization patterns`
   - `/context7 numpy broadcasting`

2. **For each strategy:**
   - Check `shared/indicators.py` for existing functions

3. **At completion:**
   - `/retrospective` - Document learnings
   - `/gist-report` - Share results

## Vectorization Requirement

**CRITICAL:** All signal generation MUST use vectorized operations:
```python
# Use np.where instead of loops
signal = np.where((rsi < entry) & (ibs < threshold), 1, 0)

# Use shift for position tracking
position = signal.where(signal == 1).ffill().fillna(0)
```

Row-by-row loops will cause timeout on optimization grids.

## Iterative Refinement Loop

**CRITICAL:** Do NOT accept the first optimization result. Keep iterating until metrics are close to targets.

### For Each Strategy:

1. **Initial Grid Search** - Run with standard parameter ranges
2. **Evaluate Against Targets:**
   | Metric | Tolerance |
   |--------|-----------|
   | Sharpe | ±0.15 |
   | CAGR | ±3% |
   | Max DD | ±10% |

3. **If ANY metric outside tolerance:**
   - Print: `"Iteration N: Sharpe=X.XX (target Y.YY, gap Z.ZZ)"`
   - Analyze which metric is furthest off
   - Adjust parameter grid based on gap analysis
   - **GO BACK TO STEP 1**

4. **Gap-Based Adjustments:**
   | Gap Type | Adjustment |
   |----------|------------|
   | Sharpe too low | Try longer holds (hold_days +2), tighter exits (rsi_exit -10) |
   | CAGR too low | Lower entry thresholds (rsi_entry -5), more signals |
   | DD too high | Shorter holds (hold_days -2), faster exits (rsi_exit -5) |
   | All metrics off | Try completely different signal approach (switch RSI→IBS or add filters) |

5. **Max 5 iterations per strategy** - then accept best result and move on

### Iteration History

Track and print progress for each strategy:
```
ETFMR Iteration 1: Sharpe=0.49, CAGR=7.3%, MDD=-21.6% (target: 0.82, 10.0%, -22.9%)
  → Gap: Sharpe -0.33, CAGR -2.7%
  → Adjustment: Trying rsi_thresh range [30-50] instead of [10-30]
ETFMR Iteration 2: Sharpe=0.61, CAGR=8.1%, MDD=-20.5%
  → Gap: Sharpe -0.21, CAGR -1.9%
  → Adjustment: Adding IBS filter with threshold 0.15
ETFMR Iteration 3: Sharpe=0.71, CAGR=9.2%, MDD=-22.1%
  → Within tolerance! SUCCESS.
```

### Success Criteria

Strategy is "successfully replicated" when ALL of:
- Sharpe within ±0.15 of target
- CAGR within ±3% of target
- Max DD within ±10% of target

OR max iterations (5) reached → report best attempt with explanation of gap.

### Reference: Previous Best Results

These parameters achieved good results in earlier runs - use as starting points:
```python
# ETFMR - achieved 0.71 Sharpe (target 0.82)
{'rsi_thresh': 40, 'ibs_thresh': 0.15}

# B2S2ETF - achieved 0.53 Sharpe (target 0.63)
{'down_days': 1, 'up_days': 1}

# MRETF - achieved 0.66 Sharpe (target 0.78)
{'entry': 0.1, 'exit': 0.9, 'max_hold': 15}
```

## Task

Reverse engineer trading strategies from Price Action Lab based on their published performance metrics. Create implementable Python backtests that attempt to replicate their reported results.

## Background

Price Action Lab (PAL) publishes strategy performance metrics without revealing exact rules. This task involves:
1. Forming hypotheses about likely rules based on strategy names and characteristics
2. Systematically testing parameter combinations
3. Validating against target metrics
4. Documenting successful reverse-engineered strategies

## Target Strategies

### Tier 1: ETF Mean-Reversion (Start Here - Best Data Availability)

| Strategy | Type | Markets | CAGR | MDD | Sharpe | Notes |
|----------|------|---------|------|-----|--------|-------|
| ETFMR | Mean-reversion | SPY, QQQ (long-only) | 10.0% | -22.9% | 0.82 | **Start here** |
| MRETF | Mean-reversion + breakouts | SPY, QQQ, TLT (long-only) | 5.2% | -9.3% | 0.78 | Lower CAGR but tight DD |
| B2S2ETF | Mean-reversion | SPY (long-only) | 9.1% | -30.6% | 0.63 | Single asset |

### Tier 2: Multi-Asset/Regime (More Complex)

| Strategy | Type | Markets | CAGR | MDD | Sharpe | Notes |
|----------|------|---------|------|-----|--------|-------|
| MRMOM | Regime switching | SPY, QQQ, TLT, GLD (long-only) | 10.3% | -16.8% | 1.15 | **Best risk-adjusted** |
| ETFSEAS | Seasonality | SPY, TLT, GLD (long-only) | 7.3% | -13.4% | 0.83 | Calendar effects |

### Tier 3: Stock-Level (Harder - Skip if Time Limited)

| Strategy | Type | Markets | CAGR | MDD | Sharpe | Notes |
|----------|------|---------|------|-----|--------|-------|
| B2S2DJ | Mean-reversion | Dow 30 stocks (long-only) | 14.4% | -30.8% | 0.84 | Need DJ components |
| MRDJ | Mean-reversion + breakouts | Dow 30 stocks (long-only) | 10.3% | -22.3% | 0.74 | Need DJ components |

## Methodology

### Phase 1: Hypothesis Formation

Based on strategy names and PAL's documented preferences:

**Mean-Reversion Indicators (PAL favorites):**
- RSI-2, RSI-3 (Connors-style short-term RSI)
- Cumulative RSI (sum of RSI over 2-3 days)
- IBS (Internal Bar Strength) - (Close - Low) / (High - Low)
- Consecutive down days (2-4 days)
- Price deviation from SMA (20, 50, 200)
- Bollinger Band %B position

**Breakout Indicators:**
- New N-day highs/lows (20, 50, 252 days)
- Donchian channel breakouts
- Volatility expansion (ATR ratio)

**Regime Detection (for MRMOM):**
- Momersion indicator (PAL's proprietary - we have it in shared/indicators.py)
- MA crossover regime (price vs SMA-200)
- Volatility regime (short vol vs long vol)

### Phase 2: Parameter Grids

For each strategy type, test these parameter ranges:

**RSI Parameters:**
- Period: [2, 3, 5, 7, 10, 14]
- Entry threshold: [5, 10, 15, 20, 25, 30]
- Exit threshold: [50, 60, 70, 80]

**IBS Parameters:**
- Entry threshold: [0.1, 0.15, 0.2, 0.25, 0.3]
- Exit threshold: [0.5, 0.6, 0.7, 0.8]

**Moving Average Parameters:**
- Short MA: [5, 10, 20, 50]
- Long MA: [50, 100, 200]

**Lookback Windows:**
- Short-term: [5, 10, 20]
- Medium-term: [50, 100]
- Long-term: [200, 252]

### Phase 3: Strategy-Specific Hypotheses

#### ETFMR (SPY, QQQ mean-reversion)
Most likely rules:
1. RSI-2 < 10 on both SPY and QQQ → Buy both
2. RSI-2 < 5 on either → Buy that one
3. Cumulative RSI-2 < 35 → Entry signal
4. IBS < 0.2 combined with RSI conditions
5. Exit: RSI > 50 or after N days

#### MRETF (SPY, QQQ, TLT with breakouts)
Most likely rules:
1. Mean-reversion entries on SPY/QQQ (RSI-based)
2. TLT as risk-off allocation when equity signals weak
3. Breakout component: buy new 20-day high, sell on reversal
4. Possibly rotate between MR and momentum based on regime

#### MRMOM (Regime-switching)
Most likely rules:
1. Use Momersion indicator for regime detection
2. Momersion > 50: Use momentum rules (buy on strength)
3. Momersion < 50: Use mean-reversion rules (buy on weakness)
4. Allocate across SPY, QQQ, TLT, GLD based on regime

#### B2S2ETF / B2S2DJ
"B2S2" likely means "Buy 2 Sell 2" or similar pattern:
1. Buy after 2 consecutive down days
2. Sell after 2 consecutive up days (or fixed holding period)
3. Possibly combined with RSI/IBS filter

### Phase 4: Validation Criteria

A strategy is "successfully reverse-engineered" if:
- Sharpe ratio: within ±0.15 of target
- CAGR: within ±3% of target
- Max drawdown: within ±10% of target

Secondary validation:
- Equity curve shape similarity (visual inspection)
- Trade frequency reasonableness (not over-trading)
- Out-of-sample stability

### Phase 5: Testing Protocol

For each candidate strategy:
1. Split data: 2005-2020 (train), 2021-2026 (test)
2. Optimize on training period
3. Validate on test period
4. Accept if test period metrics remain within bounds

## Available Resources

### Cached Data (in data/samples/)
- SPY, QQQ, TLT, GLD (all needed for Tier 1 & 2)
- IWM, EFA, EEM, VNQ, DBC (additional ETFs)
- Sector ETFs: XLF, XLE, XLK, XLV, XLI, XLY, XLP, XLU, XLB, XLRE
- VIX (for volatility regime)
- FF factors and FRED macro data

### Indicators Library (shared/indicators.py)
- rsi(), cumulative_rsi()
- ibs()
- momersion(), regime_momersion()
- bollinger_bands(), bollinger_pct_b()
- donchian_channel(), keltner_channel()
- atr(), atr_percent()
- consecutive_days()
- rate_of_change()
- williams_r()
- Performance metrics: sharpe_ratio(), max_drawdown(), cagr(), beta()

### Template (tools/_templates/quant-research/)
Copy and adapt for each strategy.

## Constraints

- Use `data/samples/` for cached data (no network fetching needed)
- Long-only strategies only (no shorting)
- Daily rebalancing frequency
- Transaction costs: 10 bps per trade (0.001 per side)
- No leverage unless strategy name suggests it
- Reinvest dividends (data already adjusted)

## Assumptions (use these, don't ask)

- Start with simplest rules that match strategy type
- Use equal-weight position sizing across assets
- Daily close prices for both signals and execution
- Random seed 42 for any stochastic elements
- Assume PAL backtests start around 2005-2010

## Output Requirements

### 1. Create Strategy Implementations

Save to `tools/pal-strategies/`:
```
tools/pal-strategies/
├── __init__.py
├── base.py              # Shared backtest infrastructure
├── etfmr.py             # ETFMR strategy
├── mretf.py             # MRETF strategy
├── mrmom.py             # MRMOM strategy
├── b2s2etf.py           # B2S2ETF strategy
├── etfseas.py           # ETFSEAS strategy (if time)
├── run_all.py           # Run all strategies and compare
└── README.md            # Documentation
```

### 2. Results Summary

Save to `output/{branch}/pal-reverse-engineering/`:
- `results_summary.md` - Overall findings
- `{strategy}_equity.png` - Equity curves for each
- `{strategy}_comparison.md` - Detailed comparison to target

### 3. Performance Tracking

For each strategy, document:
- Final parameter combination
- Metrics achieved vs target
- In-sample vs out-of-sample performance
- Trade statistics (count, win rate, avg trade)
- Assumptions made during reverse-engineering

### 4. Gist Report

Create shareable report with:
- Executive summary of findings
- Which strategies were successfully reverse-engineered
- Best-fit parameters for each
- Equity curve comparisons
- Limitations and caveats

## Execution Order

1. **ETFMR** - Start here (simple 2-asset MR, good Sharpe)
   - Try RSI-2 and cumulative RSI combinations
   - Test IBS combinations
   - Test consecutive down days

2. **B2S2ETF** - Second (single asset, name gives clue)
   - Focus on "2 down, 2 up" patterns
   - Test with RSI filters

3. **MRETF** - Third (adds TLT, may need regime element)
   - Build on ETFMR logic
   - Add TLT as flight-to-quality component

4. **MRMOM** - Fourth (regime-switching, highest Sharpe target)
   - Use Momersion indicator
   - Test regime-based allocation

5. **ETFSEAS** - Last (seasonality, may need calendar analysis)
   - Test month-of-year effects
   - Test turn-of-month effects

## Quality Checks

Before marking a strategy as "reverse-engineered":
- [ ] In-sample metrics within tolerance
- [ ] Out-of-sample metrics reasonable (not >50% degradation)
- [ ] Rules are simple and interpretable
- [ ] No obvious data leakage
- [ ] Transaction costs included
- [ ] Equity curve doesn't show suspiciously perfect behavior

## Example Strategy Code Structure

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "numpy>=1.24", "matplotlib>=3.8"]
# ///
"""
ETFMR Strategy - Reverse Engineered from PAL
Target: CAGR 10%, MDD -22.9%, Sharpe 0.82
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.indicators import rsi, cumulative_rsi, sharpe_ratio, max_drawdown, cagr

def load_data():
    """Load SPY and QQQ from cached samples."""
    pass

def generate_signals(df, rsi_period=2, rsi_threshold=10):
    """Generate entry/exit signals."""
    pass

def backtest(df, cost_bps=10):
    """Run backtest with transaction costs."""
    pass

def optimize(param_grid):
    """Grid search for best parameters."""
    pass

def main():
    # Load data
    # Split into train/test
    # Optimize on train
    # Validate on test
    # Save results
    pass

if __name__ == "__main__":
    main()
```

## Success Metrics

| Outcome | Description |
|---------|-------------|
| Full Success | All 5 strategies within tolerance |
| Partial Success | 3+ strategies within tolerance |
| Limited Success | 1-2 strategies within tolerance |
| Research Value | Document what was learned even if targets not hit |

Focus on learning from the process - understanding WHY certain parameter combinations work provides value even if exact replication fails.
