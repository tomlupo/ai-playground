# Task

 Generate an autonomous prompt for reverse engineering trading strategies from Price Action Lab.

 Output

 The prompt below can be saved to a file and run with:
 claude -p "$(cat docs/prompts/reverse-engineer-pal-strategies.md)" --dangerously-skip-permissions

 ---
 Autonomous Prompt

 ## Execution Mode: AUTONOMOUS

 ## Task
 Reverse engineer trading strategies from Price Action Lab based on their published performance metrics. Create
 implementable Python backtests that attempt to replicate their reported results.

 ## Target Strategies (prioritized by feasibility)

 ### Tier 1: ETF Mean-Reversion (best data availability)
 | Strategy | Type | Markets | CAGR | MDD | Sharpe |
 |----------|------|---------|------|-----|--------|
 | MRETF | Mean-reversion/breakouts | SPY, QQQ, TLT (long-only) | 5.2% | -9.3% | 0.78 |
 | ETFMR | Mean-reversion | SPY, QQQ (long-only) | 10.0% | -22.9% | 0.82 |
 | B2S2ETF | Mean-reversion | SPY (long-only) | 9.1% | -30.6% | 0.63 |

 ### Tier 2: Multi-Asset/Regime (more complex)
 | Strategy | Type | Markets | CAGR | MDD | Sharpe |
 |----------|------|---------|------|-----|--------|
 | MRMOM | Regime switching | SPY, QQQ, TLT, GLD (long-only) | 10.3% | -16.8% | 1.15 |
 | ETFSEAS | Seasonality | SPY, TLT, GLD (long-only) | 7.3% | -13.4% | 0.83 |

 ### Tier 3: Stock-level (harder to replicate)
 | Strategy | Type | Markets | CAGR | MDD | Sharpe |
 |----------|------|---------|------|-----|--------|
 | B2S2DJ | Mean-reversion | Dow 30 stocks (long-only) | 14.4% | -30.8% | 0.84 |
 | MRDJ | Mean-reversion/breakouts | Dow 30 stocks (long-only) | 10.3% | -22.3% | 0.74 |

 ## Methodology

 For each strategy, follow this process:

 ### Phase 1: Hypothesis Formation
 Based on the strategy name and performance characteristics, form hypotheses about the likely rules:
 - "Mean-reversion" → RSI, Bollinger Bands, price deviation from MA
 - "Breakouts" → Price making new highs/lows, volatility expansion
 - "Regime switching" → Trend detection (MA crossovers, ADX), volatility regimes
 - "Seasonality" → Month-of-year effects, turn-of-month, holiday effects

 ### Phase 2: Parameter Grid Search
 Test parameter combinations that could produce the target metrics:
 - RSI periods: 2, 3, 5, 7, 10, 14
 - MA periods: 5, 10, 20, 50, 100, 200
 - Lookback windows: 5, 10, 20, 50, 100, 200 days
 - Entry/exit thresholds: Various percentages

 ### Phase 3: Validation
 For each candidate strategy:
 1. Run backtest from strategy start year to 2025
 2. Compare CAGR, MDD, Sharpe to targets
 3. Check correlation and beta to benchmark (SPY)
 4. Accept if metrics are within ±20% of targets

 ### Phase 4: Documentation
 For successful reverse-engineered strategies:
 1. Document exact rules (entry, exit, position sizing)
 2. Save backtest code to `tools/pal-strategies/`
 3. Generate performance tearsheet
 4. Note assumptions and limitations

 ## Constraints

 - Use `data/samples/` for cached data where available
 - Fetch additional data via market-data-fetcher skill if needed
 - Long-only strategies only (no shorting unless explicitly stated)
 - Daily rebalancing frequency
 - 10 bps transaction costs assumption
 - No leverage unless strategy name suggests it (ETFMO)

 ## Success Criteria

 A strategy is "successfully reverse-engineered" if:
 - Sharpe ratio within ±0.15 of target
 - CAGR within ±3% of target
 - Max drawdown within ±10% of target
 - Correlation to benchmark within ±0.2 of target

 ## Assumptions (use these, don't ask)

 - Start with simplest possible rules that match the strategy type
 - Use standard technical indicators (RSI-2, RSI-14, SMA-20/50/200)
 - Assume equal-weight position sizing unless metrics suggest otherwise
 - Daily close prices for signals and execution
 - Reinvest dividends
 - No slippage beyond the 10 bps transaction cost assumption

 ## Output Requirements

 1. Save all code to `tools/pal-strategies/`
 2. Save results summary to `output/{branch}/pal-strategies-results.md`
 3. For each successfully reverse-engineered strategy:
    - Python implementation in `tools/pal-strategies/{strategy_name}.py`
    - Performance comparison table
    - Equity curve chart saved to `output/{branch}/`
 4. Create gist-report with findings

 ## Priority Order

 Start with ETFMR (simplest: just SPY and QQQ, mean-reversion, good Sharpe).
 Then MRETF (adds TLT, still mean-reversion).
 Then MRMOM (regime switching, higher complexity but best Sharpe).