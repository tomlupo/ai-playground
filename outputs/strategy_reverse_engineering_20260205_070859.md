# PriceActionLab Strategy Reverse Engineering Report
Generated: 2026-02-05 07:08:59

## Executive Summary

This report attempts to reverse-engineer trading strategies from PriceActionLab
based on publicly available information:

- Performance metrics (CAGR, MDD, Sharpe, correlation, beta, alpha)
- Strategy descriptions (mean-reversion, trend-following, regime-switching)
- Published indicator formulas (Momersion indicator)
- Academic literature on similar approaches (Connors RSI, calendar effects)

**Disclaimer**: These are approximations based on inference. The actual strategies
may differ significantly from these reconstructions.

## Target vs Achieved Metrics

| Strategy | Target CAGR | Achieved CAGR | Target MDD | Achieved MDD | Target Sharpe | Achieved Sharpe |
|----------|-------------|---------------|------------|--------------|---------------|------------------|
| MRETF | 5.2% | 4.0% | -9.3% | -14.7% | 0.78 | 0.31 |
| MRMOM | 10.3% | 4.8% | -16.8% | -16.4% | 1.15 | 0.36 |
| B2S2ETF | 9.1% | 7.7% | -30.6% | -35.8% | 0.63 | 0.39 |
| ETFSEAS | 7.3% | 8.6% | -13.4% | -25.2% | 0.83 | 0.48 |
| ETFMR | 10.0% | 3.6% | -22.9% | -15.5% | 0.82 | 0.20 |
| TFDLS | 14.1% | -0.6% | -32.2% | -36.1% | 0.82 | -0.24 |

## Strategy Descriptions


### MRETF (Mean-Reversion ETF - Long Only)
**Target Markets**: SPY, QQQ, TLT

**Inferred Logic**:
- Uses RSI(2) or similar short-term mean-reversion indicator
- Buys when RSI falls below oversold threshold (e.g., 10)
- Exits when price crosses above short-term MA (e.g., 5-day)
- Likely includes 200-day MA trend filter
- Single position at a time, rotates between ETFs

**Key Characteristics**:
- Very low beta (0.14) suggests highly selective entry
- Low exposure, high win rate (~69%)
- Short holding periods (~7 days)


### MRETFLS (Mean-Reversion ETF - Long/Short)
**Target Markets**: SPY, QQQ, TLT

**Inferred Logic**:
- Same base logic as MRETF but allows short positions
- Short when RSI reaches overbought levels
- Likely uses different thresholds for long vs short

**Key Characteristics**:
- Near-zero correlation and beta suggests market-neutral tendency
- Higher volatility than long-only version


### B2S2DJ (Buy 2 Sell 2 - Dow Jones)
**Target Markets**: Dow 30 stocks

**Inferred Logic**:
- "No parameters, no filters" suggests extremely simple rules
- Name suggests "Buy after 2 down days, Sell after 2 up days"
- Ranks stocks by rate-of-change for position selection
- Maximum 10 concurrent positions
- Equal-weight allocation

**Key Characteristics**:
- High correlation (0.84) with market
- High beta (0.78) indicates leveraged market exposure
- Long backtest period (1993) suggests robust approach


### B2S2ETF (Buy 2 Sell 2 - SPY ETF)
**Target Markets**: SPY only

**Inferred Logic**:
- Same "Buy 2 Sell 2" approach as B2S2DJ
- Applied only to SPY ETF
- Simpler implementation, single instrument

**Key Characteristics**:
- Similar metrics to B2S2DJ but single instrument
- Higher exposure than stock version


### TFDLS (Trend Following - Long/Short)
**Target Markets**: 23 Futures contracts

**Inferred Logic**:
- Classic channel breakout system (Donchian-style)
- Enter long on new N-day high, short on new N-day low
- Uses ATR-based position sizing and stops
- Trades all 23 markets with identical parameters

**Key Characteristics**:
- Negative correlation (-0.07) with SPY - crisis alpha
- Requires significant capital (~$1M for proper sizing)
- Wide diversification across asset classes


### MRMOM (Mean-Reversion/Momentum Regime Switching)
**Target Markets**: SPY, QQQ, TLT, GLD

**Inferred Logic**:
- Uses Momersion indicator for regime detection
- Momersion(250) = 100 * MOMc / (MOMc + MRc)
- Below 50: Mean-reversion regime - buy oversold
- Above 50: Momentum regime - follow trends
- Rotates between 4 ETFs with equal weight

**Key Characteristics**:
- Highest Sharpe ratio (1.15) of all strategies
- Moderate correlation (0.67) suggests adaptive behavior
- Combines best of both approaches


### ETFMR (ETF Mean-Reversion)
**Target Markets**: SPY, QQQ

**Inferred Logic**:
- "Based on two popular indicators" - likely RSI + something else
- Could be RSI(2) + Williams %R, or RSI(2) + Bollinger Bands
- Similar to MRETF but different parameter set
- 100% position sizing (fully invested when signal triggers)

**Key Characteristics**:
- Higher CAGR (10%) than MRETF
- Higher drawdown (-22.9%) suggests more aggressive
- ~4.3 day average holding period


### ETFSEAS (ETF Seasonality)
**Target Markets**: SPY, TLT, GLD

**Inferred Logic**:
- Calendar-based effects (day-of-week, end-of-month)
- Likely exploits:
  - Turn-of-month effect (last day to first 3 days)
  - End-of-month effect (last 4 trading days)
  - Possibly day-of-week patterns
- Very short holding period (~1 day)

**Key Characteristics**:
- Low correlation (0.22) - uncorrelated to market
- Very low beta (0.10) - minimal market exposure
- Low exposure (28.4%) - only trades on specific days


### ETFMO (ETF Momentum - Leveraged)
**Target Markets**: Leveraged ETF (unspecified)

**Inferred Logic**:
- Seasonality strategy applied to leveraged ETF
- Likely trades TQQQ, UPRO, or TMF
- ~3-day holding period
- Very selective entry (13.6% exposure)

**Key Characteristics**:
- Negative beta (-0.07) suggests non-equity focus
- Could be trading leveraged bond ETF (TMF)
- High alpha (7.8%) indicates unique edge


### MRDJ (Mean-Reversion Dow Jones)
**Target Markets**: Dow 30 stocks

**Inferred Logic**:
- Similar to B2S2DJ with possible breakout component
- "Mean-reversion/breakouts" description
- Could combine dip-buying with breakout entries
- Stock selection based on lowest ROC

**Key Characteristics**:
- Moderate correlation (0.52) vs B2S2DJ's 0.84
- Lower beta (0.39) suggests more selective
- May have trend filter that B2S2 lacks


## Implementation Notes


### Data Requirements
- **ETF Strategies**: Yahoo Finance data sufficient
- **Stock Strategies (B2S2DJ, MRDJ)**: Requires Norgate Data for delisted stocks
- **Futures Strategies (TFDLS)**: Requires continuous back-adjusted contracts

### Key Indicators Used
1. **RSI(2)**: Short-term oversold/overbought detection
2. **Momersion**: Regime detection (momentum vs mean-reversion)
3. **Rate of Change**: Stock ranking for selection
4. **Donchian Channels**: Trend-following breakouts
5. **ATR**: Position sizing and stops

### Risk Warnings
- Mean-reversion strategies typically don't use stops
- High win rate compensates for low payoff ratio
- Regime changes can cause strategy failure
- Transaction costs significantly impact high-turnover strategies

## Sources


- [PriceActionLab Strategies](https://www.priceactionlab.com/Blog/trading-strategies-for-sale/)
- [Momersion Indicator](https://www.priceactionlab.com/Blog/2015/08/momersion-indicator/)
- [Mean-Reversion Optimization Space](https://www.priceactionlab.com/Blog/2025/01/the-huge-optimization-space-of-mean-reversion/)
- [MRMOM Regime Switching](https://www.priceactionlab.com/Blog/2024/01/mean-reversion-and-momentum-regime-switching/)
- [Larry Connors RSI(2)](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/rsi-2)
