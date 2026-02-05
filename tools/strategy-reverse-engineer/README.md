# PriceActionLab Strategy Reverse Engineering Tool

Attempts to reverse-engineer trading strategies from [PriceActionLab](https://www.priceactionlab.com/Blog/trading-strategies-for-sale/) based on publicly available performance metrics, strategy descriptions, and published indicator formulas.

## Strategies Analyzed

| Strategy | Type | Markets | Key Finding |
|----------|------|---------|-------------|
| MRETF | Mean-reversion | SPY, QQQ, TLT | RSI(2) with 200-day MA filter |
| MRETFLS | Mean-reversion L/S | SPY, QQQ, TLT | Same as MRETF + short capability |
| B2S2DJ | Mean-reversion | Dow 30 | "Buy 2 Sell 2" consecutive days |
| B2S2ETF | Mean-reversion | SPY | Same as B2S2DJ, single instrument |
| TFDLS | Trend-following | 23 Futures | Donchian breakout system |
| MRMOM | Regime-switching | SPY, QQQ, TLT, GLD | Uses Momersion indicator |
| ETFMR | Mean-reversion | SPY, QQQ | Two-indicator approach |
| ETFSEAS | Seasonality | SPY, TLT, GLD | Calendar effects |
| ETFMO | Seasonality | Leveraged ETF | Calendar effects + leverage |
| MRDJ | Mean-reversion | Dow 30 | MR + breakout hybrid |

## Key Insights

### 1. Momersion Indicator (Discovered Formula)
```
Momersion(n) = 100 × MOMc / (MOMc + MRc)

Where:
- MOMc = count of same-sign consecutive returns (momentum)
- MRc = count of opposite-sign consecutive returns (mean-reversion)
- n = lookback period (typically 250 days)

Interpretation:
- Above 50%: Momentum-dominated regime
- Below 50%: Mean-reversion-dominated regime
```

### 2. RSI(2) Mean-Reversion Pattern
```
Entry: RSI(2) < 10 (oversold)
Exit: Price > 5-day MA
Filter: Price > 200-day MA (optional)
```

### 3. B2S2 Pattern
```
Entry: 2+ consecutive down days
Exit: 2+ consecutive up days
No parameters, no filters
```

### 4. Seasonality Patterns
- End-of-month effect (last 4 trading days)
- Turn-of-month effect (last day + first 3 days)
- Day-of-week patterns (Monday/Friday bias)

## Usage

```bash
uv run tools/strategy-reverse-engineer/main.py
```

## Output

- `outputs/strategy_equity_curves_*.png` - Equity curve comparison chart
- `outputs/strategy_reverse_engineering_*.md` - Detailed analysis report

## Disclaimer

These are educated approximations based on public information. The actual PriceActionLab strategies may differ significantly. This tool is for educational and research purposes only.

## Sources

- [PriceActionLab Trading Strategies](https://www.priceactionlab.com/Blog/trading-strategies-for-sale/)
- [Momersion Indicator](https://www.priceactionlab.com/Blog/2015/08/momersion-indicator/)
- [Larry Connors RSI(2)](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/rsi-2)
- [Mean-Reversion Optimization Space](https://www.priceactionlab.com/Blog/2025/01/the-huge-optimization-space-of-mean-reversion/)
