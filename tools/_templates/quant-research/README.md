# Quant Research Template

Standard template for quantitative trading research. Use this as a starting point for new strategy development.

## Quick Start

```bash
# Run with defaults
uv run tools/_templates/quant-research/main.py

# Run with config file
uv run tools/_templates/quant-research/main.py --config tools/_templates/quant-research/config.yaml

# Override via CLI
uv run tools/_templates/quant-research/main.py --symbol QQQ --rsi-period 3 --rsi-threshold 15
```

## Creating a New Strategy

1. Copy this template:
   ```bash
   cp -r tools/_templates/quant-research tools/my-strategy
   ```

2. Edit `config.yaml` with your parameters

3. Modify `generate_signals()` in `main.py` with your strategy logic

4. Run and iterate

## Template Structure

```
tools/_templates/quant-research/
├── main.py        # Entry point with standard sections
├── config.yaml    # All parameters in one place
└── README.md      # This file
```

## Key Functions to Modify

### `generate_signals(df, **params) -> pd.DataFrame`

This is where your strategy logic lives. The default implements a simple RSI + IBS mean-reversion strategy.

**Input:** DataFrame with OHLCV columns
**Output:** DataFrame with `signal` and `position` columns

### `calculate_metrics(df) -> dict`

Add any custom metrics you want to track.

## Available Indicators

From `shared/indicators.py`:

**Oscillators:**
- `rsi(prices, period)` - Relative Strength Index
- `cumulative_rsi(prices, rsi_period, cum_period)` - Sum of RSI
- `ibs(high, low, close)` - Internal Bar Strength
- `williams_r(high, low, close, period)` - Williams %R

**Volatility:**
- `atr(high, low, close, period)` - Average True Range
- `atr_percent(high, low, close, period)` - ATR as % of price
- `bollinger_bands(prices, period, num_std)` - Returns (middle, upper, lower)
- `bollinger_pct_b(prices, period, num_std)` - Position within bands

**Trend/Momentum:**
- `momersion(returns, lookback)` - Momentum vs mean-reversion regime
- `rate_of_change(prices, period)` - Simple momentum
- `consecutive_days(returns, direction)` - Count up/down streaks

**Channels:**
- `donchian_channel(high, low, period)` - Returns (upper, lower)
- `keltner_channel(high, low, close, ema_period, atr_period, atr_mult)` - Returns (middle, upper, lower)

**Regime Detection:**
- `regime_momersion(returns, lookback, threshold)` - 1=momentum, 0=mean-reversion
- `regime_volatility(prices, short_period, long_period)` - 1=high vol, 0=low vol

**Performance Metrics:**
- `sharpe_ratio(returns, rf, annualize)` - Risk-adjusted return
- `max_drawdown(returns)` - Maximum peak-to-trough decline
- `cagr(returns)` - Compound Annual Growth Rate
- `beta(returns, benchmark)` - Market sensitivity

## Cached Data

Available in `data/samples/`:
- SPY (S&P 500 ETF)
- QQQ (Nasdaq 100 ETF)
- TLT (20+ Year Treasury ETF)
- GLD (Gold ETF)
- BTC-USD (Bitcoin)
- WIG20 (Polish stock index)

## Output

Results are saved to `output/`:
- `quant_research_{symbol}_{timestamp}.png` - Equity curves chart
- `quant_research_{symbol}_{timestamp}.md` - Full report with metrics

## Best Practices

1. **Use cached data** from `data/samples/` when possible
2. **Set random seeds** for reproducibility
3. **Plan parameter ranges** before starting optimization
4. **Use `/qrd`** to create structured specifications
5. **Check `/context7`** for library documentation
