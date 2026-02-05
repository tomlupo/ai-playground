# Quant Research Data Samples

Pre-downloaded market data covering ~90% of typical quant investing research needs.

**Period**: 2005-01-03 to 2026-02-04 (~21 years, includes 2008 and 2020 crises)

## Data Files

### Tier 1: Essential Assets (~70% coverage)

| File | Description | Rows |
|------|-------------|------|
| `SPY.csv` | S&P 500 ETF (large cap US) | 5,306 |
| `QQQ.csv` | Nasdaq 100 ETF (tech-heavy) | 5,306 |
| `TLT.csv` | 20+ Year Treasury ETF | 5,306 |
| `IEF.csv` | 7-10 Year Treasury ETF | 5,306 |
| `SHY.csv` | 1-3 Year Treasury ETF | 5,306 |
| `GLD.csv` | Gold ETF | 5,306 |
| `VIX.csv` | CBOE Volatility Index | 5,306 |

### Tier 2: Diversification (~15% more)

| File | Description | Rows | Notes |
|------|-------------|------|-------|
| `IWM.csv` | Russell 2000 ETF (small cap) | 5,306 | |
| `EFA.csv` | Developed Markets ex-US | 5,306 | |
| `EEM.csv` | Emerging Markets | 5,306 | |
| `VNQ.csv` | Real Estate ETF | 5,306 | |
| `DBC.csv` | Commodities ETF | 5,031 | From 2006 |
| `BTC_USD.csv` | Bitcoin | 4,159 | From 2014 |

### Tier 3: Sector ETFs

| File | Sector | Rows |
|------|--------|------|
| `XLF.csv` | Financials | 5,306 |
| `XLE.csv` | Energy | 5,306 |
| `XLK.csv` | Technology | 5,306 |
| `XLV.csv` | Healthcare | 5,306 |
| `XLI.csv` | Industrials | 5,306 |
| `XLY.csv` | Consumer Discretionary | 5,306 |
| `XLP.csv` | Consumer Staples | 5,306 |
| `XLU.csv` | Utilities | 5,306 |
| `XLB.csv` | Materials | 5,306 |
| `XLRE.csv` | Real Estate | 2,596 | From 2015 |

### Factor Data

| File | Description | Rows |
|------|-------------|------|
| `FF5_factors.csv` | Fama-French 5 Factors (daily) | 15,731 |
| `FF_momentum.csv` | Momentum Factor (daily) | 26,050 |

Fama-French factors:
- `Mkt-RF`: Market excess return
- `SMB`: Small minus Big (size)
- `HML`: High minus Low (value)
- `RMW`: Robust minus Weak (profitability)
- `CMA`: Conservative minus Aggressive (investment)
- `RF`: Risk-free rate
- `Mom`: Momentum factor

### Economic Indicators (FRED)

| File | Description | Frequency |
|------|-------------|-----------|
| `FRED_GDP.csv` | Gross Domestic Product | Quarterly |
| `FRED_CPIAUCSL.csv` | Consumer Price Index | Monthly |
| `FRED_UNRATE.csv` | Unemployment Rate | Monthly |
| `FRED_FEDFUNDS.csv` | Federal Funds Rate | Monthly |
| `FRED_T10Y2Y.csv` | 10Y-2Y Treasury Spread | Daily |
| `FRED_DGS10.csv` | 10-Year Treasury Yield | Daily |
| `FRED_DGS2.csv` | 2-Year Treasury Yield | Daily |

## Column Format

### Price Data (ETFs, Stocks)
```
Date, Close, High, Low, Open, Volume
```

### Fama-French Factors
```
Date, Mkt-RF, SMB, HML, RMW, CMA, RF
```
Values are in decimal form (0.01 = 1%)

### FRED Data
```
Date, Value
```

## Usage

```python
import pandas as pd

# Load price data
spy = pd.read_csv('data/samples/SPY.csv', parse_dates=['Date'])

# Load factors
ff5 = pd.read_csv('data/samples/FF5_factors.csv', parse_dates=['Date'])

# Merge on date
merged = spy.merge(ff5, on='Date')
```

## Refresh Data

```bash
# Re-download all samples
uv run scripts/download_quant_samples.py --start 2005-01-01

# Update factors and FRED only
uv run scripts/download_factors_fred.py
```

## Data Sources

- **Price Data**: Yahoo Finance via yfinance
- **Fama-French**: Kenneth French Data Library
- **FRED**: Federal Reserve Economic Data
