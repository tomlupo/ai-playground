# Kenneth French Data Library

Source of Fama-French factor data and portfolio returns for academic research.

**Website**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

## Overview

Kenneth French's Data Library provides free access to:
- Fama-French factor returns (3-factor, 5-factor, momentum)
- Industry portfolio returns (5, 10, 17, 30, 49 industries)
- Size and value sorted portfolios
- Various other portfolio constructions

Data is updated monthly and includes:
- Daily returns from 1926 to present
- Monthly returns from 1926 to present

## Authentication

**None required** - All data is freely available.

## Rate Limits

No documented rate limits, but requests should be reasonable (1-2 second delay between requests).

## Data Format

### Raw Format

Data is provided as ZIP files containing CSV:
- Values are expressed as **percentages** (e.g., 1.23 means 1.23%)
- Dates are YYYYMMDD (daily) or YYYYMM (monthly)
- Files may contain multiple tables (we extract the first/primary table)

### Standardized Format (after fetching)

All values are converted to **decimals** (divide by 100):
- 0.0123 represents 1.23% return
- Consistent with other data sources in the skill

## Available Datasets

### Factor Datasets

| Dataset ID | Description | Columns |
|------------|-------------|---------|
| `FF5` | Fama-French 5 Factors (monthly) | Mkt-RF, SMB, HML, RMW, CMA, RF |
| `FF5_daily` | Fama-French 5 Factors (daily) | Mkt-RF, SMB, HML, RMW, CMA, RF |
| `FF3` | Fama-French 3 Factors (monthly) | Mkt-RF, SMB, HML, RF |
| `FF3_daily` | Fama-French 3 Factors (daily) | Mkt-RF, SMB, HML, RF |
| `MOM` | Momentum Factor (monthly) | Mom |
| `MOM_daily` | Momentum Factor (daily) | Mom |

### Industry Portfolios

| Dataset ID | Description |
|------------|-------------|
| `IND5` / `IND5_daily` | 5 Industry Portfolios |
| `IND10` / `IND10_daily` | 10 Industry Portfolios |
| `IND17` / `IND17_daily` | 17 Industry Portfolios |
| `IND30` / `IND30_daily` | 30 Industry Portfolios |
| `IND49` / `IND49_daily` | 49 Industry Portfolios |

### Size & Value Portfolios

| Dataset ID | Description |
|------------|-------------|
| `SIZE` / `SIZE_daily` | Portfolios formed on market equity |
| `VALUE` / `VALUE_daily` | Portfolios formed on book-to-market |
| `SIZE_VALUE` / `SIZE_VALUE_daily` | 6 portfolios (2x3 size/value) |
| `SIZE_MOM` / `SIZE_MOM_daily` | 25 portfolios (5x5 size/momentum) |

## Factor Definitions

### Fama-French 5 Factors

| Factor | Name | Definition |
|--------|------|------------|
| **Mkt-RF** | Market Risk Premium | Market return minus risk-free rate |
| **SMB** | Small Minus Big | Small cap minus large cap returns |
| **HML** | High Minus Low | High B/M minus low B/M returns |
| **RMW** | Robust Minus Weak | Robust profitability minus weak profitability |
| **CMA** | Conservative Minus Aggressive | Conservative investment minus aggressive investment |
| **RF** | Risk-Free Rate | Treasury bill rate (daily/monthly) |

### Momentum Factor

| Factor | Name | Definition |
|--------|------|------------|
| **Mom** / **UMD** / **WML** | Momentum | Winners minus losers (12-month return, skip 1 month) |

## Usage Examples

### CLI Usage

```bash
# List available datasets
uv run fetch_kenneth_french.py --list

# Fetch 5-factor daily data
uv run fetch_kenneth_french.py FF5_daily 2020-01-01 2024-12-31

# Fetch momentum factor
uv run fetch_kenneth_french.py MOM_daily 2020-01-01

# Fetch industry portfolios
uv run fetch_kenneth_french.py IND10_daily 2024-01-01
```

### Python Usage

```python
from fetch_kenneth_french import fetch_kenneth_french, KennethFrenchFetcher

# Fetch full 5-factor dataset
df = fetch_kenneth_french('FF5_daily', start_date='2020-01-01', end_date='2024-12-31')
print(df.head())
#          Date    Mkt-RF       SMB       HML       RMW       CMA        RF
# 0  2020-01-02  0.008563  0.002091  0.003652 -0.001247 -0.004199  0.000006
# 1  2020-01-03 -0.006831 -0.002138 -0.002509  0.000851 -0.001138  0.000006

# Fetch single factor
df = fetch_kenneth_french('SMB', start_date='2024-01-01')  # Routes to FF5
print(df.head())
#          Date       SMB
# 0  2024-01-02  0.002091
# 1  2024-01-03 -0.002138

# Via unified fetcher (auto-routes)
from fetch_unified import fetch_market_data
df = fetch_market_data('FF5_daily', start_date='2024-01-01')
df = fetch_market_data('MOM_daily', start_date='2024-01-01')
```

### Factor Symbol Routing

These symbols automatically route to Kenneth French via the unified fetcher:

```python
# All route to FF5_daily and extract the specific column
df = fetch_market_data('MKT-RF', start_date='2024-01-01')
df = fetch_market_data('SMB', start_date='2024-01-01')
df = fetch_market_data('HML', start_date='2024-01-01')
df = fetch_market_data('RMW', start_date='2024-01-01')
df = fetch_market_data('CMA', start_date='2024-01-01')
df = fetch_market_data('RF', start_date='2024-01-01')
df = fetch_market_data('MOM', start_date='2024-01-01')
```

## Common Use Cases

### 1. Factor-Based Portfolio Analysis

```python
import pandas as pd
from fetch_kenneth_french import fetch_kenneth_french

# Get factors and portfolio returns
factors = fetch_kenneth_french('FF5_daily', '2020-01-01', '2024-12-31')

# Run factor regression
# portfolio_returns = ... (your portfolio)
# factors_X = factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
# model = LinearRegression().fit(factors_X, portfolio_returns)
```

### 2. Risk Premium Analysis

```python
# Analyze market risk premium over time
factors = fetch_kenneth_french('FF3', '1990-01-01')  # Monthly data

# Annualized market premium
annual_premium = factors['Mkt-RF'].mean() * 12
print(f"Annualized market premium: {annual_premium:.2%}")
```

### 3. Industry Rotation Strategy

```python
# Get industry returns for sector rotation
industries = fetch_kenneth_french('IND10_daily', '2024-01-01')

# Calculate momentum for each industry
momentum = industries.set_index('Date').rolling(21).sum()
```

## Caching

- **Default cache duration**: 1 week (168 hours)
- **Reason**: Data updates monthly
- **Cache location**: `data/cache/market_data/kenneth_french/`

## Limitations

1. **US data only** - No international factor data
2. **Monthly updates** - Not suitable for real-time analysis
3. **Historical only** - No forward-looking data
4. **Academic focus** - May not align with commercial factor definitions

## References

- Fama, E.F. and French, K.R. (1993). "Common risk factors in the returns on stocks and bonds"
- Fama, E.F. and French, K.R. (2015). "A five-factor asset pricing model"
- Carhart, M.M. (1997). "On Persistence in Mutual Fund Performance" (Momentum factor)

## See Also

- `sources_overview.md` - Comparison with other data sources
- `fetch_kenneth_french.py` - Source code
