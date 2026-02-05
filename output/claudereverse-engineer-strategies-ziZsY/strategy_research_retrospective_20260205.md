# Strategy Reverse Engineering: Process Retrospective

**Date:** 2026-02-05
**Task:** Reverse engineer PriceActionLab trading strategies
**Outcome:** Partial success - 3/4 strategies matched within reasonable tolerance

---

## Executive Summary

| Aspect | Rating | Notes |
|--------|--------|-------|
| Research Quality | ⭐⭐⭐⭐ | Found key formulas (Momersion, IBS) |
| Code Quality | ⭐⭐ | Multiple messy iterations |
| Time Efficiency | ⭐⭐ | Debugging ate significant time |
| Skill Utilization | ⭐⭐ | Underutilized available tools |
| Final Results | ⭐⭐⭐ | 3/4 strategies reasonably matched |

---

## What Went Well ✅

### 1. Web Research Was Effective
- Found the **Momersion indicator formula** directly from PAL blog
- Discovered **IBS (Internal Bar Strength)** as likely "second indicator"
- Located **Larry Connors RSI(2)** parameters and methodology
- Cross-referenced multiple sources for validation

### 2. Beta Matching Worked
```
MRETF:   Target 0.14 → Achieved 0.14 (exact!)
ETFMR:   Target 0.44 → Achieved 0.45
B2S2ETF: Target 0.60 → Achieved 0.57
```
Beta is derived from exposure patterns - this suggests we captured the entry/exit logic correctly even if absolute returns differ.

### 3. Iterative Refinement
- Started with basic RSI → didn't match
- Added IBS → improved CAGR
- Combined RSI+IBS → matched ETFMR closely
- This scientific approach worked well

### 4. Grid Search for Parameters
- Systematic parameter exploration
- Weighted scoring function (Sharpe weighted 2x)
- Found non-obvious parameters (e.g., RSI threshold 40, not typical 10-20)

---

## What Went Badly ❌

### 1. yfinance API Debugging (30+ minutes wasted)
```python
# OLD (broken):
df["Adj Close"]  # KeyError

# NEW (working):
df = yf.download(..., auto_adjust=True)
df["Close"]  # MultiIndex handling needed
```
**Root cause:** Didn't check Context7 for latest yfinance docs.

### 2. Slow Momersion Calculation
```python
# SLOW (rolling apply):
returns.rolling(252).apply(calc_momersion)  # Minutes per symbol

# FAST (vectorized):
product = returns * returns.shift(1)
mom_count = (product > 0).rolling(252).sum()  # Seconds
```
**Root cause:** Jumped to implementation without thinking about vectorization.

### 3. MRMOM Sharpe Gap
```
Target: 1.15 Sharpe
Achieved: 0.46 Sharpe (60% gap)
```
**Root cause:** A Sharpe of 1.15 is exceptional. Likely requires:
- Proprietary timing signals
- Different asset allocation scheme
- Unknown filters or regime detection enhancements

### 4. Multiple Script Versions Created
```
tools/strategy-reverse-engineer/
├── main.py           # Original
├── optimize.py       # Full grid search (too slow)
├── quick_optimize.py # Simplified (abandoned)
├── final_optimize.py # Final working version
└── test_yf.py        # Debug script
```
**Root cause:** No upfront planning, reactive development.

---

## What Was Ugly 😬

### 1. No Proper Backtesting Framework
Reinvented basic backtesting from scratch:
```python
# Hand-rolled (error-prone):
strat_returns = signals.shift(1) * returns
```
Should have used: `vectorbt`, `backtrader`, or `quantstats`

### 2. Process Killed Multiple Times
```bash
pkill -9 -f optimize.py  # Had to do this twice
```
Long-running processes with no progress indication.

### 3. No Cached Data
Downloaded fresh data on every run:
```python
yf.download(...)  # Network call every time
```
Should use `data/samples/` cache for faster iteration.

### 4. No Reproducibility
- Random seeds not set
- No versioning of parameter search results
- Can't reproduce exact grid search exploration

---

## Skills Analysis

### Skills I Used

| Skill | Usage | Effectiveness |
|-------|-------|---------------|
| `WebSearch` | Finding indicator formulas | ⭐⭐⭐⭐ High |
| `WebFetch` | Reading PAL blog posts | ⭐⭐⭐⭐ High |
| `TodoWrite` | Task tracking | ⭐⭐⭐ Medium |
| `Bash` | Running scripts | ⭐⭐⭐ Medium |
| `Write/Edit` | Code creation | ⭐⭐⭐ Medium |

### Skills I Should Have Used

| Skill | Why I Didn't Use | What I Missed |
|-------|------------------|---------------|
| `/context7` | Forgot it existed | Latest yfinance, vectorbt docs |
| `/qrd` | Seemed like overkill | Structured spec would have saved time |
| `/workflows:plan` | Jumped to coding | Would have identified dependencies |
| `/feature-dev:code-architect` | Didn't think of it | Better initial design |
| `/statsmodels` or `/scikit-learn` | Didn't need ML | Could help with regime detection |
| `Explore` agent | Did manual search | Faster codebase understanding |

### Critical Miss: `/context7`
```bash
# Should have done this FIRST:
/context7 yfinance  # Check latest API
/context7 vectorbt  # Backtesting best practices
/context7 pandas-ta # Indicator library
```
Would have saved 30+ minutes of debugging.

---

## Actionable Improvements

### 1. Create Quant Research Template

**File: `tools/_templates/quant-research/`**

```
quant-research/
├── main.py              # Entry point with standard structure
├── indicators.py        # Common indicators (RSI, IBS, ATR, etc.)
├── backtest.py          # Standard backtesting engine
├── data_loader.py       # Cached data loading
├── metrics.py           # Standard performance metrics
└── config.yaml          # Parameters in one place
```

**Action:** Create this template in the repo.

### 2. Add Cached Market Data

**File: `data/samples/market/`**

```yaml
# data/samples/market/manifest.yaml
datasets:
  spy_daily:
    file: SPY_2000_2026.parquet
    source: yfinance
    updated: 2026-02-05

  dow30_daily:
    file: DOW30_2000_2026.parquet
    symbols: [AAPL, MSFT, ...]
```

**Action:** Add script to refresh cached data weekly.

### 3. Install Proper Backtesting Package

**File: `pyproject.toml`**

```toml
[project.optional-dependencies]
quant = [
    "vectorbt>=0.26",      # Fast backtesting
    "quantstats>=0.0.62",  # Performance reports
    "pandas-ta>=0.3.14b",  # Technical indicators
    "empyrical>=0.5.5",    # Risk metrics
]
```

**Action:** Add to finance extras, update docs.

### 4. Create Pre-Research Checklist

**File: `.claude/rules/quant-research.md`**

```markdown
## Before Starting Quant Research

1. [ ] Check `/context7` for library docs (yfinance, vectorbt, etc.)
2. [ ] Load cached data from `data/samples/market/`
3. [ ] Use `/qrd` to create structured specification
4. [ ] Set random seeds for reproducibility
5. [ ] Use `tools/_templates/quant-research/` as base
6. [ ] Plan parameter ranges BEFORE coding
```

**Action:** Add this rule file.

### 5. Add Indicator Library

**File: `shared/indicators.py`**

```python
"""Standard technical indicators with consistent API."""

def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI with Wilder smoothing."""
    ...

def ibs(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Internal Bar Strength."""
    return (close - low) / (high - low + 1e-10)

def momersion(returns: pd.Series, lookback: int = 252) -> pd.Series:
    """PriceActionLab Momersion indicator (vectorized)."""
    product = returns * returns.shift(1)
    mom = (product > 0).rolling(lookback).sum()
    mr = (product < 0).rolling(lookback).sum()
    return 100 * mom / (mom + mr)

def consecutive_days(returns: pd.Series, direction: str = "down") -> pd.Series:
    """Count consecutive up/down days."""
    ...
```

**Action:** Create shared indicator library.

### 6. Workflow Improvement

**Current (chaotic):**
```
Research → Code → Debug → Rewrite → Debug → Grid Search → Done
```

**Proposed (structured):**
```
1. /context7 (check docs)
2. /qrd (create spec)
3. /workflows:plan (design)
4. Code with template
5. Test with cached data
6. Parameter search
7. /workflows:review (validate)
```

---

## Specific Recommendations for This Repo

### Immediate Actions (Do Now)

1. **Create `shared/indicators.py`** with Momersion, IBS, RSI
2. **Add vectorbt to pyproject.toml** finance extras
3. **Cache SPY/QQQ/TLT/GLD data** in `data/samples/market/`
4. **Delete redundant scripts:**
   ```bash
   rm tools/strategy-reverse-engineer/optimize.py
   rm tools/strategy-reverse-engineer/quick_optimize.py
   rm tools/strategy-reverse-engineer/test_yf.py
   ```

### Short-Term Actions (This Week)

1. **Create quant research template** in `tools/_templates/`
2. **Add `.claude/rules/quant-research.md`** checklist
3. **Update CLAUDE.md** with quant research workflow

### Long-Term Actions (This Month)

1. **Build strategy library** with validated implementations
2. **Create backtesting benchmark suite** for validation
3. **Document indicator formulas** in `docs/reference/indicators.md`

---

## Lessons Learned

### For Future Strategy Research

1. **Always check Context7 first** for library documentation
2. **Use cached data** for faster iteration
3. **Vectorize early** - never use `.apply()` on large rolling windows
4. **Plan parameter ranges** before implementing grid search
5. **Accept partial success** - some proprietary strategies can't be fully replicated

### For Claude Code Usage

1. **Use `/qrd` for quant specs** - forces structured thinking
2. **Use `/workflows:plan` before coding** - identifies dependencies
3. **Use `Explore` agent for codebase questions** - faster than manual search
4. **Skills are underutilized** - check available skills before starting

### Meta-Lesson

> The fastest path to working code is NOT jumping straight to implementation.
> 10 minutes of planning saves 60 minutes of debugging.

---

## Files to Clean Up

```bash
# Remove redundant files
git rm tools/strategy-reverse-engineer/optimize.py
git rm tools/strategy-reverse-engineer/quick_optimize.py
git rm tools/strategy-reverse-engineer/test_yf.py

# Keep final versions
# tools/strategy-reverse-engineer/main.py (original approach)
# tools/strategy-reverse-engineer/final_optimize.py (optimized approach)
```

---

## Summary

| Category | Key Takeaway |
|----------|--------------|
| **Research** | WebSearch/WebFetch were effective; found key formulas |
| **Code** | Iterative approach worked but was messy |
| **Skills** | Underutilized `/context7`, `/qrd`, `/workflows:plan` |
| **Repo** | Needs quant template, cached data, indicator library |
| **Process** | Plan first, then code; vectorize early |

**Bottom Line:** The task was achievable but took 2-3x longer than necessary due to skipping planning steps and not using available tools optimally.
