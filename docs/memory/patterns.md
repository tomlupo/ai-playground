# Patterns & Conventions

Recurring patterns and conventions discovered during development.

## Code Patterns

- **Tool structure:** Each tool in `tools/{name}/main.py` with PEP 723 inline metadata
- **Data flow:** `data/raw/` -> `data/processed/` -> `outputs/`
- **Scratch first:** All exploratory code starts in `scratch/{agent-name}/`

## Naming Conventions

- Tool directories: lowercase with hyphens (`momentum-analyzer`)
- Output files: include timestamp (`{name}_{YYYYMMDD_HHMMSS}.png`)
- Data files: descriptive with date/version if applicable

## Common Pitfalls

### yfinance duplicate index entries
`df.loc[date, "Close"]` can return a `Series` instead of a scalar when the DatetimeIndex has duplicates. Always wrap with a safe extractor:
```python
def _safe_float(val):
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    return float(val)
```
Similarly, `df.index.get_loc(date)` can return a `slice` or boolean array instead of `int`. Handle all cases.

### Paper replication: read execution details precisely
Academic papers often bury critical quantitative constraints in equations, appendices, or single sentences. Key details that dramatically change results:
- **Capital deployment limits** (e.g. "80% remains in cash") — a single sentence can be the difference between beta=0.05 and beta=0.33
- **Holding period caps** — 30 days vs 63 days completely changes turnover and P&L
- **Position sizing formulas** — Eq.17-style diminishing allocation vs flat percentage
- **Trade frequency** — cross-check total trade counts against your output

### Research replication workflow
1. **paper-reading** skill (3-pass) → extract ALL quantitative details (formulas, parameters, constraints)
2. **qrd** skill → create spec with acceptance criteria (target trade count, beta, returns)
3. **market-data-fetcher** skill → clean data acquisition with caching
4. Build implementation against spec
5. Run and compare against acceptance criteria
6. If divergent: re-read paper for missed constraints
7. **statistical-analysis** skill → proper stat tests with APA reporting
8. **/llm-external-review:code** → external review for logic errors

### Skill usage discipline
Always evaluate available skills BEFORE starting implementation:
- `paper-reading` for any academic paper analysis
- `qrd` for any quant research task (creates spec + acceptance criteria)
- `market-data-fetcher` for any financial data (replaces raw yfinance)
- `statistical-analysis` for hypothesis testing
- `context7` for library documentation
- `exploratory-data-analysis` for data profiling
- `/llm-external-review:code` after significant code changes

The `forced-eval` hook exists to enforce this — don't bypass it.

## Quant Research Patterns

### Walk-forward validation structure
```
for fold in folds:
    train_dates = all_dates[start:start+train_window]
    test_dates = all_dates[train_end:train_end+test_window]
    # Train agent on train_dates (lookahead OK)
    # Test strictly on test_dates (no lookahead)
    start += step_size
```

### Hypothesis-driven trading framework
- Define hypothesis types with named conditions, confidence, target, stop-loss
- RL agent gates execution: ε-greedy with adaptive threshold τ(c) = base + (1-c) × range
- Portfolio enforces: max positions, position sizing, capital reserves, slippage, commissions
