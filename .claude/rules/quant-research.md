# Quant Research Checklist

Pre-flight checklist before starting any quantitative research task.

## Before Writing Code

1. **Check library documentation** via `/context7` for:
   - yfinance (data fetching)
   - vectorbt (backtesting)
   - pandas/numpy (data manipulation)
   - Any other libraries you'll use

2. **Load cached data** from `data/samples/` when possible:
   - SPY, QQQ, TLT, GLD (US markets)
   - BTC-USD (crypto)
   - WIG20 (Polish markets)
   - Avoids rate limits and network issues

3. **Use `/qrd`** to create a structured specification:
   - Hypothesis
   - Data requirements
   - Entry/exit rules
   - Parameter ranges
   - Success criteria

4. **Set random seeds** for reproducibility:
   ```python
   import numpy as np
   np.random.seed(42)
   ```

5. **Use the template** at `tools/_templates/quant-research/`:
   ```bash
   cp -r tools/_templates/quant-research tools/my-strategy
   ```

6. **Plan parameter ranges BEFORE coding**:
   - Define min, max, step for each parameter
   - Estimate total combinations
   - Plan for out-of-sample testing

## During Research

- Use indicators from `shared/indicators.py`
- Save intermediate results to `output/`
- Print progress for long-running operations
- Use vectorized operations (avoid loops)

## After Research

- Save final results to `output/` with timestamps
- Document findings in report
- Use `/gist-report` to share results
- Consider adding to `docs/memory/patterns.md` if reusable

## Common Mistakes to Avoid

- Fetching data when cached version exists
- Not setting random seeds
- Optimizing without holdout period
- Starting to code before defining parameter ranges
- Forgetting to document the approach
