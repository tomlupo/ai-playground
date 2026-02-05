# Autonomous Mode

Rules for headless/batch execution via `claude -p`.

## Detection

Autonomous mode is active when:
- User prompt contains "AUTONOMOUS MODE" or "## Execution Mode: AUTONOMOUS"
- User prompt contains "--autonomous" or "/auto"
- Running via `claude -p` with `--dangerously-skip-permissions`

## Behavior in Autonomous Mode

- Make reasonable assumptions instead of asking clarifying questions
- Document assumptions in output (don't ask about them)
- Use conservative/standard defaults for parameters
- If something fails, try 2 alternatives before stopping
- Always produce output to `output/{branch}/`

## Default Assumptions

When information is missing, use these defaults:

| Category | Default |
|----------|---------|
| Data | Use `data/samples/` cached data when available |
| Indicators | RSI-14, SMA-20/50/200, EMA-12/26 |
| Train/Test | 80/20 split with chronological ordering |
| Backtest | Daily data, 5+ years, long-only |
| Success | Sharpe > 0.5, MaxDD < 25%, positive CAGR |
| File paths | `scratch/{agent-name}/` then `tools/{name}/` for promotion |
| Output | Always timestamp files, always generate gist-report |

## When to STILL Ask (even in autonomous mode)

- Destructive operations (deleting files, force-push)
- Spending money (API calls with costs)
- Accessing external systems not mentioned in prompt
- Ambiguous instructions that could lead to completely wrong results

## Prompt Structure for Autonomous Execution

Structure prompts to enable autonomous mode:

```markdown
## Task
[Clear description]

## Execution Mode: AUTONOMOUS
- Make reasonable assumptions, document them
- Skip interactive skill phases
- Always produce output

## Assumptions (use these, don't ask)
- [Pre-answer likely questions]
```

## Skills Compatibility

**Fully compatible with autonomous mode:**
- `/workflows:work` - Execution-focused
- `market-data-fetcher` - Takes direct parameters
- `gist-report` - Output generation
- `context7` - Documentation lookup

**Require adaptation for autonomous use:**
- `qrd` - Provide all 8 fields OR use AUTONOMOUS flag
- `brainstorming` - Skip, use `/workflows:plan` instead

## Skills to ALWAYS Use in Autonomous Mode

**IMPORTANT:** Autonomous mode does NOT mean skip all skills. These skills should still be invoked:

| Skill | Why | How to Use |
|-------|-----|------------|
| `/context7` | Fetch library docs before writing code | Add explicit call in prompt |
| `/retrospective` | Document learnings after completion | Add as final step |
| `/gist-report` | Share results | Add as final step |

### Example Autonomous Prompt with Skills

```bash
claude -p "
## Execution Mode: AUTONOMOUS

## Task
Build a momentum strategy backtest.

## Skills to Invoke
1. /context7 pandas vectorization - before writing signal generation
2. /context7 numpy broadcasting - for efficient array operations
3. /retrospective - at end to document learnings
4. /gist-report - at end to share results

## Assumptions
- Use data/samples/SPY.csv
- Long-only, daily rebalancing
" --dangerously-skip-permissions
```

### Performance Patterns for Quant Code

Always use vectorized operations instead of row-by-row loops:

```python
# BAD - Will hang on large datasets
for i in range(1, len(df)):
    if rsi.iloc[i] < threshold:
        signal.iloc[i] = 1

# GOOD - Vectorized
signal = np.where(rsi < threshold, 1, 0)
```

## Examples

### Autonomous QRD Execution

```bash
claude -p "
## Execution Mode: AUTONOMOUS

/qrd momentum-strategy

Objective: Test 12-month price momentum on US equities
Assets: SPY (use data/samples/SPY.csv)
Type: Momentum/trend-following
Horizon: Daily rebalancing, 1-month holding
Data: data/samples/
Constraints: Long-only, fully invested
Success: Sharpe > 0.5, MaxDD < 30%
Infrastructure: vectorbt, pandas
" --dangerously-skip-permissions
```

### Simple Strategy Backtest

```bash
claude -p "
## Execution Mode: AUTONOMOUS

Create a simple RSI mean-reversion strategy for SPY.
- RSI < 30: Buy signal
- RSI > 70: Sell signal
- Use data/samples/SPY.csv
- Save results to output/
"
```
