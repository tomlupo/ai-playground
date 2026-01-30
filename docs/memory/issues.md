# Known Issues & Workarounds

Track recurring issues and their workarounds across sessions.

## Format

```
### Issue Title
**Symptom:** What goes wrong
**Cause:** Root cause if known
**Workaround:** How to work around it
**Status:** open | resolved | monitoring
```

## Active Issues

### yfinance HES ticker delisted
**Symptom:** `HTTP Error 404: Quote not found for symbol: HES` during data fetch
**Cause:** HES (Hess Corp) was acquired by Chevron, ticker delisted
**Workaround:** Code handles gracefully (skips failed tickers), universe runs with 99/100
**Status:** monitoring — may need replacement ticker for Energy sector

### yfinance multi-ticker download returns MultiIndex columns
**Symptom:** `TypeError: float() argument must be a string or a real number, not 'Series'`
**Cause:** When downloading multiple tickers with `group_by="ticker"`, duplicate datetime indices cause `.loc[date, "Close"]` to return Series
**Workaround:** Use `_safe_float()` wrapper that calls `.iloc[0]` on Series results. Also handle `get_loc()` returning slice/bool array.
**Status:** resolved — pattern documented in patterns.md
