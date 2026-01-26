# Claude Code Web Environment Setup

This guide covers configuring the ai-playground repository for optimal use with Claude Code Web (cloud-based Claude Code sessions).

## Overview

Claude Code Web provides cloud-based execution with automatic dependency installation via SessionStart hooks. However, network access is restricted by default, requiring configuration for external data sources.

## Required Network Access

For full market-data-fetcher functionality, configure **"Full Internet"** access or add these specific domains:

### Market Data Sources
| Domain | Purpose | Required |
|--------|---------|----------|
| `stooq.pl` | Polish stocks, indices, FX | Yes (for Polish markets) |
| `api.nbp.pl` | NBP FX rates | Yes (for PLN FX) |
| `query1.finance.yahoo.com` | Yahoo Finance API | Yes (for US/intl stocks) |
| `finance.yahoo.com` | Yahoo Finance web | Yes |
| `api.tiingo.com` | Tiingo stock data | Optional (requires API key) |
| `fred.stlouisfed.org` | FRED economic data | Optional (requires API key) |

### Always Allowed (No Configuration Needed)
- `github.com`, `api.github.com` - Gist creation works via proxy
- `pypi.org`, `files.pythonhosted.org` - Package installation
- `*.googleapis.com` - Some cloud features

## Environment Variables

Set these in your Claude Code Web environment configuration (not `.env` file):

```bash
# Market Data APIs (optional - free sources work without keys)
TIINGO_API_KEY=your_tiingo_key
FRED_API_KEY=your_fred_key

# Context7 Documentation API
CONTEXT7_API_KEY=your_context7_key

# Offline mode (use sample data when network restricted)
MARKET_DATA_OFFLINE_MODE=false
```

### Getting API Keys

1. **Tiingo** (free tier available): https://api.tiingo.com/
2. **FRED** (free): https://fred.stlouisfed.org/docs/api/api_key.html
3. **Context7**: Contact Context7 for API access

## SessionStart Hook

The repository includes a SessionStart hook (`scripts/setup-env.sh`) that automatically:

1. Installs `uv` package manager
2. Installs GitHub CLI (`gh`)
3. Syncs project dependencies
4. Creates required directories
5. Pre-downloads NLP models (for pdf-skill)
6. Checks data source connectivity

## Verifying Setup

After starting a Cloud Web session, verify the environment:

```bash
# Check tools
which uv && which gh

# Check Python environment
uv run python --version

# Check data source connectivity
curl -s --max-time 5 https://stooq.pl > /dev/null && echo "Stooq: OK" || echo "Stooq: BLOCKED"
curl -s --max-time 5 https://api.nbp.pl > /dev/null && echo "NBP: OK" || echo "NBP: BLOCKED"

# Test market data fetching
uv run .claude/skills/market-data-fetcher/scripts/fetch_stooq.py WIG20 2024-01-01 2024-01-31
```

## Offline Mode / Limited Network

If network access is restricted, the market-data-fetcher can use cached sample data:

### Available Sample Data
- `data/samples/WIG20.csv` - Polish WIG20 index
- `data/samples/SPY.csv` - S&P 500 ETF
- `data/samples/BTC-USD.csv` - Bitcoin/USD

### Using Offline Mode

Set `MARKET_DATA_OFFLINE_MODE=true` in environment config, or the fetcher will automatically fall back to sample data when network requests fail.

```python
from fetch_unified import fetch_with_offline_fallback

# Will use sample data if network fails
df = fetch_with_offline_fallback('WIG20', '2024-01-01', '2024-03-31')
```

## Best Practices for Cloud Web

### 1. Use Parallel Execution
Use `&` prefix to spawn parallel tasks:
```
& Fetch AAPL data and analyze trends
& Fetch MSFT data and analyze trends
& Fetch GOOGL data and analyze trends
```

### 2. Upload PDFs Before Session
For pdf-skill, upload PDF files to the repository before starting a cloud session (or use git push).

### 3. Use QRD Skill for Specs
Create structured research specifications using `/qrd` before executing:
```
/qrd momentum
```

### 4. Commit Results
Use `/commit` to save analysis results and push to branches.

## Limitations in Cloud Web

### MCP Servers Don't Work
MCP servers are not supported in Claude Code Web. Use the `context7` skill (Python CLI) instead of MCP:

```bash
# Search for library
python3 .claude/skills/context7/scripts/context7.py search "next.js"

# Fetch docs
python3 .claude/skills/context7/scripts/context7.py context "/vercel/next.js" "app router"
```

### Local Path References
Commands referencing local paths (like `/gist-transcript`) may not work. Use cloud-compatible alternatives.

### Browser Testing
Browser-based testing commands require local browser access.

## Troubleshooting

### "Connection refused" or timeout errors
- Check if "Full Internet" access is enabled
- Verify specific domains are allowlisted
- Try offline fallback mode

### Package installation failures
- Check `pypi.org` connectivity
- Verify `uv` is properly installed
- Check SessionStart hook completed

### PDF skill model downloads fail
- Ensure `files.pythonhosted.org` is accessible
- Models are cached after first download

### GitHub gist creation fails
- Verify `gh` CLI is installed and authenticated
- Check GitHub auth via cloud proxy is working
