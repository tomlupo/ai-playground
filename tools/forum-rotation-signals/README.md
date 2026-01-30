# Bankier.pl Forum Rotation Signal Scanner

Monitors Bankier.pl stock exchange forum to identify capital rotation signals from retail investor chatter **before** analysts and mainstream media fully embrace the themes.

## Strategy

1. **Scrape** Bankier.pl "Giełda" forum (most active WSE discussion board)
2. **Extract** ticker mentions from thread titles using WSE ticker master + aliases
3. **Classify** threads into 14 macro themes (sectors, catalysts, market dynamics)
4. **Score** rotation signals: mention velocity, sentiment skew, cross-theme momentum
5. **Surface** top 10 themes + high-potential tickers with actionable signals

## Signals Detected

| Signal | Description |
|--------|-------------|
| Management Shakeup & Governance | Executive changes creating uncertainty or opportunity |
| Commodity Supercycle / Metals | Metal/mining momentum from supply/demand shifts |
| Defense & Geopolitics | Geopolitical catalysts driving defense spending |
| AI & Tech Disruption | AI/tech themes affecting Polish tech sector |
| Gaming Cycle | Game release catalysts and studio valuations |
| Biotech & Pharma Catalysts | Clinical trial results and pharma pipeline events |
| Rate Cycle & Banking | Interest rate expectations affecting bank earnings |
| Green Energy Transition | Renewable energy policy and investment themes |
| Short Squeeze & Retail Momentum | High short interest + retail coordination signals |
| Earnings Momentum | Pre/post earnings discussion velocity |

## Usage

```bash
# Default: scan 10 pages
uv run tools/forum-rotation-signals/main.py

# Deep scan: 20 pages
uv run tools/forum-rotation-signals/main.py --pages 20

# Custom output path
uv run tools/forum-rotation-signals/main.py --output outputs/my_scan.md
```

## Output

- Console: Rich-formatted tables with color-coded signals
- File: Markdown report saved to `outputs/rotation_signals_YYYYMMDD_HHMMSS.md`

## Signal Scoring

Composite score = volume + breadth + activity + sentiment_bonus

- **Volume**: ticker mention count (capped)
- **Breadth**: unique ticker count across theme
- **Activity**: thread count in theme
- **Sentiment bonus**: absolute sentiment strength (strong conviction = signal)

## Disclaimer

Social sentiment analysis tool. Forum chatter contains noise, manipulation, and misinformation. Always validate against fundamentals, technicals, and risk management.
