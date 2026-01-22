# Bankier.pl Forum Sentiment Scraper

A Python-based scraping and sentiment analysis pipeline for the Bankier.pl investor forum. Extracts retail investor sentiment signals for Polish equities (GPW).

## Features

- **Async Scraping**: Fast, rate-limited scraping of Bankier.pl forum using `httpx` and `selectolax`
- **Sentiment Analysis**: Polish financial text sentiment using lexicon-based analysis (HerBERT optional)
- **Ticker Detection**: Automatic detection of WSE ticker mentions with company name aliases
- **Signal Aggregation**: Daily sentiment metrics by ticker with momentum indicators
- **REST API**: FastAPI service for querying sentiment data
- **CSV Export**: Export functionality for backtesting

## Quick Start

### Installation

```bash
# Clone and enter directory
cd bankier-forum

# Install dependencies (using pip)
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

### Run the Pipeline

```bash
# Run full pipeline (scrape, analyze, aggregate)
python -m scripts.run_scraper --all

# Or run individual steps
python -m scripts.run_scraper --scrape --pages 5     # Scrape 5 pages
python -m scripts.run_scraper --analyze              # Analyze sentiment
python -m scripts.run_scraper --aggregate            # Aggregate daily signals
```

### Start the API

```bash
uvicorn src.api.main:app --reload
```

Then visit:
- http://localhost:8000/docs - API documentation
- http://localhost:8000/sentiment/CDR - Get CD Projekt sentiment
- http://localhost:8000/signals/top-movers - Get daily top movers

## Project Structure

```
bankier-forum/
├── config/
│   ├── settings.yaml       # Configuration
│   └── tickers.json        # WSE ticker list
├── docs/
│   └── prd.md              # Product requirements
├── src/
│   ├── scraper/            # Bankier.pl scraper
│   ├── nlp/                # Sentiment & ticker detection
│   ├── db/                 # Database models & repository
│   ├── signals/            # Aggregation logic
│   └── api/                # FastAPI service
├── scripts/
│   └── run_scraper.py      # Main pipeline script
├── data/                   # SQLite database & exports
└── tests/                  # Test files
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /sentiment/{ticker}` | Historical sentiment for a ticker |
| `GET /signals/unusual` | Tickers with unusual activity |
| `GET /signals/top-movers` | Top bullish/bearish tickers |
| `GET /tickers` | List all tickers |
| `GET /export/csv` | Export data as CSV |

## Configuration

Edit `config/settings.yaml` to customize:
- Scraping rate limits
- Forum priorities
- NLP model settings
- Database connection

## Example Output

```csv
date,ticker,post_count,avg_sentiment,bullish_ratio,bearish_ratio
2024-01-15,CDR,45,0.23,0.42,0.18
2024-01-15,PKO,23,-0.12,0.22,0.35
```

## Dependencies

- Python 3.11+
- httpx (async HTTP client)
- selectolax (fast HTML parser)
- SQLAlchemy (database ORM)
- FastAPI (REST API)
- transformers (optional, for HerBERT)

## License

MIT
