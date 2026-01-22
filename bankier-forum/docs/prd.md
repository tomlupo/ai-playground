# PRD: Bankier.pl Forum Sentiment Scraper

## Overview

Build a Python-based scraping and sentiment analysis pipeline for the Bankier.pl investor forum to extract retail investor sentiment signals for Polish equities (GPW).

**Goal:** Generate time-series sentiment data aggregated by ticker symbol that can be used as alternative data input for quantitative trading strategies.

-----

## Technical Stack

|Component       |Technology                      |Rationale                         |
|----------------|--------------------------------|----------------------------------|
|HTTP Client     |`httpx` (async)                 |Fast, modern, native async support|
|HTML Parser     |`selectolax`                    |10x faster than BeautifulSoup     |
|Database        |SQLite (dev) / PostgreSQL (prod)|Simple start, easy migration      |
|Sentiment       |`transformers` + HerBERT        |Best Polish language model        |
|Ticker Detection|Regex + custom lexicon          |WSE ticker patterns               |
|Scheduler       |`apscheduler` or cron           |Periodic scraping                 |

-----

## Data Sources

### Forum Categories to Scrape

|Category            |URL Pattern                                       |Priority|
|--------------------|--------------------------------------------------|--------|
|Giełda (Main)       |`/forum/forum_gielda,6,{page}.html`               |High    |
|Jak grać na giełdzie|`/forum/forum_jak-grac-na-gieldzie,50,{page}.html`|Medium  |
|Kryptowaluty        |`/forum/forum_kryptowaluty,55,{page}.html`        |Medium  |
|Forex               |`/forum/forum_forex,7,{page}.html`                |Low     |
|ETF                 |`/forum/forum_etf,59,{page}.html`                 |Medium  |

### URL Patterns Discovered

```
Base URL: https://www.bankier.pl

Category listing:
  /forum/forum_gielda,{forum_id},{page}.html

Thread view (summary):
  /forum/temat_{slug},{thread_id}.html

Thread posts (full content):
  /forum/pokaz-tresc?thread_id={thread_id}&strona={page}
```

-----

## Data Models

### Database Schema

```sql
-- Core tables
CREATE TABLE forums (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    url_pattern TEXT NOT NULL
);

CREATE TABLE threads (
    id INTEGER PRIMARY KEY,  -- bankier thread_id
    forum_id INTEGER REFERENCES forums(id),
    title TEXT NOT NULL,
    slug TEXT,
    author TEXT,
    created_at TIMESTAMP,
    last_scraped_at TIMESTAMP,
    post_count INTEGER DEFAULT 0,
    view_count INTEGER DEFAULT 0
);

CREATE TABLE posts (
    id INTEGER PRIMARY KEY,  -- bankier post_id
    thread_id INTEGER REFERENCES threads(id),
    author TEXT NOT NULL,
    author_ip_fragment TEXT,  -- e.g., "149.156.96.*"
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    votes_up INTEGER DEFAULT 0,
    votes_down INTEGER DEFAULT 0,
    is_op BOOLEAN DEFAULT FALSE,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(id)
);

CREATE TABLE post_sentiment (
    post_id INTEGER PRIMARY KEY REFERENCES posts(id),
    sentiment_score REAL,  -- -1.0 to 1.0
    sentiment_label TEXT,  -- 'positive', 'negative', 'neutral'
    confidence REAL,
    model_version TEXT,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ticker_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER REFERENCES posts(id),
    ticker TEXT NOT NULL,  -- e.g., 'CDR', 'PKO', '11B'
    company_name TEXT,
    mention_type TEXT,  -- 'explicit', 'inferred'
    context_snippet TEXT,  -- 50 chars around mention
    UNIQUE(post_id, ticker)
);

-- Aggregated signals (materialized view or table)
CREATE TABLE daily_sentiment (
    date DATE NOT NULL,
    ticker TEXT NOT NULL,
    post_count INTEGER,
    avg_sentiment REAL,
    sentiment_stddev REAL,
    bullish_ratio REAL,  -- % posts with sentiment > 0.2
    bearish_ratio REAL,  -- % posts with sentiment < -0.2
    unique_authors INTEGER,
    PRIMARY KEY (date, ticker)
);

-- Indexes
CREATE INDEX idx_posts_created_at ON posts(created_at);
CREATE INDEX idx_posts_thread_id ON posts(thread_id);
CREATE INDEX idx_ticker_mentions_ticker ON ticker_mentions(ticker);
CREATE INDEX idx_ticker_mentions_post_id ON ticker_mentions(post_id);
CREATE INDEX idx_daily_sentiment_ticker ON daily_sentiment(ticker);
```

### Pydantic Models

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ForumPost(BaseModel):
    post_id: int
    thread_id: int
    author: str
    author_ip_fragment: Optional[str]
    content: str
    created_at: datetime
    votes_up: int = 0
    votes_down: int = 0
    is_op: bool = False

class ThreadInfo(BaseModel):
    thread_id: int
    forum_id: int
    title: str
    slug: Optional[str]
    author: str
    created_at: datetime
    post_count: int
    view_count: int

class SentimentResult(BaseModel):
    post_id: int
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: str    # positive/negative/neutral
    confidence: float
    model_version: str

class TickerMention(BaseModel):
    post_id: int
    ticker: str
    company_name: Optional[str]
    mention_type: str  # explicit/inferred
    context_snippet: str
```

-----

## Scraping Logic

### Rate Limiting & Politeness

```python
SCRAPE_CONFIG = {
    "requests_per_second": 0.5,  # 1 request per 2 seconds
    "retry_attempts": 3,
    "retry_backoff": [5, 15, 60],  # seconds
    "user_agents": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
        # Add 5-10 realistic user agents
    ],
    "respect_robots_txt": True,
    "max_pages_per_run": 100,
}
```

### HTML Parsing Selectors

Based on the actual forum HTML structure:

```python
SELECTORS = {
    # Thread listing page
    "thread_list": {
        "container": "div.forum-thread-list",  # Verify actual selector
        "thread_link": "a[href*='/forum/temat_']",
        "thread_title": "a[href*='/forum/temat_']::text",
        "post_count": "span.post-count::text",  # Verify
        "last_activity": "span.last-activity::text",  # Verify
    },

    # Post detail page (/forum/pokaz-tresc)
    "post_detail": {
        "post_container": "li",  # Each post is in a list item
        "author": "Autor: ",  # Text pattern to find author
        "author_ip": r"\[(\d+\.\d+\.\d+\.\*)\]",  # Regex for IP fragment
        "timestamp": r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})",  # Date pattern
        "content": "post content area",  # Main text block
        "votes": r"(\d+).*?(\d+)",  # Up/down vote pattern
    }
}
```

### Core Scraper Class

```python
# src/scraper/bankier_scraper.py

import httpx
import asyncio
from selectolax.parser import HTMLParser
from datetime import datetime
import re
from typing import AsyncGenerator
import random

class BankierForumScraper:
    BASE_URL = "https://www.bankier.pl"

    def __init__(self, config: dict):
        self.config = config
        self.client = None
        self._request_timestamps = []

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": random.choice(self.config["user_agents"])}
        )
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    async def _rate_limit(self):
        """Enforce rate limiting"""
        now = datetime.now().timestamp()
        min_interval = 1 / self.config["requests_per_second"]

        if self._request_timestamps:
            elapsed = now - self._request_timestamps[-1]
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

        self._request_timestamps.append(datetime.now().timestamp())
        # Keep only last 100 timestamps
        self._request_timestamps = self._request_timestamps[-100:]

    async def fetch_page(self, url: str) -> str:
        """Fetch a page with rate limiting and retries"""
        await self._rate_limit()

        for attempt, backoff in enumerate(self.config["retry_backoff"]):
            try:
                response = await self.client.get(url)
                response.raise_for_status()
                return response.text
            except httpx.HTTPError as e:
                if attempt < len(self.config["retry_backoff"]) - 1:
                    await asyncio.sleep(backoff)
                else:
                    raise

    async def scrape_forum_page(self, forum_id: int, page: int) -> list[dict]:
        """Scrape thread listing from a forum page"""
        url = f"{self.BASE_URL}/forum/forum_gielda,{forum_id},{page}.html"
        html = await self.fetch_page(url)
        tree = HTMLParser(html)

        threads = []
        # Parse thread links - adjust selectors based on actual HTML
        for link in tree.css("a[href*='/forum/temat_']"):
            href = link.attributes.get("href", "")
            match = re.search(r"temat_([^,]+),(\d+)\.html", href)
            if match:
                threads.append({
                    "slug": match.group(1),
                    "thread_id": int(match.group(2)),
                    "title": link.text(strip=True),
                    "url": f"{self.BASE_URL}{href}"
                })

        return threads

    async def scrape_thread_posts(self, thread_id: int) -> AsyncGenerator[dict, None]:
        """Scrape all posts from a thread"""
        page = 1
        while True:
            url = f"{self.BASE_URL}/forum/pokaz-tresc?thread_id={thread_id}&strona={page}"
            html = await self.fetch_page(url)
            tree = HTMLParser(html)

            posts = self._parse_posts(tree, thread_id)
            if not posts:
                break

            for post in posts:
                yield post

            # Check for next page
            if not self._has_next_page(tree):
                break
            page += 1

    def _parse_posts(self, tree: HTMLParser, thread_id: int) -> list[dict]:
        """Parse posts from thread HTML"""
        posts = []

        # Based on observed HTML structure - posts are in list items
        # The actual structure shows posts with:
        # - Author line: "Autor: ~nickname [IP]" or "Autor: username [IP]"
        # - Date: "2017-01-23 18:47"
        # - Content: paragraph text
        # - Actions: Odpowiedz, Zgłoś do moderatora, vote counts

        # This is a simplified parser - needs refinement based on actual HTML
        post_blocks = tree.css("li")  # Adjust selector

        for block in post_blocks:
            text = block.text()

            # Extract author
            author_match = re.search(r"Autor:\s*([~\w]+)\s*\[([^\]]+)\]", text)
            if not author_match:
                continue

            # Extract timestamp
            time_match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})", text)
            if not time_match:
                continue

            # Extract post ID from link
            post_link = block.css_first("a[href*='parent_id=']")
            post_id = None
            if post_link:
                id_match = re.search(r"parent_id=(\d+)", post_link.attributes.get("href", ""))
                if id_match:
                    post_id = int(id_match.group(1))

            # Extract content (everything between author line and action links)
            # This needs refinement based on actual structure
            content = self._extract_content(block)

            if post_id and content:
                posts.append({
                    "post_id": post_id,
                    "thread_id": thread_id,
                    "author": author_match.group(1),
                    "author_ip_fragment": author_match.group(2),
                    "created_at": datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M"),
                    "content": content,
                    "votes_up": 0,  # Extract from HTML
                    "votes_down": 0,  # Extract from HTML
                })

        return posts

    def _extract_content(self, block) -> str:
        """Extract post content text"""
        # Remove navigation/action elements
        # Return clean text content
        # Implementation depends on actual HTML structure
        return block.text(strip=True)  # Simplified

    def _has_next_page(self, tree: HTMLParser) -> bool:
        """Check if there's a next page"""
        # Look for pagination links
        pagination = tree.css("a[href*='strona=']")
        return len(pagination) > 1
```

-----

## Sentiment Analysis Pipeline

### Model Selection

Use **HerBERT** (Polish BERT) with optional fine-tuning on financial text:

```python
# src/nlp/sentiment.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Tuple

class PolishFinancialSentiment:
    def __init__(self, model_name: str = "allegro/herbert-base-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3  # positive, negative, neutral
        )
        self.model.eval()

        # Financial lexicon boost
        self.bullish_keywords = {
            "kupuj", "kup", "wzrost", "hossa", "zysk", "rośnie",
            "przebicie", "wybicie", "long", "target", "cel",
            "dywidenda", "rekomendacja", "akumuluj", "trzymaj"
        }
        self.bearish_keywords = {
            "sprzedaj", "spadek", "bessa", "strata", "spada",
            "short", "unikaj", "redukuj", "ryzyko", "krach",
            "bankructwo", "dno", "dump", "manipulacja"
        }

    def analyze(self, text: str) -> Tuple[float, str, float]:
        """
        Analyze sentiment of Polish financial text

        Returns:
            score: -1.0 to 1.0
            label: 'positive', 'negative', 'neutral'
            confidence: 0.0 to 1.0
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # Map to sentiment score
        # Assuming: 0=negative, 1=neutral, 2=positive
        probs = probs[0].numpy()

        score = probs[2] - probs[0]  # positive - negative
        confidence = max(probs)

        if score > 0.2:
            label = "positive"
        elif score < -0.2:
            label = "negative"
        else:
            label = "neutral"

        # Apply lexicon boost
        score = self._apply_lexicon_boost(text.lower(), score)

        return score, label, confidence

    def _apply_lexicon_boost(self, text: str, base_score: float) -> float:
        """Adjust score based on financial keywords"""
        bullish_count = sum(1 for kw in self.bullish_keywords if kw in text)
        bearish_count = sum(1 for kw in self.bearish_keywords if kw in text)

        boost = (bullish_count - bearish_count) * 0.1
        return max(-1.0, min(1.0, base_score + boost))

    def analyze_batch(self, texts: List[str]) -> List[Tuple[float, str, float]]:
        """Batch analysis for efficiency"""
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results
```

### Ticker Detection

```python
# src/nlp/ticker_detector.py

import re
from typing import List, Tuple

class WSETickerDetector:
    def __init__(self):
        # Load WSE ticker list (from GPW or stooq)
        self.ticker_to_company = self._load_ticker_mapping()

        # Common variations
        self.aliases = {
            "cd projekt": "CDR",
            "cdprojekt": "CDR",
            "cd project": "CDR",
            "allegro": "ALE",
            "orlen": "PKN",
            "pko": "PKO",
            "pko bp": "PKO",
            "pekao": "PEO",
            "kghm": "KGH",
            "cyfrowy polsat": "CPS",
            "dino": "DNP",
            "pepco": "PCO",
            "zabka": "ZAB",
            "żabka": "ZAB",
            # Add more common aliases
        }

        # Regex for explicit tickers (3-5 uppercase letters)
        self.ticker_pattern = re.compile(r'\b([A-Z]{3,5})\b')

    def _load_ticker_mapping(self) -> dict:
        """Load ticker -> company name mapping"""
        # TODO: Load from file or scrape from GPW
        return {
            "CDR": "CD Projekt",
            "PKO": "PKO Bank Polski",
            "PKN": "PKN Orlen",
            "KGH": "KGHM",
            "PEO": "Bank Pekao",
            "ALE": "Allegro",
            "DNP": "Dino Polska",
            "11B": "11 bit studios",
            "CPS": "Cyfrowy Polsat",
            "LPP": "LPP",
            "PZU": "PZU",
            # Add full WSE listing
        }

    def detect(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Detect ticker mentions in text

        Returns:
            List of (ticker, company_name, mention_type)
        """
        mentions = []
        text_lower = text.lower()

        # Check aliases first (company names)
        for alias, ticker in self.aliases.items():
            if alias in text_lower:
                mentions.append((
                    ticker,
                    self.ticker_to_company.get(ticker, ""),
                    "inferred"
                ))

        # Check explicit tickers
        for match in self.ticker_pattern.finditer(text):
            ticker = match.group(1)
            if ticker in self.ticker_to_company:
                mentions.append((
                    ticker,
                    self.ticker_to_company[ticker],
                    "explicit"
                ))

        # Deduplicate
        seen = set()
        unique_mentions = []
        for mention in mentions:
            if mention[0] not in seen:
                seen.add(mention[0])
                unique_mentions.append(mention)

        return unique_mentions

    def get_context(self, text: str, ticker: str, window: int = 50) -> str:
        """Extract context around ticker mention"""
        text_lower = text.lower()

        # Find position of ticker or alias
        pos = text_lower.find(ticker.lower())
        if pos == -1:
            for alias, t in self.aliases.items():
                if t == ticker and alias in text_lower:
                    pos = text_lower.find(alias)
                    break

        if pos == -1:
            return text[:window*2]

        start = max(0, pos - window)
        end = min(len(text), pos + window)
        return text[start:end]
```

-----

## Aggregation & Signals

### Daily Aggregation

```python
# src/signals/aggregator.py

from datetime import date, timedelta
from typing import Dict, List
import statistics

class SentimentAggregator:
    def __init__(self, db_connection):
        self.db = db_connection

    def aggregate_daily(self, target_date: date) -> Dict[str, dict]:
        """
        Aggregate sentiment by ticker for a given date

        Returns:
            {ticker: {metrics...}}
        """
        query = """
            SELECT
                tm.ticker,
                ps.sentiment_score,
                p.author,
                p.created_at
            FROM posts p
            JOIN post_sentiment ps ON p.id = ps.post_id
            JOIN ticker_mentions tm ON p.id = tm.post_id
            WHERE DATE(p.created_at) = ?
        """

        rows = self.db.execute(query, (target_date,)).fetchall()

        # Group by ticker
        ticker_data = {}
        for ticker, score, author, created_at in rows:
            if ticker not in ticker_data:
                ticker_data[ticker] = {
                    "scores": [],
                    "authors": set()
                }
            ticker_data[ticker]["scores"].append(score)
            ticker_data[ticker]["authors"].add(author)

        # Calculate metrics
        results = {}
        for ticker, data in ticker_data.items():
            scores = data["scores"]
            results[ticker] = {
                "date": target_date,
                "ticker": ticker,
                "post_count": len(scores),
                "avg_sentiment": statistics.mean(scores),
                "sentiment_stddev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "bullish_ratio": sum(1 for s in scores if s > 0.2) / len(scores),
                "bearish_ratio": sum(1 for s in scores if s < -0.2) / len(scores),
                "unique_authors": len(data["authors"])
            }

        return results

    def calculate_momentum(self, ticker: str, days: int = 7) -> float:
        """Calculate sentiment momentum (change over period)"""
        query = """
            SELECT date, avg_sentiment
            FROM daily_sentiment
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
        """
        rows = self.db.execute(query, (ticker, days)).fetchall()

        if len(rows) < 2:
            return 0.0

        recent = rows[0][1]
        older = rows[-1][1]
        return recent - older

    def get_unusual_activity(self, target_date: date, threshold: float = 2.0) -> List[dict]:
        """Detect tickers with unusual posting activity"""
        query = """
            WITH recent_avg AS (
                SELECT ticker, AVG(post_count) as avg_posts, STDDEV(post_count) as std_posts
                FROM daily_sentiment
                WHERE date >= DATE(?, '-30 days')
                GROUP BY ticker
            )
            SELECT ds.ticker, ds.post_count, ra.avg_posts, ra.std_posts,
                   (ds.post_count - ra.avg_posts) / NULLIF(ra.std_posts, 0) as z_score
            FROM daily_sentiment ds
            JOIN recent_avg ra ON ds.ticker = ra.ticker
            WHERE ds.date = ?
            AND ABS((ds.post_count - ra.avg_posts) / NULLIF(ra.std_posts, 0)) > ?
        """
        return self.db.execute(query, (target_date, target_date, threshold)).fetchall()
```

-----

## API / Output

### FastAPI Service (Optional)

```python
# src/api/main.py

from fastapi import FastAPI, Query
from datetime import date, timedelta
from typing import List, Optional

app = FastAPI(title="Bankier Sentiment API")

@app.get("/sentiment/{ticker}")
async def get_ticker_sentiment(
    ticker: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
):
    """Get historical sentiment for a ticker"""
    if not start_date:
        start_date = date.today() - timedelta(days=30)
    if not end_date:
        end_date = date.today()

    # Query database
    return {"ticker": ticker, "data": [...]}

@app.get("/signals/unusual")
async def get_unusual_activity(target_date: Optional[date] = None):
    """Get tickers with unusual forum activity"""
    return {"date": target_date, "signals": [...]}

@app.get("/export/csv")
async def export_csv(
    start_date: date,
    end_date: date,
    tickers: Optional[List[str]] = Query(None)
):
    """Export sentiment data as CSV"""
    # Generate CSV
    return StreamingResponse(...)
```

### CSV Export Format

```csv
date,ticker,post_count,avg_sentiment,sentiment_stddev,bullish_ratio,bearish_ratio,unique_authors
2024-01-15,CDR,45,0.23,0.31,0.42,0.18,28
2024-01-15,PKO,23,-0.12,0.28,0.22,0.35,15
```

-----

## Project Structure

```
bankier-sentiment/
├── pyproject.toml
├── README.md
├── config/
│   ├── settings.yaml
│   └── tickers.json
├── src/
│   ├── __init__.py
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── bankier_scraper.py
│   │   └── rate_limiter.py
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── sentiment.py
│   │   └── ticker_detector.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── repository.py
│   ├── signals/
│   │   ├── __init__.py
│   │   └── aggregator.py
│   └── api/
│       ├── __init__.py
│       └── main.py
├── scripts/
│   ├── run_scraper.py
│   ├── backfill.py
│   └── export.py
├── tests/
│   ├── test_scraper.py
│   ├── test_sentiment.py
│   └── test_ticker_detector.py
└── data/
    ├── raw/           # Raw HTML backups
    ├── processed/     # Processed data
    └── exports/       # CSV exports
```

-----

## Implementation Phases

### Phase 1: Core Scraper (MVP)

**Duration: 2-3 days**

- [ ] Set up project structure with `uv` or `poetry`
- [ ] Implement `BankierForumScraper` class
- [ ] Test scraping on single forum category
- [ ] Store raw posts in SQLite
- [ ] Add basic rate limiting

**Deliverable:** Script that scrapes last 100 posts from Giełda forum

### Phase 2: NLP Pipeline

**Duration: 2-3 days**

- [ ] Integrate HerBERT for sentiment analysis
- [ ] Build ticker detection with WSE ticker list
- [ ] Create financial keyword lexicon
- [ ] Process stored posts through pipeline
- [ ] Add sentiment scores to database

**Deliverable:** Sentiment scores for all scraped posts

### Phase 3: Aggregation & Signals

**Duration: 1-2 days**

- [ ] Build daily aggregation queries
- [ ] Calculate momentum indicators
- [ ] Detect unusual activity
- [ ] Create `daily_sentiment` materialized table

**Deliverable:** Daily sentiment time series by ticker

### Phase 4: API & Export

**Duration: 1-2 days**

- [ ] FastAPI endpoints for querying data
- [ ] CSV export functionality
- [ ] Basic dashboard (optional)

**Deliverable:** API to query sentiment data

### Phase 5: Production Hardening

**Duration: 2-3 days**

- [ ] Add comprehensive error handling
- [ ] Implement retry logic with exponential backoff
- [ ] Add monitoring/alerting
- [ ] Set up scheduled scraping (cron/apscheduler)
- [ ] PostgreSQL migration
- [ ] Docker containerization

**Deliverable:** Production-ready system

-----

## Dependencies

```toml
[project]
name = "bankier-sentiment"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27.0",
    "selectolax>=0.3.21",
    "transformers>=4.40.0",
    "torch>=2.2.0",
    "pydantic>=2.7.0",
    "sqlalchemy>=2.0.0",
    "aiosqlite>=0.20.0",
    "apscheduler>=3.10.0",
    "fastapi>=0.111.0",
    "uvicorn>=0.29.0",
    "pandas>=2.2.0",
    "pyyaml>=6.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
]
```

-----

## Configuration

```yaml
# config/settings.yaml

scraping:
  base_url: "https://www.bankier.pl"
  requests_per_second: 0.5
  max_retries: 3
  retry_backoff: [5, 15, 60]
  max_pages_per_run: 100
  user_agents:
    - "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

forums:
  - id: 6
    name: "Giełda"
    priority: high
  - id: 50
    name: "Jak grać na giełdzie"
    priority: medium
  - id: 59
    name: "ETF"
    priority: medium

nlp:
  model_name: "allegro/herbert-base-cased"
  batch_size: 32
  sentiment_threshold:
    bullish: 0.2
    bearish: -0.2

database:
  url: "sqlite:///data/bankier_sentiment.db"
  # url: "postgresql://user:pass@localhost/bankier"

scheduler:
  scrape_interval_hours: 4
  aggregation_time: "00:30"  # Daily at 00:30
```

-----

## Success Metrics

|Metric                               |Target|
|-------------------------------------|------|
|Posts scraped per day                |500+  |
|Sentiment accuracy (vs manual labels)|>75%  |
|Ticker detection precision           |>90%  |
|API response time                    |<200ms|
|System uptime                        |>99%  |

-----

## Risks & Mitigations

|Risk                          |Impact|Mitigation                                                        |
|------------------------------|------|------------------------------------------------------------------|
|Bankier blocks scraping       |High  |Rotate IPs, respect rate limits, use residential proxies if needed|
|HTML structure changes        |Medium|Abstract selectors, add monitoring for parse failures             |
|HerBERT poor on financial text|Medium|Fine-tune on labeled financial data, maintain keyword lexicon     |
|Database grows too large      |Low   |Archive old raw HTML, keep only processed data                    |

-----

## Future Enhancements

1. **Multi-source aggregation** - Add StockWatch.pl, Pair Biznesu, Twitter/X
1. **Real-time streaming** - WebSocket API for live sentiment updates
1. **ML signal generation** - Train classifier on sentiment → price movement
1. **Company-specific models** - Fine-tune sentiment models per sector
1. **Alert system** - Notify on unusual activity or sentiment shifts
