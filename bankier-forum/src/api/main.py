"""FastAPI service for Bankier sentiment data."""

import csv
import io
from datetime import date, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..db.repository import Repository
from ..signals.aggregator import SentimentAggregator

app = FastAPI(
    title="Bankier Sentiment API",
    description="API for accessing Polish equity sentiment data from Bankier.pl forum",
    version="0.1.0",
)

# Global repository (initialized on startup)
repo: Optional[Repository] = None
aggregator = SentimentAggregator()


class DailySentimentResponse(BaseModel):
    """Daily sentiment response model."""

    date: date
    ticker: str
    post_count: int
    avg_sentiment: float
    sentiment_stddev: float
    bullish_ratio: float
    bearish_ratio: float
    unique_authors: int


class TickerSentimentResponse(BaseModel):
    """Ticker sentiment response model."""

    ticker: str
    data: list[DailySentimentResponse]
    momentum_7d: Optional[float] = None


class UnusualActivityResponse(BaseModel):
    """Unusual activity response model."""

    ticker: str
    date: date
    post_count: int
    avg_posts: float
    z_score: float
    direction: str


class TopMoversResponse(BaseModel):
    """Top movers response model."""

    bullish: list[dict]
    bearish: list[dict]


@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup."""
    global repo
    repo = Repository()
    repo.create_tables()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/sentiment/{ticker}", response_model=TickerSentimentResponse)
async def get_ticker_sentiment(
    ticker: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
):
    """
    Get historical sentiment data for a ticker.

    Args:
        ticker: Ticker symbol (e.g., CDR, PKO)
        start_date: Start date (default: 30 days ago)
        end_date: End date (default: today)
    """
    if repo is None:
        raise HTTPException(status_code=500, detail="Database not initialized")

    ticker = ticker.upper()

    if not start_date:
        start_date = date.today() - timedelta(days=30)
    if not end_date:
        end_date = date.today()

    with repo.get_session() as session:
        data = repo.get_daily_sentiment(session, ticker, start_date, end_date)

        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"No sentiment data found for ticker {ticker}",
            )

        # Calculate momentum
        momentum = aggregator.calculate_momentum(session, ticker, days=7)

        return TickerSentimentResponse(
            ticker=ticker,
            data=[
                DailySentimentResponse(
                    date=d.date,
                    ticker=d.ticker,
                    post_count=d.post_count,
                    avg_sentiment=d.avg_sentiment,
                    sentiment_stddev=d.sentiment_stddev,
                    bullish_ratio=d.bullish_ratio,
                    bearish_ratio=d.bearish_ratio,
                    unique_authors=d.unique_authors,
                )
                for d in data
            ],
            momentum_7d=momentum,
        )


@app.get("/signals/unusual", response_model=list[UnusualActivityResponse])
async def get_unusual_activity(
    target_date: Optional[date] = None,
    threshold: float = 2.0,
):
    """
    Get tickers with unusual forum activity.

    Args:
        target_date: Date to check (default: today)
        threshold: Z-score threshold for "unusual" (default: 2.0)
    """
    if repo is None:
        raise HTTPException(status_code=500, detail="Database not initialized")

    if not target_date:
        target_date = date.today()

    with repo.get_session() as session:
        unusual = aggregator.get_unusual_activity(session, target_date, threshold=threshold)

        return [
            UnusualActivityResponse(
                ticker=u["ticker"],
                date=u["date"],
                post_count=u["post_count"],
                avg_posts=u["avg_posts"],
                z_score=u["z_score"],
                direction=u["direction"],
            )
            for u in unusual
        ]


@app.get("/signals/top-movers", response_model=TopMoversResponse)
async def get_top_movers(
    target_date: Optional[date] = None,
    min_posts: int = 5,
    limit: int = 10,
):
    """
    Get top bullish and bearish tickers.

    Args:
        target_date: Date to check (default: today)
        min_posts: Minimum post count for inclusion
        limit: Number of results per direction
    """
    if repo is None:
        raise HTTPException(status_code=500, detail="Database not initialized")

    if not target_date:
        target_date = date.today()

    with repo.get_session() as session:
        return aggregator.get_top_movers(session, target_date, min_posts, limit)


@app.get("/tickers")
async def list_tickers():
    """Get list of all tickers with sentiment data."""
    if repo is None:
        raise HTTPException(status_code=500, detail="Database not initialized")

    with repo.get_session() as session:
        tickers = repo.get_all_tickers(session)
        return {"tickers": sorted(tickers), "count": len(tickers)}


@app.get("/export/csv")
async def export_csv(
    start_date: date,
    end_date: date,
    tickers: Optional[list[str]] = Query(None),
):
    """
    Export sentiment data as CSV.

    Args:
        start_date: Start date for export
        end_date: End date for export
        tickers: Optional list of tickers to filter
    """
    if repo is None:
        raise HTTPException(status_code=500, detail="Database not initialized")

    with repo.get_session() as session:
        # Get all tickers if not specified
        if not tickers:
            tickers = repo.get_all_tickers(session)

        # Collect data
        all_data = []
        for ticker in tickers:
            data = repo.get_daily_sentiment(session, ticker.upper(), start_date, end_date)
            all_data.extend(data)

        if not all_data:
            raise HTTPException(status_code=404, detail="No data found for the specified criteria")

        # Generate CSV
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "date",
            "ticker",
            "post_count",
            "avg_sentiment",
            "sentiment_stddev",
            "bullish_ratio",
            "bearish_ratio",
            "unique_authors",
        ])

        # Data rows
        for d in sorted(all_data, key=lambda x: (x.date, x.ticker)):
            writer.writerow([
                d.date.isoformat(),
                d.ticker,
                d.post_count,
                d.avg_sentiment,
                d.sentiment_stddev,
                d.bullish_ratio,
                d.bearish_ratio,
                d.unique_authors,
            ])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=sentiment_{start_date}_{end_date}.csv"
            },
        )
