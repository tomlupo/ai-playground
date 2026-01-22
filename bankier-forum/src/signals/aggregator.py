"""Sentiment aggregation and signal generation."""

import logging
import statistics
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import DailySentiment, Post, PostSentiment, TickerMention

logger = logging.getLogger(__name__)


class SentimentAggregator:
    """
    Aggregates post-level sentiment into daily ticker-level signals.

    Calculates metrics like:
    - Average sentiment per ticker
    - Sentiment volatility (stddev)
    - Bullish/bearish ratios
    - Unique author counts
    - Momentum indicators
    """

    def __init__(
        self,
        bullish_threshold: float = 0.2,
        bearish_threshold: float = -0.2,
    ):
        """
        Initialize aggregator.

        Args:
            bullish_threshold: Score threshold for bullish classification
            bearish_threshold: Score threshold for bearish classification
        """
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold

    def aggregate_daily(self, session: Session, target_date: date) -> dict[str, dict]:
        """
        Aggregate sentiment by ticker for a given date.

        Args:
            session: Database session
            target_date: Date to aggregate

        Returns:
            Dictionary mapping ticker to metrics
        """
        # Query posts with sentiment and ticker mentions for the target date
        stmt = (
            select(
                TickerMention.ticker,
                PostSentiment.sentiment_score,
                Post.author,
                Post.created_at,
            )
            .join(Post, TickerMention.post_id == Post.id)
            .join(PostSentiment, Post.id == PostSentiment.post_id)
            .where(func.date(Post.created_at) == target_date)
        )

        rows = session.execute(stmt).fetchall()

        # Group by ticker
        ticker_data: dict[str, dict] = {}
        for ticker, score, author, created_at in rows:
            if ticker not in ticker_data:
                ticker_data[ticker] = {
                    "scores": [],
                    "authors": set(),
                }
            ticker_data[ticker]["scores"].append(score)
            ticker_data[ticker]["authors"].add(author)

        # Calculate metrics
        results: dict[str, dict] = {}
        for ticker, data in ticker_data.items():
            scores = data["scores"]
            if not scores:
                continue

            avg_sentiment = statistics.mean(scores)
            sentiment_stddev = statistics.stdev(scores) if len(scores) > 1 else 0.0
            bullish_ratio = sum(1 for s in scores if s > self.bullish_threshold) / len(scores)
            bearish_ratio = sum(1 for s in scores if s < self.bearish_threshold) / len(scores)

            results[ticker] = {
                "date": target_date,
                "ticker": ticker,
                "post_count": len(scores),
                "avg_sentiment": round(avg_sentiment, 4),
                "sentiment_stddev": round(sentiment_stddev, 4),
                "bullish_ratio": round(bullish_ratio, 4),
                "bearish_ratio": round(bearish_ratio, 4),
                "unique_authors": len(data["authors"]),
            }

        logger.info(f"Aggregated {len(results)} tickers for {target_date}")
        return results

    def save_daily_aggregations(
        self, session: Session, aggregations: dict[str, dict]
    ) -> list[DailySentiment]:
        """
        Save daily aggregations to database.

        Args:
            session: Database session
            aggregations: Output from aggregate_daily()

        Returns:
            List of saved DailySentiment records
        """
        saved = []
        for ticker, metrics in aggregations.items():
            stmt = select(DailySentiment).where(
                DailySentiment.date == metrics["date"],
                DailySentiment.ticker == ticker,
            )
            existing = session.execute(stmt).scalar_one_or_none()

            if existing:
                existing.post_count = metrics["post_count"]
                existing.avg_sentiment = metrics["avg_sentiment"]
                existing.sentiment_stddev = metrics["sentiment_stddev"]
                existing.bullish_ratio = metrics["bullish_ratio"]
                existing.bearish_ratio = metrics["bearish_ratio"]
                existing.unique_authors = metrics["unique_authors"]
                saved.append(existing)
            else:
                daily = DailySentiment(
                    date=metrics["date"],
                    ticker=ticker,
                    post_count=metrics["post_count"],
                    avg_sentiment=metrics["avg_sentiment"],
                    sentiment_stddev=metrics["sentiment_stddev"],
                    bullish_ratio=metrics["bullish_ratio"],
                    bearish_ratio=metrics["bearish_ratio"],
                    unique_authors=metrics["unique_authors"],
                )
                session.add(daily)
                saved.append(daily)

        session.commit()
        return saved

    def calculate_momentum(
        self, session: Session, ticker: str, days: int = 7
    ) -> Optional[float]:
        """
        Calculate sentiment momentum (change over period).

        Args:
            session: Database session
            ticker: Ticker symbol
            days: Lookback period

        Returns:
            Momentum value (recent - older sentiment), or None if insufficient data
        """
        stmt = (
            select(DailySentiment.date, DailySentiment.avg_sentiment)
            .where(DailySentiment.ticker == ticker)
            .order_by(DailySentiment.date.desc())
            .limit(days)
        )
        rows = session.execute(stmt).fetchall()

        if len(rows) < 2:
            return None

        recent = rows[0][1]
        older = rows[-1][1]
        return round(recent - older, 4)

    def get_unusual_activity(
        self,
        session: Session,
        target_date: date,
        lookback_days: int = 30,
        threshold: float = 2.0,
    ) -> list[dict]:
        """
        Detect tickers with unusual posting activity (z-score based).

        Args:
            session: Database session
            target_date: Date to check
            lookback_days: Days to use for baseline calculation
            threshold: Z-score threshold for "unusual"

        Returns:
            List of dicts with unusual activity info
        """
        start_date = target_date - timedelta(days=lookback_days)

        # Get historical stats per ticker
        historical_stmt = (
            select(
                DailySentiment.ticker,
                func.avg(DailySentiment.post_count).label("avg_posts"),
                func.stddev(DailySentiment.post_count).label("std_posts"),
            )
            .where(
                DailySentiment.date >= start_date,
                DailySentiment.date < target_date,
            )
            .group_by(DailySentiment.ticker)
        )
        historical = {row[0]: (row[1], row[2]) for row in session.execute(historical_stmt)}

        # Get current day stats
        current_stmt = select(DailySentiment).where(DailySentiment.date == target_date)
        current = session.execute(current_stmt).scalars().all()

        unusual = []
        for record in current:
            if record.ticker not in historical:
                continue

            avg_posts, std_posts = historical[record.ticker]
            if std_posts is None or std_posts == 0:
                continue

            z_score = (record.post_count - avg_posts) / std_posts
            if abs(z_score) > threshold:
                unusual.append({
                    "ticker": record.ticker,
                    "date": target_date,
                    "post_count": record.post_count,
                    "avg_posts": round(avg_posts, 2),
                    "z_score": round(z_score, 2),
                    "direction": "high" if z_score > 0 else "low",
                })

        return sorted(unusual, key=lambda x: abs(x["z_score"]), reverse=True)

    def get_top_movers(
        self,
        session: Session,
        target_date: date,
        min_posts: int = 5,
        limit: int = 10,
    ) -> dict[str, list[dict]]:
        """
        Get top bullish and bearish tickers for a date.

        Args:
            session: Database session
            target_date: Date to check
            min_posts: Minimum posts for inclusion
            limit: Number of results per direction

        Returns:
            Dict with 'bullish' and 'bearish' lists
        """
        stmt = (
            select(DailySentiment)
            .where(
                DailySentiment.date == target_date,
                DailySentiment.post_count >= min_posts,
            )
            .order_by(DailySentiment.avg_sentiment.desc())
        )
        all_records = list(session.execute(stmt).scalars())

        def to_dict(r: DailySentiment) -> dict:
            return {
                "ticker": r.ticker,
                "avg_sentiment": r.avg_sentiment,
                "post_count": r.post_count,
                "bullish_ratio": r.bullish_ratio,
                "bearish_ratio": r.bearish_ratio,
            }

        bullish = [to_dict(r) for r in all_records[:limit]]
        bearish = [to_dict(r) for r in reversed(all_records[-limit:])]

        return {"bullish": bullish, "bearish": bearish}
