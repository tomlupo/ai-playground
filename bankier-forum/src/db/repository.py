"""Database repository for data access."""

from datetime import date, datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    Base,
    DailySentiment,
    Forum,
    Post,
    PostSentiment,
    Thread,
    TickerMention,
)


class Repository:
    """Repository for database operations."""

    def __init__(self, db_url: str = "sqlite:///data/bankier_sentiment.db"):
        """Initialize the repository with database URL."""
        # Ensure data directory exists for SQLite
        if db_url.startswith("sqlite:///"):
            db_path = Path(db_url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    # Forum operations
    def upsert_forum(self, session: Session, forum_id: int, name: str, slug: str, url_pattern: str) -> Forum:
        """Insert or update a forum."""
        forum = session.get(Forum, forum_id)
        if forum:
            forum.name = name
            forum.slug = slug
            forum.url_pattern = url_pattern
        else:
            forum = Forum(id=forum_id, name=name, slug=slug, url_pattern=url_pattern)
            session.add(forum)
        return forum

    # Thread operations
    def upsert_thread(
        self,
        session: Session,
        thread_id: int,
        forum_id: int,
        title: str,
        slug: Optional[str] = None,
        author: Optional[str] = None,
        created_at: Optional[datetime] = None,
        post_count: int = 0,
        view_count: int = 0,
    ) -> Thread:
        """Insert or update a thread."""
        thread = session.get(Thread, thread_id)
        if thread:
            thread.title = title
            thread.slug = slug
            thread.post_count = post_count
            thread.view_count = view_count
            thread.last_scraped_at = datetime.utcnow()
        else:
            thread = Thread(
                id=thread_id,
                forum_id=forum_id,
                title=title,
                slug=slug,
                author=author,
                created_at=created_at,
                post_count=post_count,
                view_count=view_count,
                last_scraped_at=datetime.utcnow(),
            )
            session.add(thread)
        return thread

    # Post operations
    def upsert_post(
        self,
        session: Session,
        post_id: int,
        thread_id: int,
        author: str,
        content: str,
        created_at: datetime,
        author_ip_fragment: Optional[str] = None,
        votes_up: int = 0,
        votes_down: int = 0,
        is_op: bool = False,
    ) -> Post:
        """Insert or update a post."""
        post = session.get(Post, post_id)
        if post:
            post.content = content
            post.votes_up = votes_up
            post.votes_down = votes_down
            post.scraped_at = datetime.utcnow()
        else:
            post = Post(
                id=post_id,
                thread_id=thread_id,
                author=author,
                author_ip_fragment=author_ip_fragment,
                content=content,
                created_at=created_at,
                votes_up=votes_up,
                votes_down=votes_down,
                is_op=is_op,
            )
            session.add(post)
        return post

    def get_posts_without_sentiment(self, session: Session, limit: int = 100) -> list[Post]:
        """Get posts that haven't been analyzed for sentiment."""
        stmt = (
            select(Post)
            .outerjoin(PostSentiment)
            .where(PostSentiment.post_id.is_(None))
            .limit(limit)
        )
        return list(session.execute(stmt).scalars())

    # Sentiment operations
    def save_sentiment(
        self,
        session: Session,
        post_id: int,
        score: float,
        label: str,
        confidence: float,
        model_version: str,
    ) -> PostSentiment:
        """Save sentiment analysis result."""
        sentiment = session.get(PostSentiment, post_id)
        if sentiment:
            sentiment.sentiment_score = score
            sentiment.sentiment_label = label
            sentiment.confidence = confidence
            sentiment.model_version = model_version
            sentiment.analyzed_at = datetime.utcnow()
        else:
            sentiment = PostSentiment(
                post_id=post_id,
                sentiment_score=score,
                sentiment_label=label,
                confidence=confidence,
                model_version=model_version,
            )
            session.add(sentiment)
        return sentiment

    # Ticker mention operations
    def save_ticker_mention(
        self,
        session: Session,
        post_id: int,
        ticker: str,
        company_name: Optional[str] = None,
        mention_type: str = "explicit",
        context_snippet: Optional[str] = None,
    ) -> TickerMention:
        """Save a ticker mention."""
        # Check if exists
        stmt = select(TickerMention).where(
            TickerMention.post_id == post_id, TickerMention.ticker == ticker
        )
        existing = session.execute(stmt).scalar_one_or_none()

        if existing:
            existing.company_name = company_name
            existing.mention_type = mention_type
            existing.context_snippet = context_snippet
            return existing

        mention = TickerMention(
            post_id=post_id,
            ticker=ticker,
            company_name=company_name,
            mention_type=mention_type,
            context_snippet=context_snippet,
        )
        session.add(mention)
        return mention

    # Daily sentiment operations
    def save_daily_sentiment(
        self,
        session: Session,
        target_date: date,
        ticker: str,
        post_count: int,
        avg_sentiment: float,
        sentiment_stddev: float,
        bullish_ratio: float,
        bearish_ratio: float,
        unique_authors: int,
    ) -> DailySentiment:
        """Save or update daily sentiment aggregation."""
        stmt = select(DailySentiment).where(
            DailySentiment.date == target_date, DailySentiment.ticker == ticker
        )
        daily = session.execute(stmt).scalar_one_or_none()

        if daily:
            daily.post_count = post_count
            daily.avg_sentiment = avg_sentiment
            daily.sentiment_stddev = sentiment_stddev
            daily.bullish_ratio = bullish_ratio
            daily.bearish_ratio = bearish_ratio
            daily.unique_authors = unique_authors
        else:
            daily = DailySentiment(
                date=target_date,
                ticker=ticker,
                post_count=post_count,
                avg_sentiment=avg_sentiment,
                sentiment_stddev=sentiment_stddev,
                bullish_ratio=bullish_ratio,
                bearish_ratio=bearish_ratio,
                unique_authors=unique_authors,
            )
            session.add(daily)
        return daily

    def get_daily_sentiment(
        self, session: Session, ticker: str, start_date: date, end_date: date
    ) -> list[DailySentiment]:
        """Get daily sentiment data for a ticker within date range."""
        stmt = (
            select(DailySentiment)
            .where(
                DailySentiment.ticker == ticker,
                DailySentiment.date >= start_date,
                DailySentiment.date <= end_date,
            )
            .order_by(DailySentiment.date)
        )
        return list(session.execute(stmt).scalars())

    def get_all_tickers(self, session: Session) -> list[str]:
        """Get list of all unique tickers."""
        stmt = select(TickerMention.ticker).distinct()
        return list(session.execute(stmt).scalars())
