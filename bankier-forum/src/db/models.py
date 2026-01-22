"""SQLAlchemy models for the Bankier sentiment database."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Forum(Base):
    """Forum category model."""

    __tablename__ = "forums"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(255), nullable=False)
    url_pattern = Column(String(500), nullable=False)

    threads = relationship("Thread", back_populates="forum")


class Thread(Base):
    """Forum thread model."""

    __tablename__ = "threads"

    id = Column(Integer, primary_key=True)  # bankier thread_id
    forum_id = Column(Integer, ForeignKey("forums.id"))
    title = Column(Text, nullable=False)
    slug = Column(String(500))
    author = Column(String(255))
    created_at = Column(DateTime)
    last_scraped_at = Column(DateTime)
    post_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0)

    forum = relationship("Forum", back_populates="threads")
    posts = relationship("Post", back_populates="thread")


class Post(Base):
    """Forum post model."""

    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)  # bankier post_id
    thread_id = Column(Integer, ForeignKey("threads.id"))
    author = Column(String(255), nullable=False)
    author_ip_fragment = Column(String(50))  # e.g., "149.156.96.*"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)
    votes_up = Column(Integer, default=0)
    votes_down = Column(Integer, default=0)
    is_op = Column(Boolean, default=False)
    scraped_at = Column(DateTime, default=datetime.utcnow)

    thread = relationship("Thread", back_populates="posts")
    sentiment = relationship("PostSentiment", back_populates="post", uselist=False)
    ticker_mentions = relationship("TickerMention", back_populates="post")

    __table_args__ = (
        Index("idx_posts_created_at", "created_at"),
        Index("idx_posts_thread_id", "thread_id"),
    )


class PostSentiment(Base):
    """Sentiment analysis result for a post."""

    __tablename__ = "post_sentiment"

    post_id = Column(Integer, ForeignKey("posts.id"), primary_key=True)
    sentiment_score = Column(Float)  # -1.0 to 1.0
    sentiment_label = Column(String(20))  # 'positive', 'negative', 'neutral'
    confidence = Column(Float)
    model_version = Column(String(100))
    analyzed_at = Column(DateTime, default=datetime.utcnow)

    post = relationship("Post", back_populates="sentiment")


class TickerMention(Base):
    """Ticker mention in a post."""

    __tablename__ = "ticker_mentions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(Integer, ForeignKey("posts.id"))
    ticker = Column(String(10), nullable=False)  # e.g., 'CDR', 'PKO', '11B'
    company_name = Column(String(255))
    mention_type = Column(String(20))  # 'explicit', 'inferred'
    context_snippet = Column(Text)  # 50 chars around mention

    post = relationship("Post", back_populates="ticker_mentions")

    __table_args__ = (
        UniqueConstraint("post_id", "ticker", name="uq_post_ticker"),
        Index("idx_ticker_mentions_ticker", "ticker"),
        Index("idx_ticker_mentions_post_id", "post_id"),
    )


class DailySentiment(Base):
    """Aggregated daily sentiment by ticker."""

    __tablename__ = "daily_sentiment"

    date = Column(Date, primary_key=True)
    ticker = Column(String(10), primary_key=True)
    post_count = Column(Integer)
    avg_sentiment = Column(Float)
    sentiment_stddev = Column(Float)
    bullish_ratio = Column(Float)  # % posts with sentiment > 0.2
    bearish_ratio = Column(Float)  # % posts with sentiment < -0.2
    unique_authors = Column(Integer)

    __table_args__ = (Index("idx_daily_sentiment_ticker", "ticker"),)
