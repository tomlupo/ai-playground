#!/usr/bin/env python3
"""
Main scraper script for Bankier.pl forum sentiment pipeline.

This script orchestrates the full pipeline:
1. Scrape forum posts
2. Analyze sentiment
3. Detect ticker mentions
4. Aggregate daily signals

Usage:
    python -m scripts.run_scraper [OPTIONS]

Options:
    --forum-id INT       Forum ID to scrape (default: 6 = Giełda)
    --pages INT          Number of pages to scrape (default: 5)
    --analyze            Run sentiment analysis on new posts
    --aggregate          Run daily aggregation
    --date DATE          Target date for aggregation (default: today)
"""

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.repository import Repository
from src.nlp.sentiment import PolishFinancialSentiment
from src.nlp.ticker_detector import WSETickerDetector
from src.scraper.bankier_scraper import BankierForumScraper
from src.signals.aggregator import SentimentAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def scrape_forum(
    repo: Repository,
    forum_id: int = 6,
    num_pages: int = 5,
    posts_per_thread: int = 10,
) -> int:
    """
    Scrape forum posts and store in database.

    Returns:
        Number of posts scraped
    """
    logger.info(f"Starting scrape of forum {forum_id} ({num_pages} pages)")
    post_count = 0

    async with BankierForumScraper() as scraper:
        with repo.get_session() as session:
            # Ensure forum exists
            repo.upsert_forum(
                session,
                forum_id,
                scraper.FORUMS.get(forum_id, "unknown"),
                scraper.FORUMS.get(forum_id, "unknown"),
                f"/forum/forum_{scraper.FORUMS.get(forum_id, 'gielda')},{forum_id},{{page}}.html",
            )
            session.commit()

            async for post in scraper.scrape_latest_posts(
                forum_id, num_pages, posts_per_thread
            ):
                # Upsert thread
                repo.upsert_thread(
                    session,
                    thread_id=post.thread_id,
                    forum_id=forum_id,
                    title=f"Thread {post.thread_id}",  # We don't have title from posts
                )

                # Upsert post
                repo.upsert_post(
                    session,
                    post_id=post.post_id,
                    thread_id=post.thread_id,
                    author=post.author,
                    content=post.content,
                    created_at=post.created_at,
                    author_ip_fragment=post.author_ip_fragment,
                    votes_up=post.votes_up,
                    votes_down=post.votes_down,
                    is_op=post.is_op,
                )

                post_count += 1
                if post_count % 10 == 0:
                    session.commit()
                    logger.info(f"Scraped {post_count} posts")

            session.commit()

    logger.info(f"Scraping complete. Total posts: {post_count}")
    return post_count


def analyze_sentiment(repo: Repository, batch_size: int = 100) -> int:
    """
    Analyze sentiment for posts without sentiment scores.

    Returns:
        Number of posts analyzed
    """
    logger.info("Starting sentiment analysis")

    sentiment_analyzer = PolishFinancialSentiment(use_model=False)  # Use lexicon for PoC
    ticker_detector = WSETickerDetector()

    analyzed_count = 0

    with repo.get_session() as session:
        while True:
            # Get unanalyzed posts
            posts = repo.get_posts_without_sentiment(session, limit=batch_size)
            if not posts:
                break

            for post in posts:
                # Analyze sentiment
                score, label, confidence = sentiment_analyzer.analyze(post.content)

                # Save sentiment
                repo.save_sentiment(
                    session,
                    post_id=post.id,
                    score=score,
                    label=label,
                    confidence=confidence,
                    model_version=sentiment_analyzer.model_version,
                )

                # Detect and save ticker mentions
                mentions = ticker_detector.detect(post.content)
                for ticker, company_name, mention_type in mentions:
                    context = ticker_detector.get_context(post.content, ticker)
                    repo.save_ticker_mention(
                        session,
                        post_id=post.id,
                        ticker=ticker,
                        company_name=company_name,
                        mention_type=mention_type,
                        context_snippet=context,
                    )

                analyzed_count += 1

            session.commit()
            logger.info(f"Analyzed {analyzed_count} posts")

    logger.info(f"Sentiment analysis complete. Total: {analyzed_count}")
    return analyzed_count


def aggregate_daily(repo: Repository, target_date: date) -> int:
    """
    Run daily aggregation for a specific date.

    Returns:
        Number of tickers aggregated
    """
    logger.info(f"Running daily aggregation for {target_date}")

    aggregator = SentimentAggregator()

    with repo.get_session() as session:
        aggregations = aggregator.aggregate_daily(session, target_date)

        if aggregations:
            aggregator.save_daily_aggregations(session, aggregations)
            logger.info(f"Saved {len(aggregations)} daily aggregations")
        else:
            logger.info("No data to aggregate for this date")

    return len(aggregations) if aggregations else 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bankier.pl forum sentiment scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--forum-id",
        type=int,
        default=6,
        help="Forum ID to scrape (default: 6 = Giełda)",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=5,
        help="Number of pages to scrape (default: 5)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run sentiment analysis on new posts",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Run daily aggregation",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date for aggregation (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default="sqlite:///data/bankier_sentiment.db",
        help="Database URL",
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Run the scraper",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full pipeline (scrape, analyze, aggregate)",
    )

    args = parser.parse_args()

    # Parse target date
    target_date = date.today()
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    # Initialize repository
    repo = Repository(args.db_url)
    repo.create_tables()
    logger.info(f"Database initialized: {args.db_url}")

    # Determine what to run
    run_scrape = args.scrape or args.all
    run_analyze = args.analyze or args.all
    run_aggregate = args.aggregate or args.all

    # If nothing specified, show help
    if not (run_scrape or run_analyze or run_aggregate):
        parser.print_help()
        print("\n\nExample usage:")
        print("  python -m scripts.run_scraper --all          # Run full pipeline")
        print("  python -m scripts.run_scraper --scrape       # Scrape only")
        print("  python -m scripts.run_scraper --analyze      # Analyze sentiment only")
        print("  python -m scripts.run_scraper --aggregate    # Aggregate only")
        return

    # Run pipeline
    if run_scrape:
        post_count = asyncio.run(
            scrape_forum(repo, args.forum_id, args.pages)
        )
        print(f"Scraped {post_count} posts")

    if run_analyze:
        analyzed_count = analyze_sentiment(repo)
        print(f"Analyzed {analyzed_count} posts")

    if run_aggregate:
        ticker_count = aggregate_daily(repo, target_date)
        print(f"Aggregated {ticker_count} tickers for {target_date}")

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
