"""Bankier.pl forum scraper."""

import asyncio
import logging
import random
import re
from datetime import datetime
from typing import AsyncGenerator, Optional
from dataclasses import dataclass

import httpx
from selectolax.parser import HTMLParser

from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class ScrapedThread:
    """Scraped thread info."""

    thread_id: int
    slug: str
    title: str
    url: str
    post_count: Optional[int] = None
    last_activity: Optional[str] = None


@dataclass
class ScrapedPost:
    """Scraped post info."""

    post_id: int
    thread_id: int
    author: str
    author_ip_fragment: Optional[str]
    content: str
    created_at: datetime
    votes_up: int = 0
    votes_down: int = 0
    is_op: bool = False


DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


class BankierForumScraper:
    """Async scraper for Bankier.pl forum."""

    BASE_URL = "https://www.bankier.pl"

    # Forum IDs from PRD
    FORUMS = {
        6: "gielda",
        50: "jak-grac-na-gieldzie",
        55: "kryptowaluty",
        7: "forex",
        59: "etf",
    }

    def __init__(
        self,
        requests_per_second: float = 0.5,
        retry_backoff: list[int] | None = None,
        user_agents: list[str] | None = None,
    ):
        """
        Initialize the scraper.

        Args:
            requests_per_second: Rate limit (default 0.5 = 1 req per 2 sec)
            retry_backoff: Retry delays in seconds (default [5, 15, 60])
            user_agents: List of user agents to rotate
        """
        self.rate_limiter = RateLimiter(requests_per_second)
        self.retry_backoff = retry_backoff or [5, 15, 60]
        self.user_agents = user_agents or DEFAULT_USER_AGENTS
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "BankierForumScraper":
        """Async context manager entry."""
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": random.choice(self.user_agents),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "pl,en;q=0.9",
            },
        )
        return self

    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    async def _fetch_page(self, url: str) -> str:
        """
        Fetch a page with rate limiting and retries.

        Args:
            url: URL to fetch

        Returns:
            HTML content as string

        Raises:
            httpx.HTTPError: If all retries fail
        """
        if not self.client:
            raise RuntimeError("Scraper not initialized. Use async with context.")

        await self.rate_limiter.acquire()

        # Rotate user agent
        self.client.headers["User-Agent"] = random.choice(self.user_agents)

        last_error: Optional[Exception] = None
        for attempt, backoff in enumerate(self.retry_backoff):
            try:
                response = await self.client.get(url)
                response.raise_for_status()
                logger.debug(f"Fetched {url} (attempt {attempt + 1})")
                return response.text
            except httpx.HTTPError as e:
                last_error = e
                logger.warning(f"Fetch failed for {url}: {e}. Retrying in {backoff}s...")
                if attempt < len(self.retry_backoff) - 1:
                    await asyncio.sleep(backoff)

        raise last_error or RuntimeError("Unknown error during fetch")

    async def scrape_forum_page(self, forum_id: int, page: int = 0) -> list[ScrapedThread]:
        """
        Scrape thread listing from a forum page.

        Args:
            forum_id: Forum ID (e.g., 6 for Giełda)
            page: Page number (0-indexed)

        Returns:
            List of thread info
        """
        forum_slug = self.FORUMS.get(forum_id, "gielda")
        url = f"{self.BASE_URL}/forum/forum_{forum_slug},{forum_id},{page}.html"

        html = await self._fetch_page(url)
        tree = HTMLParser(html)

        threads: list[ScrapedThread] = []

        # Find thread links
        for link in tree.css("a[href*='/forum/temat_']"):
            href = link.attributes.get("href", "")
            match = re.search(r"temat_([^,]+),(\d+)\.html", href)
            if match:
                thread = ScrapedThread(
                    slug=match.group(1),
                    thread_id=int(match.group(2)),
                    title=link.text(strip=True),
                    url=f"{self.BASE_URL}{href}" if href.startswith("/") else href,
                )
                # Avoid duplicates
                if not any(t.thread_id == thread.thread_id for t in threads):
                    threads.append(thread)

        logger.info(f"Found {len(threads)} threads on forum {forum_id} page {page}")
        return threads

    async def scrape_thread_posts(self, thread_id: int, max_pages: int = 10) -> AsyncGenerator[ScrapedPost, None]:
        """
        Scrape all posts from a thread.

        Args:
            thread_id: Thread ID to scrape
            max_pages: Maximum pages to scrape

        Yields:
            ScrapedPost objects
        """
        page = 1
        while page <= max_pages:
            url = f"{self.BASE_URL}/forum/pokaz-tresc?thread_id={thread_id}&strona={page}"

            try:
                html = await self._fetch_page(url)
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch thread {thread_id} page {page}: {e}")
                break

            tree = HTMLParser(html)
            posts = self._parse_posts(tree, thread_id)

            if not posts:
                logger.debug(f"No posts found on thread {thread_id} page {page}")
                break

            for post in posts:
                yield post

            # Check for next page
            if not self._has_next_page(tree):
                break

            page += 1

    def _parse_posts(self, tree: HTMLParser, thread_id: int) -> list[ScrapedPost]:
        """
        Parse posts from thread HTML.

        Args:
            tree: Parsed HTML tree
            thread_id: Thread ID for reference

        Returns:
            List of parsed posts
        """
        posts: list[ScrapedPost] = []
        post_id_counter = 0  # Fallback counter if we can't extract post ID

        # Try multiple selectors for post containers
        post_containers = tree.css("li") or tree.css("div.post") or tree.css("article")

        for block in post_containers:
            text = block.text() or ""

            # Extract author: patterns like "Autor: ~nickname [IP]" or "Autor: username"
            author_match = re.search(r"Autor:\s*([~\w]+)(?:\s*\[([^\]]+)\])?", text)
            if not author_match:
                continue

            author = author_match.group(1)
            author_ip = author_match.group(2) if author_match.lastindex >= 2 else None

            # Extract timestamp: "2017-01-23 18:47" format
            time_match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})", text)
            if not time_match:
                continue

            try:
                created_at = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M")
            except ValueError:
                continue

            # Try to extract post ID from reply link
            post_id = None
            post_link = block.css_first("a[href*='parent_id=']")
            if post_link:
                id_match = re.search(r"parent_id=(\d+)", post_link.attributes.get("href", ""))
                if id_match:
                    post_id = int(id_match.group(1))

            # Fallback: generate ID from thread + counter
            if post_id is None:
                post_id_counter += 1
                post_id = thread_id * 10000 + post_id_counter

            # Extract content
            content = self._extract_content(block, text)
            if not content:
                continue

            # Extract votes if available
            votes_up, votes_down = self._extract_votes(text)

            posts.append(
                ScrapedPost(
                    post_id=post_id,
                    thread_id=thread_id,
                    author=author,
                    author_ip_fragment=author_ip,
                    content=content,
                    created_at=created_at,
                    votes_up=votes_up,
                    votes_down=votes_down,
                    is_op=(len(posts) == 0),  # First post is OP
                )
            )

        return posts

    def _extract_content(self, block: object, full_text: str) -> str:
        """
        Extract post content from HTML block.

        Args:
            block: HTML block element
            full_text: Full text of the block

        Returns:
            Cleaned content text
        """
        # Try to find specific content div
        content_el = getattr(block, 'css_first', lambda x: None)("div.post-content") or \
                     getattr(block, 'css_first', lambda x: None)("p")

        if content_el:
            text = content_el.text(strip=True)
        else:
            text = full_text

        # Clean up: remove author line, timestamps, action links
        text = re.sub(r"Autor:\s*[~\w]+(?:\s*\[[^\]]+\])?", "", text)
        text = re.sub(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", "", text)
        text = re.sub(r"Odpowiedz|Zgłoś do moderatora|Cytuj", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _extract_votes(self, text: str) -> tuple[int, int]:
        """
        Extract vote counts from text.

        Args:
            text: Text containing vote info

        Returns:
            Tuple of (votes_up, votes_down)
        """
        # Try to find vote patterns
        vote_match = re.search(r"(\d+)\s*[+↑]\s*(\d+)\s*[-↓]", text)
        if vote_match:
            return int(vote_match.group(1)), int(vote_match.group(2))
        return 0, 0

    def _has_next_page(self, tree: HTMLParser) -> bool:
        """
        Check if there's a next page.

        Args:
            tree: Parsed HTML tree

        Returns:
            True if next page exists
        """
        pagination = tree.css("a[href*='strona=']")
        return len(pagination) > 1

    async def scrape_latest_posts(
        self, forum_id: int = 6, num_pages: int = 5, posts_per_thread: int = 10
    ) -> AsyncGenerator[ScrapedPost, None]:
        """
        Scrape latest posts from a forum.

        Args:
            forum_id: Forum ID to scrape
            num_pages: Number of forum pages to scrape
            posts_per_thread: Max posts to get per thread

        Yields:
            ScrapedPost objects
        """
        seen_thread_ids: set[int] = set()

        for page in range(num_pages):
            threads = await self.scrape_forum_page(forum_id, page)

            for thread in threads:
                if thread.thread_id in seen_thread_ids:
                    continue
                seen_thread_ids.add(thread.thread_id)

                post_count = 0
                async for post in self.scrape_thread_posts(thread.thread_id, max_pages=1):
                    yield post
                    post_count += 1
                    if post_count >= posts_per_thread:
                        break
