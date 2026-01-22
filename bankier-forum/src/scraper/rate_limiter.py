"""Rate limiter for polite scraping."""

import asyncio
from collections import deque
from datetime import datetime
from typing import Optional


class RateLimiter:
    """Token bucket rate limiter with sliding window."""

    def __init__(self, requests_per_second: float = 0.5, burst_size: int = 1):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second (default 0.5 = 1 req per 2 sec)
            burst_size: Maximum burst of requests allowed
        """
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.burst_size = burst_size
        self._timestamps: deque[float] = deque(maxlen=100)
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request is allowed."""
        async with self._lock:
            now = datetime.now().timestamp()

            if self._timestamps:
                # Calculate required wait time
                elapsed = now - self._timestamps[-1]
                if elapsed < self.min_interval:
                    wait_time = self.min_interval - elapsed
                    await asyncio.sleep(wait_time)

            self._timestamps.append(datetime.now().timestamp())

    async def __aenter__(self) -> "RateLimiter":
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        """Async context manager exit."""
        pass

    @property
    def current_rate(self) -> float:
        """Calculate current request rate over the last minute."""
        if len(self._timestamps) < 2:
            return 0.0

        now = datetime.now().timestamp()
        recent = [ts for ts in self._timestamps if now - ts < 60]
        if len(recent) < 2:
            return 0.0

        duration = recent[-1] - recent[0]
        return (len(recent) - 1) / duration if duration > 0 else 0.0
