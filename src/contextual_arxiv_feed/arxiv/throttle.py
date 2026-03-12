"""Rate limiting and retry logic for arXiv API requests.

arXiv requires polite request rates: single connection, slow requests.
Default: 3 seconds between requests (configurable via ARXIV_THROTTLE_SECONDS).
"""

from __future__ import annotations

import logging
import time

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class ArxivThrottle:
    """Manages rate limiting for arXiv requests.

    Ensures single-connection behavior with configurable delay between requests.
    Uses exponential backoff on 429/5xx errors.
    """

    def __init__(self, throttle_seconds: int = 3):
        """Initialize throttle.

        Args:
            throttle_seconds: Minimum seconds between requests. Default 3.
        """
        self._throttle_seconds = throttle_seconds
        self._last_request_time: float = 0.0

    @property
    def throttle_seconds(self) -> int:
        """Get current throttle interval."""
        return self._throttle_seconds

    def sync_wait(self) -> None:
        """Synchronous version of wait for non-async contexts."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._throttle_seconds:
            wait_time = self._throttle_seconds - elapsed
            logger.debug(f"Throttling: waiting {wait_time:.2f}s before next request")
            time.sleep(wait_time)
        self._last_request_time = time.monotonic()


class RateLimitError(Exception):
    """Raised when rate limit (429) is hit."""

    pass


class ServerError(Exception):
    """Raised on 5xx server errors."""

    pass


def create_retry_decorator(max_attempts: int = 5):
    """Create a retry decorator for arXiv requests.

    Uses exponential backoff: 1s, 2s, 4s, 8s, 16s (max 60s).
    Retries on RateLimitError and ServerError.

    Args:
        max_attempts: Maximum retry attempts. Default 5.

    Returns:
        Configured retry decorator.
    """
    return retry(
        retry=retry_if_exception_type((RateLimitError, ServerError)),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        reraise=True,
    )


arxiv_retry = create_retry_decorator()


def check_response_status(status_code: int, url: str) -> None:
    """Check HTTP response status and raise appropriate errors.

    Args:
        status_code: HTTP status code.
        url: Request URL for error messages.

    Raises:
        RateLimitError: On 429 status.
        ServerError: On 5xx status.
    """
    if status_code == 429:
        logger.warning(f"Rate limited (429) for {url}")
        raise RateLimitError(f"Rate limited by arXiv: {url}")
    if 500 <= status_code < 600:
        logger.warning(f"Server error ({status_code}) for {url}")
        raise ServerError(f"arXiv server error {status_code}: {url}")
