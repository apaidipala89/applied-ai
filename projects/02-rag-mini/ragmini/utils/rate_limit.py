import time
import random
import functools
import asyncio
import logging
from typing import Callable, Any, Iterable, Type, Optional, Tuple
from openai import RateLimitError, APIError

def with_backoff(
    retries: int = 5,
    backoff_in_seconds: float = 1.0,
    exceptions: Iterable[Type[BaseException]] = (RateLimitError, APIError),
    jitter: bool = True,
    max_delay: Optional[float] = 60.0,
    logger: Optional[logging.Logger] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory that retries the wrapped function with exponential backoff.

    Usage:
      @with_backoff(retries=4, backoff_in_seconds=1.0)
      def call_api(...): ...

    Notes:
      - `retries` is the total number of attempts (including the first).
      - Works for both sync and async functions.
      - `exceptions` may be a tuple or any iterable of exception types.
    """
    if not isinstance(exceptions, tuple):
        exceptions = tuple(exceptions)
    logger = logger or logging.getLogger(__name__)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                delay = backoff_in_seconds
                for attempt in range(retries):
                    try:
                        return await fn(*args, **kwargs)
                    except exceptions as e:
                        logger.debug("backoff async attempt %d failed: %s", attempt + 1, e)
                        if attempt == retries - 1:
                            raise
                        sleep_time = delay * (1 + random.random() if jitter else 1)
                        if max_delay is not None:
                            sleep_time = min(sleep_time, max_delay)
                        await asyncio.sleep(sleep_time)
                        delay = min(max_delay or delay * 2, delay * 2)
            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            delay = backoff_in_seconds
            for attempt in range(retries):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    logger.debug("backoff sync attempt %d failed: %s", attempt + 1, e)
                    if attempt == retries - 1:
                        raise
                    sleep_time = delay * (1 + random.random() if jitter else 1)
                    if max_delay is not None:
                        sleep_time = min(sleep_time, max_delay)
                    time.sleep(sleep_time)
                    delay = min(max_delay or delay * 2, delay * 2)
        return sync_wrapper

    return decorator