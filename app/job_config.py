"""Configuration and retry classification for durable research jobs."""

import os

import httpx
import psycopg
from ddgs.exceptions import DDGSException


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least one.")
    return value


def max_job_attempts() -> int:
    return _positive_int("RESEARCH_JOB_MAX_ATTEMPTS", 3)


def job_lease_seconds() -> int:
    return _positive_int("RESEARCH_JOB_LEASE_SECONDS", 120)


def job_poll_seconds() -> int:
    return _positive_int("RESEARCH_JOB_POLL_SECONDS", 2)


def job_backoff_seconds(attempt_count: int) -> int:
    base = _positive_int("RESEARCH_JOB_BACKOFF_SECONDS", 5)
    maximum = _positive_int("RESEARCH_JOB_MAX_BACKOFF_SECONDS", 300)
    return min(base * (2 ** max(0, attempt_count - 1)), maximum)


def is_recoverable_error(error: Exception) -> bool:
    """Retry transport, timeout, and database-connection failures only."""

    seen: set[int] = set()
    current: BaseException | None = error

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (
                TimeoutError,
                ConnectionError,
                psycopg.OperationalError,
                psycopg.InterfaceError,
                httpx.TransportError,
                DDGSException,
            ),
        ):
            return True

        if isinstance(current, httpx.HTTPStatusError):
            if current.response.status_code in {408, 429} or (
                current.response.status_code >= 500
            ):
                return True

        current = current.__cause__ or current.__context__

    return False
