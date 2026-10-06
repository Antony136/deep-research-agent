"""
Web search tool for the Deep Research Agent.

This module provides a provider-independent interface for
searching the web and converts provider results into our
own SearchResult model.

The search layer also handles temporary provider failures
and adds a delay between requests so that search providers
are not hit with rapid bursts of traffic.
"""

import random
import time
from dataclasses import dataclass

from ddgs import DDGS
from ddgs.exceptions import DDGSException


@dataclass
class SearchResult:
    """
    A single web search result.
    """

    title: str
    url: str
    snippet: str


def search_web(
    query: str,
    max_results: int = 5,
    max_retries: int = 3,
    min_delay: float = 3.0,
    max_delay: float = 6.0,
) -> list[SearchResult]:
    """
    Search the web with retry and rate-limit protection.

    Parameters
    ----------
    query:
        Search query to execute.

    max_results:
        Maximum number of results to return.

    max_retries:
        Maximum number of retry attempts after the
        initial request.

    min_delay:
        Minimum delay before executing the search.

    max_delay:
        Maximum delay before executing the search.

    Returns
    -------
    list[SearchResult]
        Normalized search results.

    Notes
    -----
    A failed search returns an empty list instead of
    crashing the entire research workflow.
    """

    if not query.strip():
        return []

    # ------------------------------------------------------
    # Delay before every search request.
    #
    # A small random delay prevents requests from being
    # sent in perfectly regular bursts.
    # ------------------------------------------------------

    delay = random.uniform(
        min_delay,
        max_delay,
    )

    print(
        f"    Waiting {delay:.1f}s before search..."
    )

    time.sleep(delay)

    # ------------------------------------------------------
    # Search with retry handling.
    # ------------------------------------------------------

    # Open a single persistent connection session for all retry attempts
    with DDGS(timeout=10) as ddgs:
        for attempt in range(max_retries + 1):
            try:
                print(f"    Search attempt {attempt + 1}/{max_retries + 1}")

                # Use the persistent context instance 'ddgs' instead of instantiating a new one
                raw_results = ddgs.text(
                    query=query,
                    max_results=max_results,
                    backend="auto",
                )

                if not raw_results:
                    raise DDGSException("No results returned from endpoint.")

                results = []
                for result in raw_results:
                    url = result.get("href", "")
                    if not url:
                        continue
                    results.append(
                        SearchResult(
                            title=result.get("title", ""),
                            url=url,
                            snippet=result.get("body", ""),
                        )
                    )
                return results

            except DDGSException as exc:
                print(f"    Search failed: {exc}")
                if attempt >= max_retries:
                    print("    Giving up on this query.")
                    return []

                jitter = random.uniform(0.5, 1.5)
                retry_delay = (4 * (2 ** attempt)) + jitter
                print(f"    Retrying in {retry_delay:.1f} seconds...")
                time.sleep(retry_delay)

            except Exception as exc:
                print(f"    Unexpected search error: {exc}")
                return []

    return []