"""
Source selection for the Deep Research Agent.

This module selects a small, useful set of web sources from
the larger collection returned by web search.

Selection is intentionally deterministic and does not call
the LLM. This keeps the research pipeline efficient and
avoids spending an LLM call merely to rank search results.
"""

from urllib.parse import urlparse

from app.schemas.research import Source


DEFAULT_MAX_SOURCES = 5


def _get_domain(url: str) -> str:
    """
    Extract the hostname from a URL.

    Example:

        https://docs.python.org/3/
        -> docs.python.org
    """

    try:
        hostname = urlparse(url).hostname

    except ValueError:
        return ""

    return (
        hostname.lower()
        if hostname
        else ""
    )


def _source_score(
    source: Source,
) -> int:
    """
    Calculate a simple deterministic quality score.

    Higher scores indicate that the source contains more
    useful information for downstream research.
    """

    score = 0

    # A readable page is more useful than a search snippet.
    if source.content.strip():
        score += 10

    # Prefer sources with a meaningful title.
    if source.title.strip():
        score += 2

    # Prefer sources with substantial content.
    content_length = len(
        source.content.strip()
    )

    if content_length >= 1000:
        score += 5

    elif content_length >= 500:
        score += 3

    elif content_length >= 200:
        score += 1

    return score


def select_sources(
    sources: list[Source],
    max_sources: int = DEFAULT_MAX_SOURCES,
) -> list[Source]:
    """
    Select the best sources from a collection of candidates.

    Selection rules:

    1. Ignore sources without a URL.
    2. Remove duplicate URLs.
    3. Rank sources using deterministic quality signals.
    4. Prefer domain diversity.
    5. Fill remaining slots with the strongest sources.
    6. Return at most `max_sources`.

    Parameters
    ----------
    sources:
        Candidate sources returned by web search.

    max_sources:
        Maximum number of sources to select.

    Returns
    -------
    list[Source]
        Selected sources in ranking order.
    """

    if max_sources <= 0:
        return []

    # ------------------------------------------------------
    # 1. Remove invalid and duplicate URLs
    # ------------------------------------------------------

    unique_sources = []
    seen_urls = set()

    for source in sources:

        url = source.url.strip()

        if not url:
            continue

        normalized_url = url.rstrip("/").lower()

        if normalized_url in seen_urls:
            continue

        seen_urls.add(
            normalized_url
        )

        unique_sources.append(
            source
        )

    if not unique_sources:
        return []

    # ------------------------------------------------------
    # 2. Rank candidates
    # ------------------------------------------------------

    ranked_sources = sorted(
        unique_sources,
        key=_source_score,
        reverse=True,
    )

    # ------------------------------------------------------
    # 3. Prefer domain diversity
    # ------------------------------------------------------

    selected = []
    selected_urls = set()
    selected_domains = set()

    # First pass:
    # choose the strongest source from each domain.

    for source in ranked_sources:

        domain = _get_domain(
            source.url
        )

        if domain in selected_domains:
            continue

        selected.append(
            source
        )

        selected_urls.add(
            source.url.rstrip("/").lower()
        )

        selected_domains.add(
            domain
        )

        if len(selected) >= max_sources:
            return selected

    # ------------------------------------------------------
    # 4. Fill remaining slots
    # ------------------------------------------------------

    for source in ranked_sources:

        normalized_url = (
            source.url
            .rstrip("/")
            .lower()
        )

        if normalized_url in selected_urls:
            continue

        selected.append(
            source
        )

        selected_urls.add(
            normalized_url
        )

        if len(selected) >= max_sources:
            break

    return selected
