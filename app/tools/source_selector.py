"""
Source selection for the Deep Research Agent.

This module selects a small, useful set of web sources from
the larger collection returned by web search.

Selection is intentionally deterministic and does not call
the LLM. This keeps the research pipeline efficient while
giving higher priority to authoritative sources.
"""

from urllib.parse import urlparse

from app.schemas.research import Source


DEFAULT_MAX_SOURCES = 5


# ------------------------------------------------------------
# Source authority
# ------------------------------------------------------------

# Domains that are generally useful for factual, scientific,
# technical, governmental, or research-oriented questions.
#
# This is intentionally a small heuristic rather than a rigid
# allowlist. Unknown domains can still be selected.
HIGH_AUTHORITY_DOMAINS = {
    # Government / international organizations
    "gov",
    "gov.uk",
    "europa.eu",
    "un.org",
    "who.int",
    "worldbank.org",
    "oecd.org",
    "fao.org",
    "wto.org",

    # Research / academic
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "ieee.org",
    "acm.org",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",

    # Major research / standards organizations
    "researchgate.net",
    "arxiv.org",
    "nist.gov",
    "mit.edu",
    "stanford.edu",
    "harvard.edu",

    # Major technical / documentation sources
    "docs.python.org",
    "developer.mozilla.org",
    "learn.microsoft.com",
    "docs.microsoft.com",
    "cloud.google.com",
    "docs.aws.amazon.com",
    "kubernetes.io",
    "docker.com",
    "postgresql.org",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


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


def _get_domain_parts(
    domain: str,
) -> list[str]:
    """
    Return normalized domain components.

    Example:

        www.example.com
        -> ["www", "example", "com"]
    """

    return [
        part
        for part in domain.lower().split(".")
        if part
    ]


def _authority_score(
    source: Source,
) -> int:
    """
    Calculate a deterministic authority score.

    This is a heuristic, not a guarantee that a source is
    factually correct.

    Higher scores are given to:
    - Government domains
    - Academic domains
    - International organizations
    - Recognized research organizations
    - Established technical documentation domains
    """

    domain = _get_domain(
        source.url
    )

    if not domain:
        return 0

    score = 0

    # Exact high-authority domains.
    if domain in HIGH_AUTHORITY_DOMAINS:
        score += 20

    # Remove www for comparisons.
    normalized_domain = domain.removeprefix(
        "www."
    )

    if normalized_domain in HIGH_AUTHORITY_DOMAINS:
        score += 20

    # Government domains.
    if (
        normalized_domain.endswith(".gov")
        or normalized_domain.endswith(".gov.uk")
    ):
        score += 18

    # Academic domains.
    if normalized_domain.endswith(".edu"):
        score += 16

    # Country-specific academic domains.
    if normalized_domain.endswith(".ac.uk"):
        score += 16

    if normalized_domain.endswith(".ac.in"):
        score += 16

    # International / non-profit organizations.
    if normalized_domain.endswith(".org"):
        score += 3

    return score


def _source_score(
    source: Source,
) -> int:
    """
    Calculate a deterministic quality score.

    Higher scores indicate that the source is more useful
    for downstream research.

    The score combines:

    1. Source authority
    2. Readable page content
    3. Meaningful title
    4. Content length
    5. HTTPS usage
    """

    score = 0

    # --------------------------------------------------------
    # Authority
    # --------------------------------------------------------

    score += _authority_score(
        source
    )

    # --------------------------------------------------------
    # URL quality
    # --------------------------------------------------------

    if source.url.lower().startswith(
        "https://"
    ):
        score += 2

    # --------------------------------------------------------
    # Readable content
    # --------------------------------------------------------

    if source.content.strip():
        score += 10

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    if source.title.strip():
        score += 2

    # --------------------------------------------------------
    # Content depth
    # --------------------------------------------------------

    content_length = len(
        source.content.strip()
    )

    if content_length >= 5000:
        score += 8

    elif content_length >= 2000:
        score += 6

    elif content_length >= 1000:
        score += 5

    elif content_length >= 500:
        score += 3

    elif content_length >= 200:
        score += 1

    return score


def _normalize_url(
    url: str,
) -> str:
    """
    Normalize a URL for duplicate detection.
    """

    return url.strip().rstrip("/").lower()


# ------------------------------------------------------------
# Public selection function
# ------------------------------------------------------------


def select_sources(
    sources: list[Source],
    max_sources: int = DEFAULT_MAX_SOURCES,
) -> list[Source]:
    """
    Select the strongest sources from a collection of
    candidates.

    Selection rules:

    1. Ignore sources without a URL.
    2. Remove duplicate URLs.
    3. Rank sources using deterministic quality and
       authority signals.
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

    # --------------------------------------------------------
    # 1. Remove invalid and duplicate URLs
    # --------------------------------------------------------

    unique_sources = []
    seen_urls = set()

    for source in sources:

        normalized_url = _normalize_url(
            source.url
        )

        if not normalized_url:
            continue

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

    # --------------------------------------------------------
    # 2. Rank candidates
    # --------------------------------------------------------

    ranked_sources = sorted(
        unique_sources,
        key=lambda source: (
            -_source_score(source),
            _get_domain(source.url),
            _normalize_url(source.url),
        ),
    )

    # --------------------------------------------------------
    # 3. Prefer domain diversity
    # --------------------------------------------------------

    selected = []
    selected_urls = set()
    selected_domains = set()

    # First pass:
    # choose the strongest source from each domain.
    #
    # Because ranked_sources is already ordered by score,
    # this automatically prefers authoritative sources.

    for source in ranked_sources:

        domain = _get_domain(
            source.url
        )

        normalized_url = _normalize_url(
            source.url
        )

        if not domain:
            continue

        if domain in selected_domains:
            continue

        selected.append(
            source
        )

        selected_urls.add(
            normalized_url
        )

        selected_domains.add(
            domain
        )

        if len(selected) >= max_sources:
            return selected

    # --------------------------------------------------------
    # 4. Fill remaining slots
    # --------------------------------------------------------

    for source in ranked_sources:

        normalized_url = _normalize_url(
            source.url
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
