"""
Unit tests for deterministic web source selection.
"""

from app.schemas.research import Source
from app.tools.source_selector import select_sources


def make_source(
    url: str,
    title: str = "Research source",
    content: str = "Useful research content. " * 50,
) -> Source:
    return Source(
        url=url,
        title=title,
        content=content,
    )


def test_empty_input_returns_empty_list():
    assert select_sources([]) == []


def test_non_positive_max_sources_returns_empty_list():
    sources = [
        make_source("https://example.com/article"),
    ]

    assert select_sources(sources, max_sources=0) == []
    assert select_sources(sources, max_sources=-1) == []


def test_duplicate_urls_are_removed():
    first = make_source("https://example.com/article")
    duplicate = make_source("https://example.com/article")

    selected = select_sources([first, duplicate])

    assert len(selected) == 1
    assert selected[0] is first


def test_selection_does_not_exceed_max_sources():
    sources = [
        make_source(f"https://site{i}.com/article")
        for i in range(10)
    ]

    selected = select_sources(sources, max_sources=3)

    assert len(selected) == 3


def test_prefers_different_domains_when_possible():
    sources = [
        make_source(
            "https://example.com/article-a",
            content="Detailed research content. " * 300,
        ),
        make_source(
            "https://example.com/article-b",
            content="Detailed research content. " * 300,
        ),
        make_source(
            "https://another-example.com/article",
            content="Short content.",
        ),
    ]

    selected = select_sources(sources, max_sources=2)

    domains = {
        source.url.split("/")[2]
        for source in selected
    }

    assert len(selected) == 2
    assert len(domains) == 2


def test_empty_url_is_ignored():
    invalid = make_source("")
    valid = make_source("https://example.com/article")

    selected = select_sources([invalid, valid])

    assert selected == [valid]


def test_https_source_scores_above_equivalent_http_source():
    http_source = make_source(
        "http://example.com/article",
    )
    https_source = make_source(
        "https://example.com/article",
    )

    selected = select_sources(
        [http_source, https_source],
        max_sources=1,
    )

    assert selected == [https_source]
