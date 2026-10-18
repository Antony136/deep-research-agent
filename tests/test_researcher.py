"""Unit tests for the Deep Research Agent researcher node."""

from types import SimpleNamespace

import pytest

from app.graph.nodes import researcher
from app.schemas.research import Source


def make_source(
    url: str,
    title: str = "Research source",
    content: str = "Search snippet",
) -> Source:
    return Source(
        url=url,
        title=title,
        content=content,
    )


def make_question(
    question: str = "What is retrieval-augmented generation?",
    search_queries: list[str] | None = None,
) -> SimpleNamespace:
    """Create a lightweight question with the attributes the node uses."""
    return SimpleNamespace(
        question=question,
        search_queries=search_queries or ["RAG overview"],
    )


def make_search_result(
    url: str,
    title: str = "Search result",
    snippet: str = "Search snippet",
) -> SimpleNamespace:
    return SimpleNamespace(
        url=url,
        title=title,
        snippet=snippet,
    )


def make_page(
    url: str,
    title: str = "Full webpage title",
    content: str = "Full webpage content",
) -> SimpleNamespace:
    return SimpleNamespace(
        url=url,
        title=title,
        content=content,
    )


def test_no_remaining_questions_marks_research_complete(monkeypatch):
    def unexpected_search(**kwargs):
        pytest.fail("Search should not run when no questions remain.")

    monkeypatch.setattr(researcher, "search_web", unexpected_search)

    state = {
        "research_questions": [],
        "current_question_index": 0,
        "research_complete": False,
        "active_research_question": make_question(),
        "active_research_question_number": 1,
    }

    result = researcher.researcher_node(state)

    assert result["research_complete"] is True
    assert result["active_research_question"] is None
    assert result["active_research_question_number"] is None


def test_researcher_deduplicates_urls_and_preserves_existing_sources(
    monkeypatch,
):
    existing = make_source(
        "https://example.com/old-article",
        content="Previously collected content",
    )

    search_calls = []

    def fake_search_web(query, max_results=5):
        search_calls.append(query)

        if query == "RAG overview":
            return [
                make_search_result(
                    "https://example.com/old-article/",
                    snippet="Duplicate content",
                ),
                make_search_result(
                    "https://newsite.com/article",
                    title="New article",
                    snippet="New article snippet",
                ),
            ]

        return [
            make_search_result(
                "https://newsite.com/article/",
                title="Duplicate result",
            ),
            make_search_result(
                "https://secondsite.com/article",
                title="Second article",
            ),
        ]

    monkeypatch.setattr(researcher, "search_web", fake_search_web)
    monkeypatch.setattr(
        researcher,
        "select_sources",
        lambda sources, max_sources: sources,
    )
    monkeypatch.setattr(
        researcher,
        "read_webpage",
        lambda url, client=None: None,
    )

    question = make_question(
        search_queries=["RAG overview", "RAG applications"],
    )

    state = {
        "research_questions": [question],
        "current_question_index": 0,
        "sources": [existing],
    }

    result = researcher.researcher_node(state)

    assert search_calls == ["RAG overview", "RAG applications"]

    # Existing sources remain in cumulative state.
    assert result["sources"][0] is existing

    # The existing URL and repeated candidate URL are not added again.
    assert [
        source.url for source in result["current_sources"]
    ] == [
        "https://newsite.com/article",
        "https://secondsite.com/article",
    ]

    assert len(result["sources"]) == 3
    assert result["current_question_index"] == 1


def test_only_selected_sources_are_read(monkeypatch):
    first = make_source(
        "https://firstsite.com/article",
        title="First article",
    )
    second = make_source(
        "https://secondsite.com/article",
        title="Second article",
    )

    read_urls = []

    monkeypatch.setattr(
        researcher,
        "search_web",
        lambda **kwargs: [
            make_search_result(first.url, first.title, "First snippet"),
            make_search_result(second.url, second.title, "Second snippet"),
        ],
    )

    monkeypatch.setattr(
        researcher,
        "select_sources",
        lambda sources, max_sources: [sources[1]],
    )

    def fake_read_webpage(url, client=None):
        read_urls.append(url)
        return None

    monkeypatch.setattr(researcher, "read_webpage", fake_read_webpage)

    result = researcher.researcher_node(
        {
            "research_questions": [make_question()],
            "current_question_index": 0,
            "sources": [],
        }
    )

    assert read_urls == [second.url]
    assert [source.url for source in result["current_sources"]] == [
        second.url
    ]


def test_successful_page_read_replaces_snippet_with_full_content(
    monkeypatch,
):
    search_result = make_search_result(
        "https://example.com/article",
        title="Search title",
        snippet="Short search snippet",
    )

    monkeypatch.setattr(
        researcher,
        "search_web",
        lambda **kwargs: [search_result],
    )
    monkeypatch.setattr(
        researcher,
        "select_sources",
        lambda sources, max_sources: sources,
    )
    monkeypatch.setattr(
        researcher,
        "read_webpage",
        lambda url, client=None: make_page(
            url="https://example.com/final-article",
            title="Full webpage title",
            content="Full webpage content",
        ),
    )

    result = researcher.researcher_node(
        {
            "research_questions": [make_question()],
            "current_question_index": 0,
            "sources": [],
        }
    )

    discovered = result["current_sources"][0]

    assert discovered.url == "https://example.com/final-article"
    assert discovered.title == "Full webpage title"
    assert discovered.content == "Full webpage content"


def test_missing_webpage_title_falls_back_to_search_title(monkeypatch):
    monkeypatch.setattr(
        researcher,
        "search_web",
        lambda **kwargs: [
            make_search_result(
                "https://example.com/article",
                title="Search title",
                snippet="Search snippet",
            )
        ],
    )
    monkeypatch.setattr(
        researcher,
        "select_sources",
        lambda sources, max_sources: sources,
    )
    monkeypatch.setattr(
        researcher,
        "read_webpage",
        lambda url, client=None: make_page(
            url=url,
            title="",
            content="Full webpage content",
        ),
    )

    result = researcher.researcher_node(
        {
            "research_questions": [make_question()],
            "current_question_index": 0,
            "sources": [],
        }
    )

    discovered = result["current_sources"][0]

    assert discovered.title == "Search title"
    assert discovered.content == "Full webpage content"


def test_failed_page_read_keeps_search_snippet(monkeypatch):
    monkeypatch.setattr(
        researcher,
        "search_web",
        lambda **kwargs: [
            make_search_result(
                "https://example.com/article",
                title="Search title",
                snippet="Useful fallback snippet",
            )
        ],
    )
    monkeypatch.setattr(
        researcher,
        "select_sources",
        lambda sources, max_sources: sources,
    )
    monkeypatch.setattr(
        researcher,
        "read_webpage",
        lambda url, client=None: None,
    )

    result = researcher.researcher_node(
        {
            "research_questions": [make_question()],
            "current_question_index": 0,
            "sources": [],
        }
    )

    discovered = result["current_sources"][0]

    assert discovered.title == "Search title"
    assert discovered.content == "Useful fallback snippet"


def test_search_errors_propagate(monkeypatch):
    def failing_search(**kwargs):
        raise RuntimeError("Search provider unavailable")

    monkeypatch.setattr(researcher, "search_web", failing_search)

    with pytest.raises(
        RuntimeError,
        match="Search provider unavailable",
    ):
        researcher.researcher_node(
            {
                "research_questions": [make_question()],
                "current_question_index": 0,
                "sources": [],
            }
        )
