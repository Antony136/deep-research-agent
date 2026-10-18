"""
Unit tests for web search, normalization, and retry behavior.
"""

import pytest

from ddgs.exceptions import DDGSException

from app.tools import web_search


class FakeDDGS:
    """Deterministic replacement for the external search provider."""

    def __init__(self, timeout):
        assert timeout == 10
        self.attempts = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def text(self, **kwargs):
        self.attempts += 1

        assert kwargs["query"] == "test query"
        assert kwargs["max_results"] == 5
        assert kwargs["backend"] == "auto"

        return [
            {
                "title": "Example result",
                "href": "https://example.com/article",
                "body": "Example snippet",
            },
            {
                "title": "Missing URL",
                "href": "",
                "body": "This result should be skipped.",
            },
        ]


def test_blank_query_returns_empty_without_search(monkeypatch):
    def unexpected_search(*_args, **_kwargs):
        raise AssertionError("Provider should not be called.")

    monkeypatch.setattr(web_search, "DDGS", unexpected_search)

    assert web_search.search_web("   ") == []


def test_successful_search_normalizes_results(monkeypatch):
    monkeypatch.setattr(web_search, "DDGS", FakeDDGS)
    monkeypatch.setattr(web_search.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(
        web_search.random,
        "uniform",
        lambda *_args: 0,
    )

    results = web_search.search_web(
        "test query",
        min_delay=0,
        max_delay=0,
    )

    assert len(results) == 1
    assert results[0].title == "Example result"
    assert results[0].url == "https://example.com/article"
    assert results[0].snippet == "Example snippet"


def test_search_retries_after_transient_provider_failure(monkeypatch):
    attempts = []

    class RecoveringDDGS(FakeDDGS):
        def text(self, **kwargs):
            attempts.append(1)

            if len(attempts) == 1:
                raise DDGSException("temporary outage")

            return [
                {
                    "title": "Recovered result",
                    "href": "https://example.com/recovered",
                    "body": "Recovered snippet",
                }
            ]

    monkeypatch.setattr(web_search, "DDGS", RecoveringDDGS)
    monkeypatch.setattr(web_search.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(
        web_search.random,
        "uniform",
        lambda *_args: 1,
    )

    results = web_search.search_web(
        "test query",
        max_retries=1,
        min_delay=0,
        max_delay=0,
    )

    assert len(attempts) == 2
    assert results[0].title == "Recovered result"


def test_empty_provider_results_are_retried(monkeypatch):
    attempts = []

    class EmptyThenSuccessfulDDGS(FakeDDGS):
        def text(self, **kwargs):
            attempts.append(1)

            if len(attempts) == 1:
                return []

            return [
                {
                    "title": "Recovered result",
                    "href": "https://example.com/recovered",
                    "body": "Recovered snippet",
                }
            ]

    monkeypatch.setattr(web_search, "DDGS", EmptyThenSuccessfulDDGS)
    monkeypatch.setattr(web_search.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(
        web_search.random,
        "uniform",
        lambda *_args: 1,
    )

    results = web_search.search_web(
        "test query",
        max_retries=1,
        min_delay=0,
        max_delay=0,
    )

    assert len(attempts) == 2
    assert results[0].title == "Recovered result"


def test_exhausted_retries_raise_provider_error(monkeypatch):
    attempts = []

    class AlwaysFailingDDGS(FakeDDGS):
        def text(self, **kwargs):
            attempts.append(1)
            raise DDGSException("provider unavailable")

    monkeypatch.setattr(web_search, "DDGS", AlwaysFailingDDGS)
    monkeypatch.setattr(web_search.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(
        web_search.random,
        "uniform",
        lambda *_args: 1,
    )

    with pytest.raises(DDGSException, match="provider unavailable"):
        web_search.search_web(
            "test query",
            max_retries=2,
            min_delay=0,
            max_delay=0,
        )

    assert len(attempts) == 3
