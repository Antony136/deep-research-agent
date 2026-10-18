"""
Isolated researcher-node benchmark.

Uses mocked search and webpage reads to measure orchestration
overhead without making live network requests.
"""

import statistics
import time
from types import SimpleNamespace
from unittest.mock import patch

from app.graph.nodes import researcher


def make_question():
    return SimpleNamespace(
        question="What is retrieval-augmented generation?",
        search_queries=[
            "RAG overview",
            "RAG architecture",
            "RAG evaluation",
        ],
    )


def fake_search_web(query, max_results=5):
    return [
        SimpleNamespace(
            url=f"https://example.com/{query.replace(' ', '-')}",
            title=f"Result for {query}",
            snippet=f"Search snippet for {query}",
        )
    ]


def fake_read_webpage(url):
    return SimpleNamespace(
        url=url,
        title="Benchmark page",
        content="Full page content for benchmarking.",
    )


def main():
    runs = 10
    durations = []

    print("=" * 70)
    print("ISOLATED RESEARCHER BENCHMARK")
    print("=" * 70)
    print(f"Runs: {runs}")
    print("Searches per run: 3")
    print("Selected sources per run: up to 5")
    print("Network requests: none")

    for run in range(1, runs + 1):
        state = {
            "research_questions": [make_question()],
            "current_question_index": 0,
            "sources": [],
        }

        with (
            patch.object(
                researcher,
                "search_web",
                side_effect=fake_search_web,
            ),
            patch.object(
                researcher,
                "read_webpage",
                side_effect=fake_read_webpage,
            ),
        ):
            started = time.perf_counter()
            result = researcher.researcher_node(state)
            elapsed = time.perf_counter() - started

        durations.append(elapsed)

        assert result["current_question_index"] == 1
        assert len(result["current_sources"]) == 3

        print(f"Run {run:02}: {elapsed * 1000:.2f} ms")

    print("-" * 70)
    print(f"Mean:   {statistics.mean(durations) * 1000:.2f} ms")
    print(f"Median: {statistics.median(durations) * 1000:.2f} ms")
    print(f"Min:    {min(durations) * 1000:.2f} ms")
    print(f"Max:    {max(durations) * 1000:.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
