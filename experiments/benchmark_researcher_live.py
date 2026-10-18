"""
Controlled live benchmark for the researcher node.

Runs one research question with one real web search query.
Uses the real source selector and webpage reader.

Warning:
This makes real external web requests and incurs the
configured search delay and any retry delays.
"""

import time
from types import SimpleNamespace

from app.graph.nodes.researcher import researcher_node


def main():
    question = SimpleNamespace(
        question="What is retrieval-augmented generation?",
        search_queries=[
            "retrieval augmented generation official documentation"
        ],
    )

    state = {
        "research_questions": [question],
        "current_question_index": 0,
        "sources": [],
    }

    print("=" * 70)
    print("LIVE RESEARCHER BENCHMARK")
    print("=" * 70)
    print("Research questions: 1")
    print("Search queries: 1")
    print("Maximum selected sources: 5")
    print("Search delay and retry policy: unchanged")
    print("\nStarting researcher node...\n")

    started = time.perf_counter()

    try:
        result = researcher_node(state)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"\nBenchmark failed after {elapsed:.2f}s")
        print(f"{type(exc).__name__}: {exc}")
        raise

    elapsed = time.perf_counter() - started

    print("\n" + "=" * 70)
    print("LIVE BENCHMARK RESULT")
    print("=" * 70)
    print(f"Total researcher-node time: {elapsed:.2f}s")
    print(f"Sources collected: {len(result['current_sources'])}")
    print(f"Total sources in state: {len(result['sources'])}")
    print(f"Question index after execution: {result['current_question_index']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
