"""
Researcher node for the Deep Research Agent.

The researcher takes one planned research question,
searches the web, selects the strongest sources, reads
those sources, and stores both cumulative sources and
question-specific sources.

A reusable HTTP client is shared across webpage reads
within the current researcher's selected-source batch.

Temporary performance diagnostics measure each stage
without changing the research workflow.
"""

import time

import httpx

from app.graph.state import ResearchState
from app.schemas.research import Source
from app.tools.source_selector import select_sources
from app.tools.web_reader import (
    DEFAULT_TIMEOUT,
    USER_AGENT,
    read_webpage,
)
from app.tools.web_search import search_web


MAX_SOURCES_PER_QUESTION = 5


def researcher_node(
    state: ResearchState,
) -> ResearchState:
    """
    Research one planned question at a time.
    """

    node_started = time.perf_counter()

    print("\n[Node] researcher")

    questions = state.get("research_questions", [])
    current_index = state.get("current_question_index", 0)

    if current_index >= len(questions):
        return {
            **state,
            "research_complete": True,
            "active_research_question": None,
            "active_research_question_number": None,
        }

    current_question = questions[current_index]
    current_question_number = current_index + 1

    print(
        f"Research question "
        f"{current_question_number}/{len(questions)}:"
    )
    print(f"  {current_question.question}")

    # ------------------------------------------------------
    # 1. Collect search candidates
    # ------------------------------------------------------

    search_started = time.perf_counter()
    existing_sources = list(state.get("sources") or [])

    existing_urls = {
        source.url.rstrip("/").lower()
        for source in existing_sources
    }

    candidate_sources = []
    search_query_times = []

    for query in current_question.search_queries:
        query_started = time.perf_counter()

        print(f"\n  Searching: {query}")

        try:
            results = search_web(
                query=query,
                max_results=5,
            )
        finally:
            query_elapsed = time.perf_counter() - query_started
            search_query_times.append((query, query_elapsed))

            print(
                f"  [Timing] Search query took "
                f"{query_elapsed:.2f}s"
            )

        print(f"  Found {len(results)} search results.")

        for result in results:
            if not result.url:
                continue

            normalized_url = result.url.rstrip("/").lower()

            if normalized_url in existing_urls:
                print(f"    Skipping duplicate: {result.url}")
                continue

            existing_urls.add(normalized_url)

            candidate_sources.append(
                Source(
                    url=result.url,
                    title=result.title,
                    content=result.snippet,
                )
            )

    search_elapsed = time.perf_counter() - search_started

    print(
        f"\n  Candidate sources collected: "
        f"{len(candidate_sources)}"
    )
    print(
        f"  [Timing] Total search and candidate collection: "
        f"{search_elapsed:.2f}s"
    )

    # ------------------------------------------------------
    # 2. Select best sources
    # ------------------------------------------------------

    selection_started = time.perf_counter()

    selected_sources = select_sources(
        sources=candidate_sources,
        max_sources=MAX_SOURCES_PER_QUESTION,
    )

    selection_elapsed = time.perf_counter() - selection_started

    print(f"  Sources selected: {len(selected_sources)}")
    print(f"  [Timing] Source selection: {selection_elapsed:.3f}s")

    # ------------------------------------------------------
    # 3. Read selected sources using one shared client
    # ------------------------------------------------------

    reading_started = time.perf_counter()
    discovered_sources = []
    webpage_read_times = []

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html, "
            "application/xhtml+xml, "
            "text/plain;q=0.9"
        ),
    }

    # The context manager guarantees that the shared client
    # is closed after the selected-source batch finishes.
    with httpx.Client(
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
        follow_redirects=False,
    ) as client:

        for source in selected_sources:
            page_started = time.perf_counter()

            print(f"\n    Reading: {source.title or source.url}")

            try:
                page = read_webpage(
                    source.url,
                    client=client,
                )
            finally:
                page_elapsed = time.perf_counter() - page_started
                webpage_read_times.append((source.url, page_elapsed))

                print(
                    f"    [Timing] Webpage read took "
                    f"{page_elapsed:.2f}s"
                )

            if page is not None:
                source = Source(
                    url=page.url,
                    title=page.title or source.title,
                    content=page.content,
                )
                print("    Page read successfully.")
            else:
                print(
                    "    Page could not be read. "
                    "Keeping search snippet."
                )

            discovered_sources.append(source)

    reading_elapsed = time.perf_counter() - reading_started

    # ------------------------------------------------------
    # 4. Combine cumulative sources
    # ------------------------------------------------------

    combined_sources = existing_sources + discovered_sources

    print(f"\n  New selected sources: {len(discovered_sources)}")
    print(f"  Total sources in state: {len(combined_sources)}")

    # ------------------------------------------------------
    # 5. Performance summary
    # ------------------------------------------------------

    node_elapsed = time.perf_counter() - node_started

    print("\n  [Researcher Timing Summary]")
    print(f"    Search/candidate collection: {search_elapsed:.2f}s")
    print(f"    Source selection:            {selection_elapsed:.3f}s")
    print(f"    Webpage reading:             {reading_elapsed:.2f}s")
    print(f"    Researcher node total:       {node_elapsed:.2f}s")

    if webpage_read_times:
        read_durations = [
            elapsed
            for _, elapsed in webpage_read_times
        ]

        mean_read_time = (
            sum(read_durations)
            / len(read_durations)
        )

        print(
            f"    Average read-call duration: "
            f"{mean_read_time:.2f}s "
            f"({len(webpage_read_times)} calls)"
        )

    # ------------------------------------------------------
    # 6. Update state
    # ------------------------------------------------------

    return {
        **state,
        "sources": combined_sources,
        "current_sources": discovered_sources,
        "active_research_question": current_question,
        "active_research_question_number": current_question_number,
        "current_question_index": current_index + 1,
    }
