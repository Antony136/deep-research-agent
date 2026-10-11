"""
Researcher node for the Deep Research Agent.

The researcher takes one planned research question,
searches the web, selects the strongest sources, reads
those sources, and stores both cumulative sources and
question-specific sources.
"""

from app.graph.state import ResearchState
from app.schemas.research import Source
from app.tools.source_selector import select_sources
from app.tools.web_reader import read_webpage
from app.tools.web_search import search_web


MAX_SOURCES_PER_QUESTION = 5


def researcher_node(
    state: ResearchState,
) -> ResearchState:
    """
    Research one planned question at a time.
    """

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

    existing_sources = list(state.get("sources") or [])

    existing_urls = {
        source.url.rstrip("/").lower()
        for source in existing_sources
    }

    candidate_sources = []

    for query in current_question.search_queries:
        print(f"\n  Searching: {query}")

        results = search_web(
            query=query,
            max_results=5,
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

    print(
        f"\n  Candidate sources collected: "
        f"{len(candidate_sources)}"
    )

    # ------------------------------------------------------
    # 2. Select the best sources
    # ------------------------------------------------------

    selected_sources = select_sources(
        sources=candidate_sources,
        max_sources=MAX_SOURCES_PER_QUESTION,
    )

    print(f"  Sources selected: {len(selected_sources)}")

    # ------------------------------------------------------
    # 3. Read selected sources
    # ------------------------------------------------------

    discovered_sources = []

    for source in selected_sources:
        print(f"\n    Reading: {source.title or source.url}")

        page = read_webpage(source.url)

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

    # ------------------------------------------------------
    # 4. Combine cumulative sources
    # ------------------------------------------------------

    combined_sources = existing_sources + discovered_sources

    print(f"\n  New selected sources: {len(discovered_sources)}")
    print(f"  Total sources in state: {len(combined_sources)}")

    # ------------------------------------------------------
    # 5. Update state
    # ------------------------------------------------------

    return {
        **state,
        "sources": combined_sources,
        "current_sources": discovered_sources,
        "active_research_question": current_question,
        "active_research_question_number": current_question_number,
        "current_question_index": current_index + 1,
    }
