"""
Researcher node for the Deep Research Agent.

The researcher takes one planned research question,
searches the web, selects the strongest sources, reads
those sources, and stores both the cumulative sources and
the sources belonging to the current research question.
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
    Research the current question using web search,
    source selection, and web page extraction.

    The planner creates multiple research questions.
    This node handles one question at a time.
    """

    print("\n[Node] researcher")

    questions = state["research_questions"]
    current_index = state["current_question_index"]

    if current_index >= len(questions):
        return {
            **state,
            "research_complete": True,
        }

    current_question = questions[current_index]

    print(
        f"Research question "
        f"{current_index + 1}/{len(questions)}:"
    )

    print(
        f"  {current_question.question}"
    )

    # ------------------------------------------------------
    # 1. Collect search candidates
    # ------------------------------------------------------

    existing_urls = {
        source.url.rstrip("/").lower()
        for source in state["sources"]
    }

    candidate_sources = []

    for query in current_question.search_queries:

        print(
            f"\n  Searching: {query}"
        )

        results = search_web(
            query=query,
            max_results=5,
        )

        print(
            f"  Found {len(results)} search results."
        )

        for result in results:

            if not result.url:
                continue

            normalized_url = (
                result.url
                .rstrip("/")
                .lower()
            )

            if normalized_url in existing_urls:

                print(
                    f"    Skipping duplicate: "
                    f"{result.url}"
                )

                continue

            existing_urls.add(
                normalized_url
            )

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

    print(
        f"  Sources selected: "
        f"{len(selected_sources)}"
    )

    # ------------------------------------------------------
    # 3. Read only selected sources
    # ------------------------------------------------------

    discovered_sources = []

    for source in selected_sources:

        print(
            f"\n    Reading: "
            f"{source.title or source.url}"
        )

        page = read_webpage(
            source.url
        )

        if page is not None:

            source = Source(
                url=page.url,
                title=(
                    page.title
                    or source.title
                ),
                content=page.content,
            )

            print(
                "    Page read successfully."
            )

        else:

            print(
                "    Page could not be read."
                " Keeping search snippet."
            )

        discovered_sources.append(
            source
        )

    # ------------------------------------------------------
    # 4. Store sources
    # ------------------------------------------------------

    existing_sources = state["sources"]

    combined_sources = (
        existing_sources
        + discovered_sources
    )

    print(
        f"\n  New selected sources: "
        f"{len(discovered_sources)}"
    )

    print(
        f"  Total sources in state: "
        f"{len(combined_sources)}"
    )

    # ------------------------------------------------------
    # 5. Update state
    # ------------------------------------------------------

    return {
        **state,
        "sources": combined_sources,

        # These are ONLY the sources belonging to the
        # research question that was just processed.
        "current_sources": discovered_sources,

        "current_question_index": (
            current_index + 1
        ),
    }
