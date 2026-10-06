"""
Researcher node for the Deep Research Agent.

The researcher takes one planned research question,
executes its search queries, and stores the discovered
web sources in the shared LangGraph state.
"""

from app.graph.state import ResearchState
from app.schemas.research import Source
from app.tools.web_search import search_web


def researcher_node(
    state: ResearchState,
) -> ResearchState:
    """
    Research the current question using web search.

    The planner creates multiple research questions.
    This node handles one question at a time.
    """

    print("\n[Node] researcher")

    questions = state["research_questions"]
    current_index = state["current_question_index"]

    # ------------------------------------------------------
    # Check whether all research questions are completed.
    # ------------------------------------------------------

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
    # Execute the planned search queries.
    # ------------------------------------------------------

    discovered_sources = []

    for query in current_question.search_queries:

        print(
            f"\n  Searching: {query}"
        )

        results = search_web(
            query=query,
            max_results=5,
        )

        print(
            f"  Found {len(results)} results."
        )

        # --------------------------------------------------
        # Convert search results into our Source schema.
        # --------------------------------------------------

        for result in results:

            if not result.url:
                continue

            discovered_sources.append(
                Source(
                    url=result.url,
                    title=result.title,
                    content=result.snippet,
                )
            )

    # ------------------------------------------------------
    # Add new sources to existing state.
    # ------------------------------------------------------

    existing_sources = state["sources"]

    combined_sources = (
        existing_sources
        + discovered_sources
    )

    print(
        f"\n  Sources collected this round: "
        f"{len(discovered_sources)}"
    )

    print(
        f"  Total sources in state: "
        f"{len(combined_sources)}"
    )

    # ------------------------------------------------------
    # Move to the next research question.
    # ------------------------------------------------------

    return {
        **state,
        "sources": combined_sources,
        "current_question_index": current_index + 1,
    }
