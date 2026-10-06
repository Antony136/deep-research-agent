"""
Entry point for the Deep Research Agent.

Runs the complete research workflow and displays
the research plan, collected sources, and extracted
evidence in a readable format.
"""

from app.graph.graph import build_research_graph


def print_separator(char="=", width=80):
    print(char * width)


def print_research_plan(research_questions):
    print_separator()

    print("RESEARCH PLAN")

    print_separator()

    print(
        f"Research questions: "
        f"{len(research_questions)}"
    )

    for index, research_question in enumerate(
        research_questions,
        start=1,
    ):

        print()
        print(
            f"[{index}] "
            f"{research_question.question}"
        )

        print(
            "    Search queries:"
        )

        for query in research_question.search_queries:

            print(
                f"      - {query}"
            )


def print_sources(sources):
    print_separator()

    print("COLLECTED SOURCES")

    print_separator()

    print(
        f"Total sources: "
        f"{len(sources)}"
    )

    for index, source in enumerate(
        sources,
        start=1,
    ):

        print()
        print(
            f"[{index}] "
            f"{source.title or 'Untitled'}"
        )

        print(
            f"    URL: {source.url}"
        )

        print(
            f"    Content length: "
            f"{len(source.content):,} characters"
        )


def print_evidence(evidence):
    print_separator()

    print("EXTRACTED EVIDENCE")

    print_separator()

    print(
        f"Total evidence items: "
        f"{len(evidence)}"
    )

    if not evidence:

        print(
            "\nNo evidence was extracted."
        )

        return

    for index, item in enumerate(
        evidence,
        start=1,
    ):

        print()
        print(
            f"[{index}] CLAIM"
        )

        print(
            f"    {item.claim}"
        )

        print()
        print(
            "    SOURCE"
        )

        print(
            f"    {item.source_url}"
        )

        print()
        print(
            "    SUPPORTING TEXT"
        )

        print(
            f"    {item.supporting_text}"
        )


def main():

    print_separator()

    print("DEEP RESEARCH AGENT")

    print_separator()

    question = (
        "Compare LangChain, LangGraph, and CrewAI "
        "for building production AI agent systems."
    )

    print("\nUSER QUESTION")
    print("-" * 80)
    print(question)

    # ------------------------------------------------------
    # INITIAL STATE
    # ------------------------------------------------------

    initial_state = {
        "question": question,
        "research_questions": [],
        "sources": [],
        "current_sources": [],
        "evidence": [],
        "current_question_index": 0,
        "research_complete": False,
        "report": None,
    }

    # ------------------------------------------------------
    # BUILD GRAPH
    # ------------------------------------------------------

    print("\n")
    print("BUILDING RESEARCH GRAPH...")

    graph = build_research_graph()

    print(
        "Graph compiled successfully."
    )

    # ------------------------------------------------------
    # RUN RESEARCH
    # ------------------------------------------------------

    print("\n")
    print_separator()

    print("STARTING RESEARCH")

    print_separator()

    final_state = graph.invoke(
        initial_state
    )

    # ------------------------------------------------------
    # DISPLAY PLAN
    # ------------------------------------------------------

    print("\n")

    print_research_plan(
        final_state["research_questions"]
    )

    # ------------------------------------------------------
    # DISPLAY SOURCES
    # ------------------------------------------------------

    print("\n")

    print_sources(
        final_state["sources"]
    )

    # ------------------------------------------------------
    # DISPLAY EVIDENCE
    # ------------------------------------------------------

    print("\n")

    print_evidence(
        final_state["evidence"]
    )

    # ------------------------------------------------------
    # FINAL STATE
    # ------------------------------------------------------

    print("\n")

    print_separator()

    print("RESEARCH WORKFLOW SUMMARY")

    print_separator()

    print(
        f"Research questions: "
        f"{len(final_state['research_questions'])}"
    )

    print(
        f"Sources collected: "
        f"{len(final_state['sources'])}"
    )

    print(
        f"Evidence items: "
        f"{len(final_state['evidence'])}"
    )

    print(
        f"Research complete: "
        f"{final_state['research_complete']}"
    )

    print_separator()

    print(
        "RESEARCH PIPELINE COMPLETED"
    )

    print_separator()


if __name__ == "__main__":
    main()
