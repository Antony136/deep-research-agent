"""
Entry point for the Deep Research Agent.

Runs the complete research workflow and displays
the research plan, collected sources, extracted
evidence, and adaptive research decisions.
"""

from app.graph.graph import build_research_graph


def print_separator(
    char="=",
    width=80,
):
    print(char * width)


def print_research_plan(
    research_questions,
):
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


def print_sources(
    sources,
):
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
            f"    URL: "
            f"{source.url}"
        )

        print(
            f"    Content length: "
            f"{len(source.content):,} characters"
        )


def print_evidence(
    evidence,
):
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
            f"[{index}] "
            f"RESEARCH QUESTION "
            f"{item.research_question_number}"
        )

        print(
            "    CLAIM"
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


def print_research_decision(
    state,
):
    print_separator()
    print("ADAPTIVE RESEARCH DECISION")
    print_separator()

    print(
        "Research round: "
        f"{state['research_round']}"
    )

    print(
        "Maximum research rounds: "
        f"{state['max_research_rounds']}"
    )

    print()

    print(
        "Research sufficient: "
        f"{state['research_sufficient']}"
    )

    print()

    print("Reason:")

    print(
        f"  {state['research_decision_reason']}"
    )

    print()

    print("Remaining research gaps:")

    research_gaps = state[
        "research_gaps"
    ]

    if not research_gaps:

        print(
            "  None"
        )

    else:

        for gap in research_gaps:

            print(
                f"  - {gap}"
            )


def main():

    print_separator()
    print("DEEP RESEARCH AGENT")
    print_separator()

    question = (
        "How is climate change affecting global coffee production, "
        "and what strategies are farmers and governments using to adapt?"
    )

    print("\nUSER QUESTION")
    print("-" * 80)
    print(question)

    initial_state = {
        "question": question,

        # --------------------------------------------------
        # RESEARCH PLAN
        # --------------------------------------------------

        "research_questions": [],

        # --------------------------------------------------
        # CURRENT RESEARCH QUESTION
        # --------------------------------------------------

        # The researcher populates these fields when it
        # begins processing a planned research question.
        "active_research_question": None,
        "active_research_question_number": None,

        # --------------------------------------------------
        # RESEARCH DATA
        # --------------------------------------------------

        "sources": [],
        "current_sources": [],
        "pending_evidence": [],
        "evidence": [],

        # --------------------------------------------------
        # WORKFLOW CONTROL
        # --------------------------------------------------

        # Pointer to the next research question that should
        # be processed by the researcher.
        "current_question_index": 0,

        "research_complete": False,

        # --------------------------------------------------
        # ADAPTIVE RESEARCH
        # --------------------------------------------------

        # Round 1 represents the original planner's
        # research plan.
        "research_round": 1,

        # Hard safety limit preventing an endless research
        # loop.
        #
        # Round 1:
        #   Initial research plan
        #
        # Round 2:
        #   First adaptive research plan
        #
        # Round 3:
        #   Second adaptive research plan
        "max_research_rounds": 3,

        "research_gaps": [],

        # --------------------------------------------------
        # SUFFICIENCY
        # --------------------------------------------------

        "research_sufficient": False,
        "research_decision_reason": "",

        # --------------------------------------------------
        # FINAL OUTPUT
        # --------------------------------------------------

        "report": None,
    }

    print("\n")
    print(
        "BUILDING RESEARCH GRAPH..."
    )

    graph = build_research_graph()

    print(
        "Graph compiled successfully."
    )

    print("\n")

    print_separator()
    print("STARTING RESEARCH")
    print_separator()

    final_state = graph.invoke(
        initial_state
    )

    print("\n")

    print_research_plan(
        final_state[
            "research_questions"
        ]
    )

    print("\n")

    print_sources(
        final_state[
            "sources"
        ]
    )

    print("\n")

    print_evidence(
        final_state[
            "evidence"
        ]
    )

    print("\n")

    print_research_decision(
        final_state
    )

    print("\n")

    print_separator()
    print("RESEARCH WORKFLOW SUMMARY")
    print_separator()

    print(
        "Research questions: "
        f"{len(final_state['research_questions'])}"
    )

    print(
        "Sources collected: "
        f"{len(final_state['sources'])}"
    )

    print(
        "Evidence items: "
        f"{len(final_state['evidence'])}"
    )

    print(
        "Research questions processed: "
        f"{final_state['current_question_index']}"
    )

    print(
        "Research round: "
        f"{final_state['research_round']}"
    )

    print(
        "Research complete: "
        f"{final_state['research_complete']}"
    )

    print(
        "Research sufficient: "
        f"{final_state['research_sufficient']}"
    )

    print_separator()

    print(
        "RESEARCH PIPELINE COMPLETED"
    )

    print_separator()


if __name__ == "__main__":
    main()
