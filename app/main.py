"""
Entry point for the Deep Research Agent.

Runs the complete research workflow and displays
a concise summary of the final research state.
"""

from app.graph.graph import build_research_graph


def print_separator(
    char="=",
    width=80,
):
    print(char * width)


def print_research_decision(
    state,
):
    print_separator()
    print("RESEARCH DECISION")
    print_separator()

    print(
        f"Research sufficient: "
        f"{state['research_sufficient']}"
    )

    print(
        f"Research round: "
        f"{state['research_round']}/"
        f"{state['max_research_rounds']}"
    )

    print(
        f"Research questions: "
        f"{len(state['research_questions'])}/"
        f"{state['max_total_research_questions']}"
    )

    print()

    print(
        f"Reason: "
        f"{state['research_decision_reason']}"
    )

    research_gaps = state["research_gaps"]

    if research_gaps:

        print()
        print("Research gaps:")

        for gap in research_gaps:
            print(f"- {gap}")

    else:

        print()
        print("Research gaps: None")


def print_summary(
    state,
):
    print_separator()
    print("RESEARCH SUMMARY")
    print_separator()

    print(
        f"Research questions: "
        f"{len(state['research_questions'])}"
    )

    print(
        f"Sources collected: "
        f"{len(state['sources'])}"
    )

    print(
        f"Verified evidence: "
        f"{len(state['evidence'])}"
    )

    print(
        f"Research questions processed: "
        f"{state['current_question_index']}"
    )

    print(
        f"Research round: "
        f"{state['research_round']}"
    )

    print(
        f"Research complete: "
        f"{state['research_complete']}"
    )

    print(
        f"Research sufficient: "
        f"{state['research_sufficient']}"
    )


def main():

    print_separator()
    print("DEEP RESEARCH AGENT")
    print_separator()

    question = (
        "How has the performance of open-source large "
        "language models changed from 2023 to 2026, and "
        "which models currently provide the best balance "
        "of reasoning ability, coding performance, and "
        "local hardware requirements?"
    )

    print("\nUSER QUESTION")
    print("-" * 80)
    print(question)

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    initial_state = {
        "question": question,

        # --------------------------------------------------
        # RESEARCH PLAN
        # --------------------------------------------------

        "research_questions": [],

        # --------------------------------------------------
        # CURRENT RESEARCH QUESTION
        # --------------------------------------------------

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

        "current_question_index": 0,
        "research_complete": False,

        # --------------------------------------------------
        # ADAPTIVE RESEARCH
        # --------------------------------------------------

        "research_round": 1,
        "max_research_rounds": 3,
        "max_total_research_questions": 8,

        # --------------------------------------------------
        # SUFFICIENCY
        # --------------------------------------------------

        "research_gaps": [],
        "research_sufficient": False,
        "research_decision_reason": "",

        # --------------------------------------------------
        # FINAL OUTPUT
        # --------------------------------------------------

        "report": None,
    }

    # --------------------------------------------------------
    # BUILD GRAPH
    # --------------------------------------------------------

    print()
    print("Building research graph...")

    graph = build_research_graph()

    print("Graph compiled.")

    # --------------------------------------------------------
    # RUN RESEARCH
    # --------------------------------------------------------

    print()
    print("Starting research...")

    final_state = graph.invoke(
        initial_state
    )

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print()
    print("Research workflow finished.")

    print()

    print(
        f"Final research plan: "
        f"{len(final_state['research_questions'])} questions"
    )

    print(
        f"Sources collected: "
        f"{len(final_state['sources'])}"
    )

    print(
        f"Verified evidence: "
        f"{len(final_state['evidence'])}"
    )

    print()

    print_research_decision(
        final_state
    )

    print()

    print_summary(
        final_state
    )

    print()

    print_separator()
    print("RESEARCH PIPELINE COMPLETED")
    print_separator()


if __name__ == "__main__":
    main()
