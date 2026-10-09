"""
Entry point for the Deep Research Agent.

Runs the complete research workflow and displays:
- research progress summary
- research decision
- unresolved research gaps
- final evidence-linked research report
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

    print(
        f"\nReason: {state['research_decision_reason']}"
    )

    research_gaps = state["research_gaps"]

    print("\nResearch gaps:")

    if research_gaps:
        for gap in research_gaps:
            print(f"- {gap}")
    else:
        print("None")


def print_summary(
    state,
):
    print_separator()
    print("RESEARCH SUMMARY")
    print_separator()

    print(f"Research questions: {len(state['research_questions'])}")
    print(f"Sources collected: {len(state['sources'])}")
    print(f"Verified evidence: {len(state['evidence'])}")
    print(
        f"Research questions processed: "
        f"{state['current_question_index']}"
    )
    print(f"Research round: {state['research_round']}")
    print(f"Research complete: {state['research_complete']}")
    print(f"Research sufficient: {state['research_sufficient']}")


def print_final_report(
    state,
):
    print_separator()
    print("FINAL RESEARCH REPORT")
    print_separator()

    report = state.get("report")

    if report is None:
        print("No final research report was generated.")
        return

    # --------------------------------------------------------
    # TITLE AND SUMMARY
    # --------------------------------------------------------

    print(f"\nTitle:\n{report.title}")
    print(f"\nSummary:\n{report.summary}")

    # --------------------------------------------------------
    # FINDINGS AND CITATIONS
    # --------------------------------------------------------

    print("\nFINDINGS")

    evidence = state["evidence"]

    if not report.findings:
        print("No evidence-linked findings were generated.")

    for index, finding in enumerate(
        report.findings,
        start=1,
    ):
        print_separator(char="-")

        print(f"\nFinding {index}")
        print(finding.text)

        question_refs = ", ".join(
            f"Q{number}"
            for number in finding.research_question_numbers
        )

        evidence_refs = ", ".join(
            f"E{number}"
            for number in finding.evidence_numbers
        )

        print(f"\nResearch questions: {question_refs}")
        print(f"Evidence references: {evidence_refs}")

        print("\nSupporting evidence:")

        for evidence_number in finding.evidence_numbers:
            # The synthesizer validates these references before
            # placing the report into the final graph state.
            item = evidence[evidence_number - 1]

            print(f"\n[E{evidence_number}] {item.claim}")
            print(f"Research question: Q{item.research_question_number}")
            print(f"Supporting passage: {item.supporting_text}")
            print(f"Source: {item.source_url}")

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    print_separator(char="-")
    print("\nSOURCES")

    if not report.sources:
        print("No cited source URLs were included in the report.")

    for index, source in enumerate(
        report.sources,
        start=1,
    ):
        print(f"[{index}] {source}")

    # --------------------------------------------------------
    # RAW REPORT OBJECT
    # --------------------------------------------------------

    print_separator(char="-")
    print("\nFINAL REPORT OBJECT")
    print(report.model_dump_json(indent=2))


def main():
    print_separator()
    print("DEEP RESEARCH AGENT")
    print_separator()

    question = (
        "What are the most effective approaches for "
        "improving retrieval quality in RAG systems, "
        "and how do vector search, reranking, and "
        "hybrid retrieval compare?"
    )

    print("\nUSER QUESTION")
    print("-" * 80)
    print(question)

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    initial_state = {
        "question": question,
        "research_questions": [],
        "active_research_question": None,
        "active_research_question_number": None,
        "sources": [],
        "current_sources": [],
        "pending_evidence": [],
        "evidence": [],
        "current_question_index": 0,
        "research_complete": False,
        "research_round": 1,
        "max_research_rounds": 3,
        "max_total_research_questions": 10,
        "research_gaps": [],
        "research_sufficient": False,
        "research_decision_reason": "",
        "report": None,
    }

    # --------------------------------------------------------
    # BUILD AND RUN GRAPH
    # --------------------------------------------------------

    print("\nBuilding research graph...")

    graph = build_research_graph()

    print("Graph compiled.")
    print("\nStarting research...")

    final_state = graph.invoke(initial_state)

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print("\nResearch workflow finished.")

    print(
        f"\nFinal research plan: "
        f"{len(final_state['research_questions'])} questions"
    )

    print(f"Sources collected: {len(final_state['sources'])}")
    print(f"Verified evidence: {len(final_state['evidence'])}")

    print()
    print_research_decision(final_state)

    print()
    print_summary(final_state)

    print()
    print_final_report(final_state)

    print()
    print_separator()
    print("RESEARCH PIPELINE COMPLETED")
    print_separator()


if __name__ == "__main__":
    main()
