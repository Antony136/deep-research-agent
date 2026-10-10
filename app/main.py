"""
Entry point for the Deep Research Agent.

Runs the research workflow and persists each completed
research session to PostgreSQL.
"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel

from app.database.repository import (
    create_session,
    save_evidence,
    save_questions,
    save_report,
    save_research_state,
    update_session_status,
)
from app.graph.graph import build_research_graph
from app.tools.citation_formatter import format_research_report


def print_separator(char="=", width=80):
    print(char * width)


def to_json_compatible(value: Any) -> Any:
    """
    Convert Pydantic models and nested values into
    JSON-compatible Python objects for PostgreSQL JSONB.
    """

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    if isinstance(value, dict):
        return {
            str(key): to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_json_compatible(item)
            for item in value
        ]

    if isinstance(value, UUID):
        return str(value)

    return value


def print_research_decision(state):
    print_separator()
    print("RESEARCH DECISION")
    print_separator()

    print(f"Research sufficient: {state['research_sufficient']}")
    print(
        f"Research round: {state['research_round']}/"
        f"{state['max_research_rounds']}"
    )
    print(
        f"Research questions: {len(state['research_questions'])}/"
        f"{state['max_total_research_questions']}"
    )
    print(f"\nReason: {state['research_decision_reason']}")

    research_gaps = state.get("research_gaps", [])

    print("\nResearch gaps:")

    if research_gaps:
        for gap in research_gaps:
            print(f"- {gap}")
    else:
        print("None")


def print_summary(state):
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


def print_final_report(state):
    print_separator()
    print("FINAL RESEARCH REPORT")
    print_separator()

    report = state.get("report")

    if report is None:
        print("No final research report was generated.")
        return

    formatted_report = format_research_report(
        report=report,
        evidence=state.get("evidence", []),
    )

    print()
    print(formatted_report)

    print()
    print_separator()
    print("FINAL REPORT OBJECT")
    print(report.model_dump_json(indent=2))


def persist_completed_session(
    session_id: UUID,
    state: dict[str, Any],
) -> None:
    """
    Persist the completed research state, questions,
    verified evidence, and final report.
    """

    state_snapshot = to_json_compatible(state)

    # Save the complete graph state.
    save_research_state(
        session_id=session_id,
        state=state_snapshot,
    )

    # Save original and adaptive research questions.
    processed_count = state["current_question_index"]

    questions = []

    for index, question in enumerate(
        state["research_questions"],
        start=1,
    ):
        question_data = question.model_dump(mode="json")

        question_data["status"] = (
            "completed"
            if index <= processed_count
            else "pending"
        )

        questions.append(question_data)

    save_questions(
        session_id=session_id,
        questions=questions,
    )

    # Persist only evidence that passed verification.
    evidence_items = [
        item.model_dump(mode="json")
        for item in state["evidence"]
    ]

    save_evidence(
        session_id=session_id,
        evidence_items=evidence_items,
    )

    # Persist the final structured report, if available.
    report = state.get("report")

    if report is not None:
        save_report(
            session_id=session_id,
            report=report.model_dump(mode="json"),
        )


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
    # CREATE PERSISTENT SESSION
    # --------------------------------------------------------

    session_id = create_session(question)

    print(f"\nResearch session ID: {session_id}")

    try:
        update_session_status(
            session_id=session_id,
            status="running",
        )

        # ----------------------------------------------------
        # INITIAL GRAPH STATE
        # ----------------------------------------------------

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
            "coverage_assessments": [],
            "research_sufficient": False,
            "research_decision_reason": "",
            "report": None,
        }

        # ----------------------------------------------------
        # BUILD AND RUN GRAPH
        # ----------------------------------------------------

        print("\nBuilding research graph...")

        graph = build_research_graph()

        print("Graph compiled.")
        print("\nStarting research...")

        final_state = graph.invoke(initial_state)

        # ----------------------------------------------------
        # PERSIST RESULTS
        # ----------------------------------------------------

        print("\nPersisting research results to PostgreSQL...")

        persist_completed_session(
            session_id=session_id,
            state=final_state,
        )

        update_session_status(
            session_id=session_id,
            status="completed",
        )

        print("Research session saved successfully.")

    except Exception as error:
        # Record the failure without hiding the original error.
        try:
            update_session_status(
                session_id=session_id,
                status="failed",
                error_message=str(error)[:4000],
            )
        except Exception as persistence_error:
            print(
                "\nWARNING: Could not record the session failure: "
                f"{persistence_error}"
            )

        print(f"\nResearch session failed: {session_id}")
        raise

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
    print(f"Persistent session ID: {session_id}")
    print_separator()


if __name__ == "__main__":
    main()
