"""
Entry point for the Deep Research Agent.

Runs the LangGraph research workflow with PostgreSQL persistence
and resumable human-in-the-loop checkpoints.

Supported reviews:
1. Initial research plan: approve, revise, or reject.
2. Initial web research: approve or reject.
3. Adaptive follow-up questions: approve or reject.
4. Final report: approve or request more research.
"""

import json
import os
from typing import Any
from urllib.parse import quote
from uuid import UUID

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
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


load_dotenv()


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_separator(char="=", width=80):
    print(char * width)


def to_json_compatible(value: Any) -> Any:
    """Convert nested values into JSON-compatible Python objects."""

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

    if hasattr(value, "model_dump"):
        return to_json_compatible(value.model_dump(mode="json"))

    return value


def print_research_decision(state):
    print_separator()
    print("RESEARCH DECISION")
    print_separator()

    print(f"Research sufficient: {state.get('research_sufficient', False)}")
    print(
        f"Research round: {state.get('research_round', 0)}/"
        f"{state.get('max_research_rounds', 0)}"
    )
    print(
        f"Research questions: {len(state.get('research_questions', []))}/"
        f"{state.get('max_total_research_questions', 0)}"
    )
    print(f"\nReason: {state.get('research_decision_reason', '')}")

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

    print(f"Research questions: {len(state.get('research_questions', []))}")
    print(f"Sources collected: {len(state.get('sources', []))}")
    print(f"Verified evidence: {len(state.get('evidence', []))}")
    print(
        "Research questions processed: "
        f"{state.get('current_question_index', 0)}"
    )
    print(f"Research round: {state.get('research_round', 0)}")
    print(f"Research complete: {state.get('research_complete', False)}")
    print(f"Research sufficient: {state.get('research_sufficient', False)}")


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
    print_separator()

    if isinstance(report, BaseModel):
        print(report.model_dump_json(indent=2))
    else:
        print(json.dumps(to_json_compatible(report), indent=2))


# ============================================================
# POSTGRESQL PERSISTENCE
# ============================================================

def persist_session_state(
    session_id: UUID,
    state: dict[str, Any],
) -> None:
    """
    Persist the latest available research state and results.

    This function supports both interrupted and completed
    sessions. LangGraph's __interrupt__ metadata is excluded
    from the application's JSONB state snapshot.
    """

    state_snapshot = dict(state)
    state_snapshot.pop("__interrupt__", None)

    state_snapshot = to_json_compatible(state_snapshot)

    save_research_state(
        session_id=session_id,
        state=state_snapshot,
    )

    processed_count = state.get("current_question_index", 0)

    questions = []

    for index, question in enumerate(
        state.get("research_questions", []),
        start=1,
    ):
        if isinstance(question, BaseModel):
            question_data = question.model_dump(mode="json")
        else:
            question_data = to_json_compatible(question)

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

    evidence_items = [
        to_json_compatible(item)
        for item in state.get("evidence", [])
    ]

    save_evidence(
        session_id=session_id,
        evidence_items=evidence_items,
    )

    report = state.get("report")

    if report is not None:
        save_report(
            session_id=session_id,
            report=to_json_compatible(report),
        )


def get_postgres_uri() -> str:
    """Build a PostgreSQL URI with URL-encoded credentials."""

    user = quote(os.environ["POSTGRES_USER"], safe="")
    password = quote(os.environ["POSTGRES_PASSWORD"], safe="")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = quote(
        os.getenv("POSTGRES_DB", "deep_research_agent"),
        safe="",
    )

    return (
        f"postgresql://{user}:{password}"
        f"@{host}:{port}/{database}"
        "?sslmode=disable"
    )


# ============================================================
# HUMAN REVIEW HELPERS
# ============================================================

def print_review_payload(payload: dict[str, Any]) -> None:
    """Display a human-readable review request."""

    print()
    print_separator()
    print(payload.get("title", "Human Review Required").upper())
    print_separator()

    print(payload.get("message", ""))

    research_question = payload.get("research_question")

    if research_question:
        print(f"\nResearch question: {research_question}")

    research_plan = payload.get(
        "research_plan",
        payload.get("proposed_questions", []),
    )

    if research_plan:
        print("\nQuestions:")

        for item in research_plan:
            question_number = item.get("question_number", "?")
            question_text = item.get("question", "")

            print(f"\n  {question_number}. {question_text}")

            queries = item.get("search_queries", [])

            for query in queries:
                print(f"       Search: {query}")

    if payload.get("current_question_count") is not None:
        print(
            "\nCurrent questions: "
            f"{payload['current_question_count']}"
        )

    if payload.get("maximum_question_count") is not None:
        print(
            "Maximum questions: "
            f"{payload['maximum_question_count']}"
        )

    if payload.get("verified_evidence_count") is not None:
        print(
            "Verified evidence items: "
            f"{payload['verified_evidence_count']}"
        )

    report = payload.get("report")

    if report is not None:
        print("\nA final research report is ready for review.")

        if isinstance(report, BaseModel):
            preview = report.model_dump(mode="json")
        else:
            preview = to_json_compatible(report)

        print(
            json.dumps(preview, indent=2, ensure_ascii=False)[:12000]
        )


def prompt_choice(
    prompt: str,
    choices: list[str],
) -> str:
    """Prompt until the user selects a supported choice."""

    allowed = {choice.lower() for choice in choices}

    while True:
        answer = input(
            f"\n{prompt} ({'/'.join(choices)}): "
        ).strip().lower()

        if answer in allowed:
            return answer

        print(f"Please enter one of: {', '.join(choices)}")


def collect_review_decision(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Collect the appropriate decision for a graph interrupt."""

    review_type = payload.get("type")

    print_review_payload(payload)

    if review_type == "initial_plan_review":
        action = prompt_choice(
            "Review the initial research plan",
            ["approve", "revise", "reject"],
        )

        if action == "revise":
            questions = payload.get("proposed_questions", [])

            while True:
                try:
                    number = int(
                        input(
                            "Question number to revise: "
                        ).strip()
                    )

                    if 1 <= number <= len(questions):
                        break

                    print(
                        f"Enter a number between 1 and {len(questions)}."
                    )

                except ValueError:
                    print("Enter a valid integer.")

            revised_question = input(
                "Enter the revised question: "
            ).strip()

            while not revised_question:
                print("The revised question cannot be empty.")
                revised_question = input(
                    "Enter the revised question: "
                ).strip()

            existing_queries = questions[number - 1].get(
                "search_queries",
                [],
            )

            replace_queries = prompt_choice(
                "Replace the search queries too?",
                ["yes", "no"],
            )

            if replace_queries == "yes":
                print(
                    "Enter one search query per line. "
                    "Submit an empty line to finish."
                )

                revised_queries = []

                while True:
                    query = input("Search query: ").strip()

                    if not query:
                        break

                    revised_queries.append(query)

                if not revised_queries:
                    print(
                        "No queries entered; keeping the existing queries."
                    )
                    revised_queries = existing_queries
            else:
                revised_queries = existing_queries

            return {
                "action": "revise",
                "question_number": number,
                "question": revised_question,
                "search_queries": revised_queries,
            }

        if action == "reject":
            reason = input(
                "Reason for rejecting the plan (optional): "
            ).strip()

            return {
                "action": "reject",
                "reason": reason or "Initial research plan rejected.",
            }

        return {"action": "approve"}

    if review_type == "initial_research_review":
        action = prompt_choice(
            "Authorize the initial web research",
            ["approve", "reject"],
        )

        if action == "reject":
            reason = input(
                "Reason for rejecting web research (optional): "
            ).strip()

            return {
                "action": "reject",
                "reason": reason or "Web research rejected by the reviewer.",
            }

        return {"action": "approve"}

    if review_type == "adaptive_research_review":
        action = prompt_choice(
            "Review proposed follow-up questions",
            ["approve", "reject"],
        )

        if action == "reject":
            reason = input(
                "Reason for rejecting follow-up questions (optional): "
            ).strip()

            return {
                "action": "reject",
                "reason": reason or "Follow-up questions rejected.",
            }

        return {"action": "approve"}

    if review_type == "final_report_review":
        action = prompt_choice(
            "Review the final research report",
            ["approve", "research_more"],
        )

        if action == "research_more":
            reason = input(
                "What should the agent investigate further? "
                "(optional): "
            ).strip()

            return {
                "action": "research_more",
                "reason": reason or "Reviewer requested additional research.",
            }

        return {"action": "approve"}

    raise ValueError(
        f"Unsupported human-review interrupt type: {review_type!r}"
    )


# ============================================================
# GRAPH EXECUTION
# ============================================================

def run_research_graph(
    graph,
    initial_state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute the graph and resume interrupts on the same thread.

    The checkpointer stores completed node outputs, so resuming
    does not restart the workflow from the initial state.
    """

    result = graph.invoke(
        initial_state,
        config=config,
    )

    while result.get("__interrupt__"):
        interrupts = result["__interrupt__"]

        if len(interrupts) != 1:
            raise RuntimeError(
                "Expected exactly one pending human-review interrupt; "
                f"received {len(interrupts)}."
            )

        interrupt_item = interrupts[0]
        payload = interrupt_item.value

        if not isinstance(payload, dict):
            raise RuntimeError(
                "The graph returned an invalid human-review payload."
            )

        decision = collect_review_decision(payload)

        print("\nResuming the existing research session...")

        result = graph.invoke(
            Command(resume=decision),
            config=config,
        )

    return result


# ============================================================
# MAIN
# ============================================================

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

            # Human-in-the-loop settings
            "require_initial_plan_approval": True,
            "require_initial_research_approval": True,
            "require_adaptive_research_approval": True,
            "require_final_report_approval": True,

            # Approval decisions
            "initial_plan_approved": False,
            "initial_research_approved": False,
            "research_authorized": False,
            "final_report_approved": False,
            "final_report_review_decision": None,

            # Research plan
            "research_questions": [],
            "proposed_research_questions": [],
            "adaptive_review_decision": None,

            # Active research
            "active_research_question": None,
            "active_research_question_number": None,
            "current_sources": [],

            # Research data
            "sources": [],
            "pending_evidence": [],
            "evidence": [],

            # Workflow control
            "current_question_index": 0,
            "research_complete": False,

            # Research limits
            "research_round": 1,
            "max_research_rounds": 3,
            "max_total_research_questions": 10,

            # Sufficiency
            "coverage_assessments": [],
            "research_gaps": [],
            "research_sufficient": False,
            "research_decision_reason": "",

            # Final output
            "report": None,
        }

        # ----------------------------------------------------
        # BUILD AND RUN CHECKPOINTED GRAPH
        # ----------------------------------------------------

        print("\nBuilding research graph...")

        with PostgresSaver.from_conn_string(
            get_postgres_uri()
        ) as checkpointer:

            checkpointer.setup()

            graph = build_research_graph(
                checkpointer=checkpointer,
            )

            print("Graph compiled.")

            config = {
                "configurable": {
                    "thread_id": str(session_id),
                }
            }

            print("\nStarting research workflow...")

            final_state = run_research_graph(
                graph=graph,
                initial_state=initial_state,
                config=config,
            )

        # ----------------------------------------------------
        # PERSIST FINAL STATE
        # ----------------------------------------------------

        print("\nPersisting research results to PostgreSQL...")

        persist_session_state(
            session_id=session_id,
            state=final_state,
        )

        update_session_status(
            session_id=session_id,
            status="completed",
        )

        print("Research session saved successfully.")

    except Exception as error:
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
        f"{len(final_state.get('research_questions', []))} questions"
    )
    print(f"Sources collected: {len(final_state.get('sources', []))}")
    print(f"Verified evidence: {len(final_state.get('evidence', []))}")

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
