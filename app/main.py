"""
Entry point for the Deep Research Agent.

Runs the LangGraph research workflow with PostgreSQL persistence,
human-in-the-loop checkpoints, and observability.
"""

import json
import logging
import os
import time
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
    save_observability_event,
    save_questions,
    save_report,
    save_research_state,
    update_session_status,
)
from app.graph.graph import build_research_graph
from app.observability.tracing import (
    clear_session_context,
    configure_observability,
    get_run_metrics,
    reset_run_metrics,
    set_session_context,
)
from app.tools.citation_formatter import format_research_report


load_dotenv()

logger = logging.getLogger("deep_research_agent")


# ============================================================
# TERMINAL DISPLAY
# ============================================================

def print_separator(char: str = "─", width: int = 68) -> None:
    print(char * width)


def print_heading(title: str) -> None:
    print()
    print_separator()
    print(title)
    print_separator()


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


def print_research_decision(state: dict[str, Any]) -> None:
    """Print the final research coverage decision concisely."""

    print_heading("RESEARCH COVERAGE")

    print(f"Sufficient: {'Yes' if state.get('research_sufficient') else 'No'}")
    print(
        f"Research round: {state.get('research_round', 0)}"
        f"/{state.get('max_research_rounds', 0)}"
    )
    print(
        f"Questions: {len(state.get('research_questions', []))}"
        f"/{state.get('max_total_research_questions', 0)}"
    )

    reason = state.get("research_decision_reason", "")
    if reason:
        print(f"Reason: {reason}")

    gaps = state.get("research_gaps", [])

    if gaps:
        print("\nRemaining gaps:")
        for gap in gaps:
            print(f"  - {gap}")
    else:
        print("Remaining gaps: None reported")


def print_summary(state: dict[str, Any]) -> None:
    """Print a compact summary of research results."""

    print_heading("RUN SUMMARY")

    print(f"Research questions: {len(state.get('research_questions', []))}")
    print(f"Sources collected: {len(state.get('sources', []))}")
    print(f"Verified evidence: {len(state.get('evidence', []))}")
    print(f"Research complete: {state.get('research_complete', False)}")
    print(f"Research sufficient: {state.get('research_sufficient', False)}")


def print_final_report(state: dict[str, Any]) -> None:
    """Print the formatted final report once."""

    report = state.get("report")

    print_heading("FINAL RESEARCH REPORT")

    if report is None:
        print("No final research report was generated.")
        return

    formatted_report = format_research_report(
        report=report,
        evidence=state.get("evidence", []),
    )

    print(formatted_report)


# ============================================================
# OBSERVABILITY PERSISTENCE
# ============================================================

def persist_event_safely(
    event_type: str,
    message: str,
    session_id: UUID | None,
    *,
    log_level: str = "INFO",
    node_name: str | None = None,
    duration_ms: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Persist an event without interrupting the research workflow."""

    try:
        save_observability_event(
            event_type=event_type,
            message=message,
            session_id=session_id,
            log_level=log_level,
            node_name=node_name,
            duration_ms=duration_ms,
            details=details or {},
        )
    except Exception:
        # This fallback must not recursively enter the same
        # database logging handler.
        logging.StreamHandler().emit(
            logging.LogRecord(
                name="deep_research_agent.persistence",
                level=logging.ERROR,
                pathname=__file__,
                lineno=0,
                msg=f"Could not persist observability event: {event_type}",
                args=(),
                exc_info=None,
            )
        )


# ============================================================
# POSTGRESQL PERSISTENCE
# ============================================================

def persist_session_state(
    session_id: UUID,
    state: dict[str, Any],
) -> None:
    """Persist the latest research state and available results."""

    state_snapshot = dict(state)
    state_snapshot.pop("__interrupt__", None)

    save_research_state(
        session_id=session_id,
        state=to_json_compatible(state_snapshot),
    )

    processed_count = state.get("current_question_index", 0)
    questions = []

    for index, question in enumerate(
        state.get("research_questions", []),
        start=1,
    ):
        question_data = to_json_compatible(question)
        question_data["status"] = (
            "completed" if index <= processed_count else "pending"
        )
        questions.append(question_data)

    save_questions(
        session_id=session_id,
        questions=questions,
    )

    save_evidence(
        session_id=session_id,
        evidence_items=[
            to_json_compatible(item)
            for item in state.get("evidence", [])
        ],
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
# HUMAN REVIEW
# ============================================================

def print_review_payload(payload: dict[str, Any]) -> None:
    """Display a readable human-review request."""

    print_heading(payload.get("title", "Human Review Required").upper())
    print(payload.get("message", ""))

    research_question = payload.get("research_question")
    if research_question:
        print(f"\nResearch question: {research_question}")

    research_plan = payload.get(
        "research_plan",
        payload.get("proposed_questions", []),
    )

    if research_plan:
        print("\nProposed questions:")

        for item in research_plan:
            number = item.get("question_number", "?")
            question = item.get("question", "")
            print(f"  {number}. {question}")

            for query in item.get("search_queries", []):
                print(f"       Search: {query}")

    for field, label in (
        ("current_question_count", "Current questions"),
        ("maximum_question_count", "Maximum questions"),
        ("verified_evidence_count", "Verified evidence"),
    ):
        if payload.get(field) is not None:
            print(f"{label}: {payload[field]}")

    report = payload.get("report")

    if report is not None:
        preview = to_json_compatible(report)
        preview_text = json.dumps(
            preview,
            indent=2,
            ensure_ascii=False,
        )

        print("\nFinal report preview:")
        print(
            preview_text[:6000]
            + ("\n... preview truncated ..." if len(preview_text) > 6000 else "")
        )


def prompt_choice(prompt: str, choices: list[str]) -> str:
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

        if action == "approve":
            return {"action": "approve"}

        if action == "reject":
            reason = input("Reason (optional): ").strip()
            return {
                "action": "reject",
                "reason": reason or "Initial research plan rejected.",
            }

        questions = payload.get("proposed_questions", [])

        if not questions:
            print("No questions are available to revise.")
            return {"action": "approve"}

        while True:
            try:
                number = int(input("Question number to revise: ").strip())
                if 1 <= number <= len(questions):
                    break
                print(f"Enter a number between 1 and {len(questions)}.")
            except ValueError:
                print("Enter a valid integer.")

        revised_question = input("Enter the revised question: ").strip()

        while not revised_question:
            revised_question = input(
                "The question cannot be empty. Enter the revised question: "
            ).strip()

        existing_queries = questions[number - 1].get("search_queries", [])

        replace_queries = prompt_choice(
            "Replace the search queries too?",
            ["yes", "no"],
        )

        if replace_queries == "yes":
            print("Enter one query per line; submit an empty line to finish.")
            revised_queries = []

            while True:
                query = input("Search query: ").strip()
                if not query:
                    break
                revised_queries.append(query)

            if not revised_queries:
                revised_queries = existing_queries
                print("Keeping the existing search queries.")

        else:
            revised_queries = existing_queries

        return {
            "action": "revise",
            "question_number": number,
            "question": revised_question,
            "search_queries": revised_queries,
        }

    if review_type == "initial_research_review":
        action = prompt_choice(
            "Authorize the initial web research",
            ["approve", "reject"],
        )

        if action == "approve":
            return {"action": "approve"}

        reason = input("Reason (optional): ").strip()
        return {
            "action": "reject",
            "reason": reason or "Web research rejected by the reviewer.",
        }

    if review_type == "adaptive_research_review":
        action = prompt_choice(
            "Review proposed follow-up questions",
            ["approve", "reject"],
        )

        if action == "approve":
            return {"action": "approve"}

        reason = input("Reason (optional): ").strip()
        return {
            "action": "reject",
            "reason": reason or "Follow-up questions rejected.",
        }

    if review_type == "final_report_review":
        action = prompt_choice(
            "Review the final research report",
            ["approve", "research_more"],
        )

        if action == "approve":
            return {"action": "approve"}

        reason = input("What should be investigated further? ").strip()
        return {
            "action": "research_more",
            "reason": reason or "Reviewer requested additional research.",
        }

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
    session_id: UUID,
) -> dict[str, Any]:
    """Execute the graph and resume human-review interrupts."""

    result = graph.invoke(initial_state, config=config)

    while result.get("__interrupt__"):
        interrupts = result["__interrupt__"]

        if len(interrupts) != 1:
            raise RuntimeError(
                "Expected one pending human-review interrupt; "
                f"received {len(interrupts)}."
            )

        payload = interrupts[0].value

        if not isinstance(payload, dict):
            raise RuntimeError("Invalid human-review payload returned by graph.")

        decision = collect_review_decision(payload)

        # Persist this event explicitly. The tracing handler does not
        # persist human-review events, preventing duplicate rows.
        persist_event_safely(
            "human_review_submitted",
            "Human review submitted",
            session_id,
            details={
                "review_type": payload.get("type"),
                "review_action": decision.get("action"),
            },
        )

        print("\nResuming research...")

        result = graph.invoke(
            Command(resume=decision),
            config=config,
        )

    return result


# ============================================================
# OBSERVABILITY SUMMARY
# ============================================================

def print_observability_summary(elapsed_seconds: float) -> None:
    """Display concise node metrics."""

    metrics = get_run_metrics()
    nodes = metrics.get("nodes", {})

    print_heading("EXECUTION METRICS")
    print(f"Total runtime: {elapsed_seconds:.1f}s")

    if not nodes:
        print("No node metrics recorded.")
        return

    print()
    print(
        f"{'Node':<27}"
        f"{'Calls':>8}"
        f"{'OK':>8}"
        f"{'Failed':>9}"
        f"{'Time (s)':>12}"
    )
    print("-" * 64)

    for node_name, data in nodes.items():
        print(
            f"{node_name:<27}"
            f"{data.get('calls', 0):>8}"
            f"{data.get('successes', 0):>8}"
            f"{data.get('failures', 0):>9}"
            f"{data.get('duration_ms', 0.0) / 1000:>12.2f}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    configure_observability()
    reset_run_metrics()

    print_heading("DEEP RESEARCH AGENT")

    question = (
        "What are the most effective approaches for improving "
        "retrieval quality in RAG systems, and how do vector search, "
        "reranking, and hybrid retrieval compare?"
    )

    print(f"Question: {question}")

    session_id = create_session(question)
    start_time = time.perf_counter()
    set_session_context(str(session_id))

    print(f"Session:  {session_id}")

    # Persist session lifecycle events only through this function.
    # tracing.py persists node events only.
    persist_event_safely(
        "research_session_started",
        "Research session started",
        session_id,
        details={"research_question": question},
    )

    try:
        update_session_status(
            session_id=session_id,
            status="running",
        )

        initial_state = {
            "question": question,
            "require_initial_plan_approval": True,
            "require_initial_research_approval": True,
            "require_adaptive_research_approval": True,
            "require_final_report_approval": True,
            "initial_plan_approved": False,
            "initial_research_approved": False,
            "research_authorized": False,
            "final_report_approved": False,
            "final_report_review_decision": None,
            "research_questions": [],
            "proposed_research_questions": [],
            "adaptive_review_decision": None,
            "active_research_question": None,
            "active_research_question_number": None,
            "current_sources": [],
            "sources": [],
            "pending_evidence": [],
            "evidence": [],
            "current_question_index": 0,
            "research_complete": False,
            "research_round": 1,
            "max_research_rounds": 3,
            "max_total_research_questions": 10,
            "coverage_assessments": [],
            "research_gaps": [],
            "research_sufficient": False,
            "research_decision_reason": "",
            "report": None,
        }

        logger.info("Preparing research workflow")
        print("\nPreparing research workflow...")

        with PostgresSaver.from_conn_string(
            get_postgres_uri()
        ) as checkpointer:
            checkpointer.setup()

            graph = build_research_graph(checkpointer=checkpointer)

            config = {
                "configurable": {
                    "thread_id": str(session_id),
                }
            }

            print("Research workflow ready.")
            print("\nResearch started. Review requests will appear when needed.")

            final_state = run_research_graph(
                graph=graph,
                initial_state=initial_state,
                config=config,
                session_id=session_id,
            )

        print("\nSaving results...")

        persist_session_state(
            session_id=session_id,
            state=final_state,
        )

        update_session_status(
            session_id=session_id,
            status="completed",
        )

        completion_details = {
            "source_count": len(final_state.get("sources", [])),
            "evidence_count": len(final_state.get("evidence", [])),
            "question_count": len(final_state.get("research_questions", [])),
        }

        persist_event_safely(
            "research_session_completed",
            "Research session completed",
            session_id,
            details=completion_details,
        )

        print("Results saved.")

    except Exception as error:
        # Persist failure once, explicitly. Do not emit the same
        # lifecycle event through logger.extra as well.
        persist_event_safely(
            "research_session_failed",
            "Research session failed",
            session_id,
            log_level="ERROR",
            details={"error": str(error)[:4000]},
        )

        logger.exception("Research session failed")

        try:
            update_session_status(
                session_id=session_id,
                status="failed",
                error_message=str(error)[:4000],
            )
        except Exception:
            logger.exception("Could not persist failure status")

        print(f"\nResearch failed. Session: {session_id}")
        raise

    finally:
        elapsed_seconds = time.perf_counter() - start_time

        persist_event_safely(
            "research_session_ended",
            "Research session execution ended",
            session_id,
            duration_ms=round(elapsed_seconds * 1000, 2),
        )

        print_observability_summary(elapsed_seconds)
        clear_session_context()

    print_summary(final_state)
    print_research_decision(final_state)
    print_final_report(final_state)

    print_heading("RESEARCH COMPLETED")
    print(f"Session: {session_id}")


if __name__ == "__main__":
    main()
