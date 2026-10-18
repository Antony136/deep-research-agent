"""
Orchestration service for the Deep Research Agent API.

Starts research sessions, initializes complete graph state,
persists progress, handles human-review interrupts, and resumes
research from PostgreSQL checkpoints.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Callable
from uuid import UUID

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from psycopg.conninfo import make_conninfo

from app.database.repository import (
    create_session,
    get_session,
    list_sessions,
    save_evidence,
    save_questions,
    save_report,
    save_research_state,
    update_session_status,
)
from app.graph.graph import build_research_graph

load_dotenv()

logger = logging.getLogger(__name__)


class ResearchCancelled(Exception):
    """Raised when a worker is asked to stop at a graph checkpoint."""


# ---------------------------------------------------------
# DATABASE CONFIGURATION
# ---------------------------------------------------------

def get_postgres_uri() -> str:
    """Build the PostgreSQL connection URI for LangGraph."""

    password = os.getenv("POSTGRES_PASSWORD")

    if not password:
        raise ValueError(
            "POSTGRES_PASSWORD is not configured."
        )

    return make_conninfo(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv(
            "POSTGRES_DB",
            "deep_research_agent",
        ),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=password,
    )


@contextmanager
def get_checkpointer():
    """Open a PostgreSQL-backed LangGraph checkpointer."""

    with PostgresSaver.from_conn_string(
        get_postgres_uri()
    ) as checkpointer:
        checkpointer.setup()
        yield checkpointer


# ---------------------------------------------------------
# SERIALIZATION
# ---------------------------------------------------------

def to_json_compatible(value: Any) -> Any:
    """Convert graph state values into JSON-compatible data."""

    if value is None:
        return None

    if hasattr(value, "model_dump"):
        return to_json_compatible(
            value.model_dump(mode="json")
        )

    if hasattr(value, "dict") and callable(value.dict):
        return to_json_compatible(value.dict())

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

    if isinstance(value, set):
        return [
            to_json_compatible(item)
            for item in value
        ]

    if isinstance(value, UUID):
        return str(value)

    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()

    if isinstance(value, (str, int, float, bool)):
        return value

    raise TypeError(
        "Unsupported value for JSON serialization: "
        f"{type(value).__name__}"
    )


# ---------------------------------------------------------
# INITIAL STATE
# ---------------------------------------------------------

def build_initial_state(
    question: str,
    require_initial_plan_approval: bool = True,
    require_initial_research_approval: bool = True,
    require_adaptive_research_approval: bool = True,
    require_final_report_approval: bool = True,
    max_research_rounds: int = 3,
    max_total_research_questions: int = 10,
) -> dict[str, Any]:
    """
    Build the complete initial state expected by ResearchState.

    TypedDict documents the state structure but does not
    automatically populate missing keys at runtime.
    """

    return {
        # User request
        "question": question,

        # Human-review settings
        "require_initial_plan_approval": (
            require_initial_plan_approval
        ),
        "require_initial_research_approval": (
            require_initial_research_approval
        ),
        "require_adaptive_research_approval": (
            require_adaptive_research_approval
        ),
        "require_final_report_approval": (
            require_final_report_approval
        ),

        # Review decisions
        "initial_plan_approved": False,
        "initial_research_approved": False,
        "research_authorized": False,
        "final_report_approved": False,
        "final_report_review_decision": None,

        # Research plan
        "research_questions": [],
        "proposed_research_questions": [],
        "adaptive_review_decision": None,

        # Active question
        "active_research_question": None,
        "active_research_question_number": None,

        # Sources and evidence
        "current_sources": [],
        "sources": [],
        "pending_evidence": [],
        "evidence": [],

        # Workflow control
        "current_question_index": 0,
        "research_complete": False,

        # Research limits
        "research_round": 0,
        "max_research_rounds": max_research_rounds,
        "max_total_research_questions": (
            max_total_research_questions
        ),

        # Sufficiency assessments
        "coverage_assessments": [],
        "research_gaps": [],
        "research_sufficient": False,
        "research_decision_reason": "",

        # Final output
        "report": None,
    }


# ---------------------------------------------------------
# STATE PERSISTENCE
# ---------------------------------------------------------

def persist_session_state(
    session_id: UUID,
    state: dict[str, Any],
) -> None:
    """Persist the latest graph state and related records."""

    serializable_state = to_json_compatible(state)

    serializable_state.pop("__interrupt__", None)

    save_research_state(
        session_id,
        serializable_state,
    )

    save_questions(
        session_id,
        serializable_state.get(
            "research_questions",
            [],
        ),
    )

    save_evidence(
        session_id,
        serializable_state.get(
            "evidence",
            [],
        ),
    )

    report = serializable_state.get("report")

    if isinstance(report, dict):
        save_report(session_id, report)


# ---------------------------------------------------------
# GRAPH CONFIGURATION
# ---------------------------------------------------------

def get_graph_config(
    session_id: UUID,
) -> dict[str, Any]:
    """Return the stable LangGraph thread configuration."""

    return {
        "configurable": {
            "thread_id": str(session_id),
        }
    }


def extract_interrupts(
    result: dict[str, Any],
) -> list[Any]:
    """Extract interrupt payloads returned by LangGraph."""

    interrupts = result.get("__interrupt__", [])

    return [
        to_json_compatible(interrupt.value)
        for interrupt in interrupts
    ]


def get_checkpoint_interrupts(
    graph: Any,
    config: dict[str, Any],
) -> list[Any]:
    """Read pending interrupts from a saved checkpoint."""

    snapshot = graph.get_state(config)

    payloads = []

    for task in snapshot.tasks:
        for interrupt in task.interrupts:
            payloads.append(
                to_json_compatible(interrupt.value)
            )

    return payloads


def get_latest_graph_state(
    graph: Any,
    config: dict[str, Any],
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Read the latest full state from LangGraph's checkpoint.

    A node's result or update is not necessarily the complete
    state. The checkpoint snapshot is the preferred source.
    """

    snapshot = graph.get_state(config)

    if snapshot is not None and snapshot.values:
        return dict(snapshot.values)

    if fallback is not None:
        return {
            key: value
            for key, value in fallback.items()
            if key != "__interrupt__"
        }

    return {}


def stream_graph_updates(
    graph: Any,
    graph_input: Any,
    config: dict[str, Any],
    session_id: UUID,
    cancellation_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Persist each completed node update while the graph is running."""

    if cancellation_check is not None:
        cancellation_check()

    state = get_latest_graph_state(graph, config)
    if not state and isinstance(graph_input, dict):
        state = dict(graph_input)

    for update_chunk in graph.stream(
        graph_input,
        config=config,
        stream_mode="updates",
    ):
        for node_update in update_chunk.values():
            if isinstance(node_update, dict):
                state.update(node_update)
        if state:
            persist_session_state(session_id, state)
        if cancellation_check is not None:
            cancellation_check()

    return get_latest_graph_state(graph, config, fallback=state)


# ---------------------------------------------------------
# REVIEW VALIDATION
# ---------------------------------------------------------

REVIEW_ACTIONS = {
    "initial_plan_review": {
        "approve",
        "revise",
        "reject",
    },
    "initial_research_review": {
        "approve",
        "reject",
    },
    "adaptive_research_review": {
        "approve",
        "reject",
    },
    "final_report_review": {
        "approve",
        "research_more",
    },
}


def validate_review_decision(
    review: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Validate a decision against the active review type."""

    review_type = review.get("type")
    action = decision.get("action")

    allowed_actions = REVIEW_ACTIONS.get(review_type)

    if allowed_actions is None:
        raise ValueError(
            f"Unsupported review type: {review_type}"
        )

    if action not in allowed_actions:
        raise ValueError(
            f"Action '{action}' is not valid for "
            f"review type '{review_type}'. "
            f"Allowed actions: {sorted(allowed_actions)}"
        )

    normalized_decision = dict(decision)

    if action == "revise":
        question_number = normalized_decision.get(
            "question_number"
        )
        question = normalized_decision.get("question")

        if (
            not isinstance(question_number, int)
            or isinstance(question_number, bool)
            or question_number < 1
        ):
            raise ValueError(
                "question_number must be a positive integer."
            )

        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            raise ValueError(
                "A non-empty question is required for revision."
            )

        normalized_decision["question"] = question.strip()

        queries = normalized_decision.get(
            "search_queries",
            [],
        )

        if not isinstance(queries, list):
            raise ValueError(
                "search_queries must be a list."
            )

        normalized_decision["search_queries"] = [
            query.strip()
            for query in queries
            if isinstance(query, str) and query.strip()
        ]

    if action == "reject":
        reason = normalized_decision.get("reason")

        if reason is not None:
            normalized_decision["reason"] = str(reason).strip()

    return normalized_decision


# ---------------------------------------------------------
# SESSION EXECUTION
# ---------------------------------------------------------

def execute_research(
    session_id: UUID,
    initial_state: dict[str, Any] | None = None,
    resume_decision: dict[str, Any] | None = None,
    *,
    raise_errors: bool = False,
    manage_session_status: bool = True,
    cancellation_check: Callable[[], None] | None = None,
) -> str:
    """
    Execute or resume a research session.

    This function is intended to run in a background worker.
    It returns when the graph completes or reaches a review
    interrupt.
    """

    config = get_graph_config(session_id)

    try:
        if manage_session_status:
            update_session_status(session_id, "running")

        if cancellation_check is not None:
            cancellation_check()

        with get_checkpointer() as checkpointer:
            graph = build_research_graph(
                checkpointer=checkpointer,
            )
            snapshot = graph.get_state(config)
            pending_reviews = get_checkpoint_interrupts(
                graph,
                config,
            )

            if resume_decision is not None and pending_reviews:
                if len(pending_reviews) != 1:
                    raise ValueError(
                        "Expected exactly one pending review."
                    )

                decision = validate_review_decision(
                    pending_reviews[0],
                    resume_decision,
                )

                result = stream_graph_updates(
                    graph,
                    Command(resume=decision),
                    config,
                    session_id,
                    cancellation_check,
                )

            elif snapshot.next:
                # A worker may have died after LangGraph committed the
                # previous node. Continue from that checkpoint, not from
                # the original input, to avoid replaying completed nodes.
                result = stream_graph_updates(
                    graph,
                    None,
                    config,
                    session_id,
                    cancellation_check,
                )

            elif snapshot.values:
                # The graph may have finished before the previous worker
                # could mark the durable job complete.
                result = snapshot.values

            elif resume_decision is not None:
                raise ValueError(
                    "The session has no resumable graph checkpoint."
                )

            else:
                if initial_state is None:
                    raise ValueError(
                        "Initial research state is required."
                    )

                # Ensure all default state keys are present even if caller
                # provided a partial dictionary.
                base_state = build_initial_state(
                    question=initial_state.get("question", ""),
                    require_initial_plan_approval=initial_state.get(
                        "require_initial_plan_approval", True
                    ),
                    require_initial_research_approval=initial_state.get(
                        "require_initial_research_approval", True
                    ),
                    require_adaptive_research_approval=initial_state.get(
                        "require_adaptive_research_approval", True
                    ),
                    require_final_report_approval=initial_state.get(
                        "require_final_report_approval", True
                    ),
                    max_research_rounds=initial_state.get(
                        "max_research_rounds", 3
                    ),
                    max_total_research_questions=initial_state.get(
                        "max_total_research_questions", 10
                    ),
                )
                base_state.update(initial_state)
                initial_state = base_state

                result = stream_graph_updates(
                    graph,
                    initial_state,
                    config,
                    session_id,
                    cancellation_check,
                )

            while True:
                if cancellation_check is not None:
                    cancellation_check()

                state = get_latest_graph_state(
                    graph=graph,
                    config=config,
                    fallback=result,
                )

                persist_session_state(
                    session_id,
                    state,
                )

                interrupts = get_checkpoint_interrupts(
                    graph,
                    config,
                )

                # The graph is paused at a human-review node.
                if interrupts:
                    if len(interrupts) != 1:
                        raise ValueError(
                            "Multiple simultaneous review interrupts "
                            "are not supported."
                        )

                    logger.info(
                        "Research session %s is waiting for review: %s",
                        session_id,
                        interrupts[0].get("type", "unknown"),
                    )

                    if manage_session_status:
                        update_session_status(
                            session_id,
                            "waiting_for_review",
                        )

                    return "awaiting_review"

                # No pending review: inspect the graph's tasks.
                snapshot = graph.get_state(config)

                if snapshot.next:
                    # A graph can have more work scheduled without
                    # an interrupt. Continue from its checkpoint.
                    result = stream_graph_updates(
                        graph,
                        None,
                        config,
                        session_id,
                        cancellation_check,
                    )
                    continue

                break

        if manage_session_status:
            update_session_status(session_id, "completed")

        logger.info(
            "Research session %s completed.",
            session_id,
        )
        return "completed"

    except Exception as exc:
        if raise_errors:
            raise

        logger.exception(
            "Research session %s failed.",
            session_id,
        )

        try:
            update_session_status(
                session_id,
                "failed",
                error_message=str(exc)[:2000],
            )
        except Exception:
            logger.exception(
                "Could not persist failure status for session %s.",
                session_id,
            )

        return "failed"


# ---------------------------------------------------------
# PUBLIC SERVICE FUNCTIONS
# ---------------------------------------------------------

def start_research(
    question: str,
    require_initial_plan_approval: bool = True,
    require_initial_research_approval: bool = True,
    require_adaptive_research_approval: bool = True,
    require_final_report_approval: bool = True,
    max_research_rounds: int = 3,
    max_total_research_questions: int = 10,
) -> UUID:
    """Create a research session and return its ID."""

    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError(
            "Research question cannot be empty."
        )

    session_id = create_session(normalized_question)

    initial_state = build_initial_state(
        question=normalized_question,
        require_initial_plan_approval=(
            require_initial_plan_approval
        ),
        require_initial_research_approval=(
            require_initial_research_approval
        ),
        require_adaptive_research_approval=(
            require_adaptive_research_approval
        ),
        require_final_report_approval=(
            require_final_report_approval
        ),
        max_research_rounds=max_research_rounds,
        max_total_research_questions=(
            max_total_research_questions
        ),
    )

    execute_research(
        session_id=session_id,
        initial_state=initial_state,
    )

    return session_id


def submit_review(
    session_id: UUID,
    decision: dict[str, Any],
) -> None:
    """Resume a session using a submitted human decision."""

    session = get_session(session_id)

    if session is None:
        raise ValueError(
            f"Research session not found: {session_id}"
        )

    if session["status"] != "waiting_for_review":
        raise ValueError(
            "The session is not waiting for human review."
        )

    execute_research(
        session_id=session_id,
        resume_decision=decision,
    )


def get_research_session(
    session_id: UUID,
) -> dict[str, Any] | None:
    """Retrieve a persisted research session."""

    return get_session(session_id)


def list_research_sessions(
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List recent research sessions."""

    return list_sessions(limit=limit)


def get_pending_review(
    session_id: UUID,
) -> dict[str, Any] | None:
    """Return the review payload for a paused session."""

    session = get_session(session_id)

    if session is None:
        raise ValueError(
            f"Research session not found: {session_id}"
        )

    if session["status"] != "waiting_for_review":
        return None

    config = get_graph_config(session_id)

    with get_checkpointer() as checkpointer:
        graph = build_research_graph(
            checkpointer=checkpointer,
        )

        reviews = get_checkpoint_interrupts(
            graph,
            config,
        )

    if not reviews:
        return None

    return reviews[0]
