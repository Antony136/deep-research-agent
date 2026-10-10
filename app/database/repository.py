"""
Repository operations for persistent research sessions.
"""

from typing import Any
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb

from app.database.connection import get_connection


def create_session(question: str) -> UUID:
    """Create a new research session and return its ID."""

    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("Research question cannot be empty.")

    session_id = uuid4()

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO research_sessions (
                    id,
                    question,
                    status
                )
                VALUES (%s, %s, 'created')
                """,
                (session_id, normalized_question),
            )

    return session_id


def update_session_status(
    session_id: UUID,
    status: str,
    error_message: str | None = None,
) -> None:
    """Update a session's status and optional error message."""

    allowed_statuses = {
        "created",
        "running",
        "completed",
        "failed",
    }

    if status not in allowed_statuses:
        raise ValueError(f"Invalid session status: {status}")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE research_sessions
                SET
                    status = %s,
                    error_message = %s,
                    updated_at = NOW(),
                    completed_at = CASE
                        WHEN %s IN ('completed', 'failed')
                            THEN NOW()
                        ELSE NULL
                    END
                WHERE id = %s
                """,
                (
                    status,
                    error_message,
                    status,
                    session_id,
                ),
            )

            if cursor.rowcount == 0:
                raise ValueError(
                    f"Research session not found: {session_id}"
                )


def save_research_state(
    session_id: UUID,
    state: dict[str, Any],
) -> None:
    """Persist a serializable snapshot of the research state."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE research_sessions
                SET
                    research_state = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (Jsonb(state), session_id),
            )

            if cursor.rowcount == 0:
                raise ValueError(
                    f"Research session not found: {session_id}"
                )


def save_questions(
    session_id: UUID,
    questions: list[dict[str, Any]],
) -> None:
    """Replace the stored question list for a session."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM research_questions WHERE session_id = %s",
                (session_id,),
            )

            for number, question in enumerate(questions, start=1):
                cursor.execute(
                    """
                    INSERT INTO research_questions (
                        session_id,
                        question_number,
                        question,
                        search_queries,
                        parent_question_number,
                        status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        session_id,
                        number,
                        question["question"],
                        Jsonb(question.get("search_queries", [])),
                        question.get("parent_question_number"),
                        question.get("status", "pending"),
                    ),
                )


def save_evidence(
    session_id: UUID,
    evidence_items: list[dict[str, Any]],
) -> None:
    """Replace the stored verified evidence for a session."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM research_evidence WHERE session_id = %s",
                (session_id,),
            )

            for number, item in enumerate(
                evidence_items,
                start=1,
            ):
                cursor.execute(
                    """
                    INSERT INTO research_evidence (
                        session_id,
                        evidence_number,
                        research_question_number,
                        claim,
                        source_url,
                        supporting_text
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        session_id,
                        number,
                        item["research_question_number"],
                        item["claim"],
                        item["source_url"],
                        item["supporting_text"],
                    ),
                )


def save_report(
    session_id: UUID,
    report: dict[str, Any],
) -> None:
    """Save the final structured report."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE research_sessions
                SET
                    final_report = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (Jsonb(report), session_id),
            )

            if cursor.rowcount == 0:
                raise ValueError(
                    f"Research session not found: {session_id}"
                )


def get_session(session_id: UUID) -> dict[str, Any] | None:
    """Retrieve a session and its stored state/report."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    question,
                    status,
                    research_state,
                    final_report,
                    error_message,
                    created_at,
                    updated_at,
                    completed_at
                FROM research_sessions
                WHERE id = %s
                """,
                (session_id,),
            )

            row = cursor.fetchone()

    if row is None:
        return None

    return {
        "id": str(row[0]),
        "question": row[1],
        "status": row[2],
        "research_state": row[3],
        "final_report": row[4],
        "error_message": row[5],
        "created_at": row[6],
        "updated_at": row[7],
        "completed_at": row[8],
    }


def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    """List the most recently updated research sessions."""

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, question, status, created_at, updated_at
                FROM research_sessions
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (limit,),
            )

            rows = cursor.fetchall()

    return [
        {
            "id": str(row[0]),
            "question": row[1],
            "status": row[2],
            "created_at": row[3],
            "updated_at": row[4],
        }
        for row in rows
    ]
