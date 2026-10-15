"""
Repository operations for persistent research sessions
and observability events.
"""

from typing import Any
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb

from app.database.connection import get_connection


RESEARCH_JOB_COLUMNS = """
    id,
    session_id,
    payload,
    status,
    attempt_count,
    max_attempts,
    cancel_requested,
    last_error,
    created_at,
    updated_at,
    started_at,
    completed_at
"""


def _job_from_row(row: tuple[Any, ...]) -> dict[str, Any]:
    """Convert a durable-job row into its API/service representation."""

    return {
        "id": str(row[0]),
        "session_id": str(row[1]),
        "payload": row[2],
        "status": row[3],
        "attempt_count": row[4],
        "max_attempts": row[5],
        "cancel_requested": row[6],
        "last_error": row[7],
        "created_at": row[8],
        "updated_at": row[9],
        "started_at": row[10],
        "completed_at": row[11],
    }


# ---------------------------------------------------------
# RESEARCH SESSIONS
# ---------------------------------------------------------

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


def create_session_with_job(
    question: str,
    payload: dict[str, Any],
    max_attempts: int,
) -> tuple[UUID, UUID]:
    """Atomically create a session and its initial durable job."""

    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Research question cannot be empty.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one.")

    session_id = uuid4()
    job_id = uuid4()

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO research_sessions (id, question, status)
                VALUES (%s, %s, 'created')
                """,
                (session_id, normalized_question),
            )
            cursor.execute(
                """
                INSERT INTO research_jobs (
                    id,
                    session_id,
                    payload,
                    max_attempts
                )
                VALUES (%s, %s, %s, %s)
                """,
                (job_id, session_id, Jsonb(payload), max_attempts),
            )

    return session_id, job_id


def enqueue_review_job(
    session_id: UUID,
    decision: dict[str, Any],
    max_attempts: int,
) -> dict[str, Any]:
    """Atomically queue a validated review decision for graph resumption."""

    job_id = uuid4()
    payload = {"resume_decision": decision}

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {RESEARCH_JOB_COLUMNS}
                FROM research_jobs
                WHERE session_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                FOR UPDATE
                """,
                (session_id,),
            )
            previous = cursor.fetchone()
            if previous is None or previous[3] != "awaiting_review":
                raise ValueError("No pending research job is awaiting review.")

            cursor.execute(
                """
                SELECT status
                FROM research_sessions
                WHERE id = %s
                FOR UPDATE
                """,
                (session_id,),
            )
            session_row = cursor.fetchone()
            if session_row is None:
                raise ValueError(f"Research session not found: {session_id}")
            if session_row[0] != "waiting_for_review":
                raise ValueError("The session is not waiting for human review.")

            cursor.execute(
                """
                INSERT INTO research_jobs (
                    id,
                    session_id,
                    payload,
                    max_attempts
                )
                VALUES (%s, %s, %s, %s)
                RETURNING
                    id, session_id, payload, status, attempt_count,
                    max_attempts, cancel_requested, last_error,
                    created_at, updated_at, started_at, completed_at
                """,
                (job_id, session_id, Jsonb(payload), max_attempts),
            )
            job_row = cursor.fetchone()
            cursor.execute(
                """
                UPDATE research_sessions
                SET status = 'running',
                    error_message = NULL,
                    updated_at = NOW(),
                    completed_at = NULL
                WHERE id = %s
                """,
                (session_id,),
            )

    return _job_from_row(job_row)


def enqueue_retry_job(
    session_id: UUID,
    max_attempts: int,
    fallback_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Queue a failed or legacy stalled session from its graph checkpoint."""

    job_id = uuid4()

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload, status
                FROM research_jobs
                WHERE session_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                FOR UPDATE
                """,
                (session_id,),
            )
            previous = cursor.fetchone()

            cursor.execute(
                """
                SELECT status
                FROM research_sessions
                WHERE id = %s
                FOR UPDATE
                """,
                (session_id,),
            )
            session_row = cursor.fetchone()
            if session_row is None:
                raise ValueError(f"Research session not found: {session_id}")

            if previous is None:
                cursor.execute(
                    """
                    SELECT payload, status
                    FROM research_jobs
                    WHERE session_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    (session_id,),
                )
                previous = cursor.fetchone()

            if previous is not None:
                if previous[1] != "failed":
                    raise ValueError("The latest research job is not retryable.")
                payload = previous[0]
                if session_row[0] != "failed":
                    raise ValueError("Only failed research sessions can be retried.")
            else:
                if fallback_payload is None:
                    raise ValueError("The session has no retryable research job.")
                if session_row[0] not in {"failed", "running"}:
                    raise ValueError(
                        "Only failed or legacy stalled sessions can be resumed."
                    )
                payload = fallback_payload

            cursor.execute(
                """
                INSERT INTO research_jobs (
                    id,
                    session_id,
                    payload,
                    max_attempts
                )
                VALUES (%s, %s, %s, %s)
                RETURNING
                    id, session_id, payload, status, attempt_count,
                    max_attempts, cancel_requested, last_error,
                    created_at, updated_at, started_at, completed_at
                """,
                (job_id, session_id, Jsonb(payload), max_attempts),
            )
            job_row = cursor.fetchone()
            cursor.execute(
                """
                UPDATE research_sessions
                SET status = 'running',
                    error_message = NULL,
                    updated_at = NOW(),
                    completed_at = NULL
                WHERE id = %s
                """,
                (session_id,),
            )

    return _job_from_row(job_row)


def get_job(job_id: UUID) -> dict[str, Any] | None:
    """Retrieve one durable research job."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {RESEARCH_JOB_COLUMNS}
                FROM research_jobs
                WHERE id = %s
                """,
                (job_id,),
            )
            row = cursor.fetchone()

    return _job_from_row(row) if row else None


def get_latest_job(session_id: UUID) -> dict[str, Any] | None:
    """Retrieve the most recently created job for a research session."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {RESEARCH_JOB_COLUMNS}
                FROM research_jobs
                WHERE session_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (session_id,),
            )
            row = cursor.fetchone()

    return _job_from_row(row) if row else None


def claim_next_job(
    worker_id: str,
    lease_seconds: int,
) -> dict[str, Any] | None:
    """Atomically claim a queued job or reclaim an expired lease."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE research_jobs
                SET status = CASE
                        WHEN cancel_requested THEN 'cancelled'
                        ELSE 'failed'
                    END,
                    last_error = CASE
                        WHEN cancel_requested THEN last_error
                        ELSE COALESCE(
                            last_error,
                            'Worker lease expired after the final allowed attempt.'
                        )
                    END,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = NOW(),
                    completed_at = NOW()
                WHERE status = 'running'
                    AND lease_expires_at <= NOW()
                    AND attempt_count >= max_attempts
                RETURNING session_id, status, last_error
                """
            )
            for session_id, job_status, error_message in cursor.fetchall():
                cursor.execute(
                    """
                    UPDATE research_sessions
                    SET status = %s,
                        error_message = CASE
                            WHEN %s = 'failed' THEN %s
                            ELSE NULL
                        END,
                        updated_at = NOW(),
                        completed_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        job_status,
                        job_status,
                        error_message,
                        session_id,
                    ),
                )

            cursor.execute(
                f"""
                SELECT {RESEARCH_JOB_COLUMNS}
                FROM research_jobs
                WHERE (
                    status IN ('queued', 'retry_wait')
                    AND available_at <= NOW()
                )
                OR (
                    status = 'running'
                    AND lease_expires_at <= NOW()
                    AND attempt_count < max_attempts
                )
                ORDER BY created_at, id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """
            )
            row = cursor.fetchone()
            if row is None:
                return None

            cursor.execute(
                """
                UPDATE research_jobs
                SET status = 'running',
                    attempt_count = attempt_count + 1,
                    lease_owner = %s,
                    lease_expires_at = NOW() + make_interval(secs => %s),
                    cancel_requested = FALSE,
                    updated_at = NOW(),
                    started_at = COALESCE(started_at, NOW())
                WHERE id = %s
                RETURNING
                    id, session_id, payload, status, attempt_count,
                    max_attempts, cancel_requested, last_error,
                    created_at, updated_at, started_at, completed_at
                """,
                (worker_id, lease_seconds, row[0]),
            )
            claimed_row = cursor.fetchone()
            cursor.execute(
                """
                UPDATE research_sessions
                SET status = 'running',
                    error_message = NULL,
                    updated_at = NOW(),
                    completed_at = NULL
                WHERE id = %s
                """,
                (claimed_row[1],),
            )

    return _job_from_row(claimed_row)


def renew_job_lease(
    job_id: UUID,
    worker_id: str,
    lease_seconds: int,
) -> bool:
    """Extend a job lease while its worker remains alive."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE research_jobs
                SET lease_expires_at = NOW() + make_interval(secs => %s),
                    updated_at = NOW()
                WHERE id = %s
                    AND status = 'running'
                    AND lease_owner = %s
                """,
                (lease_seconds, job_id, worker_id),
            )
            return cursor.rowcount == 1


def release_job_for_recovery(job_id: UUID, worker_id: str) -> str | None:
    """Return a gracefully stopped job to the queue without consuming an attempt."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT session_id
                FROM research_jobs
                WHERE id = %s
                    AND status = 'running'
                    AND lease_owner = %s
                FOR UPDATE
                """,
                (job_id, worker_id),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            session_id = row[0]
            cursor.execute(
                """
                SELECT id
                FROM research_sessions
                WHERE id = %s
                FOR UPDATE
                """,
                (session_id,),
            )
            if cursor.fetchone() is None:
                return None

            cursor.execute(
                """
                SELECT cancel_requested
                FROM research_jobs
                WHERE id = %s
                    AND status = 'running'
                    AND lease_owner = %s
                FOR UPDATE
                """,
                (job_id, worker_id),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            outcome = "cancelled" if row[0] else "queued"
            cursor.execute(
                """
                UPDATE research_jobs
                SET status = %s,
                    attempt_count = GREATEST(attempt_count - 1, 0),
                    available_at = NOW(),
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = NOW(),
                    completed_at = CASE
                        WHEN %s = 'cancelled' THEN NOW()
                        ELSE NULL
                    END
                WHERE id = %s
                    AND status = 'running'
                    AND lease_owner = %s
                """,
                (outcome, outcome, job_id, worker_id),
            )
            if outcome == "cancelled":
                cursor.execute(
                    """
                    UPDATE research_sessions
                    SET status = 'cancelled',
                        error_message = NULL,
                        updated_at = NOW(),
                        completed_at = NOW()
                    WHERE id = %s
                    """,
                    (session_id,),
                )

    return outcome


def get_job_control(job_id: UUID, worker_id: str) -> bool | None:
    """Return cancellation state, or None when this worker lost its lease."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT cancel_requested
                FROM research_jobs
                WHERE id = %s
                    AND status = 'running'
                    AND lease_owner = %s
                """,
                (job_id, worker_id),
            )
            row = cursor.fetchone()

    return row[0] if row else None


def complete_job(
    job_id: UUID,
    worker_id: str,
    outcome: str,
) -> bool:
    """Complete a job and update the session in the same transaction."""

    if outcome not in {"completed", "awaiting_review", "cancelled"}:
        raise ValueError(f"Invalid job outcome: {outcome}")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT session_id, cancel_requested
                FROM research_jobs
                WHERE id = %s
                    AND status = 'running'
                    AND lease_owner = %s
                FOR UPDATE
                """,
                (job_id, worker_id),
            )
            row = cursor.fetchone()
            if row is None:
                return False

            session_id, cancel_requested = row
            if cancel_requested:
                outcome = "cancelled"

            cursor.execute(
                """
                SELECT id
                FROM research_sessions
                WHERE id = %s
                FOR UPDATE
                """,
                (session_id,),
            )
            if cursor.fetchone() is None:
                return False

            cursor.execute(
                """
                UPDATE research_jobs
                SET status = %s,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = NOW(),
                    completed_at = CASE
                        WHEN %s = 'awaiting_review' THEN NULL
                        ELSE NOW()
                    END
                WHERE id = %s
                    AND status = 'running'
                    AND lease_owner = %s
                RETURNING session_id
                """,
                (outcome, outcome, job_id, worker_id),
            )
            completed_row = cursor.fetchone()
            if completed_row is None:
                return False

            session_status = {
                "completed": "completed",
                "awaiting_review": "waiting_for_review",
                "cancelled": "cancelled",
            }[outcome]
            cursor.execute(
                """
                UPDATE research_sessions
                SET status = %s,
                    error_message = NULL,
                    updated_at = NOW(),
                    completed_at = CASE
                        WHEN %s IN ('completed', 'cancelled')
                            THEN NOW()
                        ELSE NULL
                    END
                WHERE id = %s
                """,
                (session_status, session_status, completed_row[0]),
            )

    return True


def fail_or_retry_job(
    job_id: UUID,
    worker_id: str,
    error_message: str,
    retryable: bool,
    backoff_seconds: int,
) -> str | None:
    """Persist a failure and either schedule a retry or mark the job terminal."""

    normalized_error = error_message[:2000]

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT session_id
                FROM research_jobs
                WHERE id = %s
                    AND status = 'running'
                    AND lease_owner = %s
                """,
                (job_id, worker_id),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            session_id = row[0]
            cursor.execute(
                """
                SELECT id
                FROM research_sessions
                WHERE id = %s
                FOR UPDATE
                """,
                (session_id,),
            )
            if cursor.fetchone() is None:
                return None

            cursor.execute(
                """
                SELECT session_id, attempt_count, max_attempts,
                    cancel_requested
                FROM research_jobs
                WHERE id = %s
                    AND status = 'running'
                    AND lease_owner = %s
                FOR UPDATE
                """,
                (job_id, worker_id),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            session_id, attempts, max_attempts, cancel_requested = row
            if cancel_requested:
                outcome = "cancelled"
            elif retryable and attempts < max_attempts:
                outcome = "retry_wait"
            else:
                outcome = "failed"

            cursor.execute(
                """
                UPDATE research_jobs
                SET status = %s,
                    last_error = %s,
                    available_at = CASE
                        WHEN %s = 'retry_wait'
                            THEN NOW() + make_interval(secs => %s)
                        ELSE available_at
                    END,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = NOW(),
                    completed_at = CASE
                        WHEN %s IN ('failed', 'cancelled')
                            THEN NOW()
                        ELSE NULL
                    END
                WHERE id = %s
                """,
                (
                    outcome,
                    normalized_error,
                    outcome,
                    backoff_seconds,
                    outcome,
                    job_id,
                ),
            )

            if outcome == "failed":
                cursor.execute(
                    """
                    UPDATE research_sessions
                    SET status = 'failed',
                        error_message = %s,
                        updated_at = NOW(),
                        completed_at = NOW()
                    WHERE id = %s
                    """,
                    (normalized_error, session_id),
                )
            elif outcome == "cancelled":
                cursor.execute(
                    """
                    UPDATE research_sessions
                    SET status = 'cancelled',
                        error_message = NULL,
                        updated_at = NOW(),
                        completed_at = NOW()
                    WHERE id = %s
                    """,
                    (session_id,),
                )

    return outcome


def cancel_latest_job(session_id: UUID) -> dict[str, Any]:
    """Cancel queued work immediately or request cooperative worker cancellation."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {RESEARCH_JOB_COLUMNS}
                FROM research_jobs
                WHERE session_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                FOR UPDATE
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    "SELECT id FROM research_sessions WHERE id = %s",
                    (session_id,),
                )
                if cursor.fetchone() is None:
                    raise ValueError(f"Research session not found: {session_id}")
                raise ValueError("The session has no research job to cancel.")

            cursor.execute(
                """
                SELECT id
                FROM research_sessions
                WHERE id = %s
                FOR UPDATE
                """,
                (session_id,),
            )
            if cursor.fetchone() is None:
                raise ValueError(f"Research session not found: {session_id}")

            cursor.execute(
                f"""
                SELECT {RESEARCH_JOB_COLUMNS}
                FROM research_jobs
                WHERE session_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                FOR UPDATE
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError("The session has no research job to cancel.")

            job = _job_from_row(row)
            if job["status"] in {"queued", "retry_wait", "awaiting_review"}:
                cursor.execute(
                    """
                    UPDATE research_jobs
                    SET status = 'cancelled',
                        cancel_requested = TRUE,
                        updated_at = NOW(),
                        completed_at = NOW()
                    WHERE id = %s
                    """,
                    (UUID(job["id"]),),
                )
                cursor.execute(
                    """
                    UPDATE research_sessions
                    SET status = 'cancelled',
                        error_message = NULL,
                        updated_at = NOW(),
                        completed_at = NOW()
                    WHERE id = %s
                    """,
                    (session_id,),
                )
                job["status"] = "cancelled"
                job["cancel_requested"] = True
            elif job["status"] == "running":
                cursor.execute(
                    """
                    UPDATE research_jobs
                    SET cancel_requested = TRUE,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (UUID(job["id"]),),
                )
                job["cancel_requested"] = True
            else:
                raise ValueError(
                    f"Cannot cancel a job with status {job['status']}."
                )

    return job


def update_session_status(
    session_id: UUID,
    status: str,
    error_message: str | None = None,
) -> None:
    """Update a session's status and optional error message."""

    allowed_statuses = {
        "created",
        "running",
        "waiting_for_review",
        "completed",
        "failed",
        "cancelled",
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
                        WHEN %s IN ('completed', 'failed', 'cancelled')
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


# ---------------------------------------------------------
# OBSERVABILITY EVENTS
# ---------------------------------------------------------

def save_observability_event(
    event_type: str,
    message: str,
    session_id: UUID | str | None = None,
    node_name: str | None = None,
    log_level: str = "INFO",
    duration_ms: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """
    Persist one structured observability event.

    Events can be associated with a research session or
    stored without a session for application-level events.
    """

    if not event_type.strip():
        raise ValueError("event_type cannot be empty.")

    if not message.strip():
        raise ValueError("message cannot be empty.")

    allowed_log_levels = {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    normalized_level = log_level.upper()

    if normalized_level not in allowed_log_levels:
        raise ValueError(
            f"Invalid log level: {log_level}"
        )

    if duration_ms is not None and duration_ms < 0:
        raise ValueError("duration_ms cannot be negative.")

    normalized_session_id = (
        UUID(str(session_id))
        if session_id is not None
        else None
    )

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO observability_events (
                    session_id,
                    event_type,
                    node_name,
                    log_level,
                    message,
                    duration_ms,
                    details
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    normalized_session_id,
                    event_type.strip(),
                    node_name,
                    normalized_level,
                    message.strip(),
                    duration_ms,
                    Jsonb(details or {}),
                ),
            )


# ---------------------------------------------------------
# SESSION RETRIEVAL
# ---------------------------------------------------------

def get_session(session_id: UUID) -> dict[str, Any] | None:
    """Retrieve a session and its stored state/report."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    session.id,
                    session.question,
                    session.status,
                    session.research_state,
                    session.final_report,
                    session.error_message,
                    session.created_at,
                    session.updated_at,
                    session.completed_at,
                    job_id,
                    job_status,
                    attempt_count,
                    max_attempts,
                    cancel_requested,
                    last_error
                FROM research_sessions AS session
                LEFT JOIN LATERAL (
                    SELECT
                        id AS job_id,
                        status AS job_status,
                        attempt_count,
                        max_attempts,
                        cancel_requested,
                        last_error
                    FROM research_jobs
                    WHERE session_id = session.id
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                ) latest_job ON TRUE
                WHERE session.id = %s
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
        "job": (
            {
                "id": str(row[9]),
                "status": row[10],
                "attempt_count": row[11],
                "max_attempts": row[12],
                "cancel_requested": row[13],
                "last_error": row[14],
            }
            if row[9] is not None
            else None
        ),
    }


def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    """List the most recently updated research sessions."""

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    session.id,
                    session.question,
                    session.status,
                    session.created_at,
                    session.updated_at,
                    job_id,
                    job_status,
                    attempt_count,
                    max_attempts,
                    cancel_requested,
                    last_error
                FROM research_sessions AS session
                LEFT JOIN LATERAL (
                    SELECT
                        id AS job_id,
                        status AS job_status,
                        attempt_count,
                        max_attempts,
                        cancel_requested,
                        last_error
                    FROM research_jobs
                    WHERE session_id = session.id
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                ) latest_job ON TRUE
                ORDER BY session.updated_at DESC
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
            "job": (
                {
                    "id": str(row[5]),
                    "status": row[6],
                    "attempt_count": row[7],
                    "max_attempts": row[8],
                    "cancel_requested": row[9],
                    "last_error": row[10],
                }
                if row[5] is not None
                else None
            ),
        }
        for row in rows
    ]


def get_observability_events(
    session_id: UUID | str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Retrieve recorded observability events."""

    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500.")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            if session_id is not None:
                cursor.execute(
                    """
                    SELECT
                        id,
                        session_id,
                        event_type,
                        node_name,
                        log_level,
                        message,
                        duration_ms,
                        details,
                        created_at
                    FROM observability_events
                    WHERE session_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s
                    """,
                    (UUID(str(session_id)), limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT
                        id,
                        session_id,
                        event_type,
                        node_name,
                        log_level,
                        message,
                        duration_ms,
                        details,
                        created_at
                    FROM observability_events
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )

            rows = cursor.fetchall()

    events = [
        {
            "id": row[0],
            "session_id": str(row[1]) if row[1] else None,
            "event_type": row[2],
            "node_name": row[3],
            "log_level": row[4],
            "message": row[5],
            "duration_ms": row[6],
            "details": row[7] or {},
            "created_at": row[8],
        }
        for row in rows
    ]

    if session_id is not None:
        events.reverse()

    return events


def get_node_execution_metrics() -> dict[str, dict[str, Any]]:
    """Aggregate node metrics across all API and worker processes."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    node_name,
                    COUNT(*) FILTER (
                        WHERE event_type = 'node_started'
                    ) AS calls,
                    COUNT(*) FILTER (
                        WHERE event_type = 'node_completed'
                    ) AS successes,
                    COUNT(*) FILTER (
                        WHERE event_type = 'node_failed'
                    ) AS failures,
                    COALESCE(
                        SUM(duration_ms) FILTER (
                            WHERE event_type IN (
                                'node_completed',
                                'node_failed'
                            )
                        ),
                        0
                    ) AS duration_ms,
                    (
                        ARRAY_AGG(
                            CASE event_type
                                WHEN 'node_completed' THEN 'success'
                                WHEN 'node_failed' THEN 'failure'
                            END
                            ORDER BY created_at DESC, id DESC
                        ) FILTER (
                            WHERE event_type IN (
                                'node_completed',
                                'node_failed'
                            )
                        )
                    )[1] AS last_status
                FROM observability_events
                WHERE node_name IS NOT NULL
                    AND event_type IN (
                        'node_started',
                        'node_completed',
                        'node_failed'
                    )
                GROUP BY node_name
                """
            )
            rows = cursor.fetchall()

    return {
        row[0]: {
            "calls": row[1],
            "successes": row[2],
            "failures": row[3],
            "duration_ms": float(row[4]),
            "last_status": row[5],
        }
        for row in rows
    }


def delete_session(session_id: UUID) -> bool:
    """Delete a research session and its associated records."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT status
                FROM research_jobs
                WHERE session_id = %s
                    AND status IN (
                        'queued',
                        'running',
                        'retry_wait',
                        'awaiting_review'
                    )
                LIMIT 1
                FOR UPDATE
                """,
                (session_id,),
            )
            if cursor.fetchone() is not None:
                raise ValueError(
                    "Cancel or finish the active research job before deleting this session."
                )

            cursor.execute(
                "DELETE FROM research_sessions WHERE id = %s",
                (session_id,),
            )
            deleted = cursor.rowcount > 0
        connection.commit()

    return deleted
