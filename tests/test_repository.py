"""Focused unit tests for database repository operations."""

from unittest.mock import MagicMock, call
from uuid import uuid4

import pytest

from app.database import repository
from psycopg.types.json import Jsonb



@pytest.fixture
def mock_database(monkeypatch):
    connection = MagicMock()
    cursor = MagicMock()

    # Support: with get_connection() as connection
    connection.__enter__.return_value = connection

    # Support: with connection.cursor() as cursor
    connection.cursor.return_value.__enter__.return_value = cursor

    monkeypatch.setattr(
        repository,
        "get_connection",
        lambda: connection,
    )

    return connection, cursor


def test_create_session_strips_question(mock_database):
    _, cursor = mock_database

    session_id = repository.create_session(
        "  How many research papers exist?  "
    )

    assert session_id is not None
    assert cursor.execute.call_count == 1

    sql, params = cursor.execute.call_args.args

    assert "INSERT INTO research_sessions" in sql
    assert params[0] == session_id
    assert params[1] == "How many research papers exist?"


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
        "\n\t ",
    ],
)
def test_create_session_rejects_empty_question(
    mock_database,
    question,
):
    _, cursor = mock_database

    with pytest.raises(
        ValueError,
        match="Research question cannot be empty",
    ):
        repository.create_session(question)

    cursor.execute.assert_not_called()


@pytest.mark.parametrize(
    "question,max_attempts",
    [
        ("", 3),
        ("   ", 3),
        ("Valid question", 0),
        ("Valid question", -1),
    ],
)
def test_create_session_with_job_rejects_invalid_input(
    mock_database,
    question,
    max_attempts,
):
    _, cursor = mock_database

    with pytest.raises(ValueError):
        repository.create_session_with_job(
            question=question,
            payload={"question": question},
            max_attempts=max_attempts,
        )

    cursor.execute.assert_not_called()


def test_create_session_with_job_inserts_session_and_job(
    mock_database,
):
    _, cursor = mock_database

    session_id, job_id = repository.create_session_with_job(
        question="  Research AI safety  ",
        payload={"question": "Research AI safety"},
        max_attempts=3,
    )

    assert session_id is not None
    assert job_id is not None
    assert session_id != job_id
    assert cursor.execute.call_count == 2

    session_sql, session_params = (
        cursor.execute.call_args_list[0].args
    )
    job_sql, job_params = (
        cursor.execute.call_args_list[1].args
    )

    assert "INSERT INTO research_sessions" in session_sql
    assert session_params[0] == session_id
    assert session_params[1] == "Research AI safety"

    assert "INSERT INTO research_jobs" in job_sql
    assert job_params[0] == job_id
    assert job_params[1] == session_id
    assert job_params[3] == 3


def make_job_row(
    session_id,
    *,
    job_id=None,
    payload=None,
    status="queued",
):
    """Build a row matching RESEARCH_JOB_COLUMNS."""
    return (
        job_id or uuid4(),
        session_id,
        payload or {"question": "Research AI safety"},
        status,
        0,
        3,
        False,
        False,
        None,
        None,
        None,
        None,
        None,
    )


def test_enqueue_review_job_rejects_missing_pending_job(
    mock_database,
):
    _, cursor = mock_database
    cursor.fetchone.return_value = None

    with pytest.raises(
        ValueError,
        match="No pending research job is awaiting review",
    ):
        repository.enqueue_review_job(
            session_id=uuid4(),
            decision={"approved": True},
            max_attempts=3,
        )

    assert cursor.execute.call_count == 1


def test_enqueue_review_job_rejects_wrong_job_status(
    mock_database,
):
    _, cursor = mock_database
    session_id = uuid4()
    cursor.fetchone.return_value = make_job_row(
        session_id,
        status="failed",
    )

    with pytest.raises(
        ValueError,
        match="No pending research job is awaiting review",
    ):
        repository.enqueue_review_job(
            session_id=session_id,
            decision={"approved": True},
            max_attempts=3,
        )

    assert cursor.execute.call_count == 1


def test_enqueue_review_job_rejects_missing_session(
    mock_database,
):
    _, cursor = mock_database
    session_id = uuid4()

    cursor.fetchone.side_effect = [
        make_job_row(session_id, status="awaiting_review"),
        None,
    ]

    with pytest.raises(
        ValueError,
        match="Research session not found",
    ):
        repository.enqueue_review_job(
            session_id=session_id,
            decision={"approved": True},
            max_attempts=3,
        )

    assert cursor.execute.call_count == 2


def test_enqueue_review_job_rejects_wrong_session_status(
    mock_database,
):
    _, cursor = mock_database
    session_id = uuid4()

    cursor.fetchone.side_effect = [
        make_job_row(session_id, status="awaiting_review"),
        ("running",),
    ]

    with pytest.raises(
        ValueError,
        match="not waiting for human review",
    ):
        repository.enqueue_review_job(
            session_id=session_id,
            decision={"approved": True},
            max_attempts=3,
        )

    assert cursor.execute.call_count == 2


def test_enqueue_review_job_inserts_job_and_resumes_session(
    mock_database,
):
    _, cursor = mock_database
    session_id = uuid4()

    inserted_row = make_job_row(
        session_id,
        status="queued",
        payload={"resume_decision": {"approved": True}},
    )

    cursor.fetchone.side_effect = [
        make_job_row(session_id, status="awaiting_review"),
        ("waiting_for_review",),
        inserted_row,
    ]

    result = repository.enqueue_review_job(
        session_id=session_id,
        decision={"approved": True},
        max_attempts=4,
    )

    assert result["status"] == "queued"
    assert result["session_id"] == str(session_id)
    assert result["payload"] == {
        "resume_decision": {"approved": True}
    }
    assert cursor.execute.call_count == 4

    insert_sql, insert_params = cursor.execute.call_args_list[2].args
    update_sql, update_params = cursor.execute.call_args_list[3].args

    assert "INSERT INTO research_jobs" in insert_sql
    assert isinstance(insert_params[2], Jsonb)
    assert insert_params[3] == 4

    assert "UPDATE research_sessions" in update_sql
    assert "status = 'running'" in update_sql
    assert update_params == (session_id,)


def test_enqueue_retry_job_rejects_missing_session(
    mock_database,
):
    _, cursor = mock_database
    cursor.fetchone.side_effect = [None, None]

    with pytest.raises(
        ValueError,
        match="Research session not found",
    ):
        repository.enqueue_retry_job(
            session_id=uuid4(),
            max_attempts=3,
            fallback_payload={"question": "Retry"},
        )

    assert cursor.execute.call_count == 2


def test_enqueue_retry_job_rejects_missing_job_without_fallback(
    mock_database,
):
    _, cursor = mock_database
    cursor.fetchone.side_effect = [
        None,          # No previous job
        ("failed",),   # Session exists and is failed
        None,          # Second job lookup
    ]

    with pytest.raises(
        ValueError,
        match="no retryable research job",
    ):
        repository.enqueue_retry_job(
            session_id=uuid4(),
            max_attempts=3,
        )


def test_enqueue_retry_job_rejects_nonfailed_job(
    mock_database,
):
    _, cursor = mock_database
    session_id = uuid4()

    cursor.fetchone.side_effect = [
        ({"question": "Original"}, "completed"),
        ("completed",),
    ]

    with pytest.raises(
        ValueError,
        match="latest research job is not retryable",
    ):
        repository.enqueue_retry_job(
            session_id=session_id,
            max_attempts=3,
        )


def test_enqueue_retry_job_rejects_failed_job_with_wrong_session_state(
    mock_database,
):
    _, cursor = mock_database

    cursor.fetchone.side_effect = [
        ({"question": "Original"}, "failed"),
        ("running",),
    ]

    with pytest.raises(
        ValueError,
        match="Only failed research sessions can be retried",
    ):
        repository.enqueue_retry_job(
            session_id=uuid4(),
            max_attempts=3,
        )


def test_enqueue_retry_job_rejects_ineligible_legacy_session(
    mock_database,
):
    _, cursor = mock_database

    cursor.fetchone.side_effect = [
        None,
        ("completed",),
        None,
    ]

    with pytest.raises(
        ValueError,
        match="Only failed or legacy stalled sessions",
    ):
        repository.enqueue_retry_job(
            session_id=uuid4(),
            max_attempts=3,
            fallback_payload={"question": "Retry"},
        )


def test_enqueue_retry_job_uses_fallback_payload(
    mock_database,
):
    _, cursor = mock_database
    session_id = uuid4()
    fallback_payload = {"question": "Resume legacy session"}

    inserted_row = make_job_row(
        session_id,
        payload=fallback_payload,
        status="queued",
    )

    cursor.fetchone.side_effect = [
        None,          # No previous job
        ("running",),  # Legacy stalled session
        None,          # Repeated job lookup
        inserted_row,  # Inserted job
    ]

    result = repository.enqueue_retry_job(
        session_id=session_id,
        max_attempts=4,
        fallback_payload=fallback_payload,
    )

    assert result["status"] == "queued"
    assert result["session_id"] == str(session_id)
    assert result["payload"] == fallback_payload

    insert_sql, insert_params = cursor.execute.call_args_list[3].args

    assert "INSERT INTO research_jobs" in insert_sql
    assert isinstance(insert_params[2], Jsonb)
    assert insert_params[3] == 4


def test_enqueue_retry_job_reuses_failed_job_payload(
    mock_database,
):
    _, cursor = mock_database
    session_id = uuid4()
    original_payload = {"question": "Original research"}

    inserted_row = make_job_row(
        session_id,
        payload=original_payload,
        status="queued",
    )

    cursor.fetchone.side_effect = [
        (original_payload, "failed"),
        ("failed",),
        inserted_row,
    ]

    result = repository.enqueue_retry_job(
        session_id=session_id,
        max_attempts=3,
    )

    assert result["status"] == "queued"
    assert result["payload"] == original_payload

    insert_sql, insert_params = cursor.execute.call_args_list[2].args

    assert "INSERT INTO research_jobs" in insert_sql
    assert isinstance(insert_params[2], Jsonb)
    assert insert_params[3] == 3


def test_get_job_returns_converted_row(mock_database):
    _, cursor = mock_database
    job_id = uuid4()
    session_id = uuid4()

    cursor.fetchone.return_value = make_job_row(
        session_id,
        job_id=job_id,
        status="queued",
    )

    result = repository.get_job(job_id)

    assert result["id"] == str(job_id)
    assert result["session_id"] == str(session_id)
    assert result["status"] == "queued"

    sql, params = cursor.execute.call_args.args
    assert "FROM research_jobs" in sql
    assert params == (job_id,)


def test_get_job_returns_none_when_missing(mock_database):
    _, cursor = mock_database
    cursor.fetchone.return_value = None

    assert repository.get_job(uuid4()) is None


def test_get_latest_job_returns_converted_row(mock_database):
    _, cursor = mock_database
    session_id = uuid4()

    cursor.fetchone.return_value = make_job_row(
        session_id,
        status="failed",
    )

    result = repository.get_latest_job(session_id)

    assert result["session_id"] == str(session_id)
    assert result["status"] == "failed"

    sql, params = cursor.execute.call_args.args
    assert "ORDER BY created_at DESC, id DESC" in sql
    assert params == (session_id,)


def test_get_latest_job_returns_none_when_missing(mock_database):
    _, cursor = mock_database
    cursor.fetchone.return_value = None

    assert repository.get_latest_job(uuid4()) is None


@pytest.mark.parametrize(
    ("rowcount", "expected"),
    [
        (1, True),
        (0, False),
        (2, False),
    ],
)
def test_renew_job_lease_checks_updated_row_count(
    mock_database,
    rowcount,
    expected,
):
    _, cursor = mock_database
    cursor.rowcount = rowcount

    job_id = uuid4()
    result = repository.renew_job_lease(
        job_id=job_id,
        worker_id="worker-1",
        lease_seconds=60,
    )

    assert result is expected

    sql, params = cursor.execute.call_args.args
    assert "lease_owner = %s" in sql
    assert params == (60, job_id, "worker-1")


def test_release_job_for_recovery_returns_none_for_wrong_owner(
    mock_database,
):
    _, cursor = mock_database
    cursor.fetchone.return_value = None

    result = repository.release_job_for_recovery(
        job_id=uuid4(),
        worker_id="wrong-worker",
    )

    assert result is None
    assert cursor.execute.call_count == 1


def test_release_job_for_recovery_returns_none_if_session_missing(
    mock_database,
):
    _, cursor = mock_database
    cursor.fetchone.side_effect = [
        (uuid4(),),  # Owned running job
        None,        # Session does not exist
    ]

    result = repository.release_job_for_recovery(
        job_id=uuid4(),
        worker_id="worker-1",
    )

    assert result is None
    assert cursor.execute.call_count == 2


@pytest.mark.parametrize(
    ("cancel_requested", "pause_requested", "expected"),
    [
        (True, False, "cancelled"),
        (False, True, "paused"),
        (False, False, "queued"),
    ],
)
def test_release_job_for_recovery_selects_correct_outcome(
    mock_database,
    cancel_requested,
    pause_requested,
    expected,
):
    _, cursor = mock_database
    session_id = uuid4()

    cursor.fetchone.side_effect = [
        (session_id,),  # Running job owned by worker
        (session_id,),  # Session exists
        (cancel_requested, pause_requested),
    ]

    result = repository.release_job_for_recovery(
        job_id=uuid4(),
        worker_id="worker-1",
    )

    assert result == expected

    # The third execution updates the job.
    update_sql, update_params = cursor.execute.call_args_list[3].args

    assert "UPDATE research_jobs" in update_sql
    assert update_params[0] == expected

    # Cancellation and pause also update the session.
    if expected in {"cancelled", "paused"}:
        session_sql, session_params = (
            cursor.execute.call_args_list[4].args
        )
        assert "UPDATE research_sessions" in session_sql
        assert session_params[0] == expected
    else:
        assert cursor.execute.call_count == 4

def test_get_job_control_returns_flags(mock_database):
    _, cursor = mock_database
    cursor.fetchone.return_value = (True, False)

    job_id = uuid4()

    result = repository.get_job_control(
        job_id=job_id,
        worker_id="worker-1",
    )

    assert result == {
        "cancel_requested": True,
        "pause_requested": False,
    }

    sql, params = cursor.execute.call_args.args
    assert "status = 'running'" in sql
    assert "lease_owner = %s" in sql
    assert params == (job_id, "worker-1")


def test_get_job_control_returns_none_when_lease_is_lost(
    mock_database,
):
    _, cursor = mock_database
    cursor.fetchone.return_value = None

    result = repository.get_job_control(
        job_id=uuid4(),
        worker_id="worker-1",
    )

    assert result is None


def test_pause_running_job_returns_none_when_job_is_not_owned(
    mock_database,
):
    _, cursor = mock_database
    cursor.fetchone.return_value = None

    result = repository.pause_running_job(
        job_id=uuid4(),
        worker_id="worker-1",
    )

    assert result is None
    assert cursor.execute.call_count == 1


def test_pause_running_job_returns_none_when_session_is_missing(
    mock_database,
):
    _, cursor = mock_database
    cursor.fetchone.side_effect = [
        (uuid4(),),  # Running job belongs to this worker.
        None,        # Its session is missing.
    ]

    result = repository.pause_running_job(
        job_id=uuid4(),
        worker_id="worker-1",
    )

    assert result is None
    assert cursor.execute.call_count == 2


@pytest.mark.parametrize(
    ("cancel_requested", "pause_requested", "expected"),
    [
        (True, False, "cancelled"),
        (False, True, "paused"),
    ],
)
def test_pause_running_job_handles_control_requests(
    mock_database,
    cancel_requested,
    pause_requested,
    expected,
):
    _, cursor = mock_database
    session_id = uuid4()

    cursor.fetchone.side_effect = [
        (session_id,),  # Owned running job.
        (session_id,),  # Existing session.
        (cancel_requested, pause_requested),
    ]

    result = repository.pause_running_job(
        job_id=uuid4(),
        worker_id="worker-1",
    )

    assert result == expected

    job_sql, job_params = cursor.execute.call_args_list[3].args
    assert "UPDATE research_jobs" in job_sql
    assert job_params[0] == expected

    session_sql, session_params = cursor.execute.call_args_list[4].args
    assert "UPDATE research_sessions" in session_sql
    assert session_params[0] == expected


def test_pause_running_job_does_nothing_without_control_request(
    mock_database,
):
    _, cursor = mock_database
    session_id = uuid4()

    cursor.fetchone.side_effect = [
        (session_id,),
        (session_id,),
        (False, False),
    ]

    result = repository.pause_running_job(
        job_id=uuid4(),
        worker_id="worker-1",
    )

    assert result is None
    assert cursor.execute.call_count == 3

def test_complete_job_rejects_invalid_outcome():
    with pytest.raises(ValueError, match="Invalid job outcome"):
        repository.complete_job(
            job_id=uuid4(),
            worker_id="worker-1",
            outcome="failed",
        )


def test_complete_job_returns_false_when_job_is_not_owned(
    mock_database,
):
    _, cursor = mock_database
    cursor.fetchone.return_value = None

    result = repository.complete_job(
        job_id=uuid4(),
        worker_id="worker-1",
        outcome="completed",
    )

    assert result is False
    assert cursor.execute.call_count == 1


def test_complete_job_returns_false_when_session_is_missing(
    mock_database,
):
    _, cursor = mock_database
    session_id = uuid4()

    cursor.fetchone.side_effect = [
        (session_id, False, False),
        None,
    ]

    result = repository.complete_job(
        job_id=uuid4(),
        worker_id="worker-1",
        outcome="completed",
    )

    assert result is False
    assert cursor.execute.call_count == 2


@pytest.mark.parametrize(
    ("outcome", "expected_session_status"),
    [
        ("completed", "completed"),
        ("awaiting_review", "waiting_for_review"),
        ("cancelled", "cancelled"),
        ("paused", "paused"),
    ],
)
def test_complete_job_updates_job_and_session(
    mock_database,
    outcome,
    expected_session_status,
):
    _, cursor = mock_database
    session_id = uuid4()

    cursor.fetchone.side_effect = [
        (session_id, False, False),
        (session_id,),
        (session_id,),
    ]

    result = repository.complete_job(
        job_id=uuid4(),
        worker_id="worker-1",
        outcome=outcome,
    )

    assert result is True

    job_sql, job_params = cursor.execute.call_args_list[2].args
    assert "UPDATE research_jobs" in job_sql
    assert job_params[0] == outcome

    session_sql, session_params = cursor.execute.call_args_list[3].args
    assert "UPDATE research_sessions" in session_sql
    assert session_params[0] == expected_session_status


@pytest.mark.parametrize(
    ("cancel_requested", "pause_requested", "expected"),
    [
        (True, False, "cancelled"),
        (False, True, "paused"),
    ],
)
def test_complete_job_prioritizes_control_requests(
    mock_database,
    cancel_requested,
    pause_requested,
    expected,
):
    _, cursor = mock_database
    session_id = uuid4()

    cursor.fetchone.side_effect = [
        (session_id, cancel_requested, pause_requested),
        (session_id,),
        (session_id,),
    ]

    result = repository.complete_job(
        job_id=uuid4(),
        worker_id="worker-1",
        outcome="completed",
    )

    assert result is True

    _, job_params = cursor.execute.call_args_list[2].args
    assert job_params[0] == expected

    _, session_params = cursor.execute.call_args_list[3].args
    assert session_params[0] == expected


def test_complete_job_returns_false_if_update_loses_ownership(
    mock_database,
):
    _, cursor = mock_database
    session_id = uuid4()

    cursor.fetchone.side_effect = [
        (session_id, False, False),
        (session_id,),
        None,
    ]

    result = repository.complete_job(
        job_id=uuid4(),
        worker_id="worker-1",
        outcome="completed",
    )

    assert result is False
    assert cursor.execute.call_count == 3

# ---------------------------------------------------------
# SESSION RETRIEVAL TESTS
# ---------------------------------------------------------

def test_get_session_returns_session_with_latest_job(mock_database):
    connection, cursor = mock_database

    session_id = uuid4()
    job_id = uuid4()

    cursor.fetchone.return_value = (
        session_id,
        "Research AI safety",
        "completed",
        {"current_node": "synthesis"},
        {"title": "AI Safety Report"},
        None,
        "created",
        "updated",
        "completed",
        job_id,
        "completed",
        2,
        3,
        False,
        False,
        None,
    )

    result = repository.get_session(session_id)

    assert result is not None
    assert result["id"] == str(session_id)
    assert result["question"] == "Research AI safety"
    assert result["status"] == "completed"
    assert result["research_state"] == {
        "current_node": "synthesis"
    }
    assert result["final_report"] == {
        "title": "AI Safety Report"
    }

    assert result["job"] == {
        "id": str(job_id),
        "status": "completed",
        "attempt_count": 2,
        "max_attempts": 3,
        "cancel_requested": False,
        "pause_requested": False,
        "last_error": None,
    }

    sql, params = cursor.execute.call_args.args
    assert "LEFT JOIN LATERAL" in sql
    assert params == (session_id,)


def test_get_session_returns_none_when_missing(mock_database):
    connection, cursor = mock_database
    cursor.fetchone.return_value = None

    result = repository.get_session(uuid4())

    assert result is None


def test_get_session_handles_session_without_job(mock_database):
    connection, cursor = mock_database

    session_id = uuid4()

    cursor.fetchone.return_value = (
        session_id,
        "Research AI safety",
        "queued",
        None,
        None,
        None,
        "created",
        "updated",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    result = repository.get_session(session_id)

    assert result is not None
    assert result["id"] == str(session_id)
    assert result["job"] is None


@pytest.mark.parametrize("limit", [0, -1, 101])
def test_list_sessions_rejects_invalid_limit(limit):
    with pytest.raises(ValueError, match="limit must be between 1 and 100"):
        repository.list_sessions(limit=limit)


def test_list_sessions_returns_session_summaries(mock_database):
    connection, cursor = mock_database

    session_id = uuid4()
    job_id = uuid4()

    cursor.fetchall.return_value = [
        (
            session_id,
            "Research AI safety",
            "running",
            "created",
            "updated",
            job_id,
            "running",
            1,
            3,
            False,
            False,
            None,
        )
    ]

    result = repository.list_sessions(limit=10)

    assert len(result) == 1

    session = result[0]
    assert session["id"] == str(session_id)
    assert session["question"] == "Research AI safety"
    assert session["status"] == "running"
    assert session["job"]["id"] == str(job_id)
    assert session["job"]["attempt_count"] == 1

    sql, params = cursor.execute.call_args.args
    assert "LEFT JOIN LATERAL" in sql
    assert "ORDER BY session.updated_at DESC" in sql
    assert params == (10,)


def test_list_sessions_returns_empty_list(mock_database):
    connection, cursor = mock_database
    cursor.fetchall.return_value = []

    assert repository.list_sessions() == []


# ---------------------------------------------------------
# RESEARCH STATE PERSISTENCE TESTS
# ---------------------------------------------------------

def test_save_research_state_updates_session(mock_database):
    connection, cursor = mock_database

    session_id = uuid4()
    state = {
        "current_node": "researcher",
        "questions": ["Question 1"],
    }

    cursor.rowcount = 1

    repository.save_research_state(session_id, state)

    sql, params = cursor.execute.call_args.args

    assert "UPDATE research_sessions" in sql
    assert "research_state = %s" in sql
    assert params[1] == session_id


def test_save_research_state_raises_when_session_missing(
    mock_database,
):
    connection, cursor = mock_database

    session_id = uuid4()
    cursor.rowcount = 0

    with pytest.raises(
        ValueError,
        match="Research session not found",
    ):
        repository.save_research_state(
            session_id,
            {"current_node": "researcher"},
        )


# ---------------------------------------------------------
# REPORT PERSISTENCE TESTS
# ---------------------------------------------------------

def test_save_report_updates_session(mock_database):
    connection, cursor = mock_database

    session_id = uuid4()
    report = {
        "title": "AI Safety Report",
        "sections": [],
    }

    cursor.rowcount = 1

    repository.save_report(session_id, report)

    sql, params = cursor.execute.call_args.args

    assert "UPDATE research_sessions" in sql
    assert "final_report = %s" in sql
    assert params[1] == session_id


def test_save_report_raises_when_session_missing(mock_database):
    connection, cursor = mock_database

    session_id = uuid4()
    cursor.rowcount = 0

    with pytest.raises(
        ValueError,
        match="Research session not found",
    ):
        repository.save_report(
            session_id,
            {"title": "AI Safety Report"},
        )

# ---------------------------------------------------------
# QUESTION PERSISTENCE TESTS
# ---------------------------------------------------------

def test_save_questions_replaces_questions_and_applies_defaults(
    mock_database,
):
    connection, cursor = mock_database
    session_id = uuid4()

    questions = [
        {
            "question": "What is AI safety?",
            "search_queries": ["AI safety research"],
        },
        {
            "question": "What are its risks?",
            "search_queries": ["AI risks"],
            "parent_question_number": 1,
            "status": "completed",
        },
    ]

    repository.save_questions(session_id, questions)

    assert cursor.execute.call_count == 3

    delete_sql, delete_params = (
        cursor.execute.call_args_list[0].args
    )
    assert "DELETE FROM research_questions" in delete_sql
    assert delete_params == (session_id,)

    first_sql, first_params = (
        cursor.execute.call_args_list[1].args
    )
    assert "INSERT INTO research_questions" in first_sql
    assert first_params[0] == session_id
    assert first_params[1] == 1
    assert first_params[2] == "What is AI safety?"
    assert first_params[4] is None
    assert first_params[5] == "pending"

    second_sql, second_params = (
        cursor.execute.call_args_list[2].args
    )
    assert "INSERT INTO research_questions" in second_sql
    assert second_params[1] == 2
    assert second_params[2] == "What are its risks?"
    assert second_params[4] == 1
    assert second_params[5] == "completed"


def test_save_questions_handles_empty_list(mock_database):
    connection, cursor = mock_database
    session_id = uuid4()

    repository.save_questions(session_id, [])

    assert cursor.execute.call_count == 1

    sql, params = cursor.execute.call_args.args
    assert "DELETE FROM research_questions" in sql
    assert params == (session_id,)


# ---------------------------------------------------------
# EVIDENCE PERSISTENCE TESTS
# ---------------------------------------------------------

def test_save_evidence_replaces_evidence_records(mock_database):
    connection, cursor = mock_database
    session_id = uuid4()

    evidence_items = [
        {
            "research_question_number": 1,
            "claim": "AI safety is an active research field.",
            "source_url": "https://example.com/ai-safety",
            "supporting_text": "Researchers study AI safety.",
        },
        {
            "research_question_number": 2,
            "claim": "Evaluation helps identify model risks.",
            "source_url": "https://example.com/evaluation",
            "supporting_text": "Evaluations measure model behavior.",
        },
    ]

    repository.save_evidence(session_id, evidence_items)

    assert cursor.execute.call_count == 3

    delete_sql, delete_params = (
        cursor.execute.call_args_list[0].args
    )
    assert "DELETE FROM research_evidence" in delete_sql
    assert delete_params == (session_id,)

    first_sql, first_params = (
        cursor.execute.call_args_list[1].args
    )
    assert "INSERT INTO research_evidence" in first_sql
    assert first_params == (
        session_id,
        1,
        1,
        "AI safety is an active research field.",
        "https://example.com/ai-safety",
        "Researchers study AI safety.",
    )

    second_sql, second_params = (
        cursor.execute.call_args_list[2].args
    )
    assert "INSERT INTO research_evidence" in second_sql
    assert second_params[1] == 2
    assert second_params[2] == 2


def test_save_evidence_handles_empty_list(mock_database):
    connection, cursor = mock_database
    session_id = uuid4()

    repository.save_evidence(session_id, [])

    assert cursor.execute.call_count == 1

    sql, params = cursor.execute.call_args.args
    assert "DELETE FROM research_evidence" in sql
    assert params == (session_id,)

# ---------------------------------------------------------
# OBSERVABILITY EVENT PERSISTENCE TESTS
# ---------------------------------------------------------

@pytest.mark.parametrize(
    ("event_type", "message"),
    [
        ("", "Something happened"),
        ("   ", "Something happened"),
        ("node_started", ""),
        ("node_started", "   "),
    ],
)
def test_save_observability_event_rejects_empty_values(
    event_type,
    message,
):
    with pytest.raises(ValueError):
        repository.save_observability_event(
            event_type=event_type,
            message=message,
        )


@pytest.mark.parametrize(
    "log_level",
    ["TRACE", "INVALID", ""],
)
def test_save_observability_event_rejects_invalid_log_level(
    log_level,
):
    with pytest.raises(ValueError, match="Invalid log level"):
        repository.save_observability_event(
            event_type="node_started",
            message="Starting research",
            log_level=log_level,
        )


def test_save_observability_event_rejects_negative_duration():
    with pytest.raises(
        ValueError,
        match="duration_ms cannot be negative",
    ):
        repository.save_observability_event(
            event_type="node_completed",
            message="Research completed",
            duration_ms=-1,
        )


def test_save_observability_event_inserts_normalized_values(
    mock_database,
):
    connection, cursor = mock_database

    session_id = uuid4()

    repository.save_observability_event(
        event_type=" node_completed ",
        message=" Research completed ",
        session_id=session_id,
        node_name="researcher",
        log_level="warning",
        duration_ms=125.5,
        details={"sources_found": 4},
    )

    sql, params = cursor.execute.call_args.args

    assert "INSERT INTO observability_events" in sql
    assert params[0] == session_id
    assert params[1] == "node_completed"
    assert params[2] == "researcher"
    assert params[3] == "WARNING"
    assert params[4] == "Research completed"
    assert params[5] == 125.5


def test_save_observability_event_supports_application_events(
    mock_database,
):
    connection, cursor = mock_database

    repository.save_observability_event(
        event_type="application_started",
        message="Worker started",
        details=None,
    )

    sql, params = cursor.execute.call_args.args

    assert "INSERT INTO observability_events" in sql
    assert params[0] is None
    assert params[1] == "application_started"
    assert params[3] == "INFO"
    assert params[5] is None


def test_save_observability_event_rejects_invalid_session_uuid():
    with pytest.raises(ValueError):
        repository.save_observability_event(
            event_type="node_started",
            message="Starting",
            session_id="not-a-valid-uuid",
        )


# ---------------------------------------------------------
# OBSERVABILITY EVENT RETRIEVAL TESTS
# ---------------------------------------------------------

@pytest.mark.parametrize("limit", [0, -1, 501])
def test_get_observability_events_rejects_invalid_limit(limit):
    with pytest.raises(
        ValueError,
        match="limit must be between 1 and 500",
    ):
        repository.get_observability_events(limit=limit)


def test_get_observability_events_filters_by_session(
    mock_database,
):
    connection, cursor = mock_database

    session_id = uuid4()
    event_id = uuid4()

    cursor.fetchall.return_value = [
        (
            event_id,
            session_id,
            "node_started",
            "researcher",
            "INFO",
            "Research started",
            None,
            None,
            "created",
        ),
        (
            uuid4(),
            session_id,
            "node_completed",
            "researcher",
            "INFO",
            "Research completed",
            250.0,
            {"sources": 3},
            "completed",
        ),
    ]

    events = repository.get_observability_events(
        session_id=session_id,
        limit=20,
    )

    # Session-specific events are reversed into chronological order.
    assert len(events) == 2
    assert events[0]["event_type"] == "node_completed"
    assert events[1]["event_type"] == "node_started"

    assert events[0]["session_id"] == str(session_id)
    assert events[0]["details"] == {"sources": 3}
    assert events[1]["details"] == {}

    sql, params = cursor.execute.call_args.args
    assert "WHERE session_id = %s" in sql
    assert params == (session_id, 20)


def test_get_observability_events_returns_global_events(
    mock_database,
):
    connection, cursor = mock_database

    event_id = uuid4()

    cursor.fetchall.return_value = [
        (
            event_id,
            None,
            "application_started",
            None,
            "INFO",
            "Worker started",
            None,
            None,
            "created",
        )
    ]

    events = repository.get_observability_events(limit=10)

    assert len(events) == 1
    assert events[0]["session_id"] is None
    assert events[0]["details"] == {}

    sql, params = cursor.execute.call_args.args
    assert "WHERE session_id = %s" not in sql
    assert params == (10,)


def test_get_observability_events_returns_empty_list(
    mock_database,
):
    connection, cursor = mock_database
    cursor.fetchall.return_value = []

    assert repository.get_observability_events() == []


def test_get_observability_events_rejects_invalid_session_uuid():
    with pytest.raises(ValueError):
        repository.get_observability_events(
            session_id="not-a-valid-uuid",
        )


# ---------------------------------------------------------
# NODE EXECUTION METRICS TESTS
# ---------------------------------------------------------

def test_get_node_execution_metrics_aggregates_rows(
    mock_database,
):
    connection, cursor = mock_database

    cursor.fetchall.return_value = [
        ("researcher", 5, 4, 1, 1250, "success"),
        ("synthesizer", 3, 2, 1, 900.5, "failure"),
    ]

    metrics = repository.get_node_execution_metrics()

    assert metrics == {
        "researcher": {
            "calls": 5,
            "successes": 4,
            "failures": 1,
            "duration_ms": 1250.0,
            "last_status": "success",
        },
        "synthesizer": {
            "calls": 3,
            "successes": 2,
            "failures": 1,
            "duration_ms": 900.5,
            "last_status": "failure",
        },
    }

    sql = cursor.execute.call_args.args[0]

    assert "GROUP BY node_name" in sql
    assert "node_started" in sql
    assert "node_completed" in sql
    assert "node_failed" in sql


def test_get_node_execution_metrics_returns_empty_dict(
    mock_database,
):
    connection, cursor = mock_database
    cursor.fetchall.return_value = []

    assert repository.get_node_execution_metrics() == {}


def test_delete_session_success(mock_database):
    connection, cursor = mock_database
    session_id = uuid4()

    # No active job
    cursor.fetchone.return_value = None

    # Session exists and is deleted
    cursor.rowcount = 1

    result = repository.delete_session(session_id)

    assert result is True

    calls = cursor.execute.call_args_list

    assert len(calls) == 2

    # First query checks for active jobs
    assert "SELECT status" in calls[0].args[0]
    assert calls[0].args[1] == (session_id,)

    # Second query deletes the session
    assert "DELETE FROM research_sessions" in calls[1].args[0]
    assert calls[1].args[1] == (session_id,)

    connection.commit.assert_called_once()


def test_delete_session_not_found(mock_database):
    connection, cursor = mock_database
    session_id = uuid4()

    cursor.fetchone.return_value = None
    cursor.rowcount = 0

    result = repository.delete_session(session_id)

    assert result is False
    connection.commit.assert_called_once()


@pytest.mark.parametrize(
    "job_status",
    [
        "queued",
        "running",
        "retry_wait",
        "awaiting_review",
        "paused",
    ],
)
def test_delete_session_rejects_active_job(
    mock_database,
    job_status,
):
    connection, cursor = mock_database
    session_id = uuid4()

    # An active job was found
    cursor.fetchone.return_value = (job_status,)

    with pytest.raises(
        ValueError,
        match="Cancel or finish the active research job",
    ):
        repository.delete_session(session_id)

    # Must not delete the session
    assert cursor.execute.call_count == 1
    assert "SELECT status" in cursor.execute.call_args.args[0]

    connection.commit.assert_not_called()

def test_pause_latest_job_raises_when_session_does_not_exist(
    mock_database,
):
    _, cursor = mock_database
    session_id = uuid4()

    # No job found; the session lookup also finds nothing.
    cursor.fetchone.side_effect = [
        None,
        None,
    ]

    with pytest.raises(
        ValueError,
        match="Research session not found",
    ):
        repository.pause_latest_job(session_id)


def test_pause_latest_job_raises_when_session_has_no_job(
    mock_database,
):
    _, cursor = mock_database
    session_id = uuid4()

    # Session exists, but there is no associated job.
    cursor.fetchone.side_effect = [
        None,
        (session_id,),
    ]

    with pytest.raises(
        ValueError,
        match="session has no research job",
    ):
        repository.pause_latest_job(session_id)


def test_pause_latest_job_raises_when_running_job_is_being_cancelled(
    mock_database,
):
    _, cursor = mock_database
    session_id = uuid4()
    job_id = uuid4()

    running_row = (
        job_id,
        session_id,
        {"question": "Research AI safety"},
        "running",
        1,
        3,
        True,   # cancel_requested
        False,  # pause_requested
        None,
        None,
        None,
        None,
        None,
    )

    cursor.fetchone.side_effect = [
        running_row,  # Initial job lookup
        (session_id,),  # Session lock
        running_row,  # Fresh job lookup
    ]

    with pytest.raises(
        ValueError,
        match="Cancellation has already been requested",
    ):
        repository.pause_latest_job(session_id)


def test_pause_latest_job_returns_already_paused_job(
    mock_database,
):
    _, cursor = mock_database
    session_id = uuid4()
    job_id = uuid4()

    paused_row = (
        job_id,
        session_id,
        {"question": "Research AI safety"},
        "paused",
        1,
        3,
        False,
        False,
        None,
        None,
        None,
        None,
        None,
    )

    cursor.fetchone.side_effect = [
        paused_row,    # Initial job lookup
        (session_id,), # Session lock
        paused_row,    # Fresh job lookup
    ]

    result = repository.pause_latest_job(session_id)

    assert result["id"] == str(job_id)
    assert result["status"] == "paused"
    assert cursor.execute.call_count == 3


@pytest.mark.parametrize(
    "job_status",
    ["completed", "failed", "cancelled", "awaiting_review"],
)
def test_pause_latest_job_rejects_non_pauseable_status(
    mock_database,
    job_status,
):
    _, cursor = mock_database
    session_id = uuid4()
    job_id = uuid4()

    job_row = (
        job_id,
        session_id,
        {"question": "Research AI safety"},
        job_status,
        1,
        3,
        False,
        False,
        None,
        None,
        None,
        None,
        None,
    )

    cursor.fetchone.side_effect = [
        job_row,       # Initial job lookup
        (session_id,), # Session lock
        job_row,       # Fresh job lookup
    ]

    with pytest.raises(
        ValueError,
        match=f"Cannot pause a job with status {job_status}",
    ):
        repository.pause_latest_job(session_id)
