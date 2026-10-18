from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api import routes
from app.schemas.api import ResearchRequest


def make_session(session_id=None, **overrides):
    session_id = session_id or uuid4()

    session = {
        "id": str(session_id),
        "question": "Research AI safety",
        "status": "running",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "research_state": None,
        "final_report": None,
        "error_message": None,
        "completed_at": None,
        "job": None,
    }

    session.update(overrides)
    return session


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------


def test_health_check_returns_ok():
    response = routes.health_check()

    assert response.status == "ok"
    assert response.service == "deep-research-agent"


# ---------------------------------------------------------
# CREATE RESEARCH
# ---------------------------------------------------------


def test_create_research_strips_question_and_queues_job(
    monkeypatch,
):
    session_id = uuid4()
    job_id = uuid4()
    captured = {}

    monkeypatch.setattr(
        routes,
        "build_initial_state",
        lambda **kwargs: captured.setdefault("state_args", kwargs)
        or {"question": kwargs["question"]},
    )

    # Return a predictable state without requiring the graph.
    monkeypatch.setattr(
        routes,
        "build_initial_state",
        lambda **kwargs: (
            captured.update(state_args=kwargs)
            or {"question": kwargs["question"]}
        ),
    )

    monkeypatch.setattr(
        routes,
        "to_json_compatible",
        lambda state: state,
    )

    monkeypatch.setattr(
        routes,
        "max_job_attempts",
        lambda: 3,
    )

    def fake_create_session_with_job(**kwargs):
        captured["job_args"] = kwargs
        return session_id, job_id

    monkeypatch.setattr(
        routes,
        "create_session_with_job",
        fake_create_session_with_job,
    )

    response = routes.create_research(
        ResearchRequest(question="  Research AI safety  ")
    )

    assert response.session_id == str(session_id)
    assert response.job_id == str(job_id)
    assert response.status == "created"
    assert captured["job_args"]["question"] == "Research AI safety"
    assert captured["job_args"]["max_attempts"] == 3
    assert captured["state_args"]["question"] == "Research AI safety"


def test_create_research_rejects_whitespace_question():
    with pytest.raises(HTTPException) as exc_info:
        routes.create_research(
            ResearchRequest(question="   ")
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Research question cannot be empty."


def test_create_research_returns_500_when_creation_fails(
    monkeypatch,
):
    monkeypatch.setattr(
        routes,
        "build_initial_state",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("internal failure")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        routes.create_research(
            ResearchRequest(question="Research AI safety")
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Could not create the research session."


# ---------------------------------------------------------
# LIST RESEARCH SESSIONS
# ---------------------------------------------------------


def test_get_research_sessions_returns_summaries(monkeypatch):
    session_id = uuid4()
    session = make_session(session_id)

    monkeypatch.setattr(
        routes,
        "list_research_sessions",
        lambda limit: [session],
    )

    response = routes.get_research_sessions(limit=10)

    assert len(response.sessions) == 1
    assert response.sessions[0].id == str(session_id)
    assert response.sessions[0].question == "Research AI safety"


def test_get_research_sessions_returns_500_on_service_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        routes,
        "list_research_sessions",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("database failure")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        routes.get_research_sessions(limit=10)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Could not retrieve research sessions."


# ---------------------------------------------------------
# GET A RESEARCH SESSION
# ---------------------------------------------------------


def test_get_research_session_returns_session(monkeypatch):
    session_id = uuid4()
    session = make_session(session_id)

    monkeypatch.setattr(
        routes,
        "get_research_session",
        lambda requested_id: (
            session if requested_id == session_id else None
        ),
    )

    response = routes.get_research_session_endpoint(session_id)

    assert response.id == str(session_id)
    assert response.question == "Research AI safety"


def test_get_research_session_returns_404_when_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        routes,
        "get_research_session",
        lambda _session_id: None,
    )

    with pytest.raises(HTTPException) as exc_info:
        routes.get_research_session_endpoint(uuid4())

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Research session not found."


def test_get_research_session_returns_500_on_service_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        routes,
        "get_research_session",
        lambda _session_id: (_ for _ in ()).throw(
            RuntimeError("database failure")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        routes.get_research_session_endpoint(uuid4())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Could not retrieve the research session."


# ---------------------------------------------------------
# PENDING HUMAN REVIEW
# ---------------------------------------------------------


def test_get_pending_review_returns_review(monkeypatch):
    session_id = uuid4()
    session = make_session(
        session_id,
        status="waiting_for_review",
    )
    review = {"value": {"action": "approve"}}

    monkeypatch.setattr(
        routes,
        "get_research_session",
        lambda _session_id: session,
    )
    monkeypatch.setattr(
        routes,
        "get_pending_review",
        lambda _session_id: review,
    )

    response = routes.get_pending_review_endpoint(session_id)

    assert response.session_id == str(session_id)
    assert response.review == review


def test_get_pending_review_returns_404_for_missing_session(
    monkeypatch,
):
    monkeypatch.setattr(
        routes,
        "get_research_session",
        lambda _session_id: None,
    )

    with pytest.raises(HTTPException) as exc_info:
        routes.get_pending_review_endpoint(uuid4())

    assert exc_info.value.status_code == 404


def test_get_pending_review_returns_409_when_no_review(
    monkeypatch,
):
    monkeypatch.setattr(
        routes,
        "get_research_session",
        lambda _session_id: make_session(),
    )
    monkeypatch.setattr(
        routes,
        "get_pending_review",
        lambda _session_id: None,
    )

    with pytest.raises(HTTPException) as exc_info:
        routes.get_pending_review_endpoint(uuid4())

    assert exc_info.value.status_code == 409


# ---------------------------------------------------------
# RETRY
# ---------------------------------------------------------


def test_retry_research_queues_job(monkeypatch):
    session_id = uuid4()
    job_id = str(uuid4())
    captured = {}

    monkeypatch.setattr(
        routes,
        "get_research_session",
        lambda _session_id: make_session(
            session_id,
            question="Research AI safety",
        ),
    )
    monkeypatch.setattr(
        routes,
        "build_initial_state",
        lambda question: {"question": question},
    )
    monkeypatch.setattr(
        routes,
        "to_json_compatible",
        lambda state: state,
    )
    monkeypatch.setattr(routes, "max_job_attempts", lambda: 3)

    def fake_enqueue(session, max_attempts, fallback_payload):
        captured.update(
            session=session,
            max_attempts=max_attempts,
            fallback_payload=fallback_payload,
        )
        return {
            "id": job_id,
            "status": "queued",
            "attempt_count": 0,
            "max_attempts": 3,
        }

    monkeypatch.setattr(routes, "enqueue_retry_job", fake_enqueue)

    response = routes.retry_research(session_id)

    assert response.job_id == job_id
    assert response.job_status == "queued"
    assert captured["session"] == session_id
    assert captured["max_attempts"] == 3
    assert captured["fallback_payload"]["initial_state"] == {
        "question": "Research AI safety"
    }


def test_retry_research_returns_409_for_invalid_transition(
    monkeypatch,
):
    monkeypatch.setattr(
        routes,
        "get_research_session",
        lambda _session_id: make_session(),
    )
    monkeypatch.setattr(
        routes,
        "build_initial_state",
        lambda question: {"question": question},
    )
    monkeypatch.setattr(
        routes,
        "to_json_compatible",
        lambda state: state,
    )
    monkeypatch.setattr(routes, "max_job_attempts", lambda: 3)
    monkeypatch.setattr(
        routes,
        "enqueue_retry_job",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("A retry is not allowed.")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        routes.retry_research(uuid4())

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "A retry is not allowed."


# ---------------------------------------------------------
# CANCEL, PAUSE AND RESUME
# ---------------------------------------------------------


@pytest.mark.parametrize(
    ("endpoint_name", "repository_name", "job_status", "expected_status"),
    [
        ("cancel_research", "cancel_latest_job", "cancelled", "cancelled"),
        ("pause_research", "pause_latest_job", "paused", "paused"),
        ("resume_research", "resume_paused_job", "queued", "running"),
    ],
)
def test_job_control_endpoints_return_job_state(
    monkeypatch,
    endpoint_name,
    repository_name,
    job_status,
    expected_status,
):
    session_id = uuid4()
    job = {
        "id": str(uuid4()),
        "status": job_status,
        "attempt_count": 1,
        "max_attempts": 3,
        "cancel_requested": False,
        "pause_requested": False,
        "last_error": None,
    }

    monkeypatch.setattr(
        routes,
        repository_name,
        lambda *_args, **_kwargs: job,
    )

    response = getattr(routes, endpoint_name)(session_id)

    assert response.session_id == str(session_id)
    assert response.job_status == job_status
    assert response.status == expected_status


@pytest.mark.parametrize(
    "endpoint_name",
    ["cancel_research", "pause_research", "resume_research"],
)
def test_job_control_endpoints_return_409_for_invalid_transition(
    monkeypatch,
    endpoint_name,
):
    repository_name = {
        "cancel_research": "cancel_latest_job",
        "pause_research": "pause_latest_job",
        "resume_research": "resume_paused_job",
    }[endpoint_name]

    monkeypatch.setattr(
        routes,
        repository_name,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("Invalid job transition.")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        getattr(routes, endpoint_name)(uuid4())

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Invalid job transition."


# ---------------------------------------------------------
# DELETE SESSION
# ---------------------------------------------------------


def test_delete_research_session_returns_deleted(monkeypatch):
    session_id = uuid4()

    monkeypatch.setattr(
        routes,
        "get_research_session",
        lambda _session_id: make_session(),
    )

    from app.database import repository

    monkeypatch.setattr(
        repository,
        "delete_session",
        lambda _session_id: True,
    )

    response = routes.delete_research_session_endpoint(session_id)

    assert response.deleted is True
    assert response.session_id == str(session_id)


def test_delete_research_session_returns_404_when_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        routes,
        "get_research_session",
        lambda _session_id: None,
    )

    with pytest.raises(HTTPException) as exc_info:
        routes.delete_research_session_endpoint(uuid4())

    assert exc_info.value.status_code == 404


# ---------------------------------------------------------
# OBSERVABILITY
# ---------------------------------------------------------


def test_get_session_events_returns_events(monkeypatch):
    from app.database import repository

    session_id = uuid4()
    event = {
        "id": 1,
        "session_id": str(session_id),
        "event_type": "node_started",
        "log_level": "INFO",
        "message": "Research started.",
        "created_at": datetime.now(timezone.utc),
    }
    captured = {}

    def fake_events(**kwargs):
        captured.update(kwargs)
        return [event]

    monkeypatch.setattr(
        repository,
        "get_observability_events",
        fake_events,
    )

    response = routes.get_session_events_endpoint(
        session_id,
        limit=25,
    )

    assert len(response.events) == 1
    assert response.events[0].id == 1
    assert captured == {
        "session_id": session_id,
        "limit": 25,
    }

def test_get_global_events_returns_events(monkeypatch):
    from app.database import repository

    captured = {}

    def fake_events(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        repository,
        "get_observability_events",
        fake_events,
    )

    response = routes.get_global_events_endpoint(limit=10)

    assert response.events == []
    assert captured == {
        "session_id": None,
        "limit": 10,
    }


def test_metrics_endpoint_returns_snapshot(monkeypatch):
    from app.database import repository

    monkeypatch.setattr(
        repository,
        "get_node_execution_metrics",
        lambda: {"researcher": {"calls": 2}},
    )

    response = routes.get_metrics_endpoint()

    assert response["started_at"] is None
    assert response["nodes"]["researcher"]["calls"] == 2
