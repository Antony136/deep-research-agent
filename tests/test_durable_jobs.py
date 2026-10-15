from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace
from typing import TypedDict
from uuid import UUID, uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.database import repository
from app.job_config import is_recoverable_error, job_backoff_seconds
from app.services import research_service
from app.tools import web_search
from app.worker import process_job
from ddgs.exceptions import DDGSException


def _job(payload=None):
    return {
        "id": str(uuid4()),
        "session_id": str(uuid4()),
        "payload": payload or {"initial_state": {"value": 0}},
        "status": "running",
        "attempt_count": 1,
        "max_attempts": 3,
    }


def test_worker_persists_transient_failure_for_backoff_retry(monkeypatch):
    job = _job()
    failures = []
    monkeypatch.setattr(
        "app.worker.execute_research",
        lambda **_kwargs: (_ for _ in ()).throw(ConnectionError("temporary outage")),
    )
    monkeypatch.setattr(
        "app.worker.repository.get_job_control",
        lambda *_args: {"cancel_requested": False, "pause_requested": False},
    )
    monkeypatch.setattr(
        "app.worker.repository.renew_job_lease",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        "app.worker.repository.fail_or_retry_job",
        lambda **failure: failures.append(failure),
    )

    process_job(job, "worker-a", Event(), lease_seconds=300)

    assert failures[0]["retryable"] is True
    assert failures[0]["error_message"] == "temporary outage"
    assert failures[0]["backoff_seconds"] > 0


def test_worker_does_not_retry_permanent_failure(monkeypatch):
    job = _job()
    failures = []
    monkeypatch.setattr(
        "app.worker.execute_research",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("invalid graph state")),
    )
    monkeypatch.setattr(
        "app.worker.repository.get_job_control",
        lambda *_args: {"cancel_requested": False, "pause_requested": False},
    )
    monkeypatch.setattr(
        "app.worker.repository.renew_job_lease",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        "app.worker.repository.fail_or_retry_job",
        lambda **failure: failures.append(failure),
    )

    process_job(job, "worker-a", Event(), lease_seconds=300)

    assert failures[0]["retryable"] is False


def test_worker_shutdown_requeues_without_consuming_an_attempt(monkeypatch):
    job = _job()
    released = []
    shutdown = Event()
    shutdown.set()

    def fake_execute(**kwargs):
        kwargs["cancellation_check"]()
        return "completed"

    monkeypatch.setattr("app.worker.execute_research", fake_execute)
    monkeypatch.setattr(
        "app.worker.repository.get_job_control",
        lambda *_args: {"cancel_requested": False, "pause_requested": False},
    )
    monkeypatch.setattr(
        "app.worker.repository.renew_job_lease",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        "app.worker.repository.release_job_for_recovery",
        lambda *args: released.append(args),
    )

    process_job(job, "worker-a", shutdown, lease_seconds=300)

    assert released == [(UUID(job["id"]), "worker-a")]


def test_retry_policy_recognizes_search_provider_outages(monkeypatch):
    monkeypatch.setenv("RESEARCH_JOB_BACKOFF_SECONDS", "4")
    monkeypatch.setenv("RESEARCH_JOB_MAX_BACKOFF_SECONDS", "10")

    assert is_recoverable_error(DDGSException("provider unavailable"))
    assert job_backoff_seconds(1) == 4
    assert job_backoff_seconds(2) == 8
    assert job_backoff_seconds(3) == 10


def test_exhausted_web_search_error_reaches_worker_retry_policy(monkeypatch):
    class FailingSearch:
        def __init__(self, timeout):
            assert timeout == 10

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def text(self, **_kwargs):
            raise DDGSException("search service unavailable")

    monkeypatch.setattr(web_search, "DDGS", FailingSearch)
    monkeypatch.setattr(web_search.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(web_search.random, "uniform", lambda *_args: 0)

    with pytest.raises(DDGSException, match="unavailable"):
        web_search.search_web(
            "test search",
            max_retries=0,
            min_delay=0,
            max_delay=0,
        )


def test_worker_marks_job_cancelled_at_graph_checkpoint(monkeypatch):
    job = _job()
    completed = []
    shutdown = Event()
    shutdown.set()

    def fake_execute(**kwargs):
        kwargs["cancellation_check"]()
        return "completed"

    monkeypatch.setattr("app.worker.execute_research", fake_execute)
    monkeypatch.setattr(
        "app.worker.repository.get_job_control",
        lambda *_args: {"cancel_requested": True, "pause_requested": False},
    )
    monkeypatch.setattr(
        "app.worker.repository.renew_job_lease",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        "app.worker.repository.complete_job",
        lambda *args: completed.append(args) or True,
    )

    process_job(job, "worker-a", shutdown, lease_seconds=300)

    assert completed[0][2] == "cancelled"


def test_worker_pauses_job_at_graph_checkpoint(monkeypatch):
    job = _job()
    paused = []

    def fake_execute(**kwargs):
        kwargs["cancellation_check"]()
        return "completed"

    monkeypatch.setattr("app.worker.execute_research", fake_execute)
    monkeypatch.setattr(
        "app.worker.repository.get_job_control",
        lambda *_args: {"cancel_requested": False, "pause_requested": True},
    )
    monkeypatch.setattr(
        "app.worker.repository.renew_job_lease",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        "app.worker.repository.pause_running_job",
        lambda *args: paused.append(args),
    )

    process_job(job, "worker-a", Event(), lease_seconds=300)

    assert paused == [(UUID(job["id"]), "worker-a")]


def test_worker_resumes_pending_human_review_as_a_durable_job(monkeypatch):
    decision = {"action": "approve"}
    job = _job({"resume_decision": decision})
    completed = []
    received = []

    def fake_execute(**kwargs):
        received.append(kwargs)
        return "awaiting_review"

    monkeypatch.setattr("app.worker.execute_research", fake_execute)
    monkeypatch.setattr(
        "app.worker.repository.get_job_control",
        lambda *_args: {"cancel_requested": False, "pause_requested": False},
    )
    monkeypatch.setattr(
        "app.worker.repository.renew_job_lease",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        "app.worker.repository.complete_job",
        lambda *args: completed.append(args) or True,
    )

    process_job(job, "worker-a", Event(), lease_seconds=300)

    assert received[0]["resume_decision"] == decision
    assert received[0]["initial_state"] is None
    assert completed[0][2] == "awaiting_review"


class RecoveryState(TypedDict):
    value: int
    finished: bool


def test_crash_recovery_resumes_from_latest_langgraph_checkpoint(monkeypatch):
    checkpointer = InMemorySaver()
    calls = []
    fail_once = {"enabled": True}

    def first_node(state):
        calls.append("first")
        return {"value": state["value"] + 1}

    def flaky_node(_state):
        calls.append("flaky")
        if fail_once["enabled"]:
            fail_once["enabled"] = False
            raise ConnectionError("worker lost during node")
        return {"finished": True}

    builder = StateGraph(RecoveryState)
    builder.add_node("first", first_node)
    builder.add_node("flaky", flaky_node)
    builder.add_edge(START, "first")
    builder.add_edge("first", "flaky")
    builder.add_edge("flaky", END)
    graph = builder.compile(checkpointer=checkpointer)

    @contextmanager
    def use_checkpointer():
        yield checkpointer

    monkeypatch.setattr(research_service, "get_checkpointer", use_checkpointer)
    monkeypatch.setattr(research_service, "build_research_graph", lambda **_kwargs: graph)
    monkeypatch.setattr(research_service, "build_initial_state", lambda **_kwargs: {})
    monkeypatch.setattr(research_service, "persist_session_state", lambda *_args: None)

    session_id = uuid4()
    with pytest.raises(ConnectionError, match="worker lost"):
        research_service.execute_research(
            session_id,
            initial_state={"value": 0, "finished": False},
            raise_errors=True,
            manage_session_status=False,
        )

    outcome = research_service.execute_research(
        session_id,
        initial_state={"value": 0, "finished": False},
        raise_errors=True,
        manage_session_status=False,
    )

    assert outcome == "completed"
    assert calls == ["first", "flaky", "flaky"]
    snapshot = graph.get_state(research_service.get_graph_config(session_id))
    assert snapshot.values["value"] == 1
    assert snapshot.values["finished"] is True


def test_pause_checkpoint_resumes_without_replaying_completed_nodes(monkeypatch):
    checkpointer = InMemorySaver()
    calls = []

    def first_node(state):
        calls.append("first")
        return {"value": state["value"] + 1}

    def final_node(_state):
        calls.append("final")
        return {"finished": True}

    builder = StateGraph(RecoveryState)
    builder.add_node("first", first_node)
    builder.add_node("final", final_node)
    builder.add_edge(START, "first")
    builder.add_edge("first", "final")
    builder.add_edge("final", END)
    graph = builder.compile(checkpointer=checkpointer)

    @contextmanager
    def use_checkpointer():
        yield checkpointer

    monkeypatch.setattr(research_service, "get_checkpointer", use_checkpointer)
    monkeypatch.setattr(research_service, "build_research_graph", lambda **_kwargs: graph)
    monkeypatch.setattr(research_service, "build_initial_state", lambda **_kwargs: {})
    monkeypatch.setattr(research_service, "persist_session_state", lambda *_args: None)

    session_id = uuid4()
    checkpoint_checks = {"count": 0}

    def pause_after_first_checkpoint():
        checkpoint_checks["count"] += 1
        if checkpoint_checks["count"] == 2:
            raise RuntimeError("pause at checkpoint")

    with pytest.raises(RuntimeError, match="pause at checkpoint"):
        research_service.execute_research(
            session_id,
            initial_state={"value": 0, "finished": False},
            raise_errors=True,
            manage_session_status=False,
            cancellation_check=pause_after_first_checkpoint,
        )

    outcome = research_service.execute_research(
        session_id,
        initial_state={"value": 0, "finished": False},
        raise_errors=True,
        manage_session_status=False,
    )

    snapshot = graph.get_state(research_service.get_graph_config(session_id))
    assert outcome == "completed"
    assert snapshot.values["value"] == 1
    assert snapshot.values["finished"] is True
    assert calls == ["first", "final"]


class ReviewState(TypedDict):
    approved: bool
    finished: bool


def test_human_review_interrupt_survives_as_a_resumable_checkpoint(monkeypatch):
    checkpointer = InMemorySaver()
    calls = []

    def review_node(_state):
        calls.append("review")
        interrupt({"type": "initial_plan_review"})
        return {"approved": True}

    builder = StateGraph(ReviewState)
    builder.add_node("review", review_node)
    builder.add_edge(START, "review")
    builder.add_edge("review", END)
    graph = builder.compile(checkpointer=checkpointer)

    @contextmanager
    def use_checkpointer():
        yield checkpointer

    monkeypatch.setattr(research_service, "get_checkpointer", use_checkpointer)
    monkeypatch.setattr(research_service, "build_research_graph", lambda **_kwargs: graph)
    monkeypatch.setattr(research_service, "build_initial_state", lambda **_kwargs: {})
    monkeypatch.setattr(research_service, "persist_session_state", lambda *_args: None)

    session_id = uuid4()
    first_outcome = research_service.execute_research(
        session_id,
        initial_state={"approved": False, "finished": False},
        raise_errors=True,
        manage_session_status=False,
    )
    resume_outcome = research_service.execute_research(
        session_id,
        resume_decision={"action": "approve"},
        raise_errors=True,
        manage_session_status=False,
    )

    snapshot = graph.get_state(research_service.get_graph_config(session_id))
    assert first_outcome == "awaiting_review"
    assert resume_outcome == "completed"
    assert snapshot.values["approved"] is True
    assert calls == ["review", "review"]


def test_retry_after_review_decision_resumes_without_replaying_review(
    monkeypatch,
):
    checkpointer = InMemorySaver()
    calls = []
    fail_once = {"enabled": True}

    def review_node(_state):
        calls.append("review")
        interrupt({"type": "initial_plan_review"})
        return {"approved": True}

    def flaky_node(_state):
        calls.append("flaky")
        if fail_once["enabled"]:
            fail_once["enabled"] = False
            raise ConnectionError("failure after review decision")
        return {"finished": True}

    builder = StateGraph(ReviewState)
    builder.add_node("review", review_node)
    builder.add_node("flaky", flaky_node)
    builder.add_edge(START, "review")
    builder.add_edge("review", "flaky")
    builder.add_edge("flaky", END)
    graph = builder.compile(checkpointer=checkpointer)

    @contextmanager
    def use_checkpointer():
        yield checkpointer

    monkeypatch.setattr(research_service, "get_checkpointer", use_checkpointer)
    monkeypatch.setattr(research_service, "build_research_graph", lambda **_kwargs: graph)
    monkeypatch.setattr(research_service, "build_initial_state", lambda **_kwargs: {})
    monkeypatch.setattr(research_service, "persist_session_state", lambda *_args: None)

    session_id = uuid4()
    assert research_service.execute_research(
        session_id,
        initial_state={"approved": False, "finished": False},
        raise_errors=True,
        manage_session_status=False,
    ) == "awaiting_review"

    decision = {"action": "approve"}
    with pytest.raises(ConnectionError, match="after review"):
        research_service.execute_research(
            session_id,
            resume_decision=decision,
            raise_errors=True,
            manage_session_status=False,
        )

    outcome = research_service.execute_research(
        session_id,
        resume_decision=decision,
        raise_errors=True,
        manage_session_status=False,
    )

    assert outcome == "completed"
    assert calls == ["review", "review", "flaky", "flaky"]


def test_reclaim_query_uses_expired_lease_and_skip_locked(monkeypatch):
    job_id = uuid4()
    session_id = uuid4()
    row = (
        job_id,
        session_id,
        {"initial_state": {"value": 0}},
        "running",
        2,
        3,
        False,
        False,
        "previous attempt failed",
        "created",
        "updated",
        "started",
        None,
    )

    class FakeCursor:
        query = ""
        queries = []
        rowcount = 1

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, query, _params=None):
            self.query = query
            self.queries.append(query)

        def fetchall(self):
            return []

        def fetchone(self):
            if "UPDATE research_jobs" in self.query and "attempt_count = attempt_count + 1" in self.query:
                return (*row[:4], row[4] + 1, *row[5:])
            if "FROM research_jobs" in self.query:
                return row
            return (session_id,)

    cursor = FakeCursor()

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return cursor

    @contextmanager
    def fake_get_connection():
        yield FakeConnection()

    monkeypatch.setattr(repository, "get_connection", fake_get_connection)
    claimed = repository.claim_next_job("worker-b", lease_seconds=90)

    assert claimed["attempt_count"] == 3
    assert claimed["payload"] == row[2]
    assert any("lease_expires_at <= NOW()" in query for query in cursor.queries)
    assert any("FOR UPDATE SKIP LOCKED" in query for query in cursor.queries)


def test_cancel_queued_job_marks_session_cancelled(monkeypatch):
    job_id = uuid4()
    session_id = uuid4()
    row = (
        job_id,
        session_id,
        {"initial_state": {"value": 0}},
        "queued",
        0,
        3,
        False,
        False,
        None,
        "created",
        "created",
        None,
        None,
    )

    class FakeCursor:
        rowcount = 1

        def __init__(self):
            self.query = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, query, _params=None):
            self.query = query

        def fetchone(self):
            if "FROM research_jobs" in self.query:
                return row
            return (session_id,)

    cursor = FakeCursor()

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return cursor

    @contextmanager
    def fake_get_connection():
        yield FakeConnection()

    monkeypatch.setattr(repository, "get_connection", fake_get_connection)
    result = repository.cancel_latest_job(session_id)

    assert result["status"] == "cancelled"
    assert result["cancel_requested"] is True


def test_pause_queued_job_moves_job_and_session_to_paused(monkeypatch):
    job_id = uuid4()
    session_id = uuid4()
    row = (
        job_id,
        session_id,
        {"initial_state": {"value": 1}},
        "queued",
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

    class FakeCursor:
        rowcount = 1

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, query, _params=None):
            self.query = query

        def fetchone(self):
            if "FROM research_jobs" in self.query:
                return row
            return (session_id,)

    cursor = FakeCursor()

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return cursor

    @contextmanager
    def fake_get_connection():
        yield FakeConnection()

    monkeypatch.setattr(repository, "get_connection", fake_get_connection)

    job = repository.pause_latest_job(session_id)

    assert job["status"] == "paused"
    assert job["pause_requested"] is False


def test_resume_paused_job_requeues_same_job_and_checkpoint(monkeypatch):
    job_id = uuid4()
    session_id = uuid4()
    payload = {"resume_decision": {"action": "approve"}}
    paused_row = (
        job_id,
        session_id,
        payload,
        "paused",
        2,
        3,
        False,
        False,
        None,
        None,
        None,
        None,
        None,
    )
    queued_row = (*paused_row[:3], "queued", *paused_row[4:])

    class FakeCursor:
        rowcount = 1

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, query, _params=None):
            self.query = query

        def fetchone(self):
            if "UPDATE research_jobs" in self.query:
                return queued_row
            if "FROM research_jobs" in self.query:
                return paused_row
            return ("paused",)

    cursor = FakeCursor()

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return cursor

    @contextmanager
    def fake_get_connection():
        yield FakeConnection()

    monkeypatch.setattr(repository, "get_connection", fake_get_connection)

    job = repository.resume_paused_job(session_id)

    assert job["id"] == str(job_id)
    assert job["payload"] == payload
    assert job["status"] == "queued"
    assert job["attempt_count"] == 2


def test_pause_transition_prefers_concurrent_cancellation(monkeypatch):
    job_id = uuid4()
    session_id = uuid4()
    cursor = SimpleNamespace(query="", params=[], queries=[])

    def execute(query, params=None):
        cursor.query = query
        cursor.params = params
        cursor.queries.append(query)

    def fetchone():
        if "SELECT cancel_requested, pause_requested" in cursor.query:
            return True, True
        return (session_id,)

    cursor.execute = execute
    cursor.fetchone = fetchone

    class FakeCursor:
        def __enter__(self):
            return cursor

        def __exit__(self, *_args):
            return None

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return FakeCursor()

    @contextmanager
    def fake_get_connection():
        yield FakeConnection()

    monkeypatch.setattr(repository, "get_connection", fake_get_connection)

    outcome = repository.pause_running_job(job_id, "worker-a")

    assert outcome == "cancelled"
    assert any(
        "SET status = %s" in query and "UPDATE research_jobs" in query
        for query in cursor.queries
    )


def test_node_metrics_are_read_from_persisted_cross_process_events(monkeypatch):
    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, query):
            assert "FROM observability_events" in query
            assert "GROUP BY node_name" in query

        def fetchall(self):
            return [("researcher", 3, 2, 1, 1500.5, "failure")]

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return FakeCursor()

    @contextmanager
    def fake_get_connection():
        yield FakeConnection()

    monkeypatch.setattr(repository, "get_connection", fake_get_connection)

    metrics = repository.get_node_execution_metrics()

    assert metrics["researcher"] == {
        "calls": 3,
        "successes": 2,
        "failures": 1,
        "duration_ms": 1500.5,
        "last_status": "failure",
    }


def test_legacy_running_session_can_be_resumed_without_job_history(monkeypatch):
    session_id = uuid4()
    job_id = uuid4()
    payload = {"initial_state": {"question": "resume this"}}
    job_row = (
        job_id,
        session_id,
        payload,
        "queued",
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

    class FakeCursor:
        def __init__(self):
            self.responses = [None, ("running",), None, job_row]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, _query, _params=None):
            return None

        def fetchone(self):
            return self.responses.pop(0)

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return FakeCursor()

    @contextmanager
    def fake_get_connection():
        yield FakeConnection()

    monkeypatch.setattr(repository, "get_connection", fake_get_connection)

    job = repository.enqueue_retry_job(
        session_id,
        max_attempts=3,
        fallback_payload=payload,
    )

    assert job["id"] == str(job_id)
    assert job["payload"] == payload
    assert job["status"] == "queued"


@pytest.mark.parametrize(
    ("attempt_count", "cancel_requested", "pause_requested", "retryable", "expected"),
    [
        (1, False, False, True, "retry_wait"),
        (3, False, False, True, "failed"),
        (1, False, False, False, "failed"),
        (1, True, False, True, "cancelled"),
        (1, False, True, True, "paused"),
    ],
)
def test_failure_transition_respects_attempt_limit_and_cancellation(
    monkeypatch,
    attempt_count,
    cancel_requested,
    pause_requested,
    retryable,
    expected,
):
    job_id = uuid4()
    session_id = uuid4()
    cursor = SimpleNamespace(query="", queries=[])

    def execute(query, _params=None):
        cursor.query = query
        cursor.queries.append(query)

    def fetchone():
        if "FROM research_jobs" in cursor.query:
            if "attempt_count" in cursor.query:
                return (
                    session_id,
                    attempt_count,
                    3,
                    cancel_requested,
                    pause_requested,
                )
            return (session_id,)
        return (session_id,)

    cursor.execute = execute
    cursor.fetchone = fetchone

    class FakeCursor:
        def __enter__(self):
            return cursor

        def __exit__(self, *_args):
            return None

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return FakeCursor()

    @contextmanager
    def fake_get_connection():
        yield FakeConnection()

    monkeypatch.setattr(repository, "get_connection", fake_get_connection)
    result = repository.fail_or_retry_job(
        job_id=job_id,
        worker_id="worker-a",
        error_message="test failure",
        retryable=retryable,
        backoff_seconds=7,
    )

    assert result == expected
    assert any("make_interval(secs => %s)" in query for query in cursor.queries)
