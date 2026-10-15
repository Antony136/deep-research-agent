import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from app import api_main
from app.api import routes
from app.observability import tracing


def test_workbench_refresh_returns_session_events_and_pending_review(
    monkeypatch,
):
    session_id = uuid4()
    session = {
        "id": str(session_id),
        "question": "What is the research question?",
        "status": "waiting_for_review",
    }
    review = {"value": {"action": "approve"}}
    events = [
        {
            "id": 1,
            "session_id": str(session_id),
            "event_type": "node_started",
            "log_level": "INFO",
            "message": "Node started.",
            "created_at": datetime.now(timezone.utc),
        }
    ]
    event_requests = []
    review_requests = []
    monkeypatch.setattr(
        routes,
        "get_research_session",
        lambda requested_id: session if requested_id == session_id else None,
    )
    monkeypatch.setattr(
        routes,
        "get_observability_events",
        lambda **kwargs: event_requests.append(kwargs) or events,
    )
    monkeypatch.setattr(
        routes,
        "get_pending_review",
        lambda requested_id: review_requests.append(requested_id) or review,
    )

    response = routes.get_research_workbench_refresh(session_id)

    assert response.session.id == str(session_id)
    assert [event.id for event in response.events] == [1]
    assert response.pending_review == review
    assert response.events_error is None
    assert event_requests == [{"session_id": session_id, "limit": 150}]
    assert review_requests == [session_id]


def test_workbench_refresh_surfaces_node_event_failure(monkeypatch):
    session_id = uuid4()
    session = {
        "id": str(session_id),
        "question": "What is the research question?",
        "status": "running",
    }
    review_requests = []
    monkeypatch.setattr(
        routes,
        "get_research_session",
        lambda _requested_id: session,
    )
    monkeypatch.setattr(
        routes,
        "get_observability_events",
        lambda **_kwargs: (_ for _ in ()).throw(ConnectionError("database unavailable")),
    )
    monkeypatch.setattr(
        routes,
        "get_pending_review",
        lambda requested_id: review_requests.append(requested_id),
    )

    response = routes.get_research_workbench_refresh(session_id)

    assert response.session.id == str(session_id)
    assert response.events == []
    assert response.events_error == "Could not retrieve node events."
    assert response.pending_review is None
    assert review_requests == []


def test_api_lifespan_configures_node_event_persistence(monkeypatch):
    configured = []
    monkeypatch.setattr(
        api_main,
        "configure_observability",
        lambda: configured.append(True),
    )

    async def run_lifespan():
        async with api_main.lifespan(api_main.app):
            assert configured == [True]

    asyncio.run(run_lifespan())


def test_instrumented_node_persists_started_and_completed_events(
    monkeypatch,
):
    session_id = uuid4()
    events = []
    monkeypatch.setattr(
        tracing,
        "save_observability_event",
        lambda **event: events.append(event),
    )

    logger = logging.getLogger("deep_research_agent")
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    handler = tracing.PostgreSQLEventHandler()
    logger.addHandler(handler)
    try:
        node = tracing.instrument_node(
            "researcher",
            lambda state: {**state, "completed": True},
        )
        result = node(
            {"started": True},
            config={"configurable": {"thread_id": str(session_id)}},
        )
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)

    assert result == {"started": True, "completed": True}
    assert [event["event_type"] for event in events] == [
        "node_started",
        "node_completed",
    ]
    assert all(event["session_id"] == session_id for event in events)
    assert all(event["node_name"] == "researcher" for event in events)
