import asyncio
import logging
from uuid import uuid4

from app import api_main
from app.observability import tracing


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
