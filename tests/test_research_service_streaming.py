from contextlib import contextmanager
from types import SimpleNamespace
from typing import TypedDict
from uuid import uuid4

from app.database import repository
from app.services.research_service import stream_graph_updates
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class StreamingTestState(TypedDict):
    value: int
    verified: int


def test_stream_graph_updates_persists_each_node_snapshot(monkeypatch):
    snapshots = [
        {"research_questions": [{"question": "Question 1"}], "evidence": []},
        {
            "research_questions": [{"question": "Question 1"}],
            "evidence": [{"claim": "Verified claim"}],
        },
    ]
    persisted = []

    class FakeGraph:
        def __init__(self):
            self.current_state = {}

        def stream(self, graph_input, config, stream_mode):
            assert graph_input == {"question": "Test"}
            assert config == {"configurable": {"thread_id": "session"}}
            assert stream_mode == "updates"
            self.current_state = dict(graph_input)
            for state in snapshots:
                self.current_state = {**self.current_state, **state}
                yield {"node": state}

        def get_state(self, config):
            return SimpleNamespace(values=self.current_state)

    monkeypatch.setattr(
        "app.services.research_service.persist_session_state",
        lambda session_id, state: persisted.append(state.copy()),
    )
    session_id = uuid4()
    graph = FakeGraph()

    result = stream_graph_updates(
        graph,
        {"question": "Test"},
        {"configurable": {"thread_id": "session"}},
        session_id,
    )

    expected_snapshots = [
        {"question": "Test", **state}
        for state in snapshots
    ]
    assert persisted == expected_snapshots
    assert result == expected_snapshots[-1]


def test_stream_graph_updates_reads_each_checkpointer_snapshot(monkeypatch):
    persisted = []

    def first_node(state):
        return {"value": state["value"] + 1}

    def verification_node(state):
        return {"verified": state["value"] + 1}

    builder = StateGraph(StreamingTestState)
    builder.add_node("first", first_node)
    builder.add_node("verification", verification_node)
    builder.add_edge(START, "first")
    builder.add_edge("first", "verification")
    builder.add_edge("verification", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    monkeypatch.setattr(
        "app.services.research_service.persist_session_state",
        lambda session_id, state: persisted.append(state.copy()),
    )
    session_id = uuid4()
    config = {"configurable": {"thread_id": str(session_id)}}

    result = stream_graph_updates(
        graph,
        {"value": 0, "verified": 0},
        config,
        session_id,
    )

    assert persisted == [
        {"value": 1, "verified": 0},
        {"value": 1, "verified": 2},
    ]
    assert result == persisted[-1]


def test_session_events_returns_most_recent_events_in_time_order(monkeypatch):
    rows = [
        (3, "session", "node_completed", "verifier", "INFO", "done", 12.0, {}, "t3"),
        (2, "session", "node_started", "verifier", "INFO", "started", None, {}, "t2"),
        (1, "session", "node_started", "researcher", "INFO", "started", None, {}, "t1"),
    ]

    class FakeCursor:
        query = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, query, _params):
            self.query = query

        def fetchall(self):
            return rows[:2]

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

    events = repository.get_observability_events(
        session_id=uuid4(),
        limit=2,
    )

    assert "ORDER BY created_at DESC, id DESC" in cursor.query
    assert [event["id"] for event in events] == [2, 3]
