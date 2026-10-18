from datetime import date, datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.services import research_service


# ------------------------------------------------------------
# JSON SERIALIZATION
# ------------------------------------------------------------

def test_to_json_compatible_converts_pydantic_models():
    from pydantic import BaseModel

    class Example(BaseModel):
        name: str
        count: int

    result = research_service.to_json_compatible(
        Example(name="research", count=3)
    )

    assert result == {
        "name": "research",
        "count": 3,
    }


def test_to_json_compatible_converts_nested_values():
    session_id = uuid4()
    timestamp = datetime(2026, 1, 2, tzinfo=timezone.utc)

    result = research_service.to_json_compatible(
        {
            "id": session_id,
            "date": date(2026, 1, 2),
            "timestamp": timestamp,
            "values": (1, 2),
            "tags": {"a", "b"},
        }
    )

    assert result["id"] == str(session_id)
    assert result["date"] == "2026-01-02"
    assert result["timestamp"] == timestamp.isoformat()
    assert result["values"] == [1, 2]
    assert set(result["tags"]) == {"a", "b"}


def test_to_json_compatible_rejects_unsupported_objects():
    with pytest.raises(TypeError):
        research_service.to_json_compatible(object())


# ------------------------------------------------------------
# GRAPH CONFIGURATION AND INTERRUPTS
# ------------------------------------------------------------

def test_get_graph_config_uses_session_id_as_thread_id():
    session_id = uuid4()

    result = research_service.get_graph_config(session_id)

    assert result == {
        "configurable": {
            "thread_id": str(session_id),
        }
    }


def test_extract_interrupts_returns_empty_list_for_missing_interrupts():
    assert research_service.extract_interrupts({}) == []
    assert research_service.extract_interrupts(
        {"other_key": "value"}
    ) == []


def test_extract_interrupts_returns_interrupt_payloads():
    interrupts = [
        SimpleNamespace(
            value={"type": "initial_plan_review"}
        ),
        SimpleNamespace(
            value={"type": "final_report_review"}
        ),
    ]

    result = research_service.extract_interrupts(
        {"__interrupt__": interrupts}
    )

    assert result == [
        {"type": "initial_plan_review"},
        {"type": "final_report_review"},
    ]


# ------------------------------------------------------------
# INITIAL STATE
# ------------------------------------------------------------

def test_build_initial_state_contains_required_defaults():
    state = research_service.build_initial_state(
        question="Research AI safety",
    )

    assert state["question"] == "Research AI safety"
    assert state["research_questions"] == []
    assert state["evidence"] == []
    assert state["research_complete"] is False
    assert state["current_question_index"] == 0


# ------------------------------------------------------------
# REVIEW DECISION VALIDATION
# ------------------------------------------------------------

@pytest.mark.parametrize(
    ("review_type", "decision"),
    [
        ("initial_plan_review", {"action": "approve"}),
        ("initial_research_review", {"action": "approve"}),
        ("adaptive_research_review", {"action": "reject"}),
        ("final_report_review", {"action": "research_more"}),
    ],
)
def test_validate_review_decision_accepts_valid_actions(
    review_type,
    decision,
):
    result = research_service.validate_review_decision(
        {"type": review_type},
        decision,
    )

    assert result["action"] == decision["action"]


@pytest.mark.parametrize(
    "review_type",
    [
        "initial_plan_review",
        "initial_research_review",
        "adaptive_research_review",
        "final_report_review",
    ],
)
def test_validate_review_decision_rejects_unknown_actions(review_type):
    with pytest.raises(ValueError):
        research_service.validate_review_decision(
            {"type": review_type},
            {"action": "unknown"},
        )


def test_validate_review_decision_normalizes_revision():
    result = research_service.validate_review_decision(
        {"type": "initial_plan_review"},
        {
            "action": "revise",
            "question_number": 1,
            "question": "  Revised research question  ",
            "search_queries": [
                "  first query ",
                "",
                "second query",
            ],
        },
    )

    assert result["action"] == "revise"
    assert result["question_number"] == 1
    assert result["question"] == "Revised research question"
    assert result["search_queries"] == [
        "first query",
        "second query",
    ]


@pytest.mark.parametrize(
    "decision",
    [
        {"action": "revise"},
        {
            "action": "revise",
            "question_number": True,
            "question": "Valid",
        },
        {
            "action": "revise",
            "question_number": 0,
            "question": "Valid",
        },
        {
            "action": "revise",
            "question_number": 1,
            "question": "  ",
        },
        {
            "action": "revise",
            "question_number": 1,
            "question": "Valid",
            "search_queries": "not a list",
        },
    ],
)
def test_validate_review_decision_rejects_invalid_revisions(decision):
    with pytest.raises(ValueError):
        research_service.validate_review_decision(
            {"type": "initial_plan_review"},
            decision,
        )


def test_validate_review_decision_rejects_unknown_review_type():
    with pytest.raises(ValueError):
        research_service.validate_review_decision(
            {"type": "unknown_review"},
            {"action": "approve"},
        )
