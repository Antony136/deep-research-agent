import pytest

from app.graph.nodes.initial_plan_review import (
    initial_plan_review_node,
)
from app.graph.nodes.initial_research_review import (
    initial_research_review_node,
)
from app.graph.nodes.adaptive_review import (
    adaptive_review_node,
)
from app.graph.nodes.final_report_review import (
    final_report_review_node,
)
from app.schemas.research import ResearchQuestion


def question(text, queries=None, parent=None):
    return ResearchQuestion(
        question=text,
        search_queries=queries or [f"search {text}"],
        parent_question_number=parent,
    )


# =========================================================
# INITIAL PLAN REVIEW
# =========================================================

def test_initial_plan_review_skips_empty_plan():
    result = initial_plan_review_node({
        "research_questions": [],
    })

    assert result["initial_plan_approved"] is False
    assert result["research_complete"] is True


def test_initial_plan_review_auto_approves_when_disabled():
    result = initial_plan_review_node({
        "research_questions": [question("Test question")],
        "require_initial_plan_approval": False,
    })

    assert result == {"initial_plan_approved": True}


def test_initial_plan_review_interrupt_contains_plan(monkeypatch):
    from app.graph.nodes import initial_plan_review

    captured = {}

    def fake_interrupt(payload):
        captured.update(payload)
        return {"action": "approve"}

    monkeypatch.setattr(initial_plan_review, "interrupt", fake_interrupt)

    result = initial_plan_review_node({
        "question": "Main question",
        "research_questions": [
            question("First question", ["query one"], parent=2),
        ],
        "require_initial_plan_approval": True,
    })

    assert captured["type"] == "initial_plan_review"
    assert captured["proposed_questions"][0]["question"] == "First question"
    assert captured["proposed_questions"][0]["question_number"] == 1
    assert result["initial_plan_approved"] is True


def test_initial_plan_review_revises_question_and_preserves_parent(monkeypatch):
    from app.graph.nodes import initial_plan_review

    monkeypatch.setattr(
        initial_plan_review,
        "interrupt",
        lambda _payload: {
            "action": "revise",
            "question_number": 1,
            "question": "Revised question",
            "search_queries": ["  updated query  ", "second query"],
        },
    )

    result = initial_plan_review_node({
        "research_questions": [
            question("Original question", ["old query"], parent=3),
        ],
    })

    revised = result["research_questions"][0]

    assert revised.question == "Revised question"
    assert revised.search_queries == ["updated query", "second query"]
    assert revised.parent_question_number == 3
    assert result["initial_plan_approved"] is False


@pytest.mark.parametrize(
    ("decision", "message"),
    [
        (
            {
                "action": "revise",
                "question_number": 0,
                "question": "Updated",
            },
            "outside the current plan",
        ),
        (
            {
                "action": "revise",
                "question_number": 1,
                "question": " ",
            },
            "cannot be empty",
        ),
        (
            {
                "action": "revise",
                "question_number": 1,
                "question": "Updated",
                "search_queries": "not a list",
            },
            "search_queries must be a list",
        ),
    ],
)
def test_initial_plan_review_rejects_invalid_revisions(
    monkeypatch,
    decision,
    message,
):
    from app.graph.nodes import initial_plan_review

    monkeypatch.setattr(
        initial_plan_review,
        "interrupt",
        lambda _payload: decision,
    )

    with pytest.raises(ValueError, match=message):
        initial_plan_review_node({
            "research_questions": [question("Original")],
        })


def test_initial_plan_review_rejects_plan_with_reason(monkeypatch):
    from app.graph.nodes import initial_plan_review

    monkeypatch.setattr(
        initial_plan_review,
        "interrupt",
        lambda _payload: {
            "action": "reject",
            "reason": "  Insufficient scope  ",
        },
    )

    result = initial_plan_review_node({
        "research_questions": [question("Test question")],
    })

    assert result["initial_plan_approved"] is False
    assert result["research_complete"] is True
    assert result["research_decision_reason"] == "Insufficient scope"


def test_initial_plan_review_rejects_unknown_action(monkeypatch):
    from app.graph.nodes import initial_plan_review

    monkeypatch.setattr(
        initial_plan_review,
        "interrupt",
        lambda _payload: {"action": "unknown"},
    )

    with pytest.raises(ValueError, match="Invalid initial plan review"):
        initial_plan_review_node({
            "research_questions": [question("Test question")],
        })


# =========================================================
# INITIAL RESEARCH AUTHORIZATION
# =========================================================

def test_initial_research_review_stops_if_research_complete():
    result = initial_research_review_node({
        "research_complete": True,
        "initial_plan_approved": True,
    })

    assert result["initial_research_approved"] is False
    assert result["research_authorized"] is False


def test_initial_research_review_requires_approved_plan():
    result = initial_research_review_node({
        "initial_plan_approved": False,
    })

    assert result["initial_research_approved"] is False
    assert result["research_authorized"] is False


def test_initial_research_review_reuses_existing_authorization():
    result = initial_research_review_node({
        "initial_plan_approved": True,
        "research_authorized": True,
    })

    assert result["initial_research_approved"] is True
    assert result["research_authorized"] is True


def test_initial_research_review_auto_authorizes_when_disabled():
    result = initial_research_review_node({
        "initial_plan_approved": True,
        "research_questions": [question("Test question")],
        "require_initial_research_approval": False,
    })

    assert result["initial_research_approved"] is True
    assert result["research_authorized"] is True


def test_initial_research_review_stops_when_plan_is_empty():
    result = initial_research_review_node({
        "initial_plan_approved": True,
        "research_questions": [],
    })

    assert result["research_complete"] is True
    assert result["research_authorized"] is False


def test_initial_research_review_approves_authorization(monkeypatch):
    from app.graph.nodes import initial_research_review

    captured = {}

    def fake_interrupt(payload):
        captured.update(payload)
        return {"action": "approve"}

    monkeypatch.setattr(initial_research_review, "interrupt", fake_interrupt)

    result = initial_research_review_node({
        "question": "Main question",
        "initial_plan_approved": True,
        "research_questions": [question("Research question")],
    })

    assert captured["type"] == "initial_research_review"
    assert len(captured["research_plan"]) == 1
    assert result["research_authorized"] is True


def test_initial_research_review_rejects_authorization(monkeypatch):
    from app.graph.nodes import initial_research_review

    monkeypatch.setattr(
        initial_research_review,
        "interrupt",
        lambda _payload: {
            "action": "reject",
            "reason": "Not authorized",
        },
    )

    result = initial_research_review_node({
        "initial_plan_approved": True,
        "research_questions": [question("Research question")],
    })

    assert result["research_complete"] is True
    assert result["research_authorized"] is False
    assert result["research_decision_reason"] == "Not authorized"


def test_initial_research_review_rejects_unknown_action(monkeypatch):
    from app.graph.nodes import initial_research_review

    monkeypatch.setattr(
        initial_research_review,
        "interrupt",
        lambda _payload: {"action": "revise"},
    )

    with pytest.raises(ValueError, match="Invalid research authorization"):
        initial_research_review_node({
            "initial_plan_approved": True,
            "research_questions": [question("Research question")],
        })


# =========================================================
# ADAPTIVE RESEARCH REVIEW
# =========================================================

def test_adaptive_review_skips_empty_proposals():
    result = adaptive_review_node({
        "proposed_research_questions": [],
    })

    assert result["proposed_research_questions"] == []
    assert result["adaptive_review_decision"]["action"] == "reject"


def test_adaptive_review_respects_question_limit():
    result = adaptive_review_node({
        "research_questions": [
            question("Existing one"),
            question("Existing two"),
        ],
        "proposed_research_questions": [
            question("Proposed question"),
        ],
        "max_total_research_questions": 2,
    })

    assert result["proposed_research_questions"] == []
    assert result["adaptive_review_decision"]["action"] == "reject"


def test_adaptive_review_approves_proposals_within_budget(monkeypatch):
    from app.graph.nodes import adaptive_review

    monkeypatch.setattr(
        adaptive_review,
        "interrupt",
        lambda _payload: {
            "action": "approve",
            "reason": "Useful follow-up",
        },
    )

    existing = [question("Existing question")]
    proposed = [
        question("Follow-up one", parent=1),
        question("Follow-up two", parent=1),
    ]

    result = adaptive_review_node({
        "research_questions": existing,
        "proposed_research_questions": proposed,
        "current_question_index": 1,
        "max_total_research_questions": 3,
    })

    assert [q.question for q in result["research_questions"]] == [
        "Existing question",
        "Follow-up one",
        "Follow-up two",
    ]
    assert result["current_question_index"] == 1
    assert result["proposed_research_questions"] == []
    assert result["adaptive_review_decision"]["action"] == "approve"
    assert result["research_complete"] is False


def test_adaptive_review_rejects_proposals(monkeypatch):
    from app.graph.nodes import adaptive_review

    monkeypatch.setattr(
        adaptive_review,
        "interrupt",
        lambda _payload: {
            "action": "reject",
            "reason": "Not relevant",
        },
    )

    result = adaptive_review_node({
        "research_questions": [question("Existing question")],
        "proposed_research_questions": [question("Follow-up")],
        "max_total_research_questions": 5,
    })

    assert result["proposed_research_questions"] == []
    assert result["adaptive_review_decision"]["action"] == "reject"
    assert result["adaptive_review_decision"]["reason"] == "Not relevant"


def test_adaptive_review_truncates_proposals_to_available_capacity(
    monkeypatch,
):
    from app.graph.nodes import adaptive_review

    monkeypatch.setattr(
        adaptive_review,
        "interrupt",
        lambda _payload: {"action": "approve"},
    )

    result = adaptive_review_node({
        "research_questions": [question("Existing")],
        "proposed_research_questions": [
            question("Follow-up one"),
            question("Follow-up two"),
            question("Follow-up three"),
        ],
        "max_total_research_questions": 2,
    })

    assert len(result["research_questions"]) == 2
    assert result["research_questions"][-1].question == "Follow-up one"


# =========================================================
# FINAL REPORT REVIEW
# =========================================================

def test_final_report_review_requests_more_research_without_report():
    result = final_report_review_node({
        "report": None,
    })

    assert result["final_report_approved"] is False
    assert result["final_report_review_decision"]["action"] == "research_more"
    assert result["research_complete"] is False
    assert result["research_sufficient"] is False


def test_final_report_review_auto_approves_when_disabled():
    result = final_report_review_node({
        "report": {"title": "Report"},
        "require_final_report_approval": False,
    })

    assert result["final_report_approved"] is True
    assert result["final_report_review_decision"]["action"] == "approve"
    assert result["research_complete"] is True


def test_final_report_review_approves_report(monkeypatch):
    from app.graph.nodes import final_report_review

    captured = {}

    def fake_interrupt(payload):
        captured.update(payload)
        return {"action": "approve", "reason": "Verified"}

    monkeypatch.setattr(final_report_review, "interrupt", fake_interrupt)

    result = final_report_review_node({
        "question": "Main question",
        "report": {"title": "Research report"},
        "research_round": 2,
        "evidence": [{"claim": "Verified finding"}],
        "require_final_report_approval": True,
    })

    assert captured["type"] == "final_report_review"
    assert captured["verified_evidence_count"] == 1
    assert result["final_report_approved"] is True
    assert result["research_complete"] is True
    assert result["final_report_review_decision"]["reason"] == "Verified"


def test_final_report_review_requests_more_research(monkeypatch):
    from app.graph.nodes import final_report_review

    monkeypatch.setattr(
        final_report_review,
        "interrupt",
        lambda _payload: {
            "action": "research_more",
            "reason": "Missing comparison",
        },
    )

    result = final_report_review_node({
        "report": {"title": "Research report"},
        "require_final_report_approval": True,
    })

    assert result["final_report_approved"] is False
    assert result["research_complete"] is False
    assert result["research_sufficient"] is False
    assert result["research_decision_reason"] == "Missing comparison"


def test_final_report_review_rejects_unknown_action(monkeypatch):
    from app.graph.nodes import final_report_review

    monkeypatch.setattr(
        final_report_review,
        "interrupt",
        lambda _payload: {"action": "reject"},
    )

    with pytest.raises(ValueError, match="Invalid final report review"):
        final_report_review_node({
            "report": {"title": "Research report"},
            "require_final_report_approval": True,
        })
