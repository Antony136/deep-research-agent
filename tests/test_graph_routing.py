import pytest

from app.graph.graph import (
    auto_accept_adaptive_questions,
    route_after_final_report_review,
    route_after_initial_plan_review,
    route_after_initial_research_review,
    should_continue_after_adaptive_review,
    should_continue_research,
    should_review_adaptive_questions,
)


# ============================================================
# INITIAL PLAN REVIEW
# ============================================================


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {"research_complete": True},
            "end",
        ),
        (
            {
                "research_complete": False,
                "initial_plan_approved": True,
            },
            "research_authorization",
        ),
        (
            {
                "research_complete": False,
                "initial_plan_approved": False,
            },
            "plan_review",
        ),
    ],
)
def test_route_after_initial_plan_review(state, expected):
    assert route_after_initial_plan_review(state) == expected


# ============================================================
# INITIAL RESEARCH AUTHORIZATION
# ============================================================


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {"research_complete": True},
            "end",
        ),
        (
            {
                "initial_research_approved": True,
                "research_authorized": True,
            },
            "research",
        ),
        (
            {
                "initial_research_approved": True,
                "research_authorized": False,
            },
            "end",
        ),
        (
            {
                "initial_research_approved": False,
                "research_authorized": True,
            },
            "end",
        ),
    ],
)
def test_route_after_initial_research_review(state, expected):
    assert route_after_initial_research_review(state) == expected


# ============================================================
# RESEARCH SUFFICIENCY
# ============================================================


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {
                "current_question_index": 0,
                "research_questions": [{"question": "Q1"}],
            },
            "research",
        ),
        (
            {
                "current_question_index": 1,
                "research_questions": [{"question": "Q1"}],
                "research_sufficient": True,
            },
            "synthesis",
        ),
        (
            {
                "current_question_index": 1,
                "research_questions": [{"question": "Q1"}],
                "research_sufficient": False,
                "max_total_research_questions": 1,
            },
            "synthesis",
        ),
        (
            {
                "current_question_index": 1,
                "research_questions": [{"question": "Q1"}],
                "research_sufficient": False,
                "max_total_research_questions": 5,
                "research_round": 1,
                "max_research_rounds": 3,
            },
            "adaptive_planner",
        ),
        (
            {
                "current_question_index": 1,
                "research_questions": [{"question": "Q1"}],
                "research_sufficient": False,
                "max_total_research_questions": 5,
                "research_round": 3,
                "max_research_rounds": 3,
            },
            "synthesis",
        ),
    ],
)
def test_should_continue_research(state, expected):
    assert should_continue_research(state) == expected


# ============================================================
# ADAPTIVE QUESTION REVIEW
# ============================================================


def test_should_review_adaptive_questions_without_proposals():
    state = {
        "proposed_research_questions": [],
        "research_questions": [],
        "max_total_research_questions": 5,
    }

    assert should_review_adaptive_questions(state) == "synthesis"


def test_should_review_adaptive_questions_when_budget_is_full():
    state = {
        "proposed_research_questions": [{"question": "Q2"}],
        "research_questions": [{"question": "Q1"}],
        "max_total_research_questions": 1,
    }

    assert should_review_adaptive_questions(state) == "synthesis"


def test_should_review_adaptive_questions_requires_approval():
    state = {
        "proposed_research_questions": [{"question": "Q2"}],
        "research_questions": [{"question": "Q1"}],
        "max_total_research_questions": 5,
        "require_adaptive_research_approval": True,
    }

    assert should_review_adaptive_questions(state) == "review"


def test_should_review_adaptive_questions_auto_accepts_when_approval_disabled():
    state = {
        "proposed_research_questions": [{"question": "Q2"}],
        "research_questions": [{"question": "Q1"}],
        "max_total_research_questions": 5,
        "require_adaptive_research_approval": False,
    }

    assert should_review_adaptive_questions(state) == "auto_accept"


# ============================================================
# AUTOMATIC ACCEPTANCE
# ============================================================


def test_auto_accept_adaptive_questions_respects_question_budget():
    state = {
        "research_questions": [{"question": "Q1"}],
        "proposed_research_questions": [
            {"question": "Q2"},
            {"question": "Q3"},
            {"question": "Q4"},
        ],
        "max_total_research_questions": 2,
    }

    result = auto_accept_adaptive_questions(state)

    assert result["research_questions"] == [
        {"question": "Q1"},
        {"question": "Q2"},
    ]
    assert result["current_question_index"] == 1
    assert result["proposed_research_questions"] == []
    assert result["adaptive_review_decision"]["action"] == "approve"
    assert result["research_complete"] is False


def test_auto_accept_adaptive_questions_rejects_when_no_capacity():
    state = {
        "research_questions": [{"question": "Q1"}],
        "proposed_research_questions": [{"question": "Q2"}],
        "max_total_research_questions": 1,
    }

    result = auto_accept_adaptive_questions(state)

    assert result["proposed_research_questions"] == []
    assert result["adaptive_review_decision"]["action"] == "reject"


# ============================================================
# CONTINUE AFTER ADAPTIVE REVIEW
# ============================================================


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {"adaptive_review_decision": {"action": "reject"}},
            "synthesis",
        ),
        (
            {
                "adaptive_review_decision": {"action": "approve"},
                "current_question_index": 1,
                "research_questions": [{"question": "Q1"}],
            },
            "synthesis",
        ),
        (
            {
                "adaptive_review_decision": {"action": "approve"},
                "current_question_index": 1,
                "research_questions": [
                    {"question": "Q1"},
                    {"question": "Q2"},
                ],
                "max_total_research_questions": 2,
            },
            "research",
        ),
        (
            {
                "adaptive_review_decision": {"action": "approve"},
                "current_question_index": 0,
                "research_questions": [
                    {"question": "Q1"},
                    {"question": "Q2"},
                ],
                "max_total_research_questions": 1,
            },
            "synthesis",
        ),
    ],
)
def test_should_continue_after_adaptive_review(state, expected):
    assert should_continue_after_adaptive_review(state) == expected


# ============================================================
# FINAL REPORT REVIEW
# ============================================================


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {"final_report_approved": True},
            "end",
        ),
        (
            {
                "final_report_approved": False,
                "final_report_review_decision": None,
            },
            "end",
        ),
        (
            {
                "final_report_approved": False,
                "final_report_review_decision": {"action": "approve"},
            },
            "end",
        ),
        (
            {
                "final_report_approved": False,
                "final_report_review_decision": {
                    "action": "research_more"
                },
                "research_questions": [{"question": "Q1"}],
                "max_total_research_questions": 5,
                "research_round": 1,
                "max_research_rounds": 3,
            },
            "adaptive_planner",
        ),
        (
            {
                "final_report_approved": False,
                "final_report_review_decision": {
                    "action": "research_more"
                },
                "research_questions": [{"question": "Q1"}],
                "max_total_research_questions": 1,
                "research_round": 1,
                "max_research_rounds": 3,
            },
            "end",
        ),
        (
            {
                "final_report_approved": False,
                "final_report_review_decision": {
                    "action": "research_more"
                },
                "research_questions": [{"question": "Q1"}],
                "max_total_research_questions": 5,
                "research_round": 3,
                "max_research_rounds": 3,
            },
            "end",
        ),
    ],
)
def test_route_after_final_report_review(state, expected):
    assert route_after_final_report_review(state) == expected
