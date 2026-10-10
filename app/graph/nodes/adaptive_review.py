"""
Human review node for adaptive research planning.

Pauses the workflow before proposed follow-up questions are
added to the active research plan.

Supported decisions:
- approve: accept the proposed follow-up questions.
- reject: decline the proposals and continue to final synthesis.
"""

from typing import Any

from langgraph.types import interrupt

from app.graph.state import (
    AdaptiveReviewDecision,
    ResearchState,
)


def adaptive_review_node(
    state: ResearchState,
) -> dict[str, Any]:
    """
    Request approval for proposed adaptive research questions.

    The interrupt payload is presented to the application.
    The workflow resumes when the application supplies a decision.

    Approved proposals are appended to the active plan.
    Rejected proposals are discarded.
    """

    proposed_questions = state.get(
        "proposed_research_questions",
        [],
    )

    current_questions = state.get(
        "research_questions",
        [],
    )

    max_total_questions = state.get(
        "max_total_research_questions",
        10,
    )

    if not proposed_questions:
        print(
            "\n[Node] adaptive_review: "
            "No proposed follow-up questions to review."
        )

        return {
            "proposed_research_questions": [],
            "adaptive_review_decision": {
                "action": "reject",
                "reason": "No valid follow-up questions were proposed.",
            },
        }

    available_capacity = max(
        0,
        max_total_questions - len(current_questions),
    )

    proposed_questions = proposed_questions[
        :available_capacity
    ]

    if not proposed_questions:
        print(
            "\n[Node] adaptive_review: "
            "The research-question limit has been reached."
        )

        return {
            "proposed_research_questions": [],
            "adaptive_review_decision": {
                "action": "reject",
                "reason": "No remaining capacity for follow-up questions.",
            },
        }

    proposed_question_data = [
        {
            "question": question.question,
            "search_queries": question.search_queries,
            "parent_question_number": (
                question.parent_question_number
            ),
        }
        for question in proposed_questions
    ]

    print(
        "\n[Node] adaptive_review: "
        "Waiting for approval of follow-up questions."
    )

    decision = interrupt(
        {
            "type": "adaptive_research_review",
            "title": "Review proposed follow-up research",
            "message": (
                "The agent found unresolved research gaps. "
                "Approve the proposed follow-up questions "
                "to continue research, or reject them to "
                "proceed to final synthesis."
            ),
            "available_actions": [
                "approve",
                "reject",
            ],
            "current_question_count": len(current_questions),
            "maximum_question_count": max_total_questions,
            "proposed_questions": proposed_question_data,
        }
    )

    if not isinstance(decision, dict):
        decision = {
            "action": str(decision).strip().lower(),
        }

    action = str(
        decision.get("action", "")
    ).strip().lower()

    reason = str(
        decision.get("reason", "")
    ).strip()

    if action != "approve":
        print(
            "  Follow-up questions rejected. "
            "Proceeding to final synthesis."
        )

        review_decision: AdaptiveReviewDecision = {
            "action": "reject",
            "reason": reason or "Follow-up questions rejected by the reviewer.",
        }

        return {
            "proposed_research_questions": [],
            "adaptive_review_decision": review_decision,
        }

    # Re-check the limit when accepting the proposals.
    # This prevents exceeding the configured question budget.
    accepted_questions = proposed_questions[
        :max_total_questions - len(current_questions)
    ]

    if not accepted_questions:
        return {
            "proposed_research_questions": [],
            "adaptive_review_decision": {
                "action": "reject",
                "reason": "No remaining capacity for approved questions.",
            },
        }

    updated_questions = (
        current_questions + accepted_questions
    )

    print(
        f"  Approved {len(accepted_questions)} "
        "follow-up question(s)."
    )

    for index, question in enumerate(
        accepted_questions,
        start=len(current_questions) + 1,
    ):
        print(
            f"    Q{index}: {question.question}"
        )

    review_decision = {
        "action": "approve",
        "reason": reason or "Follow-up questions approved.",
    }

    return {
        "research_questions": updated_questions,
        "current_question_index": len(current_questions),
        "proposed_research_questions": [],
        "adaptive_review_decision": review_decision,
        "research_complete": False,
    }
