"""
Human review node for the final research report.

Allows the reviewer to approve the report or request another
research pass before the session is marked as completed.
"""

from typing import Any

from langgraph.types import interrupt

from app.graph.state import (
    FinalReportReviewDecision,
    ResearchState,
)


def final_report_review_node(
    state: ResearchState,
) -> dict[str, Any]:
    """
    Request approval of the generated research report.

    Actions:
    - approve: accept the report and finish the workflow.
    - research_more: request additional research.

    This node does not perform research or modify the report.
    """

    print("\n[Node] final_report_review")

    report = state.get("report")

    if not report:
        return {
            "final_report_approved": False,
            "final_report_review_decision": {
                "action": "research_more",
                "reason": "Final report is unavailable for review.",
            },
            "research_complete": False,
            "research_sufficient": False,
            "research_decision_reason": (
                "Final report is unavailable for review."
            ),
        }

    # --------------------------------------------------------
    # Optional review
    # --------------------------------------------------------

    if not state.get(
        "require_final_report_approval",
        False,
    ):
        print("  Final report approval is disabled.")

        return {
            "final_report_approved": True,
            "final_report_review_decision": {
                "action": "approve",
                "reason": (
                    "Final report approval is disabled."
                ),
            },
            "research_complete": True,
        }

    # --------------------------------------------------------
    # Human review checkpoint
    # --------------------------------------------------------

    decision = interrupt(
        {
            "type": "final_report_review",
            "title": "Review the final research report",
            "message": (
                "The research report is ready. Approve it to "
                "complete the session, or request additional "
                "research before generating another report."
            ),
            "available_actions": [
                "approve",
                "research_more",
            ],
            "research_question": state["question"],
            "research_round": state.get("research_round", 0),
            "verified_evidence_count": len(
                state.get("evidence", [])
            ),
            "report": report,
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

    # --------------------------------------------------------
    # Approve report
    # --------------------------------------------------------

    if action == "approve":
        print("  Final report approved.")

        review_decision: FinalReportReviewDecision = {
            "action": "approve",
            "reason": reason or "Final report approved.",
        }

        return {
            "final_report_approved": True,
            "final_report_review_decision": review_decision,
            "research_complete": True,
        }

    # --------------------------------------------------------
    # Request additional research
    # --------------------------------------------------------

    if action == "research_more":
        print("  Additional research requested.")

        review_decision = {
            "action": "research_more",
            "reason": (
                reason
                or "Reviewer requested additional research."
            ),
        }

        return {
            "final_report_approved": False,
            "final_report_review_decision": review_decision,
            "research_complete": False,
            "research_sufficient": False,
            "research_decision_reason": review_decision["reason"],
        }

    # --------------------------------------------------------
    # Invalid decision
    # --------------------------------------------------------

    raise ValueError(
        "Invalid final report review decision. "
        "Expected 'approve' or 'research_more'."
    )
