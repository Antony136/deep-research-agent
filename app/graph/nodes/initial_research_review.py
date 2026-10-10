"""
Human review node for authorizing initial web research.

Requests approval once before the agent begins searching the
web and reading source pages for the approved research plan.

No approval is requested for individual searches or page reads.
"""

from typing import Any

from langgraph.types import interrupt

from app.graph.state import ResearchState


def initial_research_review_node(
    state: ResearchState,
) -> dict[str, Any]:
    """
    Authorize the initial research phase.

    The decision is persisted through LangGraph checkpointing.
    Once approved, research_authorized remains true for the
    remainder of the workflow, including adaptive research.
    """

    print(
        "\n[Node] initial_research_review"
    )

    # Never begin web research if the initial plan was rejected.
    if state.get("research_complete", False):
        print(
            "  Research was already marked complete. "
            "Skipping web research authorization."
        )

        return {
            "initial_research_approved": False,
            "research_authorized": False,
        }

    if not state.get("initial_plan_approved", False):
        print(
            "  Initial plan has not been approved. "
            "Web research is not authorized."
        )

        return {
            "initial_research_approved": False,
            "research_authorized": False,
        }

    # Avoid asking for approval again after the workflow resumes
    # or proceeds through additional research rounds.
    if state.get("research_authorized", False):
        print(
            "  Web research was already authorized."
        )

        return {
            "initial_research_approved": True,
            "research_authorized": True,
        }

    # Allow an application to disable this checkpoint.
    if not state.get(
        "require_initial_research_approval",
        True,
    ):
        print(
            "  Research approval is disabled. "
            "Authorizing web research."
        )

        return {
            "initial_research_approved": True,
            "research_authorized": True,
        }

    research_plan = [
        {
            "question_number": index,
            "question": question.question,
            "search_queries": question.search_queries,
        }
        for index, question in enumerate(
            state.get("research_questions", []),
            start=1,
        )
    ]

    if not research_plan:
        print(
            "  No approved research questions are available."
        )

        return {
            "initial_research_approved": False,
            "research_authorized": False,
            "research_complete": True,
            "research_decision_reason": (
                "Cannot start web research without an approved plan."
            ),
        }

    decision = interrupt(
        {
            "type": "initial_research_review",
            "title": "Authorize web research",
            "message": (
                "The research plan is approved. Allow the agent "
                "to perform web searches and read relevant pages "
                "for the plan? This approval covers the research "
                "phase; individual searches and page reads will "
                "not trigger additional approval prompts."
            ),
            "available_actions": [
                "approve",
                "reject",
            ],
            "research_question": state["question"],
            "research_plan": research_plan,
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

    if action == "approve":
        print(
            "  Web research authorized for this session."
        )

        return {
            "initial_research_approved": True,
            "research_authorized": True,
        }

    if action == "reject":
        print(
            "  Web research was not authorized."
        )

        return {
            "initial_research_approved": False,
            "research_authorized": False,
            "research_complete": True,
            "research_decision_reason": (
                reason or "Web research was rejected by the reviewer."
            ),
        }

    raise ValueError(
        "Invalid research authorization decision. "
        "Expected approve or reject."
    )