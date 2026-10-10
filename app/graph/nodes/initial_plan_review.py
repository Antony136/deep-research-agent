"""
Human review node for the initial research plan.

Pauses the LangGraph workflow before web research begins.

Supported decisions:
- approve: accept the proposed research plan.
- revise: replace one research question and optionally update
  its search queries.
- reject: stop the research workflow before web research.
"""

from typing import Any

from langgraph.types import interrupt

from app.graph.state import ResearchState
from app.schemas.research import ResearchQuestion


def initial_plan_review_node(
    state: ResearchState,
) -> dict[str, Any]:
    """
    Request approval or revision of the initial research plan.

    The node uses LangGraph interrupt/resume so the review
    decision can be persisted with the graph checkpoint.
    """

    research_questions = state.get(
        "research_questions",
        [],
    )

    if not research_questions:
        print(
            "\n[Node] initial_plan_review: "
            "No research questions were generated."
        )

        return {
            "initial_plan_approved": False,
            "research_complete": True,
        }

    if not state.get(
        "require_initial_plan_approval",
        True,
    ):
        print(
            "\n[Node] initial_plan_review: "
            "Approval disabled; accepting the plan."
        )

        return {
            "initial_plan_approved": True,
        }

    proposed_plan = [
        {
            "question_number": index,
            "question": question.question,
            "search_queries": question.search_queries,
            "parent_question_number": (
                question.parent_question_number
            ),
        }
        for index, question in enumerate(
            research_questions,
            start=1,
        )
    ]

    print(
        "\n[Node] initial_plan_review: "
        "Waiting for initial plan approval."
    )

    decision = interrupt(
        {
            "type": "initial_plan_review",
            "title": "Review the initial research plan",
            "message": (
                "Review the proposed research questions before "
                "the agent performs any web research."
            ),
            "available_actions": [
                "approve",
                "revise",
                "reject",
            ],
            "research_question": state["question"],
            "proposed_questions": proposed_plan,
        }
    )

    if not isinstance(decision, dict):
        decision = {
            "action": str(decision).strip().lower(),
        }

    action = str(
        decision.get("action", "")
    ).strip().lower()

    if action == "approve":
        print(
            "  Initial research plan approved."
        )

        return {
            "initial_plan_approved": True,
        }

    if action == "revise":
        question_number = decision.get(
            "question_number"
        )

        revised_text = str(
            decision.get("question", "")
        ).strip()

        revised_queries = decision.get(
            "search_queries"
        )

        if not isinstance(question_number, int):
            raise ValueError(
                "Plan revision requires an integer question_number."
            )

        if not 1 <= question_number <= len(research_questions):
            raise ValueError(
                "The question_number is outside the current plan."
            )

        if not revised_text:
            raise ValueError(
                "A revised research question cannot be empty."
            )

        if revised_queries is None:
            revised_queries = research_questions[
                question_number - 1
            ].search_queries

        if (
            not isinstance(revised_queries, list)
            or not all(
                isinstance(query, str) and query.strip()
                for query in revised_queries
            )
        ):
            raise ValueError(
                "search_queries must be a list of non-empty strings."
            )

        revised_queries = [
            query.strip()
            for query in revised_queries
        ]

        original_question = research_questions[
            question_number - 1
        ]

        updated_question = ResearchQuestion(
            question=revised_text,
            search_queries=revised_queries,
            parent_question_number=(
                original_question.parent_question_number
            ),
        )

        updated_questions = list(
            research_questions
        )

        updated_questions[
            question_number - 1
        ] = updated_question

        print(
            f"  Revised research question Q{question_number}."
        )

        # Return to the review node so the revised plan can
        # be approved before any web research starts.
        return {
            "research_questions": updated_questions,
            "initial_plan_approved": False,
        }

    if action == "reject":
        reason = str(
            decision.get(
                "reason",
                "Initial research plan rejected.",
            )
        ).strip()

        print(
            "  Initial research plan rejected."
        )

        return {
            "initial_plan_approved": False,
            "research_complete": True,
            "research_decision_reason": reason,
        }

    raise ValueError(
        "Invalid initial plan review decision. "
        "Expected approve, revise, or reject."
    )
