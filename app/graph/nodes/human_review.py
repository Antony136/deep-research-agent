"""
Human-in-the-loop checkpoints for the Deep Research Agent.

Approvals happen at meaningful workflow boundaries rather
than before every individual tool call.
"""

from langgraph.types import interrupt

from app.graph.state import ResearchState


def initial_plan_review_node(
    state: ResearchState,
) -> dict:
    """
    Let the user approve, revise, or cancel the initial plan.
    """

    print("\n[Node] initial_plan_review")

    payload = {
        "kind": "initial_plan",
        "question": state["question"],
        "research_questions": [
            question.model_dump(mode="json")
            for question in state["research_questions"]
        ],
        "message": (
            "Review the initial research plan before "
            "any web research begins."
        ),
    }

    decision = interrupt(payload)
    action = decision.get("action", "cancel")

    if action == "cancel":
        print("  Initial research plan cancelled.")

        return {
            **state,
            "research_cancelled": True,
            "plan_approved": False,
        }

    if action == "revise":
        revisions = decision.get("revisions", {})
        updated_questions = list(state["research_questions"])

        for question_number, revision in revisions.items():
            try:
                index = int(question_number) - 1
            except (TypeError, ValueError):
                continue

            if not 0 <= index < len(updated_questions):
                continue

            updated_question = revision.get("question", "").strip()
            updated_queries = revision.get("search_queries", [])

            if not updated_question:
                continue

            original = updated_questions[index]

            updated_questions[index] = original.model_copy(
                update={
                    "question": updated_question,
                    "search_queries": (
                        updated_queries
                        if updated_queries
                        else original.search_queries
                    ),
                }
            )

        print("  Research plan revised. Requesting approval again.")

        return {
            **state,
            "research_questions": updated_questions,
            "plan_approved": False,
            "research_cancelled": False,
        }

    if action == "approve":
        print("  Initial research plan approved.")

        return {
            **state,
            "plan_approved": True,
            "research_cancelled": False,
        }

    raise ValueError(
        f"Unsupported initial-plan decision: {action}"
    )


def initial_research_approval_node(
    state: ResearchState,
) -> dict:
    """
    Ask once before starting the initial web-research phase.

    Once approved, all searches and selected-page reads for
    the approved plan can run without further interruptions.
    """

    print("\n[Node] initial_research_approval")

    if not state.get("require_initial_research_approval", True):
        print("  Initial research approval is disabled.")

        return {
            **state,
            "initial_research_approved": True,
        }

    decision = interrupt(
        {
            "kind": "initial_research",
            "question": state["question"],
            "research_questions": [
                question.model_dump(mode="json")
                for question in state["research_questions"]
            ],
            "message": (
                "Approve web research for the entire initial "
                "plan? Individual searches will not require "
                "additional approval."
            ),
        }
    )

    action = decision.get("action", "stop")

    if action == "approve":
        print("  Initial web research approved.")

        return {
            **state,
            "initial_research_approved": True,
        }

    if action == "stop":
        print("  Research stopped by user.")

        return {
            **state,
            "initial_research_approved": False,
            "research_cancelled": True,
        }

    raise ValueError(
        f"Unsupported initial-research decision: {action}"
    )


def adaptive_research_review_node(
    state: ResearchState,
) -> dict:
    """
    Review newly proposed follow-up questions before research
    continues.

    The adaptive planner sets current_question_index to the
    first newly added question, so questions before that index
    are preserved if the proposal is rejected.
    """

    print("\n[Node] adaptive_research_review")

    start_index = state["current_question_index"]
    questions = state["research_questions"]
    proposed_questions = questions[start_index:]

    # No proposed questions means no approval is necessary.
    if not proposed_questions:
        print("  No follow-up questions require approval.")
        return state

    payload = {
        "kind": "adaptive_research",
        "question": state["question"],
        "research_gaps": state.get("research_gaps", []),
        "proposed_questions": [
            {
                "question_number": start_index + offset + 1,
                **question.model_dump(mode="json"),
            }
            for offset, question in enumerate(proposed_questions)
        ],
        "message": (
            "The initial research has been evaluated. "
            "Review these proposed follow-up questions before "
            "the agent performs additional research."
        ),
    }

    decision = interrupt(payload)
    action = decision.get("action", "reject")

    if action == "approve":
        print(
            f"  Approved {len(proposed_questions)} "
            "follow-up question(s)."
        )

        return state

    if action == "reject":
        print("  Follow-up proposal rejected.")

        return {
            **state,
            "research_questions": questions[:start_index],
            "research_complete": False,
        }

    raise ValueError(
        f"Unsupported adaptive-research decision: {action}"
    )
