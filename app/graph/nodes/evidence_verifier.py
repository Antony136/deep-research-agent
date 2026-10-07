"""
Evidence verification node for the Deep Research Agent.

This node validates LLM-extracted evidence against the actual
sources collected for the current research question.

Only evidence that passes deterministic validation is promoted
to the trusted evidence collection.
"""

from app.graph.state import ResearchState
from app.tools.evidence_validator import validate_evidence_batch


def evidence_verifier_node(
    state: ResearchState,
) -> ResearchState:
    """
    Verify pending evidence and promote valid evidence.
    """

    print("\n[Node] evidence_verifier")

    pending_evidence = state["pending_evidence"]
    current_sources = state["current_sources"]

    if not pending_evidence:
        print("  No pending evidence to verify.")

        research_complete = (
            state["current_question_index"]
            >= len(state["research_questions"])
        )

        return {
            **state,
            "pending_evidence": [],
            "research_complete": research_complete,
        }

    print(
        f"  Pending evidence items: "
        f"{len(pending_evidence)}"
    )

    valid_evidence, rejected_evidence = (
        validate_evidence_batch(
            evidence_items=pending_evidence,
            sources=current_sources,
        )
    )

    print(
        f"  Valid evidence: "
        f"{len(valid_evidence)}"
    )

    print(
        f"  Rejected evidence: "
        f"{len(rejected_evidence)}"
    )

    for evidence, reason in rejected_evidence:
        print()
        print("  REJECTED EVIDENCE")
        print(f"    Claim: {evidence.claim}")
        print(f"    Reason: {reason}")

    existing_evidence = state["evidence"]

    combined_evidence = (
        existing_evidence
        + valid_evidence
    )

    research_complete = (
        state["current_question_index"]
        >= len(state["research_questions"])
    )

    if research_complete:
        print(
            "\n  All planned research questions "
            "have been processed."
        )

    print(
        f"  Trusted evidence total: "
        f"{len(combined_evidence)}"
    )

    return {
        **state,
        "pending_evidence": [],
        "evidence": combined_evidence,
        "research_complete": research_complete,
    }
