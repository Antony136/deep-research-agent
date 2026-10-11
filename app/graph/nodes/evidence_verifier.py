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

    # Read state safely to avoid KeyError when optional
    # fields have not been initialized.
    pending_evidence = state.get("pending_evidence", [])
    current_sources = state.get("current_sources", [])
    existing_evidence = state.get("evidence", [])
    research_questions = state.get("research_questions", [])
    current_question_index = state.get(
        "current_question_index",
        0,
    )

    if not pending_evidence:
        print("  No pending evidence to verify.")

        research_complete = (
            current_question_index >= len(research_questions)
        )

        return {
            **state,
            "pending_evidence": [],
            "evidence": existing_evidence,
            "research_complete": research_complete,
        }

    print(
        f"  Pending evidence items: "
        f"{len(pending_evidence)}"
    )

    current_question_number = state.get(
        "active_research_question_number"
    )

    if current_question_number is not None:
        print(
            f"  Verifying evidence for research "
            f"question {current_question_number}."
        )

    valid_evidence, rejected_evidence = (
        validate_evidence_batch(
            evidence_items=pending_evidence,
            sources=current_sources,
        )
    )

    print(f"  Valid evidence: {len(valid_evidence)}")
    print(f"  Rejected evidence: {len(rejected_evidence)}")

    for evidence, reason in rejected_evidence:
        print("\n  REJECTED EVIDENCE")
        print(
            f"    Research question: "
            f"{evidence.research_question_number}"
        )
        print(f"    Claim: {evidence.claim}")
        print(f"    Reason: {reason}")

    combined_evidence = existing_evidence + valid_evidence

    research_complete = (
        current_question_index >= len(research_questions)
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
