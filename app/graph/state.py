"""
LangGraph state for the Deep Research Agent.

The state is the shared data structure that flows through
every node in the research workflow.

Human-in-the-loop fields store approval settings, decisions,
and proposed adaptive questions without mixing them into
the active research plan prematurely.
"""

from typing import Literal, TypedDict

from app.schemas.research import (
    Evidence,
    ResearchQuestion,
    ResearchReport,
    Source,
)


class CoverageAssessmentState(TypedDict):
    """Sufficiency assessment for one research question."""

    research_question_number: int
    covered: bool
    evidence_numbers: list[int]
    reason: str


class AdaptiveReviewDecision(TypedDict):
    """Decision about proposed follow-up research questions."""

    action: Literal["approve", "reject"]
    reason: str


class FinalReportReviewDecision(TypedDict):
    """Decision about the generated research report."""

    action: Literal["approve", "research_more"]
    reason: str


class ResearchState(TypedDict):
    """Complete state of a research session."""

    # ========================================================
    # USER REQUEST
    # ========================================================

    question: str

    # ========================================================
    # HUMAN-IN-THE-LOOP SETTINGS
    # ========================================================

    require_initial_plan_approval: bool
    require_initial_research_approval: bool
    require_adaptive_research_approval: bool
    require_final_report_approval: bool

    # Initial approval decisions
    initial_plan_approved: bool
    initial_research_approved: bool

    # True after the user authorizes web research.
    # This authorization is reused throughout the session.
    research_authorized: bool

    # Final report approval and explicit decision
    final_report_approved: bool
    final_report_review_decision: (
        FinalReportReviewDecision | None
    )

    # ========================================================
    # RESEARCH PLAN
    # ========================================================

    # Only accepted questions belong in the active plan.
    research_questions: list[ResearchQuestion]

    # ========================================================
    # ADAPTIVE RESEARCH PROPOSALS
    # ========================================================

    # Proposed questions stay separate until approved.
    proposed_research_questions: list[ResearchQuestion]

    adaptive_review_decision: AdaptiveReviewDecision | None

    # ========================================================
    # CURRENT RESEARCH QUESTION
    # ========================================================

    active_research_question: ResearchQuestion | None

    # One-based number of the active research question.
    active_research_question_number: int | None

    # Sources found for the active question.
    current_sources: list[Source]

    # ========================================================
    # RESEARCH DATA
    # ========================================================

    # All sources collected during the session.
    sources: list[Source]

    # Newly extracted evidence awaiting validation.
    pending_evidence: list[Evidence]

    # Evidence that passed deterministic validation.
    evidence: list[Evidence]

    # ========================================================
    # WORKFLOW CONTROL
    # ========================================================

    # Pointer to the next question to process.
    current_question_index: int

    # Indicates that the research workflow should stop.
    research_complete: bool

    # ========================================================
    # ADAPTIVE RESEARCH LIMITS
    # ========================================================

    research_round: int
    max_research_rounds: int
    max_total_research_questions: int

    # ========================================================
    # SUFFICIENCY ASSESSMENTS
    # ========================================================

    coverage_assessments: list[CoverageAssessmentState]

    research_gaps: list[str]

    research_sufficient: bool
    research_decision_reason: str

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    report: ResearchReport | None