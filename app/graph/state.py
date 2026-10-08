"""
LangGraph state for the Deep Research Agent.

The state is the shared data structure that flows through
every node in the research workflow.
"""

from typing import TypedDict

from app.schemas.research import (
    Evidence,
    ResearchQuestion,
    ResearchReport,
    Source,
)


class ResearchState(TypedDict):
    """
    Complete state of a research session.
    """

    # USER REQUEST
    question: str

    # RESEARCH PLAN
    research_questions: list[ResearchQuestion]

    # CURRENT RESEARCH QUESTION
    #
    # current_question_index is the pointer used by the
    # workflow to determine which question should be
    # researched next.
    #
    # active_research_question is the question whose
    # sources/evidence are currently being processed.
    #
    # These are intentionally separate because the workflow
    # advances the next-question pointer before later nodes
    # finish processing the current question.
    active_research_question: ResearchQuestion | None

    # 1-based number of the research question currently
    # being processed.
    active_research_question_number: int | None

    # Sources belonging to the research question currently
    # being processed.
    current_sources: list[Source]

    # RESEARCH DATA
    sources: list[Source]

    # Evidence extracted by the LLM for the current
    # research question, before deterministic validation.
    pending_evidence: list[Evidence]

    # Evidence that passed deterministic validation
    # and is trusted by the rest of the pipeline.
    evidence: list[Evidence]

    # WORKFLOW CONTROL
    current_question_index: int

    research_complete: bool

    # ADAPTIVE RESEARCH
    #
    # research_round = 1 means the initial research plan.
    #
    # If the initial plan is exhausted and the evidence is
    # still insufficient, the adaptive planner can create a
    # targeted follow-up plan.
    research_round: int

    # Maximum number of research rounds.
    #
    # This controls how many adaptive cycles are allowed.
    max_research_rounds: int

    # Absolute maximum number of research questions that
    # may be processed during the entire research session.
    #
    # This is separate from max_research_rounds.
    #
    # Example:
    #
    #   Initial plan       = 5 questions
    #   Adaptive additions = up to 3 questions
    #   Total              = 8 questions maximum
    #
    # This prevents research from expanding indefinitely
    # through repeated adaptive planning.
    max_total_research_questions: int

    # Information that the sufficiency evaluator determined
    # is still missing from the research.
    research_gaps: list[str]

    # Sufficiency decision
    research_sufficient: bool
    research_decision_reason: str

    # FINAL OUTPUT
    report: ResearchReport | None
