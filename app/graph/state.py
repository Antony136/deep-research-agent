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


class CoverageAssessmentState(TypedDict):
    """
    Persist the sufficiency evaluator's decision for one
    original research question.
    """

    research_question_number: int
    covered: bool
    evidence_numbers: list[int]
    reason: str


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
    max_research_rounds: int
    max_total_research_questions: int

    # SUFFICIENCY ASSESSMENTS
    #
    # Retain the evaluator's decision for each original
    # research question so later nodes can use the same
    # decision instead of independently guessing coverage.
    coverage_assessments: list[CoverageAssessmentState]

    # Information that the sufficiency evaluator determined
    # is still missing from the research.
    research_gaps: list[str]

    # Overall sufficiency decision
    research_sufficient: bool
    research_decision_reason: str

    # FINAL OUTPUT
    report: ResearchReport | None
