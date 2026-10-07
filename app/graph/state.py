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

    # RESEARCH DATA
    sources: list[Source]

    # Sources belonging to the research question
    # currently being processed.
    current_sources: list[Source]

    # Evidence extracted by the LLM for the current
    # research question, before deterministic validation.
    pending_evidence: list[Evidence]

    # Evidence that passed deterministic validation
    # and is trusted by the rest of the pipeline.
    evidence: list[Evidence]

    # WORKFLOW CONTROL
    current_question_index: int

    research_complete: bool

    # Adaptive research decision
    research_sufficient: bool
    research_decision_reason: str

    # FINAL OUTPUT
    report: ResearchReport | None
