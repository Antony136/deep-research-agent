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

    evidence: list[Evidence]

    # WORKFLOW CONTROL
    current_question_index: int

    research_complete: bool

    # FINAL OUTPUT
    report: ResearchReport | None
