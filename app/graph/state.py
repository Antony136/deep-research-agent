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

    # Sources belonging to the research question
    # currently being processed.
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
    # targeted follow-up plan and advance the round.
    research_round: int

    # Hard application-level limit preventing the agent
    # from researching indefinitely.
    max_research_rounds: int

    # Information that the sufficiency evaluator determined
    # is still missing from the research.
    research_gaps: list[str]

    # Sufficiency decision
    research_sufficient: bool
    research_decision_reason: str

    # FINAL OUTPUT
    report: ResearchReport | None