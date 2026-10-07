"""
Main LangGraph workflow for the Deep Research Agent.

Current workflow:

    START
      |
      v
   Planner
      |
      v
  Researcher
      |
      v
Evidence Extractor
      |
      v
Evidence Verifier
      |
      v
  More research?
    /       \
  yes       no
   |         |
   v         v
Researcher  END
"""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.evidence_extractor import (
    evidence_extractor_node,
)
from app.graph.nodes.evidence_verifier import (
    evidence_verifier_node,
)
from app.graph.nodes.planner import planner_node
from app.graph.nodes.researcher import researcher_node
from app.graph.state import ResearchState


def should_continue_research(
    state: ResearchState,
) -> str:
    """
    Decide whether another planned research question
    still needs to be investigated.
    """

    if (
        state["current_question_index"]
        < len(state["research_questions"])
    ):
        return "research"

    return "end"


def build_research_graph():

    graph = StateGraph(ResearchState)

    # ------------------------------------------------------
    # NODES
    # ------------------------------------------------------

    graph.add_node(
        "planner",
        planner_node,
    )

    graph.add_node(
        "researcher",
        researcher_node,
    )

    graph.add_node(
        "evidence_extractor",
        evidence_extractor_node,
    )

    graph.add_node(
        "evidence_verifier",
        evidence_verifier_node,
    )

    # ------------------------------------------------------
    # INITIAL FLOW
    # ------------------------------------------------------

    graph.add_edge(
        START,
        "planner",
    )

    graph.add_edge(
        "planner",
        "researcher",
    )

    graph.add_edge(
        "researcher",
        "evidence_extractor",
    )

    # ------------------------------------------------------
    # EVIDENCE VERIFICATION
    # ------------------------------------------------------

    graph.add_edge(
        "evidence_extractor",
        "evidence_verifier",
    )

    # ------------------------------------------------------
    # RESEARCH LOOP
    # ------------------------------------------------------

    graph.add_conditional_edges(
        "evidence_verifier",
        should_continue_research,
        {
            "research": "researcher",
            "end": END,
        },
    )

    return graph.compile()
