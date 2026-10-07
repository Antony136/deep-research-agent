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
Research Sufficiency
      |
      v
   Enough?
    /    \
  yes     no
   |       |
  END   More questions?
           /     \
         yes      no
          |        |
          v        v
     Researcher   END
"""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.evidence_extractor import evidence_extractor_node
from app.graph.nodes.evidence_verifier import evidence_verifier_node
from app.graph.nodes.planner import planner_node
from app.graph.nodes.researcher import researcher_node
from app.graph.nodes.research_sufficiency import (
    research_sufficiency_node,
)
from app.graph.state import ResearchState


def should_continue_research(
    state: ResearchState,
) -> str:
    """
    Decide whether the research workflow should continue.

    The sufficiency evaluator gets the first opportunity
    to stop the research early.

    If the evidence is sufficient, the workflow ends.

    Otherwise, if planned research questions remain,
    another research cycle is started.

    If all planned questions have already been processed,
    the workflow ends even if the evidence is insufficient.
    """

    if state["research_sufficient"]:
        print(
            "\n[Router] Research is sufficient."
        )

        return "end"

    if (
        state["current_question_index"]
        < len(state["research_questions"])
    ):
        print(
            "\n[Router] More research is required."
        )

        return "research"

    print(
        "\n[Router] No planned research questions remain."
    )

    return "end"


def build_research_graph():
    """
    Build and compile the complete research workflow.
    """

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

    graph.add_node(
        "research_sufficiency",
        research_sufficiency_node,
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

    # ------------------------------------------------------
    # RESEARCH CYCLE
    # ------------------------------------------------------

    graph.add_edge(
        "researcher",
        "evidence_extractor",
    )

    graph.add_edge(
        "evidence_extractor",
        "evidence_verifier",
    )

    graph.add_edge(
        "evidence_verifier",
        "research_sufficiency",
    )

    # ------------------------------------------------------
    # ADAPTIVE ROUTING
    # ------------------------------------------------------

    graph.add_conditional_edges(
        "research_sufficiency",
        should_continue_research,
        {
            "research": "researcher",
            "end": END,
        },
    )

    return graph.compile()
