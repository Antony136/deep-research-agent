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
  More research?
    /       \
  yes       no
   |         |
   └───┐     v
       │    END
       │
       └── Researcher
"""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.planner import planner_node
from app.graph.nodes.researcher import researcher_node
from app.graph.state import ResearchState


def should_continue_research(
    state: ResearchState,
) -> str:
    """
    Decide whether another research question remains.
    """

    if state["current_question_index"] < len(
        state["research_questions"]
    ):
        return "research"

    return "end"


def build_research_graph():
    """
    Build and compile the complete research workflow.
    """

    graph = StateGraph(
        ResearchState
    )

    # ------------------------------------------------------
    # Nodes
    # ------------------------------------------------------

    graph.add_node(
        "planner",
        planner_node,
    )

    graph.add_node(
        "researcher",
        researcher_node,
    )

    # ------------------------------------------------------
    # Initial flow
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
    # Research loop
    # ------------------------------------------------------

    graph.add_conditional_edges(
        "researcher",
        should_continue_research,
        {
            "research": "researcher",
            "end": END,
        },
    )

    return graph.compile()
