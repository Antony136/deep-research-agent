"""
Main LangGraph workflow for the Deep Research Agent.

Current workflow:

    START
      |
      v
   Planner
      |
      v
     END

More research nodes will be added as the system grows.
"""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.planner import planner_node
from app.graph.state import ResearchState


def build_research_graph():
    """
    Build and compile the research workflow.
    """

    graph = StateGraph(
        ResearchState
    )

    graph.add_node(
        "planner",
        planner_node,
    )

    graph.add_edge(
        START,
        "planner",
    )

    graph.add_edge(
        "planner",
        END,
    )

    return graph.compile()
