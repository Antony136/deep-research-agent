"""
Experiment 07: First LangGraph workflow.

Purpose:
    Understand the three fundamental pieces of LangGraph:

    1. State
    2. Nodes
    3. Edges

This experiment does not use an LLM yet.

We first learn how LangGraph itself executes a workflow.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# ------------------------------------------------------------
# State
# ------------------------------------------------------------

class ResearchState(TypedDict):
    question: str
    plan: str
    status: str


# ------------------------------------------------------------
# Node 1: Create research plan
# ------------------------------------------------------------

def create_plan(
    state: ResearchState,
) -> ResearchState:

    print("\n[Node] create_plan")

    question = state["question"]

    plan = (
        f"Research the question: {question}"
    )

    return {
        **state,
        "plan": plan,
        "status": "planned",
    }


# ------------------------------------------------------------
# Node 2: Start research
# ------------------------------------------------------------

def start_research(
    state: ResearchState,
) -> ResearchState:

    print("[Node] start_research")

    return {
        **state,
        "status": "researching",
    }


# ------------------------------------------------------------
# Build graph
# ------------------------------------------------------------

def build_graph():

    graph = StateGraph(ResearchState)

    # --------------------------------------------------------
    # Register nodes
    # --------------------------------------------------------

    graph.add_node(
        "create_plan",
        create_plan,
    )

    graph.add_node(
        "start_research",
        start_research,
    )

    # --------------------------------------------------------
    # Connect nodes
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "create_plan",
    )

    graph.add_edge(
        "create_plan",
        "start_research",
    )

    graph.add_edge(
        "start_research",
        END,
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    return graph.compile()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("LANGGRAPH BASIC WORKFLOW")
    print("=" * 70)

    graph = build_graph()

    initial_state = {
        "question": (
            "What are the latest developments "
            "in AI agent frameworks?"
        ),
        "plan": "",
        "status": "started",
    }

    print("\nINITIAL STATE")
    print("-" * 70)

    print(initial_state)

    final_state = graph.invoke(
        initial_state
    )

    print("\nFINAL STATE")
    print("-" * 70)

    print(final_state)

    print("\n" + "=" * 70)
    print("LANGGRAPH BASIC WORKFLOW COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
