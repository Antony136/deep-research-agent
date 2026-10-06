"""
Experiment 09: LangGraph planning workflow.

Purpose:
    Build a multi-node workflow where one LLM node creates
    a research plan and another LLM node uses that plan
    to perform research.

Architecture:

    START
      |
      v
    Planner
      |
      v
    Researcher
      |
      v
     END
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


load_dotenv()


MODEL_NAME = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5-coder:7b",
)

BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


# ------------------------------------------------------------
# State
# ------------------------------------------------------------

class ResearchState(TypedDict):
    question: str
    plan: str
    research: str


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = ChatOllama(
    model=MODEL_NAME,
    base_url=BASE_URL,
    temperature=0,
)


# ------------------------------------------------------------
# Planner node
# ------------------------------------------------------------

def planner_node(
    state: ResearchState,
) -> ResearchState:

    print("\n[Node] planner")

    question = state["question"]

    response = model.invoke(
        [
            HumanMessage(
                content=(
                    "You are a research planner.\n\n"
                    "Create a concise research plan for the "
                    "following question.\n\n"
                    f"Question: {question}\n\n"
                    "List the main things that should be "
                    "investigated."
                )
            )
        ]
    )

    return {
        **state,
        "plan": response.content,
    }


# ------------------------------------------------------------
# Researcher node
# ------------------------------------------------------------

def researcher_node(
    state: ResearchState,
) -> ResearchState:

    print("[Node] researcher")

    question = state["question"]
    plan = state["plan"]

    response = model.invoke(
        [
            HumanMessage(
                content=(
                    "You are a research assistant.\n\n"
                    "Use the research plan below to answer "
                    "the user's question.\n\n"
                    f"Question:\n{question}\n\n"
                    f"Research plan:\n{plan}\n\n"
                    "Provide a concise research response."
                )
            )
        ]
    )

    return {
        **state,
        "research": response.content,
    }


# ------------------------------------------------------------
# Build graph
# ------------------------------------------------------------

def build_graph():

    graph = StateGraph(
        ResearchState
    )

    graph.add_node(
        "planner",
        planner_node,
    )

    graph.add_node(
        "researcher",
        researcher_node,
    )

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
        END,
    )

    return graph.compile()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("LANGGRAPH PLANNING WORKFLOW")
    print("=" * 70)

    graph = build_graph()

    initial_state = {
        "question": (
            "Compare LangChain, LangGraph, and CrewAI "
            "for production AI agent development."
        ),
        "plan": "",
        "research": "",
    }

    print("\nINITIAL STATE")
    print("-" * 70)

    print(initial_state)

    final_state = graph.invoke(
        initial_state
    )

    print("\nRESEARCH PLAN")
    print("-" * 70)

    print(final_state["plan"])

    print("\nRESEARCH RESULT")
    print("-" * 70)

    print(final_state["research"])

    print("\n" + "=" * 70)
    print("LANGGRAPH PLANNING WORKFLOW COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
