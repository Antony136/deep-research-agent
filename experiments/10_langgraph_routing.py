"""
Experiment 10: LangGraph conditional routing.

Purpose:
    Learn how LangGraph can route execution to different
    nodes based on the current state.

Architecture:

                    START
                      |
                      v
                   Planner
                      |
                      v
                Decide Action
                 /          \
                /            \
               v              v
            Search          Answer
               \              /
                \            /
                 v          v
                     END
"""

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


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
    action: str
    result: str


# ------------------------------------------------------------
# Structured decision
# ------------------------------------------------------------

class ActionDecision(BaseModel):
    action: Literal["search", "answer"] = Field(
        description=(
            "The next action. Choose search when "
            "external information is required. "
            "Choose answer when no external research "
            "is necessary."
        )
    )


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = ChatOllama(
    model=MODEL_NAME,
    base_url=BASE_URL,
    temperature=0,
)


structured_model = model.with_structured_output(
    ActionDecision
)


# ------------------------------------------------------------
# Planner node
# ------------------------------------------------------------

def planner_node(
    state: ResearchState,
) -> ResearchState:

    print("\n[Node] planner")

    question = state["question"]

    response = structured_model.invoke(
        [
            HumanMessage(
                content=(
                    "Decide what should happen next.\n\n"
                    "Choose 'search' if the question requires "
                    "external or up-to-date information.\n"
                    "Choose 'answer' if it can be answered "
                    "without external research.\n\n"
                    f"Question: {question}"
                )
            )
        ]
    )

    print(
        f"Decision: {response.action}"
    )

    return {
        **state,
        "action": response.action,
    }


# ------------------------------------------------------------
# Search node
# ------------------------------------------------------------

def search_node(
    state: ResearchState,
) -> ResearchState:

    print("[Node] search")

    # This is deliberately a placeholder.
    # The real web-search tool will be added later.

    return {
        **state,
        "result": (
            "Search would be performed here."
        ),
    }


# ------------------------------------------------------------
# Answer node
# ------------------------------------------------------------

def answer_node(
    state: ResearchState,
) -> ResearchState:

    print("[Node] answer")

    response = model.invoke(
        [
            HumanMessage(
                content=(
                    "Answer this question concisely:\n\n"
                    f"{state['question']}"
                )
            )
        ]
    )

    return {
        **state,
        "result": response.content,
    }


# ------------------------------------------------------------
# Routing function
# ------------------------------------------------------------

def route_action(
    state: ResearchState,
) -> Literal["search", "answer"]:

    return state["action"]


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
        "search",
        search_node,
    )

    graph.add_node(
        "answer",
        answer_node,
    )

    graph.add_edge(
        START,
        "planner",
    )

    # --------------------------------------------------------
    # Conditional edge
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "planner",
        route_action,
        {
            "search": "search",
            "answer": "answer",
        },
    )

    graph.add_edge(
        "search",
        END,
    )

    graph.add_edge(
        "answer",
        END,
    )

    return graph.compile()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("LANGGRAPH CONDITIONAL ROUTING")
    print("=" * 70)

    graph = build_graph()

    questions = [
        (
            "What are the latest developments "
            "in AI agent frameworks?"
        ),
        (
            "What is 25 multiplied by 8?"
        ),
    ]

    for question in questions:

        print("\n" + "-" * 70)
        print(f"QUESTION: {question}")
        print("-" * 70)

        initial_state = {
            "question": question,
            "action": "",
            "result": "",
        }

        final_state = graph.invoke(
            initial_state
        )

        print("\nFINAL STATE")
        print("-" * 70)

        print(final_state)

    print("\n" + "=" * 70)
    print("LANGGRAPH CONDITIONAL ROUTING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
