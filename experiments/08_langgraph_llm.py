"""
Experiment 08: LLM inside LangGraph.

Purpose:
    Use our local Qwen model inside a LangGraph node.

Architecture:

    START
      |
      v
    LLM Node
      |
      v
     END

The LangGraph node receives state, calls the LLM,
and writes the result back into state.
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
    answer: str


# ------------------------------------------------------------
# Create model
# ------------------------------------------------------------

model = ChatOllama(
    model=MODEL_NAME,
    base_url=BASE_URL,
    temperature=0,
)


# ------------------------------------------------------------
# LLM node
# ------------------------------------------------------------

def research_node(
    state: ResearchState,
) -> ResearchState:

    print("\n[Node] research_node")

    question = state["question"]

    messages = [
        HumanMessage(
            content=(
                "Answer the following research question "
                "in one concise paragraph.\n\n"
                f"Question: {question}"
            )
        )
    ]

    response = model.invoke(
        messages
    )

    return {
        **state,
        "answer": response.content,
    }


# ------------------------------------------------------------
# Build graph
# ------------------------------------------------------------

def build_graph():

    graph = StateGraph(
        ResearchState
    )

    graph.add_node(
        "research",
        research_node,
    )

    graph.add_edge(
        START,
        "research",
    )

    graph.add_edge(
        "research",
        END,
    )

    return graph.compile()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("LANGGRAPH + QWEN")
    print("=" * 70)

    graph = build_graph()

    initial_state = {
        "question": (
            "What is the difference between "
            "LangChain and LangGraph?"
        ),
        "answer": "",
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

    print("\nLLM ANSWER")
    print("-" * 70)

    print(final_state["answer"])

    print("\n" + "=" * 70)
    print("LANGGRAPH + QWEN TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
