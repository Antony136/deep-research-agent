"""
Experiment 06: Structured research decision.

Purpose:
    Use LangChain structured output to make a predictable
    decision about what the research system should do next.

This experiment replaces unreliable free-form tool-call text
with a Pydantic decision that our application can safely use.
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
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
# Structured decision schema
# ------------------------------------------------------------

class ResearchDecision(BaseModel):
    action: str = Field(
        description=(
            "The next action to take. "
            "Must be one of: search, calculate, answer."
        )
    )

    reasoning: str = Field(
        description=(
            "Short explanation of why this action "
            "is appropriate."
        )
    )

    query: str | None = Field(
        default=None,
        description=(
            "Search query when the action is search. "
            "Otherwise null."
        ),
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("STRUCTURED RESEARCH DECISION")
    print("=" * 70)

    model = ChatOllama(
        model=MODEL_NAME,
        base_url=BASE_URL,
        temperature=0,
    )

    # --------------------------------------------------------
    # Structured model
    # --------------------------------------------------------

    structured_model = model.with_structured_output(
        ResearchDecision
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the decision component of a research agent.

Choose the next action for the research system.

Available actions:

- search:
  Use when external information is required.

- calculate:
  Use when mathematical computation is required.

- answer:
  Use when the question can be answered without
  external research or calculation.

Return only the requested structured output.
""",
            ),
            (
                "human",
                "User question:\n{question}",
            ),
        ]
    )

    chain = prompt | structured_model

    # --------------------------------------------------------
    # Test question
    # --------------------------------------------------------

    question = (
        "What are the latest major developments in "
        "AI agent frameworks?"
    )

    decision = chain.invoke(
        {
            "question": question,
        }
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\nDECISION")
    print("-" * 70)

    print(
        f"Action: "
        f"{decision.action}"
    )

    print(
        f"Reasoning: "
        f"{decision.reasoning}"
    )

    print(
        f"Query: "
        f"{decision.query}"
    )

    print("\nDECISION OBJECT")
    print("-" * 70)

    print(decision)

    print("\n" + "=" * 70)
    print("STRUCTURED DECISION TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
