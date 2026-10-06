"""
Experiment 03: Structured research planning.

Purpose:
    Use LangChain prompt templates and structured output
    to make the LLM produce a predictable research plan.
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
# Structured output schema
# ------------------------------------------------------------

class ResearchPlan(BaseModel):
    objective: str = Field(
        description="The main objective of the research."
    )

    sub_questions: list[str] = Field(
        description=(
            "Important questions that must be answered "
            "to complete the research."
        )
    )

    search_queries: list[str] = Field(
        description=(
            "Search queries that should be used to "
            "research the topic."
        )
    )

    research_depth: int = Field(
        description=(
            "Recommended research depth from 1 to 5."
        )
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("STRUCTURED RESEARCH PLAN")
    print("=" * 70)

    model = ChatOllama(
        model=MODEL_NAME,
        base_url=BASE_URL,
        temperature=0,
    )

    # Ask LangChain to enforce the Pydantic structure.
    structured_model = model.with_structured_output(
        ResearchPlan
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert research planner.

Your job is to convert a user's research question
into a clear research plan.

Break complex questions into useful sub-questions.
Create practical search queries.
Choose a research depth from 1 to 5.

Return only the requested structured output.
""",
            ),
            (
                "human",
                "Research question:\n{question}",
            ),
        ]
    )

    chain = prompt | structured_model

    question = (
        "Compare LangChain, LangGraph, and CrewAI "
        "for building production AI agent systems."
    )

    plan = chain.invoke(
        {
            "question": question,
        }
    )

    print("\nRESEARCH PLAN")
    print("-" * 70)

    print(f"\nObjective:\n{plan.objective}")

    print("\nSub-questions:")
    for index, item in enumerate(
        plan.sub_questions,
        start=1,
    ):
        print(f"{index}. {item}")

    print("\nSearch queries:")
    for index, query in enumerate(
        plan.search_queries,
        start=1,
    ):
        print(f"{index}. {query}")

    print(
        f"\nResearch depth: "
        f"{plan.research_depth}"
    )

    print("\n" + "=" * 70)
    print("STRUCTURED PLANNING TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
