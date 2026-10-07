"""
Research sufficiency node for the Deep Research Agent.

This node evaluates whether the currently verified evidence is
sufficient to answer the user's main research question.

The decision is made by the LLM using only trusted evidence.
It does not inspect pending or rejected evidence.

The result is stored in the LangGraph state so that the graph
can decide whether to continue researching or move forward.
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from app.graph.state import ResearchState

load_dotenv()

MODEL_NAME = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5-coder:7b",
)

BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

model = ChatOllama(
    model=MODEL_NAME,
    base_url=BASE_URL,
    temperature=0,
)


class ResearchSufficiencyOutput(BaseModel):
    sufficient: bool = Field(
        description=(
            "Whether the verified evidence is sufficient "
            "to answer the main research question."
        )
    )

    reason: str = Field(
        description=(
            "A concise explanation based only on the "
            "verified evidence."
        )
    )


structured_model = model.with_structured_output(
    ResearchSufficiencyOutput
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the research sufficiency evaluator in a deep
research agent.

Your task is to determine whether the VERIFIED evidence
collected so far is sufficient to answer the user's main
research question.

Important rules:

1. Use ONLY the verified evidence supplied to you.
2. Do not use your general knowledge.
3. Do not infer unsupported facts.
4. Do not consider rejected evidence.
5. Do not consider raw web sources unless they are represented
   in the verified evidence.
6. The evidence should cover the important aspects of the
   user's question.
7. For comparison questions, look for evidence covering the
   major entities or dimensions being compared.
8. Prefer multiple independent sources when possible.
9. If important parts of the question remain unsupported,
   mark the research as insufficient.
10. If the evidence provides enough factual coverage to
    produce a reliable answer, mark it as sufficient.

The reason must briefly explain the evidence coverage or the
important gaps.

Return only the requested structured output.
""",
        ),
        (
            "human",
            """
Main research question:

{question}

Verified evidence:

{evidence}
""",
        ),
    ]
)

chain = prompt | structured_model


def _format_evidence(state: ResearchState) -> str:
    """
    Format trusted evidence for the sufficiency evaluator.
    """

    evidence = state["evidence"]

    if not evidence:
        return "No verified evidence has been collected."

    sections = []

    for index, item in enumerate(evidence, start=1):
        sections.append(
            f"""
EVIDENCE {index}

Claim:
{item.claim}

Source URL:
{item.source_url}

Supporting text:
{item.supporting_text}
""".strip()
        )

    return "\n\n".join(sections)


def research_sufficiency_node(
    state: ResearchState,
) -> ResearchState:
    """
    Evaluate whether the current verified evidence is
    sufficient to answer the main research question.
    """

    print("\n[Node] research_sufficiency")

    evidence = state["evidence"]

    print(
        f"  Verified evidence available: "
        f"{len(evidence)}"
    )

    # With no trusted evidence, continuing research is
    # always preferable to declaring the question answered.
    if not evidence:
        print(
            "  No verified evidence available. "
            "Research is insufficient."
        )

        return {
            **state,
            "research_sufficient": False,
            "research_decision_reason": (
                "No verified evidence has been collected."
            ),
        }

    formatted_evidence = _format_evidence(state)

    try:
        response = chain.invoke(
            {
                "question": state["question"],
                "evidence": formatted_evidence,
            }
        )

    except Exception as exc:
        print(
            "  Sufficiency evaluation failed:"
        )
        print(
            f"  {exc}"
        )

        # Fail closed: if the evaluator cannot determine
        # sufficiency, continue research when possible.
        return {
            **state,
            "research_sufficient": False,
            "research_decision_reason": (
                "Sufficiency evaluation failed; "
                "additional research is safer."
            ),
        }

    print(
        f"  Research sufficient: "
        f"{response.sufficient}"
    )

    print(
        f"  Reason: "
        f"{response.reason}"
    )

    return {
        **state,
        "research_sufficient": response.sufficient,
        "research_decision_reason": response.reason,
    }
