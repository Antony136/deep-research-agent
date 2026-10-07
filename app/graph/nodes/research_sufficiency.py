"""
Research sufficiency node for the Deep Research Agent.

This node evaluates whether the currently verified evidence is
sufficient to answer the user's complete research question.

The evaluator considers:

- the original user question
- the complete research plan
- which planned questions have already been processed
- the verified evidence collected so far

If the research is insufficient, the evaluator identifies
specific factual gaps that can later be handled by the
adaptive planner.

The evaluator does not perform web searches.
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
    """
    Structured decision returned by the sufficiency evaluator.
    """

    sufficient: bool = Field(
        description=(
            "Whether the verified evidence is sufficient "
            "to answer the complete main research question."
        )
    )

    reason: str = Field(
        description=(
            "A concise explanation of the overall evidence "
            "coverage and any important gaps."
        )
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description=(
            "Specific factual research gaps that must be "
            "addressed before the main question can be "
            "answered reliably. Return an empty list only "
            "when the research is genuinely sufficient."
        ),
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
collected so far is sufficient to answer the user's COMPLETE
research question reliably.

You must evaluate the research as a whole.

The original user question is the primary authority.

The research plan is supporting context only. It describes
possible investigations, but it does NOT define what must
automatically be researched.

IMPORTANT RULES:

1. Use ONLY the supplied verified evidence.

2. Do NOT use your general knowledge.

3. Do NOT invent facts.

4. Do NOT treat raw sources as evidence unless the relevant
   information appears in the verified evidence.

5. Do NOT consider rejected evidence.

6. Determine what the original user question actually requires
   before judging whether the research is sufficient.

7. Evaluate whether the verified evidence covers those
   requirements.

8. Do NOT assume that every research-plan question must be
   answered.

9. A research-plan question may be unnecessary if the verified
   evidence already provides the information needed to answer
   the original question.

10. Do NOT create new requirements merely because a research
    dimension sounds useful or interesting.

11. Missing information must be directly relevant to the
    original user question.

12. Prefer a small number of important evidence gaps over a
    large list of speculative gaps.

13. If the available evidence is sufficient to produce a
    reliable answer, return sufficient=true even if some
    potentially interesting information has not been researched.

14. If an important part of the original question cannot be
    answered reliably from the verified evidence, return
    sufficient=false.

REASONING PROCESS:

First determine the information requirements implied by the
original user question.

For example, a comparison question may require:

- coverage of the major entities being compared
- coverage of the dimensions explicitly requested
- evidence describing meaningful differences
- evidence supporting important tradeoffs

A recommendation question may require:

- evidence about the available options
- evidence about the criteria relevant to the recommendation
- enough information to distinguish the options

A factual question may require:

- evidence directly supporting the requested fact
- enough supporting context to avoid an unreliable conclusion

These are GENERAL STRUCTURAL PATTERNS.

Do not assume that every question has these exact dimensions.

Determine the appropriate requirements from the actual user
question.

EVIDENCE COVERAGE:

For each important requirement implied by the original question,
ask:

1. Is there verified evidence addressing this requirement?
2. Is the evidence sufficiently specific?
3. Is the evidence directly relevant?
4. Can the requirement be answered without unsupported
   assumptions?

If an important requirement is unsupported, research is
insufficient.

If all important requirements are adequately supported,
research is sufficient.

RESEARCH PLAN:

Use the research plan to understand what has already been
investigated and what may still be available.

However:

- Do NOT require every planned question to be processed.
- Do NOT assume an unprocessed question represents a missing
  requirement.
- Do NOT create a missing-information item merely because a
  planned question has not been processed.
- Only treat an unprocessed question as important when the
  original user question genuinely requires the information
  that question would provide.

RESEARCH PROGRESS:

The current research progress indicates how much of the plan
has been processed.

Progress alone must never determine sufficiency.

For example:

- A partially processed plan can be sufficient if the evidence
  already answers the original question.
- A fully processed plan can still be insufficient if important
  requirements remain unsupported.

MISSING INFORMATION:

When research is insufficient, describe WHAT important factual
information is missing.

Each missing-information item must:

- be directly connected to the original user question
- describe a factual information gap
- be specific enough for another planner to create a useful
  research question
- avoid prescribing the search method

Good structural forms:

- "Evidence about [specific missing capability]"
- "Comparison of [entity A] and [entity B] on [specific dimension]"
- "Real-world evidence about [specific aspect]"
- "Evidence needed to determine [specific tradeoff]"

These are structural examples only.

Do NOT copy their subjects into the current answer unless the
original question requires them.

Bad forms:

- "Need more research"
- "Search the internet"
- "Find better sources"
- "Research question 4"
- "Get more information"

Do not identify a gap merely because additional information
could be interesting.

SUFFICIENCY:

Return sufficient=true ONLY when the verified evidence provides
enough reliable coverage to answer the COMPLETE original user
question.

Return sufficient=false when one or more important requirements
of the original question remain unsupported.

If sufficient=true:

- missing_information MUST be an empty list
- reason must briefly explain why the important requirements
  are adequately covered

If sufficient=false:

- missing_information MUST contain only the important remaining
  factual gaps
- reason must explain why those gaps prevent a reliable answer

IMPORTANT:

Do not optimize for exhaustive research.

Optimize for sufficient evidence to answer the original
question reliably.

Return only the requested structured output.
""",
        ),
        (
            "human",
            """
ORIGINAL USER QUESTION:

{question}


RESEARCH PLAN:

{research_plan}


RESEARCH PROGRESS:

{research_progress}


VERIFIED EVIDENCE:

{evidence}
""",
        ),
    ]
)


chain = prompt | structured_model


def _format_research_plan(
    state: ResearchState,
) -> str:
    """
    Format the complete research plan and indicate which
    questions have already been processed.
    """

    questions = state[
        "research_questions"
    ]

    current_index = state[
        "current_question_index"
    ]

    if not questions:
        return "No research plan has been created."

    sections = []

    for index, question in enumerate(
        questions,
        start=1,
    ):
        if index <= current_index:
            status = "PROCESSED"
        else:
            status = "NOT YET PROCESSED"

        sections.append(
            f"""
QUESTION {index}
STATUS: {status}

Research question:
{question.question}

Search queries:
{", ".join(question.search_queries)}
""".strip()
        )

    return "\n\n".join(
        sections
    )


def _format_research_progress(
    state: ResearchState,
) -> str:
    """
    Provide explicit progress information to the evaluator.
    """

    questions = state[
        "research_questions"
    ]

    current_index = state[
        "current_question_index"
    ]

    processed = min(
        current_index,
        len(questions),
    )

    remaining = max(
        len(questions) - processed,
        0,
    )

    return (
        f"Research round: "
        f"{state['research_round']}\n"
        f"Questions processed: "
        f"{processed}/{len(questions)}\n"
        f"Questions remaining: "
        f"{remaining}"
    )


def _format_evidence(
    state: ResearchState,
) -> str:
    """
    Format trusted evidence for the sufficiency evaluator.
    """

    evidence = state[
        "evidence"
    ]

    if not evidence:
        return (
            "No verified evidence has been collected."
        )

    sections = []

    for index, item in enumerate(
        evidence,
        start=1,
    ):
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

    return "\n\n".join(
        sections
    )


def research_sufficiency_node(
    state: ResearchState,
) -> ResearchState:
    """
    Evaluate whether the current verified evidence is
    sufficient to answer the complete main question.
    """

    print(
        "\n[Node] research_sufficiency"
    )

    evidence = state[
        "evidence"
    ]

    print(
        "  Verified evidence available: "
        f"{len(evidence)}"
    )

    print(
        "  Research progress: "
        f"{state['current_question_index']}/"
        f"{len(state['research_questions'])}"
    )

    if not evidence:

        print(
            "  No verified evidence available. "
            "Research is insufficient."
        )

        return {
            **state,
            "research_sufficient": False,
            "research_decision_reason": (
                "No verified evidence has been "
                "collected for the research objective."
            ),
            "research_gaps": [
                "Reliable evidence covering the "
                "main research question"
            ],
        }

    try:

        response = chain.invoke(
            {
                "question": state[
                    "question"
                ],
                "research_plan": (
                    _format_research_plan(
                        state
                    )
                ),
                "research_progress": (
                    _format_research_progress(
                        state
                    )
                ),
                "evidence": (
                    _format_evidence(
                        state
                    )
                ),
            }
        )

    except Exception as exc:

        print(
            "  Sufficiency evaluation failed:"
        )

        print(
            f"  {exc}"
        )

        return {
            **state,
            "research_sufficient": False,
            "research_decision_reason": (
                "Sufficiency evaluation failed; "
                "additional research is safer."
            ),
            "research_gaps": [
                "Reliable evidence coverage "
                "could not be evaluated."
            ],
        }

    print(
        "  Research sufficient: "
        f"{response.sufficient}"
    )

    print(
        "  Reason:"
    )

    print(
        f"    {response.reason}"
    )

    if response.missing_information:

        print(
            "  Missing information:"
        )

        for gap in response.missing_information:
            print(
                f"    - {gap}"
            )

    else:

        print(
            "  Missing information: none"
        )

    return {
        **state,
        "research_sufficient": (
            response.sufficient
        ),
        "research_decision_reason": (
            response.reason
        ),
        "research_gaps": (
            response.missing_information
        ),
    }
