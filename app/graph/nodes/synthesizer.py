"""
Final synthesis node for the Deep Research Agent.

Converts verified research evidence into the final research report.

The synthesizer must:
- use verified evidence as the factual basis
- preserve source URLs
- acknowledge unresolved research gaps
- never invent unsupported claims
- clearly distinguish conclusions from limitations
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.schemas.research import ResearchReport


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


structured_model = model.with_structured_output(
    ResearchReport
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the final synthesis component of a deep research agent.

Your job is to produce a factual research report from the
verified evidence collected by the research system.

CRITICAL RULES:

1. Use ONLY the supplied verified evidence as factual support.

2. Do NOT use your own outside knowledge to fill missing information.

3. Do NOT invent facts, statistics, comparisons, model rankings,
   dates, benchmark results, or conclusions.

4. Every important factual claim in the report must be supported
   by the supplied evidence.

5. Preserve the source URL associated with supporting evidence
   whenever the report schema allows it.

6. If a research question is insufficiently supported, explicitly
   acknowledge the limitation.

7. Do NOT pretend that insufficient evidence is sufficient.

8. Do NOT manufacture an answer simply because the user asked
   the question.

9. Combine related evidence into clear conclusions rather than
   listing evidence mechanically.

10. Resolve contradictions carefully. If the supplied evidence
    genuinely conflicts, report the conflict instead of choosing
    an unsupported answer.

11. The final report should directly answer the user's original
    research question as far as the evidence allows.

12. The report should be useful, concise, and logically organized.

The research process may terminate because the research budget
was exhausted even when some questions remain insufficient.

In that case, clearly distinguish:

- findings supported by evidence
- conclusions that can reasonably be drawn
- unresolved questions
- limitations caused by insufficient evidence
""",
        ),
        (
            "human",
            """
Original user research question:

{question}


Research questions investigated:

{research_questions}


Verified evidence:

{evidence}


Research gaps that remain unresolved:

{research_gaps}


Research sufficient:

{research_sufficient}


Research decision:

{research_decision_reason}


Create the final research report.

The report must be based only on the verified evidence above.

If evidence is insufficient for part of the original question,
state that limitation clearly instead of guessing.
""",
        ),
    ]
)


chain = prompt | structured_model


def _format_research_questions(
    research_questions,
) -> str:
    """
    Format research questions for the synthesis prompt.
    """

    if not research_questions:
        return "No research questions were generated."

    lines = []

    for index, research_question in enumerate(
        research_questions,
        start=1,
    ):
        parent = research_question.parent_question_number

        if parent is None:
            relationship = "original question"
        else:
            relationship = (
                f"follow-up to original question Q{parent}"
            )

        lines.append(
            f"Q{index} ({relationship}): "
            f"{research_question.question}"
        )

    return "\n".join(lines)


def _format_evidence(
    evidence,
) -> str:
    """
    Format verified evidence for the synthesis model.

    Only fields that actually exist on the Evidence schema
    are used here.
    """

    if not evidence:
        return "No verified evidence was collected."

    blocks = []

    for index, item in enumerate(
        evidence,
        start=1,
    ):
        blocks.append(
            "\n".join(
                [
                    f"Evidence {index}",
                    f"Research question: "
                    f"Q{item.research_question_number}",
                    f"Claim: {item.claim}",
                    f"Supporting passage: "
                    f"{item.supporting_text}",
                    f"Source URL: {item.source_url}",
                ]
            )
        )

    return "\n\n".join(blocks)


def _format_research_gaps(
    research_gaps,
) -> str:
    """
    Format unresolved research gaps.
    """

    if not research_gaps:
        return "No unresolved research gaps."

    return "\n".join(
        f"- {gap}"
        for gap in research_gaps
    )


def synthesis_node(
    state,
):
    """
    Generate the final research report from verified evidence.
    """

    print(
        "\n[Node] synthesizer"
    )

    print(
        f"Verified evidence: "
        f"{len(state['evidence'])}"
    )

    print(
        f"Research sufficient: "
        f"{state['research_sufficient']}"
    )

    print(
        "Generating final research report..."
    )

    response = chain.invoke(
        {
            "question": state["question"],
            "research_questions": (
                _format_research_questions(
                    state["research_questions"]
                )
            ),
            "evidence": _format_evidence(
                state["evidence"]
            ),
            "research_gaps": _format_research_gaps(
                state["research_gaps"]
            ),
            "research_sufficient": (
                state["research_sufficient"]
            ),
            "research_decision_reason": (
                state["research_decision_reason"]
            ),
        }
    )

    print(
        "[Node] Final synthesis completed."
    )

    return {
        **state,
        "report": response,
    }
