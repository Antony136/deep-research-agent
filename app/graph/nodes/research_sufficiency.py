"""
Research sufficiency evaluator for the Deep Research Agent.

This node determines whether the collected verified evidence
is sufficient to answer the planned research questions.

The evaluator works question-by-question instead of sending
the entire evidence collection to one large LLM prompt.

Evidence ownership is explicit:

    Research Question
          ↓
    research_question_number
          ↓
    Owned verified evidence
          ↓
    Relevance filtering
          ↓
    LLM coverage evaluation

Evidence numbering is GLOBAL and remains stable across the
entire research session.
"""

import os
import re

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from app.graph.state import ResearchState
from app.schemas.research import Evidence, ResearchQuestion


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
# Configuration
# ------------------------------------------------------------

MAX_EVIDENCE_PER_QUESTION = 8
MIN_TOKEN_OVERLAP = 0.08


# ------------------------------------------------------------
# Structured output
# ------------------------------------------------------------


class CoverageAssessment(BaseModel):
    """
    Evidence mapping for one research question.
    """

    evidence_numbers: list[int] = Field(
        default_factory=list,
        description=(
            "GLOBAL 1-based numbers of verified evidence "
            "items that directly support this research "
            "question."
        ),
    )

    missing_information: str = Field(
        default="",
        description=(
            "Specific information still required to answer "
            "the research question."
        ),
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
    CoverageAssessment
)


# ------------------------------------------------------------
# Prompt
# ------------------------------------------------------------


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the evidence coverage evaluator of a deep
research agent.

Your task is to determine which VERIFIED EVIDENCE items
directly support ONE research question.

You are NOT answering the question.

Use ONLY the supplied evidence.

IMPORTANT RULES:

1. Only select evidence that directly helps answer the
   research question.

2. Do NOT select evidence merely because it mentions the
   same company, framework, or general topic.

3. Read both the CLAIM and SUPPORTING TEXT.

4. The evidence number is a GLOBAL evidence number.
   Preserve the exact numbers supplied in the input.

5. Do not invent evidence numbers.

6. Do not modify evidence numbers.

7. An evidence item can be selected only when its actual
   content contributes useful information to the question.

8. For strengths/weaknesses questions, select evidence
   describing actual strengths, limitations, tradeoffs,
   adoption concerns, reliability concerns, or similar
   characteristics.

9. For comparison questions, select evidence containing
   actual differences, capabilities, architecture,
   workflow, or tradeoffs between the relevant systems.

10. For real-world/case-study questions, select evidence
    describing actual deployments, customers, applications,
    use cases, or case studies.

11. If the available evidence does not adequately answer
    the question, return an empty evidence_numbers list.

12. When evidence_numbers is empty, explain specifically
    what information is still missing.

13. Do NOT use outside knowledge.

14. Do NOT infer facts that are not supported by the
    supplied evidence.

15. Prefer direct evidence over indirect evidence.

Return only the requested structured output.
""",
        ),
        (
            "human",
            """
ORIGINAL USER QUESTION:

{question}


RESEARCH QUESTION:

{research_question}


VERIFIED EVIDENCE CANDIDATES:

{evidence}


Determine which GLOBAL evidence numbers directly support
this research question.
""",
        ),
    ]
)


chain = prompt | structured_model


# ------------------------------------------------------------
# Text helpers
# ------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """
    Convert text into a lightweight set of lowercase tokens.
    """

    return {
        token
        for token in re.findall(
            r"[a-zA-Z0-9]+",
            text.lower(),
        )
        if len(token) > 2
    }


def _question_tokens(
    question: ResearchQuestion,
) -> set[str]:
    """
    Build tokens from the research question and its search
    queries.
    """

    text = " ".join(
        [
            question.question,
            *question.search_queries,
        ]
    )

    return _tokenize(text)


def _evidence_relevance_score(
    question_tokens: set[str],
    evidence: Evidence,
) -> float:
    """
    Calculate lightweight lexical relevance.

    This is NOT the final coverage decision.

    It only reduces the amount of evidence sent to the LLM.
    """

    evidence_tokens = _tokenize(
        f"{evidence.claim} {evidence.supporting_text}"
    )

    if not question_tokens or not evidence_tokens:
        return 0.0

    overlap = (
        len(question_tokens & evidence_tokens)
        / len(question_tokens)
    )

    return overlap


def _select_candidate_evidence(
    question: ResearchQuestion,
    evidence: list[Evidence],
) -> list[tuple[int, Evidence]]:
    """
    Select a small set of potentially relevant evidence items.

    The evidence passed into this function MUST already belong
    to the current research question.

    Evidence numbers remain GLOBAL.

    Returns:

        [
            (1, Evidence(...)),
            (7, Evidence(...)),
            ...
        ]
    """

    question_tokens = _question_tokens(question)

    scored = []

    for index, item in enumerate(
        evidence,
        start=1,
    ):

        score = _evidence_relevance_score(
            question_tokens=question_tokens,
            evidence=item,
        )

        if score >= MIN_TOKEN_OVERLAP:

            scored.append(
                (
                    score,
                    index,
                    item,
                )
            )

    scored.sort(
        key=lambda value: (
            -value[0],
            value[1],
        )
    )

    selected = scored[
        :MAX_EVIDENCE_PER_QUESTION
    ]

    return [
        (
            index,
            item,
        )
        for _, index, item in selected
    ]


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------


def _format_evidence(
    evidence: list[tuple[int, Evidence]],
) -> str:
    """
    Format candidate evidence while preserving GLOBAL IDs.
    """

    if not evidence:
        return "No relevant verified evidence candidates."

    sections = []

    for number, item in evidence:

        sections.append(
            f"""
GLOBAL EVIDENCE {number}

Claim:
{item.claim}

Source:
{item.source_url}

Supporting text:
{item.supporting_text}
""".strip()
        )

    return "\n\n".join(sections)


# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------


def _normalize_evidence_numbers(
    numbers: list[int],
    candidate_numbers: set[int],
) -> list[int]:
    """
    Keep only valid GLOBAL evidence numbers that were actually
    supplied to the LLM.
    """

    normalized = []

    for number in numbers:

        if (
            number in candidate_numbers
            and number not in normalized
        ):
            normalized.append(number)

    return normalized


# ------------------------------------------------------------
# Single-question evaluation
# ------------------------------------------------------------


def _evaluate_question(
    original_question: str,
    research_question: ResearchQuestion,
    evidence: list[Evidence],
    global_evidence_numbers: list[int],
) -> CoverageAssessment:
    """
    Evaluate one research question using only evidence that
    belongs to that research question.

    global_evidence_numbers contains the GLOBAL numbers
    corresponding to the supplied evidence list.
    """

    # --------------------------------------------------------
    # Local candidate selection
    # --------------------------------------------------------

    question_tokens = _question_tokens(
        research_question
    )

    scored = []

    for global_number, item in zip(
        global_evidence_numbers,
        evidence,
    ):

        score = _evidence_relevance_score(
            question_tokens=question_tokens,
            evidence=item,
        )

        if score >= MIN_TOKEN_OVERLAP:

            scored.append(
                (
                    score,
                    global_number,
                    item,
                )
            )

    scored.sort(
        key=lambda value: (
            -value[0],
            value[1],
        )
    )

    selected = scored[
        :MAX_EVIDENCE_PER_QUESTION
    ]

    candidates = [
        (
            global_number,
            item,
        )
        for _, global_number, item in selected
    ]

    print(
        f"\n  Evaluating:"
        f" {research_question.question}"
    )

    print(
        f"  Owned verified evidence: "
        f"{len(evidence)}"
    )

    print(
        f"  Candidate evidence: "
        f"{len(candidates)}"
    )

    if not candidates:

        return CoverageAssessment(
            evidence_numbers=[],
            missing_information=(
                "No relevant verified evidence was "
                "found for this research requirement."
            ),
        )

    candidate_numbers = {
        number
        for number, _ in candidates
    }

    try:

        response = chain.invoke(
            {
                "question": original_question,
                "research_question": (
                    research_question.question
                ),
                "evidence": _format_evidence(
                    candidates
                ),
            }
        )

    except Exception as exc:

        print(
            "  Coverage evaluation failed:"
        )

        print(
            f"  {exc}"
        )

        return CoverageAssessment(
            evidence_numbers=[],
            missing_information=(
                "Coverage evaluation failed; "
                "additional verification is required."
            ),
        )

    valid_numbers = _normalize_evidence_numbers(
        numbers=response.evidence_numbers,
        candidate_numbers=candidate_numbers,
    )

    if not valid_numbers:

        missing_information = (
            response.missing_information.strip()
        )

        if not missing_information:

            missing_information = (
                "The available verified evidence does "
                "not directly answer this requirement."
            )

        return CoverageAssessment(
            evidence_numbers=[],
            missing_information=missing_information,
        )

    return CoverageAssessment(
        evidence_numbers=valid_numbers,
        missing_information="",
    )


# ------------------------------------------------------------
# Main node
# ------------------------------------------------------------


def research_sufficiency_node(
    state: ResearchState,
) -> ResearchState:

    print("\n[Node] research_sufficiency")

    research_questions = state[
        "research_questions"
    ]

    evidence = state[
        "evidence"
    ]

    print(
        "  Planned research questions: "
        f"{len(research_questions)}"
    )

    print(
        "  Verified evidence available: "
        f"{len(evidence)}"
    )

    if not research_questions:

        print(
            "  No research questions available."
        )

        return {
            **state,
            "research_sufficient": False,
            "research_gaps": [
                "No research plan was generated."
            ],
            "research_decision_reason": (
                "Research sufficiency cannot be evaluated "
                "without a research plan."
            ),
        }

    if not evidence:

        print(
            "  No verified evidence available."
        )

        gaps = [
            question.question
            for question in research_questions
        ]

        return {
            **state,
            "research_sufficient": False,
            "research_gaps": gaps,
            "research_decision_reason": (
                "No verified evidence has been collected "
                "for the research objective."
            ),
        }

    # --------------------------------------------------------
    # Evaluate each research question independently
    # --------------------------------------------------------

    assessments = []

    for index, research_question in enumerate(
        research_questions,
        start=1,
    ):

        print(
            f"\n  Research requirement "
            f"{index}/{len(research_questions)}"
        )

        # ----------------------------------------------------
        # CRITICAL OWNERSHIP FILTER
        #
        # Only evidence explicitly produced for this
        # research question is allowed to participate in
        # its coverage evaluation.
        # ----------------------------------------------------

        owned_evidence_with_numbers = [
            (
                global_number,
                item,
            )
            for global_number, item in enumerate(
                evidence,
                start=1,
            )
            if (
                item.research_question_number
                == index
            )
        ]

        owned_global_numbers = [
            global_number
            for global_number, _ in
            owned_evidence_with_numbers
        ]

        owned_evidence = [
            item
            for _, item in
            owned_evidence_with_numbers
        ]

        print(
            "  Evidence belonging to this question: "
            f"{len(owned_evidence)}"
        )

        assessment = _evaluate_question(
            original_question=state["question"],
            research_question=research_question,
            evidence=owned_evidence,
            global_evidence_numbers=owned_global_numbers,
        )

        assessments.append(
            (
                index,
                assessment,
            )
        )

    # --------------------------------------------------------
    # Build research gaps
    # --------------------------------------------------------

    research_gaps = []

    for index, assessment in assessments:

        if assessment.evidence_numbers:
            continue

        question = research_questions[
            index - 1
        ]

        missing_information = (
            assessment.missing_information.strip()
        )

        if not missing_information:

            missing_information = (
                "Additional verified evidence is required "
                f"to answer: {question.question}"
            )

        gap = (
            f"{question.question} "
            f"— missing: "
            f"{missing_information}"
        )

        if gap not in research_gaps:

            research_gaps.append(
                gap
            )

    research_sufficient = not research_gaps

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    if research_sufficient:

        reason = (
            "Every planned research requirement has "
            "verified evidence directly mapped to it."
        )

    else:

        uncovered_count = len(
            research_gaps
        )

        reason = (
            f"{uncovered_count} research requirement(s) "
            "still lack directly mapped verified evidence."
        )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    print(
        "\n  Research sufficiency: "
        f"{research_sufficient}"
    )

    print(
        "  Coverage assessment:"
    )

    for index, assessment in assessments:

        status = (
            "COVERED"
            if assessment.evidence_numbers
            else "NOT COVERED"
        )

        print(
            f"    [{index}] "
            f"{status}"
        )

        if assessment.evidence_numbers:

            print(
                "        Evidence: "
                + ", ".join(
                    str(number)
                    for number
                    in assessment.evidence_numbers
                )
            )

        else:

            print(
                "        Missing: "
                f"{assessment.missing_information}"
            )

    print(
        "  Research gaps:"
    )

    if not research_gaps:

        print(
            "    None"
        )

    else:

        for gap in research_gaps:

            print(
                f"    - {gap}"
            )

    return {
        **state,
        "research_sufficient": research_sufficient,
        "research_gaps": research_gaps,
        "research_decision_reason": reason,
    }
