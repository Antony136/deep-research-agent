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
    Requirement coverage evaluation
          ↓
    Research gaps

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
MAX_REQUIREMENTS_PER_QUESTION = 5


# ------------------------------------------------------------
# Structured output
# ------------------------------------------------------------


class RequirementAssessment(BaseModel):
    """
    Coverage assessment for one factual requirement.

    The requirement text is kept together with its coverage
    decision so Python does not need to compare independently
    generated strings from the LLM.
    """

    requirement: str = Field(
        description=(
            "One concrete factual requirement that must be "
            "established to adequately answer the research "
            "question."
        )
    )

    covered: bool = Field(
        description=(
            "True only when the supplied verified evidence "
            "directly supports this requirement."
        )
    )

    evidence_numbers: list[int] = Field(
        default_factory=list,
        description=(
            "GLOBAL 1-based evidence numbers that directly "
            "support this requirement."
        )
    )

    reason: str = Field(
        default="",
        description=(
            "Brief explanation of why the supplied evidence "
            "does or does not support this requirement."
        )
    )


class CoverageAssessment(BaseModel):
    """
    Requirement-level evidence coverage for one research
    question.
    """

    requirements: list[RequirementAssessment] = Field(
        default_factory=list,
        description=(
            "Important factual requirements needed to answer "
            "the research question and their evidence coverage."
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

Your task is to determine whether VERIFIED EVIDENCE
adequately covers the important factual requirements
needed to answer ONE research question.

You are NOT writing the final answer.

Use ONLY the supplied evidence.

IMPORTANT RULES:

1. First identify the important factual requirements
   needed to answer the research question.

2. Break the research question into a small number of
   concrete answer requirements.

3. Create no more than 5 requirements.

4. Requirements must represent distinct information
   that a good answer should establish.

5. Do not create unnecessary or overly detailed
   requirements.

6. Do not invent requirements unrelated to the research
   question.

7. Evaluate EVERY requirement against the supplied
   VERIFIED EVIDENCE.

8. Set covered=true ONLY when the supplied evidence
   directly supports the requirement.

9. Set covered=false when the evidence is insufficient,
   indirect, unrelated, or only partially supports the
   requirement.

10. One evidence item may support multiple requirements.

11. Do not mark a requirement as covered merely because
    an evidence item is generally related to the topic.

12. Read BOTH the CLAIM and SUPPORTING TEXT.

13. evidence_numbers must contain only GLOBAL evidence
    numbers supplied in the input.

14. Do not invent evidence numbers.

15. Do not modify evidence numbers.

16. If covered=true, provide the evidence numbers that
    directly support that requirement.

17. If covered=false, evidence_numbers should normally be
    empty.

18. Do not use outside knowledge.

19. Do not infer facts that are not supported by the
    supplied evidence.

20. Keep each requirement specific and actionable.

21. Do not turn missing requirements into broad new
    research topics.

22. Keep the number of requirements small.

23. Return one RequirementAssessment for every important
    requirement you identify.

24. Return only the requested structured output.
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


Evaluate the evidence coverage for this research question.

First identify the important factual requirements.

Then evaluate every requirement independently.

For each requirement:

- State the requirement.
- Decide whether it is directly supported.
- If covered, provide the GLOBAL evidence numbers.
- If not covered, leave evidence_numbers empty.
- Give a short reason.

Do not evaluate based on outside knowledge.
Use only the supplied verified evidence.

Return only the requested structured output.
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
    Keep only valid GLOBAL evidence numbers that were
    actually supplied to the LLM.
    """

    normalized = []

    for number in numbers:

        if (
            number in candidate_numbers
            and number not in normalized
        ):
            normalized.append(number)

    return normalized


def _normalize_requirement(
    requirement: str,
) -> str:
    """
    Normalize a single requirement description.
    """

    return " ".join(
        requirement.strip().split()
    )


def _deduplicate_requirements(
    requirements: list[RequirementAssessment],
) -> list[RequirementAssessment]:
    """
    Normalize and deduplicate requirement assessments.
    """

    normalized = []
    seen = set()

    for assessment in requirements:

        requirement = _normalize_requirement(
            assessment.requirement
        )

        if not requirement:
            continue

        key = requirement.lower()

        if key in seen:
            continue

        seen.add(key)

        normalized.append(
            RequirementAssessment(
                requirement=requirement,
                covered=assessment.covered,
                evidence_numbers=assessment.evidence_numbers,
                reason=assessment.reason.strip(),
            )
        )

        if len(normalized) >= MAX_REQUIREMENTS_PER_QUESTION:
            break

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
            requirements=[
                RequirementAssessment(
                    requirement=(
                        research_question.question
                    ),
                    covered=False,
                    evidence_numbers=[],
                    reason=(
                        "No relevant verified evidence "
                        "was available."
                    ),
                )
            ]
        )

    candidate_numbers = {
        number
        for number, _ in candidates
    }

    # --------------------------------------------------------
    # LLM coverage evaluation
    # --------------------------------------------------------

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
            requirements=[
                RequirementAssessment(
                    requirement=(
                        research_question.question
                    ),
                    covered=False,
                    evidence_numbers=[],
                    reason=(
                        "Coverage evaluation failed; "
                        "additional verified evidence "
                        "is required."
                    ),
                )
            ]
        )

    # --------------------------------------------------------
    # Normalize model output
    # --------------------------------------------------------

    requirements = _deduplicate_requirements(
        response.requirements
    )

    # --------------------------------------------------------
    # Validate evidence numbers and coverage
    # --------------------------------------------------------

    normalized_requirements = []

    for requirement in requirements:

        valid_numbers = _normalize_evidence_numbers(
            numbers=requirement.evidence_numbers,
            candidate_numbers=candidate_numbers,
        )

        # A requirement cannot be considered covered if the
        # model did not provide valid supporting evidence.
        covered = (
            requirement.covered
            and bool(valid_numbers)
        )

        normalized_requirements.append(
            RequirementAssessment(
                requirement=requirement.requirement,
                covered=covered,
                evidence_numbers=(
                    valid_numbers
                    if covered
                    else []
                ),
                reason=requirement.reason,
            )
        )

    # --------------------------------------------------------
    # Handle malformed/empty structured output
    # --------------------------------------------------------

    if not normalized_requirements:

        return CoverageAssessment(
            requirements=[
                RequirementAssessment(
                    requirement=(
                        research_question.question
                    ),
                    covered=False,
                    evidence_numbers=[],
                    reason=(
                        "The evaluator did not return "
                        "usable coverage requirements."
                    ),
                )
            ]
        )

    return CoverageAssessment(
        requirements=normalized_requirements
    )


# ------------------------------------------------------------
# Main node
# ------------------------------------------------------------


def research_sufficiency_node(
    state: ResearchState,
) -> ResearchState:

    print(
        "\n[Node] research_sufficiency"
    )

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
            for global_number, _ in owned_evidence_with_numbers
        ]

        owned_evidence = [
            item
            for _, item in owned_evidence_with_numbers
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

        missing_requirements = [
            requirement.requirement
            for requirement in assessment.requirements
            if not requirement.covered
        ]

        if not missing_requirements:
            continue

        question = research_questions[
            index - 1
        ]

        missing_information = "; ".join(
            missing_requirements
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
            "Every planned research question has all "
            "identified important information requirements "
            "covered by verified evidence."
        )

    else:

        uncovered_count = len(
            research_gaps
        )

        reason = (
            f"{uncovered_count} research question(s) "
            "still have one or more missing information "
            "requirements."
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

        missing_requirements = [
            requirement
            for requirement in assessment.requirements
            if not requirement.covered
        ]

        covered_requirements = [
            requirement
            for requirement in assessment.requirements
            if requirement.covered
        ]

        status = (
            "COVERED"
            if not missing_requirements
            else "PARTIALLY COVERED"
        )

        print(
            f"    [{index}] "
            f"{status}"
        )

        if assessment.requirements:

            print(
                "        Requirements:"
            )

            for requirement in (
                assessment.requirements
            ):

                coverage_status = (
                    "COVERED"
                    if requirement.covered
                    else "MISSING"
                )

                print(
                    f"          - "
                    f"[{coverage_status}] "
                    f"{requirement.requirement}"
                )

                if requirement.reason:

                    print(
                        f"            Reason: "
                        f"{requirement.reason}"
                    )

                if requirement.evidence_numbers:

                    print(
                        "            Evidence: "
                        + ", ".join(
                            str(number)
                            for number
                            in requirement.evidence_numbers
                        )
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
