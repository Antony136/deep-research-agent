"""
Research sufficiency evaluator for the Deep Research Agent.

Evaluates each original research question independently and
persists its coverage assessment in LangGraph state.

Adaptive follow-up questions contribute evidence to their
original parent's coverage assessment.
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


MAX_EVIDENCE_PER_QUESTION = 8
MIN_TOKEN_OVERLAP = 0.05


class CoverageAssessment(BaseModel):
    """Sufficiency decision for one original research question."""

    covered: bool = Field(
        description=(
            "True only when verified evidence is sufficient "
            "to answer the original research question."
        )
    )

    evidence_numbers: list[int] = Field(
        default_factory=list,
        description=(
            "Global 1-based evidence numbers directly supporting "
            "the coverage decision."
        ),
    )

    reason: str = Field(
        default="",
        description=(
            "Concise explanation of why the evidence is or is "
            "not sufficient."
        ),
    )


model = ChatOllama(
    model=MODEL_NAME,
    base_url=BASE_URL,
    temperature=0,
)

structured_model = model.with_structured_output(
    CoverageAssessment
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the research sufficiency evaluator of a deep
research agent.

Decide whether VERIFIED EVIDENCE is sufficient to answer
ONE ORIGINAL RESEARCH QUESTION. You are not writing the
final answer.

RULES:

1. The original research question defines the complete scope.
2. Do not invent additional requirements.
3. Do not import requirements from other original questions.
4. Evaluate only the supplied verified evidence.
5. Read both the evidence claim and supporting text.
6. Judge semantic relevance, not just keyword overlap.
7. Evidence must directly support the answer.
8. A comparison question requires enough evidence to make
   the requested comparison.
9. Questions with multiple explicit dimensions require
   sufficient evidence for those dimensions.
10. Do not require unrelated background, recommendations,
    examples, limitations, or implementation details unless
    the original question requires them.
11. Be conservative. If important information is missing,
    set covered=false.
12. Do not use outside knowledge or infer unsupported facts.
13. Use only the global evidence numbers provided.
14. If covered=true, provide valid evidence numbers.
15. If covered=false, evidence_numbers should normally be empty.
16. Explain any missing information concisely.

Return only the requested structured output.
""",
        ),
        (
            "human",
            """
ORIGINAL USER QUESTION:

{question}

ORIGINAL RESEARCH QUESTION BEING EVALUATED:

{research_question}

QUESTION TREE:

{question_tree}

VERIFIED EVIDENCE:

{evidence}

Evaluate only the original research question.

Evidence from its follow-up descendants may be used when it
directly helps answer the original question. Do not import
requirements from other original questions.

If the evidence is sufficient, set covered=true and provide
global evidence numbers supporting that decision.

If important information is missing, set covered=false,
normally return an empty evidence_numbers list, and explain
what is missing.
""",
        ),
    ]
)

chain = prompt | structured_model


def _tokenize(text: str) -> set[str]:
    """Convert text into lowercase tokens."""

    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2
    }


def _question_tokens(question: ResearchQuestion) -> set[str]:
    """Build tokens from a question and its search queries."""

    return _tokenize(
        " ".join(
            [
                question.question,
                *question.search_queries,
            ]
        )
    )


def _combined_question_tokens(
    questions: list[ResearchQuestion],
) -> set[str]:
    """Build tokens for candidate selection only."""

    tokens = set()

    for question in questions:
        tokens.update(_question_tokens(question))

    return tokens


def _evidence_relevance_score(
    question_tokens: set[str],
    evidence: Evidence,
) -> float:
    """Calculate lexical relevance for candidate selection."""

    evidence_tokens = _tokenize(
        f"{evidence.claim} {evidence.supporting_text}"
    )

    if not question_tokens or not evidence_tokens:
        return 0.0

    return (
        len(question_tokens & evidence_tokens)
        / len(question_tokens)
    )


def _format_evidence(
    evidence: list[tuple[int, Evidence]],
) -> str:
    """Format evidence while preserving global IDs."""

    if not evidence:
        return "No relevant verified evidence candidates."

    sections = []

    for number, item in evidence:
        sections.append(
            f"""
GLOBAL EVIDENCE {number}

Research question number:
Q{item.research_question_number}

Claim:
{item.claim}

Source:
{item.source_url}

Supporting text:
{item.supporting_text}
""".strip()
        )

    return "\n\n".join(sections)


def _format_question_tree(
    question_tree: list[tuple[int, ResearchQuestion]],
) -> str:
    """Format the actual question-tree lineage."""

    if not question_tree:
        return "No question-tree information available."

    sections = []

    for number, question in question_tree:
        parent = question.parent_question_number

        relationship = (
            "original question"
            if parent is None
            else f"follow-up to Q{parent}"
        )

        sections.append(
            f"Q{number} ({relationship})\n\n"
            f"{question.question}"
        )

    return "\n\n".join(sections)


def _normalize_evidence_numbers(
    numbers: list[int],
    candidate_numbers: set[int],
) -> list[int]:
    """Keep unique global IDs that were supplied to the LLM."""

    normalized = []

    for number in numbers:
        if (
            number in candidate_numbers
            and number not in normalized
        ):
            normalized.append(number)

    return normalized


def _get_descendant_question_numbers(
    root_question_number: int,
    research_questions: list[ResearchQuestion],
) -> set[int]:
    """Return a root question and all its recursive descendants."""

    descendants = {root_question_number}

    changed = True

    while changed:
        changed = False

        for index, question in enumerate(
            research_questions,
            start=1,
        ):
            if index in descendants:
                continue

            if question.parent_question_number in descendants:
                descendants.add(index)
                changed = True

    return descendants


def _get_evidence_for_question_tree(
    root_question_number: int,
    research_questions: list[ResearchQuestion],
    evidence: list[Evidence],
) -> list[tuple[int, Evidence]]:
    """Collect evidence from a root question and its descendants."""

    question_numbers = _get_descendant_question_numbers(
        root_question_number=root_question_number,
        research_questions=research_questions,
    )

    print(
        "  Evidence lineage questions: "
        + ", ".join(
            f"Q{number}"
            for number in sorted(question_numbers)
        )
    )

    return [
        (global_number, item)
        for global_number, item in enumerate(
            evidence,
            start=1,
        )
        if item.research_question_number in question_numbers
    ]


def _get_questions_for_question_tree(
    root_question_number: int,
    research_questions: list[ResearchQuestion],
) -> list[tuple[int, ResearchQuestion]]:
    """Return question objects and IDs for one question tree."""

    question_numbers = _get_descendant_question_numbers(
        root_question_number=root_question_number,
        research_questions=research_questions,
    )

    return [
        (index, question)
        for index, question in enumerate(
            research_questions,
            start=1,
        )
        if index in question_numbers
    ]


def _evaluate_question(
    research_question: ResearchQuestion,
    evidence: list[Evidence],
    global_evidence_numbers: list[int],
    question_tree: list[tuple[int, ResearchQuestion]],
) -> CoverageAssessment:
    """Evaluate whether evidence answers one original question."""

    question_tree_objects = [
        question
        for _, question in question_tree
    ]

    question_tokens = _combined_question_tokens(
        question_tree_objects
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
                (score, global_number, item)
            )

    scored.sort(
        key=lambda value: (
            -value[0],
            value[1],
        )
    )

    candidates = [
        (global_number, item)
        for _, global_number, item in scored[
            :MAX_EVIDENCE_PER_QUESTION
        ]
    ]

    print(f"\n  Evaluating: {research_question.question}")
    print(
        "  Question tree: "
        + ", ".join(
            f"Q{number}"
            for number, _ in question_tree
        )
    )
    print(f"  Verified evidence in question tree: {len(evidence)}")
    print(f"  Candidate evidence: {len(candidates)}")

    if not candidates:
        return CoverageAssessment(
            covered=False,
            evidence_numbers=[],
            reason=(
                "No relevant verified evidence was available "
                "to answer this research question."
            ),
        )

    candidate_numbers = {
        number
        for number, _ in candidates
    }

    try:
        response = chain.invoke(
            {
                "question": research_question.question,
                "research_question": research_question.question,
                "question_tree": _format_question_tree(
                    question_tree
                ),
                "evidence": _format_evidence(candidates),
            }
        )
    except Exception as exc:
        print("  Coverage evaluation failed:")
        print(f"  {exc}")

        return CoverageAssessment(
            covered=False,
            evidence_numbers=[],
            reason=(
                "Coverage evaluation failed; additional "
                "verified evidence is required."
            ),
        )

    valid_numbers = _normalize_evidence_numbers(
        numbers=response.evidence_numbers,
        candidate_numbers=candidate_numbers,
    )

    covered = response.covered and bool(valid_numbers)

    reason = response.reason.strip() or (
        "The supplied verified evidence was judged sufficient."
        if covered
        else "The supplied verified evidence was judged insufficient."
    )

    if not covered:
        valid_numbers = []

    return CoverageAssessment(
        covered=covered,
        evidence_numbers=valid_numbers,
        reason=reason,
    )


def _serialize_assessment(
    question_number: int,
    assessment: CoverageAssessment,
) -> dict:
    """Convert an assessment into the graph-state representation."""

    return {
        "research_question_number": question_number,
        "covered": assessment.covered,
        "evidence_numbers": list(assessment.evidence_numbers),
        "reason": assessment.reason,
    }


def research_sufficiency_node(
    state: ResearchState,
) -> dict:
    """Evaluate sufficiency and persist every root assessment."""

    print("\n[Node] research_sufficiency")

    research_questions = state["research_questions"]
    evidence = state["evidence"]

    print(f"  Planned research questions: {len(research_questions)}")
    print(f"  Verified evidence available: {len(evidence)}")

    if not research_questions:
        print("  No research questions available.")

        return {
            **state,
            "coverage_assessments": [],
            "research_sufficient": False,
            "research_gaps": [
                "No research plan was generated."
            ],
            "research_decision_reason": (
                "Research sufficiency cannot be evaluated "
                "without a research plan."
            ),
        }

    root_questions = [
        (index, question)
        for index, question in enumerate(
            research_questions,
            start=1,
        )
        if question.parent_question_number is None
    ]

    print(f"  Original research questions: {len(root_questions)}")
    print(
        "  Adaptive follow-up questions: "
        f"{len(research_questions) - len(root_questions)}"
    )

    # Handle the no-evidence case while still recording a
    # sufficiency assessment for every original question.
    if not evidence:
        print("  No verified evidence available.")

        assessments = []
        gaps = []

        for question_number, question in root_questions:
            reason = (
                "No verified evidence has been collected "
                "for this research question."
            )

            assessment = CoverageAssessment(
                covered=False,
                evidence_numbers=[],
                reason=reason,
            )

            assessments.append(
                _serialize_assessment(
                    question_number,
                    assessment,
                )
            )

            gaps.append(
                f"{question.question} — missing: {reason}"
            )

        return {
            **state,
            "coverage_assessments": assessments,
            "research_sufficient": False,
            "research_gaps": gaps,
            "research_decision_reason": (
                "No verified evidence has been collected "
                "for the research objective."
            ),
        }

    # Evaluate every original question independently.
    assessment_records = []

    for position, (
        question_number,
        research_question,
    ) in enumerate(root_questions, start=1):
        print(
            f"\n  Research requirement "
            f"{position}/{len(root_questions)}"
        )
        print(f"  Root question number: Q{question_number}")

        evidence_with_numbers = _get_evidence_for_question_tree(
            root_question_number=question_number,
            research_questions=research_questions,
            evidence=evidence,
        )

        question_tree = _get_questions_for_question_tree(
            root_question_number=question_number,
            research_questions=research_questions,
        )

        owned_global_numbers = [
            global_number
            for global_number, _ in evidence_with_numbers
        ]

        owned_evidence = [
            item
            for _, item in evidence_with_numbers
        ]

        print(
            "  Questions in tree: "
            + ", ".join(
                f"Q{number}"
                for number, _ in question_tree
            )
        )
        print(f"  Evidence in question tree: {len(owned_evidence)}")

        assessment = _evaluate_question(
            research_question=research_question,
            evidence=owned_evidence,
            global_evidence_numbers=owned_global_numbers,
            question_tree=question_tree,
        )

        assessment_records.append(
            (
                question_number,
                research_question,
                assessment,
            )
        )

    # Persist assessments separately from human-readable gaps.
    coverage_assessments = [
        _serialize_assessment(question_number, assessment)
        for question_number, _, assessment in assessment_records
    ]

    # Build gaps from the same decisions.
    research_gaps = []

    for _, question, assessment in assessment_records:
        if assessment.covered:
            continue

        reason = assessment.reason.strip() or (
            "Additional verified evidence is required."
        )

        gap = f"{question.question} — missing: {reason}"

        if gap not in research_gaps:
            research_gaps.append(gap)

    research_sufficient = not research_gaps

    if research_sufficient:
        reason = (
            "Every original research question is sufficiently "
            "supported by verified evidence, including evidence "
            "collected through adaptive follow-up research."
        )
    else:
        reason = (
            f"{len(research_gaps)} original research question(s) "
            "still require additional verified evidence."
        )

    print(f"\n  Research sufficiency: {research_sufficient}")
    print("  Coverage assessment:")

    for question_number, _, assessment in assessment_records:
        status = "COVERED" if assessment.covered else "INSUFFICIENT"

        print(f"    [Q{question_number}] {status}")
        print(f"        Reason: {assessment.reason}")

        if assessment.evidence_numbers:
            print(
                "        Evidence: "
                + ", ".join(
                    str(number)
                    for number in assessment.evidence_numbers
                )
            )

    print("  Research gaps:")

    if not research_gaps:
        print("    None")
    else:
        for gap in research_gaps:
            print(f"    - {gap}")

    return {
        **state,
        "coverage_assessments": coverage_assessments,
        "research_sufficient": research_sufficient,
        "research_gaps": research_gaps,
        "research_decision_reason": reason,
    }
