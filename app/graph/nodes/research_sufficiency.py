"""
Research sufficiency evaluator for the Deep Research Agent.

Evaluates each original research question independently,
uses evidence from its follow-up descendants, and persists
validated coverage assessments in LangGraph state.

The evaluator is deliberately conservative: weak evidence
matching, invalid evidence references, and failed LLM calls
must never produce a successful coverage decision.
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

# An evidence item must match at least this proportion of the
# tokens in one relevant question or follow-up question.
MIN_TOKEN_OVERLAP = 0.12

MAX_REASON_LENGTH = 1000


class CoverageAssessment(BaseModel):
    """Sufficiency decision for one original research question."""

    covered: bool = Field(
        description=(
            "True only when verified evidence adequately "
            "answers the original research question."
        )
    )

    evidence_numbers: list[int] = Field(
        default_factory=list,
        description=(
            "Global 1-based evidence numbers directly supporting "
            "the sufficiency decision."
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

Determine whether the VERIFIED EVIDENCE adequately answers
ONE ORIGINAL RESEARCH QUESTION.

The evidence has already passed evidence verification.
That does not automatically mean it answers the question.

RULES:

1. Evaluate the complete original research question.
2. Use only the supplied question tree and verified evidence.
3. Read both each evidence claim and its supporting text.
4. Evaluate direct semantic relevance, not keyword overlap alone.
5. Do not assume that a claim is true merely because its
   wording resembles the question.
6. Every selected evidence item must directly support the
   answer or a necessary part of the answer.
7. A comparison question requires evidence sufficient to
   support the requested comparison, not merely descriptions
   of the compared subjects.
8. A question with multiple explicit dimensions requires
   evidence addressing those dimensions.
9. Evidence from follow-up descendants may contribute when
   it directly answers part of the original question.
10. Evidence belonging to another original question must not
    be used to satisfy this question.
11. Do not use outside knowledge or invent missing facts.
12. Be conservative when important evidence is missing.
13. Return only the global evidence numbers supplied in the
    candidate evidence. Never invent evidence numbers.
14. If evidence is insufficient, set covered=false and return
    an empty evidence_numbers list.
15. If covered=true, return every evidence number needed to
    justify that decision, and no irrelevant evidence numbers.
16. Explain significant missing information when coverage
    is incomplete.
17. A collection of related facts is not automatically a
    complete answer to the original question.

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

VERIFIED EVIDENCE CANDIDATES:

{evidence}

Decide whether the evidence adequately answers the original
research question.

For a sufficient answer, provide the global evidence numbers
that directly support the complete answer.

For an insufficient answer, return covered=false and an
empty evidence_numbers list. Explain what important evidence
is missing.
""",
        ),
    ]
)

chain = prompt | structured_model


def _tokenize(text: str) -> set[str]:
    """Convert text into lowercase alphanumeric tokens."""

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
    """Build tokens for one question and its search queries."""

    return _tokenize(
        " ".join(
            [
                question.question,
                *question.search_queries,
            ]
        )
    )


def _evidence_relevance_score(
    question_tokens: set[str],
    evidence: Evidence,
) -> float:
    """
    Calculate lexical relevance for candidate selection.

    This score only filters candidates. It does not establish
    semantic support or determine final coverage.
    """

    evidence_tokens = _tokenize(
        f"{evidence.claim} {evidence.supporting_text}"
    )

    if not question_tokens or not evidence_tokens:
        return 0.0

    overlap = len(question_tokens & evidence_tokens)

    return overlap / len(question_tokens)


def _best_question_relevance(
    question_tree: list[tuple[int, ResearchQuestion]],
    evidence: Evidence,
) -> float:
    """
    Score evidence against individual questions separately.

    Taking the best individual score avoids diluting relevance
    by combining tokens from many unrelated follow-up queries.
    """

    scores = [
        _evidence_relevance_score(
            question_tokens=_question_tokens(question),
            evidence=evidence,
        )
        for _, question in question_tree
    ]

    return max(scores, default=0.0)


def _format_evidence(
    evidence: list[tuple[int, Evidence]],
) -> str:
    """Format evidence while preserving its global identifiers."""

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
    """Format question objects and their actual parent lineage."""

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
    """Collect evidence owned by a root question or its descendants."""

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
    """Return question objects and identifiers for one question tree."""

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


def _normalize_evidence_numbers(
    numbers: list[int],
    candidate_numbers: set[int],
) -> list[int] | None:
    """
    Validate global evidence references.

    Return None if the model invents an evidence identifier.
    Invalid references are not silently ignored because they
    may indicate an unreliable coverage assessment.
    """

    if not numbers:
        return []

    normalized = []

    for number in numbers:
        if number not in candidate_numbers:
            return None

        if number not in normalized:
            normalized.append(number)

    return normalized


def _failed_assessment(
    reason: str,
) -> CoverageAssessment:
    """Create a conservative insufficient-coverage decision."""

    return CoverageAssessment(
        covered=False,
        evidence_numbers=[],
        reason=reason[:MAX_REASON_LENGTH],
    )


def _evaluate_question(
    research_question: ResearchQuestion,
    evidence: list[Evidence],
    global_evidence_numbers: list[int],
    question_tree: list[tuple[int, ResearchQuestion]],
) -> CoverageAssessment:
    """Evaluate whether verified evidence answers one root question."""

    scored = []

    for global_number, item in zip(
        global_evidence_numbers,
        evidence,
    ):
        score = _best_question_relevance(
            question_tree=question_tree,
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
        return _failed_assessment(
            "No sufficiently relevant verified evidence was "
            "available to answer this research question."
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

        return _failed_assessment(
            "Coverage evaluation failed. Additional verified "
            "evidence is required before coverage can be confirmed."
        )

    valid_numbers = _normalize_evidence_numbers(
        numbers=response.evidence_numbers,
        candidate_numbers=candidate_numbers,
    )

    if valid_numbers is None:
        print("  Coverage rejected: invalid evidence references.")

        return _failed_assessment(
            "The coverage evaluator returned evidence references "
            "that were not present in the supplied candidates."
        )

    if not response.covered:
        return _failed_assessment(
            response.reason.strip()
            or "The verified evidence does not adequately answer "
            "the research question."
        )

    if not valid_numbers:
        return _failed_assessment(
            "The evaluator marked the question covered but did "
            "not identify supporting evidence."
        )

    # Recheck every selected item against the question tree.
    # This is a deterministic relevance check, not a substitute
    # for the semantic judgment performed by the LLM.
    selected_evidence = {
        number: item
        for number, item in candidates
    }

    weak_references = [
        number
        for number in valid_numbers
        if _best_question_relevance(
            question_tree=question_tree,
            evidence=selected_evidence[number],
        ) < MIN_TOKEN_OVERLAP
    ]

    if weak_references:
        return _failed_assessment(
            "The selected evidence did not pass the minimum "
            "relevance check for this question."
        )

    reason = (
        response.reason.strip()
        or "The supplied verified evidence was judged sufficient."
    )

    return CoverageAssessment(
        covered=True,
        evidence_numbers=valid_numbers,
        reason=reason[:MAX_REASON_LENGTH],
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
    """Evaluate every original question and persist its assessment."""

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

    if not root_questions:
        return {
            **state,
            "coverage_assessments": [],
            "research_sufficient": False,
            "research_gaps": [
                "The research plan contains no original root questions."
            ],
            "research_decision_reason": (
                "Research sufficiency cannot be evaluated because "
                "the research plan has no root questions."
            ),
        }

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

    coverage_assessments = [
        _serialize_assessment(question_number, assessment)
        for question_number, _, assessment in assessment_records
    ]

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

    research_sufficient = (
        bool(root_questions)
        and all(
            assessment.covered
            for _, _, assessment in assessment_records
        )
    )

    if research_sufficient:
        reason = (
            "Every original research question was assessed as "
            "sufficiently supported by verified evidence, including "
            "relevant evidence from adaptive follow-up research."
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
