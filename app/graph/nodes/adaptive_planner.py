"""
Adaptive research planner for the Deep Research Agent.

Creates focused follow-up research questions when the current
evidence is insufficient to answer one or more original
research questions.

Validated follow-up questions are stored separately in
proposed_research_questions. They are not added to the active
research plan until the human-review node approves them.
"""

import os
import re

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from app.schemas.research import ResearchQuestion


load_dotenv()


MODEL_NAME = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5-coder:7b",
)

BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

MAX_FOLLOW_UP_QUESTIONS = 3


class AdaptiveFollowUpQuestion(BaseModel):
    """A follow-up question targeting one specific research gap."""

    question: str = Field(
        ...,
        description=(
            "A focused research question that investigates "
            "missing information from an unresolved research gap."
        ),
    )

    search_queries: list[str] = Field(
        ...,
        description=(
            "Focused web search queries for the follow-up question."
        ),
    )

    parent_question_number: int = Field(
        ...,
        description=(
            "The 1-based number of the original research question "
            "whose unresolved gap this follow-up addresses."
        ),
    )


class AdaptiveResearchOutput(BaseModel):
    """Structured output from the adaptive planner."""

    research_questions: list[AdaptiveFollowUpQuestion] = Field(
        default_factory=list,
        description="Focused follow-up research questions.",
    )


class FollowUpValidationOutput(BaseModel):
    """LLM judgment of whether a proposed follow-up is useful."""

    valid: bool = Field(
        ...,
        description=(
            "Whether the follow-up is focused, relevant, "
            "and useful for resolving the research gap."
        ),
    )

    reason: str = Field(
        ...,
        description="Brief explanation of the validation decision.",
    )


model = ChatOllama(
    model=MODEL_NAME,
    base_url=BASE_URL,
    temperature=0,
)

structured_model = model.with_structured_output(
    AdaptiveResearchOutput
)

validation_model = model.with_structured_output(
    FollowUpValidationOutput
)


planning_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the adaptive planning component of a deep research agent.

Create focused follow-up questions to investigate unresolved
research gaps.

Rules:
1. Every follow-up must address a specific unresolved gap.
2. Every follow-up must belong to one original parent question.
3. The parent number must refer to the original research plan.
4. Do not create follow-ups for sufficiently covered questions.
5. Do not repeat, rephrase, or broaden existing questions.
6. Do not introduce unrelated topics.
7. Do not answer the questions yourself.
8. Search queries must be specific and independently useful.
9. Prefer a few focused questions over many broad questions.
10. Return no question when a gap cannot be investigated
    through a specific, independently searchable question.
""",
        ),
        (
            "human",
            """
Original research questions:

{research_questions}

Unresolved research gaps:

{research_gaps}

Maximum follow-up questions:

{max_questions}

Create follow-up questions for the highest-priority gaps.

Each question must:
- identify the missing information;
- be narrower than its parent question;
- preserve the parent's scope;
- identify the correct original parent question number;
- provide focused search queries;
- avoid repeating existing questions.

Return no follow-up if no useful question can be generated.
""",
        ),
    ]
)


validation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict validator for an adaptive research planner.

A candidate is valid only if it:
1. Belongs to the assigned original parent question.
2. Directly addresses the unresolved gap.
3. Is narrower than the parent question.
4. Investigates missing information.
5. Is not a duplicate or rewording of an existing question.
6. Introduces no unrelated topic.
7. Is independently researchable.
8. Could materially help resolve the gap.

Reject a candidate if any condition fails.
Judge only from the supplied information.
""",
        ),
        (
            "human",
            """
Original parent research question:

{parent_question}

Unresolved research gap:

{gap}

Proposed follow-up question:

{candidate_question}

Determine whether the candidate is valid.
""",
        ),
    ]
)


planning_chain = planning_prompt | structured_model
validation_chain = validation_prompt | validation_model


def _normalize_text(text: str) -> str:
    """Normalize text for deterministic comparison."""

    return re.sub(
        r"\s+",
        " ",
        text.strip().lower(),
    )


def _tokenize(text: str) -> set[str]:
    """Return normalized content tokens."""

    return {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            _normalize_text(text),
        )
        if len(token) > 2
    }


def _token_similarity(first: str, second: str) -> float:
    """Calculate Jaccard similarity between two texts."""

    first_tokens = _tokenize(first)
    second_tokens = _tokenize(second)

    if not first_tokens or not second_tokens:
        return 0.0

    return len(first_tokens & second_tokens) / len(
        first_tokens | second_tokens
    )


def _is_duplicate_question(
    candidate_question: str,
    existing_questions: list[ResearchQuestion],
) -> bool:
    """Reject duplicate or near-duplicate questions."""

    normalized_candidate = _normalize_text(candidate_question)

    for existing in existing_questions:
        normalized_existing = _normalize_text(
            existing.question
        )

        if normalized_candidate == normalized_existing:
            return True

        if _token_similarity(
            candidate_question,
            existing.question,
        ) >= 0.80:
            return True

    return False


def _is_question_too_broad(
    candidate_question: str,
    parent_question: str,
) -> bool:
    """Detect candidates that effectively restate the parent."""

    candidate_tokens = _tokenize(candidate_question)
    parent_tokens = _tokenize(parent_question)

    if not candidate_tokens or not parent_tokens:
        return True

    overlap = candidate_tokens & parent_tokens

    parent_coverage = len(overlap) / len(parent_tokens)
    candidate_coverage = len(overlap) / len(candidate_tokens)

    return (
        parent_coverage >= 0.80
        and candidate_coverage >= 0.65
    )


def _extract_gap_parent_number(
    gap: str,
    research_questions: list[ResearchQuestion],
) -> int | None:
    """
    Find the original question associated with a gap.

    Supports the existing '— missing:' gap format.
    """

    gap_question = gap.split(
        "— missing:",
        1,
    )[0].strip()

    normalized_gap = _normalize_text(gap_question)

    for index, question in enumerate(
        research_questions,
        start=1,
    ):
        if _normalize_text(question.question) == normalized_gap:
            return index

    return None


def _gap_priority_score(gap: str) -> int:
    """Assign a deterministic priority score to a gap."""

    normalized_gap = _normalize_text(gap)
    score = 0

    high_priority_terms = (
        "best",
        "compare",
        "comparison",
        "versus",
        "vs",
        "ranking",
        "rank",
        "which",
        "changed",
        "change",
        "historical",
        "history",
        "over time",
        "trend",
    )

    medium_priority_terms = (
        "requirement",
        "requirements",
        "performance",
        "difference",
        "impact",
        "effect",
        "limitation",
        "limiting",
        "criteria",
        "factor",
        "factors",
    )

    score += sum(
        3
        for term in high_priority_terms
        if term in normalized_gap
    )

    score += sum(
        1
        for term in medium_priority_terms
        if term in normalized_gap
    )

    return score


def _prioritize_gaps(
    gaps: list[str],
    research_questions: list[ResearchQuestion],
) -> list[tuple[str, int | None, int]]:
    """Order gaps by priority, preserving original order for ties."""

    prioritized = []

    for original_index, gap in enumerate(gaps):
        parent_number = _extract_gap_parent_number(
            gap=gap,
            research_questions=research_questions,
        )

        prioritized.append(
            (
                gap,
                parent_number,
                _gap_priority_score(gap),
                original_index,
            )
        )

    prioritized.sort(
        key=lambda item: (-item[2], item[3])
    )

    return [
        (gap, parent_number, score)
        for gap, parent_number, score, _ in prioritized
    ]


def _valid_gap_parent_numbers(
    gaps: list[str],
    research_questions: list[ResearchQuestion],
) -> set[int]:
    """Return original question numbers that have unresolved gaps."""

    return {
        parent_number
        for gap in gaps
        if (
            parent_number := _extract_gap_parent_number(
                gap,
                research_questions,
            )
        ) is not None
    }


def _format_research_questions(
    research_questions: list[ResearchQuestion],
) -> str:
    """Format the current plan for the LLM."""

    if not research_questions:
        return "No research questions."

    lines = []

    for index, question in enumerate(
        research_questions,
        start=1,
    ):
        relationship = (
            "original"
            if question.parent_question_number is None
            else f"follow-up to Q{question.parent_question_number}"
        )

        lines.append(
            f"Q{index} [{relationship}]: {question.question}"
        )

    return "\n".join(lines)


def _format_research_gaps(
    gaps: list[str],
    research_questions: list[ResearchQuestion],
) -> str:
    """Format unresolved gaps in priority order."""

    if not gaps:
        return "No research gaps."

    prioritized_gaps = _prioritize_gaps(
        gaps,
        research_questions,
    )

    lines = []

    for index, (
        gap,
        parent_number,
        priority_score,
    ) in enumerate(
        prioritized_gaps,
        start=1,
    ):
        parent_label = (
            f"Q{parent_number}"
            if parent_number is not None
            else "UNKNOWN"
        )

        priority_label = (
            "HIGH"
            if priority_score >= 3
            else "MEDIUM"
            if priority_score >= 1
            else "LOW"
        )

        lines.append(
            f"GAP {index} [{priority_label}]\n"
            f"Parent research question: {parent_label}\n"
            f"{gap}"
        )

    return "\n\n".join(lines)


def _build_generic_fallback_question(
    gap: str,
    parent_question: str,
    parent_number: int,
) -> ResearchQuestion:
    """Build a domain-independent fallback question."""

    missing_part = gap.split(
        "— missing:",
        1,
    )[-1].strip()

    if not missing_part:
        missing_part = gap.strip()

    question = (
        "What specific evidence, factors, criteria, or mechanisms "
        "are needed to resolve the following missing aspect of "
        f"the research question: {missing_part}"
    )

    search_queries = [
        missing_part,
        f"{missing_part} evidence",
        f"{missing_part} research findings",
    ]

    return ResearchQuestion(
        question=question,
        search_queries=search_queries,
        parent_question_number=parent_number,
    )


def _repair_search_queries(
    candidate_question: str,
    queries: list[str],
) -> list[str]:
    """Clean and deduplicate search queries."""

    repaired = []

    for query in queries:
        if not query or not query.strip():
            continue

        cleaned = re.sub(
            r"\s+",
            " ",
            query.strip(),
        )

        if len(_tokenize(cleaned)) < 2:
            continue

        if cleaned not in repaired:
            repaired.append(cleaned)

    if not repaired:
        repaired = [
            candidate_question,
            f"{candidate_question} evidence",
        ]

    return repaired[:3]


def _validate_candidate(
    candidate: AdaptiveFollowUpQuestion,
    gap: str,
    parent_question: str,
) -> bool:
    """Apply deterministic checks and semantic LLM validation."""

    question = candidate.question.strip()

    if not question or not candidate.search_queries:
        return False

    if _is_question_too_broad(
        candidate_question=question,
        parent_question=parent_question,
    ):
        print(
            "  Rejected follow-up question: "
            "candidate is too broad or restates its parent."
        )
        return False

    try:
        validation = validation_chain.invoke(
            {
                "parent_question": parent_question,
                "gap": gap,
                "candidate_question": question,
            }
        )
    except Exception as exc:
        print(
            "  Follow-up semantic validation failed: "
            f"{exc}"
        )
        return False

    if not validation.valid:
        print(
            "  Rejected follow-up question: "
            "semantic validation failed."
        )
        print(f"    Reason: {validation.reason}")
        return False

    return True


def _build_fallback_questions(
    gaps: list[str],
    research_questions: list[ResearchQuestion],
    valid_parent_numbers: set[int],
    max_questions: int,
) -> list[ResearchQuestion]:
    """Build deterministic fallback candidates for unresolved gaps."""

    fallback_questions = []
    used_parent_numbers: set[int] = set()

    prioritized_gaps = _prioritize_gaps(
        gaps,
        research_questions,
    )

    for gap, parent_number, _ in prioritized_gaps:
        if parent_number is None:
            continue

        if parent_number not in valid_parent_numbers:
            continue

        if parent_number in used_parent_numbers:
            continue

        parent_question = research_questions[
            parent_number - 1
        ].question

        candidate = _build_generic_fallback_question(
            gap=gap,
            parent_question=parent_question,
            parent_number=parent_number,
        )

        if _is_duplicate_question(
            candidate_question=candidate.question,
            existing_questions=research_questions,
        ):
            continue

        fallback_questions.append(candidate)
        used_parent_numbers.add(parent_number)

        if len(fallback_questions) >= max_questions:
            break

    return fallback_questions


def adaptive_planner_node(state) -> dict:
    """
    Generate and validate proposed follow-up questions.

    The active research plan is deliberately left unchanged.
    The human-review node decides whether proposals are accepted.
    """

    gaps = state.get("research_gaps", [])
    research_questions = state.get("research_questions", [])

    remaining_capacity = (
        state.get("max_total_research_questions", 10)
        - len(research_questions)
    )

    max_questions = min(
        MAX_FOLLOW_UP_QUESTIONS,
        max(0, remaining_capacity),
    )

    print("\n[Node] adaptive_planner")
    print(f"  Research gaps identified: {len(gaps)}")

    prioritized_gaps = _prioritize_gaps(
        gaps,
        research_questions,
    )

    for index, (
        gap,
        parent_number,
        priority_score,
    ) in enumerate(
        prioritized_gaps,
        start=1,
    ):
        priority_label = (
            "HIGH"
            if priority_score >= 3
            else "MEDIUM"
            if priority_score >= 1
            else "LOW"
        )

        parent_label = (
            f"Q{parent_number}"
            if parent_number is not None
            else "UNKNOWN"
        )

        print(
            f"    [{index}] {priority_label} "
            f"({parent_label}) {gap}"
        )

    def finish_without_proposals() -> dict:
        return {
            "proposed_research_questions": [],
            "adaptive_review_decision": None,
            "research_round": state.get("research_round", 0) + 1,
        }

    if not gaps:
        print("  No research gaps remain.")
        return finish_without_proposals()

    if max_questions <= 0:
        print("  No remaining capacity for follow-up questions.")
        return finish_without_proposals()

    formatted_questions = _format_research_questions(
        research_questions
    )

    formatted_gaps = _format_research_gaps(
        gaps,
        research_questions,
    )

    try:
        response = planning_chain.invoke(
            {
                "research_questions": formatted_questions,
                "research_gaps": formatted_gaps,
                "max_questions": max_questions,
            }
        )
        candidates = response.research_questions
    except Exception as exc:
        print(f"  Adaptive planner LLM error: {exc}")
        candidates = []

    valid_parent_numbers = _valid_gap_parent_numbers(
        gaps,
        research_questions,
    )

    follow_up_candidates = []

    for candidate in candidates:
        question = candidate.question.strip()
        parent_number = candidate.parent_question_number

        if (
            parent_number not in valid_parent_numbers
            or parent_number < 1
            or parent_number > len(research_questions)
        ):
            print(
                "  Rejected follow-up question: "
                f"invalid parent question number Q{parent_number}."
            )
            continue

        parent_question = research_questions[
            parent_number - 1
        ].question

        gap_for_parent = next(
            (
                gap
                for gap, gap_parent_number, _ in prioritized_gaps
                if gap_parent_number == parent_number
            ),
            "",
        )

        if not gap_for_parent:
            print(
                "  Rejected follow-up question: "
                "no matching unresolved gap."
            )
            continue

        if _is_duplicate_question(
            candidate_question=question,
            existing_questions=research_questions,
        ):
            print(
                "  Rejected follow-up question: "
                "duplicate or near-duplicate."
            )
            print(f"    Question: {question}")
            continue

        repaired_queries = _repair_search_queries(
            candidate_question=question,
            queries=candidate.search_queries,
        )

        candidate.search_queries = repaired_queries

        if not _validate_candidate(
            candidate=candidate,
            gap=gap_for_parent,
            parent_question=parent_question,
        ):
            continue

        priority = next(
            (
                score
                for _, gap_parent_number, score in prioritized_gaps
                if gap_parent_number == parent_number
            ),
            0,
        )

        follow_up_candidates.append(
            (
                priority,
                candidate,
                question,
                repaired_queries,
            )
        )

    follow_up_candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    follow_up_questions = []
    seen_parent_numbers: set[int] = set()

    for _, candidate, question, search_queries in follow_up_candidates:
        parent_number = candidate.parent_question_number

        # Keep at most one follow-up for each original parent
        # in a single adaptive planning round.
        if parent_number in seen_parent_numbers:
            continue

        follow_up_questions.append(
            ResearchQuestion(
                question=question,
                search_queries=search_queries,
                parent_question_number=parent_number,
            )
        )

        seen_parent_numbers.add(parent_number)

        if len(follow_up_questions) >= max_questions:
            break

    if not follow_up_questions:
        print(
            "  Qwen produced no usable follow-ups. "
            "Trying domain-independent fallback questions."
        )

        fallback_candidates = _build_fallback_questions(
            gaps=gaps,
            research_questions=research_questions,
            valid_parent_numbers=valid_parent_numbers,
            max_questions=max_questions,
        )

        for fallback in fallback_candidates:
            parent_number = fallback.parent_question_number

            if parent_number is None:
                continue

            parent_question = research_questions[
                parent_number - 1
            ].question

            gap_for_parent = next(
                (
                    gap
                    for gap, gap_parent_number, _ in prioritized_gaps
                    if gap_parent_number == parent_number
                ),
                "",
            )

            candidate = AdaptiveFollowUpQuestion(
                question=fallback.question,
                search_queries=fallback.search_queries,
                parent_question_number=parent_number,
            )

            if _validate_candidate(
                candidate=candidate,
                gap=gap_for_parent,
                parent_question=parent_question,
            ):
                follow_up_questions.append(fallback)

            if len(follow_up_questions) >= max_questions:
                break

    if not follow_up_questions:
        print("  No valid follow-up questions could be generated.")
        return finish_without_proposals()

    print(
        f"  Generated {len(follow_up_questions)} "
        "proposed follow-up question(s)."
    )

    start_number = len(research_questions) + 1

    for offset, question in enumerate(follow_up_questions):
        print(
            f"    Proposed Q{start_number + offset} "
            f"(parent Q{question.parent_question_number}): "
            f"{question.question}"
        )

    # CRITICAL: do not append proposals to research_questions.
    # The adaptive-review node will make that decision.
    return {
        "proposed_research_questions": follow_up_questions,
        "adaptive_review_decision": None,
        "research_round": state.get("research_round", 0) + 1,
        "research_complete": False,
    }
