"""
Adaptive research planner for the Deep Research Agent.

Creates focused follow-up research questions when the current
evidence is insufficient to answer one or more original
research questions.

The planner is intentionally domain-independent. It must infer
the missing research dimension from the unresolved gap rather
than relying on topic-specific rules.
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
    """
    A follow-up question created to resolve one specific
    research gap.
    """

    question: str = Field(
        ...,
        description=(
            "A specific, narrower research question that "
            "directly investigates missing information from "
            "one unresolved research gap."
        ),
    )

    search_queries: list[str] = Field(
        ...,
        description=(
            "Focused search queries that directly investigate "
            "the follow-up question."
        ),
    )

    parent_question_number: int = Field(
        ...,
        description=(
            "The 1-based number of the original research "
            "question whose unresolved gap this follow-up "
            "addresses."
        ),
    )


class AdaptiveResearchOutput(BaseModel):
    """
    Structured output produced by the adaptive planner.
    """

    research_questions: list[AdaptiveFollowUpQuestion] = Field(
        default_factory=list,
        description=(
            "Focused follow-up research questions needed to "
            "resolve the identified research gaps."
        ),
    )


class FollowUpValidationOutput(BaseModel):
    """
    Structured judgment of whether a proposed follow-up
    question is genuinely useful.
    """

    valid: bool = Field(
        ...,
        description=(
            "Whether the follow-up question is sufficiently "
            "focused, relevant, and useful for resolving the gap."
        ),
    )

    reason: str = Field(
        ...,
        description=(
            "Brief explanation for the validation decision."
        ),
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

The research agent has already investigated several original
research questions.

Some original questions remain insufficiently supported.

Your job is to create focused follow-up research questions
that investigate the missing information.

The planner must work for arbitrary research domains.

STRICT RULES:

1. Every follow-up question must directly address one
   unresolved research gap.

2. Every follow-up question must be a genuine sub-question
   of its parent research question.

3. A follow-up question must be narrower than its parent
   question.

4. A follow-up question must investigate missing information,
   not merely repeat or rephrase the parent question.

5. A follow-up question must investigate missing information,
   not merely repeat or rephrase the research gap.

6. Do not answer the question yourself.

7. Do not introduce a new topic that is not required by the
   parent question or its unresolved gap.

8. Do not broaden the research scope.

9. Preserve the intended meaning and scope of the parent
   question.

10. Each follow-up must have exactly one valid original
    parent question.

11. The parent_question_number must refer to an ORIGINAL
    research question, not another follow-up question.

12. Do not create follow-ups for sufficiently covered
    questions.

13. Search queries must be directly useful for investigating
    the follow-up question.

14. Search queries should contain concrete concepts from the
    follow-up question rather than vague generic wording.

15. Prefer a small number of highly focused follow-ups over
    many broad questions.

16. If a gap can be resolved by investigating multiple
    distinct dimensions, create separate focused questions
    only when the dimensions are genuinely independent.

17. Do not manufacture missing dimensions that are not implied
    by the unresolved gap.

18. Do not use outside knowledge to invent a research scope.

19. A follow-up that is substantially equivalent to an
    existing research question is invalid.

20. A follow-up that merely changes the wording of an existing
    question is invalid.

21. A follow-up that asks for "more information" without
    identifying what information is missing is invalid.

22. Every follow-up must be independently searchable on the
    web.

23. The final follow-up questions should make measurable
    progress toward resolving the unresolved gap.
""",
        ),
        (
            "human",
            """
Original research questions:

{research_questions}


Identified unresolved research gaps:

{research_gaps}


Maximum follow-up questions allowed:

{max_questions}


Create focused follow-up research questions for the highest
priority unresolved gaps.

For every follow-up:

- identify the exact missing information
- make the question narrower than its parent
- preserve the parent's scope
- assign the correct original parent question number
- provide focused search queries
- avoid repeating existing questions
- avoid restating the gap as a question

Return no follow-up when a gap cannot be converted into a
specific, independently researchable sub-question.
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

Determine whether a proposed follow-up research question is
valid.

A valid follow-up must satisfy ALL of these conditions:

1. It belongs to the assigned original parent question.

2. It directly addresses the unresolved research gap.

3. It is narrower than the parent question.

4. It investigates missing information rather than repeating
   information that has already been requested.

5. It is not merely a rewording of the parent question.

6. It is not merely a rewording of the research gap.

7. It does not introduce an unrelated topic.

8. It is independently researchable.

9. It can provide evidence that would materially help resolve
   the gap.

Reject the candidate if any condition fails.

Judge only from the supplied parent question, gap, and
candidate. Do not use outside knowledge.
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


Determine whether this proposed follow-up is valid.
""",
        ),
    ]
)


planning_chain = planning_prompt | structured_model
validation_chain = validation_prompt | validation_model


def _normalize_text(
    text: str,
) -> str:
    """
    Normalize text for deterministic comparison.
    """

    return re.sub(
        r"\s+",
        " ",
        text.strip().lower(),
    )


def _tokenize(
    text: str,
) -> set[str]:
    """
    Return normalized content tokens.
    """

    normalized = _normalize_text(text)

    return {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            normalized,
        )
        if len(token) > 2
    }


def _token_similarity(
    first: str,
    second: str,
) -> float:
    """
    Calculate Jaccard similarity between two texts.
    """

    first_tokens = _tokenize(first)
    second_tokens = _tokenize(second)

    if not first_tokens or not second_tokens:
        return 0.0

    intersection = first_tokens & second_tokens
    union = first_tokens | second_tokens

    return len(intersection) / len(union)


def _is_duplicate_question(
    candidate_question: str,
    existing_questions: list[ResearchQuestion],
) -> bool:
    """
    Reject questions that duplicate or almost duplicate
    previously investigated questions.
    """

    normalized_candidate = _normalize_text(
        candidate_question
    )

    for existing in existing_questions:

        normalized_existing = _normalize_text(
            existing.question
        )

        if normalized_candidate == normalized_existing:
            return True

        similarity = _token_similarity(
            candidate_question,
            existing.question,
        )

        if similarity >= 0.80:
            return True

    return False


def _is_question_too_broad(
    candidate_question: str,
    parent_question: str,
) -> bool:
    """
    Detect candidates that are effectively the same scope
    as the parent question.
    """

    candidate_tokens = _tokenize(
        candidate_question
    )

    parent_tokens = _tokenize(
        parent_question
    )

    if not candidate_tokens or not parent_tokens:
        return True

    overlap = (
        candidate_tokens & parent_tokens
    )

    parent_coverage = (
        len(overlap) / len(parent_tokens)
    )

    candidate_coverage = (
        len(overlap) / len(candidate_tokens)
    )

    # If the candidate contains almost all of the parent's
    # concepts while adding very little new specificity,
    # it is probably a restatement.
    if (
        parent_coverage >= 0.80
        and candidate_coverage >= 0.65
    ):
        return True

    return False


def _extract_gap_parent_number(
    gap: str,
    research_questions: list[ResearchQuestion],
) -> int | None:
    """
    Determine which original research question owns a gap.

    The gap format is expected to contain the original
    research question followed by '-- missing:'.
    """

    gap_question = gap.split(
        "— missing:",
        1,
    )[0].strip()

    normalized_gap = _normalize_text(
        gap_question
    )

    for index, question in enumerate(
        research_questions,
        start=1,
    ):

        if _normalize_text(
            question.question
        ) == normalized_gap:

            return index

    return None


def _gap_priority_score(
    gap: str,
) -> int:
    """
    Assign a deterministic priority score to an unresolved gap.

    The score is based only on the wording of the gap.
    """

    normalized_gap = _normalize_text(
        gap
    )

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

    for term in high_priority_terms:

        if term in normalized_gap:
            score += 3

    for term in medium_priority_terms:

        if term in normalized_gap:
            score += 1

    return score


def _prioritize_gaps(
    gaps: list[str],
    research_questions: list[ResearchQuestion],
) -> list[tuple[str, int | None, int]]:
    """
    Order unresolved gaps by research importance.
    """

    prioritized = []

    for original_index, gap in enumerate(
        gaps
    ):

        parent_number = _extract_gap_parent_number(
            gap=gap,
            research_questions=research_questions,
        )

        score = _gap_priority_score(
            gap
        )

        prioritized.append(
            (
                gap,
                parent_number,
                score,
                original_index,
            )
        )

    prioritized.sort(
        key=lambda item: (
            -item[2],
            item[3],
        )
    )

    return [
        (
            gap,
            parent_number,
            score,
        )
        for (
            gap,
            parent_number,
            score,
            _,
        ) in prioritized
    ]


def _valid_gap_parent_numbers(
    gaps: list[str],
    research_questions: list[ResearchQuestion],
) -> set[int]:
    """
    Return original question numbers that currently have
    unresolved gaps.
    """

    parent_numbers: set[int] = set()

    for gap in gaps:

        parent_number = _extract_gap_parent_number(
            gap=gap,
            research_questions=research_questions,
        )

        if parent_number is not None:
            parent_numbers.add(
                parent_number
            )

    return parent_numbers


def _format_research_questions(
    research_questions: list[ResearchQuestion],
) -> str:
    """
    Format all existing research questions.
    """

    if not research_questions:
        return "No research questions."

    lines = []

    for index, question in enumerate(
        research_questions,
        start=1,
    ):

        if question.parent_question_number is None:
            relationship = "original"
        else:
            relationship = (
                f"follow-up to Q"
                f"{question.parent_question_number}"
            )

        lines.append(
            f"Q{index} [{relationship}]: "
            f"{question.question}"
        )

    return "\n".join(lines)


def _format_research_gaps(
    gaps: list[str],
    research_questions: list[ResearchQuestion],
) -> str:
    """
    Format unresolved gaps in priority order.
    """

    if not gaps:
        return "No research gaps."

    prioritized_gaps = _prioritize_gaps(
        gaps=gaps,
        research_questions=research_questions,
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

        if priority_score >= 3:
            priority_label = "HIGH"

        elif priority_score >= 1:
            priority_label = "MEDIUM"

        else:
            priority_label = "LOW"

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
    """
    Build a domain-independent fallback question.

    This fallback deliberately does not assume anything about
    the research domain.
    """

    missing_part = gap.split(
        "— missing:",
        1,
    )[-1].strip()

    if not missing_part:
        missing_part = gap.strip()

    question = (
        "What specific evidence, factors, criteria, or mechanisms "
        f"are needed to resolve the following missing aspect of "
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
    """
    Clean and deduplicate candidate search queries.
    """

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
            repaired.append(
                cleaned
            )

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
    """
    Validate a candidate using both deterministic checks and
    a semantic LLM validator.
    """

    question = candidate.question.strip()

    if not question:
        return False

    if not candidate.search_queries:
        return False

    if _is_question_too_broad(
        candidate_question=question,
        parent_question=parent_question,
    ):
        print(
            "  Rejected follow-up question: "
            "candidate is too broad or restates the parent."
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

        print(
            f"    Reason: {validation.reason}"
        )

        return False

    return True


def _build_fallback_questions(
    gaps: list[str],
    research_questions: list[ResearchQuestion],
    valid_parent_numbers: set[int],
    max_questions: int,
) -> list[ResearchQuestion]:
    """
    Build deterministic domain-independent follow-ups when
    the LLM planner produces no usable questions.
    """

    fallback_questions: list[ResearchQuestion] = []

    prioritized_gaps = _prioritize_gaps(
        gaps=gaps,
        research_questions=research_questions,
    )

    used_parent_numbers: set[int] = set()

    for (
        gap,
        parent_number,
        _,
    ) in prioritized_gaps:

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

        fallback_questions.append(
            candidate
        )

        used_parent_numbers.add(
            parent_number
        )

        if len(fallback_questions) >= max_questions:
            break

    return fallback_questions


def adaptive_planner_node(
    state,
) -> dict:
    """
    Create follow-up research questions from unresolved gaps.
    """

    gaps = state.get(
        "research_gaps",
        [],
    )

    research_questions = state.get(
        "research_questions",
        [],
    )

    remaining_capacity = (
        state.get(
            "max_total_research_questions",
            10,
        )
        - len(research_questions)
    )

    max_questions = min(
        MAX_FOLLOW_UP_QUESTIONS,
        max(
            0,
            remaining_capacity,
        ),
    )

    print(
        "\n[Node] adaptive_planner"
    )

    print(
        f"  Research gaps identified: {len(gaps)}"
    )

    prioritized_gaps = _prioritize_gaps(
        gaps=gaps,
        research_questions=research_questions,
    )

    for index, (
        gap,
        parent_number,
        priority_score,
    ) in enumerate(
        prioritized_gaps,
        start=1,
    ):

        if priority_score >= 3:
            priority_label = "HIGH"

        elif priority_score >= 1:
            priority_label = "MEDIUM"

        else:
            priority_label = "LOW"

        parent_label = (
            f"Q{parent_number}"
            if parent_number is not None
            else "UNKNOWN"
        )

        print(
            f"    [{index}] "
            f"{priority_label} "
            f"({parent_label}) "
            f"{gap}"
        )

    if not gaps:

        print(
            "  No research gaps remain."
        )

        return {
            **state,
            "research_round": (
                state.get(
                    "research_round",
                    0,
                )
                + 1
            ),
        }

    if max_questions <= 0:

        print(
            "  No remaining capacity for follow-up questions."
        )

        return {
            **state,
            "research_round": (
                state.get(
                    "research_round",
                    0,
                )
                + 1
            ),
        }

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

        print(
            f"  Adaptive planner LLM error: {exc}"
        )

        candidates = []

    valid_parent_numbers = _valid_gap_parent_numbers(
        gaps=gaps,
        research_questions=research_questions,
    )

    follow_up_candidates = []

    for candidate in candidates:

        question = candidate.question.strip()

        parent_number = (
            candidate.parent_question_number
        )

        if parent_number not in valid_parent_numbers:

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
                for (
                    gap,
                    gap_parent_number,
                    _,
                ) in prioritized_gaps
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

            print(
                f"    Question: {question}"
            )

            continue

        search_queries = [
            query.strip()
            for query in candidate.search_queries
            if query and query.strip()
        ]

        repaired_queries = _repair_search_queries(
            candidate_question=question,
            queries=search_queries,
        )

        candidate.search_queries = repaired_queries

        if not _validate_candidate(
            candidate=candidate,
            gap=gap_for_parent,
            parent_question=parent_question,
        ):

            continue

        priority = 0

        for (
            _,
            gap_parent_number,
            score,
        ) in prioritized_gaps:

            if gap_parent_number == parent_number:
                priority = score
                break

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

    follow_up_questions: list[ResearchQuestion] = []

    seen_parent_numbers: set[int] = set()

    for (
        _,
        candidate,
        question,
        search_queries,
    ) in follow_up_candidates:

        parent_number = (
            candidate.parent_question_number
        )

        if parent_number in seen_parent_numbers:
            continue

        follow_up_questions.append(
            ResearchQuestion(
                question=question,
                search_queries=search_queries,
                parent_question_number=parent_number,
            )
        )

        seen_parent_numbers.add(
            parent_number
        )

        if len(follow_up_questions) >= max_questions:
            break

    if not follow_up_questions:

        print(
            "  Qwen did not produce usable follow-up questions."
        )

        print(
            "  Building domain-independent follow-up "
            "questions from the highest-priority gaps."
        )

        fallback_candidates = _build_fallback_questions(
            gaps=gaps,
            research_questions=research_questions,
            valid_parent_numbers=valid_parent_numbers,
            max_questions=max_questions,
        )

        for candidate in fallback_candidates:

            parent_number = (
                candidate.parent_question_number
            )

            parent_question = research_questions[
                parent_number - 1
            ].question

            gap_for_parent = next(
                (
                    gap
                    for (
                        gap,
                        gap_parent_number,
                        _,
                    ) in prioritized_gaps
                    if gap_parent_number == parent_number
                ),
                "",
            )

            if _validate_candidate(
                candidate=AdaptiveFollowUpQuestion(
                    question=candidate.question,
                    search_queries=candidate.search_queries,
                    parent_question_number=parent_number,
                ),
                gap=gap_for_parent,
                parent_question=parent_question,
            ):
                follow_up_questions.append(
                    candidate
                )

            if len(follow_up_questions) >= max_questions:
                break

    if not follow_up_questions:

        print(
            "  No valid follow-up research questions "
            "could be generated."
        )

        return {
            **state,
            "research_round": (
                state.get(
                    "research_round",
                    0,
                )
                + 1
            ),
        }

    start_number = len(
        research_questions
    ) + 1

    print(
        f"  Generated {len(follow_up_questions)} "
        "follow-up research question(s)."
    )

    for offset, question in enumerate(
        follow_up_questions,
    ):

        assigned_number = (
            start_number + offset
        )

        print(
            f"    Q{assigned_number} "
            f"(parent Q{question.parent_question_number}): "
            f"{question.question}"
        )

    updated_questions = (
        research_questions
        + follow_up_questions
    )

    return {
        **state,
        "research_questions": updated_questions,
        "current_question_index": len(
            research_questions
        ),
        "research_round": (
            state.get(
                "research_round",
                0,
            )
            + 1
        ),
        "research_complete": False,
    }
