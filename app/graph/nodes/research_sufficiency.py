"""
Research sufficiency evaluator for the Deep Research Agent.

This node determines whether the collected verified evidence
is sufficient to answer the original research questions.

Original research questions own the final coverage decision.

Adaptive follow-up questions are descendants of their parent
question. Their verified evidence is therefore included when
evaluating the parent.

Example:

    Q2
     ├── Q6
     │    └── evidence for Q6
     │
     └── Q7
          └── evidence for Q7

When evaluating Q2, evidence from Q2, Q6, and Q7 is considered.

The evaluator deliberately avoids asking the LLM to invent a
large set of requirements. The original research question
itself defines the scope.

This prevents one research question from accidentally inheriting
requirements from another research question.
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
MIN_TOKEN_OVERLAP = 0.05


# ------------------------------------------------------------
# Structured output
# ------------------------------------------------------------


class CoverageAssessment(BaseModel):
    """
    Final coverage decision for one original research question.

    The LLM is intentionally not allowed to invent a separate
    requirement list. The original research question defines
    the scope.
    """

    covered: bool = Field(
        description=(
            "True only when the supplied verified evidence "
            "is sufficient to answer the original research "
            "question."
        )
    )

    evidence_numbers: list[int] = Field(
        default_factory=list,
        description=(
            "GLOBAL 1-based evidence numbers that directly "
            "support the coverage decision."
        ),
    )

    reason: str = Field(
        default="",
        description=(
            "Brief explanation of why the evidence is or is "
            "not sufficient to answer the original research "
            "question."
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
You are the research sufficiency evaluator of a deep
research agent.

Your job is to decide whether VERIFIED EVIDENCE is sufficient
to answer ONE ORIGINAL RESEARCH QUESTION.

You are NOT writing the final answer.

You must evaluate ONLY the supplied verified evidence.

IMPORTANT SCOPE RULES:

1. The ORIGINAL RESEARCH QUESTION defines the complete scope.

2. Do NOT create additional requirements that are not explicitly
   required by the original research question.

3. Do NOT import requirements from other research questions.

4. Do NOT use the overall user question to expand the scope
   beyond the current original research question.

5. Adaptive follow-up questions are evidence-gathering tasks.
   Their wording may be more specific than the original
   question, but their evidence may be used if it genuinely
   helps answer the original question.

6. Judge evidence semantically, not merely by keyword overlap.

7. Read BOTH the evidence CLAIM and SUPPORTING TEXT.

8. Evidence must directly support the answer.

9. Do not mark a question covered merely because the evidence
   discusses the same general topic.

10. If the question asks for a comparison, the evidence must
    provide enough information to actually make that comparison.

11. If the question asks how something changed over time, the
    evidence must provide enough information about the relevant
    change or time periods.

12. If the question asks which option/entity is best, the
    evidence must contain enough relevant information to support
    that comparison or conclusion.

13. For questions with multiple explicit dimensions, evidence
    must cover those dimensions sufficiently.

14. Do not require unrelated:
    - historical background
    - definitions
    - examples
    - recommendations
    - implementation details
    - limitations
    - future work

    unless the original research question explicitly asks for
    them or they are necessary to answer it.

15. Be conservative.

16. If important information is missing, set covered=false.

17. If the evidence is only partially relevant, set
    covered=false.

18. If evidence is weak, generic, indirect, or merely mentions
    the topic, set covered=false.

19. Do not use outside knowledge.

20. Do not infer facts that are not present in the supplied
    evidence.

21. evidence_numbers must contain only GLOBAL evidence numbers
    supplied in the input.

22. Do not invent evidence numbers.

23. If covered=true, provide the GLOBAL evidence numbers that
    directly support the decision.

24. If covered=false, evidence_numbers should normally be empty.

25. Keep the reason concise and specific.

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


Evaluate ONLY this original research question:

{research_question}

The question tree shows adaptive follow-up questions that were
created to gather additional evidence for this original question.

Evidence from those descendants may be used when it directly
helps answer the original question.

Do NOT import requirements from any other original research
question.

Do NOT invent a new research scope.

Decide whether the supplied verified evidence is sufficient
to answer the original research question.

If the question is fully and adequately supported:

- covered = true
- provide the GLOBAL evidence numbers that support it

If important information is missing:

- covered = false
- evidence_numbers should normally be empty
- explain what is still missing in the reason

Return only the requested structured output.
""",
        ),
    ]
)


chain = prompt | structured_model


# ------------------------------------------------------------
# Text helpers
# ------------------------------------------------------------


def _tokenize(
    text: str,
) -> set[str]:
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
    Build tokens from a research question and its search queries.
    """

    text = " ".join(
        [
            question.question,
            *question.search_queries,
        ]
    )

    return _tokenize(text)


def _combined_question_tokens(
    questions: list[ResearchQuestion],
) -> set[str]:
    """
    Build lexical tokens for the complete question tree.

    This is used ONLY for candidate selection.

    It does not determine whether evidence is sufficient.
    """

    tokens = set()

    for question in questions:

        tokens.update(
            _question_tokens(question)
        )

    return tokens


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
        len(
            question_tokens
            & evidence_tokens
        )
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

        return (
            "No relevant verified evidence candidates."
        )

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
    """
    Format the actual question-tree lineage.

    Example:

        Q4: How did X change?
        Q6: What evidence explains the 2023 change?
        Q7: What evidence explains the 2026 change?
    """

    if not question_tree:

        return "No question-tree information available."

    sections = []

    for number, question in question_tree:

        parent = question.parent_question_number

        if parent is None:

            relationship = "original question"

        else:

            relationship = (
                f"follow-up to Q{parent}"
            )

        sections.append(
            f"""
Q{number} ({relationship})

{question.question}
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

            normalized.append(
                number
            )

    return normalized


# ------------------------------------------------------------
# Research-question lineage
# ------------------------------------------------------------


def _get_descendant_question_numbers(
    root_question_number: int,
    research_questions: list[ResearchQuestion],
) -> set[int]:
    """
    Return the root question and every adaptive follow-up
    question descended from it.

    Descendants are resolved recursively.

    Example:

        Q2
         └── Q6
              └── Q9

    Evaluating Q2 therefore includes:

        Q2 + Q6 + Q9
    """

    descendants = {
        root_question_number
    }

    changed = True

    while changed:

        changed = False

        for index, question in enumerate(
            research_questions,
            start=1,
        ):

            if index in descendants:
                continue

            parent_number = (
                question.parent_question_number
            )

            if parent_number in descendants:

                descendants.add(
                    index
                )

                changed = True

    return descendants


def _get_evidence_for_question_tree(
    root_question_number: int,
    research_questions: list[ResearchQuestion],
    evidence: list[Evidence],
) -> list[tuple[int, Evidence]]:
    """
    Collect verified evidence belonging to the root research
    question or any of its adaptive descendants.

    Evidence keeps its original question number.
    """

    question_numbers = (
        _get_descendant_question_numbers(
            root_question_number=root_question_number,
            research_questions=research_questions,
        )
    )

    print(
        "  Evidence lineage questions: "
        + ", ".join(
            f"Q{number}"
            for number in sorted(
                question_numbers
            )
        )
    )

    return [
        (
            global_number,
            item,
        )
        for global_number, item in enumerate(
            evidence,
            start=1,
        )
        if item.research_question_number
        in question_numbers
    ]


def _get_questions_for_question_tree(
    root_question_number: int,
    research_questions: list[ResearchQuestion],
) -> list[tuple[int, ResearchQuestion]]:
    """
    Return the actual ResearchQuestion objects and their GLOBAL
    question numbers belonging to a question tree.
    """

    question_numbers = (
        _get_descendant_question_numbers(
            root_question_number=root_question_number,
            research_questions=research_questions,
        )
    )

    return [
        (
            index,
            question,
        )
        for index, question in enumerate(
            research_questions,
            start=1,
        )
        if index in question_numbers
    ]


# ------------------------------------------------------------
# Single-question evaluation
# ------------------------------------------------------------


def _evaluate_question(
    research_question: ResearchQuestion,
    evidence: list[Evidence],
    global_evidence_numbers: list[int],
    question_tree: list[tuple[int, ResearchQuestion]],
) -> CoverageAssessment:
    """
    Evaluate one original research question.

    Only the original question defines the scope.

    Evidence from adaptive descendants may support the answer,
    but descendants cannot introduce new requirements.
    """

    # --------------------------------------------------------
    # Candidate selection
    # --------------------------------------------------------

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
        "  Question tree: "
        + ", ".join(
            f"Q{number}"
            for number, _ in question_tree
        )
    )

    print(
        "  Verified evidence in question tree: "
        f"{len(evidence)}"
    )

    print(
        "  Candidate evidence: "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not candidates:

        return CoverageAssessment(
            covered=False,
            evidence_numbers=[],
            reason=(
                "No relevant verified evidence "
                "was available to answer this "
                "research question."
            ),
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
                "question": (
                    research_question.question
                ),
                "research_question": (
                    research_question.question
                ),
                "question_tree": (
                    _format_question_tree(
                        question_tree
                    )
                ),
                "evidence": (
                    _format_evidence(
                        candidates
                    )
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
            covered=False,
            evidence_numbers=[],
            reason=(
                "Coverage evaluation failed; "
                "additional verified evidence "
                "is required."
            ),
        )

    # --------------------------------------------------------
    # Normalize model output
    # --------------------------------------------------------

    valid_numbers = (
        _normalize_evidence_numbers(
            numbers=response.evidence_numbers,
            candidate_numbers=candidate_numbers,
        )
    )

    # --------------------------------------------------------
    # Final deterministic coverage rule
    # --------------------------------------------------------

    covered = (
        response.covered
        and bool(valid_numbers)
    )

    reason = (
        response.reason.strip()
        if response.reason
        else (
            "The supplied verified evidence "
            "was judged insufficient."
            if not covered
            else
            "The supplied verified evidence "
            "was judged sufficient."
        )
    )

    if not covered:

        valid_numbers = []

    return CoverageAssessment(
        covered=covered,
        evidence_numbers=valid_numbers,
        reason=reason,
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

    # --------------------------------------------------------
    # Identify original/root research questions
    # --------------------------------------------------------

    root_questions = [
        (
            index,
            question,
        )
        for index, question in enumerate(
            research_questions,
            start=1,
        )
        if question.parent_question_number is None
    ]

    print(
        "  Original research questions: "
        f"{len(root_questions)}"
    )

    print(
        "  Adaptive follow-up questions: "
        f"{len(research_questions) - len(root_questions)}"
    )

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not evidence:

        print(
            "  No verified evidence available."
        )

        gaps = [
            (
                f"{question.question} "
                f"— missing: "
                f"verified evidence required to answer "
                f"this research question"
            )
            for _, question in root_questions
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
    # Evaluate each ORIGINAL research question independently
    #
    # Follow-up questions are NOT independent final
    # requirements.
    #
    # Their evidence belongs to the parent question tree.
    # --------------------------------------------------------

    assessments = []

    for position, (
        question_number,
        research_question,
    ) in enumerate(
        root_questions,
        start=1,
    ):

        print(
            f"\n  Research requirement "
            f"{position}/{len(root_questions)}"
        )

        print(
            f"  Root question number: "
            f"Q{question_number}"
        )

        # ----------------------------------------------------
        # Question-tree ownership
        # ----------------------------------------------------

        evidence_with_numbers = (
            _get_evidence_for_question_tree(
                root_question_number=question_number,
                research_questions=research_questions,
                evidence=evidence,
            )
        )

        question_tree = (
            _get_questions_for_question_tree(
                root_question_number=question_number,
                research_questions=research_questions,
            )
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

        print(
            "  Evidence in question tree: "
            f"{len(owned_evidence)}"
        )

        assessment = _evaluate_question(
            research_question=research_question,
            evidence=owned_evidence,
            global_evidence_numbers=owned_global_numbers,
            question_tree=question_tree,
        )

        assessments.append(
            (
                question_number,
                research_question,
                assessment,
            )
        )

    # --------------------------------------------------------
    # Build research gaps
    # --------------------------------------------------------

    research_gaps = []

    for (
        question_number,
        question,
        assessment,
    ) in assessments:

        if assessment.covered:

            continue

        reason = assessment.reason.strip()

        if not reason:

            reason = (
                "Additional verified evidence "
                "is required to answer this "
                "research question."
            )

        gap = (
            f"{question.question} "
            f"— missing: "
            f"{reason}"
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
            "Every original research question is sufficiently "
            "supported by verified evidence, including evidence "
            "collected through adaptive follow-up research."
        )

    else:

        uncovered_count = len(
            research_gaps
        )

        reason = (
            f"{uncovered_count} original research question(s) "
            "still require additional verified evidence."
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

    for (
        question_number,
        question,
        assessment,
    ) in assessments:

        status = (
            "COVERED"
            if assessment.covered
            else "INSUFFICIENT"
        )

        print(
            f"    [Q{question_number}] "
            f"{status}"
        )

        print(
            f"        Reason: "
            f"{assessment.reason}"
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
