"""
Final synthesis node for the Deep Research Agent.

Builds a source-grounded report from verified evidence while
preserving research sufficiency decisions and unresolved gaps.
"""

import os
from collections import defaultdict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from app.graph.state import ResearchState
from app.schemas.research import (
    Evidence,
    ResearchCoverage,
    ResearchFinding,
    ResearchReport,
)


# ------------------------------------------------------------
# LLM CONFIGURATION
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# STRUCTURED OUTPUT SCHEMAS
# ------------------------------------------------------------

class QuestionSynthesisOutput(BaseModel):
    findings: list[ResearchFinding] = Field(
        default_factory=list
    )


class ReportNarrativeOutput(BaseModel):
    title: str
    summary: str


question_synthesis_model = model.with_structured_output(
    QuestionSynthesisOutput
)

report_narrative_model = model.with_structured_output(
    ReportNarrativeOutput
)


# ------------------------------------------------------------
# PROMPTS
# ------------------------------------------------------------

question_synthesis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a careful research analyst.

Produce concise, factual findings using ONLY the supplied
verified evidence.

Rules:
- Do not introduce facts absent from the evidence.
- Every finding must cite one or more supplied evidence numbers.
- The current synthesis is for ONE original research question.
- Use the exact research question number supplied by the user.
- Evidence numbers in the supplied evidence list are LOCAL numbers.
- Copy those local evidence numbers into evidence_numbers.
- Never invent evidence numbers or source URLs.
- Prefer distinct, informative findings over repetition.
- If evidence is insufficient, do not manufacture a conclusion.
- Return an empty findings list only when no defensible finding
  can be produced from the supplied evidence.
""",
        ),
        (
            "human",
            """
Original research question number: Q{question_number}

Original research question:

{question}

Verified evidence for this question:

{evidence}

Produce findings that directly answer the original research question.

For every finding:
- research_question_numbers must contain {question_number}.
- evidence_numbers must contain the LOCAL evidence numbers shown
  in the evidence list, such as 1, 2, or 3.
- Do not use global evidence numbers from another question.
""",
        ),
    ]
)

report_narrative_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are writing the final report for a deep research agent.

Write a clear title and concise summary based ONLY on the
provided validated findings, coverage, and research gaps.

Do not claim that all research questions were answered when
some questions are partially supported or unresolved.

Do not introduce new factual claims.

If there are no validated findings, state that clearly.
Do not describe sufficient evidence as proof that a question
is unresolved.
""",
        ),
        (
            "human",
            """
Validated findings:

{findings}

Research coverage:

{coverage}

Research gaps:

{gaps}

Write the final report title and summary.
""",
        ),
    ]
)


# ------------------------------------------------------------
# QUESTION HELPERS
# ------------------------------------------------------------

def _build_question_roots(
    state: ResearchState,
) -> list[tuple[int, object]]:
    """Return original research questions with 1-based numbers."""

    return [
        (index, question)
        for index, question in enumerate(
            state.get("research_questions", []),
            start=1,
        )
        if question.parent_question_number is None
    ]


def _format_question_evidence(
    question_number: int,
    evidence_items: list[Evidence],
) -> str:
    """Format evidence using local numbers for the LLM prompt."""

    lines = []

    for local_number, evidence in enumerate(
        evidence_items,
        start=1,
    ):
        lines.append(
            f"[E{local_number}] "
            f"Claim: {evidence.claim}\n"
            f"Source: {evidence.source_url}\n"
            f"Supporting text: {evidence.supporting_text}"
        )

    return "\n\n".join(lines)


def _format_research_gaps(
    research_gaps: list[str],
) -> str:
    if not research_gaps:
        return "No explicit research gaps were recorded."

    return "\n".join(
        f"- {gap}"
        for gap in research_gaps
    )


def _format_findings(
    findings: list[ResearchFinding],
) -> str:
    if not findings:
        return "No validated findings were produced."

    return "\n".join(
        f"- {finding.text} "
        f"(questions: {finding.research_question_numbers}; "
        f"evidence: {finding.evidence_numbers})"
        for finding in findings
    )


# ------------------------------------------------------------
# FINDING VALIDATION
# ------------------------------------------------------------

def _validate_findings(
    findings: list[ResearchFinding],
    question_number: int,
    local_to_global_evidence: dict[int, int],
) -> list[ResearchFinding]:
    """
    Validate local evidence references and map them globally.

    Synthesis is isolated to one original question. Therefore,
    the current question number is assigned by the application
    rather than trusting the LLM to return it correctly.
    """

    validated = []
    seen = set()

    for finding in findings:
        text = finding.text.strip()

        if not text:
            continue

        if not finding.evidence_numbers:
            continue

        local_numbers = finding.evidence_numbers

        # Reject a finding if any local evidence reference is invalid.
        if any(
            number not in local_to_global_evidence
            for number in local_numbers
        ):
            print(
                f"  Rejected finding for Q{question_number}: "
                "invalid local evidence reference."
            )
            continue

        global_evidence_numbers = sorted(
            {
                local_to_global_evidence[number]
                for number in local_numbers
            }
        )

        if not global_evidence_numbers:
            continue

        # Deduplicate repeated findings while retaining their citations.
        key = text.casefold()

        if key in seen:
            continue

        seen.add(key)

        validated.append(
            ResearchFinding(
                text=text,
                research_question_numbers=[question_number],
                evidence_numbers=global_evidence_numbers,
            )
        )

    return validated


# ------------------------------------------------------------
# QUESTION-LEVEL SYNTHESIS
# ------------------------------------------------------------

def _synthesize_original_question(
    question_number: int,
    question_text: str,
    evidence_items: list[Evidence],
    global_evidence_numbers: list[int],
) -> list[ResearchFinding]:
    """Synthesize validated findings for one original question."""

    if not evidence_items:
        print(
            f"No verified evidence available for Q{question_number}."
        )
        return []

    if len(evidence_items) != len(global_evidence_numbers):
        print(
            f"Evidence mapping mismatch for Q{question_number}; "
            "skipping synthesis for this question."
        )
        return []

    local_to_global_evidence = {
        local_number: global_number
        for local_number, global_number in enumerate(
            global_evidence_numbers,
            start=1,
        )
    }

    evidence_text = _format_question_evidence(
        question_number=question_number,
        evidence_items=evidence_items,
    )

    chain = question_synthesis_prompt | question_synthesis_model

    try:
        response = chain.invoke(
            {
                "question_number": question_number,
                "question": question_text,
                "evidence": evidence_text,
            }
        )

    except Exception as exc:
        print(
            f"Question synthesis failed for Q{question_number}: {exc}"
        )
        return []

    findings = _validate_findings(
        findings=response.findings,
        question_number=question_number,
        local_to_global_evidence=local_to_global_evidence,
    )

    print(
        f"  Q{question_number}: "
        f"{len(response.findings)} proposed finding(s), "
        f"{len(findings)} validated finding(s)."
    )

    return findings


# ------------------------------------------------------------
# COVERAGE AND GAP HELPERS
# ------------------------------------------------------------

def _get_question_assessments(
    state: ResearchState,
) -> dict[int, dict]:
    """Index persisted sufficiency assessments by question number."""

    assessments = {}

    for assessment in state.get("coverage_assessments", []):
        question_number = assessment.get(
            "research_question_number"
        )

        if question_number is not None:
            assessments[int(question_number)] = assessment

    return assessments


def _gap_matches_question(
    gap: str,
    question_number: int,
    question_text: str,
) -> bool:
    """Match a research gap to its original question."""

    normalized_gap = gap.strip().lower()
    normalized_question = question_text.strip().lower()

    if normalized_gap.startswith(f"q{question_number}:"):
        return True

    return normalized_gap.startswith(normalized_question)


def _build_coverage(
    roots: list[tuple[int, object]],
    findings: list[ResearchFinding],
    assessments: dict[int, dict],
) -> list[ResearchCoverage]:
    """
    Determine coverage using both validated findings and
    persisted sufficiency assessments.
    """

    coverage = []

    for question_number, question in roots:
        question_findings = [
            finding
            for finding in findings
            if question_number in finding.research_question_numbers
        ]

        assessment = assessments.get(question_number)
        has_findings = bool(question_findings)

        is_sufficient = bool(
            assessment and assessment.get("covered", False)
        )

        reason = (
            str(assessment.get("reason", "")).strip()
            if assessment
            else "No sufficiency assessment was recorded."
        )

        if is_sufficient and has_findings:
            status = "supported"
            explanation = (
                "The sufficiency evaluator judged the evidence "
                "sufficient, and validated findings address this question."
            )

        elif has_findings:
            status = "partially_supported"
            explanation = (
                "Some validated findings address this question, "
                "but the available evidence was not judged sufficient."
            )

            if reason:
                explanation += f" Remaining limitation: {reason}"

        else:
            status = "unresolved"
            explanation = (
                "No validated findings address this question. "
                "The available evidence cannot be presented as a "
                "validated answer."
            )

            if reason:
                explanation += f" Assessment details: {reason}"

        coverage.append(
            ResearchCoverage(
                research_question_number=question_number,
                question=question.question,
                status=status,
                finding_numbers=[
                    index
                    for index, finding in enumerate(
                        findings,
                        start=1,
                    )
                    if question_number
                    in finding.research_question_numbers
                ],
                explanation=explanation,
            )
        )

    return coverage


def _merge_research_gaps(
    original_gaps: list[str],
    roots: list[tuple[int, object]],
    coverage: list[ResearchCoverage],
    assessments: dict[int, dict],
) -> list[str]:
    """Preserve gaps for partially supported and unresolved questions."""

    merged = []

    for gap in original_gaps:
        matched_question_number = None

        for question_number, question in roots:
            if _gap_matches_question(
                gap,
                question_number,
                question.question,
            ):
                matched_question_number = question_number
                break

        if matched_question_number is None:
            merged.append(gap)
            continue

        assessment = assessments.get(matched_question_number)

        if assessment is not None and assessment.get("covered", False):
            continue

        merged.append(gap)

    existing_gaps = {
        gap.strip().lower()
        for gap in merged
    }

    coverage_by_question = {
        item.research_question_number: item
        for item in coverage
    }

    for question_number, question in roots:
        item = coverage_by_question[question_number]

        if item.status == "supported":
            continue

        if any(
            _gap_matches_question(
                gap,
                question_number,
                question.question,
            )
            for gap in merged
        ):
            continue

        assessment = assessments.get(question_number, {})
        reason = str(assessment.get("reason", "")).strip()

        gap = (
            f"Q{question_number}: {question.question} — "
            f"{reason or item.explanation}"
        )

        if gap.lower() not in existing_gaps:
            merged.append(gap)
            existing_gaps.add(gap.lower())

    return list(
        dict.fromkeys(
            gap.strip()
            for gap in merged
            if gap.strip()
        )
    )


# ------------------------------------------------------------
# FINAL REPORT VALIDATION
# ------------------------------------------------------------

def _validate_report(
    report: ResearchReport,
    evidence: list[Evidence],
    research_questions: list | None = None,
    research_gaps: list[str] | None = None,
    roots: list[tuple[int, object]] | None = None,
) -> ResearchReport:
    """
    Validate evidence references, question lineage, coverage,
    gaps, and source URLs.

    Only sources cited by validated findings are included.
    """

    questions = research_questions or []

    if roots is None:
        roots = [
            (number, question)
            for number, question in enumerate(
                questions,
                start=1,
            )
            if question.parent_question_number is None
        ]

    valid_question_numbers = {
        number for number, _ in roots
    }

    # Validate parent references.
    for number, question in enumerate(questions, start=1):
        parent = question.parent_question_number

        if parent is not None and not (1 <= parent < number):
            raise ValueError(
                "Research question has an invalid or "
                "forward-referenced parent."
            )

        if parent is None and number not in valid_question_numbers:
            raise ValueError(
                "Original research question is missing from roots."
            )

    # Validate evidence question numbers.
    for item in evidence:
        if not (
            1 <= item.research_question_number <= len(questions)
        ):
            raise ValueError(
                "Evidence references an unknown research question."
            )

    question_by_number = {
        number: question
        for number, question in enumerate(questions, start=1)
    }

    def original_question_number(
        question_number: int,
    ) -> int | None:
        """Resolve a question or follow-up to its original root."""

        visited = set()

        while question_number is not None:
            if question_number in visited:
                return None

            visited.add(question_number)

            question = question_by_number.get(question_number)

            if question is None:
                return None

            if question.parent_question_number is None:
                return question_number

            question_number = question.parent_question_number

        return None

    validated_findings = []

    for finding in report.findings:
        evidence_numbers = finding.evidence_numbers

        if not finding.text.strip() or not evidence_numbers:
            continue

        if any(
            number < 1 or number > len(evidence)
            for number in evidence_numbers
        ):
            raise ValueError("Invalid evidence reference.")

        derived_questions = {
            root_number
            for number in evidence_numbers
            if (
                root_number := original_question_number(
                    evidence[number - 1].research_question_number
                )
            ) is not None
        }

        supplied_questions = set(
            finding.research_question_numbers
        )

        if not supplied_questions:
            supplied_questions = derived_questions

        elif not supplied_questions.issubset(valid_question_numbers):
            raise ValueError(
                "Finding references an invalid research question."
            )

        if not supplied_questions:
            continue

        # Each cited evidence item must belong to the claimed root
        # question or one of its follow-up questions.
        for number in evidence_numbers:
            evidence_root = original_question_number(
                evidence[number - 1].research_question_number
            )

            if evidence_root not in supplied_questions:
                raise ValueError(
                    "Evidence does not belong to the finding's "
                    "research question lineage."
                )

        validated_findings.append(
            ResearchFinding(
                text=finding.text.strip(),
                research_question_numbers=sorted(supplied_questions),
                evidence_numbers=sorted(set(evidence_numbers)),
            )
        )

    # Rebuild coverage from validated findings.
    validated_coverage = []

    for question_number, question in roots:
        finding_numbers = [
            index
            for index, finding in enumerate(
                validated_findings,
                start=1,
            )
            if question_number in finding.research_question_numbers
        ]

        existing = next(
            (
                item
                for item in report.coverage
                if item.research_question_number == question_number
            ),
            None,
        )

        if existing is not None:
            # Preserve an existing assessment, but never claim a
            # question is supported when no validated finding exists.
            if not finding_numbers:
                status = "unresolved"
                explanation = (
                    "No validated findings address this question."
                )
            else:
                status = existing.status
                explanation = existing.explanation

        elif finding_numbers:
            # Backward-compatible fallback: when no coverage
            # assessment exists, validated evidence-backed findings
            # establish supported coverage.
            status = "supported"
            explanation = (
                "Validated findings address this question."
            )

        else:
            status = "unresolved"
            explanation = (
                "No validated findings address this question."
            )

        validated_coverage.append(
            ResearchCoverage(
                research_question_number=question_number,
                question=question.question,
                status=status,
                finding_numbers=finding_numbers,
                explanation=explanation,
            )
        )



    # Include only sources cited by validated findings.
    referenced_evidence_numbers = {
        evidence_number
        for finding in validated_findings
        for evidence_number in finding.evidence_numbers
    }

    trusted_sources = list(
        dict.fromkeys(
            evidence[number - 1].source_url
            for number in sorted(referenced_evidence_numbers)
        )
    )

    # Preserve caller-supplied gaps and model-generated report gaps.
    preserved_gaps = list(
        dict.fromkeys(
            (research_gaps or []) + report.research_gaps
        )
    )

    # Remove question-specific gaps when validated coverage supports
    # the question; keep gaps for partial or unresolved questions.
    filtered_gaps = []

    for gap in preserved_gaps:
        matched_question_number = None

        for question_number, question in roots:
            if _gap_matches_question(
                gap,
                question_number,
                question.question,
            ):
                matched_question_number = question_number
                break

        if matched_question_number is None:
            filtered_gaps.append(gap)
            continue

        coverage_item = next(
            item
            for item in validated_coverage
            if item.research_question_number == matched_question_number
        )

        if coverage_item.status != "supported":
            filtered_gaps.append(gap)

    preserved_gaps = filtered_gaps

    existing_gap_questions = {
        gap.split(":", 1)[0].strip().lower()
        for gap in preserved_gaps
        if ":" in gap
    }

    # Ensure each partially supported or unresolved question has a gap.
    for item in validated_coverage:
        if item.status == "supported":
            continue

        gap_key = f"q{item.research_question_number}"

        if gap_key not in existing_gap_questions:
            preserved_gaps.append(
                f"Q{item.research_question_number}: "
                f"{item.question} — {item.explanation}"
            )
            existing_gap_questions.add(gap_key)

    preserved_gaps = list(
        dict.fromkeys(
            gap.strip()
            for gap in preserved_gaps
            if gap.strip()
        )
    )

    return ResearchReport(
        title=report.title,
        summary=report.summary,
        findings=validated_findings,
        coverage=validated_coverage,
        research_gaps=preserved_gaps,
        sources=trusted_sources,
    )


# ------------------------------------------------------------
# MAIN SYNTHESIS NODE
# ------------------------------------------------------------

def synthesis_node(
    state: ResearchState,
) -> dict:
    """Build the final source-grounded research report."""

    roots = _build_question_roots(state)
    evidence = state.get("evidence", [])
    questions = state.get("research_questions", [])
    assessments = _get_question_assessments(state)

    # Group evidence by its original root question.
    evidence_by_question = defaultdict(list)
    evidence_numbers_by_question = defaultdict(list)

    for evidence_number, item in enumerate(evidence, start=1):
        if not (
            1 <= item.research_question_number <= len(questions)
        ):
            continue

        root_number = item.research_question_number
        visited = set()

        while root_number is not None:
            if root_number in visited:
                root_number = None
                break

            visited.add(root_number)
            current_question = questions[root_number - 1]

            if current_question.parent_question_number is None:
                break

            root_number = current_question.parent_question_number

        if root_number is not None:
            evidence_by_question[root_number].append(item)
            evidence_numbers_by_question[root_number].append(
                evidence_number
            )

    # Synthesize each original question independently.
    findings = []

    print("\nSynthesizing verified evidence into findings...")

    for question_number, question in roots:
        question_findings = _synthesize_original_question(
            question_number=question_number,
            question_text=question.question,
            evidence_items=evidence_by_question.get(
                question_number,
                [],
            ),
            global_evidence_numbers=evidence_numbers_by_question.get(
                question_number,
                [],
            ),
        )

        findings.extend(question_findings)

    # Deduplicate finding text.
    unique_findings = []
    seen_findings = set()

    for finding in findings:
        key = finding.text.strip().casefold()

        if key in seen_findings:
            continue

        seen_findings.add(key)
        unique_findings.append(finding)

    findings = unique_findings

    # Build coverage and preserve unresolved research gaps.
    coverage = _build_coverage(
        roots=roots,
        findings=findings,
        assessments=assessments,
    )

    research_gaps = _merge_research_gaps(
        original_gaps=state.get("research_gaps", []),
        roots=roots,
        coverage=coverage,
        assessments=assessments,
    )

    # Generate report title and summary.
    narrative_chain = (
        report_narrative_prompt
        | report_narrative_model
    )

    try:
        narrative = narrative_chain.invoke(
            {
                "findings": _format_findings(findings),
                "coverage": "\n".join(
                    f"- Q{item.research_question_number}: "
                    f"{item.status} — {item.explanation}"
                    for item in coverage
                ),
                "gaps": _format_research_gaps(research_gaps),
            }
        )

        title = narrative.title
        summary = narrative.summary

    except Exception as exc:
        print(f"Report narrative generation failed: {exc}")

        title = "Deep Research Report"

        summary = (
            f"The research produced {len(findings)} validated findings "
            f"across {len(roots)} original questions. "
            f"{sum(item.status == 'supported' for item in coverage)} "
            "questions were fully supported, "
            f"{sum(item.status == 'partially_supported' for item in coverage)} "
            "were partially supported, and "
            f"{sum(item.status == 'unresolved' for item in coverage)} "
            "remain unresolved."
        )

    report = ResearchReport(
        title=title,
        summary=summary,
        findings=findings,
        coverage=coverage,
        research_gaps=research_gaps,
        sources=[],
    )

    report = _validate_report(
        report=report,
        evidence=evidence,
        research_questions=questions,
        research_gaps=research_gaps,
        roots=roots,
    )

    print("\n" + "=" * 80)
    print("FINAL RESEARCH REPORT")
    print("=" * 80)
    print(f"\nTitle: {report.title}")
    print(f"\nSummary:\n{report.summary}")
    print(f"\nValidated findings: {len(report.findings)}")

    supported_count = sum(
        item.status == "supported"
        for item in report.coverage
    )
    partial_count = sum(
        item.status == "partially_supported"
        for item in report.coverage
    )
    unresolved_count = sum(
        item.status == "unresolved"
        for item in report.coverage
    )

    print(
        f"Coverage: {supported_count} supported, "
        f"{partial_count} partially supported, "
        f"{unresolved_count} unresolved "
        f"out of {len(report.coverage)} original questions"
    )
    print(f"Sources: {len(report.sources)}")
    print(f"Research gaps: {len(report.research_gaps)}")

    return {
        **state,
        "report": report,
        "research_gaps": report.research_gaps,
    }
