"""
Citation formatting utilities for the Deep Research Agent.

Converts a validated research report and its evidence into
a readable report with traceable evidence citations.
"""

from app.schemas.research import Evidence, ResearchReport


def format_finding_citations(
    evidence_numbers: list[int],
    evidence: list[Evidence],
) -> str:
    """
    Format valid, one-based evidence references.

    Example:
        [E1] [E3]

    Invalid references are omitted defensively.
    """

    citations = []

    for number in evidence_numbers:
        if 1 <= number <= len(evidence):
            marker = f"[E{number}]"

            if marker not in citations:
                citations.append(marker)

    return " ".join(citations)


def format_sources(
    report: ResearchReport,
    evidence: list[Evidence],
) -> str:
    """
    Format only evidence explicitly referenced by report findings.

    Each evidence citation includes its source URL and
    the verified supporting text.
    """

    referenced_numbers = []

    for finding in report.findings:
        for number in finding.evidence_numbers:
            if (
                1 <= number <= len(evidence)
                and number not in referenced_numbers
            ):
                referenced_numbers.append(number)

    if not referenced_numbers:
        return "No verified evidence was cited in the report."

    sections = []

    for number in referenced_numbers:
        item = evidence[number - 1]

        sections.append(
            f"[E{number}] {item.source_url}\n"
            f"    Research question: "
            f"{item.research_question_number}\n"
            f"    Supporting evidence: "
            f"{item.supporting_text.strip()}"
        )

    return "\n\n".join(sections)


def format_research_report(
    report: ResearchReport,
    evidence: list[Evidence],
) -> str:
    """
    Render the validated research report as readable text.

    Citation numbers correspond to positions in the verified
    evidence list and remain consistent throughout the report.
    """

    sections = [
        report.title,
        "=" * len(report.title),
        "",
        "SUMMARY",
        report.summary,
        "",
        "FINDINGS",
    ]

    if report.findings:
        for index, finding in enumerate(
            report.findings,
            start=1,
        ):
            citations = format_finding_citations(
                evidence_numbers=finding.evidence_numbers,
                evidence=evidence,
            )

            sections.extend(
                [
                    "",
                    f"{index}. {finding.text}",
                    f"   Research questions: "
                    f"{finding.research_question_numbers}",
                    f"   Evidence: "
                    f"{citations or 'No valid citations'}",
                ]
            )
    else:
        sections.append("No supported findings were produced.")

    sections.extend(
        [
            "",
            "RESEARCH COVERAGE",
        ]
    )

    if report.coverage:
        for item in report.coverage:
            sections.append(
                f"- Q{item.research_question_number}: "
                f"{item.status} — {item.explanation}"
            )
    else:
        sections.append("No coverage assessments are available.")

    sections.extend(
        [
            "",
            "RESEARCH GAPS",
        ]
    )

    if report.research_gaps:
        for gap in report.research_gaps:
            sections.append(f"- {gap}")
    else:
        sections.append("No unresolved research gaps were reported.")

    sections.extend(
        [
            "",
            "SOURCES AND VERIFIED EVIDENCE",
            "",
            format_sources(report, evidence),
        ]
    )

    return "\n".join(sections)
