"""
Tests for the Deep Research Agent citation formatter.
"""

from app.schemas.research import (
    Evidence,
    ResearchCoverage,
    ResearchFinding,
    ResearchReport,
)
from app.tools.citation_formatter import (
    format_finding_citations,
    format_sources,
    format_research_report,
)


def create_evidence(
    question_number: int,
    claim: str,
    source_url: str,
    supporting_text: str,
) -> Evidence:
    return Evidence(
        research_question_number=question_number,
        claim=claim,
        source_url=source_url,
        supporting_text=supporting_text,
    )


def create_report() -> ResearchReport:
    return ResearchReport(
        title="Improving RAG Retrieval Quality",
        summary="Retrieval quality can be improved through "
                "hybrid search and reranking.",
        findings=[
            ResearchFinding(
                text="Hybrid retrieval combines semantic "
                     "and keyword search.",
                research_question_numbers=[1],
                evidence_numbers=[1, 2],
            ),
        ],
        coverage=[
            ResearchCoverage(
                research_question_number=1,
                question="How does hybrid retrieval work?",
                status="supported",
                finding_numbers=[1],
                explanation="The finding is supported by evidence.",
            ),
            ResearchCoverage(
                research_question_number=2,
                question="What are the limitations?",
                status="unresolved",
                finding_numbers=[],
                explanation="Insufficient verified evidence.",
            ),
        ],
        research_gaps=[
            "Q2: More evidence is needed about limitations."
        ],
        sources=[
            "https://example.com/hybrid-search",
            "https://example.com/keyword-search",
        ],
    )


def create_evidence_list() -> list[Evidence]:
    return [
        create_evidence(
            question_number=1,
            claim="Hybrid retrieval combines search methods.",
            source_url="https://example.com/hybrid-search",
            supporting_text=(
                "Hybrid retrieval combines semantic similarity "
                "with keyword matching."
            ),
        ),
        create_evidence(
            question_number=1,
            claim="Keyword matching contributes lexical relevance.",
            source_url="https://example.com/keyword-search",
            supporting_text=(
                "Keyword matching identifies documents containing "
                "relevant terms."
            ),
        ),
        create_evidence(
            question_number=2,
            claim="An unrelated source was collected.",
            source_url="https://example.com/unreferenced",
            supporting_text="This evidence is not cited by the report.",
        ),
    ]


def test_finding_citations_use_one_based_evidence_numbers():
    evidence = create_evidence_list()

    result = format_finding_citations(
        evidence_numbers=[1, 2],
        evidence=evidence,
    )

    assert result == "[E1] [E2]"


def test_invalid_and_duplicate_citations_are_omitted():
    evidence = create_evidence_list()

    result = format_finding_citations(
        evidence_numbers=[1, 1, 99, 0, 2],
        evidence=evidence,
    )

    assert result == "[E1] [E2]"


def test_sources_include_only_referenced_evidence():
    report = create_report()
    evidence = create_evidence_list()

    result = format_sources(
        report=report,
        evidence=evidence,
    )

    assert "[E1] https://example.com/hybrid-search" in result
    assert "[E2] https://example.com/keyword-search" in result
    assert "https://example.com/unreferenced" not in result


def test_repeated_evidence_is_listed_only_once():
    report = create_report()

    report.findings.append(
        ResearchFinding(
            text="A second finding uses the same evidence.",
            research_question_numbers=[1],
            evidence_numbers=[1, 2],
        )
    )

    result = format_sources(
        report=report,
        evidence=create_evidence_list(),
    )

    assert result.count("[E1] https://example.com/hybrid-search") == 1
    assert result.count("[E2] https://example.com/keyword-search") == 1


def test_sources_handle_report_without_findings():
    report = create_report()
    report.findings = []

    result = format_sources(
        report=report,
        evidence=create_evidence_list(),
    )

    assert result == (
        "No verified evidence was cited in the report."
    )


def test_final_report_includes_citations_coverage_and_gaps():
    report = create_report()
    evidence = create_evidence_list()

    result = format_research_report(
        report=report,
        evidence=evidence,
    )

    assert "Improving RAG Retrieval Quality" in result
    assert "SUMMARY" in result
    assert "FINDINGS" in result
    assert "[E1] [E2]" in result
    assert "RESEARCH COVERAGE" in result
    assert "Q1: supported" in result
    assert "Q2: unresolved" in result
    assert "RESEARCH GAPS" in result
    assert "Q2: More evidence is needed about limitations." in result
    assert "SOURCES AND VERIFIED EVIDENCE" in result
    assert "https://example.com/unreferenced" not in result
