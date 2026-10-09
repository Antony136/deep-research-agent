"""
Tests for final report quality and citation integrity.

These tests do not invoke Ollama. They test the deterministic
validation logic in the synthesizer.
"""

import pytest

from app.graph.nodes.synthesizer import _validate_report
from app.schemas.research import (
    Evidence,
    ResearchFinding,
    ResearchQuestion,
    ResearchReport,
)


@pytest.fixture
def research_questions():
    return [
        ResearchQuestion(
            question="How does vector search work?",
            search_queries=["vector search"],
        ),
        ResearchQuestion(
            question="How does reranking improve retrieval?",
            search_queries=["retrieval reranking"],
        ),
        ResearchQuestion(
            question="What are the benefits of hybrid retrieval?",
            search_queries=["hybrid retrieval"],
        ),
        ResearchQuestion(
            question="How can hybrid retrieval be improved?",
            search_queries=["hybrid retrieval improvements"],
            parent_question_number=3,
        ),
    ]


@pytest.fixture
def evidence():
    return [
        Evidence(
            research_question_number=1,
            claim="Vector search retrieves semantically similar items.",
            source_url="https://example.com/vector-search",
            supporting_text=(
                "Vector search compares vector representations "
                "to retrieve semantically similar items."
            ),
        ),
        Evidence(
            research_question_number=2,
            claim="Reranking reorders retrieved candidates.",
            source_url="https://example.com/reranking",
            supporting_text=(
                "A reranker scores retrieved candidates and "
                "reorders them by estimated relevance."
            ),
        ),
        Evidence(
            research_question_number=4,
            claim="Hybrid retrieval combines retrieval methods.",
            source_url="https://example.com/hybrid-retrieval",
            supporting_text=(
                "Hybrid retrieval combines lexical and dense "
                "retrieval signals."
            ),
        ),
    ]


@pytest.fixture
def valid_report():
    return ResearchReport(
        title="RAG Retrieval Methods",
        summary=(
            "The collected evidence describes vector search "
            "and reranking. Hybrid retrieval is also represented."
        ),
        findings=[
            ResearchFinding(
                text=(
                    "Vector search retrieves semantically "
                    "similar items."
                ),
                research_question_numbers=[1],
                evidence_numbers=[1],
            ),
            ResearchFinding(
                text="Reranking reorders retrieved candidates.",
                research_question_numbers=[2],
                evidence_numbers=[2],
            ),
            ResearchFinding(
                text=(
                    "Hybrid retrieval combines lexical "
                    "and dense retrieval signals."
                ),
                research_question_numbers=[3],
                evidence_numbers=[3],
            ),
        ],
    )


def test_valid_findings_keep_citations_and_sources(
    valid_report,
    evidence,
    research_questions,
):
    report = _validate_report(
        report=valid_report,
        evidence=evidence,
        research_questions=research_questions,
        research_gaps=[],
    )

    assert len(report.findings) == 3

    assert report.sources == [
        "https://example.com/vector-search",
        "https://example.com/reranking",
        "https://example.com/hybrid-retrieval",
    ]

    assert report.findings[0].evidence_numbers == [1]
    assert report.findings[1].evidence_numbers == [2]
    assert report.findings[2].evidence_numbers == [3]


def test_missing_question_references_are_derived(
    evidence,
    research_questions,
):
    report = ResearchReport(
        title="Test report",
        summary="Test summary",
        findings=[
            ResearchFinding(
                text="Hybrid retrieval combines retrieval signals.",
                evidence_numbers=[3],
            ),
        ],
    )

    validated = _validate_report(
        report=report,
        evidence=evidence,
        research_questions=research_questions,
    )

    # Evidence E3 belongs to follow-up Q4, whose original parent is Q3.
    assert validated.findings[0].research_question_numbers == [3]


def test_invalid_evidence_reference_is_rejected(
    evidence,
    research_questions,
):
    report = ResearchReport(
        title="Test report",
        summary="Test summary",
        findings=[
            ResearchFinding(
                text="Unsupported finding.",
                research_question_numbers=[1],
                evidence_numbers=[99],
            ),
        ],
    )

    with pytest.raises(ValueError, match="Invalid evidence reference"):
        _validate_report(
            report=report,
            evidence=evidence,
            research_questions=research_questions,
        )


def test_invalid_question_lineage_is_rejected(
    evidence,
    research_questions,
):
    report = ResearchReport(
        title="Test report",
        summary="Test summary",
        findings=[
            ResearchFinding(
                text="Vector search retrieves similar items.",
                research_question_numbers=[2],
                evidence_numbers=[1],
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="does not belong",
    ):
        _validate_report(
            report=report,
            evidence=evidence,
            research_questions=research_questions,
        )


def test_unreferenced_model_source_urls_are_not_trusted(
    evidence,
    research_questions,
):
    report = ResearchReport(
        title="Test report",
        summary="Test summary",
        findings=[
            ResearchFinding(
                text="Vector search retrieves similar items.",
                research_question_numbers=[1],
                evidence_numbers=[1],
            ),
        ],
        sources=[
            "https://fake.example/invented-source",
        ],
    )

    validated = _validate_report(
        report=report,
        evidence=evidence,
        research_questions=research_questions,
    )

    assert validated.sources == [
        "https://example.com/vector-search",
    ]

    assert "https://fake.example/invented-source" not in (
        validated.sources
    )


def test_all_original_questions_receive_coverage(
    valid_report,
    evidence,
    research_questions,
):
    report = _validate_report(
        report=valid_report,
        evidence=evidence,
        research_questions=research_questions,
    )

    assert [
        item.research_question_number
        for item in report.coverage
    ] == [1, 2, 3]

    assert report.coverage[0].status == "supported"
    assert report.coverage[1].status == "supported"
    assert report.coverage[2].status == "supported"


def test_uncovered_question_is_reported_as_a_gap(
    evidence,
    research_questions,
):
    report = ResearchReport(
        title="Test report",
        summary="Vector search was investigated.",
        findings=[
            ResearchFinding(
                text="Vector search retrieves similar items.",
                research_question_numbers=[1],
                evidence_numbers=[1],
            ),
        ],
    )

    validated = _validate_report(
        report=report,
        evidence=evidence,
        research_questions=research_questions,
        research_gaps=[],
    )

    assert validated.coverage[0].status == "supported"
    assert validated.coverage[1].status == "unresolved"
    assert validated.coverage[2].status == "unresolved"

    assert any(
        gap.startswith("Q2:")
        for gap in validated.research_gaps
    )

    assert any(
        gap.startswith("Q3:")
        for gap in validated.research_gaps
    )


def test_existing_research_gaps_are_preserved(
    evidence,
    research_questions,
):
    report = ResearchReport(
        title="Test report",
        summary="Research remains incomplete.",
        findings=[
            ResearchFinding(
                text="Vector search retrieves similar items.",
                research_question_numbers=[1],
                evidence_numbers=[1],
            ),
        ],
    )

    gap = "The sources did not establish comparative latency."

    validated = _validate_report(
        report=report,
        evidence=evidence,
        research_questions=research_questions,
        research_gaps=[gap],
    )

    assert gap in validated.research_gaps


def test_invalid_evidence_question_number_is_rejected(
    research_questions,
):
    evidence_with_invalid_question = [
        Evidence(
            research_question_number=99,
            claim="Some claim.",
            source_url="https://example.com/source",
            supporting_text="Supporting passage.",
        ),
    ]

    report = ResearchReport(
        title="Test report",
        summary="Test summary",
        findings=[
            ResearchFinding(
                text="A finding.",
                research_question_numbers=[1],
                evidence_numbers=[1],
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="unknown research question",
    ):
        _validate_report(
            report=report,
            evidence=evidence_with_invalid_question,
            research_questions=research_questions,
        )


def test_original_question_cannot_have_a_parent():
    invalid_questions = [
        ResearchQuestion(
            question="Original question",
            parent_question_number=2,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="invalid or forward-referenced parent",
    ):
        _validate_report(
            report=ResearchReport(
                title="Test",
                summary="Test",
            ),
            evidence=[],
            research_questions=invalid_questions,
        )
