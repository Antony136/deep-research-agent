"""
Regression tests for synthesis evidence filtering and validation.

Tests the real synthesis_node while replacing LLM calls and
semantic validation with deterministic test doubles.

No Ollama or external services are required.
"""

from langchain_core.runnables import RunnableLambda

from app.schemas.research import (
    Evidence,
    ResearchFinding,
    ResearchQuestion,
)
from app.graph.nodes import synthesizer


def make_finding(text, evidence_numbers=None):
    return ResearchFinding(
        text=text,
        research_question_numbers=[1],
        evidence_numbers=evidence_numbers or [1],
    )


def make_state(evidence):
    question = ResearchQuestion(
        question="How does vector search support RAG?",
        search_queries=["vector search RAG"],
    )

    return {
        "question": question.question,
        "research_questions": [question],
        "evidence": evidence,
        "coverage_assessments": [
            {
                "research_question_number": 1,
                "covered": True,
                "evidence_numbers": list(
                    range(1, len(evidence) + 1)
                ),
                "reason": "Evidence approved for synthesis.",
            },
        ],
        "research_gaps": [],
    }


def install_model_mocks(
    monkeypatch,
    finding_text=(
        "Vector search retrieves semantically similar documents."
    ),
):
    captured_prompts = []

    def fake_question_synthesis(prompt_value):
        captured_prompts.append(prompt_value.to_string())

        return synthesizer.QuestionSynthesisOutput(
            findings=[
                make_finding(finding_text),
            ],
        )

    def fake_report_narrative(_prompt_value):
        return synthesizer.ReportNarrativeOutput(
            title="Vector Search in RAG",
            summary="A report about vector search.",
        )

    monkeypatch.setattr(
        synthesizer,
        "question_synthesis_model",
        RunnableLambda(fake_question_synthesis),
    )

    monkeypatch.setattr(
        synthesizer,
        "report_narrative_model",
        RunnableLambda(fake_report_narrative),
    )

    return captured_prompts


def make_evidence(
    claim=(
        "Vector search retrieves semantically similar documents."
    ),
    supporting_text=(
        "Vector search retrieves semantically similar documents "
        "by comparing their vector representations."
    ),
    source_url="https://example.com/vector-search",
):
    return Evidence(
        research_question_number=1,
        claim=claim,
        source_url=source_url,
        supporting_text=supporting_text,
    )


def test_synthesis_uses_only_sufficiency_approved_evidence(
    monkeypatch,
):
    captured_prompts = []

    question = ResearchQuestion(
        question="How does vector search support RAG?",
        search_queries=["vector search RAG"],
    )

    unapproved_evidence = Evidence(
        research_question_number=1,
        claim="Keyword search matches literal words.",
        source_url="https://example.com/keyword-search",
        supporting_text=(
            "Keyword search matches literal words rather "
            "than comparing semantic vector representations."
        ),
    )

    approved_evidence = make_evidence()

    def fake_question_synthesis(prompt_value):
        captured_prompts.append(prompt_value.to_string())

        return synthesizer.QuestionSynthesisOutput(
            findings=[
                make_finding(
                    "Vector search retrieves semantically similar documents."
                ),
            ],
        )

    def fake_report_narrative(_prompt_value):
        return synthesizer.ReportNarrativeOutput(
            title="Vector Search in RAG",
            summary="Vector search retrieves similar documents.",
        )

    monkeypatch.setattr(
        synthesizer,
        "question_synthesis_model",
        RunnableLambda(fake_question_synthesis),
    )

    monkeypatch.setattr(
        synthesizer,
        "report_narrative_model",
        RunnableLambda(fake_report_narrative),
    )

    # Isolate this test from Ollama while keeping the actual
    # synthesis, evidence filtering, mapping, and validation active.
    monkeypatch.setattr(
        synthesizer,
        "validate_claim_support",
        lambda claim, supporting_text: (
            True,
            "Claim is supported by the supplied passage.",
        ),
    )

    state = {
        "question": question.question,
        "research_questions": [question],
        "evidence": [
            unapproved_evidence,
            approved_evidence,
        ],
        "coverage_assessments": [
            {
                "research_question_number": 1,
                "covered": True,
                "evidence_numbers": [2],
                "reason": "Only evidence 2 was approved.",
            },
        ],
        "research_gaps": [],
    }

    result = synthesizer.synthesis_node(state)
    report = result["report"]

    assert len(captured_prompts) == 1

    prompt_text = captured_prompts[0]

    assert approved_evidence.supporting_text in prompt_text
    assert unapproved_evidence.supporting_text not in prompt_text

    assert len(report.findings) == 1

    finding = report.findings[0]

    assert finding.research_question_numbers == [1]
    assert finding.evidence_numbers == [2]

    assert report.sources == [approved_evidence.source_url]
    assert unapproved_evidence.source_url not in report.sources

    assert len(report.coverage) == 1
    assert report.coverage[0].status == "supported"
    assert report.coverage[0].finding_numbers == [1]
    assert report.research_gaps == []


def test_semantically_supported_finding_is_kept(monkeypatch):
    evidence = make_evidence()
    install_model_mocks(monkeypatch)

    monkeypatch.setattr(
        synthesizer,
        "validate_claim_support",
        lambda claim, supporting_text: (
            True,
            "The passage supports the finding.",
        ),
    )

    result = synthesizer.synthesis_node(
        make_state([evidence])
    )

    report = result["report"]

    assert len(report.findings) == 1
    assert report.findings[0].evidence_numbers == [1]
    assert report.sources == [evidence.source_url]


def test_semantically_unsupported_finding_is_rejected(
    monkeypatch,
):
    evidence = make_evidence()
    install_model_mocks(
        monkeypatch,
        finding_text=(
            "Vector search always guarantees perfect retrieval accuracy."
        ),
    )

    monkeypatch.setattr(
        synthesizer,
        "validate_claim_support",
        lambda claim, supporting_text: (
            False,
            "The passage does not establish this guarantee.",
        ),
    )

    result = synthesizer.synthesis_node(
        make_state([evidence])
    )

    report = result["report"]

    assert report.findings == []
    assert report.sources == []
    assert report.coverage[0].status == "unresolved"
    assert report.research_gaps


def test_semantic_verifier_failure_fails_closed(monkeypatch):
    evidence = make_evidence()
    install_model_mocks(monkeypatch)

    def verifier_failure(claim, supporting_text):
        return False, "Semantic verification failed."

    monkeypatch.setattr(
        synthesizer,
        "validate_claim_support",
        verifier_failure,
    )

    result = synthesizer.synthesis_node(
        make_state([evidence])
    )

    report = result["report"]

    assert report.findings == []
    assert report.sources == []
    assert report.coverage[0].status == "unresolved"
    assert report.research_gaps
