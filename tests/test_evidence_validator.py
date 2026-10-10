"""Tests for deterministic and semantic evidence validation."""

from unittest.mock import Mock

import pytest

from app.schemas.research import Evidence, Source
from app.tools import evidence_validator


@pytest.fixture
def source():
    return Source(
        url="https://example.com/vector-search",
        title="Vector Search Guide",
        content=(
            "Vector indexes organize embeddings to support "
            "efficient similarity search. HNSW is a graph-based "
            "indexing algorithm that supports approximate nearest "
            "neighbor search."
        ),
    )


@pytest.fixture
def valid_evidence():
    return Evidence(
        research_question_number=1,
        claim="HNSW supports approximate nearest neighbor search.",
        source_url="https://example.com/vector-search",
        supporting_text=(
            "HNSW is a graph-based indexing algorithm that "
            "supports approximate nearest neighbor search."
        ),
    )


def mock_semantic_result(
    supported=True,
    reason="The passage directly supports the claim.",
):
    return evidence_validator.SemanticValidationOutput(
        supported=supported,
        reason=reason,
    )


def mock_semantic_chain(monkeypatch, *, result=None, side_effect=None):
    """Replace the entire chain instead of modifying its invoke method."""

    mocked_chain = Mock()

    if side_effect is not None:
        mocked_chain.invoke.side_effect = side_effect
    else:
        mocked_chain.invoke.return_value = result

    monkeypatch.setattr(
        evidence_validator,
        "semantic_chain",
        mocked_chain,
    )

    return mocked_chain


def test_valid_evidence_is_accepted(
    monkeypatch,
    source,
    valid_evidence,
):
    chain = mock_semantic_chain(
        monkeypatch,
        result=mock_semantic_result(),
    )

    valid, reason = evidence_validator.validate_evidence(
        valid_evidence,
        [source],
    )

    assert valid is True
    assert "semantic" in reason.lower()
    chain.invoke.assert_called_once()


def test_semantically_unsupported_claim_is_rejected(
    monkeypatch,
    source,
    valid_evidence,
):
    mock_semantic_chain(
        monkeypatch,
        result=mock_semantic_result(
            supported=False,
            reason="The passage does not establish this claim.",
        ),
    )

    valid, reason = evidence_validator.validate_evidence(
        valid_evidence,
        [source],
    )

    assert valid is False
    assert "does not adequately support" in reason.lower()


def test_fabricated_supporting_passage_is_rejected(
    monkeypatch,
    source,
    valid_evidence,
):
    valid_evidence.supporting_text = (
        "HNSW reduces database costs by exactly 73 percent."
    )

    chain = mock_semantic_chain(
        monkeypatch,
        result=mock_semantic_result(),
    )

    valid, reason = evidence_validator.validate_evidence(
        valid_evidence,
        [source],
    )

    assert valid is False
    assert "matched strongly enough" in reason.lower()
    chain.invoke.assert_not_called()


def test_uncollected_source_url_is_rejected(
    monkeypatch,
    source,
    valid_evidence,
):
    chain = mock_semantic_chain(
        monkeypatch,
        result=mock_semantic_result(),
    )

    valid_evidence.source_url = (
        "https://other-example.com/article"
    )

    valid, reason = evidence_validator.validate_evidence(
        valid_evidence,
        [source],
    )

    assert valid is False
    assert "not collected" in reason.lower()
    chain.invoke.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/vector-search",
        "javascript:alert(1)",
        "https://",
        "https://user:password@example.com/vector-search",
    ],
)
def test_invalid_source_urls_are_rejected(
    source,
    valid_evidence,
    url,
):
    valid_evidence.source_url = url

    valid, reason = evidence_validator.validate_evidence(
        valid_evidence,
        [source],
    )

    assert valid is False
    assert "url" in reason.lower()


def test_url_fragment_and_default_port_are_normalized(
    monkeypatch,
    source,
    valid_evidence,
):
    valid_evidence.source_url = (
        "HTTPS://EXAMPLE.COM:443/vector-search#section-2"
    )

    mock_semantic_chain(
        monkeypatch,
        result=mock_semantic_result(),
    )

    valid, _ = evidence_validator.validate_evidence(
        valid_evidence,
        [source],
    )

    assert valid is True


def test_oversized_claim_is_rejected(
    monkeypatch,
    source,
    valid_evidence,
):
    valid_evidence.claim = (
        "A" * (evidence_validator.MAX_CLAIM_LENGTH + 1)
    )

    chain = mock_semantic_chain(
        monkeypatch,
        result=mock_semantic_result(),
    )

    valid, reason = evidence_validator.validate_evidence(
        valid_evidence,
        [source],
    )

    assert valid is False
    assert "maximum allowed length" in reason.lower()
    chain.invoke.assert_not_called()


def test_semantic_verification_failure_rejects_evidence(
    monkeypatch,
    source,
    valid_evidence,
):
    mock_semantic_chain(
        monkeypatch,
        side_effect=RuntimeError("Ollama unavailable"),
    )

    valid, reason = evidence_validator.validate_evidence(
        valid_evidence,
        [source],
    )

    assert valid is False
    assert "verification failed" in reason.lower()


def test_batch_validation_continues_after_rejection(
    monkeypatch,
    source,
    valid_evidence,
):
    unsupported_evidence = Evidence(
        research_question_number=1,
        claim="An unsupported claim.",
        source_url=source.url,
        supporting_text="This passage does not exist in the source.",
    )

    chain = mock_semantic_chain(
        monkeypatch,
        result=mock_semantic_result(),
    )

    valid_items, rejected_items = (
        evidence_validator.validate_evidence_batch(
            [
                unsupported_evidence,
                valid_evidence,
            ],
            [source],
        )
    )

    assert valid_items == [valid_evidence]
    assert len(rejected_items) == 1
    assert rejected_items[0][0] == unsupported_evidence

    # The fabricated passage is rejected before semantic validation.
    assert chain.invoke.call_count == 1


def test_empty_source_list_rejects_evidence(
    monkeypatch,
    valid_evidence,
):
    chain = mock_semantic_chain(
        monkeypatch,
        result=mock_semantic_result(),
    )

    valid, reason = evidence_validator.validate_evidence(
        valid_evidence,
        [],
    )

    assert valid is False
    assert "no collected sources" in reason.lower()
    chain.invoke.assert_not_called()
