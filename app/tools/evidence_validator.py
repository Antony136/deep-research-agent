"""
Deterministic evidence validation for the Deep Research Agent.

This module checks whether extracted evidence is actually
grounded in the sources collected by the research system.

The validator does not use an LLM. It performs deterministic
checks before evidence is allowed to move further through
the research pipeline.
"""

import re
from difflib import SequenceMatcher

from app.schemas.research import Evidence, Source


# Minimum similarity required when the supporting text is not
# an exact substring of the source content.
MIN_SUPPORTING_TEXT_SIMILARITY = 0.75


def _normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    This removes differences caused by:
    - whitespace
    - line breaks
    - repeated spaces
    - capitalization
    - simple surrounding punctuation
    """

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = text.strip()

    return text


def _url_key(url: str) -> str:
    """
    Normalize a URL for basic comparison.
    """

    return url.strip().rstrip("/").lower()


def _supporting_text_exists(
    supporting_text: str,
    source_content: str,
) -> bool:
    """
    Check whether supporting text appears in the source.

    First performs an exact normalized substring check.

    If that fails, performs a conservative similarity check
    against source content windows.
    """

    supporting_text = _normalize_text(
        supporting_text
    )

    source_content = _normalize_text(
        source_content
    )

    if not supporting_text or not source_content:
        return False

    # ------------------------------------------------------
    # Exact normalized match
    # ------------------------------------------------------

    if supporting_text in source_content:
        return True

    # ------------------------------------------------------
    # Similarity fallback
    # ------------------------------------------------------

    # Avoid expensive comparison for extremely large text.
    # Supporting passages are expected to be relatively short.
    if len(supporting_text) > 2000:
        return False

    target_length = len(
        supporting_text
    )

    # Compare against windows approximately the same size
    # as the claimed supporting passage.
    window_size = max(
        target_length,
        100,
    )

    step = max(
        window_size // 3,
        50,
    )

    for start in range(
        0,
        len(source_content),
        step,
    ):

        window = source_content[
            start:start + window_size
        ]

        if not window:
            break

        similarity = SequenceMatcher(
            None,
            supporting_text,
            window,
        ).ratio()

        if similarity >= MIN_SUPPORTING_TEXT_SIMILARITY:
            return True

    return False


def validate_evidence(
    evidence: Evidence,
    sources: list[Source],
) -> tuple[bool, str]:
    """
    Validate one evidence item against collected sources.

    Returns:

        (True, reason)
        (False, reason)
    """

    # ------------------------------------------------------
    # 1. Basic evidence validation
    # ------------------------------------------------------

    if not evidence.claim.strip():

        return (
            False,
            "Evidence claim is empty.",
        )

    if not evidence.source_url.strip():

        return (
            False,
            "Evidence source URL is empty.",
        )

    if not evidence.supporting_text.strip():

        return (
            False,
            "Evidence supporting text is empty.",
        )

    # ------------------------------------------------------
    # 2. Verify source URL
    # ------------------------------------------------------

    evidence_url = _url_key(
        evidence.source_url
    )

    source_map = {
        _url_key(source.url): source
        for source in sources
    }

    source = source_map.get(
        evidence_url
    )

    if source is None:

        return (
            False,
            "Evidence references a source URL "
            "that was not collected.",
        )

    # ------------------------------------------------------
    # 3. Verify supporting text
    # ------------------------------------------------------

    if not _supporting_text_exists(
        evidence.supporting_text,
        source.content,
    ):

        return (
            False,
            "Supporting text could not be "
            "matched to the cited source.",
        )

    return (
        True,
        "Evidence is grounded in the cited source.",
    )


def validate_evidence_batch(
    evidence_items: list[Evidence],
    sources: list[Source],
) -> tuple[
    list[Evidence],
    list[tuple[Evidence, str]],
]:
    """
    Validate a collection of evidence items.

    Returns:

        valid_evidence
        rejected_evidence

    Rejected evidence is retained together with the reason
    so that the system can later expose verification failures
    during debugging or evaluation.
    """

    valid_evidence = []

    rejected_evidence = []

    for evidence in evidence_items:

        valid, reason = validate_evidence(
            evidence=evidence,
            sources=sources,
        )

        if valid:

            valid_evidence.append(
                evidence
            )

        else:

            rejected_evidence.append(
                (
                    evidence,
                    reason,
                )
            )

    return (
        valid_evidence,
        rejected_evidence,
    )
