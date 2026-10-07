"""
Deterministic evidence validation for the Deep Research Agent.

This module checks whether extracted evidence is actually
grounded in the sources collected by the research system.

The validator does not use an LLM. It performs deterministic
checks before evidence is allowed to move further through
the research pipeline.

The matching logic is intentionally tolerant of common web
extraction differences such as:

- Unicode punctuation
- HTML text formatting
- Different whitespace
- Line breaks
- Line-break hyphenation
- Minor text extraction differences

It still requires strong textual overlap before evidence
is considered grounded.
"""

import re
import unicodedata
from difflib import SequenceMatcher

from app.schemas.research import Evidence, Source


# Maximum supporting passage length accepted for fuzzy matching.
MAX_SUPPORTING_TEXT_LENGTH = 2000

# Minimum similarity required by the fuzzy fallback.
MIN_FUZZY_SIMILARITY = 0.85

# Minimum token overlap required by the fuzzy fallback.
MIN_TOKEN_OVERLAP = 0.80


def _normalize_text(text: str) -> str:
    """
    Normalize extracted web text for comparison.

    This does not modify the stored evidence. It only creates
    a comparison-friendly representation.
    """

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    # Remove soft hyphens.
    text = text.replace(
        "\u00ad",
        "",
    )

    # Normalize common Unicode dash characters.
    text = re.sub(
        r"[\u2010\u2011\u2012\u2013\u2014\u2212]",
        "-",
        text,
    )

    # Join words that were split across a line by a hyphen.
    #
    # Example:
    #
    #   produc-
    #   tion
    #
    # becomes:
    #
    #   production
    #
    text = re.sub(
        r"(?<=\w)-\s+(?=\w)",
        "",
        text,
    )

    text = text.lower()

    # Replace punctuation with spaces.
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _tokenize(text: str) -> list[str]:
    """
    Convert normalized text into word-like tokens.
    """

    normalized = _normalize_text(text)

    if not normalized:
        return []

    return re.findall(
        r"\b\w+\b",
        normalized,
    )


def _url_key(url: str) -> str:
    """
    Normalize URLs for deterministic comparison.
    """

    return url.strip().rstrip("/").lower()


def _exact_match(
    supporting_text: str,
    source_content: str,
) -> bool:
    """
    Check whether the normalized supporting text appears
    directly inside the normalized source content.
    """

    normalized_support = _normalize_text(
        supporting_text
    )

    normalized_source = _normalize_text(
        source_content
    )

    if not normalized_support:
        return False

    if not normalized_source:
        return False

    return normalized_support in normalized_source


def _token_sequence_match(
    supporting_text: str,
    source_content: str,
) -> bool:
    """
    Check whether the supporting text appears as an exact
    sequence of normalized tokens in the source.

    This handles cases where punctuation or formatting differs
    but the actual words are preserved.
    """

    support_tokens = _tokenize(
        supporting_text
    )

    source_tokens = _tokenize(
        source_content
    )

    if not support_tokens:
        return False

    if len(support_tokens) > len(source_tokens):
        return False

    support_length = len(support_tokens)

    first_token = support_tokens[0]

    candidate_positions = [
        index
        for index, token in enumerate(source_tokens)
        if token == first_token
    ]

    for start in candidate_positions:

        end = start + support_length

        if end > len(source_tokens):
            continue

        if (
            source_tokens[start:end]
            == support_tokens
        ):
            return True

    return False


def _calculate_token_overlap(
    supporting_tokens: list[str],
    source_tokens: list[str],
) -> float:
    """
    Calculate how many supporting-text tokens are represented
    in a source chunk.

    Duplicate tokens are intentionally preserved because
    repeated words can be meaningful in a passage.
    """

    if not supporting_tokens:
        return 0.0

    source_token_set = set(
        source_tokens
    )

    matched_tokens = sum(
        1
        for token in supporting_tokens
        if token in source_token_set
    )

    return (
        matched_tokens
        / len(supporting_tokens)
    )


def _fuzzy_match(
    supporting_text: str,
    source_content: str,
) -> bool:
    """
    Perform a conservative fuzzy comparison.

    The source is examined in windows approximately the same
    size as the supporting passage.

    A match requires BOTH:

    - sufficiently high character similarity
    - sufficiently high token overlap

    This prevents weak partial matches from being accepted.
    """

    normalized_support = _normalize_text(
        supporting_text
    )

    normalized_source = _normalize_text(
        source_content
    )

    if not normalized_support:
        return False

    if not normalized_source:
        return False

    if len(normalized_support) > MAX_SUPPORTING_TEXT_LENGTH:
        return False

    support_tokens = _tokenize(
        normalized_support
    )

    source_tokens = _tokenize(
        normalized_source
    )

    if not support_tokens:
        return False

    support_token_count = len(
        support_tokens
    )

    if support_token_count > len(source_tokens):
        return False

    # Search token-based windows rather than every character
    # position. This keeps validation reasonably efficient
    # for large web pages.
    window_size = support_token_count

    # Small overlap between windows prevents a relevant passage
    # from falling between two windows.
    step = max(
        window_size // 3,
        10,
    )

    for start in range(
        0,
        len(source_tokens),
        step,
    ):
        end = start + window_size

        if end > len(source_tokens):
            break

        source_window_tokens = (
            source_tokens[start:end]
        )

        token_overlap = (
            _calculate_token_overlap(
                support_tokens,
                source_window_tokens,
            )
        )

        if (
            token_overlap
            < MIN_TOKEN_OVERLAP
        ):
            continue

        source_window = " ".join(
            source_window_tokens
        )

        similarity = SequenceMatcher(
            None,
            normalized_support,
            source_window,
        ).ratio()

        if (
            similarity
            >= MIN_FUZZY_SIMILARITY
        ):
            return True

    return False


def _supporting_text_exists(
    supporting_text: str,
    source_content: str,
) -> bool:
    """
    Determine whether supporting text can be grounded
    in the source content.

    Matching is attempted from strongest to weakest:

    1. Normalized exact substring
    2. Exact normalized token sequence
    3. Conservative fuzzy token/character matching
    """

    if not supporting_text.strip():
        return False

    if not source_content.strip():
        return False

    # Strongest check.
    if _exact_match(
        supporting_text,
        source_content,
    ):
        return True

    # Handles punctuation and formatting differences.
    if _token_sequence_match(
        supporting_text,
        source_content,
    ):
        return True

    # Conservative fallback for small extraction differences.
    if _fuzzy_match(
        supporting_text,
        source_content,
    ):
        return True

    return False


def validate_evidence(
    evidence: Evidence,
    sources: list[Source],
) -> tuple[bool, str]:
    """
    Validate a single evidence item.

    Evidence is considered valid only when:

    - claim exists
    - source URL exists
    - supporting text exists
    - cited URL belongs to a collected source
    - supporting text can be grounded in that source
    """

    if not evidence.claim.strip():
        return False, (
            "Evidence claim is empty."
        )

    if not evidence.source_url.strip():
        return False, (
            "Evidence source URL is empty."
        )

    if not evidence.supporting_text.strip():
        return False, (
            "Evidence supporting text is empty."
        )

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
        return False, (
            "Evidence references a source URL "
            "that was not collected."
        )

    if not _supporting_text_exists(
        supporting_text=evidence.supporting_text,
        source_content=source.content,
    ):
        return False, (
            "Supporting text could not be matched "
            "strongly enough to the cited source."
        )

    return True, (
        "Evidence is grounded in the cited source."
    )


def validate_evidence_batch(
    evidence_items: list[Evidence],
    sources: list[Source],
) -> tuple[
    list[Evidence],
    list[tuple[Evidence, str]],
]:
    """
    Validate multiple evidence items.

    Returns:

        (
            valid_evidence,
            rejected_evidence,
        )

    Rejected evidence includes the original evidence item
    together with the deterministic rejection reason.
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
