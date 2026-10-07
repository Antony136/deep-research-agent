"""
Evidence extraction node for the Deep Research Agent.

The extractor identifies factual claims from focused source
passages for the current research question.

The implementation is intentionally source-first:

    Source
      ↓
    Focused passages
      ↓
    Small LLM batches
      ↓
    Claim + passage_id
      ↓
    Deterministic grounding
      ↓
    Evidence

The LLM is never allowed to invent the supporting text.
Supporting text is always copied directly from the selected
source passage.
"""

import os
import re
from difflib import SequenceMatcher

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from app.graph.state import ResearchState
from app.schemas.research import Evidence, Source


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

load_dotenv()

MODEL_NAME = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5-coder:7b",
)

BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

# Maximum number of passages considered from one source.
MAX_PASSAGES_PER_SOURCE = 3

# Maximum total focused passages considered for one
# research question.
MAX_FOCUSED_PASSAGES = 12

# Number of passages sent to Qwen in one extraction call.
#
# Keeping this small is important when running Qwen 7B
# locally on limited VRAM.
PASSAGES_PER_LLM_BATCH = 6

# Maximum length of an individual passage sent to Qwen.
MAX_PASSAGE_LENGTH = 1200

# Maximum number of evidence items accepted from one
# LLM batch.
MAX_EVIDENCE_PER_BATCH = 3

# Maximum trusted evidence proposals produced for the
# current research question.
MAX_EVIDENCE_PER_QUESTION = 5

# Minimum lexical overlap required before semantic
# verification.
MIN_TOKEN_OVERLAP = 0.20

# Minimum fuzzy similarity required when lexical overlap
# is weak but the claim is otherwise a close restatement.
MIN_FUZZY_SIMILARITY = 0.45


# ----------------------------------------------------------
# Stop words
# ----------------------------------------------------------

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
    "which",
    "what",
    "how",
    "why",
    "than",
    "they",
    "these",
    "those",
    "using",
    "used",
    "use",
}


# ----------------------------------------------------------
# LLM schema
# ----------------------------------------------------------


class ExtractedEvidence(BaseModel):
    """
    LLM-selected evidence.

    The model identifies a claim and the exact passage
    that supports it. The Python code later copies the
    actual passage text from the source.
    """

    claim: str = Field(
        description=(
            "A concise factual claim directly supported "
            "by exactly one supplied passage."
        )
    )

    passage_id: int = Field(
        description=(
            "The numeric ID of the single passage that "
            "directly supports the claim."
        )
    )


class EvidenceExtractionOutput(BaseModel):
    """
    Structured output returned by one LLM batch.
    """

    evidence: list[ExtractedEvidence] = Field(
        default_factory=list,
        description=(
            "Evidence claims directly supported by the "
            "supplied passages."
        ),
    )


# ----------------------------------------------------------
# Model
# ----------------------------------------------------------

model = ChatOllama(
    model=MODEL_NAME,
    base_url=BASE_URL,
    temperature=0,
)

structured_model = model.with_structured_output(
    EvidenceExtractionOutput
)


# ----------------------------------------------------------
# Prompt
# ----------------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the evidence extraction component of a deep
research agent.

Your task is to extract factual claims from the supplied
source passages that directly help answer the research
question.

STRICT GROUNDING RULES:

1. Use ONLY the supplied passages.
2. Every claim must be directly supported by ONE passage.
3. Do not combine information from multiple passages.
4. Do not use outside knowledge.
5. Do not infer facts that are not explicitly supported.
6. Do not make the claim broader than the passage.
7. Do not add conclusions, comparisons, opinions, or
   implications unless the passage explicitly states them.
8. Use the passage_id of the passage that supports the claim.
9. Prefer precise factual claims over vague summaries.
10. If a passage does not contain useful evidence, ignore it.
11. Return at most 3 useful evidence items.
12. A claim may be a concise paraphrase of the passage.
13. Preserve important qualifiers such as "can", "may",
    "supports", "designed for", "according to", or
    other limitations expressed by the source.

IMPORTANT:

The Python system will copy the actual supporting passage
itself. You must NOT invent or reproduce supporting text.

Return only the requested structured output.
""",
        ),
        (
            "human",
            """
Research question:

{research_question}

SOURCE PASSAGES:

{passages}
""",
        ),
    ]
)


chain = prompt | structured_model


# ----------------------------------------------------------
# Text normalization
# ----------------------------------------------------------


def _normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    """

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _tokens(text: str) -> set[str]:
    """
    Extract meaningful normalized tokens.
    """

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower(),
    )

    return {
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) > 2
    }


def _token_overlap(
    claim: str,
    passage: str,
) -> float:
    """
    Calculate the proportion of claim tokens that occur
    in the supporting passage.

    This is intentionally directional:

        claim tokens ∩ passage tokens
        ------------------------------
              claim tokens

    This helps prevent the model from adding new factual
    concepts that are not present in the source.
    """

    claim_tokens = _tokens(claim)

    if not claim_tokens:
        return 0.0

    passage_tokens = _tokens(passage)

    overlap = claim_tokens.intersection(
        passage_tokens
    )

    return len(overlap) / len(claim_tokens)


def _fuzzy_similarity(
    claim: str,
    passage: str,
) -> float:
    """
    Calculate a lightweight fuzzy similarity between the
    claim and the passage.

    This is only a secondary grounding signal. It is not
    sufficient by itself to establish evidence validity.
    """

    normalized_claim = _normalize_text(
        claim
    )

    normalized_passage = _normalize_text(
        passage
    )

    if not normalized_claim or not normalized_passage:
        return 0.0

    return SequenceMatcher(
        None,
        normalized_claim,
        normalized_passage,
    ).ratio()


# ----------------------------------------------------------
# Passage preparation
# ----------------------------------------------------------


def _split_into_passages(
    content: str,
) -> list[str]:
    """
    Split source content into reasonably sized passages.

    Paragraph boundaries are preferred, but long paragraphs
    are further divided so that individual LLM inputs remain
    manageable.
    """

    normalized = " ".join(
        content.split()
    )

    if not normalized:
        return []

    paragraphs = re.split(
        r"(?<=[.!?])\s+",
        normalized,
    )

    passages = []

    current = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if not current:
            current = paragraph
            continue

        candidate = (
            f"{current} {paragraph}"
        )

        if len(candidate) <= MAX_PASSAGE_LENGTH:
            current = candidate

        else:

            passages.append(
                current[:MAX_PASSAGE_LENGTH]
            )

            current = paragraph

    if current:
        passages.append(
            current[:MAX_PASSAGE_LENGTH]
        )

    return passages


def _relevance_score(
    research_question: str,
    passage: str,
) -> float:
    """
    Score a passage against the research question.

    This is deliberately deterministic so that the LLM sees
    only passages that are already relevant to the question.
    """

    question_tokens = _tokens(
        research_question
    )

    passage_tokens = _tokens(
        passage
    )

    if not question_tokens or not passage_tokens:
        return 0.0

    overlap = question_tokens.intersection(
        passage_tokens
    )

    token_score = (
        len(overlap)
        / len(question_tokens)
    )

    fuzzy_score = SequenceMatcher(
        None,
        _normalize_text(research_question),
        _normalize_text(passage),
    ).ratio()

    return (
        0.70 * token_score
        + 0.30 * fuzzy_score
    )


def _select_focused_passages(
    research_question: str,
    sources: list[Source],
) -> list[tuple[int, str, str]]:
    """
    Select the most relevant passages from the supplied
    sources.

    Returns tuples:

        (passage_id, source_url, passage_text)
    """

    candidates = []

    passage_id = 1

    for source in sources:

        passages = _split_into_passages(
            source.content
        )

        scored_passages = []

        for passage in passages:

            score = _relevance_score(
                research_question,
                passage,
            )

            scored_passages.append(
                (
                    score,
                    passage,
                )
            )

        scored_passages.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        for score, passage in scored_passages[
            :MAX_PASSAGES_PER_SOURCE
        ]:

            candidates.append(
                (
                    score,
                    passage_id,
                    source.url,
                    passage,
                )
            )

            passage_id += 1

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    focused = []

    for (
        score,
        current_passage_id,
        source_url,
        passage,
    ) in candidates[
        :MAX_FOCUSED_PASSAGES
    ]:

        focused.append(
            (
                current_passage_id,
                source_url,
                passage,
            )
        )

    return focused


# ----------------------------------------------------------
# LLM extraction
# ----------------------------------------------------------


def _format_passages(
    passages: list[tuple[int, str, str]],
) -> str:
    """
    Format a small group of passages for the LLM.
    """

    sections = []

    for (
        passage_id,
        source_url,
        passage,
    ) in passages:

        sections.append(
            f"""
PASSAGE {passage_id}

Source URL:
{source_url}

Text:
{passage}
""".strip()
        )

    return "\n\n".join(
        sections
    )


def _extract_from_batch(
    research_question: str,
    passages: list[tuple[int, str, str]],
) -> list[ExtractedEvidence]:
    """
    Run one small evidence-extraction call.
    """

    if not passages:
        return []

    response = chain.invoke(
        {
            "research_question": research_question,
            "passages": _format_passages(
                passages
            ),
        }
    )

    return response.evidence[
        :MAX_EVIDENCE_PER_BATCH
    ]


# ----------------------------------------------------------
# Deterministic grounding
# ----------------------------------------------------------


def _is_grounded_claim(
    claim: str,
    passage: str,
) -> bool:
    """
    Determine whether a claim is sufficiently grounded in
    its selected source passage.

    Two signals are used:

    1. Directional token overlap.
    2. Fuzzy similarity.

    The claim must have meaningful lexical support. Fuzzy
    similarity can help with paraphrasing, but cannot by
    itself establish support.
    """

    if not claim.strip():
        return False

    overlap = _token_overlap(
        claim,
        passage,
    )

    fuzzy = _fuzzy_similarity(
        claim,
        passage,
    )

    if overlap >= MIN_TOKEN_OVERLAP:
        return True

    if (
        overlap >= 0.10
        and fuzzy >= MIN_FUZZY_SIMILARITY
    ):
        return True

    return False


# ----------------------------------------------------------
# Main node
# ----------------------------------------------------------


def evidence_extractor_node(
    state: ResearchState,
) -> ResearchState:

    print("\n[Node] evidence_extractor")

    # ------------------------------------------------------
    # Use the explicitly active research question.
    #
    # IMPORTANT:
    # current_question_index already points to the NEXT
    # question because the researcher increments it after
    # collecting the current question's sources.
    # ------------------------------------------------------

    current_question = state[
        "active_research_question"
    ]

    current_question_number = state[
        "active_research_question_number"
    ]

    if (
        current_question is None
        or current_question_number is None
    ):
        print(
            "  No active research question is available."
        )

        return {
            **state,
            "pending_evidence": [],
        }

    research_question = current_question.question

    current_sources = state[
        "current_sources"
    ]

    print(
        "  Research question:"
    )

    print(
        f"    {research_question}"
    )

    print(
        "  Research question number: "
        f"{current_question_number}"
    )

    print(
        "  Sources available: "
        f"{len(current_sources)}"
    )

    focused_passages = (
        _select_focused_passages(
            research_question,
            current_sources,
        )
    )

    print(
        "  Relevant passages selected: "
        f"{len(focused_passages)}"
    )

    source_counts = {}

    for (
        _passage_id,
        source_url,
        _passage,
    ) in focused_passages:

        source_counts[source_url] = (
            source_counts.get(
                source_url,
                0,
            )
            + 1
        )

    for source_number, (
        source_url,
        count,
    ) in enumerate(
        source_counts.items(),
        start=1,
    ):

        print(
            f"    Source {source_number}: "
            f"{count} focused passage(s)"
        )

    # ------------------------------------------------------
    # Split the focused passages into small LLM batches.
    # ------------------------------------------------------

    batches = [
        focused_passages[
            index:index + PASSAGES_PER_LLM_BATCH
        ]
        for index in range(
            0,
            len(focused_passages),
            PASSAGES_PER_LLM_BATCH,
        )
    ]

    print(
        "  LLM extraction batches: "
        f"{len(batches)}"
    )

    all_proposals = []

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):

        print(
            f"  Running extraction batch "
            f"{batch_number}/{len(batches)} "
            f"({len(batch)} passages)..."
        )

        try:

            proposals = _extract_from_batch(
                research_question,
                batch,
            )

        except Exception as exc:

            print(
                "  Evidence extraction batch "
                "failed:"
            )

            print(
                f"    {exc}"
            )

            continue

        print(
            "  Batch proposals: "
            f"{len(proposals)}"
        )

        all_proposals.extend(
            proposals
        )

    print(
        "  Evidence items proposed by LLM: "
        f"{len(all_proposals)}"
    )

    # ------------------------------------------------------
    # Build a passage lookup.
    # ------------------------------------------------------

    passage_lookup = {
        passage_id: (
            source_url,
            passage,
        )
        for (
            passage_id,
            source_url,
            passage,
        ) in focused_passages
    }

    evidence_items = []

    seen_claims = set()

    # ------------------------------------------------------
    # Deterministic validation of every proposal.
    # ------------------------------------------------------

    for proposal in all_proposals:

        claim = proposal.claim.strip()

        if not claim:
            continue

        normalized_claim = (
            _normalize_text(
                claim
            )
        )

        if normalized_claim in seen_claims:
            continue

        passage_data = passage_lookup.get(
            proposal.passage_id
        )

        if passage_data is None:

            print()
            print(
                "  REJECTED CLAIM"
            )

            print(
                "    Invalid passage ID: "
                f"{proposal.passage_id}"
            )

            continue

        source_url, supporting_text = (
            passage_data
        )

        if not _is_grounded_claim(
            claim,
            supporting_text,
        ):

            print()
            print(
                "  REJECTED CLAIM"
            )

            print(
                "    Claim is not sufficiently "
                "grounded in selected passage."
            )

            print(
                f"    Claim: {claim}"
            )

            print(
                f"    Passage ID: "
                f"{proposal.passage_id}"
            )

            continue

        evidence_items.append(
            Evidence(
                research_question_number=(
                    current_question_number
                ),
                claim=claim,
                source_url=source_url,
                supporting_text=supporting_text,
            )
        )

        seen_claims.add(
            normalized_claim
        )

        print()
        print(
            "  ACCEPTED CLAIM"
        )

        print(
            f"    Question: "
            f"{current_question_number}"
        )

        print(
            f"    Claim: {claim}"
        )

        print(
            f"    Passage ID: "
            f"{proposal.passage_id}"
        )

        print(
            f"    Source: {source_url}"
        )

        if len(evidence_items) >= (
            MAX_EVIDENCE_PER_QUESTION
        ):
            break

    print()
    print(
        "  Evidence extraction summary:"
    )

    print(
        f"    LLM proposals: "
        f"{len(all_proposals)}"
    )

    print(
        f"    Accepted: "
        f"{len(evidence_items)}"
    )

    print(
        f"    Rejected: "
        f"{len(all_proposals) - len(evidence_items)}"
    )

    return {
        **state,
        "pending_evidence": evidence_items,
    }
