"""
Evidence extraction node for the Deep Research Agent.

The LLM identifies:

- a factual claim
- the source that supports the claim

The LLM is NOT trusted to generate the supporting passage.

Instead, the application deterministically extracts a
supporting passage from the actual collected source content.

This creates a stronger grounding boundary:

    LLM claim
        +
    LLM source selection
        ↓
    deterministic passage extraction
        ↓
    deterministic evidence validation
        ↓
    trusted evidence
"""

import os
import re
import unicodedata
from difflib import SequenceMatcher

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from app.graph.state import ResearchState
from app.schemas.research import Evidence, Source


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


class ExtractedEvidence(BaseModel):
    """
    Evidence returned by the LLM.

    The model identifies the source using its numeric
    position in the supplied source list.

    The model does NOT generate supporting text.
    """

    claim: str = Field(
        description=(
            "One specific factual claim that is directly "
            "supported by the selected source."
        )
    )

    source_index: int = Field(
        description=(
            "The numeric index of the source supporting "
            "this claim. Prefer 1-based indexing: 1 means "
            "the first source, 2 means the second source, "
            "and so on."
        )
    )


class EvidenceExtractionOutput(BaseModel):
    """
    Structured output returned by the evidence extractor.
    """

    evidence: list[ExtractedEvidence] = Field(
        default_factory=list,
        description=(
            "Specific factual claims supported by the "
            "supplied research sources."
        ),
    )


structured_model = model.with_structured_output(
    EvidenceExtractionOutput
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the evidence extraction component of a deep
research agent.

Your task is to identify factual claims from the supplied
web sources that directly help answer the research question.

IMPORTANT SOURCE RULES:

1. Every source has a numeric SOURCE INDEX.
2. Select the source using its SOURCE INDEX.
3. Prefer 1-based indexing.
4. SOURCE INDEX 1 means the first source.
5. SOURCE INDEX 2 means the second source.
6. Never invent a URL.
7. Never modify a URL.
8. Never use information from your general knowledge.

IMPORTANT CLAIM RULES:

9. State one specific factual claim.
10. The claim must be directly supported by the selected
    source.
11. Do not invent facts.
12. Do not use your general knowledge.
13. Do not combine information from multiple sources into
    one claim.
14. Ignore opinions, advertisements, navigation text,
    and unrelated information.
15. Prefer concrete facts, documented capabilities,
    limitations, comparisons, examples, and production
    characteristics.
16. If a source does not contain useful evidence, do not
    use that source.
17. If no useful evidence exists, return an empty evidence
    list.

IMPORTANT:

Do NOT generate a supporting quote or passage.

The application will extract the supporting passage
directly from the actual source content after you select
the source.

Your responsibility is ONLY:

    factual claim + source index

Return only the requested structured output.
""",
        ),
        (
            "human",
            """
Research question:

{question}

Sources:

{sources}
""",
        ),
    ]
)


chain = prompt | structured_model


def _format_sources(
    sources: list[Source],
) -> str:
    """
    Format sources with explicit numeric identifiers.
    """

    sections = []

    for index, source in enumerate(
        sources,
        start=1,
    ):
        sections.append(
            f"""
SOURCE INDEX: {index}

Title:
{source.title}

URL:
{source.url}

Content:
{source.content}
""".strip()
        )

    if not sections:
        return "No sources were available."

    return "\n\n".join(sections)


def _normalize_text(text: str) -> str:
    """
    Normalize text for deterministic comparison.
    """

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.lower()

    text = re.sub(
        r"[\u2010\u2011\u2012\u2013\u2014\u2212]",
        "-",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _tokenize(text: str) -> list[str]:
    """
    Convert text into normalized word tokens.
    """

    normalized = _normalize_text(text)

    if not normalized:
        return []

    return re.findall(
        r"\b\w+\b",
        normalized,
    )


def _split_into_passages(
    content: str,
) -> list[str]:
    """
    Split source content into reasonably sized passages.

    Paragraphs are preferred. Long paragraphs are further
    split into sentences.
    """

    paragraphs = re.split(
        r"\n\s*\n+",
        content,
    )

    passages = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) <= 1200:
            passages.append(
                paragraph
            )
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            paragraph,
        )

        current = ""

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            if not current:
                current = sentence
                continue

            candidate = (
                f"{current} {sentence}"
            )

            if len(candidate) <= 1200:
                current = candidate
            else:
                passages.append(
                    current
                )
                current = sentence

        if current:
            passages.append(
                current
            )

    return passages


def _token_overlap(
    claim_tokens: list[str],
    passage_tokens: list[str],
) -> float:
    """
    Calculate the percentage of claim tokens that occur
    in the candidate passage.
    """

    if not claim_tokens:
        return 0.0

    passage_token_set = set(
        passage_tokens
    )

    matched = sum(
        1
        for token in claim_tokens
        if token in passage_token_set
    )

    return (
        matched
        / len(claim_tokens)
    )


def _score_passage(
    claim: str,
    passage: str,
) -> float:
    """
    Score how strongly a source passage supports a claim.

    The score combines:

    - token overlap
    - character similarity

    This is deterministic and does not call an LLM.
    """

    claim_tokens = _tokenize(
        claim
    )

    passage_tokens = _tokenize(
        passage
    )

    if not claim_tokens:
        return 0.0

    if not passage_tokens:
        return 0.0

    overlap = _token_overlap(
        claim_tokens,
        passage_tokens,
    )

    normalized_claim = _normalize_text(
        claim
    )

    normalized_passage = _normalize_text(
        passage
    )

    similarity = SequenceMatcher(
        None,
        normalized_claim,
        normalized_passage,
    ).ratio()

    # Token overlap is more important because the claim may
    # legitimately paraphrase the source.
    return (
        (overlap * 0.70)
        + (similarity * 0.30)
    )


def _extract_supporting_text(
    claim: str,
    source: Source,
) -> str | None:
    """
    Find the strongest source passage supporting the claim.

    The returned text is copied directly from the actual
    source content. The LLM never generates this text.
    """

    passages = _split_into_passages(
        source.content
    )

    if not passages:
        return None

    best_passage = None
    best_score = 0.0

    for passage in passages:

        score = _score_passage(
            claim=claim,
            passage=passage,
        )

        if score > best_score:
            best_score = score
            best_passage = passage

    if best_passage is None:
        return None

    claim_tokens = _tokenize(
        claim
    )

    passage_tokens = _tokenize(
        best_passage
    )

    overlap = _token_overlap(
        claim_tokens,
        passage_tokens,
    )

    # Require meaningful lexical support.
    #
    # This prevents the extractor from selecting an arbitrary
    # paragraph merely because it shares a few common words.
    if overlap < 0.35:
        return None

    return best_passage


def _resolve_source_index(
    source_index: int,
    source_count: int,
    zero_based_mode: bool,
) -> int | None:
    """
    Convert the LLM's source index into a zero-based Python
    list index.

    Qwen has occasionally returned 0-based indexes despite
    receiving explicit 1-based instructions.

    When the batch contains index 0, the batch is treated as
    zero-based for compatibility.
    """

    if zero_based_mode:

        if (
            0
            <= source_index
            < source_count
        ):
            return source_index

        return None

    if (
        1
        <= source_index
        <= source_count
    ):
        return source_index - 1

    return None


def _convert_extracted_evidence(
    extracted_items: list[ExtractedEvidence],
    sources: list[Source],
) -> list[Evidence]:
    """
    Convert LLM claims and source indexes into Evidence.

    Supporting text is extracted deterministically from the
    actual source content.
    """

    evidence = []

    if not sources:
        return evidence

    zero_based_mode = any(
        item.source_index == 0
        for item in extracted_items
    )

    if zero_based_mode:
        print(
            "  Detected zero-based source indexes "
            "from model output."
        )

    for item in extracted_items:

        source_position = (
            _resolve_source_index(
                source_index=item.source_index,
                source_count=len(sources),
                zero_based_mode=zero_based_mode,
            )
        )

        if source_position is None:
            print(
                "  Ignoring evidence with invalid "
                f"source index: {item.source_index}"
            )
            continue

        source = sources[
            source_position
        ]

        supporting_text = (
            _extract_supporting_text(
                claim=item.claim,
                source=source,
            )
        )

        if supporting_text is None:
            print(
                "  Could not find a sufficiently "
                "relevant supporting passage for claim:"
            )
            print(
                f"    {item.claim}"
            )
            continue

        evidence.append(
            Evidence(
                claim=item.claim,
                source_url=source.url,
                supporting_text=supporting_text,
            )
        )

    return evidence


def evidence_extractor_node(
    state: ResearchState,
) -> ResearchState:
    """
    Extract evidence for the current research question.
    """

    print(
        "\n[Node] evidence_extractor"
    )

    questions = state[
        "research_questions"
    ]

    current_index = state[
        "current_question_index"
    ]

    if current_index <= 0:
        return {
            **state,
            "pending_evidence": [],
        }

    question = questions[
        current_index - 1
    ]

    sources = state[
        "current_sources"
    ]

    if not sources:
        print(
            "  No sources available "
            "for evidence extraction."
        )

        return {
            **state,
            "pending_evidence": [],
        }

    print(
        "  Extracting evidence for: "
        f"{question.question}"
    )

    print(
        "  Sources available: "
        f"{len(sources)}"
    )

    formatted_sources = _format_sources(
        sources
    )

    try:

        response = chain.invoke(
            {
                "question": question.question,
                "sources": formatted_sources,
            }
        )

    except Exception as exc:

        print(
            "  Evidence extraction failed:"
        )

        print(
            f"  {exc}"
        )

        return {
            **state,
            "pending_evidence": [],
        }

    extracted_items = (
        response.evidence
    )

    print(
        "  Evidence items extracted: "
        f"{len(extracted_items)}"
    )

    extracted_evidence = (
        _convert_extracted_evidence(
            extracted_items=extracted_items,
            sources=sources,
        )
    )

    print(
        "  Evidence items mapped to "
        "real source passages: "
        f"{len(extracted_evidence)}"
    )

    return {
        **state,
        "pending_evidence": extracted_evidence,
    }
