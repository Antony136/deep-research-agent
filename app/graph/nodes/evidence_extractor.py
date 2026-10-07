"""
Evidence extraction node for the Deep Research Agent.

The extractor identifies multiple factual claims from the
currently researched sources.

The LLM is responsible only for identifying:
    - the claim
    - the source containing the claim

Supporting text is always selected deterministically from
the actual source content.

This prevents the LLM from inventing or rewriting evidence.
"""

import os
import re
from difflib import SequenceMatcher

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from app.graph.state import ResearchState
from app.schemas.research import Evidence


load_dotenv()


MODEL_NAME = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5-coder:7b",
)

BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

MAX_EVIDENCE_ITEMS = 5


model = ChatOllama(
    model=MODEL_NAME,
    base_url=BASE_URL,
    temperature=0,
)


class ExtractedEvidence(BaseModel):
    """
    LLM-selected evidence claim.

    The LLM identifies the factual claim and the source
    containing it.

    It does NOT generate supporting text.
    """

    claim: str = Field(
        description=(
            "A specific factual claim directly supported "
            "by one of the supplied sources."
        )
    )

    source_index: int = Field(
        description=(
            "The numeric index of the source containing "
            "direct support for the claim."
        )
    )


class EvidenceExtractionOutput(BaseModel):
    """
    Structured output returned by the LLM.
    """

    evidence: list[ExtractedEvidence] = Field(
        default_factory=list,
        description=(
            "Independent factual claims supported by the "
            "supplied sources."
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
sources that help answer the research question.

IMPORTANT RULES:

1. Use ONLY information explicitly supported by the
   supplied sources.

2. Do NOT use your general knowledge.

3. Do NOT invent facts.

4. Do NOT combine information from different sources into
   one claim.

5. Each claim must be independently supported by exactly
   one source.

6. Return the source index that directly supports each claim.

7. Claims should be specific and factual.

8. Prefer useful information about:
   - capabilities
   - architecture
   - production characteristics
   - scalability
   - persistence
   - state management
   - reliability
   - deployment
   - real-world usage
   - limitations
   - tradeoffs

9. Avoid vague claims such as:
   - "This framework is useful."
   - "It has many features."
   - "It is good for production."

10. Avoid opinions unless the source explicitly presents
    them as findings, limitations, or documented tradeoffs.

11. Do not generate supporting quotations.

12. The application will independently extract the exact
    supporting passage from the source.

13. Extract multiple independent claims when the sources
    contain multiple useful facts.

14. Do not create multiple claims that express essentially
    the same fact.

15. Prefer high-value claims over minor implementation
    details.

16. Return up to 5 strong evidence items.

17. If the sources do not contain useful evidence, return
    an empty list.

SOURCE INDEXING:

Sources are numbered starting from 1.

For example:

SOURCE 1
...

SOURCE 2
...

If a claim is supported by SOURCE 2, return:

source_index = 2

Return only the requested structured output.
""",
        ),
        (
            "human",
            """
Research question:

{research_question}

Sources:

{sources}
""",
        ),
    ]
)


chain = prompt | structured_model


def _format_sources(sources) -> str:
    """
    Format sources for the LLM with stable numeric indexes.
    """

    if not sources:
        return "No sources available."

    sections = []

    for index, source in enumerate(sources, start=1):
        sections.append(
            f"""
SOURCE {index}

Title:
{source.title}

URL:
{source.url}

Content:
{source.content}
""".strip()
        )

    return "\n\n".join(sections)


def _normalize_text(text: str) -> str:
    """
    Normalize text for deterministic matching.
    """

    text = text.lower()

    text = text.replace("\u00ad", "")

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")

    text = re.sub(
        r"-\s*\n\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^\w\s%$.-]",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _tokenize(text: str) -> list[str]:
    """
    Convert text into normalized tokens.
    """

    normalized = _normalize_text(text)

    return normalized.split()


def _split_into_passages(
    content: str,
    max_passage_length: int = 1800,
) -> list[str]:
    """
    Split source content into reasonably sized passages.

    Paragraph boundaries are preferred.

    Very large paragraphs are further divided into sentences.
    """

    if not content.strip():
        return []

    paragraphs = re.split(
        r"\n\s*\n+",
        content,
    )

    passages = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) <= max_passage_length:
            passages.append(paragraph)
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

            candidate = f"{current} {sentence}"

            if len(candidate) <= max_passage_length:
                current = candidate
            else:
                passages.append(current)
                current = sentence

        if current:
            passages.append(current)

    return passages


def _score_passage(
    claim: str,
    passage: str,
) -> float:
    """
    Calculate deterministic relevance between a claim
    and a candidate passage.

    Score:

        70% token overlap
        30% sequence similarity
    """

    claim_tokens = set(_tokenize(claim))
    passage_tokens = set(_tokenize(passage))

    if not claim_tokens or not passage_tokens:
        return 0.0

    overlap = (
        len(claim_tokens & passage_tokens)
        / len(claim_tokens)
    )

    similarity = SequenceMatcher(
        None,
        _normalize_text(claim),
        _normalize_text(passage),
    ).ratio()

    return (
        (0.70 * overlap)
        + (0.30 * similarity)
    )


def _extract_supporting_text(
    claim: str,
    source_content: str,
) -> str | None:
    """
    Select the strongest real passage supporting a claim.

    The returned text always comes directly from the source.
    """

    passages = _split_into_passages(
        source_content
    )

    if not passages:
        return None

    best_passage = None
    best_score = 0.0

    for passage in passages:

        score = _score_passage(
            claim,
            passage,
        )

        if score > best_score:
            best_score = score
            best_passage = passage

    if best_passage is None:
        return None

    claim_tokens = set(
        _tokenize(claim)
    )

    passage_tokens = set(
        _tokenize(best_passage)
    )

    if not claim_tokens:
        return None

    token_overlap = (
        len(claim_tokens & passage_tokens)
        / len(claim_tokens)
    )

    if token_overlap < 0.35:
        return None

    return best_passage


def _resolve_source_index(
    source_index: int,
    source_count: int,
    zero_based: bool,
) -> int | None:
    """
    Convert the LLM source index into a zero-based Python
    list index.
    """

    if zero_based:
        index = source_index
    else:
        index = source_index - 1

    if index < 0 or index >= source_count:
        return None

    return index


def _detect_zero_based_indexes(
    extracted_items: list[ExtractedEvidence],
) -> bool:
    """
    Detect whether the model appears to be using zero-based
    source indexes.

    A source_index of 0 is strong evidence of zero-based
    indexing.
    """

    return any(
        item.source_index == 0
        for item in extracted_items
    )


def _deduplicate_claims(
    extracted_items: list[ExtractedEvidence],
) -> list[ExtractedEvidence]:
    """
    Remove duplicate or near-duplicate claims.
    """

    unique_items = []

    for item in extracted_items:

        normalized_claim = _normalize_text(
            item.claim
        )

        if not normalized_claim:
            continue

        duplicate = False

        for existing in unique_items:

            existing_claim = _normalize_text(
                existing.claim
            )

            similarity = SequenceMatcher(
                None,
                normalized_claim,
                existing_claim,
            ).ratio()

            if similarity >= 0.90:
                duplicate = True
                break

        if not duplicate:
            unique_items.append(item)

    return unique_items


def evidence_extractor_node(
    state: ResearchState,
) -> ResearchState:

    print("\n[Node] evidence_extractor")

    current_sources = state["current_sources"]

    if not current_sources:
        print("  No current sources available.")

        return {
            **state,
            "pending_evidence": [],
        }

    current_index = (
        state["current_question_index"] - 1
    )

    research_questions = state[
        "research_questions"
    ]

    if (
        current_index < 0
        or current_index >= len(research_questions)
    ):
        print("  Invalid research question index.")

        return {
            **state,
            "pending_evidence": [],
        }

    research_question = research_questions[
        current_index
    ]

    print(
        "  Extracting evidence for: "
        f"{research_question.question}"
    )

    print(
        f"  Sources available: "
        f"{len(current_sources)}"
    )

    try:

        response = chain.invoke(
            {
                "research_question": (
                    research_question.question
                ),
                "sources": _format_sources(
                    current_sources
                ),
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

    extracted_items = response.evidence

    print(
        "  Evidence items extracted: "
        f"{len(extracted_items)}"
    )

    if not extracted_items:

        return {
            **state,
            "pending_evidence": [],
        }

    extracted_items = _deduplicate_claims(
        extracted_items
    )

    if len(extracted_items) > MAX_EVIDENCE_ITEMS:

        extracted_items = extracted_items[
            :MAX_EVIDENCE_ITEMS
        ]

    zero_based = _detect_zero_based_indexes(
        extracted_items
    )

    if zero_based:

        print(
            "  Detected zero-based source indexes "
            "from model output."
        )

    pending_evidence = []

    mapped_count = 0

    for item in extracted_items:

        source_index = _resolve_source_index(
            source_index=item.source_index,
            source_count=len(current_sources),
            zero_based=zero_based,
        )

        if source_index is None:

            print(
                "  Invalid source index for claim:"
            )

            print(
                f"    {item.claim}"
            )

            continue

        source = current_sources[
            source_index
        ]

        supporting_text = (
            _extract_supporting_text(
                claim=item.claim,
                source_content=source.content,
            )
        )

        if supporting_text is None:

            print(
                "  Could not find a sufficiently "
                "relevant supporting passage "
                "for claim:"
            )

            print(
                f"    {item.claim}"
            )

            continue

        pending_evidence.append(
            Evidence(
                claim=item.claim.strip(),
                source_url=source.url,
                supporting_text=supporting_text,
            )
        )

        mapped_count += 1

    print(
        "  Evidence items mapped to real "
        "source passages: "
        f"{mapped_count}"
    )

    return {
        **state,
        "pending_evidence": pending_evidence,
    }
