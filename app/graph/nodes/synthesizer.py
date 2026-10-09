"""
Final synthesis node for the Deep Research Agent.

Converts verified research evidence into an evidence-linked report.

Responsibilities:
- use verified evidence as the factual basis
- associate findings with original research questions
- validate evidence references
- preserve source URLs deterministically
- acknowledge unresolved research gaps
- reject invalid citations
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.schemas.research import ResearchReport


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


structured_model = model.with_structured_output(
    ResearchReport
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the final synthesis component of a deep research agent.

Produce a factual research report using ONLY the supplied
verified evidence.

STRICT RULES:

1. Do not introduce outside facts or unsupported conclusions.

2. Every finding must be supported by one or more supplied
   evidence items.

3. Every finding must include evidence_numbers containing
   the 1-based positions of the evidence supporting it.

4. Include research_question_numbers identifying the original
   research questions addressed by each finding.

5. Use only evidence numbers that exist in the input.

6. Original research questions use their original Q numbers.
   Follow-up questions belong to the original question
   identified by their parent_question_number.

7. Do not invent source URLs. The application populates
   report sources from cited verified evidence.

8. Address the original user question directly.

9. Preserve important comparisons, trade-offs, and distinctions
   requested by the original question.

10. Do not claim that a question is answered merely because
    related evidence exists.

11. If evidence conflicts, explain the conflict rather than
    inventing a resolution.

12. If evidence is insufficient, acknowledge the limitation.

13. Keep findings concise, specific, and non-duplicative.

14. The summary must reflect the findings and limitations.

The report's sources field should be empty. The application
will populate it deterministically.
""",
        ),
        (
            "human",
            """
Original user research question:

{question}


Research questions investigated:

{research_questions}


Verified evidence, numbered by position:

{evidence}


Unresolved research gaps:

{research_gaps}


Research sufficient:

{research_sufficient}


Research decision:

{research_decision_reason}


Generate the final structured research report.

Address each original research question as far as the
available evidence allows.

Every finding must include valid evidence_numbers.
Include research_question_numbers whenever possible.
Do not invent evidence or source URLs.
""",
        ),
    ]
)


chain = prompt | structured_model


def _format_research_questions(
    research_questions,
) -> str:
    """Format original and follow-up research questions."""

    if not research_questions:
        return "No research questions were generated."

    lines = []

    for index, research_question in enumerate(
        research_questions,
        start=1,
    ):
        parent = research_question.parent_question_number

        if parent is None:
            relationship = "original question"
        else:
            relationship = (
                f"follow-up to original question Q{parent}"
            )

        lines.append(
            f"Q{index} ({relationship}): "
            f"{research_question.question}"
        )

    return "\n".join(lines)


def _format_evidence(
    evidence,
) -> str:
    """Number verified evidence for citation by the model."""

    if not evidence:
        return "No verified evidence was collected."

    blocks = []

    for index, item in enumerate(
        evidence,
        start=1,
    ):
        blocks.append(
            "\n".join(
                [
                    f"Evidence {index}",
                    f"Research question: "
                    f"Q{item.research_question_number}",
                    f"Claim: {item.claim}",
                    f"Supporting passage: "
                    f"{item.supporting_text}",
                    f"Source URL: {item.source_url}",
                ]
            )
        )

    return "\n\n".join(blocks)


def _format_research_gaps(
    research_gaps,
) -> str:
    """Format unresolved research gaps."""

    if not research_gaps:
        return "No unresolved research gaps."

    return "\n".join(
        f"- {gap}"
        for gap in research_gaps
    )


def _build_question_roots(
    research_questions,
) -> tuple[dict[int, int], set[int]]:
    """
    Map every question number to its original question number.

    Original questions map to themselves.
    Follow-up questions map to their declared parent.
    """

    question_roots = {}
    original_question_numbers = set()

    for index, research_question in enumerate(
        research_questions,
        start=1,
    ):
        parent = research_question.parent_question_number

        if parent is None:
            question_roots[index] = index
            original_question_numbers.add(index)
        else:
            if parent not in original_question_numbers:
                raise ValueError(
                    f"Research question Q{index} has an invalid "
                    f"or forward-referenced parent Q{parent}."
                )

            question_roots[index] = parent

    return question_roots, original_question_numbers


def _validate_report(
    report: ResearchReport,
    evidence,
    research_questions,
) -> ResearchReport:
    """
    Validate finding citations and construct source URLs.

    Missing question references are derived from the cited
    evidence's research-question lineage.

    Invalid evidence references are never repaired silently.
    """

    question_roots, original_question_numbers = (
        _build_question_roots(research_questions)
    )

    validated_findings = []
    cited_evidence_numbers = set()

    for finding in report.findings:
        if not finding.text.strip():
            raise ValueError(
                "The synthesizer produced an empty finding."
            )

        evidence_numbers = list(
            dict.fromkeys(finding.evidence_numbers)
        )

        if not evidence_numbers:
            raise ValueError(
                f"Finding has no evidence references: "
                f"{finding.text}"
            )

        # Validate evidence references before using them.
        for evidence_number in evidence_numbers:
            if (
                evidence_number < 1
                or evidence_number > len(evidence)
            ):
                raise ValueError(
                    f"Invalid evidence reference E{evidence_number} "
                    f"in finding: {finding.text}"
                )

        # Derive the original question lineage from the cited
        # evidence instead of relying entirely on the LLM.
        derived_question_numbers = set()

        for evidence_number in evidence_numbers:
            item = evidence[evidence_number - 1]
            question_number = item.research_question_number

            if question_number not in question_roots:
                raise ValueError(
                    f"Evidence E{evidence_number} references "
                    f"unknown research question Q{question_number}."
                )

            derived_question_numbers.add(
                question_roots[question_number]
            )

        supplied_question_numbers = list(
            dict.fromkeys(finding.research_question_numbers)
        )

        # If Qwen omits question references, derive them from
        # the validated evidence lineage.
        if not supplied_question_numbers:
            final_question_numbers = sorted(
                derived_question_numbers
            )
        else:
            for question_number in supplied_question_numbers:
                if question_number not in original_question_numbers:
                    raise ValueError(
                        f"Invalid original research question "
                        f"reference Q{question_number} in finding: "
                        f"{finding.text}"
                    )

                if question_number not in derived_question_numbers:
                    raise ValueError(
                        f"Finding references Q{question_number}, "
                        f"but its cited evidence does not belong "
                        f"to that question or its follow-ups: "
                        f"{finding.text}"
                    )

            final_question_numbers = supplied_question_numbers

        validated_finding = finding.model_copy(
            update={
                "evidence_numbers": evidence_numbers,
                "research_question_numbers": (
                    final_question_numbers
                ),
            }
        )

        validated_findings.append(validated_finding)

        cited_evidence_numbers.update(evidence_numbers)

    # Construct source URLs exclusively from the actual evidence.
    # Never trust the URLs generated by the language model.
    source_urls = list(
        dict.fromkeys(
            evidence[number - 1].source_url
            for number in sorted(cited_evidence_numbers)
        )
    )

    return report.model_copy(
        update={
            "findings": validated_findings,
            "sources": source_urls,
        }
    )


def synthesis_node(
    state,
):
    """Generate and validate the final research report."""

    print("\n[Node] synthesizer")

    evidence = state["evidence"]
    research_questions = state["research_questions"]

    print(f"Verified evidence: {len(evidence)}")
    print(
        f"Research sufficient: "
        f"{state['research_sufficient']}"
    )
    print("Generating evidence-linked research report...")

    response = chain.invoke(
        {
            "question": state["question"],
            "research_questions": _format_research_questions(
                research_questions
            ),
            "evidence": _format_evidence(evidence),
            "research_gaps": _format_research_gaps(
                state["research_gaps"]
            ),
            "research_sufficient": state["research_sufficient"],
            "research_decision_reason": (
                state["research_decision_reason"]
            ),
        }
    )

    report = _validate_report(
        report=response,
        evidence=evidence,
        research_questions=research_questions,
    )

    print(f"Validated findings: {len(report.findings)}")
    print(f"Cited source URLs: {len(report.sources)}")
    print("[Node] Final synthesis completed.")

    return {
        **state,
        "report": report,
    }