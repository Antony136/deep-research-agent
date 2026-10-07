"""
Core schemas for the Deep Research Agent.

These Pydantic models define the structured data that moves
through the research system.
"""

from pydantic import BaseModel, Field


class ResearchQuestion(BaseModel):
    """
    A single question that the research agent must investigate.
    """

    question: str = Field(
        description="A specific research question to investigate."
    )

    search_queries: list[str] = Field(
        default_factory=list,
        description="Search queries that can help answer the question.",
    )


class Source(BaseModel):
    """
    A web source discovered during research.
    """

    url: str = Field(
        description="The URL of the source."
    )

    title: str = Field(
        default="",
        description="Title of the source.",
    )

    content: str = Field(
        default="",
        description="Relevant extracted content from the source.",
    )


class Evidence(BaseModel):
    """
    A piece of evidence extracted from a source.

    Each evidence item is explicitly associated with the
    research question it was extracted for.
    """

    research_question_number: int = Field(
        description=(
            "1-based number of the research question "
            "this evidence supports."
        )
    )

    claim: str = Field(
        description="The factual claim supported by the evidence."
    )

    source_url: str = Field(
        description="URL of the source supporting the claim."
    )

    supporting_text: str = Field(
        description="Text from the source that supports the claim."
    )


class ResearchReport(BaseModel):
    """
    Final structured research report.
    """

    title: str = Field(
        description="Title of the research report."
    )

    summary: str = Field(
        description="Concise summary of the research findings."
    )

    findings: list[str] = Field(
        default_factory=list,
        description="Important findings from the research.",
    )

    sources: list[str] = Field(
        default_factory=list,
        description="URLs used to support the report.",
    )
