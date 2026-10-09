"""
Deterministic integration test for the Deep Research Agent.

Exercises the compiled LangGraph workflow without calling:
- Ollama / Qwen
- External web search
- Website extraction
- Any external service

The real graph and its routing remain in use.
Individual nodes are replaced with deterministic test doubles.
"""

import app.graph.graph as graph_module

from app.schemas.research import (
    Evidence,
    ResearchCoverage,
    ResearchFinding,
    ResearchQuestion,
    ResearchReport,
    Source,
)


def test_complete_research_workflow_without_external_services(
    monkeypatch,
):
    """
    Verify that a research task travels through the graph
    and produces a source-linked final report.
    """

    executed_nodes = []

    question = ResearchQuestion(
        question="How does vector search support RAG?",
        search_queries=["vector search RAG"],
    )

    source = Source(
        url="https://example.com/vector-search",
        title="Vector Search",
        content=(
            "Vector search retrieves items using "
            "similarity between vector representations."
        ),
    )

    evidence = Evidence(
        research_question_number=1,
        claim=(
            "Vector search retrieves items using "
            "similarity between vector representations."
        ),
        source_url=source.url,
        supporting_text=source.content,
    )

    report = ResearchReport(
        title="Vector Search in RAG",
        summary=(
            "Vector search retrieves semantically similar "
            "items using vector representations."
        ),
        findings=[
            ResearchFinding(
                text=(
                    "Vector search retrieves items by "
                    "comparing vector representations."
                ),
                research_question_numbers=[1],
                evidence_numbers=[1],
            ),
        ],
        coverage=[
            ResearchCoverage(
                research_question_number=1,
                question=question.question,
                status="supported",
                finding_numbers=[1],
                explanation=(
                    "The collected evidence directly "
                    "addresses the research question."
                ),
            ),
        ],
        research_gaps=[],
        sources=[source.url],
    )

    # --------------------------------------------------
    # Deterministic node replacements
    # --------------------------------------------------

    def fake_planner_node(state):
        executed_nodes.append("planner")

        return {
            **state,
            "research_questions": [question],
            "active_question": question,
            "active_question_number": 1,
            "current_question_index": 0,
            "research_round": 1,
            "max_research_rounds": 1,
            "max_total_research_questions": 1,
            "research_complete": False,
            "research_sufficient": False,
            "research_gaps": [],
        }

    def fake_researcher_node(state):
        executed_nodes.append("researcher")

        return {
            **state,
            "active_question": question,
            "active_question_number": 1,
            "current_sources": [source],
            "sources": [source],
        }

    def fake_evidence_extractor_node(state):
        executed_nodes.append("evidence_extractor")

        return {
            **state,
            "pending_evidence": [evidence],
        }

    def fake_evidence_verifier_node(state):
        executed_nodes.append("evidence_verifier")

        return {
            **state,
            "evidence": state.get(
                "pending_evidence",
                [],
            ),
            "pending_evidence": [],
            "current_question_index": len(
                state["research_questions"]
            ),
            "research_complete": True,
        }

    def fake_research_sufficiency_node(state):
        executed_nodes.append("research_sufficiency")

        return {
            **state,
            "research_sufficient": True,
            "research_decision_reason": (
                "The deterministic test evidence is sufficient."
            ),
            "research_gaps": [],
            "coverage_assessments": [
                {
                    "research_question_number": 1,
                    "covered": True,
                    "evidence_numbers": [1],
                    "reason": "Evidence directly supports the question.",
                },
            ],
        }

    def fake_adaptive_planner_node(state):
        executed_nodes.append("adaptive_planner")

        # This should not be needed because sufficiency is True.
        return {
            **state,
            "research_complete": True,
        }

    def fake_synthesis_node(state):
        executed_nodes.append("synthesizer")

        return {
            **state,
            "report": report,
            "research_gaps": [],
        }

    # --------------------------------------------------
    # Patch graph node references before compiling.
    # --------------------------------------------------

    node_replacements = {
        "planner_node": fake_planner_node,
        "researcher_node": fake_researcher_node,
        "evidence_extractor_node": fake_evidence_extractor_node,
        "evidence_verifier_node": fake_evidence_verifier_node,
        "research_sufficiency_node": fake_research_sufficiency_node,
        "adaptive_planner_node": fake_adaptive_planner_node,
        "synthesis_node": fake_synthesis_node,
    }

    for name, replacement in node_replacements.items():
        monkeypatch.setattr(
            graph_module,
            name,
            replacement,
            raising=False,
        )

    # --------------------------------------------------
    # Execute the actual compiled graph.
    # --------------------------------------------------

    graph = graph_module.build_research_graph()

    initial_state = {
        "question": "How does vector search support RAG?",
        "research_questions": [],
        "active_question": None,
        "active_question_number": 0,
        "current_sources": [],
        "sources": [],
        "pending_evidence": [],
        "evidence": [],
        "current_question_index": 0,
        "research_complete": False,
        "research_round": 0,
        "max_research_rounds": 1,
        "max_total_research_questions": 1,
        "research_gaps": [],
        "research_sufficient": False,
        "research_decision_reason": "",
        "coverage_assessments": [],
        "report": None,
    }

    final_state = graph.invoke(initial_state)

    # --------------------------------------------------
    # Verify the workflow result.
    # --------------------------------------------------

    assert "planner" in executed_nodes
    assert "researcher" in executed_nodes
    assert "evidence_extractor" in executed_nodes
    assert "evidence_verifier" in executed_nodes
    assert "research_sufficiency" in executed_nodes
    assert "synthesizer" in executed_nodes

    assert "adaptive_planner" not in executed_nodes

    assert final_state["research_sufficient"] is True
    assert len(final_state["research_questions"]) == 1
    assert len(final_state["evidence"]) == 1
    assert final_state["pending_evidence"] == []

    final_report = final_state["report"]

    assert final_report is not None
    assert len(final_report.findings) == 1
    assert final_report.findings[0].evidence_numbers == [1]
    assert final_report.coverage[0].status == "supported"
    assert final_report.sources[0] == source.url
    assert final_state["research_gaps"] == []
