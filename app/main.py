"""
Entry point for the Deep Research Agent.

This currently runs the planning stage of the research
workflow. Additional research stages will be added later.
"""

from app.graph.graph import build_research_graph


def main():

    print("=" * 70)
    print("DEEP RESEARCH AGENT")
    print("=" * 70)

    graph = build_research_graph()

    initial_state = {
        "question": (
            "Compare LangChain, LangGraph, and CrewAI "
            "for building production AI agent systems."
        ),
        "research_questions": [],
        "sources": [],
        "evidence": [],
        "current_question_index": 0,
        "research_complete": False,
        "report": None,
    }

    print("\nRESEARCH QUESTION")
    print("-" * 70)
    print(initial_state["question"])

    final_state = graph.invoke(
        initial_state
    )

    print("\nRESEARCH PLAN")
    print("-" * 70)

    for index, research_question in enumerate(
        final_state["research_questions"],
        start=1,
    ):
        print(
            f"\n{index}. "
            f"{research_question.question}"
        )

        print("   Search queries:")

        for query in research_question.search_queries:
            print(f"   - {query}")

    print("\n" + "=" * 70)
    print("PLANNING STAGE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
