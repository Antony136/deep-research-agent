"""
Inspect research-plan search queries without running web searches.

Reports query counts, blank queries, exact duplicates, and
normalized duplicates.
"""

import re

from app.graph.nodes.planner import chain


def normalize_query(query: str) -> str:
    """Normalize whitespace and case for diagnostic comparisons."""
    return re.sub(r"\s+", " ", query.strip()).casefold()


def main():
    question = input("Enter a research question: ").strip()

    if not question:
        print("Research question cannot be empty.")
        return

    print("\nGenerating research plan...")

    plan = chain.invoke({"question": question})
    questions = plan.research_questions

    all_queries = []

    print(f"\nResearch questions generated: {len(questions)}")

    for index, research_question in enumerate(questions, start=1):
        queries = research_question.search_queries or []

        print(f"\nQuestion {index}: {research_question.question}")
        print(f"Search queries: {len(queries)}")

        for query in queries:
            print(f"  - {query!r}")
            all_queries.append(query)

    normalized = [normalize_query(query) for query in all_queries]

    nonblank = [query for query in normalized if query]

    print("\n" + "=" * 60)
    print("QUERY DIAGNOSTICS")
    print("=" * 60)
    print(f"Total queries: {len(all_queries)}")
    print(f"Blank queries: {sum(not query.strip() for query in all_queries)}")
    print(f"Exact duplicates: {len(all_queries) - len(set(all_queries))}")
    print(f"Normalized duplicates: {len(nonblank) - len(set(nonblank))}")


if __name__ == "__main__":
    main()
