"""
Baseline latency benchmark for web search and webpage reading.

Measures the existing functions without changing their behavior.
"""

import statistics
import time

from app.tools.web_reader import read_webpage
from app.tools.web_search import search_web


SEARCH_QUERIES = [
    "Python official documentation",
    "PostgreSQL official documentation",
]

WEBPAGE_URLS = [
    "https://docs.python.org/3/",
    "https://www.postgresql.org/docs/",
]

MIN_SEARCH_DELAY = 3.0
MAX_SEARCH_DELAY = 6.0


def measure_searches():
    durations = []

    print("\n" + "=" * 70)
    print("WEB SEARCH BASELINE")
    print("=" * 70)

    print(
        f"Configured initial delay: "
        f"{MIN_SEARCH_DELAY:.1f}–{MAX_SEARCH_DELAY:.1f} seconds per query"
    )

    for query in SEARCH_QUERIES:
        print(f"\nQuery: {query}")

        started = time.perf_counter()

        try:
            results = search_web(
                query=query,
                max_results=5,
                min_delay=MIN_SEARCH_DELAY,
                max_delay=MAX_SEARCH_DELAY,
            )

            elapsed = time.perf_counter() - started
            durations.append(elapsed)

            print(f"Results: {len(results)}")
            print(f"Total elapsed: {elapsed:.2f} seconds")

        except Exception as exc:
            elapsed = time.perf_counter() - started

            print(f"Search failed after {elapsed:.2f} seconds")
            print(f"Error: {type(exc).__name__}: {exc}")

    if durations:
        print(f"\nSuccessful-search mean: {statistics.mean(durations):.2f}s")


def measure_page_reads():
    durations = []

    print("\n" + "=" * 70)
    print("WEBPAGE READ BASELINE")
    print("=" * 70)

    for url in WEBPAGE_URLS:
        print(f"\nURL: {url}")

        started = time.perf_counter()

        try:
            page = read_webpage(url)
            elapsed = time.perf_counter() - started
            durations.append(elapsed)

            if page is None:
                print("Page: unavailable")
            else:
                print(f"Title: {page.title}")
                print(f"Content characters: {len(page.content)}")

            print(f"Elapsed: {elapsed:.2f} seconds")

        except Exception as exc:
            elapsed = time.perf_counter() - started

            print(f"Read failed after {elapsed:.2f} seconds")
            print(f"Error: {type(exc).__name__}: {exc}")

    if durations:
        print(f"\nSuccessful-call mean: {statistics.mean(durations):.2f}s")


def main():
    measure_searches()
    measure_page_reads()


if __name__ == "__main__":
    main()
