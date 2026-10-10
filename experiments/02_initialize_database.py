"""
Initialize the Deep Research Agent database.
"""

from app.database.schema import initialize_database


def main() -> None:
    print("=" * 60)
    print("DATABASE INITIALIZATION")
    print("=" * 60)

    initialize_database()

    print("\nCreated or verified:")
    print("- research_sessions")
    print("- research_questions")
    print("- research_evidence")
    print("- Supporting indexes")

    print("\nDatabase initialization successful.")


if __name__ == "__main__":
    main()