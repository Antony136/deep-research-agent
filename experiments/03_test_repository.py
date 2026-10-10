"""
Test research-session repository operations.
"""

from app.database.repository import (
    create_session,
    get_session,
    list_sessions,
    save_report,
    save_research_state,
    update_session_status,
)


def main() -> None:
    question = (
        "What are the main approaches to improving "
        "retrieval quality in RAG systems?"
    )

    print("=" * 60)
    print("RESEARCH REPOSITORY TEST")
    print("=" * 60)

    session_id = create_session(question)
    print(f"\nCreated session: {session_id}")

    save_research_state(
        session_id,
        {
            "question": question,
            "research_questions": [],
            "evidence": [],
            "research_complete": False,
        },
    )
    print("Research state saved.")

    save_report(
        session_id,
        {
            "title": "Repository Test Report",
            "summary": "Testing PostgreSQL persistence.",
            "findings": [],
            "sources": [],
        },
    )
    print("Report saved.")

    update_session_status(session_id, "completed")
    print("Session marked as completed.")

    session = get_session(session_id)

    assert session is not None
    assert session["status"] == "completed"
    assert session["research_state"]["question"] == question
    assert session["final_report"]["title"] == (
        "Repository Test Report"
    )

    print("\nRetrieved session:")
    print(f"ID: {session['id']}")
    print(f"Status: {session['status']}")
    print(f"Report: {session['final_report']['title']}")

    sessions = list_sessions(limit=5)
    print(f"\nRecent sessions retrieved: {len(sessions)}")

    print("\nRepository test passed.")


if __name__ == "__main__":
    main()
