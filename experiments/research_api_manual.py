"""
End-to-end API tests for the Deep Research Agent.

Tests:
1. Health check
2. Root endpoint
3. List research sessions
4. Create a research session
5. Retrieve session status and results
6. Retrieve and submit human-review decisions
7. Resume research and verify completion

Run the FastAPI server separately before executing this script.
"""

import json
import sys
import time
from typing import Any

import requests


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BASE_URL = "http://127.0.0.1:8000"
API_URL = f"{BASE_URL}/api"

REQUEST_TIMEOUT = 30
POLL_INTERVAL_SECONDS = 2
MAX_WAIT_SECONDS = 600

# Change this question if desired.
RESEARCH_QUESTION = (
    "What are the main applications of retrieval-augmented "
    "generation, and what are its limitations?"
)


# ---------------------------------------------------------
# TEST UTILITIES
# ---------------------------------------------------------

def print_header(title: str) -> None:
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def print_response(response: requests.Response) -> None:
    print(f"HTTP Status: {response.status_code}")

    try:
        print(
            json.dumps(
                response.json(),
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    except ValueError:
        print(response.text)


def request(
    method: str,
    endpoint: str,
    **kwargs: Any,
) -> requests.Response:
    """Send an HTTP request and report connection errors."""

    url = f"{BASE_URL}{endpoint}"

    try:
        return requests.request(
            method,
            url,
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
    except requests.RequestException as exc:
        print(f"\nRequest failed: {method} {url}")
        print(exc)
        raise


def require_status(
    response: requests.Response,
    expected_status: int,
) -> dict[str, Any]:
    """Validate the HTTP status and return a JSON response."""

    print_response(response)

    if response.status_code != expected_status:
        raise AssertionError(
            f"Expected HTTP {expected_status}, "
            f"received HTTP {response.status_code}."
        )

    try:
        return response.json()
    except ValueError as exc:
        raise AssertionError(
            "Expected a JSON response."
        ) from exc


# ---------------------------------------------------------
# TEST 1: HEALTH CHECK
# ---------------------------------------------------------

def test_health() -> None:
    print_header("TEST 1: HEALTH CHECK")

    response = request("GET", "/api/health")
    data = require_status(response, 200)

    assert data["status"] == "ok"
    assert data["service"] == "deep-research-agent"

    print("PASS: Health check")


# ---------------------------------------------------------
# TEST 2: ROOT ENDPOINT
# ---------------------------------------------------------

def test_root() -> None:
    print_header("TEST 2: ROOT ENDPOINT")

    response = request("GET", "/")
    data = require_status(response, 200)

    assert data["service"] == "Deep Research Agent API"
    assert data["docs"] == "/docs"

    print("PASS: Root endpoint")


# ---------------------------------------------------------
# TEST 3: LIST SESSIONS
# ---------------------------------------------------------

def test_list_sessions() -> None:
    print_header("TEST 3: LIST RESEARCH SESSIONS")

    response = request(
        "GET",
        "/api/research",
        params={"limit": 5},
    )

    data = require_status(response, 200)

    assert "sessions" in data
    assert isinstance(data["sessions"], list)

    print("PASS: List sessions")


# ---------------------------------------------------------
# TEST 4: CREATE SESSION
# ---------------------------------------------------------

def test_create_session() -> str:
    print_header("TEST 4: CREATE RESEARCH SESSION")

    payload = {
        "question": RESEARCH_QUESTION,
        "require_initial_plan_approval": True,
        "require_initial_research_approval": True,
        "require_adaptive_research_approval": True,
        "require_final_report_approval": True,
        "max_research_rounds": 2,
        "max_total_research_questions": 5,
    }

    response = request(
        "POST",
        "/api/research",
        json=payload,
    )

    data = require_status(response, 202)

    session_id = data.get("session_id")

    assert session_id, "The API did not return a session ID."

    print(f"\nCreated session: {session_id}")
    print("PASS: Create research session")

    return session_id


# ---------------------------------------------------------
# TEST 5: GET SESSION
# ---------------------------------------------------------

def test_get_session(
    session_id: str,
) -> dict[str, Any]:
    print_header("TEST 5: GET RESEARCH SESSION")

    response = request(
        "GET",
        f"/api/research/{session_id}",
    )

    data = require_status(response, 200)

    assert data["id"] == session_id
    assert data["question"] == RESEARCH_QUESTION
    assert data["status"] in {
        "created",
        "running",
        "waiting_for_review",
        "completed",
        "failed",
    }

    print("PASS: Get session")
    return data


# ---------------------------------------------------------
# POLL SESSION STATUS
# ---------------------------------------------------------

def wait_for_session_state(
    session_id: str,
) -> dict[str, Any]:
    """
    Wait until the session needs review or reaches a terminal state.
    """

    print_header("WAITING FOR RESEARCH")

    deadline = time.monotonic() + MAX_WAIT_SECONDS

    while time.monotonic() < deadline:
        response = request(
            "GET",
            f"/api/research/{session_id}",
        )

        data = require_status(response, 200)
        current_status = data["status"]

        print(f"\nCurrent status: {current_status}")

        if current_status in {
            "waiting_for_review",
            "completed",
            "failed",
        }:
            return data

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Session did not reach a review or terminal state "
        f"within {MAX_WAIT_SECONDS} seconds."
    )


# ---------------------------------------------------------
# TEST 6: GET AND SUBMIT REVIEW
# ---------------------------------------------------------

def test_review_workflow(
    session_id: str,
) -> None:
    """
    Retrieve each pending review and submit a safe test decision.

    Decisions:
    - Initial plan: approve
    - Initial research: approve
    - Adaptive review: approve
    - Final report: approve

    The API returns to waiting_for_review whenever another
    human decision is needed.
    """

    print_header("TEST 6: HUMAN REVIEW WORKFLOW")

    while True:
        session = wait_for_session_state(session_id)
        current_status = session["status"]

        if current_status == "completed":
            print("Research completed.")
            return

        if current_status == "failed":
            raise AssertionError(
                "Research failed: "
                f"{session.get('error_message')}"
            )

        if current_status != "waiting_for_review":
            raise AssertionError(
                f"Unexpected session status: {current_status}"
            )

        # Retrieve the pending review.
        response = request(
            "GET",
            f"/api/research/{session_id}/review",
        )

        review_data = require_status(response, 200)
        review = review_data["review"]
        review_type = review.get("type")

        print(f"\nPending review type: {review_type}")
        print("Review payload:")
        print(
            json.dumps(
                review,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        # Automatically approve each supported review type.
        # Initial-plan revision is deliberately not tested here.
        actions = {
            "initial_plan_review": "approve",
            "initial_research_review": "approve",
            "adaptive_research_review": "approve",
            "final_report_review": "approve",
        }

        action = actions.get(review_type)

        if action is None:
            raise AssertionError(
                f"Unknown review type: {review_type}"
            )

        decision = {
            "action": action,
            "reason": "Approved by automated API integration test.",
        }

        print(f"\nSubmitting decision: {decision}")

        response = request(
            "POST",
            f"/api/research/{session_id}/review",
            json=decision,
        )

        result = require_status(response, 202)

        assert result["session_id"] == session_id

        print("PASS: Review decision accepted")

        # Polling will detect the next review or completion.
        time.sleep(POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------
# TEST 7: VERIFY FINAL RESULT
# ---------------------------------------------------------

def test_final_result(
    session_id: str,
) -> None:
    print_header("TEST 7: VERIFY FINAL RESULT")

    response = request(
        "GET",
        f"/api/research/{session_id}",
    )

    data = require_status(response, 200)

    assert data["status"] == "completed", (
        f"Expected completed status, got {data['status']}"
    )

    print("\nFinal session status:", data["status"])
    print("\nFinal report:")

    report = data.get("final_report")

    if report is None:
        print("No final_report was returned.")
    else:
        print(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

    print("\nResearch state keys:")
    research_state = data.get("research_state") or {}

    for key in research_state:
        print(f"- {key}")

    print("PASS: Final result verified")


# ---------------------------------------------------------
# TEST 8: VALIDATION ERROR
# ---------------------------------------------------------

def test_invalid_request() -> None:
    print_header("TEST 8: INVALID REQUEST VALIDATION")

    response = request(
        "POST",
        "/api/research",
        json={"question": ""},
    )

    require_status(response, 422)

    print("PASS: Empty question rejected")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:
    print_header("DEEP RESEARCH AGENT API TEST SUITE")

    session_id = None

    try:
        test_health()
        test_root()
        test_list_sessions()
        test_invalid_request()

        session_id = test_create_session()

        test_get_session(session_id)

        test_review_workflow(session_id)

        test_final_result(session_id)

        print_header("ALL API TESTS PASSED")
        print(f"Tested session ID: {session_id}")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")

        if session_id:
            print(f"Session ID: {session_id}")

        sys.exit(130)

    except Exception as exc:
        print_header("API TEST FAILED")
        print(f"{type(exc).__name__}: {exc}")

        if session_id:
            print(f"Session ID: {session_id}")
            print(
                f"Inspect session: "
                f"{API_URL}/research/{session_id}"
            )

        sys.exit(1)


if __name__ == "__main__":
    main()
