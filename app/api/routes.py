"""
FastAPI routes for the Deep Research Agent.

Provides endpoints for research sessions, session status,
pending human reviews, and review submissions.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.api import (
    HealthResponse,
    ObservabilityEventResponse,
    ObservabilityEventsListResponse,
    PendingReviewResponse,
    ResearchCreatedResponse,
    ResearchRequest,
    ResearchSessionListResponse,
    ResearchSessionResponse,
    ResearchSessionSummary,
    ReviewDecisionRequest,
    ReviewSubmissionResponse,
    SessionDeletedResponse,
)
from app.services.research_service import (
    get_pending_review,
    get_research_session,
    list_research_sessions,
    start_research,
    submit_review,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Research"])

# Temporary in-process executor for background research jobs.
# A durable distributed worker can replace this during
# production hardening.
executor = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="research-worker",
)


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
)
def health_check() -> HealthResponse:
    """Check whether the API process is responding."""

    return HealthResponse(status="ok")


# ---------------------------------------------------------
# CREATE RESEARCH
# ---------------------------------------------------------

@router.post(
    "/research",
    response_model=ResearchCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_research(
    request: ResearchRequest,
) -> ResearchCreatedResponse:
    """Create a session and schedule its research workflow."""

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Research question cannot be empty.",
        )

    try:
        # Create the session immediately so the frontend
        # receives its ID before the research starts.
        from app.database.repository import create_session

        session_id = create_session(question)

        from app.services.research_service import (
            build_initial_state,
            execute_research,
        )

        initial_state = build_initial_state(
            question=question,
            require_initial_plan_approval=request.require_initial_plan_approval,
            require_initial_research_approval=request.require_initial_research_approval,
            require_adaptive_research_approval=request.require_adaptive_research_approval,
            require_final_report_approval=request.require_final_report_approval,
            max_research_rounds=request.max_research_rounds,
            max_total_research_questions=request.max_total_research_questions,
        )

        executor.submit(
            execute_research,
            session_id,
            initial_state,
            None,
        )

        return ResearchCreatedResponse(
            session_id=str(session_id),
            status="created",
            message="Research session created and scheduled.",
        )

    except Exception as exc:
        logger.exception("Failed to create research session.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create the research session.",
        ) from exc


# ---------------------------------------------------------
# LIST RESEARCH SESSIONS
# ---------------------------------------------------------

@router.get(
    "/research",
    response_model=ResearchSessionListResponse,
)
def get_research_sessions(
    limit: int = Query(default=20, ge=1, le=100),
) -> ResearchSessionListResponse:
    """List recent research sessions."""

    try:
        sessions = list_research_sessions(limit=limit)

        return ResearchSessionListResponse(
            sessions=[
                ResearchSessionSummary(**session)
                for session in sessions
            ]
        )

    except Exception as exc:
        logger.exception("Failed to list research sessions.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve research sessions.",
        ) from exc


# ---------------------------------------------------------
# GET A RESEARCH SESSION
# ---------------------------------------------------------

@router.get(
    "/research/{session_id}",
    response_model=ResearchSessionResponse,
)
def get_research_session_endpoint(
    session_id: UUID,
) -> ResearchSessionResponse:
    """Retrieve the status and persisted results of a session."""

    try:
        session = get_research_session(session_id)

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Research session not found.",
            )

        return ResearchSessionResponse(**session)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Failed to retrieve research session %s.",
            session_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve the research session.",
        ) from exc


# ---------------------------------------------------------
# GET PENDING HUMAN REVIEW
# ---------------------------------------------------------

@router.get(
    "/research/{session_id}/review",
    response_model=PendingReviewResponse,
)
def get_pending_review_endpoint(
    session_id: UUID,
) -> PendingReviewResponse:
    """Retrieve the review currently awaiting a decision."""

    try:
        session = get_research_session(session_id)

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Research session not found.",
            )

        review = get_pending_review(session_id)

        if review is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This session has no pending human review."
                ),
            )

        return PendingReviewResponse(
            session_id=str(session_id),
            review=review,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Failed to retrieve pending review for %s.",
            session_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve the pending review.",
        ) from exc


# ---------------------------------------------------------
# SUBMIT HUMAN REVIEW
# ---------------------------------------------------------

@router.post(
    "/research/{session_id}/review",
    response_model=ReviewSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_review_endpoint(
    session_id: UUID,
    request: ReviewDecisionRequest,
) -> ReviewSubmissionResponse:
    """Validate a review and schedule graph resumption."""

    try:
        session = get_research_session(session_id)

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Research session not found.",
            )

        if session["status"] != "waiting_for_review":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "The session is not waiting for human review."
                ),
            )

        review = get_pending_review(session_id)

        if review is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No pending review was found.",
            )

        decision = request.model_dump(exclude_none=True)

        # Validate against the actual interrupt before scheduling.
        from app.services.research_service import (
            validate_review_decision,
        )

        validated_decision = validate_review_decision(
            review,
            decision,
        )

        # Queue only after validation succeeds.
        from app.services.research_service import execute_research

        executor.submit(
            execute_research,
            session_id,
            None,
            validated_decision,
        )

        return ReviewSubmissionResponse(
            session_id=str(session_id),
            status="running",
            message="Review accepted; research resumption scheduled.",
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to submit review for session %s.",
            session_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not submit the review.",
        ) from exc


# ---------------------------------------------------------
# RETRY A FAILED OR INTERRUPTED SESSION
# ---------------------------------------------------------

@router.post(
    "/research/{session_id}/retry",
    response_model=ReviewSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def retry_research(
    session_id: UUID,
) -> ReviewSubmissionResponse:
    """Retry a failed or stalled research session."""

    try:
        session = get_research_session(session_id)

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Research session not found.",
            )

        from app.database.repository import update_session_status
        from app.services.research_service import (
            build_initial_state,
            execute_research,
        )

        update_session_status(session_id, "running")

        initial_state = build_initial_state(
            question=session["question"],
        )

        executor.submit(
            execute_research,
            session_id,
            initial_state,
            None,
        )

        return ReviewSubmissionResponse(
            session_id=str(session_id),
            status="running",
            message="Research retry scheduled.",
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Failed to retry research session %s.",
            session_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retry the research session.",
        ) from exc


# ---------------------------------------------------------
# DELETE RESEARCH SESSION
# ---------------------------------------------------------

@router.delete(
    "/research/{session_id}",
    response_model=SessionDeletedResponse,
)
def delete_research_session_endpoint(
    session_id: UUID,
) -> SessionDeletedResponse:
    """Delete a research session and all associated data."""

    try:
        session = get_research_session(session_id)

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Research session not found.",
            )

        from app.database.repository import delete_session

        deleted = delete_session(session_id)

        return SessionDeletedResponse(
            deleted=deleted,
            session_id=str(session_id),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Failed to delete research session %s.",
            session_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not delete the research session.",
        ) from exc


# ---------------------------------------------------------
# SESSION OBSERVABILITY EVENTS
# ---------------------------------------------------------

@router.get(
    "/research/{session_id}/events",
    response_model=ObservabilityEventsListResponse,
)
def get_session_events_endpoint(
    session_id: UUID,
    limit: int = Query(default=100, ge=1, le=500),
) -> ObservabilityEventsListResponse:
    """Retrieve observability events recorded for this session."""

    try:
        from app.database.repository import get_observability_events

        events = get_observability_events(session_id=session_id, limit=limit)

        return ObservabilityEventsListResponse(
            events=[
                ObservabilityEventResponse(**event)
                for event in events
            ]
        )

    except Exception as exc:
        logger.exception(
            "Failed to retrieve events for session %s.",
            session_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve session observability events.",
        ) from exc


# ---------------------------------------------------------
# GLOBAL OBSERVABILITY EVENTS
# ---------------------------------------------------------

@router.get(
    "/observability/events",
    response_model=ObservabilityEventsListResponse,
)
def get_global_events_endpoint(
    limit: int = Query(default=50, ge=1, le=500),
) -> ObservabilityEventsListResponse:
    """Retrieve recent observability events across all sessions."""

    try:
        from app.database.repository import get_observability_events

        events = get_observability_events(session_id=None, limit=limit)

        return ObservabilityEventsListResponse(
            events=[
                ObservabilityEventResponse(**event)
                for event in events
            ]
        )

    except Exception as exc:
        logger.exception("Failed to retrieve global observability events.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve observability events.",
        ) from exc


# ---------------------------------------------------------
# RUN METRICS
# ---------------------------------------------------------

@router.get(
    "/observability/metrics",
)
def get_metrics_endpoint():
    """Retrieve node execution metrics snapshot."""

    try:
        from app.observability.tracing import get_run_metrics

        return get_run_metrics()

    except Exception as exc:
        logger.exception("Failed to retrieve run metrics.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve run metrics.",
        ) from exc

