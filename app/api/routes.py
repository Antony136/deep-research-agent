"""
FastAPI routes for the Deep Research Agent.

Provides endpoints for research sessions, session status,
pending human reviews, and review submissions.
"""

import logging
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
from app.job_config import max_job_attempts
from app.database.repository import (
    cancel_latest_job,
    create_session_with_job,
    enqueue_retry_job,
    enqueue_review_job,
    pause_latest_job,
    resume_paused_job,
)
from app.services.research_service import (
    get_pending_review,
    get_research_session,
    list_research_sessions,
    build_initial_state,
    to_json_compatible,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Research"])

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
        initial_state = build_initial_state(
            question=question,
            require_initial_plan_approval=request.require_initial_plan_approval,
            require_initial_research_approval=request.require_initial_research_approval,
            require_adaptive_research_approval=request.require_adaptive_research_approval,
            require_final_report_approval=request.require_final_report_approval,
            max_research_rounds=request.max_research_rounds,
            max_total_research_questions=request.max_total_research_questions,
        )
        session_id, job_id = create_session_with_job(
            question=question,
            payload={
                "initial_state": to_json_compatible(initial_state),
            },
            max_attempts=max_job_attempts(),
        )

        return ResearchCreatedResponse(
            session_id=str(session_id),
            status="created",
            message="Research session and durable job queued.",
            job_id=str(job_id),
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

        try:
            job = enqueue_review_job(
                session_id,
                validated_decision,
                max_job_attempts(),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

        return ReviewSubmissionResponse(
            session_id=str(session_id),
            status="running",
            message="Review accepted; research resumption queued.",
            job_id=job["id"],
            job_status=job["status"],
            attempt_count=job["attempt_count"],
            max_attempts=job["max_attempts"],
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
    """Retry failed work or resume a legacy session without a job record."""

    try:
        session = get_research_session(session_id)

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Research session not found.",
            )

        initial_state = build_initial_state(session["question"])
        job = enqueue_retry_job(
            session_id,
            max_job_attempts(),
            fallback_payload={
                "initial_state": to_json_compatible(initial_state),
            },
        )

        return ReviewSubmissionResponse(
            session_id=str(session_id),
            status="running",
            message="Research retry queued from its latest checkpoint.",
            job_id=job["id"],
            job_status=job["status"],
            attempt_count=job["attempt_count"],
            max_attempts=job["max_attempts"],
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to retry research session %s.",
            session_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retry the research session.",
        ) from exc


@router.post(
    "/research/{session_id}/cancel",
    response_model=ReviewSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def cancel_research(
    session_id: UUID,
) -> ReviewSubmissionResponse:
    """Cancel queued research or request a safe worker stop."""

    try:
        job = cancel_latest_job(session_id)
        session_status = (
            "cancelled"
            if job["status"] == "cancelled"
            else "running"
        )
        message = (
            "Research job cancelled."
            if job["status"] == "cancelled"
            else "Cancellation requested; the worker will stop at its next checkpoint."
        )
        return ReviewSubmissionResponse(
            session_id=str(session_id),
            status=session_status,
            message=message,
            job_id=job["id"],
            job_status=job["status"],
            attempt_count=job["attempt_count"],
            max_attempts=job["max_attempts"],
            cancel_requested=job["cancel_requested"],
            last_error=job["last_error"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to cancel research session %s.",
            session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not cancel the research session.",
        ) from exc


@router.post(
    "/research/{session_id}/pause",
    response_model=ReviewSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def pause_research(
    session_id: UUID,
) -> ReviewSubmissionResponse:
    """Pause immediately if queued, otherwise at the next graph checkpoint."""

    try:
        job = pause_latest_job(session_id)
        is_paused = job["status"] == "paused"
        return ReviewSubmissionResponse(
            session_id=str(session_id),
            status="paused" if is_paused else "running",
            message=(
                "Research paused."
                if is_paused
                else "Pause requested; research will pause at the next workflow checkpoint."
            ),
            job_id=job["id"],
            job_status=job["status"],
            attempt_count=job["attempt_count"],
            max_attempts=job["max_attempts"],
            cancel_requested=job["cancel_requested"],
            pause_requested=job["pause_requested"],
            last_error=job["last_error"],
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to pause research session %s.", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not pause the research session.",
        ) from exc


@router.post(
    "/research/{session_id}/resume",
    response_model=ReviewSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def resume_research(
    session_id: UUID,
) -> ReviewSubmissionResponse:
    """Queue a paused research job from its latest graph checkpoint."""

    try:
        job = resume_paused_job(session_id)
        return ReviewSubmissionResponse(
            session_id=str(session_id),
            status="running",
            message="Research resumed from its latest workflow checkpoint.",
            job_id=job["id"],
            job_status=job["status"],
            attempt_count=job["attempt_count"],
            max_attempts=job["max_attempts"],
            cancel_requested=job["cancel_requested"],
            pause_requested=job["pause_requested"],
            last_error=job["last_error"],
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to resume research session %s.", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not resume the research session.",
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

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

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
        from app.database.repository import get_node_execution_metrics

        return {
            "started_at": None,
            "nodes": get_node_execution_metrics(),
        }

    except Exception as exc:
        logger.exception("Failed to retrieve run metrics.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve run metrics.",
        ) from exc
