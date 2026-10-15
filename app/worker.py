"""Separate PostgreSQL-backed worker for durable research jobs.

Run with ``python -m app.worker`` from the project root.
"""

import logging
import os
import signal
import socket
import threading
from typing import Any
from uuid import UUID, uuid4

from app.database import repository
from app.job_config import (
    is_recoverable_error,
    job_backoff_seconds,
    job_lease_seconds,
    job_poll_seconds,
)
from app.observability.tracing import configure_observability
from app.services.research_service import (
    ResearchCancelled,
    execute_research,
)


logger = logging.getLogger(__name__)


class JobLeaseLost(Exception):
    """Raised when this worker no longer owns its job lease."""


class WorkerStopping(Exception):
    """Raised when the process is shutting down at a safe checkpoint."""


def process_job(
    job: dict[str, Any],
    worker_id: str,
    shutdown_event: threading.Event,
    lease_seconds: int,
) -> None:
    """Run one claimed job while renewing its lease in the background."""

    job_id = UUID(job["id"])
    session_id = UUID(job["session_id"])
    stop_heartbeat = threading.Event()
    lease_lost = threading.Event()
    heartbeat_interval = max(1, lease_seconds // 3)

    def renew_lease() -> None:
        while not stop_heartbeat.wait(heartbeat_interval):
            try:
                if not repository.renew_job_lease(
                    job_id,
                    worker_id,
                    lease_seconds,
                ):
                    lease_lost.set()
                    logger.warning(
                        "Worker %s lost lease for job %s.",
                        worker_id,
                        job_id,
                    )
                    return
            except Exception:
                logger.exception(
                    "Could not renew lease for job %s.",
                    job_id,
                )
                lease_lost.set()

    heartbeat = threading.Thread(
        target=renew_lease,
        name=f"lease-{job_id}",
        daemon=True,
    )
    heartbeat.start(    )

    def check_control() -> None:
        if lease_lost.is_set():
            raise JobLeaseLost()

        cancel_requested = repository.get_job_control(
            job_id,
            worker_id,
        )
        if cancel_requested is None:
            lease_lost.set()
            raise JobLeaseLost()
        if cancel_requested:
            raise ResearchCancelled()
        if shutdown_event.is_set():
            raise WorkerStopping()

    payload = job["payload"]
    try:
        outcome = execute_research(
            session_id=session_id,
            initial_state=payload.get("initial_state"),
            resume_decision=payload.get("resume_decision"),
            raise_errors=True,
            manage_session_status=False,
            cancellation_check=check_control,
        )
        check_control()
        if not repository.complete_job(job_id, worker_id, outcome):
            logger.warning(
                "Job %s finished after its lease was lost; leaving it to recovery.",
                job_id,
            )
    except ResearchCancelled:
        repository.complete_job(job_id, worker_id, "cancelled")
    except WorkerStopping:
        repository.release_job_for_recovery(job_id, worker_id)
    except JobLeaseLost:
        logger.warning(
            "Stopping job %s because worker %s no longer owns its lease.",
            job_id,
            worker_id,
        )
    except Exception as exc:
        logger.exception("Research job %s failed.", job_id)
        repository.fail_or_retry_job(
            job_id=job_id,
            worker_id=worker_id,
            error_message=str(exc),
            retryable=is_recoverable_error(exc),
            backoff_seconds=job_backoff_seconds(job["attempt_count"]),
        )
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=heartbeat_interval + 1)


def run_worker(
    worker_id: str | None = None,
    shutdown_event: threading.Event | None = None,
    poll_seconds: int | None = None,
    lease_seconds: int | None = None,
) -> None:
    """Continuously claim jobs until the shutdown event is set."""

    worker_id = worker_id or (
        f"{socket.gethostname()}:{os.getpid()}:{uuid4()}"
    )
    shutdown_event = shutdown_event or threading.Event()
    poll_seconds = poll_seconds or job_poll_seconds()
    lease_seconds = lease_seconds or job_lease_seconds()

    logger.info("Research worker %s started.", worker_id)

    while not shutdown_event.is_set():
        try:
            job = repository.claim_next_job(worker_id, lease_seconds)
        except Exception:
            logger.exception("Could not poll the durable research queue.")
            shutdown_event.wait(poll_seconds)
            continue

        if job is None:
            shutdown_event.wait(poll_seconds)
            continue

        logger.info(
            "Claimed research job %s (attempt %s/%s).",
            job["id"],
            job["attempt_count"],
            job["max_attempts"],
        )
        try:
            process_job(job, worker_id, shutdown_event, lease_seconds)
        except Exception:
            logger.exception(
                "Worker failed while finalizing job %s; its lease will be recovered.",
                job["id"],
            )

    logger.info("Research worker %s stopped.", worker_id)


def main() -> None:
    """Configure logging and handle process shutdown signals."""

    configure_observability()
    shutdown_event = threading.Event()

    def request_shutdown(signum: int, frame: Any) -> None:
        del signum, frame
        logger.info("Research worker shutdown requested.")
        shutdown_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    run_worker(shutdown_event=shutdown_event)


if __name__ == "__main__":
    main()
