"""
Structured logging and node-level observability for the
Deep Research Agent.

Provides:
- Readable console logging.
- Node execution metrics.
- PostgreSQL persistence for node events only.
- Session-aware tracing.
"""

import inspect
import logging
import os
import time

from contextvars import ContextVar
from threading import Lock
from typing import Any, Callable
from uuid import UUID

from app.database.repository import save_observability_event
from langchain_core.runnables import RunnableConfig

# ============================================================
# SESSION CONTEXT
# ============================================================

_session_id: ContextVar[str | None] = ContextVar(
    "research_session_id",
    default=None,
)


def set_session_context(session_id: str) -> None:
    """Associate subsequent logs with a research session."""
    _session_id.set(str(session_id))


def clear_session_context() -> None:
    """Clear the active research session context."""
    _session_id.set(None)


# ============================================================
# RUN METRICS
# ============================================================

_metrics_lock = Lock()

_run_metrics: dict[str, Any] = {
    "nodes": {},
    "started_at": None,
}


def reset_run_metrics() -> None:
    """Reset node metrics before a new research run."""
    with _metrics_lock:
        _run_metrics["nodes"] = {}
        _run_metrics["started_at"] = time.perf_counter()


def get_run_metrics() -> dict[str, Any]:
    """Return a snapshot of the current run metrics."""
    with _metrics_lock:
        return {
            "started_at": _run_metrics["started_at"],
            "nodes": {
                name: metrics.copy()
                for name, metrics in _run_metrics["nodes"].items()
            },
        }


def _record_node_result(
    node_name: str,
    status: str,
    duration_ms: float,
) -> None:
    """Update aggregate metrics for a node."""
    with _metrics_lock:
        metrics = _run_metrics["nodes"].setdefault(
            node_name,
            {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "duration_ms": 0.0,
                "last_status": None,
            },
        )

        metrics["calls"] += 1
        metrics["duration_ms"] = round(
            metrics["duration_ms"] + duration_ms,
            2,
        )
        metrics["last_status"] = status

        if status == "success":
            metrics["successes"] += 1
        elif status == "failure":
            metrics["failures"] += 1


# ============================================================
# READABLE CONSOLE FORMATTER
# ============================================================

class ConsoleFormatter(logging.Formatter):
    """Format application logs as concise, readable lines."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        event_type = getattr(record, "event_type", None)
        node_name = getattr(record, "node", None)
        duration_ms = getattr(record, "duration_ms", None)

        if event_type == "node_started":
            message = f"[Node] {node_name} started"

        elif event_type == "node_completed":
            message = f"[Node] {node_name} completed"
            if duration_ms is not None:
                message += f" ({duration_ms / 1000:.2f}s)"

        elif event_type == "node_failed":
            message = f"[Node] {node_name} failed"

        elif event_type == "node_interrupted":
            message = f"[Node] {node_name} paused for review"

        prefix = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARN",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "CRITICAL",
        }.get(record.levelno, record.levelname)

        return f"{prefix}: {message}"


# ============================================================
# POSTGRESQL EVENT HANDLER
# ============================================================

class PostgreSQLEventHandler(logging.Handler):
    """
    Persist node execution events only.

    Session lifecycle and human-review events are persisted by
    their explicit application call sites to avoid duplicates.
    """

    PERSISTED_EVENT_TYPES = {
        "node_started",
        "node_completed",
        "node_failed",
    }

    def emit(self, record: logging.LogRecord) -> None:
        event_type = getattr(record, "event_type", None)

        if event_type not in self.PERSISTED_EVENT_TYPES:
            return

        raw_session_id = getattr(
            record,
            "session_id",
            _session_id.get(),
        )

        try:
            session_id = (
                UUID(str(raw_session_id))
                if raw_session_id
                else None
            )

            reserved_fields = (
                logging.makeLogRecord({}).__dict__.keys()
            )

            details = {
                key: value
                for key, value in record.__dict__.items()
                if key not in reserved_fields
                and key not in {
                    "event_type",
                    "session_id",
                    "node",
                    "duration_ms",
                }
                and not key.startswith("_")
            }

            save_observability_event(
                event_type=event_type,
                message=record.getMessage(),
                session_id=session_id,
                node_name=getattr(record, "node", None),
                log_level=record.levelname,
                duration_ms=getattr(record, "duration_ms", None),
                details=details,
            )

        except Exception:
            # Never recursively log through the same handler.
            try:
                fallback = logging.StreamHandler()
                fallback.setFormatter(ConsoleFormatter())
                fallback.emit(
                    logging.LogRecord(
                        name="observability.persistence",
                        level=logging.ERROR,
                        pathname=__file__,
                        lineno=0,
                        msg=(
                            "Failed to persist observability event "
                            f"{event_type!r}"
                        ),
                        args=(),
                        exc_info=None,
                    )
                )
            except Exception:
                pass


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

def configure_observability() -> None:
    """Configure readable console logging and PostgreSQL storage."""

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    root_logger = logging.getLogger()

    if not getattr(
        root_logger,
        "_research_observability_configured",
        False,
    ):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ConsoleFormatter())

        database_handler = PostgreSQLEventHandler()
        database_handler.setLevel(logging.INFO)

        root_logger.addHandler(console_handler)
        root_logger.addHandler(database_handler)

        root_logger._research_observability_configured = True

    root_logger.setLevel(
        getattr(logging, log_level, logging.INFO)
    )

    # Suppress routine third-party HTTP and networking logs.
    # Warnings and errors remain visible.
    for logger_name in (
        "httpx",
        "httpcore",
        "urllib3",
        "primp",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


logger = logging.getLogger("deep_research_agent")


# ============================================================
# NODE INSTRUMENTATION
# ============================================================

def instrument_node(
    node_name: str,
    node_function: Callable,
) -> Callable:
    """
    Wrap a LangGraph node with timing, metrics, logging,
    and PostgreSQL persistence.
    """

    signature = inspect.signature(node_function)

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    accepts_config = (
        "config" in signature.parameters
        or accepts_kwargs
    )

    accepted_kwargs = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    def wrapped_node(
        state: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        started_at = time.perf_counter()

        configurable = (
            config.get("configurable", {})
            if isinstance(config, dict)
            else {}
        )

        session_id = configurable.get(
            "thread_id",
            _session_id.get(),
        )

        extra = {
            "node": node_name,
            "session_id": (
                str(session_id)
                if session_id is not None
                else None
            ),
        }

        logger.info(
            "Node started",
            extra={
                **extra,
                "event_type": "node_started",
            },
        )

        try:
            if accepts_kwargs:
                forwarded_kwargs = dict(kwargs)
            else:
                forwarded_kwargs = {
                    key: value
                    for key, value in kwargs.items()
                    if key in accepted_kwargs
                }

            if accepts_config and config is not None:
                forwarded_kwargs["config"] = config

            result = node_function(
                state,
                **forwarded_kwargs,
            )

            duration_ms = (
                time.perf_counter() - started_at
            ) * 1000

            _record_node_result(
                node_name,
                "success",
                duration_ms,
            )

            logger.info(
                "Node completed",
                extra={
                    **extra,
                    "event_type": "node_completed",
                    "duration_ms": round(duration_ms, 2),
                },
            )

            return result

        except Exception as error:
            duration_ms = (
                time.perf_counter() - started_at
            ) * 1000

            if type(error).__name__ in {
                "GraphInterrupt",
                "NodeInterrupt",
            }:
                logger.info(
                    "Node paused for graph interruption",
                    extra={
                        **extra,
                        "event_type": "node_interrupted",
                        "duration_ms": round(duration_ms, 2),
                    },
                )
                raise

            _record_node_result(
                node_name,
                "failure",
                duration_ms,
            )

            logger.exception(
                "Node failed",
                extra={
                    **extra,
                    "event_type": "node_failed",
                    "duration_ms": round(duration_ms, 2),
                },
            )

            raise

    wrapped_node.__name__ = getattr(
        node_function,
        "__name__",
        node_name,
    )
    wrapped_node.__doc__ = getattr(
        node_function,
        "__doc__",
        None,
    )

    return wrapped_node
