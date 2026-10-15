"""
Database schema initialization for the Deep Research Agent.
"""

from app.database.connection import get_connection


# ---------------------------------------------------------
# RESEARCH SESSIONS
# ---------------------------------------------------------

CREATE_RESEARCH_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS research_sessions (
    id UUID PRIMARY KEY,
    question TEXT NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'created',
    research_state JSONB,
    final_report JSONB,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    CONSTRAINT research_sessions_status_check
        CHECK (
            status IN (
                'created',
                'running',
                'waiting_for_review',
                'completed',
                'failed',
                'cancelled',
                'paused'
            )
        )
);
"""


# ---------------------------------------------------------
# DURABLE RESEARCH JOBS
# ---------------------------------------------------------

CREATE_RESEARCH_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS research_jobs (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL
        REFERENCES research_sessions(id)
        ON DELETE CASCADE,
    payload JSONB NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'queued',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    lease_owner VARCHAR(255),
    lease_expires_at TIMESTAMPTZ,
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
    pause_requested BOOLEAN NOT NULL DEFAULT FALSE,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    CONSTRAINT research_jobs_status_check
        CHECK (
            status IN (
                'queued',
                'running',
                'retry_wait',
                'awaiting_review',
                'completed',
                'failed',
                'cancelled',
                'paused'
            )
        ),
    CONSTRAINT research_jobs_attempts_check
        CHECK (
            attempt_count >= 0
            AND max_attempts >= 1
        )
);
"""


# ---------------------------------------------------------
# RESEARCH QUESTIONS
# ---------------------------------------------------------

CREATE_RESEARCH_QUESTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS research_questions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id UUID NOT NULL
        REFERENCES research_sessions(id)
        ON DELETE CASCADE,
    question_number INTEGER NOT NULL,
    question TEXT NOT NULL,
    search_queries JSONB NOT NULL DEFAULT '[]'::jsonb,
    parent_question_number INTEGER,
    status VARCHAR(30) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT research_questions_session_number_unique
        UNIQUE (session_id, question_number),

    CONSTRAINT research_questions_status_check
        CHECK (
            status IN (
                'pending',
                'researching',
                'completed',
                'failed'
            )
        )
);
"""


# ---------------------------------------------------------
# RESEARCH EVIDENCE
# ---------------------------------------------------------

CREATE_RESEARCH_EVIDENCE_TABLE = """
CREATE TABLE IF NOT EXISTS research_evidence (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id UUID NOT NULL
        REFERENCES research_sessions(id)
        ON DELETE CASCADE,
    evidence_number INTEGER NOT NULL,
    research_question_number INTEGER NOT NULL,
    claim TEXT NOT NULL,
    source_url TEXT NOT NULL,
    supporting_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT research_evidence_session_number_unique
        UNIQUE (session_id, evidence_number)
);
"""


# ---------------------------------------------------------
# OBSERVABILITY EVENTS
# ---------------------------------------------------------

CREATE_OBSERVABILITY_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS observability_events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    session_id UUID
        REFERENCES research_sessions(id)
        ON DELETE SET NULL,

    event_type VARCHAR(100) NOT NULL,
    node_name VARCHAR(150),
    log_level VARCHAR(20) NOT NULL DEFAULT 'INFO',
    message TEXT NOT NULL,
    duration_ms DOUBLE PRECISION,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


# ---------------------------------------------------------
# INDEXES
# ---------------------------------------------------------

CREATE_INDEXES = [
    """
    CREATE INDEX IF NOT EXISTS idx_research_sessions_created_at
    ON research_sessions (created_at DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_research_sessions_status
    ON research_sessions (status);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_research_jobs_available
    ON research_jobs (available_at, created_at)
    WHERE status IN ('queued', 'retry_wait');
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_research_jobs_expired_lease
    ON research_jobs (lease_expires_at)
    WHERE status = 'running';
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_research_jobs_session_created
    ON research_jobs (session_id, created_at DESC, id DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_research_questions_session
    ON research_questions (session_id, question_number);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_research_evidence_session
    ON research_evidence (session_id, evidence_number);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_observability_events_session_time
    ON observability_events (session_id, created_at DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_observability_events_type
    ON observability_events (event_type, created_at DESC);
    """,
]


# ---------------------------------------------------------
# INITIALIZE DATABASE
# ---------------------------------------------------------

def initialize_database() -> None:
    """
    Create research tables, observability storage, and indexes.

    Also migrate the existing session-status constraint and
    create durable job storage.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_RESEARCH_SESSIONS_TABLE)

            # Update the constraint on existing databases.
            cursor.execute(
                """
                ALTER TABLE research_sessions
                DROP CONSTRAINT IF EXISTS
                    research_sessions_status_check;
                """
            )

            cursor.execute(
                """
                ALTER TABLE research_sessions
                ADD CONSTRAINT research_sessions_status_check
                CHECK (
                    status IN (
                        'created',
                        'running',
                        'waiting_for_review',
                        'completed',
                        'failed',
                        'cancelled',
                        'paused'
                    )
                );
                """
            )

            cursor.execute(CREATE_RESEARCH_JOBS_TABLE)
            cursor.execute(
                """
                ALTER TABLE research_jobs
                ADD COLUMN IF NOT EXISTS pause_requested
                    BOOLEAN NOT NULL DEFAULT FALSE;
                """
            )
            cursor.execute(
                """
                ALTER TABLE research_jobs
                DROP CONSTRAINT IF EXISTS research_jobs_status_check;
                """
            )
            cursor.execute(
                """
                ALTER TABLE research_jobs
                ADD CONSTRAINT research_jobs_status_check
                CHECK (
                    status IN (
                        'queued',
                        'running',
                        'retry_wait',
                        'awaiting_review',
                        'completed',
                        'failed',
                        'cancelled',
                        'paused'
                    )
                );
                """
            )
            cursor.execute(
                """
                DROP INDEX IF EXISTS idx_research_jobs_one_active_session;
                """
            )
            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_research_jobs_one_active_session
                ON research_jobs (session_id)
                WHERE status IN (
                    'queued',
                    'running',
                    'retry_wait',
                    'paused'
                );
                """
            )

            cursor.execute(CREATE_RESEARCH_QUESTIONS_TABLE)
            cursor.execute(CREATE_RESEARCH_EVIDENCE_TABLE)
            cursor.execute(CREATE_OBSERVABILITY_EVENTS_TABLE)

            for query in CREATE_INDEXES:
                cursor.execute(query)

        connection.commit()
