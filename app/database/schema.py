"""
Database schema initialization for the Deep Research Agent.
"""

from app.database.connection import get_connection


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
                'completed',
                'failed'
            )
        )
);
"""


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
    CREATE INDEX IF NOT EXISTS idx_research_questions_session
    ON research_questions (session_id, question_number);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_research_evidence_session
    ON research_evidence (session_id, evidence_number);
    """,
]


def initialize_database() -> None:
    """
    Create the research tables and indexes if they do not exist.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_RESEARCH_SESSIONS_TABLE)
            cursor.execute(CREATE_RESEARCH_QUESTIONS_TABLE)
            cursor.execute(CREATE_RESEARCH_EVIDENCE_TABLE)

            for query in CREATE_INDEXES:
                cursor.execute(query)

        connection.commit()