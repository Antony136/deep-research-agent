"""Tests for database schema initialization."""

from unittest.mock import MagicMock, call

import pytest

from app.database import schema


@pytest.fixture
def mock_database(monkeypatch):
    """Provide a mocked connection and cursor."""
    connection = MagicMock()
    cursor = MagicMock()

    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor

    monkeypatch.setattr(
        schema,
        "get_connection",
        lambda: connection,
    )

    return connection, cursor


def normalize_sql(sql: str) -> str:
    """Normalize whitespace for readable SQL assertions."""
    return " ".join(sql.split()).upper()


def test_initialize_database_creates_all_tables_and_commits(
    mock_database,
):
    connection, cursor = mock_database

    schema.initialize_database()

    executed_sql = [
        normalize_sql(item.args[0])
        for item in cursor.execute.call_args_list
    ]

    expected_tables = [
        "CREATE TABLE IF NOT EXISTS RESEARCH_SESSIONS",
        "CREATE TABLE IF NOT EXISTS RESEARCH_JOBS",
        "CREATE TABLE IF NOT EXISTS RESEARCH_QUESTIONS",
        "CREATE TABLE IF NOT EXISTS RESEARCH_EVIDENCE",
        "CREATE TABLE IF NOT EXISTS OBSERVABILITY_EVENTS",
    ]

    for table_sql in expected_tables:
        assert any(
            sql.startswith(table_sql)
            for sql in executed_sql
        ), f"Missing SQL operation: {table_sql}"

    assert len(schema.CREATE_INDEXES) == 9

    for index_sql in schema.CREATE_INDEXES:
        normalized_index = normalize_sql(index_sql)

        assert any(
            sql == normalized_index
            for sql in executed_sql
        ), f"Index was not created: {normalized_index}"

    connection.commit.assert_called_once()


def test_initialize_database_migrates_session_status_constraint(
    mock_database,
):
    _, cursor = mock_database

    schema.initialize_database()

    executed_sql = [
        normalize_sql(item.args[0])
        for item in cursor.execute.call_args_list
    ]

    assert any(
        "ALTER TABLE RESEARCH_SESSIONS DROP CONSTRAINT IF EXISTS"
        in sql
        for sql in executed_sql
    )

    assert any(
        "ALTER TABLE RESEARCH_SESSIONS ADD CONSTRAINT"
        in sql
        and "'WAITING_FOR_REVIEW'" in sql
        and "'CANCELLED'" in sql
        and "'PAUSED'" in sql
        for sql in executed_sql
    )


def test_initialize_database_migrates_job_schema(
    mock_database,
):
    _, cursor = mock_database

    schema.initialize_database()

    executed_sql = [
        normalize_sql(item.args[0])
        for item in cursor.execute.call_args_list
    ]

    assert any(
        "ALTER TABLE RESEARCH_JOBS ADD COLUMN IF NOT EXISTS PAUSE_REQUESTED"
        in sql
        for sql in executed_sql
    )

    assert any(
        "ALTER TABLE RESEARCH_JOBS DROP CONSTRAINT IF EXISTS"
        in sql
        for sql in executed_sql
    )

    assert any(
        "ALTER TABLE RESEARCH_JOBS ADD CONSTRAINT"
        in sql
        and "'RETRY_WAIT'" in sql
        and "'AWAITING_REVIEW'" in sql
        and "'PAUSED'" in sql
        for sql in executed_sql
    )


def test_initialize_database_recreates_active_session_unique_index(
    mock_database,
):
    _, cursor = mock_database

    schema.initialize_database()

    executed_sql = [
        normalize_sql(item.args[0])
        for item in cursor.execute.call_args_list
    ]

    assert any(
        "DROP INDEX IF EXISTS IDX_RESEARCH_JOBS_ONE_ACTIVE_SESSION"
        in sql
        for sql in executed_sql
    )

    active_index_sql = [
        sql
        for sql in executed_sql
        if "CREATE UNIQUE INDEX IF NOT EXISTS IDX_RESEARCH_JOBS_ONE_ACTIVE_SESSION"
        in sql
    ]

    assert len(active_index_sql) == 1

    assert "'QUEUED'" in active_index_sql[0]
    assert "'RUNNING'" in active_index_sql[0]
    assert "'RETRY_WAIT'" in active_index_sql[0]
    assert "'PAUSED'" in active_index_sql[0]


def test_initialize_database_executes_schema_operations_in_order(
    mock_database,
):
    _, cursor = mock_database

    schema.initialize_database()

    executed_sql = [
        normalize_sql(item.args[0])
        for item in cursor.execute.call_args_list
    ]

    session_table_index = next(
        index
        for index, sql in enumerate(executed_sql)
        if sql.startswith(
            "CREATE TABLE IF NOT EXISTS RESEARCH_SESSIONS"
        )
    )

    jobs_table_index = next(
        index
        for index, sql in enumerate(executed_sql)
        if sql.startswith(
            "CREATE TABLE IF NOT EXISTS RESEARCH_JOBS"
        )
    )

    questions_table_index = next(
        index
        for index, sql in enumerate(executed_sql)
        if sql.startswith(
            "CREATE TABLE IF NOT EXISTS RESEARCH_QUESTIONS"
        )
    )

    evidence_table_index = next(
        index
        for index, sql in enumerate(executed_sql)
        if sql.startswith(
            "CREATE TABLE IF NOT EXISTS RESEARCH_EVIDENCE"
        )
    )

    events_table_index = next(
        index
        for index, sql in enumerate(executed_sql)
        if sql.startswith(
            "CREATE TABLE IF NOT EXISTS OBSERVABILITY_EVENTS"
        )
    )

    assert (
        session_table_index
        < jobs_table_index
        < questions_table_index
        < evidence_table_index
        < events_table_index
    )


def test_initialize_database_does_not_commit_if_sql_execution_fails(
    mock_database,
):
    connection, cursor = mock_database

    cursor.execute.side_effect = RuntimeError("Database migration failed")

    with pytest.raises(
        RuntimeError,
        match="Database migration failed",
    ):
        schema.initialize_database()

    connection.commit.assert_not_called()
