"""
PostgreSQL connection management for the Deep Research Agent.
"""

import os

import psycopg
from dotenv import load_dotenv


load_dotenv()


def get_connection():
    """
    Create and return a PostgreSQL database connection.

    Connection settings are loaded from environment variables.
    """

    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv(
            "POSTGRES_DB",
            "deep_research_agent",
        ),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.environ["POSTGRES_PASSWORD"],
        connect_timeout=5,
    )
