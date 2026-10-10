"""
Test PostgreSQL connectivity for the Deep Research Agent.
"""

from app.database.connection import get_connection


def main():
    print("=" * 60)
    print("POSTGRESQL CONNECTION TEST")
    print("=" * 60)

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    current_database(),
                    current_user,
                    version();
                """
            )

            database, user, version = cursor.fetchone()

            print(f"\nDatabase: {database}")
            print(f"User: {user}")
            print(f"PostgreSQL: {version.split(',')[0]}")

    print("\nDatabase connection successful.")


if __name__ == "__main__":
    main()