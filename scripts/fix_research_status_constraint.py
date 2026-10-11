"""
Update the research_sessions status constraint.

Adds 'waiting_for_review' to the allowed session statuses.
"""

from app.database.connection import get_connection


def main():
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                ALTER TABLE research_sessions
                DROP CONSTRAINT IF EXISTS research_sessions_status_check;
            """)

            cursor.execute("""
                ALTER TABLE research_sessions
                ADD CONSTRAINT research_sessions_status_check
                CHECK (
                    status IN (
                        'created',
                        'running',
                        'waiting_for_review',
                        'completed',
                        'failed'
                    )
                );
            """)

        connection.commit()

        print("Research session status constraint updated successfully.")

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()
