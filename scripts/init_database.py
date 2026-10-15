"""Initialize or migrate the PostgreSQL schema for this project."""

from app.database.schema import initialize_database


def main() -> None:
    initialize_database()
    print("Database schema initialized.")


if __name__ == "__main__":
    main()
