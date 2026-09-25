"""AstrumWeaver PostgreSQL schema migration entrypoint."""

from __future__ import annotations

import argparse
import os
from importlib import resources
from pathlib import Path
from typing import Any

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]


SCHEMA_MIGRATIONS_TABLE = "schema_migrations"


def _migration_files() -> tuple[Any, ...]:
    migration_root = resources.files("astrumweaver").joinpath("migrations")
    migrations: list[Any] = []
    try:
        migrations = sorted(
            (
                item
                for item in migration_root.iterdir()
                if item.name.endswith(".sql")
            ),
            key=lambda item: item.name,
        )
    except (FileNotFoundError, NotADirectoryError):
        migrations = []

    if not migrations:
        # Editable/source-tree fallback. Installed wheels include migrations
        # under the astrumweaver package through hatch force-include.
        source_root = Path(__file__).resolve().parents[3] / "migrations"
        if source_root.is_dir():
            migrations = sorted(
                source_root.glob("*.sql"),
                key=lambda item: item.name,
            )

    if not migrations:
        raise RuntimeError("no AstrumWeaver migrations were found")
    return tuple(migrations)


def required_migration_names() -> tuple[str, ...]:
    """Return the canonical migration set required by this binary."""
    return tuple(migration.name for migration in _migration_files())


def _recorded_migrations(connection: Any) -> set[str]:
    table = connection.execute(
        "SELECT to_regclass('public.schema_migrations') AS table_name"
    ).fetchone()
    if not table or table.get("table_name") is None:
        return set()
    rows = connection.execute(
        "SELECT name FROM schema_migrations ORDER BY name"
    ).fetchall()
    return {str(row["name"]) for row in rows}


def apply_migrations(database_url: str) -> list[str]:
    if not database_url:
        raise RuntimeError("database URL is required")
    if psycopg is None:
        raise RuntimeError("psycopg is required for PostgreSQL migrations")

    migrations = _migration_files()

    with psycopg.connect(database_url, autocommit=True) as connection:
        recorded = _recorded_migrations(connection)

        for migration in migrations:
            if migration.name in recorded:
                continue

            # Migration SQL remains independently idempotent. This matters for
            # legacy databases that already contain 001-era tables but predate
            # schema_migrations tracking: replay records the canonical history
            # without requiring destructive schema reconstruction.
            connection.execute(migration.read_text(encoding="utf-8"))

            table = connection.execute(
                "SELECT to_regclass('public.schema_migrations') AS table_name"
            ).fetchone()
            if table and table.get("table_name") is not None:
                connection.execute(
                    """
                    INSERT INTO schema_migrations (name)
                    VALUES (%s)
                    ON CONFLICT (name) DO NOTHING
                    """,
                    (migration.name,),
                )
                recorded.add(migration.name)

        final = _recorded_migrations(connection)

    required = required_migration_names()
    missing = [name for name in required if name not in final]
    if missing:
        raise RuntimeError(
            "database migration tracking is incomplete: " + ", ".join(missing)
        )
    return list(required)


def main() -> None:
    parser = argparse.ArgumentParser(prog="astrumweaver-migrate")
    parser.parse_args()

    database_url = os.environ.get("ASTRUMWEAVER_DATABASE_URL", "")
    applied = apply_migrations(database_url)
    for name in applied:
        print(name)


if __name__ == "__main__":
    main()
