"""AstrumWeaver PostgreSQL schema migration entrypoint."""

from __future__ import annotations

import argparse
import os
from importlib import resources

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]


def apply_migrations(database_url: str) -> list[str]:
    if not database_url:
        raise RuntimeError("database URL is required")
    if psycopg is None:
        raise RuntimeError("psycopg is required for PostgreSQL migrations")

    migration_root = resources.files("astrumweaver").joinpath("migrations")
    migrations = sorted(
        item for item in migration_root.iterdir()
        if item.name.endswith(".sql")
    )
    if not migrations:
        raise RuntimeError("no packaged migrations were found")

    applied: list[str] = []
    with psycopg.connect(database_url, autocommit=True) as connection:
        for migration in migrations:
            connection.execute(migration.read_text(encoding="utf-8"))
            applied.append(migration.name)
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(prog="astrumweaver-migrate")
    parser.parse_args()

    database_url = os.environ.get("ASTRUMWEAVER_DATABASE_URL", "")
    applied = apply_migrations(database_url)
    for name in applied:
        print(name)


if __name__ == "__main__":
    main()
