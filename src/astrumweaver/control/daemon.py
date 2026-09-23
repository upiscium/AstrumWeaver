"""astrumweaver-control service entrypoint."""

from __future__ import annotations

import argparse
import os
import tomllib
from pathlib import Path
from typing import Any

import uvicorn

from .api import create_app
from .postgres import PostgresControlRepository


def _load_toml(path: str) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def build_app(config_path: str):
    config = _load_toml(config_path)
    section = dict(config.get("control") or {})

    database_url = os.environ.get("ASTRUMWEAVER_DATABASE_URL", "")
    client_token = os.environ.get("ASTRUMWEAVER_CLIENT_TOKEN", "")
    worker_token = os.environ.get("ASTRUMWEAVER_WORKER_TOKEN", "")

    if not database_url:
        raise RuntimeError("ASTRUMWEAVER_DATABASE_URL is required")
    if not client_token:
        raise RuntimeError("ASTRUMWEAVER_CLIENT_TOKEN is required")
    if not worker_token:
        raise RuntimeError("ASTRUMWEAVER_WORKER_TOKEN is required")

    repository = PostgresControlRepository(
        database_url,
        worker_ttl_seconds=int(section.get("worker_ttl_seconds", 60)),
        lease_seconds=int(section.get("lease_seconds", 300)),
    )
    return create_app(
        repository,
        client_token=client_token,
        worker_token=worker_token,
        maintenance_interval_seconds=float(section.get("maintenance_interval_seconds", 5.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="astrumweaver-control")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = _load_toml(args.config)
    section = dict(config.get("control") or {})
    host = str(section.get("host", "127.0.0.1"))
    port = int(section.get("port", 9000))

    app = build_app(args.config)
    uvicorn.run(
        app,
        host=host,
        port=port,
        access_log=bool(section.get("access_log", False)),
    )


if __name__ == "__main__":
    main()
