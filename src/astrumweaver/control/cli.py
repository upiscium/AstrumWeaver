"""Command-line entrypoint for the AstrumWeaver Control daemon."""

from __future__ import annotations

import argparse
import os

import uvicorn

from ..config import (
    ConfigurationError,
    load_toml,
    positive_float,
    positive_int,
    table,
)
from ..transport import AuthConfig
from .postgres import PostgresControlRepository
from .service import create_app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrumweaver-control")
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        config = load_toml(args.config)
        control = table(config, "control")
        host = str(control.get("host", "127.0.0.1"))
        port = positive_int(control, "port", context="control", default=8080)
        worker_ttl = positive_int(
            control,
            "worker_ttl_seconds",
            context="control",
            default=60,
        )
        lease_seconds = positive_int(
            control,
            "lease_seconds",
            context="control",
            default=300,
        )
        maintenance_interval = positive_float(
            control,
            "maintenance_interval_seconds",
            context="control",
            default=5.0,
        )

        database_url = os.environ.get("ASTRUMWEAVER_DATABASE_URL", "").strip()
        client_token = os.environ.get("ASTRUMWEAVER_CLIENT_TOKEN", "").strip()
        worker_token = os.environ.get("ASTRUMWEAVER_WORKER_TOKEN", "").strip()
        if not database_url:
            raise ConfigurationError("ASTRUMWEAVER_DATABASE_URL is required")
        if not client_token:
            raise ConfigurationError("ASTRUMWEAVER_CLIENT_TOKEN is required")
        if not worker_token:
            raise ConfigurationError("ASTRUMWEAVER_WORKER_TOKEN is required")

        repository = PostgresControlRepository(
            database_url,
            worker_ttl_seconds=worker_ttl,
            lease_seconds=lease_seconds,
        )
        app = create_app(
            repository=repository,
            auth=AuthConfig(
                client_token=client_token,
                worker_token=worker_token,
            ),
            maintenance_interval_seconds=maintenance_interval,
        )
    except (ConfigurationError, ValueError) as exc:
        raise SystemExit(f"configuration error: {exc}") from exc

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=str(control.get("log_level", "info")),
        access_log=bool(control.get("access_log", False)),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
