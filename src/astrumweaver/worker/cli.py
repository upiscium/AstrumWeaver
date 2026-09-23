"""Command-line entrypoint for the AstrumWeaver Worker daemon."""

from __future__ import annotations

import argparse
import asyncio
import os
import signal

from ..config import ConfigurationError
from .client import ControlClient
from .config import WorkerRuntimeConfig
from .daemon import WorkerDaemon
from .loader import ExecutorLoadError, load_executor
from .preflight import PreflightError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrumweaver-worker")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate TOML and executor factory without contacting Control",
    )
    return parser


async def _run(config: WorkerRuntimeConfig) -> None:
    worker_token = os.environ.get("ASTRUMWEAVER_WORKER_TOKEN", "").strip()
    if not worker_token:
        raise ConfigurationError("ASTRUMWEAVER_WORKER_TOKEN is required")

    executor = load_executor(config.executor_factory, config.executor_settings)
    client = ControlClient(
        base_url=config.control_url,
        worker_token=worker_token,
        timeout_seconds=config.request_timeout_seconds,
    )
    daemon = WorkerDaemon(config=config, executor=executor, client=client)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover - non-POSIX event loops
            pass

    await daemon.run_forever(stop)


def main() -> None:
    args = _parser().parse_args()
    try:
        config = WorkerRuntimeConfig.from_file(args.config)
        if args.check_config:
            load_executor(config.executor_factory, config.executor_settings)
            return
        asyncio.run(_run(config))
    except (ConfigurationError, ExecutorLoadError, PreflightError, ValueError) as exc:
        raise SystemExit(f"worker startup error: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover
    main()
