"""astrumweaver-worker service entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import tomllib
from typing import Any

import uvicorn

from ..contracts import AcceleratorDevice, ResourceShape, WorkerSpec
from .client import ControlClient
from .health import create_health_app
from .runtime import (
    WorkerRuntime,
    load_executor,
    require_exact_gpu_set,
    require_executor_capabilities,
)


def _load_toml(path: str) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _build_spec(section: dict[str, Any]) -> WorkerSpec:
    gpu_uuids = tuple(str(item) for item in section.get("gpu_uuids", ()))
    capabilities = frozenset(str(item) for item in section.get("capabilities", ()))
    labels = {str(key): str(value) for key, value in dict(section.get("labels") or {}).items()}
    accelerators = tuple(
        AcceleratorDevice(
            uuid=str(item["uuid"]),
            memory_mb=int(item["memory_mb"]),
            compute_capability=item.get("compute_capability"),
            device_class=item.get("device_class"),
        )
        for item in section.get("accelerators", ())
    )
    return WorkerSpec(
        worker_id=str(section["id"]),
        worker_class=str(section["class"]),
        resources=ResourceShape(
            gpu_count=int(section.get("gpu_count", len(gpu_uuids))),
            total_vram_mb=int(section.get("total_vram_mb", 0)),
            max_single_gpu_vram_mb=int(section.get("max_single_gpu_vram_mb", 0)),
        ),
        gpu_uuids=gpu_uuids,
        accelerators=accelerators,
        capabilities=capabilities,
        labels=labels,
    )


async def run_worker(config_path: str) -> None:
    config = _load_toml(config_path)
    worker_section = dict(config.get("worker") or {})
    executor_section = dict(config.get("executor") or {})

    required = ("id", "class", "control_url")
    missing = [name for name in required if not worker_section.get(name)]
    if missing:
        raise RuntimeError(f"missing worker configuration: {', '.join(missing)}")

    executor_factory = str(executor_section.get("factory", ""))
    if not executor_factory:
        raise RuntimeError("executor.factory is required")

    worker_token = os.environ.get("ASTRUMWEAVER_WORKER_TOKEN", "")
    if not worker_token:
        raise RuntimeError("ASTRUMWEAVER_WORKER_TOKEN is required")

    spec = _build_spec(worker_section)

    if spec.gpu_uuids and bool(worker_section.get("gpu_preflight", True)):
        require_exact_gpu_set(
            spec.gpu_uuids,
            command=str(worker_section.get("nvidia_smi_command", "nvidia-smi")),
        )

    executor = load_executor(
        executor_factory,
        dict(executor_section.get("settings") or {}),
    )
    require_executor_capabilities(executor, spec.capabilities)

    client = ControlClient(
        str(worker_section["control_url"]),
        worker_token,
        timeout_seconds=float(worker_section.get("request_timeout_seconds", 10.0)),
    )

    runtime = WorkerRuntime(
        spec=spec,
        max_concurrency=int(worker_section.get("max_concurrency", 1)),
        executor=executor,
        client=client,
        poll_interval_seconds=float(worker_section.get("poll_interval_seconds", 1.0)),
        heartbeat_interval_seconds=float(
            worker_section.get("heartbeat_interval_seconds", 5.0)
        ),
    )

    health_config = uvicorn.Config(
        create_health_app(runtime),
        host=str(worker_section.get("health_host", "127.0.0.1")),
        port=int(worker_section.get("health_port", 9100)),
        log_level="warning",
        access_log=False,
    )
    health_server = uvicorn.Server(health_config)
    health_server.install_signal_handlers = lambda: None

    loop = asyncio.get_running_loop()
    stop_signal = asyncio.Event()
    drain_signal = asyncio.Event()

    def request_shutdown() -> None:
        stop_signal.set()

    def request_drain() -> None:
        # Stop local claiming immediately; the async task mirrors DRAINING
        # into durable Control state without requiring mode-switch tooling to
        # hold the Worker authority token.
        runtime.request_drain()
        drain_signal.set()

    for signal_name in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(signal_name, request_shutdown)
        except NotImplementedError:
            pass

    sigusr1 = getattr(signal, "SIGUSR1", None)
    if sigusr1 is not None:
        try:
            loop.add_signal_handler(sigusr1, request_drain)
        except NotImplementedError:
            pass

    async def drain_signal_loop() -> None:
        while True:
            await drain_signal.wait()
            drain_signal.clear()
            with contextlib.suppress(Exception):
                await runtime.drain()

    worker_task = asyncio.create_task(runtime.run_forever(), name="astrumweaver-worker")
    health_task = asyncio.create_task(health_server.serve(), name="astrumweaver-worker-health")
    signal_task = asyncio.create_task(stop_signal.wait(), name="astrumweaver-worker-signal")
    drain_task = asyncio.create_task(
        drain_signal_loop(),
        name="astrumweaver-worker-drain-signal",
    )

    try:
        done, _ = await asyncio.wait(
            {worker_task, health_task, signal_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if worker_task in done:
            exc = worker_task.exception()
            if exc is not None:
                raise exc

        if health_task in done:
            exc = health_task.exception()
            if exc is not None:
                raise exc

        forced_cancel = False
        if signal_task in done and runtime.registered:
            try:
                await runtime.drain()
            except Exception:
                pass

            grace = float(worker_section.get("shutdown_grace_seconds", 30.0))
            deadline = loop.time() + max(grace, 0.0)
            while runtime.active_job_id is not None and loop.time() < deadline:
                await asyncio.sleep(0.1)

            if runtime.active_job_id is not None:
                forced_cancel = True
                with contextlib.suppress(Exception):
                    await executor.cancel(runtime.active_job_id)

        runtime.request_stop()
        health_server.should_exit = True

        if forced_cancel and not worker_task.done():
            worker_task.cancel()

        await asyncio.gather(worker_task, health_task, return_exceptions=True)
    finally:
        signal_task.cancel()
        drain_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drain_task
        await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="astrumweaver-worker")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    asyncio.run(run_worker(args.config))


if __name__ == "__main__":
    main()
