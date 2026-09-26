"""astrumweaver-worker service entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import tomllib
from typing import Any

import uvicorn

from ..contracts import AcceleratorDevice, ResourceShape, WorkerSpec
from ..runtime import (
    RuntimeDeploymentSpec,
    RuntimeLifecycleManager,
    discover_runtime_host_facts,
    managed_runtime_from_deployment,
)
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


def _load_runtime_deployment(path: str) -> RuntimeDeploymentSpec:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot load runtime deployment manifest: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError("runtime deployment manifest must contain an object")
    try:
        return RuntimeDeploymentSpec.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("runtime deployment manifest is invalid") from exc


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


async def run_worker(
    config_path: str,
    runtime_manifest_path: str | None = None,
) -> None:
    config = _load_toml(config_path)
    worker_section = dict(config.get("worker") or {})
    executor_section = dict(config.get("executor") or {})
    runtime_section = dict(config.get("runtime") or {})

    required = ("id", "class", "control_url")
    missing = [name for name in required if not worker_section.get(name)]
    if missing:
        raise RuntimeError(f"missing worker configuration: {', '.join(missing)}")

    executor_factory = str(executor_section.get("factory", "")).strip()
    runtime_manifest = (
        str(runtime_manifest_path).strip()
        if runtime_manifest_path is not None
        else str(runtime_section.get("manifest", "")).strip()
    )
    if bool(executor_factory) == bool(runtime_manifest):
        raise RuntimeError(
            "configure exactly one of executor.factory or runtime.manifest"
        )

    worker_token = os.environ.get("ASTRUMWEAVER_WORKER_TOKEN", "")
    if not worker_token:
        raise RuntimeError("ASTRUMWEAVER_WORKER_TOKEN is required")

    spec = _build_spec(worker_section)

    if spec.gpu_uuids and bool(worker_section.get("gpu_preflight", True)):
        require_exact_gpu_set(
            spec.gpu_uuids,
            command=str(worker_section.get("nvidia_smi_command", "nvidia-smi")),
        )

    client = ControlClient(
        str(worker_section["control_url"]),
        worker_token,
        timeout_seconds=float(worker_section.get("request_timeout_seconds", 10.0)),
    )
    runtime_manager: RuntimeLifecycleManager | None = None

    try:
        if runtime_manifest:
            deployment = _load_runtime_deployment(runtime_manifest)
            managed_runtime = managed_runtime_from_deployment(
                deployment,
                worker=spec,
                host=discover_runtime_host_facts(),
            )
            runtime_manager = RuntimeLifecycleManager(managed_runtime)
            await runtime_manager.ensure_ready(
                timeout_seconds=float(
                    runtime_section.get("startup_timeout_seconds", 600.0)
                )
            )
            executor = managed_runtime.executor()
        else:
            executor = load_executor(
                executor_factory,
                dict(executor_section.get("settings") or {}),
            )

        require_executor_capabilities(executor, spec.capabilities)

        worker_runtime = WorkerRuntime(
            spec=spec,
            max_concurrency=int(worker_section.get("max_concurrency", 1)),
            executor=executor,
            client=client,
            poll_interval_seconds=float(
                worker_section.get("poll_interval_seconds", 1.0)
            ),
            heartbeat_interval_seconds=float(
                worker_section.get("heartbeat_interval_seconds", 5.0)
            ),
        )

        health_config = uvicorn.Config(
            create_health_app(worker_runtime),
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
            worker_runtime.request_drain()
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
                    await worker_runtime.drain()

        worker_task = asyncio.create_task(
            worker_runtime.run_forever(),
            name="astrumweaver-worker",
        )
        health_task = asyncio.create_task(
            health_server.serve(),
            name="astrumweaver-worker-health",
        )
        signal_task = asyncio.create_task(
            stop_signal.wait(),
            name="astrumweaver-worker-signal",
        )
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
            if signal_task in done and worker_runtime.registered:
                with contextlib.suppress(Exception):
                    await worker_runtime.drain()

                grace = float(
                    worker_section.get("shutdown_grace_seconds", 30.0)
                )
                deadline = loop.time() + max(grace, 0.0)
                while (
                    worker_runtime.active_job_id is not None
                    and loop.time() < deadline
                ):
                    await asyncio.sleep(0.1)

                if worker_runtime.active_job_id is not None:
                    forced_cancel = True
                    with contextlib.suppress(Exception):
                        await executor.cancel(worker_runtime.active_job_id)

            worker_runtime.request_stop()
            health_server.should_exit = True

            if forced_cancel and not worker_task.done():
                worker_task.cancel()

            await asyncio.gather(
                worker_task,
                health_task,
                return_exceptions=True,
            )
        finally:
            signal_task.cancel()
            drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain_task
    finally:
        await client.aclose()
        if runtime_manager is not None:
            await runtime_manager.ensure_stopped(
                timeout_seconds=float(
                    runtime_section.get(
                        "shutdown_timeout_seconds",
                        60.0,
                    )
                ),
                release=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(prog="astrumweaver-worker")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--runtime-manifest",
        help=(
            "optional RuntimeProvider deployment manifest; overrides "
            "[runtime].manifest in worker configuration"
        ),
    )
    args = parser.parse_args()
    asyncio.run(
        run_worker(
            args.config,
            runtime_manifest_path=args.runtime_manifest,
        )
    )


if __name__ == "__main__":
    main()
