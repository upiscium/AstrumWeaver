"""Generic AstrumWeaver Worker execution loop."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import subprocess
from collections.abc import Mapping
from typing import Any

from ..contracts import WorkerSpec
from ..control.models import WorkerState
from ..execution import JobExecutor, JobResult
from .client import ClaimedJob, ControlClient, ControlTransportError


def discover_nvidia_gpu_uuids(command: str = "nvidia-smi") -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            [command, "--query-gpu=uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi preflight failed") from exc
    uuids = tuple(sorted(line.strip() for line in completed.stdout.splitlines() if line.strip()))
    if not uuids:
        raise RuntimeError("no NVIDIA GPUs observed")
    if len(set(uuids)) != len(uuids):
        raise RuntimeError("observed duplicate GPU UUIDs")
    return uuids


def require_exact_gpu_set(
    expected: tuple[str, ...],
    *,
    command: str = "nvidia-smi",
) -> None:
    expected_sorted = tuple(sorted(expected))
    if not expected_sorted:
        return
    if len(set(expected_sorted)) != len(expected_sorted):
        raise RuntimeError("configured GPU UUIDs contain duplicates")
    observed = discover_nvidia_gpu_uuids(command)
    if observed != expected_sorted:
        raise RuntimeError(
            f"GPU UUID set mismatch (expected_count={len(expected_sorted)} observed_count={len(observed)})"
        )


def executor_capabilities(executor: JobExecutor) -> frozenset[str]:
    raw = getattr(executor, "capabilities", None)
    if raw is None:
        raise TypeError("configured executor must declare capabilities")
    if isinstance(raw, str):
        values = (raw,)
    else:
        values = tuple(raw)
    capabilities = frozenset(str(value).strip() for value in values)
    if not capabilities or any(not value for value in capabilities):
        raise ValueError("executor capabilities must be non-empty non-blank strings")
    return capabilities


def require_executor_capabilities(
    executor: JobExecutor,
    advertised: frozenset[str],
) -> None:
    supported = executor_capabilities(executor)
    missing = advertised - supported
    if missing:
        raise RuntimeError(
            "executor does not support advertised capabilities: "
            + ", ".join(sorted(missing))
        )


def load_executor(specifier: str, config: Mapping[str, Any] | None = None) -> JobExecutor:
    module_name, separator, attribute_name = specifier.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("executor must use module:attribute syntax")
    module = importlib.import_module(module_name)
    target = getattr(module, attribute_name)
    if isinstance(target, JobExecutor):
        executor = target
    elif callable(target):
        executor = target(dict(config or {}))
        if inspect.isawaitable(executor):
            raise TypeError("executor factory must be synchronous")
    else:
        raise TypeError("executor target must be a JobExecutor or factory")
    if not isinstance(executor, JobExecutor):
        raise TypeError("executor factory did not return JobExecutor")
    return executor


class WorkerRuntime:
    def __init__(
        self,
        *,
        spec: WorkerSpec,
        max_concurrency: int,
        executor: JobExecutor,
        client: ControlClient,
        poll_interval_seconds: float = 1.0,
        heartbeat_interval_seconds: float = 5.0,
    ) -> None:
        if max_concurrency != 1:
            raise ValueError("v1 WorkerRuntime currently requires max_concurrency=1")
        if poll_interval_seconds <= 0 or heartbeat_interval_seconds <= 0:
            raise ValueError("worker intervals must be positive")
        self.spec = spec
        self.max_concurrency = max_concurrency
        self.executor = executor
        self.client = client
        self.poll_interval_seconds = poll_interval_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._stop = asyncio.Event()
        self._draining = False
        self._registered = False
        self._active: ClaimedJob | None = None

    @property
    def ready(self) -> bool:
        return not self._stop.is_set()

    @property
    def registered(self) -> bool:
        return self._registered

    @property
    def active_job_id(self) -> str | None:
        return None if self._active is None else self._active.request.job_id

    def request_stop(self) -> None:
        self._draining = True
        self._stop.set()

    async def register(self) -> None:
        await self.client.register(
            spec=self.spec,
            max_concurrency=self.max_concurrency,
            metadata={"runtime": "astrumweaver-worker"},
        )
        self._registered = True

    async def run_forever(self) -> None:
        await self.register()
        try:
            while not self._stop.is_set():
                if self._draining:
                    await asyncio.sleep(self.poll_interval_seconds)
                    continue
                claimed = await self.client.claim(self.spec.worker_id)
                if claimed is None:
                    await self.client.heartbeat(self.spec.worker_id)
                    try:
                        await asyncio.wait_for(
                            self._stop.wait(), timeout=self.poll_interval_seconds
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue
                self._active = claimed
                try:
                    await self._execute_claim(claimed)
                finally:
                    self._active = None
        finally:
            with contextlib.suppress(ControlTransportError):
                await self.client.set_state(self.spec.worker_id, WorkerState.OFFLINE)
            self._registered = False

    async def drain(self) -> None:
        self._draining = True
        await self.client.set_state(self.spec.worker_id, WorkerState.DRAINING)

    async def _execute_claim(self, claimed: ClaimedJob) -> None:
        execution = asyncio.create_task(self.executor.execute(claimed.request))
        cancelled_remotely = False
        try:
            while not execution.done():
                done, _ = await asyncio.wait(
                    {execution},
                    timeout=self.heartbeat_interval_seconds,
                )
                if done:
                    break
                try:
                    await self.client.heartbeat(
                        self.spec.worker_id,
                        active_job_id=claimed.request.job_id,
                        lease_token=claimed.lease_token,
                    )
                except ControlTransportError as exc:
                    if exc.status_code != 409:
                        raise
                    status = await self.client.inspect_job(
                        self.spec.worker_id, claimed.request.job_id
                    )
                    if status.get("status") == "cancelled":
                        cancelled_remotely = True
                        await self.executor.cancel(claimed.request.job_id)
                        execution.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await execution
                        return
                    raise

            if cancelled_remotely:
                return

            try:
                result = await execution
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                try:
                    await self.client.fail(
                        self.spec.worker_id,
                        claimed.request.job_id,
                        claimed.lease_token,
                        error={"type": type(exc).__name__, "message": str(exc)},
                        retryable=True,
                    )
                except ControlTransportError as transport_exc:
                    if transport_exc.status_code != 409:
                        raise
                    status = await self.client.inspect_job(
                        self.spec.worker_id, claimed.request.job_id
                    )
                    if status.get("status") != "cancelled":
                        raise
                return

            if not isinstance(result, JobResult):
                try:
                    await self.client.fail(
                        self.spec.worker_id,
                        claimed.request.job_id,
                        claimed.lease_token,
                        error={
                            "type": "ExecutorProtocolError",
                            "message": "executor returned a non-JobResult value",
                        },
                        retryable=False,
                    )
                except ControlTransportError as transport_exc:
                    if transport_exc.status_code != 409:
                        raise
                    status = await self.client.inspect_job(
                        self.spec.worker_id, claimed.request.job_id
                    )
                    if status.get("status") != "cancelled":
                        raise
                return
            try:
                await self.client.complete(
                    self.spec.worker_id,
                    claimed.request.job_id,
                    claimed.lease_token,
                    result,
                )
            except ControlTransportError as exc:
                if exc.status_code != 409:
                    raise
                status = await self.client.inspect_job(
                    self.spec.worker_id, claimed.request.job_id
                )
                if status.get("status") != "cancelled":
                    raise
        finally:
            if not execution.done():
                execution.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await execution
