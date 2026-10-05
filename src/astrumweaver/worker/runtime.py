"""Generic AstrumWeaver Worker execution loop."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import math
import logging
import time
import subprocess
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from ..contracts import WorkerSpec
from ..control.models import WorkerState
from ..control.serde import worker_spec_from_dict
from ..transport import PROTOCOL_VERSION, SERVING_EXTENSION
from ..execution import JobExecutor, JobResult
from ..serving import (
    ServingJobBinding,
    WorkerServingAdvertisement,
    worker_serving_matches,
)
from ..gpu_mapping import (
    discover_gpu_mapping,
    load_reviewed_gpu_map,
    verify_isolated_gpu_access,
)
from .client import ClaimedJob, ControlClient, ControlTransportError
from .supervision import RuntimeHealthSupervisor, RuntimeUnavailable


_LOG = logging.getLogger(__name__)


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


def require_isolated_gpu_access(
    expected: tuple[str, ...],
    *,
    reviewed_map_path: str,
    command: str = "nvidia-smi",
    cuda_visible_devices: str | None,
) -> None:
    if not expected:
        return
    map_path = Path(reviewed_map_path)
    if not map_path.is_absolute():
        raise RuntimeError("isolated GPU device map path must be absolute")
    mapping = discover_gpu_mapping(nvidia_smi=command)
    reviewed = load_reviewed_gpu_map(map_path)
    verify_isolated_gpu_access(
        mapping,
        expected,
        reviewed,
        worker_gpu_order=expected,
        cuda_visible_devices=cuda_visible_devices,
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
    """One execution owner with a persistent, job-independent heartbeat deadline.

    Control RPCs are serialized with drain requests. No heartbeat task can
    outlive an attempt or race a terminal write using an obsolete lease token.
    """

    def __init__(
        self,
        *,
        spec: WorkerSpec,
        max_concurrency: int,
        executor: JobExecutor,
        client: ControlClient,
        poll_interval_seconds: float = 1.0,
        heartbeat_interval_seconds: float = 5.0,
        runtime_supervisor: RuntimeHealthSupervisor | None = None,
        serving: WorkerServingAdvertisement | None = None,
    ) -> None:
        if max_concurrency != 1:
            raise ValueError("v1 WorkerRuntime currently requires max_concurrency=1")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (poll_interval_seconds, heartbeat_interval_seconds)
        ):
            raise ValueError("worker intervals must be finite and positive")
        self.spec = spec
        self.max_concurrency = max_concurrency
        self.executor = executor
        self.client = client
        self.runtime_supervisor = runtime_supervisor
        if serving is not None and not isinstance(serving, WorkerServingAdvertisement):
            raise TypeError("serving must be WorkerServingAdvertisement")
        self.serving = serving
        self.poll_interval_seconds = poll_interval_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._stop = asyncio.Event()
        self._draining = False
        self._registered = False
        self._control_state: WorkerState | None = None
        self._control_valid_until = 0.0
        self._next_heartbeat_at = 0.0
        self._control_lock = asyncio.Lock()
        self._active: ClaimedJob | None = None

    @property
    def control_available(self) -> bool:
        return time.monotonic() < self._control_valid_until

    @property
    def control_state(self) -> str | None:
        return None if self._control_state is None else self._control_state.value

    @property
    def ready(self) -> bool:
        return (
            not self._stop.is_set()
            and self._registered
            and not self.draining
            and self.control_available
            and self._control_state is WorkerState.ONLINE
            and self.runtime_available
        )

    @property
    def runtime_available(self) -> bool:
        return self.runtime_supervisor is None or self.runtime_supervisor.available

    @property
    def runtime_state(self) -> str:
        return "unmanaged" if self.runtime_supervisor is None else self.runtime_supervisor.state

    @property
    def runtime_failure(self) -> str | None:
        return None if self.runtime_supervisor is None else self.runtime_supervisor.failure_reason

    def _require_runtime(self) -> None:
        if not self.runtime_available:
            raise RuntimeUnavailable("RuntimeProvider health is unconfirmed")

    @property
    def registered(self) -> bool:
        return self._registered

    @property
    def draining(self) -> bool:
        return (
            self._draining or self._control_state is WorkerState.DRAINING
            or (self.runtime_supervisor is not None and self.runtime_supervisor.failed.is_set())
        )

    @property
    def active_job_id(self) -> str | None:
        return None if self._active is None else self._active.request.job_id

    def request_drain(self) -> None:
        # A signal can set this while an RPC is in flight. A late ONLINE
        # response must never clear local drain/stop intent.
        self._draining = True

    def request_stop(self) -> None:
        self.request_drain()
        self._stop.set()

    def _control_failed(self, exc: ControlTransportError) -> None:
        if self._control_valid_until:
            _LOG.warning("Control acknowledgement lost (status=%s); claims paused", exc.status_code)
        self._control_valid_until = 0.0
        if exc.status_code in (401, 403, 404):
            self._registered = False
        # No immediate busy-loop retry, including after a failed claim/write.
        self._next_heartbeat_at = time.monotonic() + self.heartbeat_interval_seconds

    def _acknowledge_control(self, record: Any, *, sent_at: float) -> None:
        try:
            if not isinstance(record, dict) or record.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError("invalid Worker response")
            state = WorkerState(record["state"])
            if worker_spec_from_dict(record["spec"]) != self.spec:
                raise ValueError("Worker identity changed")
            if record["max_concurrency"] != self.max_concurrency:
                raise ValueError("Worker capacity changed")
            raw_serving = record.get("serving")
            acknowledged_serving = (
                None
                if raw_serving is None
                else WorkerServingAdvertisement.from_dict(raw_serving)
            )
            if acknowledged_serving is not None and SERVING_EXTENSION not in set(
                record.get("extensions") or ()
            ):
                raise ValueError("Worker acknowledgement lacks serving extension")
            if acknowledged_serving != self.serving:
                raise ValueError("Worker serving identity changed")
        except (KeyError, TypeError, ValueError) as exc:
            error = ControlTransportError(502, "invalid Worker acknowledgement")
            self._control_failed(error)
            raise error from exc
        if state is not self._control_state:
            _LOG.info("Acknowledged Control Worker state: %s", state.value)
        self._registered = True
        self._control_state = state
        # Count from request start, not response arrival: a delayed reply
        # cannot grant an unbounded extension of local readiness.
        self._control_valid_until = (
            sent_at + self.heartbeat_interval_seconds + self.client.timeout_seconds
        )

    async def register(self) -> None:
        async with self._control_lock:
            sent_at = time.monotonic()
            try:
                record = await self.client.register(
                    spec=self.spec,
                    max_concurrency=self.max_concurrency,
                    metadata={"runtime": "astrumweaver-worker"},
                    serving=self.serving,
                )
                self._acknowledge_control(record, sent_at=sent_at)
            except ControlTransportError as exc:
                self._control_failed(exc)
                raise
            self._next_heartbeat_at = time.monotonic() + self.heartbeat_interval_seconds
            if self._draining:
                await self._heartbeat_locked()

    async def _heartbeat_locked(self) -> None:
        """Called only by the execution owner or drain, under _control_lock."""
        active = self._active
        sent_at = time.monotonic()
        # Heartbeat is never allowed to reacquire OFFLINE resources. Control
        # rejects an ONLINE/DRAINING promotion from OFFLINE, including races.
        state = (
            WorkerState.DRAINING
            if (self._draining or (
                self.runtime_supervisor is not None and self.runtime_supervisor.failed.is_set()
            )) and self.control_available
            and self._control_state is not WorkerState.OFFLINE
            else None
        )
        try:
            record = await self.client.heartbeat(
                self.spec.worker_id,
                active_job_id=None if active is None else active.request.job_id,
                lease_token=None if active is None else active.lease_token,
                state=state,
                runtime_instance_epoch=(
                    None
                    if self.serving is None
                    else self.serving.runtime_instance.epoch
                ),
            )
            self._acknowledge_control(record, sent_at=sent_at)
        except ControlTransportError as exc:
            self._control_failed(exc)
            raise
        finally:
            self._next_heartbeat_at = time.monotonic() + self.heartbeat_interval_seconds

    async def _heartbeat_if_due(self) -> None:
        async with self._control_lock:
            if time.monotonic() >= self._next_heartbeat_at:
                await self._heartbeat_locked()

    async def _wait_for_poll(self) -> None:
        delay = min(
            self.poll_interval_seconds,
            max(0.0, self._next_heartbeat_at - time.monotonic()),
        )
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=delay)

    async def run_forever(self) -> None:
        try:
            if self._stop.is_set():
                return
            if self.runtime_supervisor is not None:
                await self.runtime_supervisor.start()
                if not self.runtime_available:
                    # Stay not-ready without an automatic service restart loop.
                    await self._stop.wait()
                    return
            await self.register()
            while not self._stop.is_set():
                try:
                    # Also runs during local/remote drain and OFFLINE pause.
                    # Completion and claim never reset this deadline.
                    await self._heartbeat_if_due()
                    if not self.ready:
                        await self._wait_for_poll()
                        continue
                    async with self._control_lock:
                        if not self.ready:
                            continue
                        # A signal-driven drain RPC might have held the lock.
                        if time.monotonic() >= self._next_heartbeat_at:
                            await self._heartbeat_locked()
                        if not self.ready:
                            continue
                        claimed = await self.client.claim(
                            self.spec.worker_id,
                            runtime_instance_epoch=(
                                None
                                if self.serving is None
                                else self.serving.runtime_instance.epoch
                            ),
                        )
                        self._active = claimed
                    if claimed is None:
                        await self._wait_for_poll()
                        continue
                    try:
                        # A claim may have been admitted before a failed probe
                        # while its response was in flight. Do not execute it.
                        self._require_runtime()
                        await self._execute_claim(claimed)
                    finally:
                        self._active = None
                except RuntimeUnavailable:
                    # Do not manufacture a failure for an unconfirmed attempt.
                    # Server lease expiry/recovery remains authoritative.
                    await self._wait_for_poll()
                except ControlTransportError as exc:
                    # Stay live but not ready. Retry a state-less heartbeat,
                    # never enrollment or automatic ownership reacquisition.
                    self._control_failed(exc)
                    await self._wait_for_poll()
        finally:
            self.request_stop()
            self._control_valid_until = 0.0
            if self.runtime_supervisor is not None:
                await self.runtime_supervisor.close()
                # Do not release an idle GPU's Control reservation before its
                # owned process is stopped. Failed cleanup keeps ownership
                # fail-closed; daemon fallback/service-manager cleanup follows.
                await self.runtime_supervisor.shutdown_owned()
            async with self._control_lock:
                if self._registered:
                    with contextlib.suppress(ControlTransportError):
                        await self.client.set_state(
                            self.spec.worker_id,
                            WorkerState.OFFLINE,
                            runtime_instance_epoch=(
                                None
                                if self.serving is None
                                else self.serving.runtime_instance.epoch
                            ),
                        )
                self._registered = False

    async def drain(self) -> None:
        self.request_drain()
        async with self._control_lock:
            if self._registered and not self._stop.is_set():
                # Do not use set_state(DRAINING), which can reacquire OFFLINE
                # resources. Heartbeat's guarded transition cannot do so.
                await self._heartbeat_locked()

    def _validate_claim_serving(self, claimed: ClaimedJob) -> None:
        metadata = claimed.request.metadata
        raw_binding = metadata.get("serving")
        claimed_values = (
            metadata.get("claimed_deployment_revision"),
            metadata.get("claimed_serving_contract_revision"),
            metadata.get("claimed_runtime_instance_epoch"),
        )
        if raw_binding is None:
            if any(value is not None for value in claimed_values):
                raise RuntimeError("legacy claim carried partial serving identity")
            return
        binding = ServingJobBinding.from_dict(raw_binding)
        if self.serving is None or not worker_serving_matches(self.serving, binding):
            raise RuntimeError("claimed serving binding does not match local deployment")
        if claimed_values != (
            self.serving.deployment_revision,
            binding.serving_contract_revision,
            self.serving.runtime_instance.epoch,
        ):
            raise RuntimeError("claimed serving attempt identity does not match local runtime")

    async def _execute_claim(self, claimed: ClaimedJob) -> None:
        # The private direct-entry test path also installs the active attempt;
        # normal callers install it under the claim lock before reaching here.
        if self._active is not None and self._active is not claimed:
            raise RuntimeError("Worker already owns a different local attempt")
        self._require_runtime()
        self._validate_claim_serving(claimed)
        self._active = claimed
        runtime_failure = (
            asyncio.create_task(self.runtime_supervisor.failed.wait(), name="astrumweaver-runtime-failure")
            if self.runtime_supervisor is not None else None
        )
        execution = asyncio.create_task(
            self.executor.execute(claimed.request), name="astrumweaver-job-execution"
        )
        try:
            while not execution.done():
                self._require_runtime()
                await self._heartbeat_if_due()
                self._require_runtime()
                await asyncio.wait(
                    {execution} if runtime_failure is None else {execution, runtime_failure},
                    timeout=max(0.0, self._next_heartbeat_at - time.monotonic()),
                )

            self._require_runtime()
            error: dict[str, str] | None = None
            retryable = False
            try:
                result = await execution
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self.runtime_supervisor is not None:
                    # Detect a dead runtime before consuming another queued
                    # job. Job validation errors on a healthy runtime retain
                    # the ordinary fenced failure path.
                    await self.runtime_supervisor.check()
                    self._require_runtime()
                error = {"type": type(exc).__name__, "message": str(exc)}
                retryable = True
            else:
                if not isinstance(result, JobResult):
                    error = {
                        "type": "ExecutorProtocolError",
                        "message": "executor returned a non-JobResult value",
                    }

            async with self._control_lock:
                # Renew first if due, even when a short execution completed
                # immediately. No terminal write races a lease heartbeat.
                if time.monotonic() >= self._next_heartbeat_at:
                    await self._heartbeat_locked()
                self._require_runtime()
                if not self.control_available:
                    raise ControlTransportError(0, "Worker authority is unconfirmed")
                if error is not None:
                    await self.client.fail(
                        self.spec.worker_id, claimed.request.job_id,
                        claimed.lease_token, error=error, retryable=retryable,
                        runtime_instance_epoch=(
                            None if self.serving is None
                            else self.serving.runtime_instance.epoch
                        ),
                    )
                else:
                    await self.client.complete(
                        self.spec.worker_id, claimed.request.job_id,
                        claimed.lease_token, result,
                        runtime_instance_epoch=(
                            None if self.serving is None
                            else self.serving.runtime_instance.epoch
                        ),
                    )
                # Do not let a waiting drain heartbeat reuse this terminal lease.
                self._active = None
        except ControlTransportError as exc:
            self._control_failed(exc)
            if exc.status_code == 409:
                async with self._control_lock:
                    status = await self.client.inspect_job(
                        self.spec.worker_id, claimed.request.job_id
                    )
                if status.get("status") == "cancelled":
                    return
            # Unknown/lost lease: abandon local work without reporting a new
            # failure. Durable recovery/fencing decides this attempt's fate.
            raise
        finally:
            try:
                if not execution.done():
                    try:
                        await self.executor.cancel(claimed.request.job_id)
                    finally:
                        execution.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await execution
            finally:
                if runtime_failure is not None:
                    runtime_failure.cancel()
                    await asyncio.gather(runtime_failure, return_exceptions=True)
                # Retrieve a completed task's exception even if health failed
                # before the normal result path. No unobserved task leaks.
                if execution.done() and not execution.cancelled():
                    execution.exception()
                self._active = None
