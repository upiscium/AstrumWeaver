"""Long-running AstrumWeaver Worker claim/execute/heartbeat loop."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..control.models import JobStatus, WorkerHeartbeat, WorkerState
from ..execution import JobExecutor
from ..transport import ClaimedJobDTO
from .client import (
    ControlClient,
    ControlClientError,
    ControlConflict,
    ControlNotFound,
    ControlUnauthorized,
    ControlUnavailable,
)
from .config import WorkerRuntimeConfig
from .preflight import validate_worker_resources


class WorkerDaemon:
    def __init__(
        self,
        *,
        config: WorkerRuntimeConfig,
        executor: JobExecutor,
        client: ControlClient,
    ) -> None:
        self.config = config
        self.executor = executor
        self.client = client
        self.worker_id = config.registration.spec.worker_id
        self.control_state = WorkerState.OFFLINE
        self.registered = False
        self.active_job_id: str | None = None
        self.last_error: str | None = None
        self.last_heartbeat_at: datetime | None = None

    def _status_payload(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "registered": self.registered,
            "control_state": self.control_state.value,
            "active_job_id": self.active_job_id,
            "last_heartbeat_at": (
                self.last_heartbeat_at.isoformat()
                if self.last_heartbeat_at is not None
                else None
            ),
            "last_error": self.last_error,
            "ready": self.registered and self.last_error is None,
            "schedulable": (
                self.registered
                and self.last_error is None
                and self.control_state is WorkerState.ONLINE
            ),
        }

    def write_status(self) -> None:
        path = self.config.status_file
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(self._status_payload(), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    async def initialize(self) -> None:
        validate_worker_resources(self.config.registration.spec)
        record = await self.client.register(self.config.registration)
        self.control_state = record.state
        self.registered = True
        self.last_error = None
        self.last_heartbeat_at = datetime.now(UTC)
        self.write_status()

    async def idle_heartbeat(self) -> None:
        record = await self.client.heartbeat(self.worker_id)
        self.control_state = record.state
        self.last_error = None
        self.last_heartbeat_at = datetime.now(UTC)
        self.write_status()

    async def _cancel_executor(self, job_id: str) -> None:
        with suppress(Exception):
            await self.executor.cancel(job_id)

    async def _monitor_claim(
        self,
        claim: ClaimedJobDTO,
        execution_task: asyncio.Task[Any],
        lease_lost: asyncio.Event,
    ) -> None:
        while not execution_task.done():
            await asyncio.sleep(self.config.heartbeat_interval_seconds)
            if execution_task.done():
                return
            try:
                record = await self.client.heartbeat(
                    self.worker_id,
                    WorkerHeartbeat(
                        active_job_id=claim.job_id,
                        lease_token=claim.lease_token,
                    ),
                )
                self.control_state = record.state
                self.last_heartbeat_at = datetime.now(UTC)
                self.last_error = None
                self.write_status()
            except ControlConflict:
                try:
                    remote = await self.client.job_status(
                        self.worker_id,
                        claim.job_id,
                    )
                except ControlClientError:
                    remote = None
                lease_lost.set()
                if remote is None or remote.status is not JobStatus.RUNNING:
                    await self._cancel_executor(claim.job_id)
                return
            except (ControlUnavailable, ControlUnauthorized, ControlNotFound):
                # Fail closed. Without a confirmed lease renewal the local
                # executor may continue consuming resources, but it no longer
                # has authority to publish a result.
                lease_lost.set()
                self.last_error = "lost control lease authority"
                self.write_status()
                await self._cancel_executor(claim.job_id)
                return

    async def execute_claim(self, claim: ClaimedJobDTO) -> None:
        self.active_job_id = claim.job_id
        self.last_error = None
        self.write_status()

        lease_lost = asyncio.Event()
        execution_task = asyncio.create_task(
            self.executor.execute(claim.to_job_request()),
            name=f"astrumweaver-job-{claim.job_id}",
        )
        monitor_task = asyncio.create_task(
            self._monitor_claim(claim, execution_task, lease_lost),
            name=f"astrumweaver-heartbeat-{claim.job_id}",
        )

        try:
            try:
                result = await execution_task
            except asyncio.CancelledError:
                await self._cancel_executor(claim.job_id)
                raise
            except Exception as exc:
                if not lease_lost.is_set():
                    try:
                        await self.client.fail(
                            worker_id=self.worker_id,
                            job_id=claim.job_id,
                            lease_token=claim.lease_token,
                            error={
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                            retryable=True,
                        )
                    except ControlConflict:
                        pass
                return

            if lease_lost.is_set():
                return

            try:
                await self.client.complete(
                    worker_id=self.worker_id,
                    job_id=claim.job_id,
                    lease_token=claim.lease_token,
                    result=result,
                )
            except ControlConflict:
                # Cancellation, expiry, or re-claim won the durable race.
                return
        finally:
            monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await monitor_task
            self.active_job_id = None
            self.write_status()

    async def run_once(self) -> bool:
        await self.idle_heartbeat()
        if self.control_state is not WorkerState.ONLINE:
            return False

        claim = await self.client.claim(self.worker_id)
        if claim is None:
            return False

        await self.execute_claim(claim)
        return True

    async def run_forever(self, stop_event: asyncio.Event | None = None) -> None:
        stop = stop_event or asyncio.Event()
        try:
            await self.initialize()
            while not stop.is_set():
                try:
                    did_work = await self.run_once()
                    self.last_error = None
                except ControlUnauthorized:
                    self.last_error = "worker credentials rejected"
                    self.write_status()
                    raise
                except ControlClientError:
                    self.last_error = "control unavailable"
                    self.write_status()
                    did_work = False

                if not did_work and not stop.is_set():
                    try:
                        await asyncio.wait_for(
                            stop.wait(),
                            timeout=self.config.poll_interval_seconds,
                        )
                    except TimeoutError:
                        pass
        finally:
            await self.client.close()
