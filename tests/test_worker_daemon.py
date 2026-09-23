from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from astrumweaver import JobResult, ResourceShape, WorkerSpec
from astrumweaver.control import InMemoryControlRepository
from astrumweaver.control.models import WorkerRegistration
from astrumweaver.control.service import create_app
from astrumweaver.execution import JobExecutor, JobRequest, ResidencyReport
from astrumweaver.transport import AuthConfig
from astrumweaver.worker import ControlClient, WorkerDaemon, WorkerRuntimeConfig


CLIENT_TOKEN = "client-secret"
WORKER_TOKEN = "worker-secret"


class RecordingExecutor:
    def __init__(self) -> None:
        self.jobs: list[JobRequest] = []
        self.cancelled: list[str] = []

    async def execute(self, job: JobRequest) -> JobResult:
        self.jobs.append(job)
        return JobResult(
            outputs={"normalized": True, "input": dict(job.payload)},
            metrics={"elapsed_ms": 1},
        )

    async def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)

    async def residency(self) -> ResidencyReport:
        return ResidencyReport()


class BlockingExecutor:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.cancel_calls: list[str] = []

    async def execute(self, job: JobRequest) -> JobResult:
        self.started.set()
        await self.cancelled.wait()
        return JobResult(outputs={"late": True})

    async def cancel(self, job_id: str) -> None:
        self.cancel_calls.append(job_id)
        self.cancelled.set()

    async def residency(self) -> ResidencyReport:
        return ResidencyReport()


def worker_config(tmp_path: Path) -> WorkerRuntimeConfig:
    return WorkerRuntimeConfig(
        registration=WorkerRegistration(
            spec=WorkerSpec(
                worker_id="worker-runtime",
                worker_class="cpu",
                resources=ResourceShape(),
                capabilities=frozenset({"metadata.normalize"}),
            )
        ),
        control_url="http://control",
        executor_factory="unused:unused",
        executor_settings={},
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.05,
        request_timeout_seconds=1.0,
        status_file=tmp_path / "status.json",
        allow_insecure_http=True,
    )


def headers(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_worker_register_claim_execute_complete_loop(tmp_path: Path) -> None:
    repository = InMemoryControlRepository()
    app = create_app(
        repository=repository,
        auth=AuthConfig(client_token=CLIENT_TOKEN, worker_token=WORKER_TOKEN),
    )
    transport = httpx.ASGITransport(app=app)
    worker_http = httpx.AsyncClient(transport=transport, base_url="http://control")
    client_http = httpx.AsyncClient(transport=transport, base_url="http://control")
    executor = RecordingExecutor()
    control = ControlClient(
        base_url="http://control",
        worker_token=WORKER_TOKEN,
        http_client=worker_http,
    )
    daemon = WorkerDaemon(
        config=worker_config(tmp_path),
        executor=executor,
        client=control,
    )

    try:
        await daemon.initialize()
        created = await client_http.post(
            "/v1/jobs",
            headers=headers(CLIENT_TOKEN),
            json={
                "capability": "metadata.normalize",
                "payload": {"value": 1},
            },
        )
        job_id = created.json()["job_id"]

        assert await daemon.run_once()

        viewed = await client_http.get(
            f"/v1/jobs/{job_id}",
            headers=headers(CLIENT_TOKEN),
        )
        assert viewed.json()["status"] == "succeeded"
        assert viewed.json()["result"]["outputs"] == {
            "normalized": True,
            "input": {"value": 1},
        }
        assert len(executor.jobs) == 1
        assert daemon.active_job_id is None
        assert (tmp_path / "status.json").exists()
    finally:
        await worker_http.aclose()
        await client_http.aclose()


@pytest.mark.asyncio
async def test_worker_obeys_remote_draining_state(tmp_path: Path) -> None:
    repository = InMemoryControlRepository()
    app = create_app(
        repository=repository,
        auth=AuthConfig(client_token=CLIENT_TOKEN, worker_token=WORKER_TOKEN),
    )
    transport = httpx.ASGITransport(app=app)
    worker_http = httpx.AsyncClient(transport=transport, base_url="http://control")
    client_http = httpx.AsyncClient(transport=transport, base_url="http://control")
    executor = RecordingExecutor()
    daemon = WorkerDaemon(
        config=worker_config(tmp_path),
        executor=executor,
        client=ControlClient(
            base_url="http://control",
            worker_token=WORKER_TOKEN,
            http_client=worker_http,
        ),
    )

    try:
        await daemon.initialize()
        await client_http.put(
            "/v1/workers/worker-runtime/state",
            headers=headers(CLIENT_TOKEN),
            json={"state": "draining"},
        )
        await client_http.post(
            "/v1/jobs",
            headers=headers(CLIENT_TOKEN),
            json={"capability": "metadata.normalize"},
        )

        assert not await daemon.run_once()
        assert daemon.control_state.value == "draining"
        assert executor.jobs == []
    finally:
        await worker_http.aclose()
        await client_http.aclose()


@pytest.mark.asyncio
async def test_worker_cancels_local_executor_when_control_cancels_job(
    tmp_path: Path,
) -> None:
    repository = InMemoryControlRepository(lease_seconds=5)
    app = create_app(
        repository=repository,
        auth=AuthConfig(client_token=CLIENT_TOKEN, worker_token=WORKER_TOKEN),
    )
    transport = httpx.ASGITransport(app=app)
    worker_http = httpx.AsyncClient(transport=transport, base_url="http://control")
    client_http = httpx.AsyncClient(transport=transport, base_url="http://control")
    executor = BlockingExecutor()
    daemon = WorkerDaemon(
        config=worker_config(tmp_path),
        executor=executor,
        client=ControlClient(
            base_url="http://control",
            worker_token=WORKER_TOKEN,
            http_client=worker_http,
        ),
    )

    try:
        await daemon.initialize()
        created = await client_http.post(
            "/v1/jobs",
            headers=headers(CLIENT_TOKEN),
            json={"capability": "metadata.normalize"},
        )
        job_id = created.json()["job_id"]

        worker_task = asyncio.create_task(daemon.run_once())
        await asyncio.wait_for(executor.started.wait(), timeout=1.0)

        cancelled = await client_http.post(
            f"/v1/jobs/{job_id}/cancel",
            headers=headers(CLIENT_TOKEN),
        )
        assert cancelled.status_code == 200

        await asyncio.wait_for(worker_task, timeout=1.0)

        viewed = await client_http.get(
            f"/v1/jobs/{job_id}",
            headers=headers(CLIENT_TOKEN),
        )
        assert viewed.json()["status"] == "cancelled"
        assert executor.cancel_calls == [job_id]
    finally:
        await worker_http.aclose()
        await client_http.aclose()


def test_worker_daemon_structurally_accepts_generic_executor(tmp_path: Path) -> None:
    assert isinstance(RecordingExecutor(), JobExecutor)
