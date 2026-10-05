from __future__ import annotations

import httpx
import pytest

from astrumweaver import (
    JobExecutionError,
    ResidencyReport,
    ResourceShape,
    WorkerSpec,
)
from astrumweaver.control import (
    InMemoryControlRepository,
    JobStatus,
    JobSubmission,
)
from astrumweaver.control.api import create_app
from astrumweaver.worker import ControlClient, WorkerRuntime


class RejectingExecutor:
    capabilities = frozenset({"test.reject"})

    async def execute(self, _job):
        raise JobExecutionError(
            "context_length_exceeded",
            "request exceeds the reviewed execution limit",
            retryable=False,
        )

    async def cancel(self, _job_id: str) -> None:
        return None

    async def residency(self) -> ResidencyReport:
        return ResidencyReport()


@pytest.mark.asyncio
async def test_nonretryable_executor_rejection_is_terminal_after_one_attempt():
    repository = InMemoryControlRepository()
    app = create_app(
        repository,
        client_token="client-secret",
        worker_token="worker-secret",
        maintenance_interval_seconds=60,
    )
    transport = httpx.ASGITransport(app=app)
    spec = WorkerSpec(
        worker_id="rejecting-worker",
        worker_class="cpu-test",
        resources=ResourceShape(),
        capabilities=frozenset({"test.reject"}),
    )

    async with ControlClient(
        "http://control",
        "worker-secret",
        transport=transport,
    ) as control:
        runtime = WorkerRuntime(
            spec=spec,
            max_concurrency=1,
            executor=RejectingExecutor(),
            client=control,
            heartbeat_interval_seconds=1,
        )
        await runtime.register()
        job = repository.submit_job(
            JobSubmission(
                capability="test.reject",
                max_attempts=3,
            )
        )
        claimed = await control.claim(spec.worker_id)
        assert claimed is not None
        await runtime._execute_claim(claimed)

    final = repository.get_job(job.job_id)
    assert final.status is JobStatus.FAILED
    assert final.attempts == 1
    assert final.error["type"] == "JobExecutionError"
    assert final.error["code"] == "context_length_exceeded"
    assert final.error["retryable"] is False
