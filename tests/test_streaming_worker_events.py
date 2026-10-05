from __future__ import annotations

import httpx
import pytest

from astrumweaver import (
    JobEvent,
    JobResult,
    ResidencyReport,
    ResourceShape,
    WorkerSpec,
)
from astrumweaver.control import InMemoryControlRepository, JobStatus, JobSubmission
from astrumweaver.control.api import create_app
from astrumweaver.worker import ControlClient, WorkerRuntime


WORKER_TOKEN = "worker-secret"


class StreamingExecutor:
    capabilities = frozenset({"debug.echo"})

    def __init__(self, *, count: int = 2) -> None:
        self.count = count
        self.used_stream = False

    async def execute(self, job):
        raise AssertionError("streaming executor used legacy execute path")

    async def execute_stream(self, job, sink):
        self.used_stream = True
        for index in range(self.count):
            await sink.emit(
                JobEvent(
                    kind="debug.chunk",
                    payload={"index": index, "value": job.payload.get("value")},
                )
            )
        return JobResult(outputs={"ok": True})

    async def cancel(self, job_id: str) -> None:
        return None

    async def residency(self):
        return ResidencyReport()


async def runtime_and_claim(repo, executor):
    app = create_app(
        repo,
        client_token=None,
        worker_token=WORKER_TOKEN,
        client_auth="none",
        maintenance_interval_seconds=60,
    )
    transport = httpx.ASGITransport(app=app)
    control = ControlClient(
        "http://control",
        WORKER_TOKEN,
        transport=transport,
    )
    spec = WorkerSpec(
        worker_id="stream-worker",
        worker_class="cpu-test",
        resources=ResourceShape(),
        capabilities=frozenset({"debug.echo"}),
    )
    runtime = WorkerRuntime(
        spec=spec,
        max_concurrency=1,
        executor=executor,
        client=control,
        heartbeat_interval_seconds=1,
    )
    await runtime.register()
    job = repo.submit_job(
        JobSubmission(
            capability="debug.echo",
            payload={"value": 7},
            max_attempts=1,
        )
    )
    claimed = await control.claim(spec.worker_id)
    assert claimed is not None and claimed.request.job_id == job.job_id
    return control, runtime, job, claimed


@pytest.mark.asyncio
async def test_streaming_executor_events_are_published_before_success():
    repo = InMemoryControlRepository()
    executor = StreamingExecutor()
    control, runtime, job, claimed = await runtime_and_claim(repo, executor)
    try:
        await runtime._execute_claim(claimed)
    finally:
        await control.aclose()

    assert executor.used_stream
    events = repo.list_job_events(job.job_id)
    assert [event.sequence for event in events] == [1, 2]
    assert [event.payload["index"] for event in events] == [0, 1]
    assert all(event.attempt == 1 for event in events)
    assert repo.get_job(job.job_id).status is JobStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_streaming_executor_buffer_overflow_fails_job_without_retry():
    repo = InMemoryControlRepository(event_max_count=1)
    executor = StreamingExecutor(count=2)
    control, runtime, job, claimed = await runtime_and_claim(repo, executor)
    try:
        await runtime._execute_claim(claimed)
    finally:
        await control.aclose()

    events = repo.list_job_events(job.job_id)
    assert len(events) == 1
    record = repo.get_job(job.job_id)
    assert record.status is JobStatus.FAILED
    assert record.attempts == 1
    assert record.error["code"] == "event_buffer_full"
    assert record.error["retryable"] is False
