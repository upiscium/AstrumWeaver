from __future__ import annotations

import httpx
import pytest

from astrumweaver import ResourceShape, WorkerSpec
from astrumweaver.control.api import create_app
from astrumweaver.control.repository import InMemoryControlRepository
from astrumweaver.executors.structured_echo import StructuredEchoExecutor
from astrumweaver.worker import ControlClient, WorkerRuntime
from astrumweaver.worker.health import create_health_app


@pytest.mark.asyncio
async def test_worker_health_becomes_not_ready_when_draining() -> None:
    repository = InMemoryControlRepository()
    control_app = create_app(
        repository,
        client_token="client-test-token",
        worker_token="worker-test-token",
        maintenance_interval_seconds=60.0,
    )
    control_transport = httpx.ASGITransport(app=control_app)
    spec = WorkerSpec(
        worker_id="health-worker",
        worker_class="cpu-test",
        resources=ResourceShape(),
        capabilities=frozenset({"debug.echo"}),
    )

    async with ControlClient(
        "http://control",
        "worker-test-token",
        transport=control_transport,
    ) as control:
        runtime = WorkerRuntime(
            spec=spec,
            max_concurrency=1,
            executor=StructuredEchoExecutor(),
            client=control,
        )
        await runtime.register()

        health_app = create_health_app(runtime)
        health_transport = httpx.ASGITransport(app=health_app)
        async with httpx.AsyncClient(
            transport=health_transport,
            base_url="http://worker",
        ) as client:
            before = await client.get("/ready")
            assert before.status_code == 200
            assert before.json()["ready"] is True

            await runtime.drain()

            health = await client.get("/health")
            ready = await client.get("/ready")

    assert health.json()["draining"] is True
    assert health.json()["registered"] is True
    assert health.json()["ready"] is False
    assert ready.status_code == 503
    assert ready.json()["ready"] is False
