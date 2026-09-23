from __future__ import annotations

import asyncio

import httpx
import pytest

from astrumweaver import ArtifactRef, JobResult, ResourceShape, WorkerSpec
from astrumweaver.control.api import create_app
from astrumweaver.control.daemon import build_app
from astrumweaver.control.repository import InMemoryControlRepository, StorageUnavailable
from astrumweaver.executors.structured_echo import StructuredEchoExecutor
from astrumweaver.transport import PROTOCOL_VERSION
from astrumweaver.worker import (
    ControlClient,
    WorkerRuntime,
    require_executor_capabilities,
)


CLIENT_TOKEN = "client-secret"
WORKER_TOKEN = "worker-secret"


def auth(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


def worker_registration(
    worker_id: str = "worker-a",
    *,
    capability: str = "image.generate",
) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "spec": {
            "worker_id": worker_id,
            "worker_class": "cpu-test",
            "gpu_uuids": [],
            "capabilities": [capability],
            "labels": {"test": "true"},
            "resources": {
                "gpu_count": 0,
                "total_vram_mb": 0,
                "max_single_gpu_vram_mb": 0,
            },
        },
        "max_concurrency": 1,
        "metadata": {},
    }


def job_submission(capability: str = "image.generate") -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "capability": capability,
        "payload": {"input": "example"},
        "requirements": {},
        "priority": 0,
        "max_attempts": 3,
    }


@pytest.fixture
def repository() -> InMemoryControlRepository:
    return InMemoryControlRepository()


@pytest.fixture
def app(repository: InMemoryControlRepository):
    return create_app(
        repository,
        client_token=CLIENT_TOKEN,
        worker_token=WORKER_TOKEN,
        maintenance_interval_seconds=60.0,
    )


@pytest.mark.asyncio
async def test_auth_boundaries_and_protocol_version(app) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        worker_submit = await client.post(
            "/v1/jobs",
            headers=auth(WORKER_TOKEN),
            json=job_submission(),
        )
        client_register = await client.post(
            "/v1/workers/register",
            headers=auth(CLIENT_TOKEN),
            json=worker_registration(),
        )
        wrong_version = await client.post(
            "/v1/jobs",
            headers=auth(CLIENT_TOKEN),
            json={**job_submission(), "protocol_version": "v999"},
        )
        health = await client.get("/v1/health")

    assert worker_submit.status_code == 401
    assert client_register.status_code == 401
    assert wrong_version.status_code == 409
    assert health.status_code == 200
    assert health.json()["protocol_version"] == "v1"


@pytest.mark.asyncio
async def test_generic_non_text_result_round_trip(app) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        registered = await client.post(
            "/v1/workers/register",
            headers=auth(WORKER_TOKEN),
            json=worker_registration(),
        )
        assert registered.status_code == 201

        created = await client.post(
            "/v1/jobs",
            headers=auth(CLIENT_TOKEN),
            json=job_submission(),
        )
        assert created.status_code == 201
        job_id = created.json()["job_id"]

        claimed = await client.post(
            "/v1/workers/worker-a/jobs/claim",
            headers=auth(WORKER_TOKEN),
        )
        assert claimed.status_code == 200
        lease_token = claimed.json()["lease_token"]

        completed = await client.post(
            f"/v1/workers/worker-a/jobs/{job_id}/complete",
            headers=auth(WORKER_TOKEN),
            json={
                "protocol_version": PROTOCOL_VERSION,
                "lease_token": lease_token,
                "result": {
                    "outputs": {"width": 1024, "height": 1024},
                    "artifacts": [
                        {
                            "uri": "artifact://job/image.png",
                            "media_type": "image/png",
                            "size_bytes": 1234,
                            "metadata": {},
                        }
                    ],
                    "metrics": {"elapsed_ms": 10.5},
                    "text": None,
                    "metadata": {"executor": "image-test"},
                },
            },
        )
        fetched = await client.get(
            f"/v1/jobs/{job_id}",
            headers=auth(CLIENT_TOKEN),
        )

    assert completed.status_code == 200
    result = fetched.json()["result"]
    assert result["text"] is None
    assert result["outputs"]["width"] == 1024
    assert result["artifacts"][0]["media_type"] == "image/png"


@pytest.mark.asyncio
async def test_draining_worker_cannot_claim_new_job(app) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        await client.post(
            "/v1/workers/register",
            headers=auth(WORKER_TOKEN),
            json=worker_registration(),
        )
        state = await client.post(
            "/v1/workers/worker-a/state",
            headers=auth(WORKER_TOKEN),
            json={"protocol_version": PROTOCOL_VERSION, "state": "draining"},
        )
        assert state.status_code == 200

        await client.post(
            "/v1/jobs",
            headers=auth(CLIENT_TOKEN),
            json=job_submission(),
        )
        claimed = await client.post(
            "/v1/workers/worker-a/jobs/claim",
            headers=auth(WORKER_TOKEN),
        )

    assert claimed.status_code == 204
    assert claimed.content == b""


@pytest.mark.asyncio
async def test_stale_fencing_token_rejected_over_transport() -> None:
    repository = InMemoryControlRepository(lease_seconds=1)
    app = create_app(
        repository,
        client_token=CLIENT_TOKEN,
        worker_token=WORKER_TOKEN,
        maintenance_interval_seconds=60.0,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        await client.post(
            "/v1/workers/register",
            headers=auth(WORKER_TOKEN),
            json=worker_registration(capability="decision.system_one"),
        )
        created = await client.post(
            "/v1/jobs",
            headers=auth(CLIENT_TOKEN),
            json=job_submission(capability="decision.system_one"),
        )
        job_id = created.json()["job_id"]

        first = await client.post(
            "/v1/workers/worker-a/jobs/claim",
            headers=auth(WORKER_TOKEN),
        )
        old_token = first.json()["lease_token"]

        await asyncio.sleep(1.05)
        expired_complete = await client.post(
            f"/v1/workers/worker-a/jobs/{job_id}/complete",
            headers=auth(WORKER_TOKEN),
            json={
                "protocol_version": PROTOCOL_VERSION,
                "lease_token": old_token,
                "result": {"outputs": {"stale": True}},
            },
        )
        assert expired_complete.status_code == 409

        repository.recover_expired_jobs()
        second = await client.post(
            "/v1/workers/worker-a/jobs/claim",
            headers=auth(WORKER_TOKEN),
        )
        new_token = second.json()["lease_token"]
        assert new_token != old_token

        stale_again = await client.post(
            f"/v1/workers/worker-a/jobs/{job_id}/complete",
            headers=auth(WORKER_TOKEN),
            json={
                "protocol_version": PROTOCOL_VERSION,
                "lease_token": old_token,
                "result": {"outputs": {"stale": True}},
            },
        )

    assert stale_again.status_code == 409


class SlowExecutor:
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.started = asyncio.Event()

    async def execute(self, job):
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)

    async def residency(self):
        from astrumweaver import ResidencyReport

        return ResidencyReport()


@pytest.mark.asyncio
async def test_worker_runtime_observes_durable_remote_cancellation(app) -> None:
    transport = httpx.ASGITransport(app=app)
    worker_spec = WorkerSpec(
        worker_id="runtime-worker",
        worker_class="cpu-test",
        resources=ResourceShape(),
        capabilities=frozenset({"custom.slow"}),
    )
    executor = SlowExecutor()

    async with ControlClient(
        "http://control",
        WORKER_TOKEN,
        transport=transport,
    ) as control:
        runtime = WorkerRuntime(
            spec=worker_spec,
            max_concurrency=1,
            executor=executor,
            client=control,
            poll_interval_seconds=0.01,
            heartbeat_interval_seconds=0.02,
        )
        await runtime.register()

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://control",
        ) as client:
            created = await client.post(
                "/v1/jobs",
                headers=auth(CLIENT_TOKEN),
                json=job_submission(capability="custom.slow"),
            )
            job_id = created.json()["job_id"]

            claimed = await control.claim(worker_spec.worker_id)
            assert claimed is not None
            execution = asyncio.create_task(runtime._execute_claim(claimed))
            await executor.started.wait()

            cancelled = await client.post(
                f"/v1/jobs/{job_id}/cancel",
                headers=auth(CLIENT_TOKEN),
            )
            assert cancelled.status_code == 200

            await asyncio.wait_for(execution, timeout=1.0)

    assert executor.cancelled == [job_id]


@pytest.mark.asyncio
async def test_worker_runtime_completes_structured_echo_result(app) -> None:
    transport = httpx.ASGITransport(app=app)
    spec = WorkerSpec(
        worker_id="echo-worker",
        worker_class="cpu-test",
        resources=ResourceShape(),
        capabilities=frozenset({"debug.echo"}),
    )
    async with ControlClient("http://control", WORKER_TOKEN, transport=transport) as control:
        runtime = WorkerRuntime(
            spec=spec,
            max_concurrency=1,
            executor=StructuredEchoExecutor(),
            client=control,
            heartbeat_interval_seconds=0.05,
        )
        await runtime.register()
        async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
            created = await client.post(
                "/v1/jobs",
                headers=auth(CLIENT_TOKEN),
                json={
                    **job_submission(capability="debug.echo"),
                    "payload": {"value": 7},
                },
            )
            job_id = created.json()["job_id"]
            claimed = await control.claim(spec.worker_id)
            assert claimed is not None
            await runtime._execute_claim(claimed)
            fetched = await client.get(f"/v1/jobs/{job_id}", headers=auth(CLIENT_TOKEN))

    result = fetched.json()["result"]
    assert result["text"] is None
    assert result["outputs"]["payload"] == {"value": 7}


@pytest.mark.asyncio
async def test_readiness_reflects_storage_failure() -> None:
    class BrokenRepository(InMemoryControlRepository):
        def check_storage(self) -> None:
            raise StorageUnavailable("database password=must-not-leak")

    broken = create_app(
        BrokenRepository(),
        client_token=CLIENT_TOKEN,
        worker_token=WORKER_TOKEN,
    )
    transport = httpx.ASGITransport(app=broken, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        response = await client.get("/v1/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "storage unavailable"}
    assert "must-not-leak" not in response.text


class DelayedStructuredExecutor:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def execute(self, job):
        self.started.set()
        await asyncio.sleep(0.06)
        return JobResult(
            outputs={"value": job.payload.get("value"), "kind": "structured"},
            text=None,
        )

    async def cancel(self, job_id: str) -> None:
        return None

    async def residency(self):
        from astrumweaver import ResidencyReport

        return ResidencyReport()


@pytest.mark.asyncio
async def test_worker_run_forever_register_claim_heartbeat_complete_loop() -> None:
    repository = InMemoryControlRepository(lease_seconds=1)
    app = create_app(
        repository,
        client_token=CLIENT_TOKEN,
        worker_token=WORKER_TOKEN,
        maintenance_interval_seconds=60.0,
    )
    transport = httpx.ASGITransport(app=app)
    spec = WorkerSpec(
        worker_id="loop-worker",
        worker_class="cpu-test",
        resources=ResourceShape(),
        capabilities=frozenset({"structured.work"}),
    )
    executor = DelayedStructuredExecutor()

    async with ControlClient("http://control", WORKER_TOKEN, transport=transport) as control:
        runtime = WorkerRuntime(
            spec=spec,
            max_concurrency=1,
            executor=executor,
            client=control,
            poll_interval_seconds=0.01,
            heartbeat_interval_seconds=0.01,
        )
        runtime_task = asyncio.create_task(runtime.run_forever())

        for _ in range(100):
            if runtime.registered:
                break
            await asyncio.sleep(0.01)
        assert runtime.registered

        async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
            created = await client.post(
                "/v1/jobs",
                headers=auth(CLIENT_TOKEN),
                json={
                    **job_submission(capability="structured.work"),
                    "payload": {"value": 9},
                },
            )
            assert created.status_code == 201
            job_id = created.json()["job_id"]

            await asyncio.wait_for(executor.started.wait(), timeout=1.0)

            result = None
            for _ in range(200):
                fetched = await client.get(
                    f"/v1/jobs/{job_id}",
                    headers=auth(CLIENT_TOKEN),
                )
                if fetched.json()["status"] == "succeeded":
                    result = fetched.json()["result"]
                    break
                await asyncio.sleep(0.01)

            assert result is not None
            assert result["text"] is None
            assert result["outputs"] == {"value": 9, "kind": "structured"}

        runtime.request_stop()
        await asyncio.wait_for(runtime_task, timeout=1.0)

    assert repository.get_worker(spec.worker_id).state.value == "offline"


def test_executor_capabilities_must_cover_worker_advertisement() -> None:
    executor = StructuredEchoExecutor()

    require_executor_capabilities(
        executor,
        frozenset({"debug.echo"}),
    )

    with pytest.raises(RuntimeError, match="does not support advertised"):
        require_executor_capabilities(
            executor,
            frozenset({"debug.echo", "image.generate"}),
        )


def test_executor_without_capability_declaration_is_rejected() -> None:
    class UndeclaredExecutor:
        async def execute(self, job):
            return JobResult()

        async def cancel(self, job_id: str) -> None:
            return None

        async def residency(self):
            from astrumweaver import ResidencyReport

            return ResidencyReport()

    with pytest.raises(TypeError, match="must declare capabilities"):
        require_executor_capabilities(
            UndeclaredExecutor(),
            frozenset({"custom.work"}),
        )


def test_control_daemon_requires_explicit_postgres_and_authority(
    tmp_path,
    monkeypatch,
) -> None:
    config = tmp_path / "control.toml"
    config.write_text(
        "[control]\nhost = \"127.0.0.1\"\nport = 9000\n",
        encoding="utf-8",
    )

    for name in (
        "ASTRUMWEAVER_DATABASE_URL",
        "ASTRUMWEAVER_CLIENT_TOKEN",
        "ASTRUMWEAVER_WORKER_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="ASTRUMWEAVER_DATABASE_URL is required"):
        build_app(str(config))

    monkeypatch.setenv(
        "ASTRUMWEAVER_DATABASE_URL",
        "postgresql://example.invalid/astrumweaver",
    )
    with pytest.raises(RuntimeError, match="ASTRUMWEAVER_CLIENT_TOKEN is required"):
        build_app(str(config))

    monkeypatch.setenv("ASTRUMWEAVER_CLIENT_TOKEN", "client")
    with pytest.raises(RuntimeError, match="ASTRUMWEAVER_WORKER_TOKEN is required"):
        build_app(str(config))

    monkeypatch.setenv("ASTRUMWEAVER_WORKER_TOKEN", "worker")
    app = build_app(str(config))

    assert app.title == "AstrumWeaver Control API"
