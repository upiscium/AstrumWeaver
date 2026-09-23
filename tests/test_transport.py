from __future__ import annotations

import asyncio
from datetime import timedelta

import httpx
import pytest

from astrumweaver import JobResult, ResourceShape, WorkerSpec
from astrumweaver.control import InMemoryControlRepository, utc_now
from astrumweaver.control.service import create_app
from astrumweaver.transport import AuthConfig


CLIENT_TOKEN = "client-secret"
WORKER_TOKEN = "worker-secret"


def client_headers() -> dict[str, str]:
    return {"authorization": f"Bearer {CLIENT_TOKEN}"}


def worker_headers() -> dict[str, str]:
    return {"authorization": f"Bearer {WORKER_TOKEN}"}


def cpu_registration(worker_id: str = "worker-cpu") -> dict[str, object]:
    return {
        "spec": {
            "worker_id": worker_id,
            "worker_class": "cpu",
            "resources": {
                "gpu_count": 0,
                "total_vram_mb": 0,
                "max_single_gpu_vram_mb": 0,
            },
            "gpu_uuids": [],
            "capabilities": ["image.generate", "metadata.normalize"],
            "labels": {"runtime_family": "test"},
        },
        "max_concurrency": 1,
        "metadata": {},
    }


@pytest.mark.asyncio
async def test_transport_separates_client_and_worker_authority() -> None:
    repository = InMemoryControlRepository()
    app = create_app(
        repository=repository,
        auth=AuthConfig(client_token=CLIENT_TOKEN, worker_token=WORKER_TOKEN),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://control",
    ) as client:
        worker_submit = await client.post(
            "/v1/jobs",
            headers=worker_headers(),
            json={"capability": "metadata.normalize"},
        )
        client_register = await client.post(
            "/v1/workers/register",
            headers=client_headers(),
            json=cpu_registration(),
        )
        no_token = await client.post(
            "/v1/jobs",
            json={"capability": "metadata.normalize"},
        )

    assert worker_submit.status_code == 401
    assert client_register.status_code == 401
    assert no_token.status_code == 401


@pytest.mark.asyncio
async def test_generic_non_text_result_crosses_transport_without_leaking_lease() -> None:
    repository = InMemoryControlRepository()
    app = create_app(
        repository=repository,
        auth=AuthConfig(client_token=CLIENT_TOKEN, worker_token=WORKER_TOKEN),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://control",
    ) as client:
        registered = await client.post(
            "/v1/workers/register",
            headers=worker_headers(),
            json=cpu_registration(),
        )
        assert registered.status_code == 201

        created = await client.post(
            "/v1/jobs",
            headers=client_headers(),
            json={
                "capability": "image.generate",
                "payload": {"prompt": "example"},
            },
        )
        assert created.status_code == 201
        job_id = created.json()["job_id"]

        claimed = await client.post(
            "/v1/workers/worker-cpu/jobs/claim",
            headers=worker_headers(),
        )
        assert claimed.status_code == 200
        lease_token = claimed.json()["lease_token"]

        completed = await client.post(
            f"/v1/jobs/{job_id}/complete",
            headers=worker_headers(),
            json={
                "worker_id": "worker-cpu",
                "lease_token": lease_token,
                "result": {
                    "outputs": {"width": 1024, "height": 1024, "seed": 42},
                    "artifacts": [
                        {
                            "uri": "artifact://job/image.png",
                            "media_type": "image/png",
                            "digest": "sha256:example",
                            "size_bytes": 1234,
                            "metadata": {},
                        }
                    ],
                    "metrics": {"elapsed_ms": 10.5},
                    "metadata": {"executor": "test"},
                },
            },
        )
        assert completed.status_code == 200

        viewed = await client.get(
            f"/v1/jobs/{job_id}",
            headers=client_headers(),
        )

    assert viewed.status_code == 200
    payload = viewed.json()
    assert payload["status"] == "succeeded"
    assert payload["result"]["text"] is None
    assert payload["result"]["outputs"]["width"] == 1024
    assert payload["result"]["artifacts"][0]["media_type"] == "image/png"
    assert "lease_token" not in payload


@pytest.mark.asyncio
async def test_stale_fencing_token_is_rejected_over_http_after_reclaim() -> None:
    repository = InMemoryControlRepository(lease_seconds=1)
    app = create_app(
        repository=repository,
        auth=AuthConfig(client_token=CLIENT_TOKEN, worker_token=WORKER_TOKEN),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://control",
    ) as client:
        await client.post(
            "/v1/workers/register",
            headers=worker_headers(),
            json=cpu_registration(),
        )
        created = await client.post(
            "/v1/jobs",
            headers=client_headers(),
            json={"capability": "image.generate"},
        )
        job_id = created.json()["job_id"]

        first = await client.post(
            "/v1/workers/worker-cpu/jobs/claim",
            headers=worker_headers(),
        )
        first_token = first.json()["lease_token"]

        await asyncio.sleep(1.05)
        repository.recover_expired_jobs()

        second = await client.post(
            "/v1/workers/worker-cpu/jobs/claim",
            headers=worker_headers(),
        )
        assert second.status_code == 200
        second_token = second.json()["lease_token"]
        assert second_token != first_token

        stale = await client.post(
            f"/v1/jobs/{job_id}/complete",
            headers=worker_headers(),
            json={
                "worker_id": "worker-cpu",
                "lease_token": first_token,
                "result": {"outputs": {"stale": True}},
            },
        )
        current = await client.post(
            f"/v1/jobs/{job_id}/complete",
            headers=worker_headers(),
            json={
                "worker_id": "worker-cpu",
                "lease_token": second_token,
                "result": {"outputs": {"ok": True}},
            },
        )

    assert stale.status_code == 409
    assert current.status_code == 200


@pytest.mark.asyncio
async def test_draining_worker_heartbeat_reflects_state_and_claims_nothing() -> None:
    repository = InMemoryControlRepository()
    app = create_app(
        repository=repository,
        auth=AuthConfig(client_token=CLIENT_TOKEN, worker_token=WORKER_TOKEN),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://control",
    ) as client:
        await client.post(
            "/v1/workers/register",
            headers=worker_headers(),
            json=cpu_registration(),
        )
        await client.post(
            "/v1/jobs",
            headers=client_headers(),
            json={"capability": "metadata.normalize"},
        )
        state = await client.put(
            "/v1/workers/worker-cpu/state",
            headers=client_headers(),
            json={"state": "draining"},
        )
        heartbeat = await client.post(
            "/v1/workers/worker-cpu/heartbeat",
            headers=worker_headers(),
            json={},
        )
        claim = await client.post(
            "/v1/workers/worker-cpu/jobs/claim",
            headers=worker_headers(),
        )

    assert state.status_code == 200
    assert heartbeat.json()["state"] == "draining"
    assert claim.status_code == 204
    assert claim.content == b""


@pytest.mark.asyncio
async def test_ready_probe_uses_repository_health() -> None:
    repository = InMemoryControlRepository()
    app = create_app(
        repository=repository,
        auth=AuthConfig(client_token=CLIENT_TOKEN, worker_token=WORKER_TOKEN),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://control",
    ) as client:
        health = await client.get("/v1/healthz")
        ready = await client.get("/v1/readyz")

    assert health.json() == {"status": "ok", "protocol_version": "v1"}
    assert ready.json() == {"status": "ready", "protocol_version": "v1"}


@pytest.mark.asyncio
async def test_unexpected_control_exception_is_redacted() -> None:
    class BrokenRepository(InMemoryControlRepository):
        def submit_job(self, submission, *, now=None):
            raise RuntimeError("database password=must-not-leak")

    app = create_app(
        repository=BrokenRepository(),
        auth=AuthConfig(client_token=CLIENT_TOKEN, worker_token=WORKER_TOKEN),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://control",
    ) as client:
        response = await client.post(
            "/v1/jobs",
            headers=client_headers(),
            json={"capability": "metadata.normalize"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
    assert "must-not-leak" not in response.text
