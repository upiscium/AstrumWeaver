from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

psycopg = pytest.importorskip("psycopg")

from astrumweaver import JobRequirements, JobResult, ResourceShape, WorkerSpec
from astrumweaver.control.api import create_app
from astrumweaver.transport import PROTOCOL_VERSION

from astrumweaver.control import (
    ConflictError,
    JobStatus,
    JobSubmission,
    PostgresControlRepository,
    WorkerRegistration,
    utc_now,
)


DATABASE_URL = os.environ.get("ASTRUMWEAVER_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="ASTRUMWEAVER_TEST_DATABASE_URL is not configured",
)


@pytest.fixture(autouse=True)
def reset_database() -> None:
    assert DATABASE_URL is not None
    migration = (
        Path(__file__).resolve().parents[1]
        / "migrations"
        / "001_control_plane.sql"
    ).read_text(encoding="utf-8")
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute(migration)
        connection.execute("TRUNCATE TABLE jobs, workers RESTART IDENTITY CASCADE")


def worker(
    worker_id: str,
    *,
    single_vram_mb: int,
    gpu_count: int = 1,
    total_vram_mb: int | None = None,
) -> WorkerRegistration:
    if total_vram_mb is None:
        total_vram_mb = single_vram_mb * gpu_count
    return WorkerRegistration(
        spec=WorkerSpec(
            worker_id=worker_id,
            worker_class="multi-gpu" if gpu_count > 1 else "modern-single",
            resources=ResourceShape(
                gpu_count=gpu_count,
                total_vram_mb=total_vram_mb,
                max_single_gpu_vram_mb=single_vram_mb,
            ),
            gpu_uuids=tuple(f"GPU-{worker_id}-{index}" for index in range(gpu_count)),
            capabilities=frozenset({"llm.chat", "code.review"}),
            labels={"runtime_family": "modern"},
        )
    )


def test_postgres_priority_fifo_and_resource_matching() -> None:
    assert DATABASE_URL is not None
    now = utc_now()
    repo = PostgresControlRepository(DATABASE_URL)
    registered = repo.register_worker(
        worker(
            "multi-worker",
            single_vram_mb=12_288,
            gpu_count=2,
            total_vram_mb=24_576,
        ),
        now=now,
    )

    blocked = repo.submit_job(
        JobSubmission(
            capability="llm.chat",
            priority=100,
            payload={"name": "needs-one-large-gpu"},
            requirements=JobRequirements(min_single_gpu_vram_mb=20_000),
        ),
        now=now,
    )
    first = repo.submit_job(
        JobSubmission(
            capability="llm.chat",
            priority=10,
            payload={"name": "first"},
            requirements=JobRequirements(
                min_gpu_count=2,
                min_total_vram_mb=20_000,
            ),
        ),
        now=now,
    )
    second = repo.submit_job(
        JobSubmission(
            capability="llm.chat",
            priority=10,
            payload={"name": "second"},
            requirements=JobRequirements(
                min_gpu_count=2,
                min_total_vram_mb=20_000,
            ),
        ),
        now=now,
    )

    claimed = repo.claim_next_job(registered.worker_id, now=now)

    assert claimed is not None
    assert claimed.job_id == first.job_id
    assert first.sequence < second.sequence
    assert repo.get_job(blocked.job_id).status is JobStatus.QUEUED


def test_postgres_lease_recovery_fences_stale_attempt() -> None:
    assert DATABASE_URL is not None
    now = utc_now()
    repo = PostgresControlRepository(DATABASE_URL, lease_seconds=1)
    registered = repo.register_worker(
        worker("single-worker", single_vram_mb=24_576),
        now=now,
    )
    job = repo.submit_job(JobSubmission(capability="llm.chat"), now=now)
    first = repo.claim_next_job(registered.worker_id, now=now)
    assert first is not None and first.lease_token

    expired = now + timedelta(seconds=2)
    with pytest.raises(ConflictError, match="lease has expired"):
        repo.complete_job(
            job.job_id,
            JobResult(outputs={"stale": True}),
            worker_id=registered.worker_id,
            lease_token=first.lease_token,
            now=expired,
        )

    recovered = repo.recover_expired_jobs(now=expired)
    assert [item.job_id for item in recovered] == [job.job_id]

    second = repo.claim_next_job(registered.worker_id, now=expired)
    assert second is not None and second.lease_token != first.lease_token

    with pytest.raises(ConflictError):
        repo.complete_job(
            job.job_id,
            JobResult(outputs={"stale": True}),
            worker_id=registered.worker_id,
            lease_token=first.lease_token,
            now=expired,
        )


def test_postgres_idempotency_is_durable() -> None:
    assert DATABASE_URL is not None
    repo = PostgresControlRepository(DATABASE_URL)
    submission = JobSubmission(
        capability="code.review",
        payload={"repository": "example/repository"},
        idempotency_key="durable-request-1",
    )

    first = repo.submit_job(submission)
    replay = repo.submit_job(submission)

    assert replay.job_id == first.job_id

    with pytest.raises(ConflictError, match="different job request"):
        repo.submit_job(
            JobSubmission(
                capability="code.review",
                payload={"repository": "different/repository"},
                idempotency_key="durable-request-1",
            )
        )


@pytest.mark.asyncio
async def test_postgres_transport_fencing_and_non_text_result() -> None:
    assert DATABASE_URL is not None
    repo = PostgresControlRepository(DATABASE_URL, lease_seconds=1)
    app = create_app(
        repo,
        client_token="client-secret",
        worker_token="worker-secret",
        maintenance_interval_seconds=60.0,
    )
    transport = httpx.ASGITransport(app=app)
    client_headers = {"authorization": "Bearer client-secret"}
    worker_headers = {"authorization": "Bearer worker-secret"}

    registration = {
        "protocol_version": PROTOCOL_VERSION,
        "spec": {
            "worker_id": "transport-worker",
            "worker_class": "cpu-test",
            "gpu_uuids": [],
            "capabilities": ["decision.system_one"],
            "labels": {},
            "resources": {
                "gpu_count": 0,
                "total_vram_mb": 0,
                "max_single_gpu_vram_mb": 0,
            },
        },
        "max_concurrency": 1,
        "metadata": {},
    }
    submission = {
        "protocol_version": PROTOCOL_VERSION,
        "capability": "decision.system_one",
        "payload": {"question": "route"},
        "requirements": {},
        "priority": 0,
        "max_attempts": 3,
    }

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://control",
    ) as client:
        registered = await client.post(
            "/v1/workers/register",
            headers=worker_headers,
            json=registration,
        )
        assert registered.status_code == 201

        created = await client.post(
            "/v1/jobs",
            headers=client_headers,
            json=submission,
        )
        assert created.status_code == 201
        job_id = created.json()["job_id"]

        first = await client.post(
            "/v1/workers/transport-worker/jobs/claim",
            headers=worker_headers,
        )
        assert first.status_code == 200
        old_token = first.json()["lease_token"]

        await asyncio.sleep(1.05)
        expired = await client.post(
            f"/v1/workers/transport-worker/jobs/{job_id}/complete",
            headers=worker_headers,
            json={
                "protocol_version": PROTOCOL_VERSION,
                "lease_token": old_token,
                "result": {"outputs": {"stale": True}},
            },
        )
        assert expired.status_code == 409

        repo.recover_expired_jobs()

        second = await client.post(
            "/v1/workers/transport-worker/jobs/claim",
            headers=worker_headers,
        )
        assert second.status_code == 200
        new_token = second.json()["lease_token"]
        assert new_token != old_token

        stale = await client.post(
            f"/v1/workers/transport-worker/jobs/{job_id}/complete",
            headers=worker_headers,
            json={
                "protocol_version": PROTOCOL_VERSION,
                "lease_token": old_token,
                "result": {"outputs": {"stale": True}},
            },
        )
        assert stale.status_code == 409

        completed = await client.post(
            f"/v1/workers/transport-worker/jobs/{job_id}/complete",
            headers=worker_headers,
            json={
                "protocol_version": PROTOCOL_VERSION,
                "lease_token": new_token,
                "result": {
                    "outputs": {
                        "choice": "local",
                        "probabilities": {"local": 0.8, "remote": 0.2},
                    },
                    "artifacts": [],
                    "metrics": {"entropy": 0.5},
                    "text": None,
                    "metadata": {"executor": "decision-test"},
                },
            },
        )
        assert completed.status_code == 200

        fetched = await client.get(
            f"/v1/jobs/{job_id}",
            headers=client_headers,
        )

    result = fetched.json()["result"]
    assert result["text"] is None
    assert result["outputs"]["choice"] == "local"
    assert result["outputs"]["probabilities"]["remote"] == 0.2
