from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import httpx
import pytest

from astrumweaver import ResourceShape, WorkerSpec
from astrumweaver.control import (
    ConflictError,
    EventBufferFull,
    InMemoryControlRepository,
    JobSubmission,
    WorkerRegistration,
    utc_now,
)
from astrumweaver.control.api import create_app
from astrumweaver.serving import (
    DeploymentIdentity,
    LogicalServingProfile,
    RuntimeInstance,
    ServingContract,
    ServingJobBinding,
    WorkerServingAdvertisement,
    resolve_profile,
)
from astrumweaver.transport import (
    JOB_EVENTS_EXTENSION,
    PROTOCOL_VERSION,
    SERVING_EXTENSION,
)


CLIENT_TOKEN = "client-secret"
WORKER_TOKEN = "worker-secret"
EPOCH = "12345678-1234-4234-9234-123456789abc"


def digest(character: str) -> str:
    return "sha256:" + character * 64


def values():
    deployment = DeploymentIdentity(
        provider_id="llama-cpp",
        runtime_artifact_sha256=digest("1"),
        adapter_artifact_sha256=digest("2"),
        model_artifact_sha256=digest("3"),
        execution_config_sha256=digest("4"),
        quantization="Q6_K",
    )
    contract = ServingContract(
        deployment_revision=deployment.revision,
        capability="llm.chat",
        operation_schema="chat-completions-v1",
        validation_evidence_sha256=digest("7"),
        features=frozenset(),
        limits={"input_tokens": 128, "output_tokens": 32},
    )
    profile = LogicalServingProfile(
        profile_id="stream-test-v1",
        deployment_revision=deployment.revision,
        serving_contract_revision=contract.revision,
        capability=contract.capability,
        operation_schema=contract.operation_schema,
    )
    binding = ServingJobBinding.from_resolved(
        resolve_profile(profile, contract, deployment)
    )
    serving = WorkerServingAdvertisement(
        deployment_revision=deployment.revision,
        runtime_instance=RuntimeInstance(deployment.revision, EPOCH),
        contracts=(contract,),
    )
    return binding, serving


def setup(repo: InMemoryControlRepository):
    now = utc_now()
    binding, serving = values()
    repo.register_worker(
        WorkerRegistration(
            spec=WorkerSpec(
                worker_id="worker-a",
                worker_class="cpu-test",
                resources=ResourceShape(),
                capabilities=frozenset({"llm.chat"}),
            ),
            serving=serving,
        ),
        now=now,
    )
    job = repo.submit_job(
        JobSubmission(
            capability="llm.chat",
            payload={"stream": True},
            max_attempts=1,
            serving=binding,
            deadline_at=now + timedelta(seconds=30),
        ),
        now=now,
    )
    claimed = repo.claim_next_job(
        "worker-a",
        runtime_instance_epoch=EPOCH,
        now=now,
    )
    assert claimed is not None and claimed.lease_token
    return now, job, claimed


def append(repo, claimed, *, kind="chat.completion.chunk", payload=None, now=None):
    return repo.append_job_event(
        claimed.job_id,
        worker_id="worker-a",
        lease_token=claimed.lease_token,
        runtime_instance_epoch=EPOCH,
        kind=kind,
        payload=payload or {"delta": "x"},
        now=now,
    )


def test_reference_job_events_are_ordered_and_paged():
    repo = InMemoryControlRepository(event_max_read=2)
    now, _, claimed = setup(repo)

    first = append(repo, claimed, payload={"delta": "a"}, now=now)
    second = append(repo, claimed, payload={"delta": "b"}, now=now)
    third = append(repo, claimed, payload={"delta": "c"}, now=now)

    assert [first.sequence, second.sequence, third.sequence] == [1, 2, 3]
    assert all(item.attempt == claimed.attempts for item in (first, second, third))
    assert all(item.runtime_instance_epoch == EPOCH for item in (first, second, third))
    assert [item.sequence for item in repo.list_job_events(claimed.job_id, limit=2)] == [1, 2]
    assert [
        item.sequence
        for item in repo.list_job_events(
            claimed.job_id,
            after_sequence=1,
            limit=2,
        )
    ] == [2, 3]


def test_reference_job_event_publication_is_attempt_fenced():
    repo = InMemoryControlRepository()
    now, _, claimed = setup(repo)

    with pytest.raises(ConflictError):
        repo.append_job_event(
            claimed.job_id,
            worker_id="worker-a",
            lease_token="stale",
            runtime_instance_epoch=EPOCH,
            kind="chat.completion.chunk",
            payload={"delta": "x"},
            now=now,
        )
    with pytest.raises(ConflictError, match="runtime instance"):
        repo.append_job_event(
            claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token,
            runtime_instance_epoch="22345678-1234-4234-9234-123456789abc",
            kind="chat.completion.chunk",
            payload={"delta": "x"},
            now=now,
        )

    repo.cancel_job(claimed.job_id, now=now)
    with pytest.raises(ConflictError, match="not running"):
        append(repo, claimed, now=now)


def test_reference_job_event_buffer_limits_fail_closed():
    count_repo = InMemoryControlRepository(event_max_count=1)
    now, _, claimed = setup(count_repo)
    append(count_repo, claimed, now=now)
    with pytest.raises(EventBufferFull, match="count"):
        append(count_repo, claimed, now=now)

    payload_repo = InMemoryControlRepository(event_max_payload_bytes=64)
    now2, _, claimed2 = setup(payload_repo)
    with pytest.raises(EventBufferFull, match="payload"):
        append(
            payload_repo,
            claimed2,
            payload={"delta": "x" * 256},
            now=now2,
        )


@pytest.mark.asyncio
async def test_worker_event_endpoint_requires_extension_and_fences_attempt():
    repo = InMemoryControlRepository()
    now, _, claimed = setup(repo)
    app = create_app(
        repo,
        client_token=CLIENT_TOKEN,
        worker_token=WORKER_TOKEN,
        maintenance_interval_seconds=60,
    )
    transport = httpx.ASGITransport(app=app)
    headers = {"authorization": f"Bearer {WORKER_TOKEN}"}

    body = {
        "protocol_version": PROTOCOL_VERSION,
        "extensions": [JOB_EVENTS_EXTENSION, SERVING_EXTENSION],
        "lease_token": claimed.lease_token,
        "runtime_instance_epoch": EPOCH,
        "kind": "chat.completion.chunk",
        "payload": {"choices": [{"delta": {"content": "hi"}}]},
    }

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://control",
    ) as client:
        missing = await client.post(
            f"/v1/workers/worker-a/jobs/{claimed.job_id}/events",
            headers=headers,
            json={**body, "extensions": [SERVING_EXTENSION]},
        )
        accepted = await client.post(
            f"/v1/workers/worker-a/jobs/{claimed.job_id}/events",
            headers=headers,
            json=body,
        )
        stale = await client.post(
            f"/v1/workers/worker-a/jobs/{claimed.job_id}/events",
            headers=headers,
            json={**body, "lease_token": "stale"},
        )

    assert missing.status_code == 422
    assert accepted.status_code == 200
    assert accepted.json()["sequence"] == 1
    assert accepted.json()["attempt"] == claimed.attempts
    assert accepted.json()["runtime_instance_epoch"] == EPOCH
    assert set(accepted.json()["extensions"]) == {
        JOB_EVENTS_EXTENSION,
        SERVING_EXTENSION,
    }
    assert stale.status_code == 409
