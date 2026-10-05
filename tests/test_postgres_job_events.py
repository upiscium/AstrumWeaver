from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest

psycopg = pytest.importorskip("psycopg")

from astrumweaver import ResourceShape, WorkerSpec
from astrumweaver.control import (
    ConflictError,
    EventBufferFull,
    JobSubmission,
    PostgresControlRepository,
    WorkerRegistration,
    utc_now,
)
from astrumweaver.control.migrate import apply_migrations
from astrumweaver.serving import (
    DeploymentIdentity,
    LogicalServingProfile,
    RuntimeInstance,
    ServingContract,
    ServingJobBinding,
    WorkerServingAdvertisement,
    resolve_profile,
)


DATABASE_URL = os.environ.get("ASTRUMWEAVER_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="ASTRUMWEAVER_TEST_DATABASE_URL is not configured",
)
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


@pytest.fixture(autouse=True)
def reset_database():
    assert DATABASE_URL is not None
    apply_migrations(DATABASE_URL)
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute(
            "TRUNCATE TABLE job_events, jobs, workers RESTART IDENTITY CASCADE"
        )


def setup(repo: PostgresControlRepository):
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


def append(repo, claimed, payload, now):
    return repo.append_job_event(
        claimed.job_id,
        worker_id="worker-a",
        lease_token=claimed.lease_token,
        runtime_instance_epoch=EPOCH,
        kind="chat.completion.chunk",
        payload=payload,
        now=now,
    )


def test_postgres_concurrent_event_appends_are_serialized():
    assert DATABASE_URL is not None
    setup_repo = PostgresControlRepository(DATABASE_URL)
    now, _, claimed = setup(setup_repo)
    repo_a = PostgresControlRepository(DATABASE_URL)
    repo_b = PostgresControlRepository(DATABASE_URL)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(append, repo_a, claimed, {"delta": "a"}, now)
        b = pool.submit(append, repo_b, claimed, {"delta": "b"}, now)
        events = [a.result(timeout=10), b.result(timeout=10)]

    assert {event.sequence for event in events} == {1, 2}
    listed = setup_repo.list_job_events(claimed.job_id)
    assert [event.sequence for event in listed] == [1, 2]
    assert all(event.attempt == claimed.attempts for event in listed)
    assert all(event.runtime_instance_epoch == EPOCH for event in listed)

    with psycopg.connect(DATABASE_URL) as connection:
        rows = connection.execute(
            """
            SELECT sequence, attempt, runtime_instance_epoch
            FROM job_events
            WHERE job_id::text = %s
            ORDER BY sequence
            """,
            (claimed.job_id,),
        ).fetchall()
    assert rows == [(1, 1, EPOCH), (2, 1, EPOCH)]


def test_postgres_event_append_rejects_stale_and_cancelled_attempt():
    assert DATABASE_URL is not None
    repo = PostgresControlRepository(DATABASE_URL)
    now, _, claimed = setup(repo)

    with pytest.raises(ConflictError):
        repo.append_job_event(
            claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token,
            runtime_instance_epoch="22345678-1234-4234-9234-123456789abc",
            kind="chat.completion.chunk",
            payload={"delta": "stale"},
            now=now,
        )

    repo.cancel_job(claimed.job_id, now=now)
    with pytest.raises(ConflictError, match="not running"):
        append(repo, claimed, {"delta": "late"}, now)


def test_postgres_event_buffer_limit_is_authoritative():
    assert DATABASE_URL is not None
    repo = PostgresControlRepository(DATABASE_URL, event_max_count=1)
    now, _, claimed = setup(repo)
    append(repo, claimed, {"delta": "first"}, now)
    with pytest.raises(EventBufferFull, match="count"):
        append(repo, claimed, {"delta": "second"}, now)
