from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
import os
from uuid import uuid4

import pytest

psycopg = pytest.importorskip("psycopg")

from astrumweaver import JobResult, ResourceShape, WorkerSpec
from astrumweaver.control import JobStatus, JobSubmission, WorkerRegistration, utc_now
from astrumweaver.control.migrate import apply_migrations
from astrumweaver.control.postgres import PostgresControlRepository
from astrumweaver.control.repository import ConflictError, NoCompatibleDeployment
from astrumweaver.serving import (
    DeploymentIdentity,
    LogicalServingProfile,
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


@pytest.fixture(autouse=True)
def reset_database() -> None:
    assert DATABASE_URL is not None
    apply_migrations(DATABASE_URL)
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute("TRUNCATE TABLE jobs, workers RESTART IDENTITY CASCADE")


def digest(ch: str) -> str:
    return "sha256:" + ch * 64


def serving_values(*, epoch: str | None = None):
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
        operation_schema="chat-v1",
        validation_evidence_sha256=digest("7"),
        features=frozenset({"tools"}),
        limits={"input_tokens": 8192, "output_tokens": 2048},
    )
    profile = LogicalServingProfile(
        profile_id="local-code-v1",
        deployment_revision=deployment.revision,
        serving_contract_revision=contract.revision,
        capability=contract.capability,
        operation_schema=contract.operation_schema,
        required_features=frozenset({"tools"}),
        limits={"input_tokens": 4096},
    )
    binding = ServingJobBinding.from_resolved(
        resolve_profile(profile, contract, deployment)
    )
    serving = WorkerServingAdvertisement(
        deployment_revision=deployment.revision,
        runtime_instance_epoch=epoch or str(uuid4()),
        contract_revisions={"llm.chat": contract.revision},
    )
    return binding, serving


def worker(worker_id: str, serving: WorkerServingAdvertisement):
    return WorkerRegistration(
        spec=WorkerSpec(
            worker_id=worker_id,
            worker_class="cpu-test",
            resources=ResourceShape(),
            capabilities=frozenset({"llm.chat"}),
        ),
        serving=serving,
    )


def submission(binding: ServingJobBinding, now, *, key: str | None = None):
    return JobSubmission(
        capability="llm.chat",
        payload={"input": "bounded"},
        serving_binding=binding,
        deadline_at=now + timedelta(seconds=10),
        idempotency_key=key,
    )


def test_postgres_persists_serving_binding_and_fences_runtime_epoch():
    assert DATABASE_URL is not None
    now = utc_now()
    repo = PostgresControlRepository(DATABASE_URL, lease_seconds=30)
    binding, serving = serving_values()

    with pytest.raises(NoCompatibleDeployment):
        repo.submit_job(submission(binding, now), now=now)

    registered = repo.register_worker(worker("worker-a", serving), now=now)
    job = repo.submit_job(submission(binding, now), now=now)
    claimed = repo.claim_next_job(
        registered.worker_id,
        runtime_instance_epoch=serving.runtime_instance_epoch,
        now=now,
    )
    assert claimed is not None
    assert claimed.serving_binding == binding
    assert claimed.attempt_runtime_instance_epoch == serving.runtime_instance_epoch

    with pytest.raises(ConflictError, match="runtime instance epoch"):
        repo.complete_job(
            job.job_id,
            JobResult(outputs={"ok": False}),
            worker_id=registered.worker_id,
            lease_token=claimed.lease_token or "",
            runtime_instance_epoch=str(uuid4()),
            now=now,
        )

    completed = repo.complete_job(
        job.job_id,
        JobResult(outputs={"ok": True}),
        worker_id=registered.worker_id,
        lease_token=claimed.lease_token or "",
        runtime_instance_epoch=serving.runtime_instance_epoch,
        now=now,
    )
    assert completed.status is JobStatus.SUCCEEDED
    assert completed.serving_binding == binding
    assert completed.attempt_runtime_instance_epoch is None


def test_postgres_deadline_caps_lease_and_disables_retry():
    assert DATABASE_URL is not None
    now = utc_now()
    repo = PostgresControlRepository(DATABASE_URL, lease_seconds=300)
    binding, serving = serving_values()
    repo.register_worker(worker("worker-a", serving), now=now)

    deadline = now + timedelta(seconds=1)
    first = repo.submit_job(
        replace(submission(binding, now), deadline_at=deadline),
        now=now,
    )
    second = repo.submit_job(
        replace(submission(binding, now), deadline_at=deadline),
        now=now,
    )
    claimed = repo.claim_next_job(
        "worker-a",
        runtime_instance_epoch=serving.runtime_instance_epoch,
        now=now,
    )
    assert claimed is not None
    assert claimed.job_id == first.job_id
    assert claimed.lease_expires_at == deadline

    later = now + timedelta(seconds=2)
    recovered = repo.recover_expired_jobs(now=later)
    assert [item.job_id for item in recovered] == [first.job_id]
    assert recovered[0].status is JobStatus.FAILED
    assert recovered[0].error["code"] == "deadline-exceeded"
    assert recovered[0].error["retryable"] is False

    expired = repo.expire_deadline_jobs(now=later)
    assert [item.job_id for item in expired] == [second.job_id]
    assert repo.get_job(second.job_id).status is JobStatus.FAILED


def test_postgres_concurrent_compatible_workers_claim_only_one_attempt():
    assert DATABASE_URL is not None
    now = utc_now()
    binding, serving_a = serving_values()
    serving_b = replace(serving_a, runtime_instance_epoch=str(uuid4()))
    setup = PostgresControlRepository(DATABASE_URL)
    setup.register_worker(worker("worker-a", serving_a), now=now)
    setup.register_worker(worker("worker-b", serving_b), now=now)
    job = setup.submit_job(submission(binding, now), now=now)

    repo_a = PostgresControlRepository(DATABASE_URL)
    repo_b = PostgresControlRepository(DATABASE_URL)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(
            repo_a.claim_next_job,
            "worker-a",
            runtime_instance_epoch=serving_a.runtime_instance_epoch,
            now=now,
        )
        future_b = pool.submit(
            repo_b.claim_next_job,
            "worker-b",
            runtime_instance_epoch=serving_b.runtime_instance_epoch,
            now=now,
        )
        claims = [future_a.result(timeout=10), future_b.result(timeout=10)]

    claimed = [item for item in claims if item is not None]
    assert len(claimed) == 1
    assert claimed[0].job_id == job.job_id
    expected_epoch = (
        serving_a.runtime_instance_epoch
        if claimed[0].assigned_worker_id == "worker-a"
        else serving_b.runtime_instance_epoch
    )
    assert claimed[0].attempt_runtime_instance_epoch == expected_epoch


def test_postgres_concurrent_idempotency_preserves_serving_intent():
    assert DATABASE_URL is not None
    now = utc_now()
    binding, serving = serving_values()
    setup = PostgresControlRepository(DATABASE_URL)
    setup.register_worker(worker("worker-a", serving), now=now)
    request = submission(binding, now, key="same-serving-request")

    repo_a = PostgresControlRepository(DATABASE_URL)
    repo_b = PostgresControlRepository(DATABASE_URL)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(repo_a.submit_job, request, now=now)
        second = pool.submit(repo_b.submit_job, request, now=now)
        records = [first.result(timeout=10), second.result(timeout=10)]

    assert records[0].job_id == records[1].job_id
    assert records[0].serving_binding == binding
    assert records[0].deadline_at == request.deadline_at

    with pytest.raises(ConflictError, match="idempotency"):
        setup.submit_job(
            replace(request, deadline_at=request.deadline_at + timedelta(seconds=1)),
            now=now,
        )
