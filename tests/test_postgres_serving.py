from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta

import pytest

psycopg = pytest.importorskip("psycopg")

from astrumweaver import ResourceShape, WorkerSpec
from astrumweaver.control import (
    ConflictError,
    JobStatus,
    NoCompatibleDeployment,
    JobSubmission,
    PostgresControlRepository,
    WorkerRegistration,
    WorkerState,
    utc_now,
)
from astrumweaver.control.migrate import apply_migrations
from astrumweaver.execution import JobResult
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


def digest(character: str) -> str:
    return "sha256:" + character * 64


def serving_values(*, epoch: str):
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
        limits={"input_tokens": 4096},
    )
    profile = LogicalServingProfile(
        profile_id="local-code-v1",
        deployment_revision=deployment.revision,
        serving_contract_revision=contract.revision,
        capability="llm.chat",
        operation_schema="chat-v1",
        required_features=frozenset({"tools"}),
        limits={"input_tokens": 4096},
    )
    binding = ServingJobBinding.from_resolved(
        resolve_profile(profile, contract, deployment)
    )
    advertisement = WorkerServingAdvertisement(
        deployment_revision=deployment.revision,
        runtime_instance=RuntimeInstance(deployment.revision, epoch),
        contracts=(contract,),
    )
    return binding, advertisement


def worker(worker_id: str, *, epoch: str) -> WorkerRegistration:
    _, advertisement = serving_values(epoch=epoch)
    return WorkerRegistration(
        spec=WorkerSpec(
            worker_id=worker_id,
            worker_class="cpu-test",
            resources=ResourceShape(),
            capabilities=frozenset({"llm.chat"}),
        ),
        serving=advertisement,
    )


@pytest.fixture(autouse=True)
def reset_database():
    assert DATABASE_URL is not None
    apply_migrations(DATABASE_URL)
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute("TRUNCATE TABLE jobs, workers RESTART IDENTITY CASCADE")


def submission(now, binding=None, *, key=None):
    if binding is None:
        binding, _ = serving_values(
            epoch="12345678-1234-4234-9234-123456789abc"
        )
    return JobSubmission(
        capability="llm.chat",
        serving=binding,
        deadline_at=now + timedelta(seconds=30),
        max_attempts=2,
        idempotency_key=key,
    )


def serving_epoch(repo, worker_id: str) -> str:
    serving = repo.get_worker(worker_id).serving
    assert serving is not None
    return serving.runtime_instance.epoch


def claim_serving(repo, worker_id: str, *, now):
    return repo.claim_next_job(
        worker_id,
        runtime_instance_epoch=serving_epoch(repo, worker_id),
        now=now,
    )


def test_postgres_serving_claim_persists_exact_attempt_identity():
    assert DATABASE_URL is not None
    now = utc_now()
    repo = PostgresControlRepository(DATABASE_URL)
    registration = worker(
        "worker", epoch="12345678-1234-4234-9234-123456789abc"
    )
    repo.register_worker(registration, now=now)
    job = repo.submit_job(submission(now), now=now)

    claim = claim_serving(repo, "worker", now=now)
    assert claim is not None
    assert claim.job_id == job.job_id
    assert claim.serving == job.serving
    assert (
        claim.claimed_deployment_revision
        == registration.serving.deployment_revision
    )
    assert (
        claim.claimed_serving_contract_revision
        == job.serving.serving_contract_revision
    )
    assert (
        claim.claimed_runtime_instance_epoch
        == registration.serving.runtime_instance.epoch
    )

    with psycopg.connect(DATABASE_URL) as connection:
        row = connection.execute(
            """
            SELECT serving, deadline_at, claimed_deployment_revision,
                   claimed_serving_contract_revision,
                   claimed_runtime_instance_epoch
            FROM jobs WHERE id::text = %s
            """,
            (job.job_id,),
        ).fetchone()
    assert row[0]["profile_revision"] == job.serving.profile_revision
    assert row[1] == job.deadline_at
    assert row[2] == claim.claimed_deployment_revision
    assert row[3] == claim.claimed_serving_contract_revision
    assert row[4] == claim.claimed_runtime_instance_epoch


def test_postgres_wrong_deployment_cannot_block_matching_worker():
    assert DATABASE_URL is not None
    now = utc_now()
    repo = PostgresControlRepository(DATABASE_URL)
    binding, _ = serving_values(
        epoch="12345678-1234-4234-9234-123456789abc"
    )
    wrong = worker(
        "wrong", epoch="22345678-1234-4234-9234-123456789abc"
    )
    wrong_deployment = replace(
        wrong.serving,
        deployment_revision=digest("a"),
        runtime_instance=RuntimeInstance(
            digest("a"), "22345678-1234-4234-9234-123456789abc"
        ),
        contracts=(
            replace(
                wrong.serving.contracts[0],
                deployment_revision=digest("a"),
            ),
        ),
    )
    repo.register_worker(replace(wrong, serving=wrong_deployment), now=now)
    right = worker(
        "right", epoch="32345678-1234-4234-9234-123456789abc"
    )
    repo.register_worker(right, now=now)
    job = repo.submit_job(submission(now, binding), now=now)

    assert claim_serving(repo, "wrong", now=now) is None
    assert claim_serving(repo, "right", now=now).job_id == job.job_id


def test_postgres_restart_epoch_fences_old_attempt_and_retry_uses_new_epoch():
    assert DATABASE_URL is not None
    now = utc_now()
    repo = PostgresControlRepository(DATABASE_URL, lease_seconds=1)
    first = worker(
        "worker", epoch="12345678-1234-4234-9234-123456789abc"
    )
    repo.register_worker(first, now=now)
    job = repo.submit_job(submission(now), now=now)
    claim1 = claim_serving(repo, "worker", now=now)
    assert claim1 is not None

    retry_at = now + timedelta(seconds=2)
    recovered = repo.recover_expired_jobs(now=retry_at)
    assert recovered[0].claimed_runtime_instance_epoch is None

    second = worker(
        "worker", epoch="42345678-1234-4234-9234-123456789abc"
    )
    repo.register_worker(second, now=retry_at)
    claim2 = claim_serving(repo, "worker", now=retry_at)
    assert claim2 is not None
    assert claim2.claimed_runtime_instance_epoch == second.serving.runtime_instance.epoch

    with pytest.raises(ConflictError, match="runtime instance is stale"):
        repo.complete_job(
            job.job_id,
            JobResult(text="old"),
            worker_id="worker",
            lease_token=claim2.lease_token,
            runtime_instance_epoch=first.serving.runtime_instance.epoch,
            now=retry_at,
        )
    assert repo.complete_job(
        job.job_id,
        JobResult(text="new"),
        worker_id="worker",
        lease_token=claim2.lease_token,
        runtime_instance_epoch=second.serving.runtime_instance.epoch,
        now=retry_at,
    ).status is JobStatus.SUCCEEDED


def test_postgres_deadline_expiry_releases_running_capacity():
    assert DATABASE_URL is not None
    now = utc_now()
    repo = PostgresControlRepository(DATABASE_URL)
    registration = worker(
        "worker", epoch="12345678-1234-4234-9234-123456789abc"
    )
    repo.register_worker(registration, now=now)
    job = repo.submit_job(
        replace(submission(now), deadline_at=now + timedelta(seconds=1)),
        now=now,
    )
    assert claim_serving(repo, "worker", now=now) is not None

    expired = repo.expire_deadline_jobs(now=now + timedelta(seconds=2))
    assert [item.job_id for item in expired] == [job.job_id]
    assert expired[0].status is JobStatus.FAILED
    assert expired[0].error["type"] == "deadline_expired"
    assert repo.get_worker("worker").active_jobs == 0


def test_postgres_competing_matching_workers_claim_serving_job_once():
    assert DATABASE_URL is not None
    now = utc_now()
    repo = PostgresControlRepository(DATABASE_URL)
    repo.register_worker(
        worker("a", epoch="12345678-1234-4234-9234-123456789abc"),
        now=now,
    )
    repo.register_worker(
        worker("b", epoch="22345678-1234-4234-9234-123456789abc"),
        now=now,
    )
    job = repo.submit_job(submission(now), now=now)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda worker_id: claim_serving(repo, worker_id, now=now), ["a", "b"])
        )

    claims = [item for item in results if item is not None]
    assert len(claims) == 1
    assert claims[0].job_id == job.job_id
    assert repo.get_job(job.job_id).status is JobStatus.RUNNING



def test_postgres_legacy_job_keeps_sql_null_serving_metadata():
    assert DATABASE_URL is not None
    now = utc_now()
    repo = PostgresControlRepository(DATABASE_URL)
    legacy = JobSubmission(capability="llm.chat")
    job = repo.submit_job(legacy, now=now)

    with psycopg.connect(DATABASE_URL) as connection:
        row = connection.execute(
            "SELECT serving, deadline_at FROM jobs WHERE id::text = %s",
            (job.job_id,),
        ).fetchone()

    assert row[0] is None
    assert row[1] is None


def test_postgres_serving_admission_excludes_draining_and_stale_workers():
    assert DATABASE_URL is not None
    now = utc_now()
    repo = PostgresControlRepository(DATABASE_URL, worker_ttl_seconds=60)
    registration = worker(
        "worker", epoch="12345678-1234-4234-9234-123456789abc"
    )
    repo.register_worker(registration, now=now)
    repo.set_worker_state(
        "worker",
        WorkerState.DRAINING,
        runtime_instance_epoch=registration.serving.runtime_instance.epoch,
        now=now,
    )

    with pytest.raises(NoCompatibleDeployment):
        repo.submit_job(submission(now), now=now)

    repo.register_worker(
        registration,
        now=now - timedelta(seconds=61),
    )
    with pytest.raises(NoCompatibleDeployment):
        repo.submit_job(submission(now), now=now)


def test_postgres_concurrent_idempotency_preserves_serving_intent():
    assert DATABASE_URL is not None
    now = utc_now()
    setup = PostgresControlRepository(DATABASE_URL)
    setup.register_worker(
        worker("worker", epoch="12345678-1234-4234-9234-123456789abc"),
        now=now,
    )
    request = submission(now, key="same-serving-request")
    repo_a = PostgresControlRepository(DATABASE_URL)
    repo_b = PostgresControlRepository(DATABASE_URL)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(repo_a.submit_job, request, now=now)
        second = pool.submit(repo_b.submit_job, request, now=now)
        records = [first.result(timeout=10), second.result(timeout=10)]

    assert records[0].job_id == records[1].job_id
    assert records[0].serving == request.serving
    assert records[0].deadline_at == request.deadline_at

    with pytest.raises(ConflictError, match="idempotency"):
        setup.submit_job(
            replace(request, deadline_at=request.deadline_at + timedelta(seconds=1)),
            now=now,
        )


def test_postgres_deadline_cancel_recovery_race_has_one_terminal_owner():
    assert DATABASE_URL is not None
    now = utc_now()
    setup = PostgresControlRepository(DATABASE_URL, lease_seconds=1)
    registration = worker(
        "worker", epoch="12345678-1234-4234-9234-123456789abc"
    )
    setup.register_worker(registration, now=now)
    job = setup.submit_job(
        replace(submission(now), deadline_at=now + timedelta(seconds=1)),
        now=now,
    )
    claimed = claim_serving(setup, "worker", now=now)
    assert claimed is not None and claimed.lease_token

    later = now + timedelta(seconds=2)
    cancel_repo = PostgresControlRepository(DATABASE_URL, lease_seconds=1)
    recover_repo = PostgresControlRepository(DATABASE_URL, lease_seconds=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        cancel_future = pool.submit(cancel_repo.cancel_job, job.job_id, now=later)
        recover_future = pool.submit(recover_repo.recover_expired_jobs, now=later)
        cancelled_view = cancel_future.result(timeout=10)
        recovered_view = recover_future.result(timeout=10)

    final = setup.get_job(job.job_id)
    assert final.status in {JobStatus.CANCELLED, JobStatus.FAILED}
    assert setup.get_worker("worker").active_jobs == 0
    assert cancelled_view.status in {JobStatus.CANCELLED, JobStatus.FAILED}
    assert all(item.status is JobStatus.FAILED for item in recovered_view)

    with pytest.raises(ConflictError):
        setup.complete_job(
            job.job_id,
            JobResult(text="stale"),
            worker_id="worker",
            lease_token=claimed.lease_token,
            runtime_instance_epoch=registration.serving.runtime_instance.epoch,
            now=later,
        )


def test_postgres_restart_fences_stale_serving_process_before_claim():
    assert DATABASE_URL is not None
    now = utc_now()
    repo = PostgresControlRepository(DATABASE_URL)
    first = worker(
        "worker", epoch="12345678-1234-4234-9234-123456789abc"
    )
    repo.register_worker(first, now=now)
    second = worker(
        "worker", epoch="82345678-1234-4234-9234-123456789abc"
    )
    repo.register_worker(second, now=now)

    with pytest.raises(ConflictError, match="runtime instance is stale"):
        repo.claim_next_job(
            "worker",
            runtime_instance_epoch=first.serving.runtime_instance.epoch,
            now=now,
        )

    assert (
        repo.claim_next_job(
            "worker",
            runtime_instance_epoch=second.serving.runtime_instance.epoch,
            now=now,
        )
        is None
    )
