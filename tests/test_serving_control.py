from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import httpx
import pytest

from astrumweaver import ResourceShape, WorkerSpec
from astrumweaver.control import (
    ConflictError,
    DeadlineExceededError,
    InMemoryControlRepository,
    JobStatus,
    NoCompatibleDeployment,
    OverloadedError,
    JobSubmission,
    WorkerHeartbeat,
    WorkerRegistration,
    WorkerState,
    utc_now,
)
from astrumweaver.control.api import create_app
from astrumweaver.execution import JobRequest, JobResult
from astrumweaver.serving import (
    DeploymentIdentity,
    LogicalServingProfile,
    RuntimeInstance,
    ServingContract,
    ServingJobBinding,
    WorkerServingAdvertisement,
    resolve_profile,
)
from astrumweaver.transport import PROTOCOL_VERSION, SERVING_EXTENSION
from astrumweaver.worker.client import ClaimedJob, ControlClient, ControlTransportError
from astrumweaver.worker.runtime import WorkerRuntime


def digest(character: str) -> str:
    return "sha256:" + character * 64


def serving_values(*, epoch: str = "12345678-1234-4234-9234-123456789abc"):
    deployment = DeploymentIdentity(
        provider_id="llama-cpp",
        runtime_artifact_sha256=digest("1"),
        adapter_artifact_sha256=digest("2"),
        model_artifact_sha256=digest("3"),
        execution_config_sha256=digest("4"),
        quantization="Q6_K",
        tokenizer_artifact_sha256=digest("5"),
        template_artifact_sha256=digest("6"),
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
    advertisement = WorkerServingAdvertisement(
        deployment_revision=deployment.revision,
        runtime_instance=RuntimeInstance(deployment.revision, epoch),
        contracts=(contract,),
    )
    return deployment, contract, profile, binding, advertisement


def serving_worker(
    worker_id: str,
    *,
    epoch: str = "12345678-1234-4234-9234-123456789abc",
) -> WorkerRegistration:
    *_, advertisement = serving_values(epoch=epoch)
    return WorkerRegistration(
        spec=WorkerSpec(
            worker_id=worker_id,
            worker_class="cpu-test",
            resources=ResourceShape(),
            capabilities=frozenset({"llm.chat"}),
        ),
        serving=advertisement,
    )


def serving_submission(
    *,
    now,
    binding: ServingJobBinding | None = None,
    idempotency_key: str | None = None,
):
    if binding is None:
        *_, binding, _ = serving_values()
    return JobSubmission(
        capability="llm.chat",
        payload={"messages": [{"role": "user", "content": "bounded"}]},
        serving=binding,
        deadline_at=now + timedelta(seconds=30),
        idempotency_key=idempotency_key,
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


def set_serving_state(repo, worker_id: str, state: WorkerState, *, now):
    return repo.set_worker_state(
        worker_id,
        state,
        runtime_instance_epoch=serving_epoch(repo, worker_id),
        now=now,
    )


def test_serving_job_requires_a_deadline():
    *_, binding, _ = serving_values()
    with pytest.raises(ValueError, match="require deadline"):
        JobSubmission(capability="llm.chat", serving=binding)


def test_legacy_job_and_worker_remain_compatible():
    repo = InMemoryControlRepository()
    worker = WorkerRegistration(
        spec=WorkerSpec(
            worker_id="legacy",
            worker_class="cpu-test",
            resources=ResourceShape(),
            capabilities=frozenset({"llm.chat"}),
        )
    )
    repo.register_worker(worker)
    job = repo.submit_job(JobSubmission(capability="llm.chat"))
    claim = repo.claim_next_job("legacy")
    assert claim is not None
    assert claim.job_id == job.job_id
    assert claim.serving is None
    assert claim.claimed_runtime_instance_epoch is None


def test_serving_job_claim_is_bound_to_matching_deployment_and_epoch():
    now = utc_now()
    repo = InMemoryControlRepository(lease_seconds=10)
    *_, binding, advertisement = serving_values()
    wrong = serving_worker(
        "wrong",
        epoch="22345678-1234-4234-9234-123456789abc",
    )
    wrong_advertisement = replace(
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
    wrong = replace(wrong, serving=wrong_advertisement)
    repo.register_worker(wrong, now=now)
    repo.register_worker(serving_worker("right"), now=now)
    job = repo.submit_job(serving_submission(now=now, binding=binding), now=now)

    assert claim_serving(repo, "wrong", now=now) is None
    claim = claim_serving(repo, "right", now=now)
    assert claim is not None
    assert claim.job_id == job.job_id
    assert claim.claimed_deployment_revision == advertisement.deployment_revision
    assert claim.claimed_serving_contract_revision == binding.serving_contract_revision
    assert (
        claim.claimed_runtime_instance_epoch
        == advertisement.runtime_instance.epoch
    )


def test_runtime_epoch_fences_heartbeat_and_terminal_write():
    now = utc_now()
    repo = InMemoryControlRepository(lease_seconds=10)
    registration = serving_worker("worker")
    repo.register_worker(registration, now=now)
    job = repo.submit_job(serving_submission(now=now), now=now)
    claim = claim_serving(repo, "worker", now=now)
    assert claim is not None and claim.lease_token

    wrong_epoch = "32345678-1234-4234-9234-123456789abc"
    with pytest.raises(ConflictError, match="runtime instance is stale"):
        repo.heartbeat_worker(
            "worker",
            WorkerHeartbeat(
                active_job_id=job.job_id,
                lease_token=claim.lease_token,
                runtime_instance_epoch=wrong_epoch,
            ),
            now=now,
        )
    with pytest.raises(ConflictError, match="runtime instance is stale"):
        repo.complete_job(
            job.job_id,
            JobResult(text="stale"),
            worker_id="worker",
            lease_token=claim.lease_token,
            runtime_instance_epoch=wrong_epoch,
            now=now,
        )

    completed = repo.complete_job(
        job.job_id,
        JobResult(text="ok"),
        worker_id="worker",
        lease_token=claim.lease_token,
        runtime_instance_epoch=registration.serving.runtime_instance.epoch,
        now=now,
    )
    assert completed.status is JobStatus.SUCCEEDED


def test_retry_after_restart_binds_new_epoch_and_rejects_old_epoch():
    now = utc_now()
    repo = InMemoryControlRepository(lease_seconds=1)
    first = serving_worker("worker")
    repo.register_worker(first, now=now)
    job = repo.submit_job(serving_submission(now=now), now=now)
    claim1 = claim_serving(repo, "worker", now=now)
    assert claim1 is not None

    retry_at = now + timedelta(seconds=2)
    recovered = repo.recover_expired_jobs(now=retry_at)
    assert recovered[0].claimed_runtime_instance_epoch is None

    second = serving_worker(
        "worker", epoch="42345678-1234-4234-9234-123456789abc"
    )
    repo.register_worker(second, now=retry_at)
    claim2 = claim_serving(repo, "worker", now=retry_at)
    assert claim2 is not None
    assert claim2.claimed_runtime_instance_epoch == second.serving.runtime_instance.epoch
    assert claim2.claimed_runtime_instance_epoch != claim1.claimed_runtime_instance_epoch

    with pytest.raises(ConflictError, match="runtime instance is stale"):
        repo.complete_job(
            job.job_id,
            JobResult(text="old"),
            worker_id="worker",
            lease_token=claim2.lease_token,
            runtime_instance_epoch=first.serving.runtime_instance.epoch,
            now=retry_at,
        )


def test_active_serving_worker_cannot_reregister_with_new_epoch():
    now = utc_now()
    repo = InMemoryControlRepository()
    first = serving_worker("worker")
    repo.register_worker(first, now=now)
    repo.submit_job(serving_submission(now=now), now=now)
    assert claim_serving(repo, "worker", now=now) is not None

    restarted = serving_worker(
        "worker", epoch="52345678-1234-4234-9234-123456789abc"
    )
    with pytest.raises(ConflictError, match="serving deployment cannot change"):
        repo.register_worker(restarted, now=now)


def test_deadline_expiry_is_terminal_and_releases_capacity():
    now = utc_now()
    repo = InMemoryControlRepository()
    registration = serving_worker("worker")
    repo.register_worker(registration, now=now)
    job = repo.submit_job(
        replace(
            serving_submission(now=now),
            deadline_at=now + timedelta(seconds=1),
        ),
        now=now,
    )
    assert claim_serving(repo, "worker", now=now) is not None

    expired = repo.expire_deadline_jobs(now=now + timedelta(seconds=2))
    assert [item.job_id for item in expired] == [job.job_id]
    record = repo.get_job(job.job_id)
    assert record.status is JobStatus.FAILED
    assert record.error["type"] == "deadline_expired"
    assert repo.get_worker("worker").active_jobs == 0


def test_idempotency_equivalence_includes_serving_binding_and_deadline():
    now = utc_now()
    repo = InMemoryControlRepository()
    repo.register_worker(serving_worker("worker"), now=now)
    submission = serving_submission(now=now, idempotency_key="same")
    first = repo.submit_job(submission, now=now)
    assert repo.submit_job(submission, now=now).job_id == first.job_id

    with pytest.raises(ConflictError, match="different job request"):
        repo.submit_job(
            replace(submission, deadline_at=now + timedelta(seconds=31)),
            now=now,
        )


def test_profile_snapshot_does_not_follow_later_profile_change():
    now = utc_now()
    deployment, contract, profile, binding, _ = serving_values()
    repo = InMemoryControlRepository()
    repo.register_worker(serving_worker("worker"), now=now)
    job = repo.submit_job(serving_submission(now=now, binding=binding), now=now)

    changed = replace(profile, profile_id="local-code-v2")
    changed_binding = ServingJobBinding.from_resolved(
        resolve_profile(changed, contract, deployment)
    )
    assert changed_binding.profile_revision != binding.profile_revision
    assert repo.get_job(job.job_id).serving == binding


def test_worker_runtime_rejects_claim_from_other_runtime_epoch():
    registration = serving_worker("worker")
    binding = serving_values()[3]
    runtime = WorkerRuntime(
        spec=registration.spec,
        max_concurrency=1,
        executor=object(),
        client=object(),
        serving=registration.serving,
    )
    request = JobRequest(
        job_id="job",
        capability="llm.chat",
        payload={},
        metadata={
            "serving": binding.to_dict(),
            "claimed_deployment_revision": binding.deployment_revision,
            "claimed_serving_contract_revision": binding.serving_contract_revision,
            "claimed_runtime_instance_epoch":
                "62345678-1234-4234-9234-123456789abc",
        },
    )
    claim = ClaimedJob(
        request=request,
        lease_token="lease",
        lease_expires_at=None,
        attempts=1,
    )
    with pytest.raises(RuntimeError, match="runtime"):
        runtime._validate_claim_serving(claim)


@pytest.mark.asyncio
async def test_v1_transport_round_trips_serving_identity_and_fences_epoch():
    now = utc_now()
    repo = InMemoryControlRepository()
    registration = serving_worker("transport")
    submission = serving_submission(now=now)
    app = create_app(
        repo,
        client_token="client-secret",
        worker_token="worker-secret",
        maintenance_interval_seconds=60,
    )
    transport = httpx.ASGITransport(app=app)
    client_headers = {"authorization": "Bearer client-secret"}
    worker_headers = {"authorization": "Bearer worker-secret"}

    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        registered = await client.post(
            "/v1/workers/register",
            headers=worker_headers,
            json={
                "protocol_version": PROTOCOL_VERSION,
                "spec": {
                    "worker_id": registration.spec.worker_id,
                    "worker_class": registration.spec.worker_class,
                    "gpu_uuids": [],
                    "accelerators": [],
                    "capabilities": ["llm.chat"],
                    "labels": {},
                    "resources": {
                        "gpu_count": 0,
                        "total_vram_mb": 0,
                        "max_single_gpu_vram_mb": 0,
                    },
                },
                "max_concurrency": 1,
                "metadata": {},
                "serving": registration.serving.to_dict(),
                "extensions": [SERVING_EXTENSION],
            },
        )
        assert registered.status_code == 201
        assert registered.json()["serving"] == registration.serving.to_dict()

        created = await client.post(
            "/v1/jobs",
            headers=client_headers,
            json={
                "protocol_version": PROTOCOL_VERSION,
                "capability": submission.capability,
                "payload": dict(submission.payload),
                "requirements": {},
                "deadline_at": submission.deadline_at.isoformat(),
                "serving": submission.serving.to_dict(),
                "extensions": [SERVING_EXTENSION],
            },
        )
        assert created.status_code == 201

        claimed = await client.post(
            "/v1/workers/transport/jobs/claim",
            headers=worker_headers,
            json={
                "protocol_version": PROTOCOL_VERSION,
                "extensions": [SERVING_EXTENSION],
                "runtime_instance_epoch":
                    registration.serving.runtime_instance.epoch,
            },
        )
        assert claimed.status_code == 200
        body = claimed.json()
        assert body["serving"] == submission.serving.to_dict()
        assert (
            body["claimed_runtime_instance_epoch"]
            == registration.serving.runtime_instance.epoch
        )

        stale = await client.post(
            f"/v1/workers/transport/jobs/{body['job_id']}/complete",
            headers=worker_headers,
            json={
                "protocol_version": PROTOCOL_VERSION,
                "lease_token": body["lease_token"],
                "runtime_instance_epoch":
                    "72345678-1234-4234-9234-123456789abc",
                "extensions": [SERVING_EXTENSION],
                "result": {"outputs": {"text": "stale"}, "text": "stale"},
            },
        )
        assert stale.status_code == 409

        completed = await client.post(
            f"/v1/workers/transport/jobs/{body['job_id']}/complete",
            headers=worker_headers,
            json={
                "protocol_version": PROTOCOL_VERSION,
                "lease_token": body["lease_token"],
                "runtime_instance_epoch":
                    registration.serving.runtime_instance.epoch,
                "extensions": [SERVING_EXTENSION],
                "result": {"outputs": {"text": "ok"}, "text": "ok"},
            },
        )
        assert completed.status_code == 200
        assert completed.json()["status"] == "succeeded"



@pytest.mark.asyncio
async def test_v1_serving_fields_require_explicit_extension_marker():
    now = utc_now()
    repo = InMemoryControlRepository()
    registration = serving_worker("extension-check")
    submission = serving_submission(now=now)
    app = create_app(
        repo,
        client_token="client-secret",
        worker_token="worker-secret",
        maintenance_interval_seconds=60,
    )
    transport = httpx.ASGITransport(app=app)
    client_headers = {"authorization": "Bearer client-secret"}
    worker_headers = {"authorization": "Bearer worker-secret"}

    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        missing_worker = await client.post(
            "/v1/workers/register",
            headers=worker_headers,
            json={
                "protocol_version": PROTOCOL_VERSION,
                "spec": {
                    "worker_id": registration.spec.worker_id,
                    "worker_class": registration.spec.worker_class,
                    "gpu_uuids": [],
                    "accelerators": [],
                    "capabilities": ["llm.chat"],
                    "labels": {},
                    "resources": {
                        "gpu_count": 0,
                        "total_vram_mb": 0,
                        "max_single_gpu_vram_mb": 0,
                    },
                },
                "max_concurrency": 1,
                "metadata": {},
                "serving": registration.serving.to_dict(),
            },
        )
        assert missing_worker.status_code == 422

        unknown = await client.post(
            "/v1/jobs",
            headers=client_headers,
            json={
                "protocol_version": PROTOCOL_VERSION,
                "capability": submission.capability,
                "payload": {},
                "requirements": {},
                "deadline_at": submission.deadline_at.isoformat(),
                "serving": submission.serving.to_dict(),
                "extensions": ["unknown-extension-v1"],
            },
        )
        assert unknown.status_code == 422

        missing_job = await client.post(
            "/v1/jobs",
            headers=client_headers,
            json={
                "protocol_version": PROTOCOL_VERSION,
                "capability": submission.capability,
                "payload": {},
                "requirements": {},
                "deadline_at": submission.deadline_at.isoformat(),
                "serving": submission.serving.to_dict(),
            },
        )
        assert missing_job.status_code == 422


def test_serving_admission_is_bounded_by_liveness_capacity_and_deadline():
    now = utc_now()
    repo = InMemoryControlRepository(worker_ttl_seconds=60)
    request = serving_submission(now=now)

    with pytest.raises(NoCompatibleDeployment):
        repo.submit_job(request, now=now)

    registration = serving_worker("worker")
    repo.register_worker(registration, now=now)
    accepted = repo.submit_job(request, now=now)
    claimed = claim_serving(repo, "worker", now=now)
    assert claimed is not None and claimed.job_id == accepted.job_id

    with pytest.raises(OverloadedError):
        repo.submit_job(serving_submission(now=now), now=now)

    with pytest.raises(DeadlineExceededError):
        repo.submit_job(
            replace(
                serving_submission(now=now),
                deadline_at=now - timedelta(seconds=1),
            ),
            now=now,
        )

    draining = InMemoryControlRepository()
    draining.register_worker(serving_worker("draining"), now=now)
    set_serving_state(
        draining, "draining", WorkerState.DRAINING, now=now
    )
    with pytest.raises(NoCompatibleDeployment):
        draining.submit_job(serving_submission(now=now), now=now)

    stale = InMemoryControlRepository(worker_ttl_seconds=60)
    stale.register_worker(
        serving_worker("stale"),
        now=now - timedelta(seconds=61),
    )
    with pytest.raises(NoCompatibleDeployment):
        stale.submit_job(serving_submission(now=now), now=now)


def test_idempotent_retry_precedes_transient_admission_recheck():
    now = utc_now()
    repo = InMemoryControlRepository()
    registration = serving_worker("worker")
    repo.register_worker(registration, now=now)
    request = serving_submission(now=now, idempotency_key="durable")
    first = repo.submit_job(request, now=now)
    claim = claim_serving(repo, "worker", now=now)
    assert claim is not None

    retry = repo.submit_job(
        request,
        now=request.deadline_at + timedelta(seconds=1),
    )
    assert retry.job_id == first.job_id


@pytest.mark.asyncio
async def test_serving_admission_http_errors_are_explicit():
    now = utc_now()
    repo = InMemoryControlRepository()
    app = create_app(
        repo,
        client_token="client-secret",
        worker_token="worker-secret",
        maintenance_interval_seconds=60,
    )
    transport = httpx.ASGITransport(app=app)
    headers = {"authorization": "Bearer client-secret"}
    *_, binding, _ = serving_values()

    def body(deadline):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "extensions": [SERVING_EXTENSION],
            "capability": "llm.chat",
            "payload": {},
            "requirements": {},
            "serving": binding.to_dict(),
            "deadline_at": deadline.isoformat(),
        }

    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        missing = await client.post(
            "/v1/jobs",
            headers=headers,
            json=body(now + timedelta(seconds=10)),
        )
        assert missing.status_code == 503

        repo.register_worker(serving_worker("worker"), now=utc_now())
        accepted = await client.post(
            "/v1/jobs",
            headers=headers,
            json=body(now + timedelta(seconds=10)),
        )
        assert accepted.status_code == 201
        claimed = claim_serving(repo, "worker", now=utc_now())
        assert claimed is not None

        overloaded = await client.post(
            "/v1/jobs",
            headers=headers,
            json=body(now + timedelta(seconds=10)),
        )
        assert overloaded.status_code == 429

        expired = await client.post(
            "/v1/jobs",
            headers=headers,
            json=body(now - timedelta(seconds=1)),
        )
        assert expired.status_code == 408


def test_restart_fences_stale_serving_process_before_new_attempt():
    now = utc_now()
    repo = InMemoryControlRepository()
    first = serving_worker("worker")
    repo.register_worker(first, now=now)
    second = serving_worker(
        "worker", epoch="82345678-1234-4234-9234-123456789abc"
    )
    repo.register_worker(second, now=now)

    with pytest.raises(ConflictError, match="runtime instance is stale"):
        repo.claim_next_job(
            "worker",
            runtime_instance_epoch=first.serving.runtime_instance.epoch,
            now=now,
        )
    with pytest.raises(ConflictError, match="runtime instance is stale"):
        repo.heartbeat_worker(
            "worker",
            WorkerHeartbeat(
                runtime_instance_epoch=first.serving.runtime_instance.epoch
            ),
            now=now,
        )
    with pytest.raises(ConflictError, match="runtime instance is stale"):
        repo.set_worker_state(
            "worker",
            WorkerState.DRAINING,
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


@pytest.mark.asyncio
async def test_serving_worker_negotiates_extension_before_registration():
    registration = serving_worker("negotiation")
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/v1/ready":
            return httpx.Response(
                200,
                json={"ready": True, "protocol_version": PROTOCOL_VERSION},
            )
        if request.url.path == "/v1/workers/register":
            pytest.fail("serving registration reached an unnegotiated Control")
        return httpx.Response(404)

    async with ControlClient(
        "http://legacy-control",
        "worker-secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(
            ControlTransportError,
            match="does not support required extension",
        ):
            await client.register(
                spec=registration.spec,
                max_concurrency=1,
                serving=registration.serving,
            )

    assert calls == ["/v1/ready"]


def test_worker_client_rejects_unnegotiated_serving_response():
    registration = serving_worker("response-check")
    response = httpx.Response(
        200,
        json={
            "protocol_version": PROTOCOL_VERSION,
            "serving": registration.serving.to_dict(),
        },
    )
    with pytest.raises(ControlTransportError, match="omitted the serving extension"):
        ControlClient._object(response)

    unknown = httpx.Response(
        200,
        json={
            "protocol_version": PROTOCOL_VERSION,
            "extensions": [SERVING_EXTENSION, "future-v2"],
        },
    )
    with pytest.raises(ControlTransportError, match="unsupported extension"):
        ControlClient._object(unknown)


def test_durable_job_record_rejects_inconsistent_serving_identity():
    now = utc_now()
    repo = InMemoryControlRepository()
    repo.register_worker(serving_worker("worker"), now=now)
    queued = repo.submit_job(serving_submission(now=now), now=now)

    with pytest.raises(ValueError, match="binding capability"):
        replace(
            queued,
            serving=replace(queued.serving, capability="other.capability"),
        )
    with pytest.raises(ValueError, match="require deadline"):
        replace(queued, deadline_at=None)

    claimed = claim_serving(repo, "worker", now=now)
    assert claimed is not None
    with pytest.raises(ValueError, match="deployment revision"):
        replace(claimed, claimed_deployment_revision=digest("a"))
    with pytest.raises(ValueError, match="contract revision"):
        replace(claimed, claimed_serving_contract_revision=digest("b"))
    with pytest.raises(ValueError, match="requires a serving binding"):
        replace(claimed, serving=None, deadline_at=None)
