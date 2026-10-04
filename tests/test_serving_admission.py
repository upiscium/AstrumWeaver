from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import json
from uuid import uuid4

import httpx
import pytest

from astrumweaver import JobResult, ResourceShape, WorkerSpec
from astrumweaver.control.api import create_app
from astrumweaver.control.models import JobStatus, JobSubmission, WorkerRegistration, WorkerState
from astrumweaver.control.repository import (
    ConflictError,
    InMemoryControlRepository,
    NoCompatibleDeployment,
    OverloadedError,
)
from astrumweaver.serving import (
    DeploymentIdentity,
    LogicalServingProfile,
    ServingContract,
    ServingJobBinding,
    WorkerServingAdvertisement,
    WorkerServingManifest,
    resolve_profile,
)
from astrumweaver.transport import PROTOCOL_VERSION, SERVING_EXTENSION
from astrumweaver.worker import ControlClient, ControlTransportError, WorkerRuntime
from astrumweaver.worker.daemon import _load_serving_manifest
from astrumweaver.control.models import utc_now


CLIENT_TOKEN = "client-secret"
WORKER_TOKEN = "worker-secret"


def digest(ch: str) -> str:
    return "sha256:" + ch * 64


def serving_values(*, model: str = "3", epoch: str | None = None):
    deployment = DeploymentIdentity(
        provider_id="llama-cpp",
        runtime_artifact_sha256=digest("1"),
        adapter_artifact_sha256=digest("2"),
        model_artifact_sha256=digest(model),
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
        capability="llm.chat",
        operation_schema="chat-v1",
        required_features=frozenset({"tools"}),
        limits={"input_tokens": 4096},
    )
    resolved = resolve_profile(profile, contract, deployment)
    binding = ServingJobBinding.from_resolved(resolved)
    advertisement = WorkerServingAdvertisement(
        deployment_revision=deployment.revision,
        runtime_instance_epoch=epoch or str(uuid4()),
        contract_revisions={"llm.chat": contract.revision},
    )
    return deployment, contract, profile, binding, advertisement


def registration(worker_id: str, advertisement: WorkerServingAdvertisement):
    return WorkerRegistration(
        spec=WorkerSpec(
            worker_id=worker_id,
            worker_class="cpu-test",
            resources=ResourceShape(),
            capabilities=frozenset({"llm.chat"}),
        ),
        serving=advertisement,
    )


def serving_submission(binding: ServingJobBinding, *, now, key: str | None = None):
    return JobSubmission(
        capability="llm.chat",
        payload={"messages": [{"role": "user", "content": "ping"}]},
        serving_binding=binding,
        deadline_at=now + timedelta(seconds=10),
        idempotency_key=key,
    )


def test_serving_bound_submission_requires_deadline():
    _, _, _, binding, _ = serving_values()
    with pytest.raises(ValueError, match="require deadline_at"):
        JobSubmission(capability="llm.chat", serving_binding=binding)


def test_reference_admission_capacity_and_epoch_fencing():
    now = utc_now()
    repo = InMemoryControlRepository(lease_seconds=30)
    _, _, _, binding, advertisement = serving_values()

    with pytest.raises(NoCompatibleDeployment):
        repo.submit_job(serving_submission(binding, now=now), now=now)

    worker = repo.register_worker(registration("worker-a", advertisement), now=now)
    job = repo.submit_job(serving_submission(binding, now=now), now=now)
    claimed = repo.claim_next_job(
        worker.worker_id,
        runtime_instance_epoch=advertisement.runtime_instance_epoch,
        now=now,
    )
    assert claimed is not None
    assert claimed.serving_binding == binding
    assert claimed.attempt_runtime_instance_epoch == advertisement.runtime_instance_epoch

    with pytest.raises(OverloadedError):
        repo.submit_job(serving_submission(binding, now=now), now=now)

    with pytest.raises(ConflictError, match="runtime instance epoch"):
        repo.complete_job(
            job.job_id,
            JobResult(outputs={"ok": False}),
            worker_id=worker.worker_id,
            lease_token=claimed.lease_token or "",
            runtime_instance_epoch=str(uuid4()),
            now=now,
        )

    done = repo.complete_job(
        job.job_id,
        JobResult(outputs={"ok": True}),
        worker_id=worker.worker_id,
        lease_token=claimed.lease_token or "",
        runtime_instance_epoch=advertisement.runtime_instance_epoch,
        now=now,
    )
    assert done.status is JobStatus.SUCCEEDED


def test_active_attempt_blocks_runtime_epoch_replacement_then_old_epoch_is_stale():
    now = utc_now()
    repo = InMemoryControlRepository()
    _, _, _, binding, old = serving_values()
    spec = registration("worker-a", old)
    repo.register_worker(spec, now=now)
    job = repo.submit_job(serving_submission(binding, now=now), now=now)
    claimed = repo.claim_next_job(
        "worker-a", runtime_instance_epoch=old.runtime_instance_epoch, now=now
    )
    assert claimed is not None

    new = replace(old, runtime_instance_epoch=str(uuid4()))
    with pytest.raises(ConflictError, match="serving identity"):
        repo.register_worker(registration("worker-a", new), now=now)

    repo.cancel_job(job.job_id, now=now)
    repo.register_worker(registration("worker-a", new), now=now)

    with pytest.raises(ConflictError, match="epoch is stale"):
        repo.claim_next_job(
            "worker-a", runtime_instance_epoch=old.runtime_instance_epoch, now=now
        )


def test_durable_job_rejects_binding_capability_mismatch():
    now = utc_now()
    repo = InMemoryControlRepository()
    _, _, _, binding, advertisement = serving_values()
    repo.register_worker(registration("worker-a", advertisement), now=now)
    job = repo.submit_job(serving_submission(binding, now=now), now=now)

    with pytest.raises(ValueError, match="binding capability"):
        replace(
            job,
            serving_binding=replace(binding, capability="other.capability"),
        )


def test_job_snapshot_and_idempotency_include_resolved_binding_and_deadline():
    now = utc_now()
    repo = InMemoryControlRepository()
    _, _, profile, binding, advertisement = serving_values()
    repo.register_worker(registration("worker-a", advertisement), now=now)

    submission = serving_submission(binding, now=now, key="same-request")
    first = repo.submit_job(submission, now=now)
    replay = repo.submit_job(submission, now=now)
    assert replay.job_id == first.job_id

    changed_profile = replace(profile, profile_id="local-code-v2")
    assert changed_profile.revision != binding.profile_revision
    assert repo.get_job(first.job_id).serving_binding == binding

    changed_binding = replace(binding, profile_revision=changed_profile.revision)
    with pytest.raises(ConflictError, match="idempotency"):
        repo.submit_job(
            replace(submission, serving_binding=changed_binding),
            now=now,
        )

    with pytest.raises(ConflictError, match="idempotency"):
        repo.submit_job(
            replace(submission, deadline_at=submission.deadline_at + timedelta(seconds=1)),
            now=now,
        )


def test_deadline_caps_lease_and_prevents_retry_or_queued_execution():
    now = utc_now()
    repo = InMemoryControlRepository(lease_seconds=300)
    _, _, _, binding, advertisement = serving_values()
    repo.register_worker(registration("worker-a", advertisement), now=now)

    deadline = now + timedelta(seconds=1)
    first = repo.submit_job(
        replace(serving_submission(binding, now=now), deadline_at=deadline),
        now=now,
    )
    second = repo.submit_job(
        replace(serving_submission(binding, now=now), deadline_at=deadline),
        now=now,
    )
    claimed = repo.claim_next_job(
        "worker-a",
        runtime_instance_epoch=advertisement.runtime_instance_epoch,
        now=now,
    )
    assert claimed is not None
    assert claimed.lease_expires_at == deadline

    later = now + timedelta(seconds=2)
    recovered = repo.recover_expired_jobs(now=later)
    assert recovered[0].status is JobStatus.FAILED
    assert recovered[0].error["code"] == "deadline-exceeded"
    assert recovered[0].error["retryable"] is False

    expired = repo.expire_deadline_jobs(now=later)
    assert [item.job_id for item in expired] == [second.job_id]
    assert repo.get_job(second.job_id).status is JobStatus.FAILED
    assert repo.get_job(first.job_id).status is JobStatus.FAILED


def test_worker_client_rejects_unnegotiated_or_unknown_serving_response():
    _, _, _, binding, advertisement = serving_values()

    missing_extension = httpx.Response(
        200,
        json={
            "protocol_version": PROTOCOL_VERSION,
            "serving": advertisement.to_dict(),
        },
    )
    with pytest.raises(ControlTransportError, match="omitted the serving-v1"):
        ControlClient._object(missing_extension)

    unknown_extension = httpx.Response(
        200,
        json={
            "protocol_version": PROTOCOL_VERSION,
            "extensions": ["serving-v1", "unknown-v2"],
            "serving_binding": binding.to_dict(),
        },
    )
    with pytest.raises(ControlTransportError, match="unsupported extension"):
        ControlClient._object(unknown_extension)


@pytest.mark.asyncio
async def test_serving_worker_negotiates_before_mutating_legacy_control():
    _, _, _, _, advertisement = serving_values()
    spec = registration("worker-a", advertisement).spec
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/v1/ready":
            # Exact legacy-v1 shape: no serving extension advertisement.
            return httpx.Response(
                200,
                json={"ready": True, "protocol_version": PROTOCOL_VERSION},
            )
        if request.url.path == "/v1/workers/register":
            pytest.fail("serving registration reached an unnegotiated Control")
        return httpx.Response(404)

    async with ControlClient(
        "http://legacy-control",
        WORKER_TOKEN,
        transport=httpx.MockTransport(handler),
    ) as control:
        with pytest.raises(
            ControlTransportError,
            match="does not support required extension",
        ):
            await control.register(
                spec=spec,
                max_concurrency=1,
                serving=advertisement,
            )

    assert calls == ["/v1/ready"]


def test_serving_transport_requires_explicit_extension():
    from astrumweaver.transport import (
        job_submission_from_dict,
        worker_registration_from_dict,
    )

    now = utc_now()
    _, _, _, binding, advertisement = serving_values()
    job = {
        "protocol_version": PROTOCOL_VERSION,
        "capability": "llm.chat",
        "payload": {},
        "requirements": {},
        "serving_binding": binding.to_dict(),
        "deadline_at": (now + timedelta(seconds=10)).isoformat(),
    }
    with pytest.raises(ValueError, match="serving-v1"):
        job_submission_from_dict(job)
    parsed = job_submission_from_dict(
        {**job, "extensions": [SERVING_EXTENSION]}
    )
    assert parsed.serving_binding == binding

    worker = {
        "protocol_version": PROTOCOL_VERSION,
        "spec": {
            "worker_id": "worker-a",
            "worker_class": "cpu-test",
            "gpu_uuids": [],
            "capabilities": ["llm.chat"],
            "labels": {},
            "resources": {
                "gpu_count": 0,
                "total_vram_mb": 0,
                "max_single_gpu_vram_mb": 0,
            },
        },
        "serving": advertisement.to_dict(),
    }
    with pytest.raises(ValueError, match="serving-v1"):
        worker_registration_from_dict(worker)
    parsed_worker = worker_registration_from_dict(
        {**worker, "extensions": [SERVING_EXTENSION]}
    )
    assert parsed_worker.serving == advertisement


class EchoExecutor:
    capabilities = frozenset({"llm.chat"})

    async def execute(self, job):
        return JobResult(outputs={"echo": dict(job.payload)})

    async def cancel(self, job_id: str) -> None:
        return None

    async def residency(self):
        from astrumweaver import ResidencyReport
        return ResidencyReport()


@pytest.mark.asyncio
async def test_serving_worker_transport_round_trip_binds_exact_runtime_epoch():
    now = utc_now()
    repository = InMemoryControlRepository()
    app = create_app(
        repository,
        client_token=CLIENT_TOKEN,
        worker_token=WORKER_TOKEN,
        maintenance_interval_seconds=60,
    )
    transport = httpx.ASGITransport(app=app)
    _, _, _, binding, advertisement = serving_values()
    spec = registration("worker-a", advertisement).spec

    async with ControlClient(
        "http://control", WORKER_TOKEN, transport=transport
    ) as control:
        runtime = WorkerRuntime(
            spec=spec,
            max_concurrency=1,
            executor=EchoExecutor(),
            client=control,
            serving=advertisement,
        )
        await runtime.register()

        async with httpx.AsyncClient(
            transport=transport, base_url="http://control"
        ) as client:
            response = await client.post(
                "/v1/jobs",
                headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
                json={
                    "protocol_version": PROTOCOL_VERSION,
                    "extensions": [SERVING_EXTENSION],
                    "capability": "llm.chat",
                    "payload": {"value": 7},
                    "requirements": {},
                    "serving_binding": binding.to_dict(),
                    "deadline_at": (now + timedelta(seconds=10)).isoformat(),
                },
            )
            assert response.status_code == 201
            job_id = response.json()["job_id"]

        claimed = await control.claim(
            spec.worker_id,
            runtime_instance_epoch=advertisement.runtime_instance_epoch,
        )
        assert claimed is not None
        assert claimed.serving_binding == binding
        assert claimed.runtime_instance_epoch == advertisement.runtime_instance_epoch
        await runtime._execute_claim(claimed)

        async with httpx.AsyncClient(
            transport=transport, base_url="http://control"
        ) as client:
            fetched = await client.get(
                f"/v1/jobs/{job_id}",
                headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
            )
        assert fetched.json()["status"] == "succeeded"
        assert fetched.json()["serving_binding"] == binding.to_dict()



def test_worker_serving_manifest_round_trip_and_epoch_is_process_owned(tmp_path):
    deployment, contract, _, _, _ = serving_values()
    manifest = WorkerServingManifest(
        deployment=deployment,
        contracts=(contract,),
    )
    path = tmp_path / "serving.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    loaded = _load_serving_manifest(str(path))
    assert loaded == manifest
    epoch_a = str(uuid4())
    epoch_b = str(uuid4())
    first = loaded.advertisement(epoch_a)
    second = loaded.advertisement(epoch_b)
    assert first.deployment_revision == deployment.revision
    assert first.contract_revisions["llm.chat"] == contract.revision
    assert first.runtime_instance_epoch == epoch_a
    assert second.runtime_instance_epoch == epoch_b
    assert first.runtime_instance_epoch != second.runtime_instance_epoch


def test_worker_serving_manifest_rejects_contract_from_other_deployment():
    deployment, contract, _, _, _ = serving_values()
    with pytest.raises(ValueError, match="deployment-revision-mismatch"):
        WorkerServingManifest(
            deployment=deployment,
            contracts=(
                replace(contract, deployment_revision=digest("9")),
            ),
        )



def test_worker_registration_rejects_serving_contract_not_in_worker_capabilities():
    _, serving = serving_values()[3:5]
    inconsistent = replace(
        serving,
        contract_revisions={"other.capability": next(iter(serving.contract_revisions.values()))},
    )
    with pytest.raises(ValueError, match="serving contract capabilities"):
        WorkerRegistration(
            spec=WorkerSpec(
                worker_id="worker-a",
                worker_class="cpu-test",
                resources=ResourceShape(),
                capabilities=frozenset({"llm.chat"}),
            ),
            serving=inconsistent,
        )



@pytest.mark.asyncio
async def test_serving_admission_http_errors_are_explicit_and_bounded():
    now = utc_now()
    repository = InMemoryControlRepository()
    app = create_app(
        repository,
        client_token=CLIENT_TOKEN,
        worker_token=WORKER_TOKEN,
        maintenance_interval_seconds=60,
    )
    transport = httpx.ASGITransport(app=app)
    _, _, _, binding, advertisement = serving_values()
    headers = {"authorization": f"Bearer {CLIENT_TOKEN}"}

    def body(deadline):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "extensions": [SERVING_EXTENSION],
            "capability": "llm.chat",
            "payload": {"value": 1},
            "requirements": {},
            "serving_binding": binding.to_dict(),
            "deadline_at": deadline.isoformat(),
        }

    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        missing = await client.post(
            "/v1/jobs",
            headers=headers,
            json=body(now + timedelta(seconds=10)),
        )
        assert missing.status_code == 503

        repository.register_worker(registration("worker-a", advertisement), now=now)
        accepted = await client.post(
            "/v1/jobs",
            headers=headers,
            json=body(now + timedelta(seconds=10)),
        )
        assert accepted.status_code == 201
        claimed = repository.claim_next_job(
            "worker-a",
            runtime_instance_epoch=advertisement.runtime_instance_epoch,
            now=utc_now(),
        )
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



def test_recovered_serving_job_can_move_to_new_epoch_of_same_deployment():
    now = utc_now()
    repo = InMemoryControlRepository(lease_seconds=1)
    _, _, _, binding, old = serving_values()
    repo.register_worker(registration("worker-a", old), now=now)
    job = repo.submit_job(serving_submission(binding, now=now), now=now)
    first = repo.claim_next_job(
        "worker-a", runtime_instance_epoch=old.runtime_instance_epoch, now=now
    )
    assert first is not None

    recovered_at = now + timedelta(seconds=2)
    recovered = repo.recover_expired_jobs(now=recovered_at)
    assert [item.job_id for item in recovered] == [job.job_id]
    assert recovered[0].status is JobStatus.QUEUED
    assert recovered[0].attempt_runtime_instance_epoch is None

    new = replace(old, runtime_instance_epoch=str(uuid4()))
    repo.register_worker(registration("worker-a", new), now=recovered_at)
    second = repo.claim_next_job(
        "worker-a",
        runtime_instance_epoch=new.runtime_instance_epoch,
        now=recovered_at,
    )
    assert second is not None
    assert second.job_id == job.job_id
    assert second.attempts == first.attempts + 1
    assert second.attempt_runtime_instance_epoch == new.runtime_instance_epoch

    with pytest.raises(ConflictError):
        repo.complete_job(
            job.job_id,
            JobResult(outputs={"stale": True}),
            worker_id="worker-a",
            lease_token=first.lease_token or "",
            runtime_instance_epoch=old.runtime_instance_epoch,
            now=recovered_at,
        )

def test_serving_worker_cannot_claim_unbound_covered_capability():
    now = utc_now()
    repo = InMemoryControlRepository()
    _, _, _, binding, advertisement = serving_values()
    repo.register_worker(registration("serving-worker", advertisement), now=now)

    legacy = repo.submit_job(
        JobSubmission(
            capability="llm.chat",
            payload={"legacy": True},
            priority=100,
        ),
        now=now,
    )
    bound = repo.submit_job(
        replace(serving_submission(binding, now=now), priority=10),
        now=now,
    )

    claimed = repo.claim_next_job(
        "serving-worker",
        runtime_instance_epoch=advertisement.runtime_instance_epoch,
        now=now,
    )
    assert claimed is not None
    assert claimed.job_id == bound.job_id
    assert repo.get_job(legacy.job_id).status is JobStatus.QUEUED

    legacy_worker = WorkerRegistration(
        spec=WorkerSpec(
            worker_id="legacy-worker",
            worker_class="cpu-test",
            resources=ResourceShape(),
            capabilities=frozenset({"llm.chat"}),
        )
    )
    repo.register_worker(legacy_worker, now=now)
    legacy_claim = repo.claim_next_job("legacy-worker", now=now)
    assert legacy_claim is not None
    assert legacy_claim.job_id == legacy.job_id
    assert legacy_claim.serving_binding is None


def test_serving_worker_preserves_unrelated_legacy_capability():
    now = utc_now()
    repo = InMemoryControlRepository()
    _, _, _, _, advertisement = serving_values()
    base = registration("worker-a", advertisement)
    repo.register_worker(
        replace(
            base,
            spec=replace(
                base.spec,
                capabilities=frozenset({"llm.chat", "debug.echo"}),
            ),
        ),
        now=now,
    )
    legacy = repo.submit_job(
        JobSubmission(capability="debug.echo", payload={"value": 1}),
        now=now,
    )

    claimed = repo.claim_next_job(
        "worker-a",
        runtime_instance_epoch=advertisement.runtime_instance_epoch,
        now=now,
    )
    assert claimed is not None
    assert claimed.job_id == legacy.job_id
    assert claimed.serving_binding is None
    assert claimed.attempt_runtime_instance_epoch is None

def test_serving_admission_excludes_draining_and_stale_workers():
    now = utc_now()
    _, _, _, binding, advertisement = serving_values()

    draining_repo = InMemoryControlRepository()
    draining_repo.register_worker(
        registration("draining-worker", advertisement),
        now=now,
    )
    draining_repo.set_worker_state(
        "draining-worker",
        WorkerState.DRAINING,
        runtime_instance_epoch=advertisement.runtime_instance_epoch,
        now=now,
    )
    with pytest.raises(NoCompatibleDeployment):
        draining_repo.submit_job(
            serving_submission(binding, now=now),
            now=now,
        )

    stale_repo = InMemoryControlRepository(worker_ttl_seconds=60)
    stale_repo.register_worker(
        registration("stale-worker", advertisement),
        now=now - timedelta(seconds=61),
    )
    with pytest.raises(NoCompatibleDeployment):
        stale_repo.submit_job(
            serving_submission(binding, now=now),
            now=now,
        )

