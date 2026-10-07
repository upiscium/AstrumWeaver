from __future__ import annotations

from astrumweaver import ResourceShape, WorkerSpec
from astrumweaver.execution import JobRequest
from astrumweaver.serving import (
    DeploymentIdentity,
    LogicalServingProfile,
    RuntimeInstance,
    ServingContract,
    ServingJobBinding,
    WorkerServingAdvertisement,
    resolve_profile,
)
from astrumweaver.worker.client import ClaimedJob
from astrumweaver.worker.runtime import WorkerRuntime


EPOCH = "12345678-1234-4234-9234-123456789abc"


def digest(ch: str) -> str:
    return "sha256:" + ch * 64


def test_worker_injects_locally_verified_semantic_revision_for_executor():
    deployment = DeploymentIdentity(
        provider_id="llama-cpp",
        runtime_artifact_sha256=digest("1"),
        adapter_artifact_sha256=digest("2"),
        model_artifact_sha256=digest("3"),
        execution_config_sha256=digest("4"),
        quantization="Q8_0",
        tokenizer_artifact_sha256=digest("5"),
    )
    semantic_revision = digest("8")
    contract = ServingContract(
        deployment_revision=deployment.revision,
        capability="text.embed",
        operation_schema="openai-embeddings-v1",
        validation_evidence_sha256=digest("6"),
        features=frozenset(
            {"float", "pooling-last", "normalization-l2"}
        ),
        limits={
            "item_tokens": 512,
            "item_bytes": 4096,
            "batch_items": 8,
            "batch_bytes": 16384,
            "aggregate_tokens": 2048,
            "request_bytes": 32768,
        },
        semantic_revision=semantic_revision,
    )
    profile = LogicalServingProfile(
        profile_id="notes-embed-v1",
        deployment_revision=deployment.revision,
        serving_contract_revision=contract.revision,
        capability="text.embed",
        operation_schema="openai-embeddings-v1",
        required_features=contract.features,
        limits={},
    )
    binding = ServingJobBinding.from_resolved(
        resolve_profile(profile, contract, deployment)
    )
    advertisement = WorkerServingAdvertisement(
        deployment_revision=deployment.revision,
        runtime_instance=RuntimeInstance(deployment.revision, EPOCH),
        contracts=(contract,),
    )
    spec = WorkerSpec(
        worker_id="embed-worker",
        worker_class="cpu-test",
        resources=ResourceShape(),
        capabilities=frozenset({"text.embed"}),
    )
    runtime = WorkerRuntime(
        spec=spec,
        max_concurrency=1,
        executor=object(),
        client=object(),
        serving=advertisement,
    )
    claimed = ClaimedJob(
        request=JobRequest(
            job_id="embed-job",
            capability="text.embed",
            payload={"schema_version": "embedding-job-v1"},
            metadata={
                "serving": binding.to_dict(),
                "claimed_deployment_revision": deployment.revision,
                "claimed_serving_contract_revision": contract.revision,
                "claimed_runtime_instance_epoch": EPOCH,
            },
        ),
        lease_token="lease",
        lease_expires_at=None,
        attempts=1,
    )

    runtime._validate_claim_serving(claimed)
    request = runtime._executor_request(claimed)

    assert request is not claimed.request
    assert request.metadata["serving_semantic_revision"] == semantic_revision
    assert request.payload == claimed.request.payload
