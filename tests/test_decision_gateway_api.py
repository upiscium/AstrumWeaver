from __future__ import annotations

import asyncio

import httpx
import pytest

from astrumweaver import JobResult, ResourceShape, WorkerSpec
from astrumweaver.control import (
    InMemoryControlRepository,
    WorkerRegistration,
)
from astrumweaver.control.api import create_app
from astrumweaver.gateway.decision import (
    DECISION_OPERATION_SCHEMA,
    DECISION_SCORE_KIND,
    LLAMA_CPP_SCORE_SEMANTICS,
    LLAMA_CPP_SYSTEM_ONE_ADAPTER,
    DecisionGatewayProfile,
    DecisionProfileCatalog,
    DecisionSemanticsIdentity,
)
from astrumweaver.gateway.decision_api import create_decision_router
from astrumweaver.serving import (
    DeploymentIdentity,
    LogicalServingProfile,
    RuntimeInstance,
    ServingContract,
    WorkerServingAdvertisement,
    resolve_profile,
)


CLIENT_TOKEN = "client-secret"
WORKER_TOKEN = "worker-secret"
EPOCH = "12345678-1234-4234-9234-123456789abc"


def digest(ch: str) -> str:
    return "sha256:" + ch * 64


def serving_values(*, timeout: float = 1.0, threshold: float = 0.6):
    deployment = DeploymentIdentity(
        provider_id="llama-cpp",
        runtime_artifact_sha256=digest("1"),
        adapter_artifact_sha256=digest("2"),
        model_artifact_sha256=digest("3"),
        execution_config_sha256=digest("4"),
        quantization="Q4_K_M",
        tokenizer_artifact_sha256=digest("5"),
    )
    semantics = DecisionSemanticsIdentity(
        deployment_revision=deployment.revision,
        adapter_id=LLAMA_CPP_SYSTEM_ONE_ADAPTER,
        score_kind=DECISION_SCORE_KIND,
        provider_score_semantics=LLAMA_CPP_SCORE_SEMANTICS,
        calibration_status="uncalibrated",
        calibration_reference_sha256=None,
        abstain_below=threshold,
    )
    features = frozenset(
        {
            "native-decision-head",
            "choice-probabilities",
            "multi-token-choice-labels",
            "provider-temperature-softmax",
        }
    )
    limits = {
        "state_bytes": 1024,
        "question_bytes": 512,
        "choice_count": 8,
        "choice_id_bytes": 64,
        "choice_label_bytes": 256,
        "state_tokens": 128,
        "question_tokens": 64,
        "choice_label_tokens": 32,
        "aggregate_tokens": 256,
        "request_bytes": 4096,
    }
    contract = ServingContract(
        deployment_revision=deployment.revision,
        capability="decision.system_one",
        operation_schema=DECISION_OPERATION_SCHEMA,
        validation_evidence_sha256=digest("6"),
        features=features,
        limits=limits,
        semantic_revision=semantics.decision_semantics_id,
    )
    logical = LogicalServingProfile(
        profile_id="decision-local-v1",
        deployment_revision=deployment.revision,
        serving_contract_revision=contract.revision,
        capability="decision.system_one",
        operation_schema=DECISION_OPERATION_SCHEMA,
        required_features=features,
        limits={},
    )
    resolved = resolve_profile(logical, contract, deployment)
    profile = DecisionGatewayProfile(
        resolved=resolved,
        semantics=semantics,
        adapter_id=LLAMA_CPP_SYSTEM_ONE_ADAPTER,
        request_timeout_seconds=timeout,
        max_attempts=2,
    )
    advertisement = WorkerServingAdvertisement(
        deployment_revision=deployment.revision,
        runtime_instance=RuntimeInstance(deployment.revision, EPOCH),
        contracts=(contract,),
    )
    return profile, advertisement


def register_worker(
    repository: InMemoryControlRepository,
    advertisement: WorkerServingAdvertisement,
) -> None:
    repository.register_worker(
        WorkerRegistration(
            spec=WorkerSpec(
                worker_id="decision-worker",
                worker_class="cpu-test",
                resources=ResourceShape(),
                capabilities=frozenset({"decision.system_one"}),
            ),
            serving=advertisement,
        )
    )


def app_with_gateway(
    repository: InMemoryControlRepository,
    profile: DecisionGatewayProfile,
):
    app = create_app(
        repository,
        client_token=CLIENT_TOKEN,
        worker_token=WORKER_TOKEN,
        maintenance_interval_seconds=60,
    )
    app.include_router(
        create_decision_router(
            repository,
            DecisionProfileCatalog((profile,)),
            client_auth="bearer",
            client_token=CLIENT_TOKEN,
            poll_interval_seconds=0.005,
        )
    )
    return app


def request(profile: DecisionGatewayProfile) -> dict:
    return {
        "profile": profile.profile_id,
        "profile_revision": profile.resolved.profile_revision,
        "state": "A test failed immediately after a dependency update.",
        "question": "What should the agent do next?",
        "choices": [
            {"id": "inspect", "label": "inspect the failure carefully"},
            {"id": "retry", "label": "retry without changing anything"},
            {"id": "escalate", "label": "ask a human for review"},
        ],
    }


def provider_result(probabilities=(0.7, 0.2, 0.1)):
    return {
        "model": "astrumweaver",
        "answers": {
            "decision": {
                "type": "choice",
                "choice": "0000",
                "probabilities": {
                    f"{index:04d}": value
                    for index, value in enumerate(probabilities)
                },
                "confidence": 0.55,
            }
        },
        "usage": {"input_tokens": 42, "output_tokens": 0},
    }


async def complete_next(
    repository: InMemoryControlRepository,
    outputs: dict,
):
    deadline = asyncio.get_running_loop().time() + 1
    while True:
        claimed = repository.claim_next_job(
            "decision-worker",
            runtime_instance_epoch=EPOCH,
        )
        if claimed is not None:
            break
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(
                "gateway did not submit a claimable decision job"
            )
        await asyncio.sleep(0.001)

    repository.complete_job(
        claimed.job_id,
        JobResult(outputs=outputs),
        worker_id="decision-worker",
        lease_token=claimed.lease_token or "",
        runtime_instance_epoch=EPOCH,
    )
    return claimed


@pytest.mark.asyncio
async def test_decision_profiles_expose_semantics_without_authority():
    repository = InMemoryControlRepository()
    profile, _ = serving_values()
    app = app_with_gateway(repository, profile)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://control",
    ) as client:
        unauthorized = await client.get("/v1/decision-profiles")
        response = await client.get(
            "/v1/decision-profiles",
            headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
        )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    item = response.json()["data"][0]
    assert item["id"] == "decision-local-v1"
    extension = item["x_astrumweaver"]
    assert extension["score_kind"] == DECISION_SCORE_KIND
    assert extension["calibration_status"] == "uncalibrated"
    assert extension["mode"] == "shadow"
    assert extension["authority"] == "recommendation-only"


@pytest.mark.asyncio
async def test_decision_request_uses_durable_binding_and_ordered_scores():
    repository = InMemoryControlRepository()
    profile, advertisement = serving_values()
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, profile)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://control",
    ) as client:
        pending = asyncio.create_task(
            client.post(
                "/v1/decisions",
                headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
                json=request(profile),
            )
        )
        claimed = await complete_next(repository, provider_result())
        response = await pending

    assert response.status_code == 200
    body = response.json()
    assert body["choice_id"] == "inspect"
    assert body["abstained"] is False
    assert [item["id"] for item in body["scores"]] == [
        "inspect",
        "retry",
        "escalate",
    ]
    assert body["x_astrumweaver"]["authority"] == "recommendation-only"
    timing = body["x_astrumweaver"]["timing"]
    assert timing["queue_wait_ms"] >= 0
    assert timing["execution_ms"] >= 0
    assert timing["durable_job_ms"] >= 0
    assert claimed.serving == profile.binding
    payload = dict(claimed.payload)
    assert payload["decision_semantics_id"] == profile.decision_semantics_id
    assert payload["choice_ids"] == ["inspect", "retry", "escalate"]
    criteria = payload["request"]["questions"]["decision"]["criteria"]
    assert list(criteria) == ["0000", "0001", "0002"]


@pytest.mark.asyncio
async def test_decision_abstention_is_success_not_request_failure():
    repository = InMemoryControlRepository()
    profile, advertisement = serving_values(threshold=0.6)
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, profile)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://control",
    ) as client:
        pending = asyncio.create_task(
            client.post(
                "/v1/decisions",
                headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
                json=request(profile),
            )
        )
        await complete_next(
            repository,
            provider_result((0.4, 0.35, 0.25)),
        )
        response = await pending

    assert response.status_code == 200
    assert response.json()["choice_id"] is None
    assert response.json()["abstained"] is True


@pytest.mark.asyncio
async def test_stale_profile_revision_rejected_before_admission():
    repository = InMemoryControlRepository()
    profile, advertisement = serving_values()
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, profile)
    transport = httpx.ASGITransport(app=app)
    body = request(profile)
    body["profile_revision"] = digest("f")

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://control",
    ) as client:
        response = await client.post(
            "/v1/decisions",
            headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
            json=body,
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "profile_revision_mismatch"
    assert repository.list_jobs() == []


@pytest.mark.asyncio
async def test_invalid_provider_probability_set_returns_502():
    repository = InMemoryControlRepository()
    profile, advertisement = serving_values()
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, profile)
    transport = httpx.ASGITransport(app=app)
    invalid = provider_result()
    del invalid["answers"]["decision"]["probabilities"]["0002"]

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://control",
    ) as client:
        pending = asyncio.create_task(
            client.post(
                "/v1/decisions",
                headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
                json=request(profile),
            )
        )
        await complete_next(repository, invalid)
        response = await pending

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_provider_response"


@pytest.mark.asyncio
async def test_decision_timeout_cancels_only_owned_job():
    repository = InMemoryControlRepository()
    profile, advertisement = serving_values(timeout=0.03)
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, profile)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://control",
    ) as client:
        response = await client.post(
            "/v1/decisions",
            headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
            json=request(profile),
        )

    assert response.status_code == 504
    jobs = repository.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].status.value == "cancelled"