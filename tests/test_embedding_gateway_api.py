from __future__ import annotations

import asyncio
import math

import httpx
import pytest

from astrumweaver import JobResult, ResourceShape, WorkerSpec
from astrumweaver.control import (
    InMemoryControlRepository,
    WorkerRegistration,
)
from astrumweaver.control.api import create_app
from astrumweaver.gateway.embedding import (
    EMBEDDING_OPERATION_SCHEMA,
    LLAMA_CPP_EMBEDDING_ADAPTER,
    EmbeddingGatewayProfile,
    EmbeddingProfileCatalog,
    EmbeddingSpaceIdentity,
    text_policy_digest,
)
from astrumweaver.gateway.embedding_api import create_embedding_router
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
QUERY_PREFIX = "Instruct: retrieve relevant notes\nQuery: "


def digest(ch: str) -> str:
    return "sha256:" + ch * 64


def serving_values(*, timeout: float = 1.0):
    deployment = DeploymentIdentity(
        provider_id="llama-cpp",
        runtime_artifact_sha256=digest("1"),
        adapter_artifact_sha256=digest("2"),
        model_artifact_sha256=digest("3"),
        execution_config_sha256=digest("4"),
        quantization="Q8_0",
        tokenizer_artifact_sha256=digest("5"),
    )
    space = EmbeddingSpaceIdentity(
        deployment_revision=deployment.revision,
        model_artifact_sha256=deployment.model_artifact_sha256,
        quantization=deployment.quantization,
        tokenizer_artifact_sha256=deployment.tokenizer_artifact_sha256 or "",
        pooling="last",
        normalization="l2",
        dimensions=3,
        query_policy_id="notes-query-v1",
        query_preprocess_sha256=text_policy_digest(QUERY_PREFIX),
        document_policy_id="notes-document-v1",
        document_preprocess_sha256=text_policy_digest(""),
        adapter_id=LLAMA_CPP_EMBEDDING_ADAPTER,
    )
    features = frozenset(
        {"float", "pooling-last", "normalization-l2"}
    )
    contract = ServingContract(
        deployment_revision=deployment.revision,
        capability="text.embed",
        operation_schema=EMBEDDING_OPERATION_SCHEMA,
        validation_evidence_sha256=digest("6"),
        features=features,
        limits={
            "item_tokens": 64,
            "item_bytes": 1024,
            "batch_items": 4,
            "batch_bytes": 2048,
            "aggregate_tokens": 128,
            "request_bytes": 4096,
        },
        semantic_revision=space.embedding_space_id,
    )
    logical = LogicalServingProfile(
        profile_id="notes-embed-v1",
        deployment_revision=deployment.revision,
        serving_contract_revision=contract.revision,
        capability="text.embed",
        operation_schema=EMBEDDING_OPERATION_SCHEMA,
        required_features=features,
        limits={},
    )
    resolved = resolve_profile(logical, contract, deployment)
    profile = EmbeddingGatewayProfile(
        resolved=resolved,
        space=space,
        adapter_id=LLAMA_CPP_EMBEDDING_ADAPTER,
        query_prefix=QUERY_PREFIX,
        query_suffix="",
        document_prefix="",
        document_suffix="",
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
                worker_id="embed-worker",
                worker_class="cpu-test",
                resources=ResourceShape(),
                capabilities=frozenset({"text.embed"}),
            ),
            serving=advertisement,
        )
    )


def app_with_gateway(
    repository: InMemoryControlRepository,
    profile: EmbeddingGatewayProfile,
):
    app = create_app(
        repository,
        client_token=CLIENT_TOKEN,
        worker_token=WORKER_TOKEN,
        maintenance_interval_seconds=60,
    )
    app.include_router(
        create_embedding_router(
            repository,
            EmbeddingProfileCatalog((profile,)),
            client_auth="bearer",
            client_token=CLIENT_TOKEN,
            poll_interval_seconds=0.005,
        )
    )
    return app


async def complete_next(
    repository: InMemoryControlRepository,
    outputs: dict,
):
    deadline = asyncio.get_running_loop().time() + 1
    while True:
        claimed = repository.claim_next_job(
            "embed-worker",
            runtime_instance_epoch=EPOCH,
        )
        if claimed is not None:
            break
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(
                "gateway did not submit a claimable embedding job"
            )
        await asyncio.sleep(0.001)

    repository.complete_job(
        claimed.job_id,
        JobResult(outputs=outputs),
        worker_id="embed-worker",
        lease_token=claimed.lease_token or "",
        runtime_instance_epoch=EPOCH,
    )
    return claimed


def provider_result(count: int = 2, dimensions: int = 3):
    value = 1.0 / math.sqrt(dimensions)
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [value] * dimensions,
                "index": index,
            }
            for index in range(count)
        ],
        "model": "astrumweaver",
        "usage": {"prompt_tokens": 6, "total_tokens": 6},
    }


@pytest.mark.asyncio
async def test_embedding_spaces_expose_namespaced_immutable_contract():
    repository = InMemoryControlRepository()
    profile, _ = serving_values()
    app = app_with_gateway(repository, profile)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://control",
    ) as client:
        unauthorized = await client.get("/v1/embedding-spaces")
        response = await client.get(
            "/v1/embedding-spaces",
            headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
        )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    item = response.json()["data"][0]
    assert item["id"] == "notes-embed-v1"
    assert item["object"] == "astrumweaver.embedding_space"
    extension = item["x_astrumweaver"]
    assert extension["embedding_space_id"] == profile.embedding_space_id
    assert extension["dimensions"] == 3
    assert extension["pooling"] == "last"
    assert extension["normalization"] == "l2"
    assert extension["input_types"] == ["query", "document"]


@pytest.mark.asyncio
async def test_embedding_request_uses_durable_binding_and_preserves_order():
    repository = InMemoryControlRepository()
    profile, advertisement = serving_values()
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, profile)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://control",
    ) as client:
        request = asyncio.create_task(
            client.post(
                "/v1/embeddings",
                headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
                json={
                    "model": "notes-embed-v1",
                    "input": ["alpha", "beta"],
                    "encoding_format": "float",
                    "x_astrumweaver_input_type": "query",
                    "x_astrumweaver_embedding_space_id": profile.embedding_space_id,
                },
            )
        )
        claimed = await complete_next(repository, provider_result())
        response = await request

    assert response.status_code == 200
    body = response.json()
    assert [item["index"] for item in body["data"]] == [0, 1]
    assert body["model"] == "notes-embed-v1"
    assert body["x_astrumweaver"]["embedding_space_id"] == (
        profile.embedding_space_id
    )
    assert body["x_astrumweaver"]["input_type"] == "query"

    payload = dict(claimed.payload)
    assert payload["embedding_space_id"] == profile.embedding_space_id
    assert payload["request"]["input"] == [
        QUERY_PREFIX + "alpha",
        QUERY_PREFIX + "beta",
    ]
    assert claimed.serving == profile.binding


@pytest.mark.asyncio
async def test_embedding_gateway_rejects_unsupported_shape_before_admission():
    repository = InMemoryControlRepository()
    profile, advertisement = serving_values()
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, profile)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://control",
    ) as client:
        response = await client.post(
            "/v1/embeddings",
            headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
            json={
                "model": "notes-embed-v1",
                "input": [[1, 2, 3]],
                "x_astrumweaver_input_type": "document",
                "x_astrumweaver_embedding_space_id": profile.embedding_space_id,
            },
        )
        dimensions = await client.post(
            "/v1/embeddings",
            headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
            json={
                "model": "notes-embed-v1",
                "input": "alpha",
                "dimensions": 2,
                "x_astrumweaver_input_type": "document",
                "x_astrumweaver_embedding_space_id": profile.embedding_space_id,
            },
        )

    assert response.status_code == 400
    assert dimensions.status_code == 400
    assert repository.list_jobs() == []


@pytest.mark.asyncio
async def test_embedding_gateway_rejects_invalid_provider_vector_batch():
    repository = InMemoryControlRepository()
    profile, advertisement = serving_values()
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, profile)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://control",
    ) as client:
        request = asyncio.create_task(
            client.post(
                "/v1/embeddings",
                headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
                json={
                    "model": "notes-embed-v1",
                    "input": ["alpha", "beta"],
                    "x_astrumweaver_input_type": "document",
                    "x_astrumweaver_embedding_space_id": profile.embedding_space_id,
                },
            )
        )
        await complete_next(
            repository,
            provider_result(dimensions=2),
        )
        response = await request

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_provider_response"


@pytest.mark.asyncio
async def test_embedding_gateway_timeout_cancels_only_owned_job():
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
            "/v1/embeddings",
            headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
            json={
                "model": "notes-embed-v1",
                "input": "alpha",
                "x_astrumweaver_input_type": "document",
                "x_astrumweaver_embedding_space_id": profile.embedding_space_id,
            },
        )

    assert response.status_code == 504
    jobs = repository.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].status.value == "cancelled"


@pytest.mark.asyncio
async def test_embedding_gateway_rejects_incompatible_space_before_admission():
    repository = InMemoryControlRepository()
    profile, advertisement = serving_values()
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, profile)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://control",
    ) as client:
        response = await client.post(
            "/v1/embeddings",
            headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
            json={
                "model": profile.profile_id,
                "input": "alpha",
                "x_astrumweaver_input_type": "document",
                "x_astrumweaver_embedding_space_id": digest("f"),
            },
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "embedding_space_mismatch"
    assert repository.list_jobs() == []
