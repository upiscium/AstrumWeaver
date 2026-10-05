from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta

import httpx
import pytest

from astrumweaver import JobResult, ResourceShape, WorkerSpec
from astrumweaver.control import (
    InMemoryControlRepository,
    JobStatus,
    WorkerRegistration,
    utc_now,
)
from astrumweaver.control.api import create_app
from astrumweaver.gateway.api import create_chat_router
from astrumweaver.gateway.chat import (
    CHAT_OPERATION_SCHEMA,
    LLAMA_CPP_CHAT_ADAPTER,
    ChatGatewayProfile,
    ChatProfileCatalog,
)
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


def digest(character: str) -> str:
    return "sha256:" + character * 64


def serving_values(*, tools: bool = True, timeout: float = 1.0):
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
        operation_schema=CHAT_OPERATION_SCHEMA,
        validation_evidence_sha256=digest("7"),
        features=frozenset({"tools"} if tools else set()),
        limits={
            "input_tokens": 4096,
            "output_tokens": 1024,
            "total_tokens": 5120,
            "request_bytes": 65536,
        },
    )
    logical = LogicalServingProfile(
        profile_id="local-code-v1",
        deployment_revision=deployment.revision,
        serving_contract_revision=contract.revision,
        capability="llm.chat",
        operation_schema=CHAT_OPERATION_SCHEMA,
        required_features=frozenset(),
        limits={},
    )
    resolved = resolve_profile(logical, contract, deployment)
    gateway = ChatGatewayProfile(
        resolved=resolved,
        adapter_id=LLAMA_CPP_CHAT_ADAPTER,
        request_timeout_seconds=timeout,
        max_attempts=2,
    )
    advertisement = WorkerServingAdvertisement(
        deployment_revision=deployment.revision,
        runtime_instance=RuntimeInstance(deployment.revision, EPOCH),
        contracts=(contract,),
    )
    return gateway, advertisement


def register_worker(
    repository: InMemoryControlRepository,
    advertisement: WorkerServingAdvertisement,
) -> None:
    repository.register_worker(
        WorkerRegistration(
            spec=WorkerSpec(
                worker_id="chat-worker",
                worker_class="cpu-test",
                resources=ResourceShape(),
                capabilities=frozenset({"llm.chat"}),
            ),
            serving=advertisement,
        )
    )


def app_with_gateway(
    repository: InMemoryControlRepository,
    gateway: ChatGatewayProfile,
):
    app = create_app(
        repository,
        client_token=CLIENT_TOKEN,
        worker_token=WORKER_TOKEN,
        maintenance_interval_seconds=60,
    )
    app.include_router(
        create_chat_router(
            repository,
            ChatProfileCatalog((gateway,)),
            client_auth="bearer",
            client_token=CLIENT_TOKEN,
            poll_interval_seconds=0.005,
        )
    )
    return app


async def complete_next(
    repository: InMemoryControlRepository,
    outputs: dict,
) -> None:
    deadline = asyncio.get_running_loop().time() + 1
    while True:
        claimed = repository.claim_next_job(
            "chat-worker",
            runtime_instance_epoch=EPOCH,
        )
        if claimed is not None:
            break
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("gateway did not submit a claimable chat job")
        await asyncio.sleep(0.001)

    repository.complete_job(
        claimed.job_id,
        JobResult(outputs=outputs),
        worker_id="chat-worker",
        lease_token=claimed.lease_token or "",
        runtime_instance_epoch=EPOCH,
    )


@pytest.mark.asyncio
async def test_models_expose_configured_profile_not_runtime_inventory():
    repository = InMemoryControlRepository()
    gateway, _ = serving_values()
    app = app_with_gateway(repository, gateway)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        unauthorized = await client.get("/v1/models")
        response = await client.get(
            "/v1/models",
            headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
        )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "local-code-v1",
                "object": "model",
                "created": 0,
                "owned_by": "astrumweaver",
            }
        ],
    }


@pytest.mark.asyncio
async def test_plain_chat_uses_durable_job_and_normalizes_response():
    repository = InMemoryControlRepository()
    gateway, advertisement = serving_values()
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, gateway)
    transport = httpx.ASGITransport(app=app)

    worker = asyncio.create_task(
        complete_next(
            repository,
            {
                "id": "provider-chat",
                "created": 12,
                "model": "runtime-alias",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 1,
                    "total_tokens": 5,
                },
            },
        )
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
            json={
                "model": "local-code-v1",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 32,
            },
        )
    await worker

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "provider-chat"
    assert body["model"] == "local-code-v1"
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["usage"]["total_tokens"] == 5

    records = repository.list_jobs()
    assert len(records) == 1
    assert records[0].status is JobStatus.SUCCEEDED
    assert records[0].serving == gateway.binding
    assert records[0].payload["schema_version"] == "chat-job-v1"
    assert records[0].payload["adapter_id"] == "llama-cpp-chat-v1"


@pytest.mark.asyncio
async def test_tool_request_result_and_final_response_preserve_structure():
    repository = InMemoryControlRepository()
    gateway, advertisement = serving_values()
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, gateway)
    transport = httpx.ASGITransport(app=app)
    headers = {"authorization": f"Bearer {CLIENT_TOKEN}"}
    arguments = '{"path":"README.md"}'

    first_worker = asyncio.create_task(
        complete_next(
            repository,
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_read",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": arguments,
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        first = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "local-code-v1",
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        )
        await first_worker
        tool_call = first.json()["choices"][0]["message"]["tool_calls"][0]
        assert tool_call["id"] == "call_read"
        assert tool_call["function"]["arguments"] == arguments

        second_worker = asyncio.create_task(
            complete_next(
                repository,
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The file is a README.",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        )
        second = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "local-code-v1",
                "messages": [
                    {"role": "user", "content": "read the file"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_read",
                        "content": "# Project",
                    },
                    {"role": "user", "content": "summarize"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        )
        await second_worker

    assert second.status_code == 200
    assert (
        second.json()["choices"][0]["message"]["content"]
        == "The file is a README."
    )
    jobs = repository.list_jobs()
    assert len(jobs) == 2
    second_messages = jobs[1].payload["request"]["messages"]
    assert second_messages[1]["tool_calls"][0]["id"] == "call_read"
    assert second_messages[2]["tool_call_id"] == "call_read"


@pytest.mark.asyncio
async def test_invalid_streaming_and_missing_tool_feature_fail_before_enqueue():
    repository = InMemoryControlRepository()
    gateway, advertisement = serving_values(tools=False)
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, gateway)
    transport = httpx.ASGITransport(app=app)
    headers = {"authorization": f"Bearer {CLIENT_TOKEN}"}

    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        streaming = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "local-code-v1",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        tools = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "local-code-v1",
                "messages": [{"role": "user", "content": "use tool"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "inspect", "parameters": {}},
                    }
                ],
            },
        )

    assert streaming.status_code == 400
    assert streaming.json()["error"]["code"] == "streaming_not_supported"
    assert tools.status_code == 400
    assert tools.json()["error"]["code"] == "unsupported_feature"
    assert repository.list_jobs() == []


@pytest.mark.asyncio
async def test_no_compatible_deployment_and_overload_are_explicit():
    repository = InMemoryControlRepository()
    gateway, advertisement = serving_values()
    app = app_with_gateway(repository, gateway)
    transport = httpx.ASGITransport(app=app)
    headers = {"authorization": f"Bearer {CLIENT_TOKEN}"}
    request = {
        "model": "local-code-v1",
        "messages": [{"role": "user", "content": "hi"}],
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        missing = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json=request,
        )
        assert missing.status_code == 503
        assert missing.json()["error"]["code"] == "no_compatible_deployment"

        register_worker(repository, advertisement)
        occupied = repository.submit_job(
            __import__("astrumweaver.control", fromlist=["JobSubmission"]).JobSubmission(
                capability="llm.chat",
                payload={"direct": True},
                deadline_at=utc_now() + timedelta(seconds=10),
                serving=gateway.binding,
            )
        )
        claim = repository.claim_next_job(
            "chat-worker",
            runtime_instance_epoch=EPOCH,
        )
        assert claim is not None and claim.job_id == occupied.job_id

        overloaded = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json=request,
        )

    assert overloaded.status_code == 429
    assert overloaded.json()["error"]["code"] == "overloaded"


@pytest.mark.asyncio
async def test_gateway_timeout_cancels_only_owned_job():
    repository = InMemoryControlRepository()
    gateway, advertisement = serving_values(timeout=0.03)
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, gateway)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
            json={
                "model": "local-code-v1",
                "messages": [{"role": "user", "content": "wait"}],
            },
        )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "gateway_timeout"
    jobs = repository.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].status is JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_known_nonretryable_executor_failure_maps_to_client_error():
    repository = InMemoryControlRepository()
    gateway, advertisement = serving_values()
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, gateway)
    transport = httpx.ASGITransport(app=app)

    async def fail_next():
        deadline = asyncio.get_running_loop().time() + 1
        while True:
            claimed = repository.claim_next_job(
                "chat-worker",
                runtime_instance_epoch=EPOCH,
            )
            if claimed is not None:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("job not submitted")
            await asyncio.sleep(0.001)
        repository.fail_job(
            claimed.job_id,
            {
                "type": "JobExecutionError",
                "code": "context_length_exceeded",
                "message": "chat input exceeds the deployed input-token limit",
            },
            retryable=False,
            worker_id="chat-worker",
            lease_token=claimed.lease_token or "",
            runtime_instance_epoch=EPOCH,
        )

    worker = asyncio.create_task(fail_next())
    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
            json={
                "model": "local-code-v1",
                "messages": [{"role": "user", "content": "too large"}],
            },
        )
    await worker

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"
