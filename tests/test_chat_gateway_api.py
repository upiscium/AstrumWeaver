from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import timedelta

import httpx
import pytest

from astrumweaver import JobResult, ResourceShape, WorkerSpec
from astrumweaver.control import (
    InMemoryControlRepository,
    JobStatus,
    JobSubmission,
    WorkerRegistration,
    utc_now,
)
from astrumweaver.control.api import create_app
from astrumweaver.gateway.api import ChatGatewayService, create_chat_router
from astrumweaver.gateway.chat import (
    CHAT_OPERATION_SCHEMA,
    LLAMA_CPP_CHAT_ADAPTER,
    ChatGatewayError,
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


async def stream_next(
    repository: InMemoryControlRepository,
    chunks: list[dict],
    *,
    fail: bool = False,
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

    for chunk in chunks:
        repository.append_job_event(
            claimed.job_id,
            worker_id="chat-worker",
            lease_token=claimed.lease_token or "",
            runtime_instance_epoch=EPOCH,
            kind="chat.completion.chunk",
            payload=chunk,
        )
        await asyncio.sleep(0)

    if fail:
        repository.fail_job(
            claimed.job_id,
            {"code": "provider_stream_failed", "message": "provider failed"},
            retryable=False,
            worker_id="chat-worker",
            lease_token=claimed.lease_token or "",
            runtime_instance_epoch=EPOCH,
        )
        return

    repository.complete_job(
        claimed.job_id,
        JobResult(outputs={"streamed": True}),
        worker_id="chat-worker",
        lease_token=claimed.lease_token or "",
        runtime_instance_epoch=EPOCH,
    )


def sse_data(response: httpx.Response) -> list[str]:
    return [
        frame[len("data: "):]
        for frame in response.text.split("\n\n")
        if frame.startswith("data: ")
    ]


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
async def test_missing_tool_feature_fails_before_enqueue():
    repository = InMemoryControlRepository()
    gateway, advertisement = serving_values(tools=False)
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, gateway)
    transport = httpx.ASGITransport(app=app)
    headers = {"authorization": f"Bearer {CLIENT_TOKEN}"}

    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
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
            JobSubmission(
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



@pytest.mark.asyncio
async def test_gateway_disconnect_cancels_only_owned_job():
    repository = InMemoryControlRepository()
    gateway, advertisement = serving_values(timeout=1.0)
    register_worker(repository, advertisement)
    service = ChatGatewayService(
        repository,
        ChatProfileCatalog((gateway,)),
        poll_interval_seconds=0.001,
    )

    class DisconnectedRequest:
        async def is_disconnected(self) -> bool:
            return True

    with pytest.raises(ChatGatewayError) as error:
        await service.complete(
            DisconnectedRequest(),
            {
                "model": "local-code-v1",
                "messages": [{"role": "user", "content": "disconnect"}],
            },
            request_size_bytes=64,
        )

    assert error.value.code == "client_disconnected"
    assert error.value.status_code == 499
    jobs = repository.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].status is JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_streaming_chat_emits_ordered_sse_and_done_after_success():
    repository = InMemoryControlRepository()
    gateway, advertisement = serving_values()
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, gateway)
    transport = httpx.ASGITransport(app=app)
    chunks = [
        {
            "id": "provider-stream",
            "object": "chat.completion.chunk",
            "created": 12,
            "model": "runtime-alias",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "hel"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "provider-stream",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "lo"},
                    "finish_reason": "stop",
                }
            ],
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
        },
    ]
    worker = asyncio.create_task(stream_next(repository, chunks))

    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
            json={
                "model": "local-code-v1",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    await worker

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = sse_data(response)
    assert frames[-1] == "[DONE]"
    parsed = [json.loads(frame) for frame in frames[:-1]]
    assert [item["model"] for item in parsed] == ["local-code-v1"] * 3
    assert "runtime-alias" not in response.text
    assert parsed[0]["choices"][0]["delta"]["content"] == "hel"
    assert parsed[1]["choices"][0]["delta"]["content"] == "lo"
    assert parsed[1]["choices"][0]["finish_reason"] == "stop"
    assert parsed[2]["choices"] == []
    assert parsed[2]["usage"]["total_tokens"] == 6
    jobs = repository.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].max_attempts == 1
    assert jobs[0].status is JobStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_streaming_tool_call_fragments_are_not_replayed_or_flattened():
    repository = InMemoryControlRepository()
    gateway, advertisement = serving_values(tools=True)
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, gateway)
    transport = httpx.ASGITransport(app=app)
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_read",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": "{\"path\":",
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": "\"README.md\"}",
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    worker = asyncio.create_task(stream_next(repository, chunks))

    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
            json={
                "model": "local-code-v1",
                "messages": [{"role": "user", "content": "read"}],
                "stream": True,
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
    await worker

    frames = sse_data(response)
    assert frames[-1] == "[DONE]"
    parsed = [json.loads(frame) for frame in frames[:-1]]
    assert len(parsed) == 2
    first = parsed[0]["choices"][0]["delta"]["tool_calls"][0]
    second = parsed[1]["choices"][0]["delta"]["tool_calls"][0]
    assert first["id"] == "call_read"
    assert first["function"]["name"] == "read_file"
    assert first["function"]["arguments"] == "{\"path\":"
    assert second["function"]["arguments"] == "\"README.md\"}"
    assert parsed[1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_partial_stream_failure_has_no_done_and_no_retry():
    repository = InMemoryControlRepository()
    gateway, advertisement = serving_values()
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, gateway)
    transport = httpx.ASGITransport(app=app)
    worker = asyncio.create_task(
        stream_next(
            repository,
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "partial"},
                            "finish_reason": None,
                        }
                    ]
                }
            ],
            fail=True,
        )
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {CLIENT_TOKEN}"},
            json={
                "model": "local-code-v1",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    await worker

    frames = sse_data(response)
    assert "[DONE]" not in frames
    assert json.loads(frames[0])["choices"][0]["delta"]["content"] == "partial"
    error = json.loads(frames[-1])["error"]
    assert error["code"] == "backend_failure"
    record = repository.list_jobs()[0]
    assert record.max_attempts == 1
    assert record.attempts == 1
    assert record.status is JobStatus.FAILED



@pytest.mark.asyncio
async def test_streaming_tool_round_trip_matches_pinned_opencode_wire_shape():
    repository = InMemoryControlRepository()
    gateway, advertisement = serving_values(tools=True)
    register_worker(repository, advertisement)
    app = app_with_gateway(repository, gateway)
    transport = httpx.ASGITransport(app=app)
    headers = {"authorization": f"Bearer {CLIENT_TOKEN}"}
    tool_chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_read",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": "{\"path\":",
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": "\"README.md\"}",
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "total_tokens": 12,
            },
        },
    ]
    first_worker = asyncio.create_task(stream_next(repository, tool_chunks))

    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
        first = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "local-code-v1",
                "messages": [{"role": "user", "content": "read README"}],
                "stream": True,
                "stream_options": {"include_usage": True},
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

        first_chunks = [
            json.loads(frame)
            for frame in sse_data(first)
            if frame != "[DONE]"
        ]
        tool_fragments = [
            item
            for chunk in first_chunks
            for choice in chunk.get("choices", [])
            for item in choice.get("delta", {}).get("tool_calls", [])
        ]
        opening = next(
            item
            for item in tool_fragments
            if item.get("id") is not None
        )
        call_id = opening["id"]
        call_name = opening["function"]["name"]
        arguments = "".join(
            item.get("function", {}).get("arguments", "")
            for item in tool_fragments
        )
        assert call_id == "call_read"
        assert call_name == "read_file"
        assert arguments == '{"path":"README.md"}'

        final_chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "README summarized."},
                        "finish_reason": "stop",
                    }
                ]
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 16,
                    "completion_tokens": 3,
                    "total_tokens": 19,
                },
            },
        ]
        second_worker = asyncio.create_task(stream_next(repository, final_chunks))
        second = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "local-code-v1",
                "messages": [
                    {"role": "user", "content": "read README"},
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "I need the file contents.",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": call_name,
                                    "arguments": arguments,
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": "# README",
                    },
                ],
                "stream": True,
                "stream_options": {"include_usage": True},
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

    second_frames = sse_data(second)
    assert second_frames[-1] == "[DONE]"
    final = json.loads(second_frames[0])
    assert final["choices"][0]["delta"]["content"] == "README summarized."
    jobs = repository.list_jobs()
    assert len(jobs) == 2
    assert all(job.max_attempts == 1 for job in jobs)
    history = jobs[1].payload["request"]["messages"]
    assert history[1]["reasoning_content"] == "I need the file contents."
    assert history[1]["tool_calls"][0]["id"] == "call_read"
    assert history[2]["tool_call_id"] == "call_read"


class _DisconnectedRequest:
    async def is_disconnected(self) -> bool:
        return True


class _ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_streaming_disconnect_cancels_owned_job_before_output():
    repository = InMemoryControlRepository()
    gateway, advertisement = serving_values()
    register_worker(repository, advertisement)
    service = ChatGatewayService(
        repository,
        ChatProfileCatalog((gateway,)),
        poll_interval_seconds=0.001,
    )
    iterator = await service.stream(
        _DisconnectedRequest(),
        {
            "model": "local-code-v1",
            "messages": [{"role": "user", "content": "disconnect"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        request_size_bytes=128,
    )
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)

    record = repository.list_jobs()[0]
    assert record.max_attempts == 1
    assert record.status is JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_streaming_caller_task_cancellation_cancels_owned_job():
    repository = InMemoryControlRepository()
    gateway, advertisement = serving_values()
    register_worker(repository, advertisement)
    service = ChatGatewayService(
        repository,
        ChatProfileCatalog((gateway,)),
        poll_interval_seconds=0.01,
    )
    iterator = await service.stream(
        _ConnectedRequest(),
        {
            "model": "local-code-v1",
            "messages": [{"role": "user", "content": "cancel"}],
            "stream": True,
        },
        request_size_bytes=128,
    )
    task = asyncio.create_task(anext(iterator))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert repository.list_jobs()[0].status is JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_streaming_timeout_emits_error_without_done_and_cancels_job():
    repository = InMemoryControlRepository()
    gateway, advertisement = serving_values(timeout=0.01)
    register_worker(repository, advertisement)
    service = ChatGatewayService(
        repository,
        ChatProfileCatalog((gateway,)),
        poll_interval_seconds=0.001,
    )
    iterator = await service.stream(
        _ConnectedRequest(),
        {
            "model": "local-code-v1",
            "messages": [{"role": "user", "content": "timeout"}],
            "stream": True,
        },
        request_size_bytes=128,
    )
    frames = [frame async for frame in iterator]

    assert all("[DONE]" not in frame for frame in frames)
    payload = json.loads(frames[-1][len("data: "):].strip())
    assert payload["error"]["code"] == "gateway_timeout"
    assert repository.list_jobs()[0].status is JobStatus.CANCELLED
