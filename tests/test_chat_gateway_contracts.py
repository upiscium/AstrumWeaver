from __future__ import annotations

from dataclasses import replace

import pytest

from astrumweaver.gateway.chat import (
    CHAT_CATALOG_SCHEMA,
    CHAT_OPERATION_SCHEMA,
    LLAMA_CPP_CHAT_ADAPTER,
    ChatGatewayError,
    ChatGatewayProfile,
    ChatProfileCatalog,
    compile_chat_request,
    normalize_chat_completion,
)
from astrumweaver.serving import (
    DeploymentIdentity,
    LogicalServingProfile,
    ServingContract,
    resolve_profile,
)


def digest(character: str) -> str:
    return "sha256:" + character * 64


def profile(*, tools: bool = True) -> ChatGatewayProfile:
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
            "input_tokens": 8192,
            "output_tokens": 2048,
            "total_tokens": 10240,
            "request_bytes": 131072,
        },
    )
    serving_profile = LogicalServingProfile(
        profile_id="local-code-v1",
        deployment_revision=deployment.revision,
        serving_contract_revision=contract.revision,
        capability="llm.chat",
        operation_schema=CHAT_OPERATION_SCHEMA,
        required_features=frozenset(),
        limits={
            "input_tokens": 4096,
            "output_tokens": 1024,
            "total_tokens": 5120,
            "request_bytes": 65536,
        },
    )
    return ChatGatewayProfile(
        resolved=resolve_profile(serving_profile, contract, deployment),
        adapter_id=LLAMA_CPP_CHAT_ADAPTER,
        request_timeout_seconds=30,
        max_attempts=2,
        created=1,
    )


def manifest(value: ChatGatewayProfile) -> dict:
    return {
        "schema_version": CHAT_CATALOG_SCHEMA,
        "profiles": [
            {
                "adapter_id": value.adapter_id,
                "request_timeout_seconds": value.request_timeout_seconds,
                "max_attempts": value.max_attempts,
                "owned_by": value.owned_by,
                "created": value.created,
                "deployment": value.resolved.deployment.to_dict(),
                "contract": value.resolved.contract.to_dict(),
                "profile": value.resolved.profile.to_dict(),
            }
        ],
    }


def test_catalog_round_trip_and_models_expose_only_profile_ids():
    original = profile()
    catalog = ChatProfileCatalog.from_dict(manifest(original))

    loaded = catalog.get("local-code-v1")
    assert loaded.resolved == original.resolved
    assert loaded.binding == original.binding
    assert catalog.models_response() == {
        "object": "list",
        "data": [
            {
                "id": "local-code-v1",
                "object": "model",
                "created": 1,
                "owned_by": "astrumweaver",
            }
        ],
    }
    with pytest.raises(ChatGatewayError) as error:
        catalog.get("runtime-model-name")
    assert error.value.code == "model_not_found"
    assert error.value.status_code == 404


def test_catalog_rejects_wrong_operation_and_unbounded_profile():
    value = profile()
    wrong_contract = replace(
        value.resolved.contract,
        operation_schema="other-chat-v1",
    )
    wrong_profile = replace(
        value.resolved.profile,
        serving_contract_revision=wrong_contract.revision,
        operation_schema="other-chat-v1",
    )
    wrong_resolved = resolve_profile(
        wrong_profile,
        wrong_contract,
        value.resolved.deployment,
    )
    with pytest.raises(ChatGatewayError, match="operation schema"):
        ChatGatewayProfile(
            resolved=wrong_resolved,
            adapter_id=LLAMA_CPP_CHAT_ADAPTER,
            request_timeout_seconds=30,
        )

    too_wide_contract = replace(
        value.resolved.contract,
        limits={
            "input_tokens": 6000,
            "output_tokens": 1024,
            "total_tokens": 5120,
            "request_bytes": 65536,
        },
    )
    too_wide_profile = replace(
        value.resolved.profile,
        serving_contract_revision=too_wide_contract.revision,
        limits={
            "input_tokens": 6000,
            "output_tokens": 1024,
            "total_tokens": 5120,
            "request_bytes": 65536,
        },
    )
    too_wide = resolve_profile(
        too_wide_profile,
        too_wide_contract,
        value.resolved.deployment,
    )
    with pytest.raises(ChatGatewayError, match="total context"):
        ChatGatewayProfile(
            resolved=too_wide,
            adapter_id=LLAMA_CPP_CHAT_ADAPTER,
            request_timeout_seconds=30,
        )


def test_plain_chat_compiles_to_opaque_llama_cpp_job_envelope():
    value = profile()
    compiled = compile_chat_request(
        {
            "model": "local-code-v1",
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.9,
            "seed": 7,
            "stream": False,
            "n": 1,
        },
        value,
        request_size_bytes=512,
    )

    payload = dict(compiled.payload)
    assert payload["schema_version"] == "chat-job-v1"
    assert payload["adapter_id"] == "llama-cpp-chat-v1"
    request = payload["request"]
    assert request["messages"] == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Hello"},
    ]
    assert request["max_tokens"] == 512
    assert request["stream"] is False
    assert "model" not in request
    assert payload["limits"] == {
        "input_tokens": 4096,
        "output_tokens": 1024,
        "total_tokens": 5120,
    }


def test_tool_round_trip_preserves_call_ids_and_argument_json_strings():
    value = profile()
    arguments = '{"path":"src/main.cpp","line":7}'
    compiled = compile_chat_request(
        {
            "model": "local-code-v1",
            "messages": [
                {"role": "user", "content": "Inspect the file"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "int main() {}",
                },
                {"role": "user", "content": "Now answer"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                        "strict": True,
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "read_file"},
            },
        },
        value,
        request_size_bytes=2048,
    )

    request = compiled.payload["request"]
    assert request["messages"][1]["tool_calls"][0]["id"] == "call_1"
    assert (
        request["messages"][1]["tool_calls"][0]["function"]["arguments"]
        == arguments
    )
    assert request["messages"][2]["tool_call_id"] == "call_1"
    assert request["tool_choice"]["function"]["name"] == "read_file"


@pytest.mark.parametrize(
    ("patch", "code"),
    [
        ({"stream": "yes"}, "invalid_request"),
        ({"n": 2}, "unsupported_field"),
        ({"response_format": {"type": "json_object"}}, "unsupported_field"),
        ({"max_tokens": 2048}, "output_limit_exceeded"),
    ],
)
def test_request_subset_fails_closed(patch, code):
    request = {
        "model": "local-code-v1",
        "messages": [{"role": "user", "content": "Hello"}],
        **patch,
    }
    with pytest.raises(ChatGatewayError) as error:
        compile_chat_request(request, profile(), request_size_bytes=256)
    assert error.value.code == code


def test_streaming_request_compiles_to_same_bounded_job_envelope():
    compiled = compile_chat_request(
        {
            "model": "local-code-v1",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "max_tokens": 32,
        },
        profile(),
        request_size_bytes=256,
    )
    assert compiled.payload["request"]["stream"] is True
    assert compiled.payload["request"]["max_tokens"] == 32
    assert compiled.payload["limits"]["total_tokens"] == 5120


def test_request_byte_bound_and_missing_tool_feature_fail_closed():
    with pytest.raises(ChatGatewayError) as error:
        compile_chat_request(
            {
                "model": "local-code-v1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            profile(),
            request_size_bytes=65537,
        )
    assert error.value.code == "request_too_large"
    assert error.value.status_code == 413

    with pytest.raises(ChatGatewayError) as error:
        compile_chat_request(
            {
                "model": "local-code-v1",
                "messages": [{"role": "user", "content": "Use a tool"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "inspect", "parameters": {}},
                    }
                ],
                "tool_choice": "auto",
            },
            profile(tools=False),
            request_size_bytes=512,
        )
    assert error.value.code == "unsupported_feature"


def test_tool_history_rejects_missing_unknown_or_duplicate_results():
    value = profile()
    with pytest.raises(ChatGatewayError, match="pending tool"):
        compile_chat_request(
            {
                "model": value.profile_id,
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "inspect",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    {"role": "user", "content": "continue"},
                ],
            },
            value,
            request_size_bytes=512,
        )

    with pytest.raises(ChatGatewayError, match="pending tool call"):
        compile_chat_request(
            {
                "model": value.profile_id,
                "messages": [
                    {"role": "user", "content": "hello"},
                    {
                        "role": "tool",
                        "tool_call_id": "missing",
                        "content": "value",
                    },
                ],
            },
            value,
            request_size_bytes=512,
        )

    with pytest.raises(ChatGatewayError, match="unique"):
        compile_chat_request(
            {
                "model": value.profile_id,
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "same",
                                "type": "function",
                                "function": {"name": "a", "arguments": "{}"},
                            },
                            {
                                "id": "same",
                                "type": "function",
                                "function": {"name": "b", "arguments": "{}"},
                            },
                        ],
                    }
                ],
            },
            value,
            request_size_bytes=512,
        )


def test_normalize_plain_response_preserves_usage_and_logical_model():
    response = normalize_chat_completion(
        {
            "id": "provider-id",
            "object": "chat.completion",
            "created": 123,
            "model": "runtime-alias",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "provider_extra": 99,
            },
        },
        profile_id="local-code-v1",
        job_id="job-id",
        created=999,
    )

    assert response["id"] == "provider-id"
    assert response["model"] == "local-code-v1"
    assert response["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Hello",
    }
    assert response["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }


def test_normalize_tool_response_preserves_exact_argument_string():
    arguments = '{"path":"README.md"}'
    response = normalize_chat_completion(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_7",
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
        profile_id="local-code-v1",
        job_id="job-id",
        created=123,
    )
    call = response["choices"][0]["message"]["tool_calls"][0]
    assert call["id"] == "call_7"
    assert call["function"]["arguments"] == arguments


def test_normalize_rejects_invalid_provider_shape():
    with pytest.raises(ChatGatewayError) as error:
        normalize_chat_completion(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            profile_id="local-code-v1",
            job_id="job",
            created=1,
        )
    assert error.value.code == "invalid_provider_response"
    assert error.value.status_code == 502



def test_parallel_tool_calls_false_is_preserved_but_true_is_unverified():
    value = profile()
    compiled = compile_chat_request(
        {
            "model": value.profile_id,
            "messages": [{"role": "user", "content": "inspect"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "inspect", "parameters": {}},
                }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        },
        value,
        request_size_bytes=512,
    )
    assert compiled.payload["request"]["parallel_tool_calls"] is False

    with pytest.raises(ChatGatewayError) as error:
        compile_chat_request(
            {
                "model": value.profile_id,
                "messages": [{"role": "user", "content": "inspect"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "inspect", "parameters": {}},
                    }
                ],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            },
            value,
            request_size_bytes=512,
        )
    assert error.value.code == "unsupported_feature"


def test_tools_profile_requires_template_artifact_identity():
    value = profile()
    deployment = replace(
        value.resolved.deployment,
        template_artifact_sha256=None,
    )
    contract = replace(
        value.resolved.contract,
        deployment_revision=deployment.revision,
    )
    logical = replace(
        value.resolved.profile,
        deployment_revision=deployment.revision,
        serving_contract_revision=contract.revision,
    )
    resolved = resolve_profile(logical, contract, deployment)

    with pytest.raises(ChatGatewayError, match="template artifact"):
        ChatGatewayProfile(
            resolved=resolved,
            adapter_id=LLAMA_CPP_CHAT_ADAPTER,
            request_timeout_seconds=30,
        )
