"""Validated Stage A chat gateway contracts.

This module owns OpenAI-compatible chat schema/profile validation only. It does
not submit Jobs, own resources, contact runtimes, or execute tools.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..serving import (
    DeploymentIdentity,
    LogicalServingProfile,
    ResolvedServingProfile,
    ServingContract,
    ServingJobBinding,
    resolve_profile,
)


CHAT_CATALOG_SCHEMA = "chat-gateway-catalog-v1"
CHAT_JOB_SCHEMA = "chat-job-v1"
LLAMA_CPP_CHAT_ADAPTER = "llama-cpp-chat-v1"
CHAT_OPERATION_SCHEMA = "openai-chat-completions-v1"

_REQUIRED_LIMITS = frozenset(
    {"input_tokens", "output_tokens", "total_tokens", "request_bytes"}
)
_ALLOWED_REQUEST_FIELDS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "stop",
        "n",
        "stream",
    }
)


class ChatGatewayError(ValueError):
    """Public-safe Stage A validation/normalization failure."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(message)


def _nonblank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ChatGatewayError("invalid_request", f"{name} must be a non-blank string")
    return value.strip()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ChatGatewayError("invalid_request", f"{name} must be a positive integer")
    return value


def _finite_number(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChatGatewayError("invalid_request", f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ChatGatewayError("invalid_request", f"{name} must be finite")
    lower_ok = numeric >= minimum if minimum_inclusive else numeric > minimum
    if not lower_ok or numeric > maximum:
        operator = ">=" if minimum_inclusive else ">"
        raise ChatGatewayError(
            "invalid_request",
            f"{name} must be {operator} {minimum} and <= {maximum}",
        )
    return value


@dataclass(frozen=True, slots=True)
class ChatGatewayProfile:
    """One configured model ID bound to an already-resolved #93 profile."""

    resolved: ResolvedServingProfile
    adapter_id: str
    request_timeout_seconds: float
    max_attempts: int = 2
    owned_by: str = "astrumweaver"
    created: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.resolved, ResolvedServingProfile):
            raise TypeError("resolved must be ResolvedServingProfile")
        adapter_id = _nonblank(self.adapter_id, "adapter_id")
        if adapter_id != LLAMA_CPP_CHAT_ADAPTER:
            raise ChatGatewayError(
                "unsupported_adapter",
                "Stage A supports only the reviewed llama.cpp chat adapter",
            )
        if self.resolved.profile.capability != "llm.chat":
            raise ChatGatewayError(
                "invalid_profile",
                "chat gateway profile must bind llm.chat",
            )
        if self.resolved.profile.operation_schema != CHAT_OPERATION_SCHEMA:
            raise ChatGatewayError(
                "invalid_profile",
                "chat gateway profile uses an unsupported operation schema",
            )
        limits = dict(self.resolved.effective_limits)
        missing = _REQUIRED_LIMITS - limits.keys()
        if missing:
            raise ChatGatewayError(
                "invalid_profile",
                "chat gateway profile is missing required bounded limits",
            )
        if limits["input_tokens"] + limits["output_tokens"] > limits["total_tokens"]:
            raise ChatGatewayError(
                "invalid_profile",
                "chat input/output limits exceed the total context limit",
            )
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, (int, float))
            or not math.isfinite(float(self.request_timeout_seconds))
            or self.request_timeout_seconds <= 0
        ):
            raise ChatGatewayError(
                "invalid_profile",
                "request_timeout_seconds must be finite and positive",
            )
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise ChatGatewayError("invalid_profile", "max_attempts must be an integer")
        if self.max_attempts < 1:
            raise ChatGatewayError("invalid_profile", "max_attempts must be positive")
        if isinstance(self.created, bool) or not isinstance(self.created, int) or self.created < 0:
            raise ChatGatewayError("invalid_profile", "created must be non-negative")
        object.__setattr__(self, "adapter_id", adapter_id)
        object.__setattr__(self, "owned_by", _nonblank(self.owned_by, "owned_by"))

    @property
    def profile_id(self) -> str:
        return self.resolved.profile.profile_id

    @property
    def binding(self) -> ServingJobBinding:
        return ServingJobBinding.from_resolved(self.resolved)

    def model_object(self) -> dict[str, object]:
        return {
            "id": self.profile_id,
            "object": "model",
            "created": self.created,
            "owned_by": self.owned_by,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ChatGatewayProfile":
        data = dict(value)
        try:
            deployment = DeploymentIdentity(**dict(data["deployment"]))
            contract = ServingContract(**dict(data["contract"]))
            profile = LogicalServingProfile(**dict(data["profile"]))
            resolved = resolve_profile(profile, contract, deployment)
        except (KeyError, TypeError, ValueError) as exc:
            raise ChatGatewayError(
                "invalid_profile",
                "configured chat profile identity is invalid",
            ) from exc
        try:
            return cls(
                resolved=resolved,
                adapter_id=data["adapter_id"],
                request_timeout_seconds=data.get("request_timeout_seconds", 60.0),
                max_attempts=data.get("max_attempts", 2),
                owned_by=data.get("owned_by", "astrumweaver"),
                created=data.get("created", 0),
            )
        except KeyError as exc:
            raise ChatGatewayError(
                "invalid_profile",
                "configured chat profile is missing a required field",
            ) from exc


class ChatProfileCatalog:
    def __init__(self, profiles: tuple[ChatGatewayProfile, ...]) -> None:
        values = tuple(profiles)
        if not values:
            raise ChatGatewayError(
                "invalid_catalog",
                "chat gateway catalog must contain at least one profile",
            )
        table: dict[str, ChatGatewayProfile] = {}
        for profile in values:
            if not isinstance(profile, ChatGatewayProfile):
                raise TypeError("profiles must contain ChatGatewayProfile values")
            if profile.profile_id in table:
                raise ChatGatewayError(
                    "invalid_catalog",
                    "chat gateway profile IDs must be unique",
                )
            table[profile.profile_id] = profile
        self._profiles = values
        self._table = MappingProxyType(table)

    @property
    def profiles(self) -> tuple[ChatGatewayProfile, ...]:
        return self._profiles

    def get(self, profile_id: str) -> ChatGatewayProfile:
        try:
            return self._table[profile_id]
        except KeyError as exc:
            raise ChatGatewayError(
                "model_not_found",
                "requested model/profile is not configured",
                status_code=404,
            ) from exc

    def models_response(self) -> dict[str, object]:
        return {
            "object": "list",
            "data": [profile.model_object() for profile in self._profiles],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ChatProfileCatalog":
        data = dict(value)
        if data.get("schema_version") != CHAT_CATALOG_SCHEMA:
            raise ChatGatewayError(
                "invalid_catalog",
                "unsupported chat gateway catalog schema",
            )
        raw_profiles = data.get("profiles")
        if not isinstance(raw_profiles, list):
            raise ChatGatewayError(
                "invalid_catalog",
                "chat gateway catalog profiles must be an array",
            )
        if not all(isinstance(item, Mapping) for item in raw_profiles):
            raise ChatGatewayError(
                "invalid_catalog",
                "chat gateway catalog profile entries must be objects",
            )
        return cls(
            tuple(ChatGatewayProfile.from_dict(item) for item in raw_profiles)
        )


@dataclass(frozen=True, slots=True)
class CompiledChatRequest:
    profile: ChatGatewayProfile
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ChatGatewayProfile):
            raise TypeError("profile must be ChatGatewayProfile")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


def _validate_tool_call(value: object, *, seen: set[str]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ChatGatewayError("invalid_request", "tool_calls must contain objects")
    data = dict(value)
    if set(data) != {"id", "type", "function"}:
        raise ChatGatewayError(
            "invalid_request",
            "tool call uses unsupported fields",
        )
    call_id = _nonblank(data["id"], "tool call id")
    if call_id in seen:
        raise ChatGatewayError("invalid_request", "tool call IDs must be unique")
    if data["type"] != "function":
        raise ChatGatewayError("invalid_request", "only function tool calls are supported")
    function = data["function"]
    if not isinstance(function, Mapping):
        raise ChatGatewayError("invalid_request", "tool call function must be an object")
    function_data = dict(function)
    if set(function_data) != {"name", "arguments"}:
        raise ChatGatewayError(
            "invalid_request",
            "tool call function uses unsupported fields",
        )
    name = _nonblank(function_data["name"], "tool call function name")
    arguments = function_data["arguments"]
    if not isinstance(arguments, str):
        raise ChatGatewayError(
            "invalid_request",
            "tool call arguments must remain a JSON string",
        )
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ChatGatewayError(
            "invalid_request",
            "tool call arguments must contain valid JSON",
        ) from exc
    if not isinstance(decoded, Mapping):
        raise ChatGatewayError(
            "invalid_request",
            "tool call arguments JSON must be an object",
        )
    seen.add(call_id)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _validate_messages(
    value: object,
) -> tuple[list[dict[str, object]], bool]:
    if not isinstance(value, list) or not value:
        raise ChatGatewayError("invalid_request", "messages must be a non-empty array")

    result: list[dict[str, object]] = []
    seen_calls: set[str] = set()
    pending_calls: set[str] = set()
    uses_tools = False

    for raw in value:
        if not isinstance(raw, Mapping):
            raise ChatGatewayError("invalid_request", "messages must contain objects")
        data = dict(raw)
        role = data.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ChatGatewayError("invalid_request", "message role is unsupported")

        if role != "tool" and pending_calls:
            raise ChatGatewayError(
                "invalid_request",
                "all pending tool calls require tool results before the next message",
            )

        allowed = {"role", "content", "name"}
        if role == "assistant":
            allowed.add("tool_calls")
        if role == "tool":
            allowed.add("tool_call_id")
        if set(data) - allowed:
            raise ChatGatewayError(
                "invalid_request",
                "message uses unsupported fields",
            )

        message: dict[str, object] = {"role": role}
        if "name" in data:
            message["name"] = _nonblank(data["name"], "message name")

        content = data.get("content")
        if role in {"system", "user", "tool"}:
            if not isinstance(content, str):
                raise ChatGatewayError(
                    "invalid_request",
                    f"{role} message content must be a string",
                )
            message["content"] = content
        else:
            if content is not None and not isinstance(content, str):
                raise ChatGatewayError(
                    "invalid_request",
                    "assistant message content must be a string or null",
                )
            message["content"] = content

        if role == "assistant":
            raw_calls = data.get("tool_calls")
            if raw_calls is not None:
                if not isinstance(raw_calls, list) or not raw_calls:
                    raise ChatGatewayError(
                        "invalid_request",
                        "assistant tool_calls must be a non-empty array",
                    )
                calls = [
                    _validate_tool_call(item, seen=seen_calls)
                    for item in raw_calls
                ]
                pending_calls.update(str(call["id"]) for call in calls)
                message["tool_calls"] = calls
                uses_tools = True
            elif content is None:
                raise ChatGatewayError(
                    "invalid_request",
                    "assistant message requires content or tool_calls",
                )

        if role == "tool":
            call_id = _nonblank(data.get("tool_call_id"), "tool_call_id")
            if call_id not in pending_calls:
                raise ChatGatewayError(
                    "invalid_request",
                    "tool result does not match a pending tool call",
                )
            pending_calls.remove(call_id)
            message["tool_call_id"] = call_id
            uses_tools = True

        result.append(message)

    if pending_calls:
        raise ChatGatewayError(
            "invalid_request",
            "tool call history is missing one or more tool results",
        )
    return result, uses_tools


def _validate_tools(value: object) -> tuple[list[dict[str, object]], set[str]]:
    if not isinstance(value, list):
        raise ChatGatewayError("invalid_request", "tools must be an array")
    result: list[dict[str, object]] = []
    names: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ChatGatewayError("invalid_request", "tools must contain objects")
        data = dict(raw)
        if set(data) != {"type", "function"} or data["type"] != "function":
            raise ChatGatewayError(
                "invalid_request",
                "Stage A supports only function tools",
            )
        function = data["function"]
        if not isinstance(function, Mapping):
            raise ChatGatewayError("invalid_request", "tool function must be an object")
        function_data = dict(function)
        if set(function_data) - {"name", "description", "parameters", "strict"}:
            raise ChatGatewayError(
                "invalid_request",
                "tool function uses unsupported fields",
            )
        name = _nonblank(function_data.get("name"), "tool function name")
        if name in names:
            raise ChatGatewayError("invalid_request", "tool function names must be unique")
        names.add(name)
        normalized: dict[str, object] = {"name": name}
        if "description" in function_data:
            if not isinstance(function_data["description"], str):
                raise ChatGatewayError(
                    "invalid_request",
                    "tool function description must be a string",
                )
            normalized["description"] = function_data["description"]
        if "parameters" in function_data:
            if not isinstance(function_data["parameters"], Mapping):
                raise ChatGatewayError(
                    "invalid_request",
                    "tool function parameters must be a JSON-schema object",
                )
            normalized["parameters"] = dict(function_data["parameters"])
        if "strict" in function_data:
            if type(function_data["strict"]) is not bool:
                raise ChatGatewayError(
                    "invalid_request",
                    "tool function strict must be boolean",
                )
            normalized["strict"] = function_data["strict"]
        result.append({"type": "function", "function": normalized})
    return result, names


def _validate_tool_choice(value: object, *, tool_names: set[str]) -> object:
    if isinstance(value, str):
        if value not in {"none", "auto", "required"}:
            raise ChatGatewayError("invalid_request", "tool_choice is unsupported")
        if value != "none" and not tool_names:
            raise ChatGatewayError(
                "invalid_request",
                "tool_choice requires configured request tools",
            )
        return value
    if not isinstance(value, Mapping):
        raise ChatGatewayError("invalid_request", "tool_choice is unsupported")
    data = dict(value)
    if set(data) != {"type", "function"} or data["type"] != "function":
        raise ChatGatewayError("invalid_request", "named tool_choice is invalid")
    function = data["function"]
    if not isinstance(function, Mapping) or set(function) != {"name"}:
        raise ChatGatewayError("invalid_request", "named tool_choice is invalid")
    name = _nonblank(function["name"], "tool_choice function name")
    if name not in tool_names:
        raise ChatGatewayError(
            "invalid_request",
            "named tool_choice does not match a request tool",
        )
    return {"type": "function", "function": {"name": name}}


def _copy_sampling_fields(body: Mapping[str, object], request: dict[str, object]) -> None:
    for name in ("temperature",):
        if name in body:
            request[name] = _finite_number(
                body[name], name, minimum=0.0, maximum=2.0
            )
    if "top_p" in body:
        request["top_p"] = _finite_number(
            body["top_p"],
            "top_p",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
    for name in ("presence_penalty", "frequency_penalty"):
        if name in body:
            request[name] = _finite_number(
                body[name], name, minimum=-2.0, maximum=2.0
            )
    if "seed" in body:
        seed = body["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ChatGatewayError("invalid_request", "seed must be an integer")
        request["seed"] = seed
    if "stop" in body:
        stop = body["stop"]
        if isinstance(stop, str):
            request["stop"] = stop
        elif (
            isinstance(stop, list)
            and 1 <= len(stop) <= 4
            and all(isinstance(item, str) for item in stop)
        ):
            request["stop"] = list(stop)
        else:
            raise ChatGatewayError(
                "invalid_request",
                "stop must be a string or an array of one to four strings",
            )


def compile_chat_request(
    body: Mapping[str, object],
    profile: ChatGatewayProfile,
    *,
    request_size_bytes: int,
) -> CompiledChatRequest:
    """Validate and compile one Stage A OpenAI-compatible request."""

    if not isinstance(body, Mapping):
        raise ChatGatewayError("invalid_request", "JSON object is required")
    if isinstance(request_size_bytes, bool) or not isinstance(request_size_bytes, int):
        raise TypeError("request_size_bytes must be an integer")
    if request_size_bytes < 0:
        raise ValueError("request_size_bytes must not be negative")

    data = dict(body)
    unknown = set(data) - _ALLOWED_REQUEST_FIELDS
    if unknown:
        raise ChatGatewayError(
            "unsupported_field",
            "request contains fields outside the Stage A subset",
        )

    if data.get("model") != profile.profile_id:
        raise ChatGatewayError(
            "model_mismatch",
            "request model does not match the resolved gateway profile",
        )

    if "stream" in data and data["stream"] is not False:
        raise ChatGatewayError(
            "streaming_not_supported",
            "stream=true requires the Stage B fenced event channel",
        )
    if "n" in data and data["n"] != 1:
        raise ChatGatewayError(
            "unsupported_field",
            "Stage A supports exactly one completion choice",
        )

    limits = dict(profile.resolved.effective_limits)
    if request_size_bytes > limits["request_bytes"]:
        raise ChatGatewayError(
            "request_too_large",
            "chat request exceeds the configured byte limit",
            status_code=413,
        )

    messages, history_uses_tools = _validate_messages(data.get("messages"))
    request: dict[str, object] = {
        "messages": messages,
        "stream": False,
    }

    tool_names: set[str] = set()
    if "tools" in data:
        tools, tool_names = _validate_tools(data["tools"])
        request["tools"] = tools
    tool_choice_uses_tools = False
    if "tool_choice" in data:
        choice = _validate_tool_choice(data["tool_choice"], tool_names=tool_names)
        request["tool_choice"] = choice
        tool_choice_uses_tools = choice != "none"

    uses_tools = history_uses_tools or bool(tool_names) or tool_choice_uses_tools
    if uses_tools and "tools" not in profile.resolved.contract.features:
        raise ChatGatewayError(
            "unsupported_feature",
            "selected profile does not advertise structured tools",
        )

    if "max_tokens" in data and "max_completion_tokens" in data:
        raise ChatGatewayError(
            "invalid_request",
            "max_tokens and max_completion_tokens are mutually exclusive",
        )
    raw_output = data.get(
        "max_tokens",
        data.get("max_completion_tokens", limits["output_tokens"]),
    )
    output_tokens = _positive_int(raw_output, "max_tokens")
    if output_tokens > limits["output_tokens"]:
        raise ChatGatewayError(
            "output_limit_exceeded",
            "requested output token budget exceeds the profile limit",
        )
    request["max_tokens"] = output_tokens

    _copy_sampling_fields(data, request)

    payload = {
        "schema_version": CHAT_JOB_SCHEMA,
        "adapter_id": profile.adapter_id,
        "request": request,
        "limits": {
            "input_tokens": limits["input_tokens"],
            "output_tokens": limits["output_tokens"],
            "total_tokens": limits["total_tokens"],
        },
    }
    return CompiledChatRequest(profile=profile, payload=payload)


def _normalize_response_tool_calls(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ChatGatewayError(
            "invalid_provider_response",
            "provider returned invalid tool_calls",
            status_code=502,
        )
    seen: set[str] = set()
    try:
        return [_validate_tool_call(item, seen=seen) for item in value]
    except ChatGatewayError as exc:
        raise ChatGatewayError(
            "invalid_provider_response",
            "provider returned invalid structured tool output",
            status_code=502,
        ) from exc


def normalize_chat_completion(
    outputs: Mapping[str, object],
    *,
    profile_id: str,
    job_id: str,
    created: int,
) -> dict[str, object]:
    """Normalize the first provider response into the Stage A public subset."""

    if not isinstance(outputs, Mapping):
        raise ChatGatewayError(
            "invalid_provider_response",
            "provider result must be an object",
            status_code=502,
        )
    data = dict(outputs)
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ChatGatewayError(
            "invalid_provider_response",
            "provider must return exactly one chat choice",
            status_code=502,
        )
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ChatGatewayError(
            "invalid_provider_response",
            "provider chat choice is invalid",
            status_code=502,
        )
    choice = dict(first)
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ChatGatewayError(
            "invalid_provider_response",
            "provider chat choice lacks an assistant message",
            status_code=502,
        )
    message_data = dict(message)
    if message_data.get("role") != "assistant":
        raise ChatGatewayError(
            "invalid_provider_response",
            "provider response role is not assistant",
            status_code=502,
        )
    content = message_data.get("content")
    if content is not None and not isinstance(content, str):
        raise ChatGatewayError(
            "invalid_provider_response",
            "provider assistant content is invalid",
            status_code=502,
        )
    normalized_message: dict[str, object] = {
        "role": "assistant",
        "content": content,
    }
    if message_data.get("tool_calls") is not None:
        normalized_message["tool_calls"] = _normalize_response_tool_calls(
            message_data["tool_calls"]
        )
    if content is None and "tool_calls" not in normalized_message:
        raise ChatGatewayError(
            "invalid_provider_response",
            "provider assistant message has no content or tool_calls",
            status_code=502,
        )

    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ChatGatewayError(
            "invalid_provider_response",
            "provider finish_reason is invalid",
            status_code=502,
        )
    index = choice.get("index", 0)
    if isinstance(index, bool) or not isinstance(index, int):
        raise ChatGatewayError(
            "invalid_provider_response",
            "provider choice index is invalid",
            status_code=502,
        )

    result: dict[str, object] = {
        "id": data.get("id") if isinstance(data.get("id"), str) else f"chatcmpl-{job_id}",
        "object": "chat.completion",
        "created": data.get("created") if isinstance(data.get("created"), int) else created,
        "model": profile_id,
        "choices": [
            {
                "index": index,
                "message": normalized_message,
                "finish_reason": finish_reason,
            }
        ],
    }

    usage = data.get("usage")
    if isinstance(usage, Mapping):
        normalized_usage: dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                normalized_usage[key] = value
        if normalized_usage:
            result["usage"] = normalized_usage
    return result


__all__ = [
    "CHAT_CATALOG_SCHEMA",
    "CHAT_JOB_SCHEMA",
    "CHAT_OPERATION_SCHEMA",
    "LLAMA_CPP_CHAT_ADAPTER",
    "ChatGatewayError",
    "ChatGatewayProfile",
    "ChatProfileCatalog",
    "CompiledChatRequest",
    "compile_chat_request",
    "normalize_chat_completion",
]
