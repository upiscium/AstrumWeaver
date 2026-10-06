"""First-class llama.cpp RuntimeProvider."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx

from ...execution import (
    JobEvent,
    JobEventSink,
    JobExecutionError,
    JobExecutor,
    JobRequest,
    JobResult,
    ResidencyItem,
    ResidencyReport,
)
from ...gateway.chat import (
    CHAT_JOB_SCHEMA,
    LLAMA_CPP_CHAT_ADAPTER,
    ChatGatewayError,
    validate_chat_stream_chunk,
)
from ...gateway.embedding import (
    EMBEDDING_JOB_SCHEMA,
    LLAMA_CPP_EMBEDDING_ADAPTER,
    EmbeddingGatewayError,
    validate_embedding_response,
)
from ..contracts import (
    CompatibilityReason,
    GPUTopology,
    ManagedRuntime,
    ModelPreparationPolicy,
    ModelTopology,
    ResidencyPolicy,
    RuntimeCompatibility,
    RuntimeCompatibilityContext,
    RuntimeHealth,
    RuntimeHealthState,
    RuntimeProvider,
    RuntimeProviderInfo,
    RuntimeSetupIntent,
)


LLAMA_CPP_PROVIDER_ID = "llama-cpp"
LLAMA_CPP_CAPABILITIES = frozenset({"llm.chat", "text.generate"})
LLAMA_CPP_EMBEDDING_CAPABILITIES = frozenset({"text.embed"})


def _nonblank(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


class LlamaCppSplitMode(StrEnum):
    NONE = "none"
    LAYER = "layer"
    ROW = "row"
    TENSOR = "tensor"


@dataclass(frozen=True, slots=True)
class LlamaCppProviderConfig:
    base_url: str = "http://127.0.0.1:8080"
    executable: str = "llama-server"
    package_reference: str = "llama-cpp"
    startup_timeout_seconds: float = 120.0
    request_timeout_seconds: float = 300.0
    context_size: int = 4096
    gpu_layers: int | str | None = None
    split_mode: LlamaCppSplitMode | None = None
    tensor_split: tuple[float, ...] | None = None
    main_gpu: int = 0
    fit: bool | None = None
    fit_target_mb: tuple[int, ...] | None = None
    cpu_moe: bool = False
    n_cpu_moe: int | None = None
    n_cpu_ffn: int | None = None
    offline: bool = True
    no_webui: bool = True
    jinja: bool = False
    chat_template_file: str | None = None
    embeddings: bool = False
    pooling: str | None = None

    def __post_init__(self) -> None:
        base_url = _nonblank(self.base_url, "base_url").rstrip("/")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.port is None
        ):
            raise ValueError(
                "llama.cpp provider requires a local loopback HTTP endpoint with port"
            )
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(
            self,
            "executable",
            _nonblank(self.executable, "executable"),
        )
        object.__setattr__(
            self,
            "package_reference",
            _nonblank(self.package_reference, "package_reference"),
        )
        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.context_size < 0:
            raise ValueError("context_size must not be negative")
        if self.main_gpu < 0:
            raise ValueError("main_gpu must not be negative")
        if type(self.jinja) is not bool:
            raise TypeError("jinja must be boolean")
        if type(self.embeddings) is not bool:
            raise TypeError("embeddings must be boolean")
        if self.embeddings:
            if self.jinja or self.chat_template_file is not None:
                raise ValueError(
                    "embedding mode cannot enable chat templates"
                )
            pooling = _nonblank(self.pooling or "", "pooling")
            if pooling not in {"mean", "cls", "last"}:
                raise ValueError(
                    "embedding mode pooling must be mean, cls, or last"
                )
            object.__setattr__(self, "pooling", pooling)
        elif self.pooling is not None:
            raise ValueError("pooling requires embeddings=true")
        if self.chat_template_file is not None:
            template = _nonblank(
                self.chat_template_file,
                "chat_template_file",
            )
            if not self.jinja:
                raise ValueError(
                    "chat_template_file requires jinja=true"
                )
            object.__setattr__(self, "chat_template_file", template)

        if self.gpu_layers is not None:
            value = self.gpu_layers
            if isinstance(value, bool):
                raise TypeError("gpu_layers must be int, auto, all, or None")
            if isinstance(value, int):
                if value < 0:
                    raise ValueError("gpu_layers must not be negative")
            elif str(value) not in {"auto", "all"}:
                raise ValueError("gpu_layers must be int, auto, all, or None")

        if self.split_mode is not None:
            object.__setattr__(
                self,
                "split_mode",
                LlamaCppSplitMode(self.split_mode),
            )

        if self.tensor_split is not None:
            tensor_split = tuple(float(value) for value in self.tensor_split)
            if not tensor_split or any(value <= 0 for value in tensor_split):
                raise ValueError("tensor_split values must be positive")
            object.__setattr__(self, "tensor_split", tensor_split)

        if self.fit_target_mb is not None:
            targets = tuple(int(value) for value in self.fit_target_mb)
            if not targets or any(value < 0 for value in targets):
                raise ValueError("fit_target_mb values must be non-negative")
            object.__setattr__(self, "fit_target_mb", targets)

        for field_name in ("n_cpu_moe", "n_cpu_ffn"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must not be negative")


@dataclass(frozen=True, slots=True)
class LlamaCppLaunchPolicy:
    gpu_layers: int | str
    split_mode: LlamaCppSplitMode
    fit: bool
    tensor_split: tuple[float, ...] | None
    fit_target_mb: tuple[int, ...] | None
    main_gpu: int
    cpu_moe: bool
    n_cpu_moe: int | None
    n_cpu_ffn: int | None


def _launch_policy(
    context: RuntimeCompatibilityContext,
    config: LlamaCppProviderConfig,
) -> LlamaCppLaunchPolicy:
    demand = context.demand
    gpu_count = context.worker.resources.gpu_count

    gpu_layers: int | str
    if config.gpu_layers is not None:
        gpu_layers = config.gpu_layers
    elif demand.residency_policy is ResidencyPolicy.VRAM_ONLY:
        gpu_layers = "all"
    else:
        gpu_layers = "auto"

    if config.split_mode is not None:
        split_mode = config.split_mode
    elif demand.gpu_topology is GPUTopology.MULTI_GPU:
        split_mode = LlamaCppSplitMode.LAYER
    else:
        split_mode = LlamaCppSplitMode.NONE

    fit = (
        config.fit
        if config.fit is not None
        else demand.residency_policy is not ResidencyPolicy.VRAM_ONLY
    )

    if gpu_count == 0:
        split_mode = LlamaCppSplitMode.NONE
        gpu_layers = 0
        fit = False

    return LlamaCppLaunchPolicy(
        gpu_layers=gpu_layers,
        split_mode=split_mode,
        fit=fit,
        tensor_split=config.tensor_split,
        fit_target_mb=config.fit_target_mb,
        main_gpu=config.main_gpu,
        cpu_moe=config.cpu_moe,
        n_cpu_moe=config.n_cpu_moe,
        n_cpu_ffn=config.n_cpu_ffn,
    )


@runtime_checkable
class LlamaCppApi(Protocol):
    async def health(self) -> bool: ...

    async def models(self) -> tuple[str, ...]: ...

    async def chat(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def chat_stream(
        self,
        payload: Mapping[str, Any],
    ) -> AsyncIterator[Mapping[str, Any]]: ...

    async def chat_input_tokens(self, payload: Mapping[str, Any]) -> int: ...

    async def completion(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def embeddings(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def tokenize(self, content: str) -> int: ...

    async def close(self) -> None: ...


class HttpLlamaCppApi:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )

    async def health(self) -> bool:
        try:
            response = await self._client.get("/health")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def _json(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        kwargs = {} if json is None else {"json": json}
        response = await self._client.request(method, path, **kwargs)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, Mapping):
            raise RuntimeError("llama.cpp server returned a non-object response")
        return body

    async def models(self) -> tuple[str, ...]:
        body = await self._json("GET", "/v1/models")
        data = body.get("data", ())
        if not isinstance(data, list):
            raise RuntimeError("llama.cpp /v1/models returned invalid data")
        result: list[str] = []
        for item in data:
            if not isinstance(item, Mapping):
                continue
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id.strip():
                result.append(model_id.strip())
        return tuple(result)

    async def chat(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return await self._json(
            "POST",
            "/v1/chat/completions",
            json=payload,
        )

    async def chat_stream(
        self,
        payload: Mapping[str, Any],
    ) -> AsyncIterator[Mapping[str, Any]]:
        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json=dict(payload),
        ) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    raise RuntimeError("llama.cpp stream returned invalid SSE framing")
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    value = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "llama.cpp stream returned invalid JSON"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise RuntimeError(
                        "llama.cpp stream returned a non-object event"
                    )
                yield value

    async def chat_input_tokens(self, payload: Mapping[str, Any]) -> int:
        body = await self._json(
            "POST",
            "/v1/chat/completions/input_tokens",
            json=payload,
        )
        value = body.get("input_tokens")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(
                "llama.cpp token-count endpoint returned invalid input_tokens"
            )
        return value

    async def completion(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return await self._json(
            "POST",
            "/v1/completions",
            json=payload,
        )

    async def embeddings(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return await self._json(
            "POST",
            "/v1/embeddings",
            json=payload,
        )

    async def tokenize(self, content: str) -> int:
        body = await self._json(
            "POST",
            "/tokenize",
            json={"content": content, "add_special": False},
        )
        tokens = body.get("tokens")
        if not isinstance(tokens, list) or any(
            isinstance(token, bool) or not isinstance(token, int)
            for token in tokens
        ):
            raise RuntimeError(
                "llama.cpp tokenize endpoint returned invalid tokens"
            )
        return len(tokens)

    async def close(self) -> None:
        await self._client.aclose()


@runtime_checkable
class LlamaCppProcessController(Protocol):
    @property
    def running(self) -> bool: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class LlamaCppSubprocessController:
    def __init__(
        self,
        *,
        executable: str,
        base_url: str,
        model_ref: str,
        model_alias: str,
        gpu_uuids: tuple[str, ...],
        context_size: int,
        launch_policy: LlamaCppLaunchPolicy,
        offline: bool,
        no_webui: bool,
        jinja: bool = False,
        chat_template_file: str | None = None,
        embeddings: bool = False,
        pooling: str | None = None,
    ) -> None:
        self.executable = executable
        self.base_url = base_url
        self.model_ref = model_ref
        self.model_alias = model_alias
        self.gpu_uuids = gpu_uuids
        self.context_size = context_size
        self.launch_policy = launch_policy
        self.offline = offline
        self.no_webui = no_webui
        self.jinja = jinja
        self.chat_template_file = chat_template_file
        self.embeddings = embeddings
        self.pooling = pooling
        self._process: asyncio.subprocess.Process | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def command(self) -> tuple[str, ...]:
        parsed = urlsplit(self.base_url)
        assert parsed.hostname is not None
        assert parsed.port is not None

        args: list[str] = [
            self.executable,
            "--host",
            parsed.hostname,
            "--port",
            str(parsed.port),
            "--model",
            self.model_ref,
            "--alias",
            self.model_alias,
            "--parallel",
            "1",
            "--ctx-size",
            str(self.context_size),
            "--n-gpu-layers",
            str(self.launch_policy.gpu_layers),
            "--fit",
            "on" if self.launch_policy.fit else "off",
        ]
        if self.no_webui:
            args.append("--no-webui")
        if self.offline:
            args.append("--offline")
        if self.jinja:
            args.append("--jinja")
        if self.chat_template_file is not None:
            args.extend(
                ["--chat-template-file", self.chat_template_file]
            )
        if self.embeddings:
            args.append("--embeddings")
            if self.pooling is None:
                raise RuntimeError(
                    "embedding-mode llama.cpp process requires pooling"
                )
            args.extend(["--pooling", self.pooling])

        if self.gpu_uuids:
            visible = [f"CUDA{index}" for index in range(len(self.gpu_uuids))]
            args.extend(["--device", ",".join(visible)])
            args.extend(
                ["--split-mode", self.launch_policy.split_mode.value]
            )
            if self.launch_policy.split_mode in {
                LlamaCppSplitMode.NONE,
                LlamaCppSplitMode.ROW,
            }:
                args.extend(
                    ["--main-gpu", str(self.launch_policy.main_gpu)]
                )
            if self.launch_policy.tensor_split is not None:
                args.extend(
                    [
                        "--tensor-split",
                        ",".join(
                            str(value)
                            for value in self.launch_policy.tensor_split
                        ),
                    ]
                )
            if self.launch_policy.fit_target_mb is not None:
                args.extend(
                    [
                        "--fit-target",
                        ",".join(
                            str(value)
                            for value in self.launch_policy.fit_target_mb
                        ),
                    ]
                )

        if self.launch_policy.cpu_moe:
            args.append("--cpu-moe")
        if self.launch_policy.n_cpu_moe is not None:
            args.extend(
                ["--n-cpu-moe", str(self.launch_policy.n_cpu_moe)]
            )
        if self.launch_policy.n_cpu_ffn is not None:
            args.extend(
                ["--n-cpu-ffn", str(self.launch_policy.n_cpu_ffn)]
            )

        return tuple(args)

    async def start(self) -> None:
        if self.running:
            return
        env = dict(os.environ)
        if self.gpu_uuids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpu_uuids)
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.command(),
                env=env,
            )
        except OSError as exc:
            raise RuntimeError(
                "failed to start llama.cpp server process"
            ) from exc

    async def stop(self) -> None:
        process = self._process
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        self._process = None


class LlamaCppExecutor(JobExecutor):
    capabilities = LLAMA_CPP_CAPABILITIES

    def __init__(
        self,
        *,
        api: LlamaCppApi,
        model_ref: str,
        model_alias: str,
        residency_metadata: Mapping[str, Any],
        tools_enabled: bool = False,
    ) -> None:
        self.api = api
        self.model_ref = _nonblank(model_ref, "model_ref")
        self.model_alias = _nonblank(model_alias, "model_alias")
        self.residency_metadata = dict(residency_metadata)
        if type(tools_enabled) is not bool:
            raise TypeError("tools_enabled must be boolean")
        self.tools_enabled = tools_enabled
        self.serving_features = {
            "llm.chat": (
                frozenset({"tools"})
                if tools_enabled
                else frozenset()
            )
        }
        self._inflight: dict[str, asyncio.Task[Any]] = {}

    def _payload(self, job: JobRequest) -> dict[str, Any]:
        payload = dict(job.payload)
        supplied_model = payload.pop("model", None)
        if supplied_model is not None and str(supplied_model) not in {
            self.model_ref,
            self.model_alias,
        }:
            raise ValueError(
                "job model does not match the model bound to this llama.cpp runtime"
            )
        payload["model"] = self.model_alias
        payload["stream"] = False
        return payload

    def _gateway_chat_payload(
        self,
        job: JobRequest,
    ) -> tuple[dict[str, Any], dict[str, int]] | None:
        envelope = dict(job.payload)
        if envelope.get("schema_version") != CHAT_JOB_SCHEMA:
            return None
        if set(envelope) != {"schema_version", "adapter_id", "request", "limits"}:
            raise JobExecutionError(
                "invalid_chat_job",
                "chat job envelope uses unsupported fields",
                retryable=False,
            )
        if envelope.get("adapter_id") != LLAMA_CPP_CHAT_ADAPTER:
            raise JobExecutionError(
                "unsupported_chat_adapter",
                "chat job targets an unsupported provider adapter",
                retryable=False,
            )
        raw_request = envelope.get("request")
        raw_limits = envelope.get("limits")
        if not isinstance(raw_request, Mapping) or not isinstance(raw_limits, Mapping):
            raise JobExecutionError(
                "invalid_chat_job",
                "chat job request/limits must be objects",
                retryable=False,
            )
        request = dict(raw_request)
        if not isinstance(request.get("messages"), list):
            raise JobExecutionError(
                "invalid_chat_job",
                "chat job requires a messages array",
                retryable=False,
            )
        if type(request.get("stream")) is not bool:
            raise JobExecutionError(
                "invalid_chat_job",
                "chat job stream flag must be boolean",
                retryable=False,
            )
        output_tokens = request.get("max_tokens")
        if (
            isinstance(output_tokens, bool)
            or not isinstance(output_tokens, int)
            or output_tokens < 1
        ):
            raise JobExecutionError(
                "invalid_chat_job",
                "chat job max_tokens must be a positive integer",
                retryable=False,
            )
        limits: dict[str, int] = {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = raw_limits.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise JobExecutionError(
                    "invalid_chat_job",
                    "chat job token limits must be positive integers",
                    retryable=False,
                )
            limits[key] = value
        if (
            limits["input_tokens"] > limits["total_tokens"]
            or limits["output_tokens"] > limits["total_tokens"]
        ):
            raise JobExecutionError(
                "invalid_chat_job",
                "chat job token limit exceeds total context",
                retryable=False,
            )
        if output_tokens > limits["output_tokens"]:
            raise JobExecutionError(
                "output_limit_exceeded",
                "chat output budget exceeds the bound serving profile",
                retryable=False,
            )
        supplied_model = request.pop("model", None)
        if supplied_model is not None:
            raise JobExecutionError(
                "invalid_chat_job",
                "gateway chat job must not supply a runtime model name",
                retryable=False,
            )
        uses_tools = bool(request.get("tools"))
        tool_choice = request.get("tool_choice")
        uses_tools = uses_tools or (
            tool_choice is not None and tool_choice != "none"
        )
        for message in request.get("messages", ()):
            if isinstance(message, Mapping) and (
                message.get("role") == "tool"
                or message.get("tool_calls") is not None
            ):
                uses_tools = True
                break
        if uses_tools and not self.tools_enabled:
            raise JobExecutionError(
                "unsupported_feature",
                "llama.cpp serving deployment does not enable structured tools",
                retryable=False,
            )

        request["model"] = self.model_alias
        return request, limits

    async def _enforce_chat_limits(
        self,
        payload: Mapping[str, Any],
        limits: Mapping[str, int],
    ) -> None:
        input_tokens = await self.api.chat_input_tokens(payload)
        output_tokens = int(payload["max_tokens"])
        if input_tokens > limits["input_tokens"]:
            raise JobExecutionError(
                "context_length_exceeded",
                "chat input exceeds the deployed input-token limit",
                retryable=False,
            )
        if input_tokens + output_tokens > limits["total_tokens"]:
            raise JobExecutionError(
                "context_length_exceeded",
                "chat input plus output reservation exceeds deployed context",
                retryable=False,
            )

    async def execute(self, job: JobRequest) -> JobResult:
        if job.capability not in self.capabilities:
            raise ValueError(
                f"llama.cpp executor does not support capability: {job.capability}"
            )

        gateway_chat = (
            self._gateway_chat_payload(job)
            if job.capability == "llm.chat"
            else None
        )
        if gateway_chat is None:
            payload = self._payload(job)
        else:
            payload, limits = gateway_chat

        if job.capability == "llm.chat":
            if not isinstance(payload.get("messages"), list):
                raise ValueError("llm.chat requires a messages list")
            if gateway_chat is not None:
                if payload.get("stream") is True:
                    raise JobExecutionError(
                        "streaming_requires_event_sink",
                        "streaming chat requires the fenced Job event channel",
                        retryable=False,
                    )
                await self._enforce_chat_limits(payload, limits)
            call = self.api.chat(payload)
        else:
            if not isinstance(payload.get("prompt"), str):
                raise ValueError("text.generate requires a prompt string")
            call = self.api.completion(payload)

        task = asyncio.create_task(call)
        self._inflight[job.job_id] = task
        try:
            response = await task
        finally:
            self._inflight.pop(job.job_id, None)

        text: str | None = None
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                if job.capability == "llm.chat":
                    message = first.get("message")
                    if isinstance(message, Mapping):
                        content = message.get("content")
                        if isinstance(content, str):
                            text = content
                else:
                    candidate = first.get("text")
                    if isinstance(candidate, str):
                        text = candidate

        metrics: dict[str, int | float] = {}
        usage = response.get("usage")
        if isinstance(usage, Mapping):
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                value = usage.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[key] = value

        return JobResult(
            outputs=dict(response),
            metrics=metrics,
            text=text,
            metadata={
                "runtime_provider": LLAMA_CPP_PROVIDER_ID,
                "model": self.model_ref,
                "model_alias": self.model_alias,
            },
        )

    async def execute_stream(
        self,
        job: JobRequest,
        sink: JobEventSink,
    ) -> JobResult:
        if job.capability != "llm.chat":
            return await self.execute(job)
        gateway_chat = self._gateway_chat_payload(job)
        if gateway_chat is None:
            return await self.execute(job)
        payload, limits = gateway_chat
        if payload.get("stream") is not True:
            return await self.execute(job)

        await self._enforce_chat_limits(payload, limits)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

        metrics: dict[str, int | float] = {}
        text_parts: list[str] = []
        current = asyncio.current_task()
        if current is None:
            raise RuntimeError("streaming execution task is unavailable")
        self._inflight[job.job_id] = current
        try:
            async for raw_chunk in self.api.chat_stream(payload):
                try:
                    chunk = validate_chat_stream_chunk(raw_chunk)
                except ChatGatewayError as exc:
                    raise JobExecutionError(
                        "invalid_provider_response",
                        "llama.cpp returned an invalid chat stream chunk",
                        retryable=False,
                    ) from exc
                choices = chunk.get("choices")
                if isinstance(choices, list) and choices:
                    choice = choices[0]
                    if isinstance(choice, Mapping):
                        delta = choice.get("delta")
                        if isinstance(delta, Mapping):
                            content = delta.get("content")
                            if isinstance(content, str):
                                text_parts.append(content)
                usage = chunk.get("usage")
                if isinstance(usage, Mapping):
                    for key in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                    ):
                        value = usage.get(key)
                        if (
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                        ):
                            metrics[key] = value
                await sink.emit(
                    JobEvent(
                        kind="chat.completion.chunk",
                        payload=chunk,
                    )
                )
        finally:
            if self._inflight.get(job.job_id) is current:
                self._inflight.pop(job.job_id, None)

        return JobResult(
            outputs={"streamed": True},
            metrics=metrics,
            text="".join(text_parts) if text_parts else None,
            metadata={
                "runtime_provider": LLAMA_CPP_PROVIDER_ID,
                "model": self.model_ref,
                "model_alias": self.model_alias,
            },
        )

    async def cancel(self, job_id: str) -> None:
        task = self._inflight.get(job_id)
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def residency(self) -> ResidencyReport:
        return ResidencyReport(
            items=(
                ResidencyItem(
                    name=self.model_ref,
                    kind="gguf-model",
                    metadata=dict(self.residency_metadata),
                ),
            ),
            metadata={"runtime_provider": LLAMA_CPP_PROVIDER_ID},
        )


class LlamaCppManagedRuntime(ManagedRuntime):
    provider_id = LLAMA_CPP_PROVIDER_ID

    def __init__(
        self,
        *,
        api: LlamaCppApi,
        process: LlamaCppProcessController,
        context: RuntimeCompatibilityContext,
        model_ref: str,
        model_alias: str,
        startup_timeout_seconds: float,
        launch_policy: LlamaCppLaunchPolicy,
        tools_enabled: bool = False,
    ) -> None:
        self.api = api
        self.process = process
        self.context = context
        self.model_ref = model_ref
        self.model_alias = model_alias
        self.startup_timeout_seconds = startup_timeout_seconds
        self.launch_policy = launch_policy
        self._closed = False
        self._executor = LlamaCppExecutor(
            api=api,
            model_ref=model_ref,
            model_alias=model_alias,
            tools_enabled=tools_enabled,
            residency_metadata={
                "residency_policy": context.demand.residency_policy.value,
                "gpu_topology": context.demand.gpu_topology.value,
                "gpu_count": context.worker.resources.gpu_count,
                "gpu_layers": launch_policy.gpu_layers,
                "split_mode": launch_policy.split_mode.value,
                "fit": launch_policy.fit,
                "tensor_split": (
                    list(launch_policy.tensor_split)
                    if launch_policy.tensor_split is not None
                    else None
                ),
                "cpu_moe": launch_policy.cpu_moe,
                "n_cpu_moe": launch_policy.n_cpu_moe,
                "n_cpu_ffn": launch_policy.n_cpu_ffn,
            },
        )

    async def _server_reachable(self) -> bool:
        return await self.api.health()

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("llama.cpp runtime has already been released")

        reachable = await self._server_reachable()
        if reachable and not self.process.running:
            raise RuntimeError(
                "an external llama.cpp server is already using the configured endpoint"
            )

        started_here = False
        if not self.process.running:
            await self.process.start()
            started_here = True

        try:
            deadline = (
                asyncio.get_running_loop().time()
                + self.startup_timeout_seconds
            )
            while True:
                if not self.process.running:
                    raise RuntimeError(
                        "llama.cpp owned process exited before readiness; "
                        "inspect Worker service logs"
                    )
                if await self._server_reachable():
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError(
                        "llama.cpp server did not become ready before timeout"
                    )
                await asyncio.sleep(0.1)

            models = await self.api.models()
            if self.model_alias not in models:
                raise RuntimeError(
                    "llama.cpp server did not expose the configured model alias"
                )
        except Exception:
            if started_here:
                with contextlib.suppress(Exception):
                    await self.process.stop()
            raise

    async def stop(self) -> None:
        await self.process.stop()

    async def health(self) -> RuntimeHealth:
        reachable = await self._server_reachable()
        if not reachable:
            return RuntimeHealth(
                state=(
                    RuntimeHealthState.STARTING
                    if self.process.running
                    else RuntimeHealthState.STOPPED
                ),
                ready=False,
                metadata={"runtime_provider": LLAMA_CPP_PROVIDER_ID},
            )
        if not self.process.running:
            return RuntimeHealth(
                state=RuntimeHealthState.FAILED,
                ready=False,
                detail="external llama.cpp process detected at managed endpoint",
                metadata={"runtime_provider": LLAMA_CPP_PROVIDER_ID},
            )
        try:
            models = await self.api.models()
        except Exception:
            return RuntimeHealth(
                state=RuntimeHealthState.DEGRADED,
                ready=False,
                detail="llama.cpp model inventory is unavailable",
                metadata={"runtime_provider": LLAMA_CPP_PROVIDER_ID},
            )
        if self.model_alias not in models:
            return RuntimeHealth(
                state=RuntimeHealthState.FAILED,
                ready=False,
                detail="configured llama.cpp model alias is unavailable",
                metadata={"runtime_provider": LLAMA_CPP_PROVIDER_ID},
            )
        # An owned process may exit during the awaited inventory request.
        if not self.process.running:
            return RuntimeHealth(
                state=RuntimeHealthState.STOPPED,
                ready=False,
                metadata={"runtime_provider": LLAMA_CPP_PROVIDER_ID},
            )
        return RuntimeHealth(
            state=RuntimeHealthState.READY,
            ready=True,
            metadata={
                "runtime_provider": LLAMA_CPP_PROVIDER_ID,
                "model": self.model_ref,
                "model_alias": self.model_alias,
            },
        )

    async def residency(self) -> ResidencyReport:
        return await self._executor.residency()

    def executor(self) -> JobExecutor:
        return self._executor

    async def release(self) -> None:
        if self._closed:
            return
        await self.api.close()
        self._closed = True


class LlamaCppProvider(RuntimeProvider):
    def __init__(
        self,
        config: LlamaCppProviderConfig | None = None,
        *,
        api_factory: Any | None = None,
        process_factory: Any | None = None,
    ) -> None:
        self.config = config or LlamaCppProviderConfig()
        self._api_factory = api_factory or (
            lambda base_url: HttpLlamaCppApi(
                base_url,
                timeout_seconds=self.config.request_timeout_seconds,
            )
        )
        self._process_factory = process_factory or (
            lambda **kwargs: LlamaCppSubprocessController(**kwargs)
        )
        self._info = RuntimeProviderInfo(
            provider_id=LLAMA_CPP_PROVIDER_ID,
            display_name="llama.cpp",
            description=(
                "GGUF runtime for GPU-resident, CPU/GPU hybrid, and "
                "heterogeneous multi-GPU inference."
            ),
        )

    @property
    def info(self) -> RuntimeProviderInfo:
        return self._info

    def compatibility(
        self,
        context: RuntimeCompatibilityContext,
    ) -> RuntimeCompatibility:
        reasons: list[CompatibilityReason] = []
        demand = context.demand
        model = demand.model
        resources = context.worker.resources
        policy = _launch_policy(context, self.config)

        if model.model_format != "gguf":
            reasons.append(
                CompatibilityReason(
                    code="model-format-unsupported",
                    message="llama.cpp provider requires model_format=gguf",
                )
            )

        if demand.residency_policy is ResidencyPolicy.VRAM_ONLY:
            if model.estimated_size_mb is None:
                reasons.append(
                    CompatibilityReason(
                        code="model-size-required-for-vram-only",
                        message=(
                            "llama.cpp vram_only planning requires an "
                            "estimated GGUF size"
                        ),
                    )
                )
            elif demand.gpu_topology is GPUTopology.SINGLE_GPU:
                if model.estimated_size_mb > resources.max_single_gpu_vram_mb:
                    reasons.append(
                        CompatibilityReason(
                            code="model-exceeds-single-gpu-vram",
                            message=(
                                "estimated GGUF size exceeds the selected "
                                "single GPU VRAM"
                            ),
                        )
                    )
            elif model.estimated_size_mb > resources.total_vram_mb:
                reasons.append(
                    CompatibilityReason(
                        code="model-exceeds-total-vram",
                        message=(
                            "estimated GGUF size exceeds Worker total VRAM"
                        ),
                    )
                )

            if policy.gpu_layers != "all" or policy.fit:
                reasons.append(
                    CompatibilityReason(
                        code="vram-only-policy-overridden",
                        message=(
                            "vram_only requires gpu_layers=all and fit=off"
                        ),
                    )
                )
            if (
                policy.cpu_moe
                or policy.n_cpu_moe not in (None, 0)
                or policy.n_cpu_ffn not in (None, 0)
            ):
                reasons.append(
                    CompatibilityReason(
                        code="cpu-offload-conflicts-with-vram-only",
                        message=(
                            "CPU FFN/MoE offload options conflict with vram_only"
                        ),
                    )
                )

        if (
            demand.gpu_topology is not GPUTopology.MULTI_GPU
            and policy.tensor_split is not None
        ):
            reasons.append(
                CompatibilityReason(
                    code="tensor-split-requires-multi-gpu",
                    message="tensor_split is only valid for multi_gpu execution",
                )
            )

        if (
            resources.gpu_count > 0
            and policy.split_mode
            in {LlamaCppSplitMode.NONE, LlamaCppSplitMode.ROW}
            and policy.main_gpu >= resources.gpu_count
        ):
            reasons.append(
                CompatibilityReason(
                    code="main-gpu-outside-visible-set",
                    message=(
                        "main_gpu must refer to a GPU inside the Worker-visible set"
                    ),
                )
            )

        if demand.gpu_topology is GPUTopology.SINGLE_GPU:
            if policy.split_mode is not LlamaCppSplitMode.NONE:
                reasons.append(
                    CompatibilityReason(
                        code="single-gpu-split-mode-invalid",
                        message="single_gpu requires split_mode=none",
                    )
                )

        if demand.gpu_topology is GPUTopology.MULTI_GPU:
            if policy.split_mode is LlamaCppSplitMode.NONE:
                reasons.append(
                    CompatibilityReason(
                        code="multi-gpu-split-mode-invalid",
                        message="multi_gpu requires a split mode other than none",
                    )
                )
            if policy.tensor_split is not None and (
                len(policy.tensor_split) != resources.gpu_count
            ):
                reasons.append(
                    CompatibilityReason(
                        code="tensor-split-length-mismatch",
                        message=(
                            "tensor_split must contain one proportion per "
                            "Worker GPU"
                        ),
                    )
                )
            if policy.fit_target_mb is not None and len(
                policy.fit_target_mb
            ) not in {1, resources.gpu_count}:
                reasons.append(
                    CompatibilityReason(
                        code="fit-target-length-mismatch",
                        message=(
                            "fit_target_mb must contain one value or one value "
                            "per Worker GPU"
                        ),
                    )
                )
            if policy.split_mode is LlamaCppSplitMode.TENSOR:
                reasons.append(
                    CompatibilityReason(
                        code="tensor-split-experimental",
                        message=(
                            "llama.cpp split_mode=tensor is upstream "
                            "experimental"
                        ),
                        blocking=False,
                    )
                )
            elif policy.tensor_split is None:
                reasons.append(
                    CompatibilityReason(
                        code="heterogeneous-auto-fit",
                        message=(
                            "llama.cpp will fit/split across visible GPUs using "
                            "its own device-memory information"
                        ),
                        blocking=False,
                    )
                )

        if demand.gpu_topology is GPUTopology.NONE:
            if self.config.tensor_split is not None:
                reasons.append(
                    CompatibilityReason(
                        code="cpu-only-tensor-split-invalid",
                        message="CPU-only execution cannot use tensor_split",
                    )
                )

        if model.topology is ModelTopology.DENSE:
            if policy.cpu_moe or policy.n_cpu_moe not in (None, 0):
                reasons.append(
                    CompatibilityReason(
                        code="moe-offload-on-dense-model",
                        message=(
                            "cpu_moe/n_cpu_moe are only valid for MoE models"
                        ),
                    )
                )
        elif model.topology is ModelTopology.MOE:
            if policy.n_cpu_ffn not in (None, 0):
                reasons.append(
                    CompatibilityReason(
                        code="dense-ffn-offload-on-moe-model",
                        message=(
                            "n_cpu_ffn is reserved for dense FFN offload; "
                            "use cpu_moe/n_cpu_moe for MoE expert weights"
                        ),
                    )
                )

        if demand.residency_policy is ResidencyPolicy.CPU_GPU_HYBRID:
            if (
                model.estimated_size_mb is not None
                and model.estimated_size_mb
                > resources.total_vram_mb + context.host.host_ram_mb
            ):
                reasons.append(
                    CompatibilityReason(
                        code="model-exceeds-combined-memory",
                        message=(
                            "estimated GGUF size exceeds combined Worker VRAM "
                            "and host RAM"
                        ),
                    )
                )
            if (
                self.config.gpu_layers is None
                and not policy.cpu_moe
                and policy.n_cpu_moe is None
                and policy.n_cpu_ffn is None
            ):
                reasons.append(
                    CompatibilityReason(
                        code="hybrid-offload-auto",
                        message=(
                            "llama.cpp will choose GPU layer placement "
                            "automatically; set provider offload options for an "
                            "explicit partition"
                        ),
                        blocking=False,
                    )
                )

        return RuntimeCompatibility(
            provider_id=LLAMA_CPP_PROVIDER_ID,
            reasons=tuple(reasons),
        )

    def setup_intent(
        self,
        context: RuntimeCompatibilityContext,
    ) -> RuntimeSetupIntent:
        report = self.compatibility(context)
        if not report.compatible:
            raise RuntimeError(
                "cannot build llama.cpp setup intent for incompatible demand"
            )
        policy = _launch_policy(context, self.config)
        return RuntimeSetupIntent(
            provider_id=LLAMA_CPP_PROVIDER_ID,
            package_references=(self.config.package_reference,),
            configuration={
                "base_url": self.config.base_url,
                "executable": self.config.executable,
                "model_ref": context.demand.model.model_ref,
                "model_alias": "astrumweaver",
                "context_size": self.config.context_size,
                "gpu_uuids": list(context.worker.gpu_uuids),
                "gpu_layers": policy.gpu_layers,
                "split_mode": policy.split_mode.value,
                "tensor_split": (
                    list(policy.tensor_split)
                    if policy.tensor_split is not None
                    else None
                ),
                "main_gpu": policy.main_gpu,
                "fit": policy.fit,
                "fit_target_mb": (
                    list(policy.fit_target_mb)
                    if policy.fit_target_mb is not None
                    else None
                ),
                "cpu_moe": policy.cpu_moe,
                "n_cpu_moe": policy.n_cpu_moe,
                "n_cpu_ffn": policy.n_cpu_ffn,
                "offline": self.config.offline,
                "no_webui": self.config.no_webui,
                "jinja": self.config.jinja,
                "chat_template_file": self.config.chat_template_file,
                "capabilities": sorted(LLAMA_CPP_CAPABILITIES),
            },
            model_preparation=ModelPreparationPolicy.REFERENCE_ONLY,
            model_ref=context.demand.model.model_ref,
            requires_privilege=True,
            metadata={"runtime_provider": LLAMA_CPP_PROVIDER_ID},
        )

    def create_runtime(
        self,
        context: RuntimeCompatibilityContext,
        setup: RuntimeSetupIntent,
    ) -> ManagedRuntime:
        if setup.provider_id != LLAMA_CPP_PROVIDER_ID:
            raise ValueError("setup intent does not belong to llama.cpp provider")
        report = self.compatibility(context)
        if not report.compatible:
            raise RuntimeError(
                "cannot create llama.cpp runtime for incompatible demand"
            )

        cfg = dict(setup.configuration)
        base_url = str(cfg.get("base_url") or self.config.base_url)
        model_ref = str(
            cfg.get("model_ref") or context.demand.model.model_ref
        )
        model_alias = str(cfg.get("model_alias") or "astrumweaver")
        executable = str(cfg.get("executable") or self.config.executable)
        jinja = cfg.get("jinja", self.config.jinja)
        if type(jinja) is not bool:
            raise ValueError("llama.cpp deployment jinja must be boolean")
        raw_template = cfg.get(
            "chat_template_file",
            self.config.chat_template_file,
        )
        chat_template_file = (
            None
            if raw_template is None
            else _nonblank(str(raw_template), "chat_template_file")
        )
        if chat_template_file is not None and not jinja:
            raise ValueError(
                "llama.cpp chat_template_file requires jinja=true"
            )

        split_mode = LlamaCppSplitMode(
            str(cfg.get("split_mode") or _launch_policy(context, self.config).split_mode)
        )
        tensor_split_raw = cfg.get("tensor_split")
        fit_target_raw = cfg.get("fit_target_mb")
        policy = LlamaCppLaunchPolicy(
            gpu_layers=cfg.get(
                "gpu_layers",
                _launch_policy(context, self.config).gpu_layers,
            ),
            split_mode=split_mode,
            fit=bool(cfg.get("fit", _launch_policy(context, self.config).fit)),
            tensor_split=(
                tuple(float(value) for value in tensor_split_raw)
                if isinstance(tensor_split_raw, (list, tuple))
                else None
            ),
            fit_target_mb=(
                tuple(int(value) for value in fit_target_raw)
                if isinstance(fit_target_raw, (list, tuple))
                else None
            ),
            main_gpu=int(cfg.get("main_gpu", self.config.main_gpu)),
            cpu_moe=bool(cfg.get("cpu_moe", self.config.cpu_moe)),
            n_cpu_moe=(
                int(cfg["n_cpu_moe"])
                if cfg.get("n_cpu_moe") is not None
                else None
            ),
            n_cpu_ffn=(
                int(cfg["n_cpu_ffn"])
                if cfg.get("n_cpu_ffn") is not None
                else None
            ),
        )

        api = self._api_factory(base_url)
        process = self._process_factory(
            executable=executable,
            base_url=base_url,
            model_ref=model_ref,
            model_alias=model_alias,
            gpu_uuids=context.worker.gpu_uuids,
            context_size=int(
                cfg.get("context_size", self.config.context_size)
            ),
            launch_policy=policy,
            offline=bool(cfg.get("offline", self.config.offline)),
            no_webui=bool(cfg.get("no_webui", self.config.no_webui)),
            jinja=jinja,
            chat_template_file=chat_template_file,
        )

        return LlamaCppManagedRuntime(
            api=api,
            process=process,
            context=context,
            model_ref=model_ref,
            model_alias=model_alias,
            startup_timeout_seconds=self.config.startup_timeout_seconds,
            launch_policy=policy,
            tools_enabled=jinja,
        )
