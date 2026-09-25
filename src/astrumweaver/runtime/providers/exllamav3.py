"""First-class ExLlamaV3 RuntimeProvider via the official TabbyAPI server."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx

from ...execution import JobExecutor, JobRequest, JobResult, ResidencyItem, ResidencyReport
from ..contracts import (
    CompatibilityReason,
    GPUTopology,
    ManagedRuntime,
    ModelPreparationPolicy,
    ResidencyPolicy,
    RuntimeCompatibility,
    RuntimeCompatibilityContext,
    RuntimeHealth,
    RuntimeHealthState,
    RuntimeProvider,
    RuntimeProviderInfo,
    RuntimeSetupIntent,
)


EXLLAMAV3_PROVIDER_ID = "exllamav3"
EXLLAMAV3_CAPABILITIES = frozenset({"llm.chat", "text.generate"})
EXLLAMAV3_MODEL_FORMATS = frozenset({"exl3", "exllamav3"})


def _nonblank(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _local_model_reference(model_ref: str) -> bool:
    path = Path(model_ref)
    return (
        path.is_absolute()
        or model_ref.startswith("./")
        or model_ref.startswith("../")
    )


def _model_preparation(model_ref: str) -> ModelPreparationPolicy:
    if _local_model_reference(model_ref):
        return ModelPreparationPolicy.REFERENCE_ONLY
    return ModelPreparationPolicy.DOWNLOAD


def _model_location(model_ref: str, default_model_dir: str) -> tuple[str, str]:
    if _local_model_reference(model_ref):
        path = Path(model_ref)
        return str(path.parent), path.name
    return default_model_dir, model_ref.rstrip("/").split("/")[-1]


class ExLlamaMultiGpuMode(StrEnum):
    AUTOSPLIT = "autosplit"
    TENSOR_PARALLEL = "tensor_parallel"


@dataclass(frozen=True, slots=True)
class ExLlamaV3ProviderConfig:
    base_url: str = "http://127.0.0.1:5000"
    executable: str = "python"
    entrypoint: str = "/opt/tabbyAPI/main.py"
    config_path: str = "/var/lib/astrumweaver/runtime/exllamav3/config.yml"
    package_references: tuple[str, ...] = ("tabbyAPI[cu12]", "exllamav3")
    startup_timeout_seconds: float = 300.0
    request_timeout_seconds: float = 300.0
    model_dir: str = "/var/lib/astrumweaver/models/exllamav3"
    multi_gpu_mode: ExLlamaMultiGpuMode = ExLlamaMultiGpuMode.AUTOSPLIT
    tensor_parallel_backend: str = "native"
    gpu_split: tuple[float, ...] | None = None
    cache_mode: str = "FP16"
    max_seq_len: int | None = None
    max_batch_size: int | None = None
    autosplit_reserve_mb: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        base_url = _nonblank(self.base_url, "base_url").rstrip("/")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.port is None
        ):
            raise ValueError(
                "ExLlamaV3 provider requires a local loopback HTTP endpoint with port"
            )
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "executable", _nonblank(self.executable, "executable"))
        object.__setattr__(self, "entrypoint", _nonblank(self.entrypoint, "entrypoint"))
        object.__setattr__(self, "config_path", _nonblank(self.config_path, "config_path"))
        object.__setattr__(self, "model_dir", _nonblank(self.model_dir, "model_dir"))
        object.__setattr__(
            self,
            "package_references",
            tuple(_nonblank(value, "package reference") for value in self.package_references),
        )
        object.__setattr__(
            self,
            "multi_gpu_mode",
            ExLlamaMultiGpuMode(self.multi_gpu_mode),
        )
        backend = _nonblank(self.tensor_parallel_backend, "tensor_parallel_backend")
        if backend not in {"native", "nccl"}:
            raise ValueError("tensor_parallel_backend must be native or nccl")
        object.__setattr__(self, "tensor_parallel_backend", backend)
        object.__setattr__(self, "cache_mode", _nonblank(self.cache_mode, "cache_mode"))

        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.max_seq_len is not None and self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive when set")
        if self.max_batch_size is not None and self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive when set")

        if self.gpu_split is not None:
            split = tuple(float(value) for value in self.gpu_split)
            if not split or any(value <= 0 for value in split):
                raise ValueError("gpu_split values must be positive")
            object.__setattr__(self, "gpu_split", split)

        if self.autosplit_reserve_mb is not None:
            reserve = tuple(int(value) for value in self.autosplit_reserve_mb)
            if not reserve or any(value < 0 for value in reserve):
                raise ValueError("autosplit_reserve_mb values must be non-negative")
            object.__setattr__(self, "autosplit_reserve_mb", reserve)


@dataclass(frozen=True, slots=True)
class ExLlamaV3LaunchPolicy:
    gpu_uuids: tuple[str, ...]
    multi_gpu_mode: ExLlamaMultiGpuMode
    tensor_parallel: bool
    tensor_parallel_backend: str
    gpu_split_auto: bool
    gpu_split: tuple[float, ...] | None
    autosplit_reserve_mb: tuple[int, ...] | None
    cache_mode: str
    max_seq_len: int | None
    max_batch_size: int | None


def _launch_policy(
    context: RuntimeCompatibilityContext,
    config: ExLlamaV3ProviderConfig,
) -> ExLlamaV3LaunchPolicy:
    multi = context.demand.gpu_topology is GPUTopology.MULTI_GPU
    tensor_parallel = multi and config.multi_gpu_mode is ExLlamaMultiGpuMode.TENSOR_PARALLEL
    gpu_split_auto = multi and config.gpu_split is None
    return ExLlamaV3LaunchPolicy(
        gpu_uuids=context.worker.gpu_uuids,
        multi_gpu_mode=config.multi_gpu_mode,
        tensor_parallel=tensor_parallel,
        tensor_parallel_backend=config.tensor_parallel_backend,
        gpu_split_auto=gpu_split_auto,
        gpu_split=config.gpu_split,
        autosplit_reserve_mb=config.autosplit_reserve_mb,
        cache_mode=config.cache_mode,
        max_seq_len=config.max_seq_len,
        max_batch_size=config.max_batch_size,
    )


def _tabby_config(
    *,
    base_url: str,
    model_dir: str,
    model_name: str,
    policy: ExLlamaV3LaunchPolicy,
) -> dict[str, Any]:
    parsed = urlsplit(base_url)
    assert parsed.hostname is not None
    assert parsed.port is not None
    model: dict[str, Any] = {
        "model_dir": model_dir,
        "model_name": model_name,
        "backend": "exllamav3",
        "inline_model_loading": False,
        "tensor_parallel": policy.tensor_parallel,
        "tensor_parallel_backend": policy.tensor_parallel_backend,
        "gpu_split_auto": policy.gpu_split_auto,
        "gpu_split": list(policy.gpu_split or ()),
        "cache_mode": policy.cache_mode,
    }
    if policy.autosplit_reserve_mb is not None:
        model["autosplit_reserve"] = list(policy.autosplit_reserve_mb)
    if policy.max_seq_len is not None:
        model["max_seq_len"] = policy.max_seq_len
    if policy.max_batch_size is not None:
        model["max_batch_size"] = policy.max_batch_size

    return {
        "network": {
            "host": parsed.hostname,
            "port": parsed.port,
            "disable_auth": True,
            "allowed_origins": [],
            "api_servers": ["OAI"],
            "access_log": False,
        },
        "logging": {
            "log_prompt": False,
            "log_generation_params": False,
            "log_requests": False,
            "log_chat_completion_requests": False,
        },
        "model": model,
        "sampling": {"override_preset": "safe_defaults"},
    }


@runtime_checkable
class ExLlamaV3Api(Protocol):
    async def health(self) -> bool: ...

    async def models(self) -> tuple[str, ...]: ...

    async def chat(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def completion(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def close(self) -> None: ...


class HttpExLlamaV3Api:
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
            raise RuntimeError("TabbyAPI returned a non-object response")
        return body

    async def health(self) -> bool:
        try:
            body = await self._json("GET", "/health")
        except (httpx.HTTPError, RuntimeError, ValueError):
            return False
        return body.get("status") == "healthy"

    async def models(self) -> tuple[str, ...]:
        body = await self._json("GET", "/v1/models")
        data = body.get("data", ())
        if not isinstance(data, list):
            raise RuntimeError("TabbyAPI /v1/models returned invalid data")
        result: list[str] = []
        for item in data:
            if not isinstance(item, Mapping):
                continue
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id.strip():
                result.append(model_id.strip())
        return tuple(result)

    async def chat(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return await self._json("POST", "/v1/chat/completions", json=payload)

    async def completion(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return await self._json("POST", "/v1/completions", json=payload)

    async def close(self) -> None:
        await self._client.aclose()


@runtime_checkable
class ExLlamaV3ProcessController(Protocol):
    @property
    def running(self) -> bool: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class ExLlamaV3SubprocessController:
    def __init__(
        self,
        *,
        executable: str,
        entrypoint: str,
        config_path: str,
        gpu_uuids: tuple[str, ...],
    ) -> None:
        self.executable = executable
        self.entrypoint = entrypoint
        self.config_path = config_path
        self.gpu_uuids = gpu_uuids
        self._process: asyncio.subprocess.Process | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def command(self) -> tuple[str, ...]:
        return (
            self.executable,
            self.entrypoint,
            "--config",
            self.config_path,
        )

    async def start(self) -> None:
        if self.running:
            return
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpu_uuids)
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.command(),
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            raise RuntimeError("failed to start ExLlamaV3 TabbyAPI process") from exc

    async def stop(self) -> None:
        process = self._process
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        self._process = None


class ExLlamaV3Executor(JobExecutor):
    capabilities = EXLLAMAV3_CAPABILITIES

    def __init__(
        self,
        *,
        api: ExLlamaV3Api,
        model_ref: str,
        served_model_name: str,
        residency_metadata: Mapping[str, Any],
    ) -> None:
        self.api = api
        self.model_ref = _nonblank(model_ref, "model_ref")
        self.served_model_name = _nonblank(served_model_name, "served_model_name")
        self.residency_metadata = dict(residency_metadata)
        self._inflight: dict[str, asyncio.Task[Mapping[str, Any]]] = {}

    def _payload(self, job: JobRequest) -> dict[str, Any]:
        payload = dict(job.payload)
        supplied_model = payload.pop("model", None)
        if supplied_model is not None and str(supplied_model) not in {
            self.model_ref,
            self.served_model_name,
        }:
            raise ValueError(
                "job model does not match the model bound to this ExLlamaV3 runtime"
            )
        payload["model"] = self.served_model_name
        payload["stream"] = False
        return payload

    async def execute(self, job: JobRequest) -> JobResult:
        if job.capability not in self.capabilities:
            raise ValueError(
                f"ExLlamaV3 executor does not support capability: {job.capability}"
            )

        payload = self._payload(job)
        if job.capability == "llm.chat":
            if not isinstance(payload.get("messages"), list):
                raise ValueError("llm.chat requires a messages list")
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
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[key] = value

        return JobResult(
            outputs=dict(response),
            metrics=metrics,
            text=text,
            metadata={
                "runtime_provider": EXLLAMAV3_PROVIDER_ID,
                "model": self.model_ref,
                "served_model_name": self.served_model_name,
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
                    kind="exl3-model",
                    metadata=dict(self.residency_metadata),
                ),
            ),
            metadata={"runtime_provider": EXLLAMAV3_PROVIDER_ID},
        )


class ExLlamaV3ManagedRuntime(ManagedRuntime):
    provider_id = EXLLAMAV3_PROVIDER_ID

    def __init__(
        self,
        *,
        api: ExLlamaV3Api,
        process: ExLlamaV3ProcessController,
        context: RuntimeCompatibilityContext,
        model_ref: str,
        served_model_name: str,
        startup_timeout_seconds: float,
        launch_policy: ExLlamaV3LaunchPolicy,
    ) -> None:
        self.api = api
        self.process = process
        self.context = context
        self.model_ref = model_ref
        self.served_model_name = served_model_name
        self.startup_timeout_seconds = startup_timeout_seconds
        self.launch_policy = launch_policy
        self._closed = False
        self._executor = ExLlamaV3Executor(
            api=api,
            model_ref=model_ref,
            served_model_name=served_model_name,
            residency_metadata={
                "residency_policy": context.demand.residency_policy.value,
                "gpu_topology": context.demand.gpu_topology.value,
                "gpu_count": context.worker.resources.gpu_count,
                "gpu_uuids": list(launch_policy.gpu_uuids),
                "multi_gpu_mode": launch_policy.multi_gpu_mode.value,
                "tensor_parallel": launch_policy.tensor_parallel,
                "tensor_parallel_backend": launch_policy.tensor_parallel_backend,
                "gpu_split_auto": launch_policy.gpu_split_auto,
                "gpu_split": (
                    list(launch_policy.gpu_split)
                    if launch_policy.gpu_split is not None
                    else None
                ),
                "cache_mode": launch_policy.cache_mode,
            },
        )

    async def _server_reachable(self) -> bool:
        return await self.api.health()

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("ExLlamaV3 runtime has already been released")

        reachable = await self._server_reachable()
        if reachable and not self.process.running:
            raise RuntimeError(
                "an external TabbyAPI server is already using the configured endpoint"
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
            while not await self._server_reachable():
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError(
                        "ExLlamaV3 TabbyAPI server did not become ready before timeout"
                    )
                await asyncio.sleep(0.2)

            models = await self.api.models()
            if self.served_model_name not in models:
                raise RuntimeError(
                    "TabbyAPI did not expose the configured ExLlamaV3 model"
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
                metadata={"runtime_provider": EXLLAMAV3_PROVIDER_ID},
            )
        if not self.process.running:
            return RuntimeHealth(
                state=RuntimeHealthState.FAILED,
                ready=False,
                detail="external TabbyAPI process detected at managed endpoint",
                metadata={"runtime_provider": EXLLAMAV3_PROVIDER_ID},
            )
        try:
            models = await self.api.models()
        except Exception:
            return RuntimeHealth(
                state=RuntimeHealthState.DEGRADED,
                ready=False,
                detail="TabbyAPI model inventory is unavailable",
                metadata={"runtime_provider": EXLLAMAV3_PROVIDER_ID},
            )
        if self.served_model_name not in models:
            return RuntimeHealth(
                state=RuntimeHealthState.FAILED,
                ready=False,
                detail="configured ExLlamaV3 model is unavailable",
                metadata={"runtime_provider": EXLLAMAV3_PROVIDER_ID},
            )
        return RuntimeHealth(
            state=RuntimeHealthState.READY,
            ready=True,
            metadata={
                "runtime_provider": EXLLAMAV3_PROVIDER_ID,
                "model": self.model_ref,
                "served_model_name": self.served_model_name,
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


class ExLlamaV3Provider(RuntimeProvider):
    def __init__(
        self,
        config: ExLlamaV3ProviderConfig | None = None,
        *,
        api_factory: Any | None = None,
        process_factory: Any | None = None,
    ) -> None:
        self.config = config or ExLlamaV3ProviderConfig()
        self._api_factory = api_factory or (
            lambda base_url: HttpExLlamaV3Api(
                base_url,
                timeout_seconds=self.config.request_timeout_seconds,
            )
        )
        self._process_factory = process_factory or (
            lambda **kwargs: ExLlamaV3SubprocessController(**kwargs)
        )
        self._info = RuntimeProviderInfo(
            provider_id=EXLLAMAV3_PROVIDER_ID,
            display_name="ExLlamaV3",
            description=(
                "EXL3 consumer-NVIDIA VRAM-resident serving through the "
                "official TabbyAPI backend."
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
        resources = context.worker.resources
        policy = _launch_policy(context, self.config)

        if context.host.architecture.lower() not in {"x86_64", "amd64"}:
            reasons.append(
                CompatibilityReason(
                    code="architecture-unsupported",
                    message=(
                        "current ExLlamaV3/TabbyAPI Linux wheel path targets x86_64"
                    ),
                )
            )

        if demand.model.model_format not in EXLLAMAV3_MODEL_FORMATS:
            reasons.append(
                CompatibilityReason(
                    code="model-format-unsupported",
                    message=(
                        "AstrumWeaver ExLlamaV3 provider requires an EXL3 model"
                    ),
                )
            )

        compute_capability = context.worker.labels.get(
            "gpu.compute_capability.min"
        )
        if compute_capability is None:
            reasons.append(
                CompatibilityReason(
                    code="compute-capability-unverified",
                    message=(
                        "ExLlamaV3 requires NVIDIA compute capability 8.0 or newer; "
                        "Worker metadata does not currently prove the minimum capability"
                    ),
                    blocking=False,
                )
            )
        else:
            try:
                major = int(str(compute_capability).split(".", 1)[0])
            except ValueError:
                reasons.append(
                    CompatibilityReason(
                        code="compute-capability-invalid",
                        message=(
                            "gpu.compute_capability.min must use a numeric value such "
                            "as 8.6"
                        ),
                    )
                )
            else:
                if major < 8:
                    reasons.append(
                        CompatibilityReason(
                            code="compute-capability-unsupported",
                            message=(
                                "ExLlamaV3 requires NVIDIA compute capability 8.0 "
                                "or newer"
                            ),
                        )
                    )

        if demand.gpu_topology is GPUTopology.NONE or resources.gpu_count == 0:
            reasons.append(
                CompatibilityReason(
                    code="gpu-required",
                    message="ExLlamaV3 requires an NVIDIA GPU Worker",
                )
            )

        if len(context.worker.gpu_uuids) != resources.gpu_count:
            reasons.append(
                CompatibilityReason(
                    code="gpu-identity-count-mismatch",
                    message="Worker GPU UUID ownership does not match GPU resource count",
                )
            )

        if demand.gpu_topology is GPUTopology.SINGLE_GPU:
            if resources.gpu_count != 1:
                reasons.append(
                    CompatibilityReason(
                        code="single-gpu-required",
                        message="single_gpu ExLlamaV3 demand requires exactly one Worker GPU",
                    )
                )
            if policy.gpu_split is not None:
                reasons.append(
                    CompatibilityReason(
                        code="gpu-split-requires-multi-gpu",
                        message="explicit ExLlamaV3 gpu_split requires multi_gpu demand",
                    )
                )

        if policy.autosplit_reserve_mb is not None:
            if len(policy.autosplit_reserve_mb) != resources.gpu_count:
                reasons.append(
                    CompatibilityReason(
                        code="autosplit-reserve-length-mismatch",
                        message=(
                            "ExLlamaV3 autosplit_reserve_mb must provide one value "
                            "for every Worker-owned GPU"
                        ),
                    )
                )

        if demand.gpu_topology is GPUTopology.MULTI_GPU:
            if resources.gpu_count < 2:
                reasons.append(
                    CompatibilityReason(
                        code="multi-gpu-required",
                        message="multi_gpu ExLlamaV3 demand requires at least two Worker GPUs",
                    )
                )
            if policy.gpu_split is not None and len(policy.gpu_split) != resources.gpu_count:
                reasons.append(
                    CompatibilityReason(
                        code="gpu-split-length-mismatch",
                        message=(
                            "ExLlamaV3 gpu_split must provide one value for every "
                            "Worker-owned GPU"
                        ),
                    )
                )
            reasons.append(
                CompatibilityReason(
                    code=(
                        "tensor-parallel-enabled"
                        if policy.tensor_parallel
                        else "multi-gpu-autosplit-enabled"
                    ),
                    message=(
                        "ExLlamaV3 will use all Worker-owned GPUs with "
                        + (
                            f"{policy.tensor_parallel_backend} tensor parallelism"
                            if policy.tensor_parallel
                            else "TabbyAPI/ExLlamaV3 autosplit"
                        )
                    ),
                    blocking=False,
                )
            )

        if demand.residency_policy is ResidencyPolicy.CPU_GPU_HYBRID:
            reasons.append(
                CompatibilityReason(
                    code="cpu-gpu-hybrid-outside-provider-scope",
                    message=(
                        "AstrumWeaver v0.x ExLlamaV3 provider is scoped to "
                        "VRAM-resident/prefer-VRAM execution; use FreeToken or "
                        "llama.cpp for RAM-heavy offload"
                    ),
                )
            )

        if (
            demand.residency_policy is ResidencyPolicy.VRAM_ONLY
            and demand.model.estimated_size_mb is None
        ):
            reasons.append(
                CompatibilityReason(
                    code="model-size-required-for-vram-only",
                    message=(
                        "ExLlamaV3 vram_only planning requires estimated model size"
                    ),
                )
            )

        if (
            demand.model.estimated_size_mb is not None
            and demand.model.estimated_size_mb > resources.total_vram_mb
        ):
            reasons.append(
                CompatibilityReason(
                    code="model-exceeds-total-vram",
                    message=(
                        "estimated EXL3 model size exceeds Worker total VRAM before "
                        "KV cache and runtime overhead"
                    ),
                )
            )
        elif demand.residency_policy is ResidencyPolicy.VRAM_ONLY:
            reasons.append(
                CompatibilityReason(
                    code="vram-overhead-not-proven-by-model-size",
                    message=(
                        "model-size fit does not include KV cache/runtime overhead; "
                        "declare explicit VRAM minima for a hard deployment floor"
                    ),
                    blocking=False,
                )
            )

        return RuntimeCompatibility(
            provider_id=EXLLAMAV3_PROVIDER_ID,
            reasons=tuple(reasons),
        )

    def setup_intent(
        self,
        context: RuntimeCompatibilityContext,
    ) -> RuntimeSetupIntent:
        report = self.compatibility(context)
        if not report.compatible:
            raise RuntimeError(
                "cannot build ExLlamaV3 setup intent for incompatible demand"
            )

        policy = _launch_policy(context, self.config)
        model_dir, model_name = _model_location(
            context.demand.model.model_ref,
            self.config.model_dir,
        )
        tabby_config = _tabby_config(
            base_url=self.config.base_url,
            model_dir=model_dir,
            model_name=model_name,
            policy=policy,
        )

        return RuntimeSetupIntent(
            provider_id=EXLLAMAV3_PROVIDER_ID,
            package_references=self.config.package_references,
            configuration={
                "base_url": self.config.base_url,
                "executable": self.config.executable,
                "entrypoint": self.config.entrypoint,
                "config_path": self.config.config_path,
                "model_ref": context.demand.model.model_ref,
                "model_dir": model_dir,
                "served_model_name": model_name,
                "gpu_uuids": list(policy.gpu_uuids),
                "multi_gpu_mode": policy.multi_gpu_mode.value,
                "tensor_parallel": policy.tensor_parallel,
                "tensor_parallel_backend": policy.tensor_parallel_backend,
                "gpu_split_auto": policy.gpu_split_auto,
                "gpu_split": (
                    list(policy.gpu_split)
                    if policy.gpu_split is not None
                    else None
                ),
                "autosplit_reserve_mb": (
                    list(policy.autosplit_reserve_mb)
                    if policy.autosplit_reserve_mb is not None
                    else None
                ),
                "cache_mode": policy.cache_mode,
                "max_seq_len": policy.max_seq_len,
                "max_batch_size": policy.max_batch_size,
                "tabby_config": tabby_config,
                "capabilities": sorted(EXLLAMAV3_CAPABILITIES),
            },
            model_preparation=_model_preparation(context.demand.model.model_ref),
            model_ref=context.demand.model.model_ref,
            requires_privilege=True,
            metadata={
                "runtime_provider": EXLLAMAV3_PROVIDER_ID,
                "serving_backend": "tabbyapi",
            },
        )

    def create_runtime(
        self,
        context: RuntimeCompatibilityContext,
        setup: RuntimeSetupIntent,
    ) -> ManagedRuntime:
        if setup.provider_id != EXLLAMAV3_PROVIDER_ID:
            raise ValueError("setup intent does not belong to ExLlamaV3 provider")

        report = self.compatibility(context)
        if not report.compatible:
            raise RuntimeError(
                "cannot create ExLlamaV3 runtime for incompatible demand"
            )

        cfg = dict(setup.configuration)
        expected_gpu_uuids = context.worker.gpu_uuids
        configured_gpu_uuids = tuple(str(v) for v in cfg.get("gpu_uuids", ()))
        if configured_gpu_uuids and configured_gpu_uuids != expected_gpu_uuids:
            raise ValueError(
                "setup intent GPU UUIDs do not match ExLlamaV3 Worker ownership"
            )

        policy = _launch_policy(context, self.config)
        configured_mode = ExLlamaMultiGpuMode(
            cfg.get("multi_gpu_mode", policy.multi_gpu_mode.value)
        )
        if configured_mode is not policy.multi_gpu_mode:
            raise ValueError(
                "setup intent multi-GPU mode does not match reviewed ExLlamaV3 policy"
            )

        base_url = str(cfg.get("base_url") or self.config.base_url)
        model_ref = str(cfg.get("model_ref") or context.demand.model.model_ref)
        served_model_name = str(
            cfg.get("served_model_name")
            or _model_location(model_ref, self.config.model_dir)[1]
        )

        api = self._api_factory(base_url)
        process = self._process_factory(
            executable=str(cfg.get("executable") or self.config.executable),
            entrypoint=str(cfg.get("entrypoint") or self.config.entrypoint),
            config_path=str(cfg.get("config_path") or self.config.config_path),
            gpu_uuids=expected_gpu_uuids,
        )

        return ExLlamaV3ManagedRuntime(
            api=api,
            process=process,
            context=context,
            model_ref=model_ref,
            served_model_name=served_model_name,
            startup_timeout_seconds=self.config.startup_timeout_seconds,
            launch_policy=policy,
        )
