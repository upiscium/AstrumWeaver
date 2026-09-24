"""First-class vLLM RuntimeProvider."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
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


VLLM_PROVIDER_ID = "vllm"
VLLM_CAPABILITIES = frozenset({"llm.chat", "text.generate"})
VLLM_MODEL_FORMATS = frozenset(
    {"huggingface", "hf", "safetensors", "vllm"}
)


def _nonblank(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


@dataclass(frozen=True, slots=True)
class VllmProviderConfig:
    base_url: str = "http://127.0.0.1:8000"
    executable: str = "vllm"
    package_reference: str = "vllm"
    startup_timeout_seconds: float = 180.0
    request_timeout_seconds: float = 300.0
    served_model_name: str = "astrumweaver"
    gpu_memory_utilization: float = 0.92
    tensor_parallel_size: int | None = None
    enable_expert_parallel: bool | None = None
    cpu_offload_gb: float = 0.0
    generation_config: str = "vllm"
    trust_remote_code: bool = False
    enforce_eager: bool = False

    def __post_init__(self) -> None:
        base_url = _nonblank(self.base_url, "base_url").rstrip("/")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.port is None
        ):
            raise ValueError(
                "vLLM provider requires a local loopback HTTP endpoint with port"
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
        object.__setattr__(
            self,
            "served_model_name",
            _nonblank(self.served_model_name, "served_model_name"),
        )
        object.__setattr__(
            self,
            "generation_config",
            _nonblank(self.generation_config, "generation_config"),
        )
        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if not (0 < self.gpu_memory_utilization <= 1):
            raise ValueError(
                "gpu_memory_utilization must be greater than 0 and at most 1"
            )
        if (
            self.tensor_parallel_size is not None
            and self.tensor_parallel_size <= 0
        ):
            raise ValueError("tensor_parallel_size must be positive")
        if self.cpu_offload_gb < 0:
            raise ValueError("cpu_offload_gb must not be negative")


@dataclass(frozen=True, slots=True)
class VllmLaunchPolicy:
    tensor_parallel_size: int
    enable_expert_parallel: bool
    gpu_memory_utilization: float
    cpu_offload_gb: float
    device_ids: tuple[str, ...]


def _launch_policy(
    context: RuntimeCompatibilityContext,
    config: VllmProviderConfig,
) -> VllmLaunchPolicy:
    gpu_count = context.worker.resources.gpu_count
    if config.tensor_parallel_size is not None:
        tp_size = config.tensor_parallel_size
    elif context.demand.gpu_topology is GPUTopology.MULTI_GPU:
        tp_size = gpu_count
    else:
        tp_size = 1

    if config.enable_expert_parallel is None:
        enable_ep = (
            context.demand.model.topology is ModelTopology.MOE
            and context.demand.gpu_topology is GPUTopology.MULTI_GPU
        )
    else:
        enable_ep = config.enable_expert_parallel

    return VllmLaunchPolicy(
        tensor_parallel_size=tp_size,
        enable_expert_parallel=enable_ep,
        gpu_memory_utilization=config.gpu_memory_utilization,
        cpu_offload_gb=config.cpu_offload_gb,
        device_ids=context.worker.gpu_uuids,
    )


def _model_preparation(model_ref: str) -> ModelPreparationPolicy:
    path = Path(model_ref)
    if path.is_absolute() or model_ref.startswith("./") or model_ref.startswith("../"):
        return ModelPreparationPolicy.REFERENCE_ONLY
    return ModelPreparationPolicy.DOWNLOAD


@runtime_checkable
class VllmApi(Protocol):
    async def health(self) -> bool: ...

    async def models(self) -> tuple[str, ...]: ...

    async def chat(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def completion(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def close(self) -> None: ...


class HttpVllmApi:
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
            raise RuntimeError("vLLM returned a non-object response")
        return body

    async def models(self) -> tuple[str, ...]:
        body = await self._json("GET", "/v1/models")
        data = body.get("data", ())
        if not isinstance(data, list):
            raise RuntimeError("vLLM /v1/models returned invalid data")
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

    async def completion(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return await self._json(
            "POST",
            "/v1/completions",
            json=payload,
        )

    async def close(self) -> None:
        await self._client.aclose()


@runtime_checkable
class VllmProcessController(Protocol):
    @property
    def running(self) -> bool: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class VllmSubprocessController:
    def __init__(
        self,
        *,
        executable: str,
        base_url: str,
        model_ref: str,
        served_model_name: str,
        launch_policy: VllmLaunchPolicy,
        generation_config: str,
        trust_remote_code: bool,
        enforce_eager: bool,
    ) -> None:
        self.executable = executable
        self.base_url = base_url
        self.model_ref = model_ref
        self.served_model_name = served_model_name
        self.launch_policy = launch_policy
        self.generation_config = generation_config
        self.trust_remote_code = trust_remote_code
        self.enforce_eager = enforce_eager
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
            "serve",
            self.model_ref,
            "--host",
            parsed.hostname,
            "--port",
            str(parsed.port),
            "--served-model-name",
            self.served_model_name,
            "--generation-config",
            self.generation_config,
            "--gpu-memory-utilization",
            str(self.launch_policy.gpu_memory_utilization),
        ]

        if self.launch_policy.device_ids:
            args.extend(
                [
                    "--device-ids",
                    ",".join(self.launch_policy.device_ids),
                ]
            )

        if self.launch_policy.tensor_parallel_size > 1:
            args.extend(
                [
                    "--tensor-parallel-size",
                    str(self.launch_policy.tensor_parallel_size),
                    "--distributed-executor-backend",
                    "mp",
                ]
            )

        if self.launch_policy.enable_expert_parallel:
            args.append("--enable-expert-parallel")

        if self.launch_policy.cpu_offload_gb > 0:
            args.extend(
                [
                    "--cpu-offload-gb",
                    str(self.launch_policy.cpu_offload_gb),
                ]
            )
        if self.trust_remote_code:
            args.append("--trust-remote-code")
        if self.enforce_eager:
            args.append("--enforce-eager")

        return tuple(args)

    async def start(self) -> None:
        if self.running:
            return
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.command(),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            raise RuntimeError("failed to start vLLM server process") from exc

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


class VllmExecutor(JobExecutor):
    capabilities = VLLM_CAPABILITIES

    def __init__(
        self,
        *,
        api: VllmApi,
        model_ref: str,
        served_model_name: str,
        residency_metadata: Mapping[str, Any],
    ) -> None:
        self.api = api
        self.model_ref = _nonblank(model_ref, "model_ref")
        self.served_model_name = _nonblank(
            served_model_name,
            "served_model_name",
        )
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
                "job model does not match the model bound to this vLLM runtime"
            )
        payload["model"] = self.served_model_name
        payload["stream"] = False
        return payload

    async def execute(self, job: JobRequest) -> JobResult:
        if job.capability not in self.capabilities:
            raise ValueError(
                f"vLLM executor does not support capability: {job.capability}"
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
                "runtime_provider": VLLM_PROVIDER_ID,
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
                    kind="vllm-model",
                    metadata=dict(self.residency_metadata),
                ),
            ),
            metadata={"runtime_provider": VLLM_PROVIDER_ID},
        )


class VllmManagedRuntime(ManagedRuntime):
    provider_id = VLLM_PROVIDER_ID

    def __init__(
        self,
        *,
        api: VllmApi,
        process: VllmProcessController,
        context: RuntimeCompatibilityContext,
        model_ref: str,
        served_model_name: str,
        startup_timeout_seconds: float,
        launch_policy: VllmLaunchPolicy,
    ) -> None:
        self.api = api
        self.process = process
        self.context = context
        self.model_ref = model_ref
        self.served_model_name = served_model_name
        self.startup_timeout_seconds = startup_timeout_seconds
        self.launch_policy = launch_policy
        self._closed = False
        self._executor = VllmExecutor(
            api=api,
            model_ref=model_ref,
            served_model_name=served_model_name,
            residency_metadata={
                "residency_policy": context.demand.residency_policy.value,
                "gpu_topology": context.demand.gpu_topology.value,
                "gpu_count": context.worker.resources.gpu_count,
                "tensor_parallel_size": (
                    launch_policy.tensor_parallel_size
                ),
                "expert_parallel": launch_policy.enable_expert_parallel,
                "gpu_memory_utilization": (
                    launch_policy.gpu_memory_utilization
                ),
                "cpu_offload_gb_per_gpu": launch_policy.cpu_offload_gb,
                "device_ids": list(launch_policy.device_ids),
            },
        )

    async def _server_reachable(self) -> bool:
        return await self.api.health()

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("vLLM runtime has already been released")

        reachable = await self._server_reachable()
        if reachable and not self.process.running:
            raise RuntimeError(
                "an external vLLM server is already using the configured endpoint"
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
                        "vLLM server did not become ready before timeout"
                    )
                await asyncio.sleep(0.2)

            models = await self.api.models()
            if self.served_model_name not in models:
                raise RuntimeError(
                    "vLLM server did not expose the configured model alias"
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
                metadata={"runtime_provider": VLLM_PROVIDER_ID},
            )
        if not self.process.running:
            return RuntimeHealth(
                state=RuntimeHealthState.FAILED,
                ready=False,
                detail="external vLLM process detected at managed endpoint",
                metadata={"runtime_provider": VLLM_PROVIDER_ID},
            )
        try:
            models = await self.api.models()
        except Exception:
            return RuntimeHealth(
                state=RuntimeHealthState.DEGRADED,
                ready=False,
                detail="vLLM model inventory is unavailable",
                metadata={"runtime_provider": VLLM_PROVIDER_ID},
            )
        if self.served_model_name not in models:
            return RuntimeHealth(
                state=RuntimeHealthState.FAILED,
                ready=False,
                detail="configured vLLM model alias is unavailable",
                metadata={"runtime_provider": VLLM_PROVIDER_ID},
            )
        return RuntimeHealth(
            state=RuntimeHealthState.READY,
            ready=True,
            metadata={
                "runtime_provider": VLLM_PROVIDER_ID,
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


class VllmProvider(RuntimeProvider):
    def __init__(
        self,
        config: VllmProviderConfig | None = None,
        *,
        api_factory: Any | None = None,
        process_factory: Any | None = None,
    ) -> None:
        self.config = config or VllmProviderConfig()
        self._api_factory = api_factory or (
            lambda base_url: HttpVllmApi(
                base_url,
                timeout_seconds=self.config.request_timeout_seconds,
            )
        )
        self._process_factory = process_factory or (
            lambda **kwargs: VllmSubprocessController(**kwargs)
        )
        self._info = RuntimeProviderInfo(
            provider_id=VLLM_PROVIDER_ID,
            display_name="vLLM",
            description=(
                "GPU-resident high-throughput serving with tensor and "
                "expert parallelism."
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

        if model.model_format not in VLLM_MODEL_FORMATS:
            reasons.append(
                CompatibilityReason(
                    code="model-format-unsupported",
                    message=(
                        "vLLM provider requires a Hugging Face/safetensors "
                        "compatible model reference"
                    ),
                )
            )

        if demand.gpu_topology is GPUTopology.NONE:
            reasons.append(
                CompatibilityReason(
                    code="gpu-required",
                    message=(
                        "first-class vLLM provider currently targets GPU Workers"
                    ),
                )
            )

        if demand.gpu_topology is GPUTopology.SINGLE_GPU:
            if policy.tensor_parallel_size != 1:
                reasons.append(
                    CompatibilityReason(
                        code="tensor-parallel-size-mismatch",
                        message="single_gpu execution requires tensor_parallel_size=1",
                    )
                )

        if demand.gpu_topology is GPUTopology.MULTI_GPU:
            if policy.tensor_parallel_size != resources.gpu_count:
                reasons.append(
                    CompatibilityReason(
                        code="tensor-parallel-size-mismatch",
                        message=(
                            "multi_gpu vLLM must use every GPU owned by the Worker"
                        ),
                    )
                )
            if (
                resources.total_vram_mb
                != resources.max_single_gpu_vram_mb * resources.gpu_count
            ):
                reasons.append(
                    CompatibilityReason(
                        code="heterogeneous-vram-unsupported",
                        message=(
                            "vLLM multi-GPU v0.x requires equal VRAM capacity "
                            "across Worker GPUs"
                        ),
                    )
                )

        if policy.enable_expert_parallel:
            if model.topology is not ModelTopology.MOE:
                reasons.append(
                    CompatibilityReason(
                        code="expert-parallel-requires-moe",
                        message="expert parallelism is only valid for MoE models",
                    )
                )
            if resources.gpu_count < 2:
                reasons.append(
                    CompatibilityReason(
                        code="expert-parallel-requires-multi-gpu",
                        message=(
                            "expert parallelism requires a multi-GPU Worker"
                        ),
                    )
                )

        if (
            model.topology is ModelTopology.MOE
            and demand.gpu_topology is GPUTopology.MULTI_GPU
            and policy.enable_expert_parallel
        ):
            reasons.append(
                CompatibilityReason(
                    code="expert-parallel-enabled",
                    message=(
                        "vLLM expert parallelism will be enabled for MoE layers"
                    ),
                    blocking=False,
                )
            )

        offload_total_mb = (
            int(policy.cpu_offload_gb * 1024) * max(resources.gpu_count, 1)
        )
        if policy.cpu_offload_gb > 0 and offload_total_mb > context.host.host_ram_mb:
            reasons.append(
                CompatibilityReason(
                    code="cpu-offload-exceeds-host-ram",
                    message=(
                        "configured vLLM CPU offload exceeds available host RAM"
                    ),
                )
            )

        if demand.residency_policy is ResidencyPolicy.VRAM_ONLY:
            if policy.cpu_offload_gb > 0:
                reasons.append(
                    CompatibilityReason(
                        code="cpu-offload-conflicts-with-vram-only",
                        message="vram_only cannot use vLLM CPU weight offload",
                    )
                )
            if model.estimated_size_mb is None:
                reasons.append(
                    CompatibilityReason(
                        code="model-size-required-for-vram-only",
                        message=(
                            "vLLM vram_only planning requires estimated model size"
                        ),
                    )
                )

        if demand.residency_policy is ResidencyPolicy.CPU_GPU_HYBRID:
            if policy.cpu_offload_gb <= 0:
                reasons.append(
                    CompatibilityReason(
                        code="cpu-offload-not-configured",
                        message=(
                            "vLLM cpu_gpu_hybrid requires explicit cpu_offload_gb"
                        ),
                    )
                )
            else:
                reasons.append(
                    CompatibilityReason(
                        code="cpu-offload-interconnect-sensitive",
                        message=(
                            "vLLM CPU weight offload transfers model data on "
                            "forward passes and benefits from a fast CPU-GPU interconnect"
                        ),
                        blocking=False,
                    )
                )

        if model.estimated_size_mb is not None and resources.gpu_count > 0:
            vram_budget_mb = int(
                resources.total_vram_mb * policy.gpu_memory_utilization
            )
            effective_budget_mb = vram_budget_mb + offload_total_mb
            if model.estimated_size_mb > effective_budget_mb:
                reasons.append(
                    CompatibilityReason(
                        code="model-exceeds-effective-memory-budget",
                        message=(
                            "estimated model size exceeds configured vLLM "
                            "VRAM plus CPU-offload budget"
                        ),
                    )
                )

        return RuntimeCompatibility(
            provider_id=VLLM_PROVIDER_ID,
            reasons=tuple(reasons),
        )

    def setup_intent(
        self,
        context: RuntimeCompatibilityContext,
    ) -> RuntimeSetupIntent:
        report = self.compatibility(context)
        if not report.compatible:
            raise RuntimeError(
                "cannot build vLLM setup intent for incompatible demand"
            )

        policy = _launch_policy(context, self.config)
        return RuntimeSetupIntent(
            provider_id=VLLM_PROVIDER_ID,
            package_references=(self.config.package_reference,),
            configuration={
                "base_url": self.config.base_url,
                "executable": self.config.executable,
                "model_ref": context.demand.model.model_ref,
                "served_model_name": self.config.served_model_name,
                "gpu_uuids": list(context.worker.gpu_uuids),
                "tensor_parallel_size": policy.tensor_parallel_size,
                "enable_expert_parallel": (
                    policy.enable_expert_parallel
                ),
                "gpu_memory_utilization": policy.gpu_memory_utilization,
                "cpu_offload_gb": policy.cpu_offload_gb,
                "generation_config": self.config.generation_config,
                "trust_remote_code": self.config.trust_remote_code,
                "enforce_eager": self.config.enforce_eager,
                "capabilities": sorted(VLLM_CAPABILITIES),
            },
            model_preparation=_model_preparation(
                context.demand.model.model_ref
            ),
            model_ref=context.demand.model.model_ref,
            requires_privilege=True,
            metadata={"runtime_provider": VLLM_PROVIDER_ID},
        )

    def create_runtime(
        self,
        context: RuntimeCompatibilityContext,
        setup: RuntimeSetupIntent,
    ) -> ManagedRuntime:
        if setup.provider_id != VLLM_PROVIDER_ID:
            raise ValueError("setup intent does not belong to vLLM provider")

        report = self.compatibility(context)
        if not report.compatible:
            raise RuntimeError(
                "cannot create vLLM runtime for incompatible demand"
            )

        cfg = dict(setup.configuration)
        base_url = str(cfg.get("base_url") or self.config.base_url)
        model_ref = str(
            cfg.get("model_ref") or context.demand.model.model_ref
        )
        served_model_name = str(
            cfg.get("served_model_name") or self.config.served_model_name
        )
        executable = str(cfg.get("executable") or self.config.executable)

        policy = VllmLaunchPolicy(
            tensor_parallel_size=int(
                cfg.get(
                    "tensor_parallel_size",
                    _launch_policy(context, self.config).tensor_parallel_size,
                )
            ),
            enable_expert_parallel=bool(
                cfg.get(
                    "enable_expert_parallel",
                    _launch_policy(
                        context,
                        self.config,
                    ).enable_expert_parallel,
                )
            ),
            gpu_memory_utilization=float(
                cfg.get(
                    "gpu_memory_utilization",
                    self.config.gpu_memory_utilization,
                )
            ),
            cpu_offload_gb=float(
                cfg.get("cpu_offload_gb", self.config.cpu_offload_gb)
            ),
            device_ids=context.worker.gpu_uuids,
        )

        api = self._api_factory(base_url)
        process = self._process_factory(
            executable=executable,
            base_url=base_url,
            model_ref=model_ref,
            served_model_name=served_model_name,
            launch_policy=policy,
            generation_config=str(
                cfg.get(
                    "generation_config",
                    self.config.generation_config,
                )
            ),
            trust_remote_code=bool(
                cfg.get(
                    "trust_remote_code",
                    self.config.trust_remote_code,
                )
            ),
            enforce_eager=bool(
                cfg.get("enforce_eager", self.config.enforce_eager)
            ),
        )

        return VllmManagedRuntime(
            api=api,
            process=process,
            context=context,
            model_ref=model_ref,
            served_model_name=served_model_name,
            startup_timeout_seconds=self.config.startup_timeout_seconds,
            launch_policy=policy,
        )
