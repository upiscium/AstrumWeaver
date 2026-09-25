"""First-class FreeToken RuntimeProvider."""

from __future__ import annotations

import asyncio
import contextlib
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


FREETOKEN_PROVIDER_ID = "freetoken"
FREETOKEN_CAPABILITIES = frozenset({"llm.chat", "text.generate"})
FREETOKEN_MODEL_FORMATS = frozenset(
    {"huggingface", "hf", "safetensors", "ftw"}
)


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


class FreeTokenMoeStrategy(StrEnum):
    AUTO = "auto"
    OFFLOAD = "offload"
    CPU = "cpu"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class FreeTokenProviderConfig:
    base_url: str = "http://127.0.0.1:1919"
    executable: str = "ft"
    package_reference: str = "freetoken[accel]"
    startup_timeout_seconds: float = 300.0
    request_timeout_seconds: float = 300.0
    served_model_name: str = "astrumweaver"
    moe_strategy: FreeTokenMoeStrategy = FreeTokenMoeStrategy.AUTO
    memory_ratio: float = 0.9
    moe_cache_size: int | None = None
    moe_cache_rate: float | None = None
    moe_cpu_threads: int | None = None
    moe_cpu_layers: str | int | float | None = None
    moe_hybrid_max_fetch: int | None = None
    max_running_requests: int = 1

    def __post_init__(self) -> None:
        base_url = _nonblank(self.base_url, "base_url").rstrip("/")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.port is None
        ):
            raise ValueError(
                "FreeToken provider requires a local loopback HTTP endpoint with port"
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
            "moe_strategy",
            FreeTokenMoeStrategy(self.moe_strategy),
        )

        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if not (0 < self.memory_ratio <= 1):
            raise ValueError("memory_ratio must be greater than 0 and at most 1")
        if self.max_running_requests <= 0:
            raise ValueError("max_running_requests must be positive")
        if self.moe_cache_size is not None and self.moe_cache_size <= 0:
            raise ValueError("moe_cache_size must be positive when set")
        if self.moe_cache_rate is not None and not (0 < self.moe_cache_rate <= 1):
            raise ValueError(
                "moe_cache_rate must be greater than 0 and at most 1 when set"
            )
        if self.moe_cache_size is not None and self.moe_cache_rate is not None:
            raise ValueError(
                "moe_cache_size and moe_cache_rate are mutually exclusive"
            )
        if self.moe_cpu_threads is not None and self.moe_cpu_threads <= 0:
            raise ValueError("moe_cpu_threads must be positive when set")
        if (
            self.moe_hybrid_max_fetch is not None
            and self.moe_hybrid_max_fetch < -1
        ):
            raise ValueError("moe_hybrid_max_fetch must be >= -1 when set")
        if isinstance(self.moe_cpu_layers, bool):
            raise TypeError("moe_cpu_layers must not be boolean")


@dataclass(frozen=True, slots=True)
class FreeTokenLaunchPolicy:
    gpu_uuid: str
    moe_strategy: FreeTokenMoeStrategy
    memory_ratio: float
    moe_cache_size: int | None
    moe_cache_rate: float | None
    moe_cpu_threads: int | None
    moe_cpu_layers: str | int | float | None
    moe_hybrid_max_fetch: int | None
    max_running_requests: int


def _launch_policy(
    context: RuntimeCompatibilityContext,
    config: FreeTokenProviderConfig,
) -> FreeTokenLaunchPolicy:
    gpu_uuid = (
        context.worker.gpu_uuids[0]
        if len(context.worker.gpu_uuids) == 1
        else ""
    )
    return FreeTokenLaunchPolicy(
        gpu_uuid=gpu_uuid,
        moe_strategy=config.moe_strategy,
        memory_ratio=config.memory_ratio,
        moe_cache_size=config.moe_cache_size,
        moe_cache_rate=config.moe_cache_rate,
        moe_cpu_threads=config.moe_cpu_threads,
        moe_cpu_layers=config.moe_cpu_layers,
        moe_hybrid_max_fetch=config.moe_hybrid_max_fetch,
        max_running_requests=config.max_running_requests,
    )


@runtime_checkable
class FreeTokenApi(Protocol):
    async def health(self) -> bool: ...

    async def models(self) -> tuple[str, ...]: ...

    async def chat(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def completion(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def stats(self) -> Mapping[str, Any]: ...

    async def close(self) -> None: ...


class HttpFreeTokenApi:
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
            raise RuntimeError("FreeToken returned a non-object response")
        return body

    async def health(self) -> bool:
        try:
            body = await self._json("GET", "/health")
        except (httpx.HTTPError, RuntimeError, ValueError):
            return False
        return body.get("status") == "ok"

    async def models(self) -> tuple[str, ...]:
        body = await self._json("GET", "/v1/models")
        data = body.get("data", ())
        if not isinstance(data, list):
            raise RuntimeError("FreeToken /v1/models returned invalid data")
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

    async def stats(self) -> Mapping[str, Any]:
        return await self._json("GET", "/v1/stats")

    async def close(self) -> None:
        await self._client.aclose()


@runtime_checkable
class FreeTokenProcessController(Protocol):
    @property
    def running(self) -> bool: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class FreeTokenSubprocessController:
    def __init__(
        self,
        *,
        executable: str,
        base_url: str,
        model_ref: str,
        served_model_name: str,
        launch_policy: FreeTokenLaunchPolicy,
    ) -> None:
        self.executable = executable
        self.base_url = base_url
        self.model_ref = model_ref
        self.served_model_name = served_model_name
        self.launch_policy = launch_policy
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
            "--model",
            self.model_ref,
            "--host",
            parsed.hostname,
            "--port",
            str(parsed.port),
            "--gpu",
            self.launch_policy.gpu_uuid,
            "--served-model-name",
            self.served_model_name,
            "--moe-strategy",
            self.launch_policy.moe_strategy.value,
            "--memory-ratio",
            str(self.launch_policy.memory_ratio),
            "--max-running-requests",
            str(self.launch_policy.max_running_requests),
        ]
        if self.launch_policy.moe_cache_size is not None:
            args.extend(
                ["--moe-cache-size", str(self.launch_policy.moe_cache_size)]
            )
        if self.launch_policy.moe_cache_rate is not None:
            args.extend(
                ["--moe-cache-rate", str(self.launch_policy.moe_cache_rate)]
            )
        if self.launch_policy.moe_cpu_threads is not None:
            args.extend(
                ["--moe-cpu-threads", str(self.launch_policy.moe_cpu_threads)]
            )
        if self.launch_policy.moe_cpu_layers is not None:
            args.extend(
                ["--moe-cpu-layers", str(self.launch_policy.moe_cpu_layers)]
            )
        if self.launch_policy.moe_hybrid_max_fetch is not None:
            args.extend(
                [
                    "--moe-hybrid-max-fetch",
                    str(self.launch_policy.moe_hybrid_max_fetch),
                ]
            )
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
            raise RuntimeError("failed to start FreeToken server process") from exc

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


class FreeTokenExecutor(JobExecutor):
    capabilities = FREETOKEN_CAPABILITIES

    def __init__(
        self,
        *,
        api: FreeTokenApi,
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
                "job model does not match the model bound to this FreeToken runtime"
            )
        payload["model"] = self.served_model_name
        payload["stream"] = False
        return payload

    async def execute(self, job: JobRequest) -> JobResult:
        if job.capability not in self.capabilities:
            raise ValueError(
                f"FreeToken executor does not support capability: {job.capability}"
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
                "runtime_provider": FREETOKEN_PROVIDER_ID,
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
        metadata = dict(self.residency_metadata)
        accelerator_memory_bytes: int | None = None
        try:
            stats = await self.api.stats()
        except Exception:
            stats = {}

        vram_bytes = stats.get("vram_bytes")
        if isinstance(vram_bytes, int) and not isinstance(vram_bytes, bool):
            if vram_bytes >= 0:
                accelerator_memory_bytes = vram_bytes

        model_stats = stats.get("model")
        if isinstance(model_stats, Mapping):
            observed_model = model_stats.get("id")
            if isinstance(observed_model, str):
                metadata["observed_model"] = observed_model
            observed_moe = model_stats.get("moe")
            if isinstance(observed_moe, bool):
                metadata["observed_moe"] = observed_moe

        return ResidencyReport(
            items=(
                ResidencyItem(
                    name=self.model_ref,
                    kind="freetoken-model",
                    accelerator_memory_bytes=accelerator_memory_bytes,
                    metadata=metadata,
                ),
            ),
            metadata={"runtime_provider": FREETOKEN_PROVIDER_ID},
        )


class FreeTokenManagedRuntime(ManagedRuntime):
    provider_id = FREETOKEN_PROVIDER_ID

    def __init__(
        self,
        *,
        api: FreeTokenApi,
        process: FreeTokenProcessController,
        context: RuntimeCompatibilityContext,
        model_ref: str,
        served_model_name: str,
        startup_timeout_seconds: float,
        launch_policy: FreeTokenLaunchPolicy,
    ) -> None:
        self.api = api
        self.process = process
        self.context = context
        self.model_ref = model_ref
        self.served_model_name = served_model_name
        self.startup_timeout_seconds = startup_timeout_seconds
        self.launch_policy = launch_policy
        self._closed = False
        self._executor = FreeTokenExecutor(
            api=api,
            model_ref=model_ref,
            served_model_name=served_model_name,
            residency_metadata={
                "residency_policy": context.demand.residency_policy.value,
                "gpu_topology": context.demand.gpu_topology.value,
                "gpu_uuid": launch_policy.gpu_uuid,
                "host_ram_mb": context.host.host_ram_mb,
                "moe_strategy": launch_policy.moe_strategy.value,
                "memory_ratio": launch_policy.memory_ratio,
                "moe_cache_size": launch_policy.moe_cache_size,
                "moe_cache_rate": launch_policy.moe_cache_rate,
                "moe_cpu_threads": launch_policy.moe_cpu_threads,
                "moe_cpu_layers": launch_policy.moe_cpu_layers,
                "moe_hybrid_max_fetch": launch_policy.moe_hybrid_max_fetch,
            },
        )

    async def _server_reachable(self) -> bool:
        return await self.api.health()

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("FreeToken runtime has already been released")

        reachable = await self._server_reachable()
        if reachable and not self.process.running:
            raise RuntimeError(
                "an external FreeToken server is already using the configured endpoint"
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
                        "FreeToken server did not become ready before timeout"
                    )
                await asyncio.sleep(0.2)

            models = await self.api.models()
            if self.served_model_name not in models:
                raise RuntimeError(
                    "FreeToken server did not expose the configured model alias"
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
                metadata={"runtime_provider": FREETOKEN_PROVIDER_ID},
            )
        if not self.process.running:
            return RuntimeHealth(
                state=RuntimeHealthState.FAILED,
                ready=False,
                detail="external FreeToken process detected at managed endpoint",
                metadata={"runtime_provider": FREETOKEN_PROVIDER_ID},
            )
        try:
            models = await self.api.models()
        except Exception:
            return RuntimeHealth(
                state=RuntimeHealthState.DEGRADED,
                ready=False,
                detail="FreeToken model inventory is unavailable",
                metadata={"runtime_provider": FREETOKEN_PROVIDER_ID},
            )
        if self.served_model_name not in models:
            return RuntimeHealth(
                state=RuntimeHealthState.FAILED,
                ready=False,
                detail="configured FreeToken model alias is unavailable",
                metadata={"runtime_provider": FREETOKEN_PROVIDER_ID},
            )
        return RuntimeHealth(
            state=RuntimeHealthState.READY,
            ready=True,
            metadata={
                "runtime_provider": FREETOKEN_PROVIDER_ID,
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


class FreeTokenProvider(RuntimeProvider):
    def __init__(
        self,
        config: FreeTokenProviderConfig | None = None,
        *,
        api_factory: Any | None = None,
        process_factory: Any | None = None,
    ) -> None:
        self.config = config or FreeTokenProviderConfig()
        self._api_factory = api_factory or (
            lambda base_url: HttpFreeTokenApi(
                base_url,
                timeout_seconds=self.config.request_timeout_seconds,
            )
        )
        self._process_factory = process_factory or (
            lambda **kwargs: FreeTokenSubprocessController(**kwargs)
        )
        self._info = RuntimeProviderInfo(
            provider_id=FREETOKEN_PROVIDER_ID,
            display_name="FreeToken",
            description=(
                "RAM-heavy, VRAM-constrained MoE serving with explicit "
                "single-GPU ownership."
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

        if context.host.architecture.lower() not in {"x86_64", "amd64"}:
            reasons.append(
                CompatibilityReason(
                    code="architecture-unsupported",
                    message=(
                        "current upstream FreeToken packages require Linux x86_64; "
                        f"host architecture is {context.host.architecture}"
                    ),
                )
            )

        if model.model_format not in FREETOKEN_MODEL_FORMATS:
            reasons.append(
                CompatibilityReason(
                    code="model-format-unsupported",
                    message=(
                        "AstrumWeaver FreeToken provider requires a Hugging Face/"
                        "safetensors or local FTW model reference"
                    ),
                )
            )

        if (
            model.model_format == "ftw"
            and not _local_model_reference(model.model_ref)
        ):
            reasons.append(
                CompatibilityReason(
                    code="ftw-reference-must-be-local",
                    message=(
                        "FTW is a local FreeToken checkpoint format; the v0.x "
                        "provider requires an explicit local FTW path"
                    ),
                )
            )

        if model.topology is not ModelTopology.MOE:
            reasons.append(
                CompatibilityReason(
                    code="model-topology-outside-provider-scope",
                    message=(
                        "AstrumWeaver v0.x intentionally scopes the FreeToken "
                        "provider to MoE/offload workloads; this is not a claim "
                        "that upstream FreeToken cannot serve dense models"
                    ),
                )
            )

        if (
            demand.gpu_topology is not GPUTopology.SINGLE_GPU
            or resources.gpu_count != 1
            or len(context.worker.gpu_uuids) != 1
        ):
            reasons.append(
                CompatibilityReason(
                    code="single-gpu-required",
                    message=(
                        "AstrumWeaver v0.x FreeToken provider requires a "
                        "single-GPU Worker and will not partially consume a "
                        "multi-GPU Worker"
                    ),
                )
            )

        if demand.residency_policy is ResidencyPolicy.VRAM_ONLY:
            reasons.append(
                CompatibilityReason(
                    code="vram-only-unsupported",
                    message=(
                        "AstrumWeaver v0.x FreeToken provider targets MoE "
                        "offload and does not claim a vram_only execution path"
                    ),
                )
            )
        elif demand.residency_policy is ResidencyPolicy.CPU_GPU_HYBRID:
            reasons.append(
                CompatibilityReason(
                    code="cpu-gpu-hybrid-supported",
                    message=(
                        "FreeToken MoE execution will use the selected "
                        f"{policy.moe_strategy.value} strategy with host RAM "
                        "available for expert offload/CPU execution"
                    ),
                    blocking=False,
                )
            )
        else:
            reasons.append(
                CompatibilityReason(
                    code="prefer-vram-moe-cache",
                    message=(
                        "FreeToken will prefer GPU residency within its memory "
                        "budget while retaining its MoE offload/cache policy"
                    ),
                    blocking=False,
                )
            )

        if policy.moe_strategy in {
            FreeTokenMoeStrategy.OFFLOAD,
            FreeTokenMoeStrategy.CPU,
            FreeTokenMoeStrategy.HYBRID,
            FreeTokenMoeStrategy.AUTO,
        }:
            reasons.append(
                CompatibilityReason(
                    code="host-ram-interconnect-sensitive",
                    message=(
                        "FreeToken MoE offload performance and feasible model "
                        "size depend on host RAM capacity/bandwidth and the "
                        "CPU-GPU interconnect; AstrumWeaver does not invent an "
                        "exact RAM requirement beyond declared demand"
                    ),
                    blocking=False,
                )
            )

        return RuntimeCompatibility(
            provider_id=FREETOKEN_PROVIDER_ID,
            reasons=tuple(reasons),
        )

    def setup_intent(
        self,
        context: RuntimeCompatibilityContext,
    ) -> RuntimeSetupIntent:
        report = self.compatibility(context)
        if not report.compatible:
            raise RuntimeError(
                "cannot build FreeToken setup intent for incompatible demand"
            )

        policy = _launch_policy(context, self.config)
        return RuntimeSetupIntent(
            provider_id=FREETOKEN_PROVIDER_ID,
            package_references=(self.config.package_reference,),
            configuration={
                "base_url": self.config.base_url,
                "executable": self.config.executable,
                "model_ref": context.demand.model.model_ref,
                "served_model_name": self.config.served_model_name,
                "gpu_uuid": policy.gpu_uuid,
                "moe_strategy": policy.moe_strategy.value,
                "memory_ratio": policy.memory_ratio,
                "moe_cache_size": policy.moe_cache_size,
                "moe_cache_rate": policy.moe_cache_rate,
                "moe_cpu_threads": policy.moe_cpu_threads,
                "moe_cpu_layers": policy.moe_cpu_layers,
                "moe_hybrid_max_fetch": policy.moe_hybrid_max_fetch,
                "max_running_requests": policy.max_running_requests,
                "capabilities": sorted(FREETOKEN_CAPABILITIES),
            },
            model_preparation=_model_preparation(
                context.demand.model.model_ref
            ),
            model_ref=context.demand.model.model_ref,
            requires_privilege=True,
            metadata={"runtime_provider": FREETOKEN_PROVIDER_ID},
        )

    def create_runtime(
        self,
        context: RuntimeCompatibilityContext,
        setup: RuntimeSetupIntent,
    ) -> ManagedRuntime:
        if setup.provider_id != FREETOKEN_PROVIDER_ID:
            raise ValueError(
                "setup intent does not belong to FreeToken provider"
            )

        report = self.compatibility(context)
        if not report.compatible:
            raise RuntimeError(
                "cannot create FreeToken runtime for incompatible demand"
            )

        cfg = dict(setup.configuration)
        expected_gpu_uuid = context.worker.gpu_uuids[0]
        configured_gpu_uuid = cfg.get("gpu_uuid")
        if (
            configured_gpu_uuid is not None
            and str(configured_gpu_uuid) != expected_gpu_uuid
        ):
            raise ValueError(
                "setup intent GPU UUID does not match FreeToken Worker ownership"
            )

        base_url = str(cfg.get("base_url") or self.config.base_url)
        model_ref = str(
            cfg.get("model_ref") or context.demand.model.model_ref
        )
        served_model_name = str(
            cfg.get("served_model_name") or self.config.served_model_name
        )
        executable = str(cfg.get("executable") or self.config.executable)
        policy = FreeTokenLaunchPolicy(
            gpu_uuid=expected_gpu_uuid,
            moe_strategy=FreeTokenMoeStrategy(
                cfg.get("moe_strategy", self.config.moe_strategy)
            ),
            memory_ratio=float(
                cfg.get("memory_ratio", self.config.memory_ratio)
            ),
            moe_cache_size=(
                int(cfg["moe_cache_size"])
                if cfg.get("moe_cache_size") is not None
                else None
            ),
            moe_cache_rate=(
                float(cfg["moe_cache_rate"])
                if cfg.get("moe_cache_rate") is not None
                else None
            ),
            moe_cpu_threads=(
                int(cfg["moe_cpu_threads"])
                if cfg.get("moe_cpu_threads") is not None
                else None
            ),
            moe_cpu_layers=cfg.get(
                "moe_cpu_layers",
                self.config.moe_cpu_layers,
            ),
            moe_hybrid_max_fetch=(
                int(cfg["moe_hybrid_max_fetch"])
                if cfg.get("moe_hybrid_max_fetch") is not None
                else None
            ),
            max_running_requests=int(
                cfg.get(
                    "max_running_requests",
                    self.config.max_running_requests,
                )
            ),
        )

        api = self._api_factory(base_url)
        process = self._process_factory(
            executable=executable,
            base_url=base_url,
            model_ref=model_ref,
            served_model_name=served_model_name,
            launch_policy=policy,
        )

        return FreeTokenManagedRuntime(
            api=api,
            process=process,
            context=context,
            model_ref=model_ref,
            served_model_name=served_model_name,
            startup_timeout_seconds=self.config.startup_timeout_seconds,
            launch_policy=policy,
        )
