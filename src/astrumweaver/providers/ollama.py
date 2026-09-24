"""First-class Ollama RuntimeProvider.

Ollama is treated as one Worker-local runtime implementation. Control remains
provider-agnostic.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import httpx

from ..execution import (
    JobExecutor,
    JobRequest,
    JobResult,
    ResidencyItem,
    ResidencyReport,
)
from ..runtime import (
    CompatibilityReason,
    ExecutionDemand,
    GPUTopology,
    ManagedRuntime,
    ModelPreparationPolicy,
    ResidencyPolicy,
    RuntimeCompatibility,
    RuntimeCompatibilityContext,
    RuntimeHealth,
    RuntimeHealthState,
    RuntimeProviderInfo,
    RuntimeSetupIntent,
)


OLLAMA_PROVIDER_ID = "ollama"
OLLAMA_DEFAULT_HOST = "127.0.0.1"
OLLAMA_DEFAULT_PORT = 11434
OLLAMA_SUPPORTED_MODEL_FORMATS = frozenset({"ollama", "ollama-model"})


def _nonblank(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


@dataclass(frozen=True, slots=True)
class OllamaRuntimeConfig:
    model: str
    host: str = OLLAMA_DEFAULT_HOST
    port: int = OLLAMA_DEFAULT_PORT
    executable: str = "ollama"
    keep_alive: int | str = -1
    startup_timeout_seconds: float = 60.0
    request_timeout_seconds: float = 300.0
    no_cloud: bool = True
    max_loaded_models: int = 1
    num_parallel: int = 1
    sched_spread: bool = False
    gpu_uuids: tuple[str, ...] = ()
    residency_policy: ResidencyPolicy = ResidencyPolicy.PREFER_VRAM
    extra_environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _nonblank(self.model, "model"))
        object.__setattr__(self, "host", _nonblank(self.host, "host"))
        object.__setattr__(
            self,
            "executable",
            _nonblank(self.executable, "executable"),
        )
        if not (1 <= self.port <= 65535):
            raise ValueError("port must be between 1 and 65535")
        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.max_loaded_models <= 0:
            raise ValueError("max_loaded_models must be positive")
        if self.num_parallel <= 0:
            raise ValueError("num_parallel must be positive")

        gpu_uuids = tuple(str(value).strip() for value in self.gpu_uuids)
        if any(not value for value in gpu_uuids):
            raise ValueError("gpu_uuids must not contain blanks")
        if len(gpu_uuids) != len(set(gpu_uuids)):
            raise ValueError("gpu_uuids must not contain duplicates")
        object.__setattr__(self, "gpu_uuids", gpu_uuids)
        object.__setattr__(
            self,
            "residency_policy",
            ResidencyPolicy(self.residency_policy),
        )
        environment = {
            _nonblank(key, "environment key"): str(value)
            for key, value in dict(self.extra_environment).items()
        }
        object.__setattr__(
            self,
            "extra_environment",
            MappingProxyType(environment),
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def environment(self) -> dict[str, str]:
        env = {
            "OLLAMA_HOST": f"{self.host}:{self.port}",
            "OLLAMA_KEEP_ALIVE": str(self.keep_alive),
            "OLLAMA_MAX_LOADED_MODELS": str(self.max_loaded_models),
            "OLLAMA_NUM_PARALLEL": str(self.num_parallel),
            "OLLAMA_NO_CLOUD": "true" if self.no_cloud else "false",
        }
        if self.gpu_uuids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpu_uuids)
        if self.sched_spread:
            env["OLLAMA_SCHED_SPREAD"] = "true"
        env.update(self.extra_environment)
        return env


class OllamaAPIError(RuntimeError):
    pass


class OllamaAPI:
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

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise OllamaAPIError("Ollama API is unavailable") from exc
        if response.status_code >= 400:
            raise OllamaAPIError(
                f"Ollama API request failed ({response.status_code})"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise OllamaAPIError("Ollama API returned invalid JSON") from exc

    async def version(self) -> str:
        body = await self._request("GET", "/api/version")
        version = body.get("version") if isinstance(body, Mapping) else None
        if not isinstance(version, str) or not version.strip():
            raise OllamaAPIError("Ollama API returned an invalid version")
        return version.strip()

    async def list_models(self) -> tuple[str, ...]:
        body = await self._request("GET", "/api/tags")
        models = body.get("models") if isinstance(body, Mapping) else None
        if not isinstance(models, list):
            raise OllamaAPIError("Ollama API returned an invalid model list")
        names: list[str] = []
        for item in models:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name") or item.get("model")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        return tuple(names)

    async def running_models(self) -> tuple[Mapping[str, Any], ...]:
        body = await self._request("GET", "/api/ps")
        models = body.get("models") if isinstance(body, Mapping) else None
        if not isinstance(models, list):
            raise OllamaAPIError("Ollama API returned invalid residency data")
        return tuple(
            MappingProxyType(dict(item))
            for item in models
            if isinstance(item, Mapping)
        )

    async def generate(
        self,
        *,
        model: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        request = dict(payload)
        request["model"] = model
        request["stream"] = False
        body = await self._request("POST", "/api/generate", json=request)
        if not isinstance(body, Mapping):
            raise OllamaAPIError("Ollama generate response is invalid")
        return MappingProxyType(dict(body))

    async def chat(
        self,
        *,
        model: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        request = dict(payload)
        request["model"] = model
        request["stream"] = False
        body = await self._request("POST", "/api/chat", json=request)
        if not isinstance(body, Mapping):
            raise OllamaAPIError("Ollama chat response is invalid")
        return MappingProxyType(dict(body))

    async def load_model(
        self,
        model: str,
        *,
        keep_alive: int | str = -1,
    ) -> None:
        await self.generate(
            model=model,
            payload={"prompt": "", "keep_alive": keep_alive},
        )

    async def unload_model(self, model: str) -> None:
        await self.generate(
            model=model,
            payload={"prompt": "", "keep_alive": 0},
        )


@runtime_checkable
class OllamaProcessController(Protocol):
    @property
    def running(self) -> bool: ...

    async def start(self, environment: Mapping[str, str]) -> None: ...

    async def stop(self) -> None: ...


class SubprocessOllamaProcess:
    def __init__(
        self,
        executable: str = "ollama",
        *,
        shutdown_timeout_seconds: float = 10.0,
    ) -> None:
        self.executable = _nonblank(executable, "executable")
        if shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown_timeout_seconds must be positive")
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self._process: asyncio.subprocess.Process | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self, environment: Mapping[str, str]) -> None:
        if self.running:
            return
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in environment.items()})
        try:
            self._process = await asyncio.create_subprocess_exec(
                self.executable,
                "serve",
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            raise RuntimeError("failed to start Ollama runtime process") from exc

    async def stop(self) -> None:
        process = self._process
        if process is None:
            return
        if process.returncode is not None:
            self._process = None
            return

        process.terminate()
        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=self.shutdown_timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        finally:
            self._process = None


class OllamaExecutor(JobExecutor):
    capabilities = frozenset({"llm.chat", "text.generate"})

    def __init__(
        self,
        api: OllamaAPI,
        *,
        model: str,
        keep_alive: int | str = -1,
    ) -> None:
        self.api = api
        self.model = _nonblank(model, "model")
        self.keep_alive = keep_alive
        self._active: dict[str, asyncio.Task[Mapping[str, Any]]] = {}

    def _request_payload(self, job: JobRequest) -> dict[str, Any]:
        payload = dict(job.payload)
        requested_model = payload.pop("model", None)
        if requested_model is not None and str(requested_model) != self.model:
            raise ValueError(
                "Ollama Worker is pinned to one configured model; "
                "job model override does not match"
            )
        payload.setdefault("keep_alive", self.keep_alive)
        return payload

    async def execute(self, job: JobRequest) -> JobResult:
        if job.job_id in self._active:
            raise RuntimeError("duplicate active Ollama job ID")

        payload = self._request_payload(job)
        if job.capability == "llm.chat":
            call = self.api.chat(model=self.model, payload=payload)
        elif job.capability == "text.generate":
            call = self.api.generate(model=self.model, payload=payload)
        else:
            raise ValueError(
                f"unsupported Ollama executor capability: {job.capability}"
            )

        task = asyncio.create_task(call)
        self._active[job.job_id] = task
        try:
            raw = await task
        finally:
            self._active.pop(job.job_id, None)

        outputs = dict(raw)
        text: str | None = None
        if job.capability == "llm.chat":
            message = outputs.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    text = content
        else:
            response = outputs.get("response")
            if isinstance(response, str):
                text = response

        metrics: dict[str, int | float] = {}
        for source, target in (
            ("total_duration", "total_duration_ns"),
            ("load_duration", "load_duration_ns"),
            ("prompt_eval_count", "prompt_eval_count"),
            ("prompt_eval_duration", "prompt_eval_duration_ns"),
            ("eval_count", "eval_count"),
            ("eval_duration", "eval_duration_ns"),
        ):
            value = outputs.get(source)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                metrics[target] = value

        return JobResult(
            outputs=outputs,
            metrics=metrics,
            text=text,
            metadata={
                "runtime_provider": OLLAMA_PROVIDER_ID,
                "model": self.model,
            },
        )

    async def cancel(self, job_id: str) -> None:
        task = self._active.get(job_id)
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def residency(self) -> ResidencyReport:
        items: list[ResidencyItem] = []
        for raw in await self.api.running_models():
            name = raw.get("name") or raw.get("model")
            if not isinstance(name, str) or not name.strip():
                continue
            size_vram = raw.get("size_vram")
            accelerator_memory_bytes = (
                int(size_vram)
                if isinstance(size_vram, int) and size_vram >= 0
                else None
            )
            metadata: dict[str, Any] = {}
            for key in (
                "size",
                "digest",
                "expires_at",
                "context_length",
            ):
                value = raw.get(key)
                if value is not None:
                    metadata[key] = value
            items.append(
                ResidencyItem(
                    name=name.strip(),
                    kind="ollama-model",
                    accelerator_memory_bytes=accelerator_memory_bytes,
                    metadata=metadata,
                )
            )
        return ResidencyReport(
            items=tuple(items),
            metadata={"runtime_provider": OLLAMA_PROVIDER_ID},
        )


class OllamaManagedRuntime(ManagedRuntime):
    provider_id = OLLAMA_PROVIDER_ID

    def __init__(
        self,
        config: OllamaRuntimeConfig,
        *,
        api: OllamaAPI | None = None,
        process: OllamaProcessController | None = None,
    ) -> None:
        self.config = config
        self.api = api or OllamaAPI(
            config.base_url,
            timeout_seconds=config.request_timeout_seconds,
        )
        self.process = process or SubprocessOllamaProcess(config.executable)
        self._executor = OllamaExecutor(
            self.api,
            model=config.model,
            keep_alive=config.keep_alive,
        )

    async def start(self) -> None:
        await self.process.start(self.config.environment())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.config.startup_timeout_seconds
        while True:
            try:
                await self.api.version()
                break
            except OllamaAPIError:
                if loop.time() >= deadline:
                    raise RuntimeError(
                        "Ollama runtime did not become ready before timeout"
                    )
                await asyncio.sleep(0.25)

        models = await self.api.list_models()
        if self.config.model not in models:
            raise RuntimeError(
                "configured Ollama model is not installed; "
                "apply the reviewed model setup plan first"
            )

        await self.api.load_model(
            self.config.model,
            keep_alive=self.config.keep_alive,
        )
        if self.config.residency_policy is ResidencyPolicy.VRAM_ONLY:
            await self._require_full_vram_residency()

    async def _require_full_vram_residency(self) -> None:
        for raw in await self.api.running_models():
            name = raw.get("name") or raw.get("model")
            if name != self.config.model:
                continue
            size = raw.get("size")
            size_vram = raw.get("size_vram")
            if (
                isinstance(size, int)
                and size > 0
                and isinstance(size_vram, int)
                and size_vram >= size
            ):
                return
            raise RuntimeError(
                "Ollama model is not fully resident in VRAM"
            )
        raise RuntimeError(
            "Ollama model residency was not visible after preload"
        )

    async def stop(self) -> None:
        with contextlib.suppress(OllamaAPIError):
            await self.api.unload_model(self.config.model)
        await self.process.stop()

    async def health(self) -> RuntimeHealth:
        try:
            version = await self.api.version()
        except OllamaAPIError:
            return RuntimeHealth(
                state=(
                    RuntimeHealthState.STARTING
                    if self.process.running
                    else RuntimeHealthState.STOPPED
                ),
                ready=False,
            )

        try:
            models = await self.api.list_models()
        except OllamaAPIError:
            return RuntimeHealth(
                state=RuntimeHealthState.DEGRADED,
                ready=False,
                detail="Ollama API is reachable but model inventory failed",
                metadata={"version": version},
            )

        if self.config.model not in models:
            return RuntimeHealth(
                state=RuntimeHealthState.DEGRADED,
                ready=False,
                detail="configured Ollama model is not installed",
                metadata={"version": version},
            )

        return RuntimeHealth(
            state=RuntimeHealthState.READY,
            ready=True,
            metadata={
                "version": version,
                "model": self.config.model,
            },
        )

    async def residency(self) -> ResidencyReport:
        return await self._executor.residency()

    def executor(self) -> JobExecutor:
        return self._executor

    async def release(self) -> None:
        with contextlib.suppress(OllamaAPIError):
            await self.api.unload_model(self.config.model)

    async def aclose(self) -> None:
        await self.api.aclose()


class OllamaProvider:
    info = RuntimeProviderInfo(
        provider_id=OLLAMA_PROVIDER_ID,
        display_name="Ollama",
        description="General-purpose local model serving via Ollama.",
    )

    def compatibility(
        self,
        context: RuntimeCompatibilityContext,
    ) -> RuntimeCompatibility:
        demand: ExecutionDemand = context.demand
        reasons: list[CompatibilityReason] = []

        if demand.model.model_format not in OLLAMA_SUPPORTED_MODEL_FORMATS:
            reasons.append(
                CompatibilityReason(
                    code="model-format-unsupported",
                    message=(
                        "Ollama provider currently requires an "
                        "Ollama model reference"
                    ),
                )
            )

        if demand.gpu_topology is GPUTopology.NONE:
            reasons.append(
                CompatibilityReason(
                    code="cpu-only-unsupported",
                    message=(
                        "first-class Ollama provider currently targets "
                        "GPU Workers"
                    ),
                )
            )

        if demand.residency_policy is ResidencyPolicy.VRAM_ONLY:
            has_budget = (
                demand.model.estimated_size_mb is not None
                or demand.min_total_vram_mb > 0
                or demand.min_single_gpu_vram_mb > 0
            )
            if not has_budget:
                reasons.append(
                    CompatibilityReason(
                        code="vram-budget-unspecified",
                        message=(
                            "vram_only Ollama execution requires an "
                            "explicit model/VRAM size budget"
                        ),
                    )
                )

        if demand.residency_policy is ResidencyPolicy.CPU_GPU_HYBRID:
            reasons.append(
                CompatibilityReason(
                    code="ollama-managed-offload",
                    message=(
                        "Ollama controls CPU/GPU offload placement; use "
                        "llama.cpp when exact layer offload control is required"
                    ),
                    blocking=False,
                )
            )

        if demand.gpu_topology is GPUTopology.MULTI_GPU:
            reasons.append(
                CompatibilityReason(
                    code="ollama-managed-multi-gpu-placement",
                    message=(
                        "Ollama controls model distribution across the "
                        "selected GPU set"
                    ),
                    blocking=False,
                )
            )

        return RuntimeCompatibility(
            provider_id=OLLAMA_PROVIDER_ID,
            reasons=tuple(reasons),
        )

    def setup_intent(
        self,
        context: RuntimeCompatibilityContext,
    ) -> RuntimeSetupIntent:
        if not self.compatibility(context).compatible:
            raise ValueError(
                "Ollama setup intent requested for incompatible execution demand"
            )

        worker = context.worker
        demand = context.demand
        config = OllamaRuntimeConfig(
            model=demand.model.model_ref,
            gpu_uuids=worker.gpu_uuids,
            residency_policy=demand.residency_policy,
            sched_spread=demand.gpu_topology is GPUTopology.MULTI_GPU,
        )
        return RuntimeSetupIntent(
            provider_id=OLLAMA_PROVIDER_ID,
            package_references=("ollama",),
            configuration={
                "host": config.host,
                "port": config.port,
                "model": config.model,
                "keep_alive": config.keep_alive,
                "no_cloud": config.no_cloud,
                "max_loaded_models": config.max_loaded_models,
                "num_parallel": config.num_parallel,
                "sched_spread": config.sched_spread,
                "cuda_visible_devices": list(config.gpu_uuids),
                "residency_policy": config.residency_policy.value,
            },
            model_preparation=ModelPreparationPolicy.DOWNLOAD,
            model_ref=config.model,
            requires_privilege=True,
            metadata={
                "runtime_provider": OLLAMA_PROVIDER_ID,
                "capabilities": sorted(OllamaExecutor.capabilities),
            },
        )

    def create_runtime(
        self,
        context: RuntimeCompatibilityContext,
        setup: RuntimeSetupIntent,
    ) -> OllamaManagedRuntime:
        if setup.provider_id != OLLAMA_PROVIDER_ID:
            raise ValueError("Ollama provider received foreign setup intent")
        if not self.compatibility(context).compatible:
            raise ValueError(
                "Ollama runtime requested for incompatible execution demand"
            )

        configuration = dict(setup.configuration)
        config = OllamaRuntimeConfig(
            model=str(configuration.get("model") or context.demand.model.model_ref),
            host=str(configuration.get("host") or OLLAMA_DEFAULT_HOST),
            port=int(configuration.get("port") or OLLAMA_DEFAULT_PORT),
            keep_alive=configuration.get("keep_alive", -1),
            no_cloud=bool(configuration.get("no_cloud", True)),
            max_loaded_models=int(
                configuration.get("max_loaded_models", 1)
            ),
            num_parallel=int(configuration.get("num_parallel", 1)),
            sched_spread=bool(configuration.get("sched_spread", False)),
            gpu_uuids=context.worker.gpu_uuids,
            residency_policy=context.demand.residency_policy,
        )
        return OllamaManagedRuntime(config)
