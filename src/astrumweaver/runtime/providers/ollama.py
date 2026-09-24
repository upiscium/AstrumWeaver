"""First-class Ollama RuntimeProvider."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

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


MIB = 1024 * 1024
OLLAMA_PROVIDER_ID = "ollama"
OLLAMA_CAPABILITIES = frozenset({"llm.chat", "text.generate"})


def _nonblank(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


@dataclass(frozen=True, slots=True)
class OllamaProviderConfig:
    base_url: str = "http://127.0.0.1:11434"
    executable: str = "ollama"
    package_reference: str = "ollama"
    keep_alive: str = "5m"
    startup_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _nonblank(self.base_url, "base_url").rstrip("/"))
        object.__setattr__(self, "executable", _nonblank(self.executable, "executable"))
        object.__setattr__(
            self,
            "package_reference",
            _nonblank(self.package_reference, "package_reference"),
        )
        object.__setattr__(self, "keep_alive", _nonblank(self.keep_alive, "keep_alive"))
        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")


@runtime_checkable
class OllamaApi(Protocol):
    async def list_models(self) -> tuple[Mapping[str, Any], ...]: ...

    async def running_models(self) -> tuple[Mapping[str, Any], ...]: ...

    async def chat(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def generate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def unload(self, model: str) -> None: ...

    async def close(self) -> None: ...


class HttpOllamaApi:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 120.0,
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
        response = await self._client.request(method, path, json=json)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, Mapping):
            raise RuntimeError("Ollama returned a non-object response")
        return body

    async def list_models(self) -> tuple[Mapping[str, Any], ...]:
        body = await self._json("GET", "/api/tags")
        models = body.get("models", ())
        if not isinstance(models, list):
            raise RuntimeError("Ollama /api/tags returned invalid models")
        return tuple(model for model in models if isinstance(model, Mapping))

    async def running_models(self) -> tuple[Mapping[str, Any], ...]:
        body = await self._json("GET", "/api/ps")
        models = body.get("models", ())
        if not isinstance(models, list):
            raise RuntimeError("Ollama /api/ps returned invalid models")
        return tuple(model for model in models if isinstance(model, Mapping))

    async def chat(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return await self._json("POST", "/api/chat", json=payload)

    async def generate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return await self._json("POST", "/api/generate", json=payload)

    async def unload(self, model: str) -> None:
        await self.generate(
            {
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            }
        )

    async def close(self) -> None:
        await self._client.aclose()


@runtime_checkable
class OllamaProcessController(Protocol):
    @property
    def running(self) -> bool: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class OllamaSubprocessController:
    """Own one local `ollama serve` child process.

    #31 may replace this with deployment-specific service controllers while
    preserving the same ManagedRuntime contract.
    """

    def __init__(
        self,
        *,
        executable: str,
        base_url: str,
        gpu_uuids: tuple[str, ...],
    ) -> None:
        self.executable = executable
        self.base_url = base_url
        self.gpu_uuids = gpu_uuids
        self._process: asyncio.subprocess.Process | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        if self.running:
            return
        env = dict(os.environ)
        host = self.base_url.removeprefix("http://").removeprefix("https://")
        env["OLLAMA_HOST"] = host
        env["OLLAMA_NUM_PARALLEL"] = "1"
        env["OLLAMA_MAX_LOADED_MODELS"] = "1"
        if self.gpu_uuids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpu_uuids)
        self._process = await asyncio.create_subprocess_exec(
            self.executable,
            "serve",
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

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


class OllamaExecutor(JobExecutor):
    capabilities = OLLAMA_CAPABILITIES

    def __init__(
        self,
        *,
        api: OllamaApi,
        model: str,
        keep_alive: str,
    ) -> None:
        self.api = api
        self.model = _nonblank(model, "model")
        self.keep_alive = _nonblank(keep_alive, "keep_alive")
        self._inflight: dict[str, asyncio.Task[Mapping[str, Any]]] = {}

    def _payload(self, job: JobRequest) -> dict[str, Any]:
        payload = dict(job.payload)
        supplied_model = payload.pop("model", None)
        if supplied_model is not None and str(supplied_model) != self.model:
            raise ValueError(
                "job model does not match the model bound to this Ollama runtime"
            )
        payload["model"] = self.model
        payload["stream"] = False
        payload.setdefault("keep_alive", self.keep_alive)
        return payload

    async def execute(self, job: JobRequest) -> JobResult:
        if job.capability not in self.capabilities:
            raise ValueError(
                f"Ollama executor does not support capability: {job.capability}"
            )

        payload = self._payload(job)
        if job.capability == "llm.chat":
            if not isinstance(payload.get("messages"), list):
                raise ValueError("llm.chat requires a messages list")
            call = self.api.chat(payload)
        else:
            if not isinstance(payload.get("prompt"), str):
                raise ValueError("text.generate requires a prompt string")
            call = self.api.generate(payload)

        task = asyncio.create_task(call)
        self._inflight[job.job_id] = task
        try:
            response = await task
        finally:
            self._inflight.pop(job.job_id, None)

        text: str | None
        if job.capability == "llm.chat":
            message = response.get("message")
            text = (
                str(message.get("content"))
                if isinstance(message, Mapping)
                and isinstance(message.get("content"), str)
                else None
            )
        else:
            candidate = response.get("response")
            text = candidate if isinstance(candidate, str) else None

        metrics: dict[str, int | float] = {}
        for key in (
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_cached_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        ):
            value = response.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = value

        return JobResult(
            outputs=dict(response),
            metrics=metrics,
            text=text,
            metadata={
                "runtime_provider": OLLAMA_PROVIDER_ID,
                "model": self.model,
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
        models = await self.api.running_models()
        items: list[ResidencyItem] = []
        for model in models:
            name = model.get("model") or model.get("name")
            if not isinstance(name, str) or name != self.model:
                continue
            size_vram = model.get("size_vram")
            size = model.get("size")
            metadata: dict[str, Any] = {}
            if isinstance(size, int) and not isinstance(size, bool):
                metadata["size_bytes"] = size
            context_length = model.get("context_length")
            if isinstance(context_length, int) and not isinstance(context_length, bool):
                metadata["context_length"] = context_length
            items.append(
                ResidencyItem(
                    name=self.model,
                    kind="ollama-model",
                    accelerator_memory_bytes=(
                        size_vram
                        if isinstance(size_vram, int)
                        and not isinstance(size_vram, bool)
                        else None
                    ),
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
        *,
        api: OllamaApi,
        process: OllamaProcessController,
        context: RuntimeCompatibilityContext,
        model: str,
        keep_alive: str,
        startup_timeout_seconds: float,
    ) -> None:
        self.api = api
        self.process = process
        self.context = context
        self.model = model
        self.keep_alive = keep_alive
        self.startup_timeout_seconds = startup_timeout_seconds
        self._executor = OllamaExecutor(
            api=api,
            model=model,
            keep_alive=keep_alive,
        )
        self._closed = False

    async def _server_reachable(self) -> bool:
        try:
            await self.api.list_models()
            return True
        except Exception:
            return False

    async def _available_models(self) -> frozenset[str]:
        models = await self.api.list_models()
        names: set[str] = set()
        for model in models:
            for key in ("model", "name"):
                value = model.get(key)
                if isinstance(value, str):
                    names.add(value)
        return frozenset(names)

    async def _resident_entry(self) -> Mapping[str, Any] | None:
        for entry in await self.api.running_models():
            name = entry.get("model") or entry.get("name")
            if name == self.model:
                return entry
        return None

    async def _load_model(self) -> None:
        await self.api.generate(
            {
                "model": self.model,
                "prompt": "",
                "stream": False,
                "keep_alive": self.keep_alive,
            }
        )

    async def _verify_loaded_topology(self) -> None:
        entry = await self._resident_entry()
        if entry is None:
            raise RuntimeError("Ollama did not report the configured model as resident")

        size = entry.get("size")
        size_vram = entry.get("size_vram")
        if not isinstance(size_vram, int) or isinstance(size_vram, bool):
            raise RuntimeError("Ollama did not report model VRAM residency")

        demand = self.context.demand
        if demand.residency_policy is ResidencyPolicy.VRAM_ONLY:
            if not isinstance(size, int) or isinstance(size, bool):
                raise RuntimeError(
                    "Ollama did not report model size required for vram_only verification"
                )
            if size_vram < size:
                raise RuntimeError(
                    "Ollama loaded part of the model outside VRAM under vram_only policy"
                )

        if demand.gpu_topology is GPUTopology.MULTI_GPU:
            single_capacity = self.context.worker.resources.max_single_gpu_vram_mb * MIB
            if size_vram <= single_capacity:
                raise RuntimeError(
                    "Ollama residency does not prove use of more than one GPU"
                )

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("Ollama runtime has already been released")

        reachable = await self._server_reachable()
        if reachable and not self.process.running:
            raise RuntimeError(
                "an external Ollama server is already using the configured endpoint"
            )

        if not self.process.running:
            await self.process.start()

        deadline = asyncio.get_running_loop().time() + self.startup_timeout_seconds
        while not await self._server_reachable():
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("Ollama server did not become ready")
            await asyncio.sleep(0.1)

        available = await self._available_models()
        if self.model not in available:
            raise RuntimeError(
                "configured Ollama model is not available after setup"
            )

        await self._load_model()
        await self._verify_loaded_topology()

    async def stop(self) -> None:
        if await self._server_reachable():
            with contextlib.suppress(Exception):
                await self.api.unload(self.model)
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
                metadata={"runtime_provider": OLLAMA_PROVIDER_ID},
            )

        if not self.process.running:
            return RuntimeHealth(
                state=RuntimeHealthState.FAILED,
                ready=False,
                detail="external Ollama process detected at managed endpoint",
                metadata={"runtime_provider": OLLAMA_PROVIDER_ID},
            )

        try:
            available = await self._available_models()
        except Exception:
            return RuntimeHealth(
                state=RuntimeHealthState.DEGRADED,
                ready=False,
                detail="Ollama model inventory is unavailable",
                metadata={"runtime_provider": OLLAMA_PROVIDER_ID},
            )

        if self.model not in available:
            return RuntimeHealth(
                state=RuntimeHealthState.FAILED,
                ready=False,
                detail="configured Ollama model is unavailable",
                metadata={"runtime_provider": OLLAMA_PROVIDER_ID},
            )

        return RuntimeHealth(
            state=RuntimeHealthState.READY,
            ready=True,
            metadata={
                "runtime_provider": OLLAMA_PROVIDER_ID,
                "model": self.model,
            },
        )

    async def residency(self) -> ResidencyReport:
        return await self._executor.residency()

    def executor(self) -> JobExecutor:
        return self._executor

    async def release(self) -> None:
        if self._closed:
            return
        if await self._server_reachable():
            with contextlib.suppress(Exception):
                await self.api.unload(self.model)
        await self.api.close()
        self._closed = True


class OllamaProvider(RuntimeProvider):
    def __init__(
        self,
        config: OllamaProviderConfig | None = None,
        *,
        api_factory: Any | None = None,
        process_factory: Any | None = None,
    ) -> None:
        self.config = config or OllamaProviderConfig()
        self._api_factory = api_factory or (
            lambda base_url: HttpOllamaApi(base_url)
        )
        self._process_factory = process_factory or (
            lambda *, executable, base_url, gpu_uuids: OllamaSubprocessController(
                executable=executable,
                base_url=base_url,
                gpu_uuids=gpu_uuids,
            )
        )
        self._info = RuntimeProviderInfo(
            provider_id=OLLAMA_PROVIDER_ID,
            display_name="Ollama",
            description="General-purpose local model serving through Ollama.",
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
        worker = context.worker
        resources = worker.resources
        model = demand.model

        if model.model_format != "ollama":
            reasons.append(
                CompatibilityReason(
                    code="model-format-unsupported",
                    message=(
                        "Ollama provider currently requires an Ollama model reference "
                        "(model_format=ollama)"
                    ),
                )
            )

        if demand.residency_policy is ResidencyPolicy.CPU_GPU_HYBRID:
            reasons.append(
                CompatibilityReason(
                    code="ollama-offload-automatic",
                    message=(
                        "Ollama controls CPU/GPU offload automatically; use llama.cpp "
                        "when explicit layer/offload control is required"
                    ),
                    blocking=False,
                )
            )

        if demand.residency_policy is ResidencyPolicy.VRAM_ONLY:
            if model.estimated_size_mb is None:
                reasons.append(
                    CompatibilityReason(
                        code="model-size-required-for-vram-only",
                        message=(
                            "Ollama vram_only planning requires estimated model size; "
                            "runtime residency is verified again after load"
                        ),
                    )
                )
            elif model.estimated_size_mb > resources.total_vram_mb:
                reasons.append(
                    CompatibilityReason(
                        code="model-exceeds-total-vram",
                        message=(
                            "estimated model size exceeds Worker total VRAM under "
                            "vram_only policy"
                        ),
                    )
                )
            else:
                reasons.append(
                    CompatibilityReason(
                        code="vram-residency-runtime-verified",
                        message=(
                            "Ollama full-VRAM residency will be verified from /api/ps "
                            "after the model is loaded"
                        ),
                        blocking=False,
                    )
                )

        if demand.gpu_topology is GPUTopology.MULTI_GPU:
            if model.estimated_size_mb is None:
                reasons.append(
                    CompatibilityReason(
                        code="multi-gpu-model-size-required",
                        message=(
                            "Ollama cannot guarantee use of multiple GPUs without "
                            "model-size evidence that exceeds one device"
                        ),
                    )
                )
            elif model.estimated_size_mb <= resources.max_single_gpu_vram_mb:
                reasons.append(
                    CompatibilityReason(
                        code="multi-gpu-not-forced",
                        message=(
                            "Ollama may keep this model on one GPU because its estimated "
                            "size fits the largest single device"
                        ),
                    )
                )
            else:
                reasons.append(
                    CompatibilityReason(
                        code="multi-gpu-runtime-verified",
                        message=(
                            "Ollama multi-GPU residency will be verified after load; "
                            "Ollama chooses the split automatically"
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
        report = self.compatibility(context)
        if not report.compatible:
            raise RuntimeError(
                "cannot build Ollama setup intent for an incompatible execution demand"
            )

        return RuntimeSetupIntent(
            provider_id=OLLAMA_PROVIDER_ID,
            package_references=(self.config.package_reference,),
            configuration={
                "base_url": self.config.base_url,
                "executable": self.config.executable,
                "model": context.demand.model.model_ref,
                "keep_alive": self.config.keep_alive,
                "startup_timeout_seconds": self.config.startup_timeout_seconds,
                "gpu_uuids": list(context.worker.gpu_uuids),
                "capabilities": sorted(OLLAMA_CAPABILITIES),
                "num_parallel": 1,
                "max_loaded_models": 1,
            },
            model_preparation=ModelPreparationPolicy.DOWNLOAD,
            model_ref=context.demand.model.model_ref,
            requires_privilege=True,
            metadata={
                "runtime_provider": OLLAMA_PROVIDER_ID,
            },
        )

    def create_runtime(
        self,
        context: RuntimeCompatibilityContext,
        setup: RuntimeSetupIntent,
    ) -> ManagedRuntime:
        if setup.provider_id != OLLAMA_PROVIDER_ID:
            raise ValueError("setup intent does not belong to Ollama provider")

        report = self.compatibility(context)
        if not report.compatible:
            raise RuntimeError(
                "cannot create Ollama runtime for an incompatible execution demand"
            )

        configuration = dict(setup.configuration)
        model = str(configuration.get("model") or context.demand.model.model_ref)
        base_url = str(configuration.get("base_url") or self.config.base_url)
        executable = str(configuration.get("executable") or self.config.executable)
        keep_alive = str(configuration.get("keep_alive") or self.config.keep_alive)
        startup_timeout_seconds = float(
            configuration.get(
                "startup_timeout_seconds",
                self.config.startup_timeout_seconds,
            )
        )
        api = self._api_factory(base_url)
        process = self._process_factory(
            executable=executable,
            base_url=base_url,
            gpu_uuids=context.worker.gpu_uuids,
        )

        return OllamaManagedRuntime(
            api=api,
            process=process,
            context=context,
            model=model,
            keep_alive=keep_alive,
            startup_timeout_seconds=startup_timeout_seconds,
        )
