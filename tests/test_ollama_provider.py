from __future__ import annotations

import asyncio
from collections.abc import Mapping

import httpx
import pytest

from astrumweaver import ResourceShape, WorkerSpec
from astrumweaver.execution import JobRequest
from astrumweaver.runtime import (
    ExecutionDemand,
    GPUTopology,
    ModelDemand,
    ModelTopology,
    ResidencyPolicy,
    RuntimeCatalog,
    RuntimeCompatibilityContext,
    RuntimeHostFacts,
    RuntimeHealthState,
    RuntimeSelection,
    RuntimeSelectionMode,
)
from astrumweaver.setup import (
    DeploymentPath,
    PrivilegeMode,
    SetupActionKind,
    SetupHostSnapshot,
    build_runtime_setup_plan,
)
from astrumweaver.runtime.providers.ollama import (
    HttpOllamaApi,
    OllamaExecutor,
    OllamaManagedRuntime,
    OllamaProvider,
    OllamaProviderConfig,
    OllamaSubprocessController,
)


class FakeApi:
    def __init__(
        self,
        *,
        model: str = "qwen3:8b",
        model_size: int = 8 * 1024 * 1024 * 1024,
        size_vram: int | None = None,
        reachable: bool = True,
        available: bool = True,
    ) -> None:
        self.model = model
        self.model_size = model_size
        self.size_vram = model_size if size_vram is None else size_vram
        self.reachable = reachable
        self.available = available
        self.loaded = False
        self.closed = False
        self.chat_payloads: list[dict] = []
        self.generate_payloads: list[dict] = []
        self.unloaded: list[str] = []
        self.block_chat = False
        self.chat_started = asyncio.Event()

    async def list_models(self):
        if not self.reachable:
            raise httpx.ConnectError("offline")
        if not self.available:
            return ()
        return (
            {
                "name": self.model,
                "model": self.model,
                "size": self.model_size,
            },
        )

    async def running_models(self):
        if not self.reachable:
            raise httpx.ConnectError("offline")
        if not self.loaded:
            return ()
        return (
            {
                "name": self.model,
                "model": self.model,
                "size": self.model_size,
                "size_vram": self.size_vram,
                "context_length": 8192,
            },
        )

    async def chat(self, payload):
        self.chat_payloads.append(dict(payload))
        self.loaded = True
        self.chat_started.set()
        if self.block_chat:
            await asyncio.Event().wait()
        return {
            "model": self.model,
            "message": {"role": "assistant", "content": "hello"},
            "done": True,
            "total_duration": 100,
            "eval_count": 4,
        }

    async def generate(self, payload):
        self.generate_payloads.append(dict(payload))
        self.loaded = True
        return {
            "model": self.model,
            "response": "generated",
            "done": True,
            "total_duration": 200,
            "prompt_eval_count": 3,
            "eval_count": 5,
        }

    async def unload(self, model: str) -> None:
        self.unloaded.append(model)
        self.loaded = False

    async def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self, api: FakeApi, *, running: bool = False) -> None:
        self.api = api
        self._running = running
        self.starts = 0
        self.stops = 0

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        self.starts += 1
        self._running = True
        self.api.reachable = True

    async def stop(self) -> None:
        self.stops += 1
        self._running = False
        self.api.reachable = False


def host() -> RuntimeHostFacts:
    return RuntimeHostFacts(
        cpu_count=16,
        host_ram_mb=65536,
        architecture="x86_64",
    )


def single_worker(vram_mb: int = 24576) -> WorkerSpec:
    return WorkerSpec(
        worker_id="ollama-single",
        worker_class="modern-single",
        resources=ResourceShape(
            gpu_count=1,
            total_vram_mb=vram_mb,
            max_single_gpu_vram_mb=vram_mb,
        ),
        gpu_uuids=("GPU-example-one",),
        capabilities=frozenset({"llm.chat", "text.generate"}),
    )


def multi_worker() -> WorkerSpec:
    return WorkerSpec(
        worker_id="ollama-multi",
        worker_class="multi-gpu",
        resources=ResourceShape(
            gpu_count=2,
            total_vram_mb=24576,
            max_single_gpu_vram_mb=12288,
        ),
        gpu_uuids=("GPU-example-a", "GPU-example-b"),
        capabilities=frozenset({"llm.chat", "text.generate"}),
    )


def demand(
    *,
    model_format: str = "ollama",
    estimated_size_mb: int | None = 8000,
    residency: ResidencyPolicy = ResidencyPolicy.PREFER_VRAM,
    topology: GPUTopology = GPUTopology.SINGLE_GPU,
) -> ExecutionDemand:
    return ExecutionDemand(
        model=ModelDemand(
            model_ref="qwen3:8b",
            model_format=model_format,
            topology=ModelTopology.DENSE,
            estimated_size_mb=estimated_size_mb,
        ),
        residency_policy=residency,
        gpu_topology=topology,
        min_gpu_count=(2 if topology is GPUTopology.MULTI_GPU else 0),
    )


def context(
    *,
    worker: WorkerSpec | None = None,
    execution: ExecutionDemand | None = None,
) -> RuntimeCompatibilityContext:
    return RuntimeCompatibilityContext(
        worker=worker or single_worker(),
        host=host(),
        demand=execution or demand(),
    )


def test_ollama_provider_rejects_non_ollama_model_format() -> None:
    report = OllamaProvider().compatibility(
        context(execution=demand(model_format="gguf"))
    )

    assert not report.compatible
    assert "model-format-unsupported" in {
        reason.code for reason in report.reasons
    }


def test_ollama_vram_only_requires_size_and_rechecks_runtime() -> None:
    provider = OllamaProvider()

    missing = provider.compatibility(
        context(
            execution=demand(
                estimated_size_mb=None,
                residency=ResidencyPolicy.VRAM_ONLY,
            )
        )
    )
    assert not missing.compatible
    assert "model-size-required-for-vram-only" in {
        reason.code for reason in missing.reasons
    }

    compatible = provider.compatibility(
        context(
            execution=demand(
                estimated_size_mb=8000,
                residency=ResidencyPolicy.VRAM_ONLY,
            )
        )
    )
    assert compatible.compatible
    assert "vram-residency-runtime-verified" in {
        reason.code for reason in compatible.reasons
    }


def test_ollama_vram_only_rejects_estimate_larger_than_total_vram() -> None:
    report = OllamaProvider().compatibility(
        context(
            worker=single_worker(vram_mb=8192),
            execution=demand(
                estimated_size_mb=12000,
                residency=ResidencyPolicy.VRAM_ONLY,
            ),
        )
    )

    assert not report.compatible
    assert "model-exceeds-total-vram" in {
        reason.code for reason in report.reasons
    }


def test_ollama_multi_gpu_uses_explicit_spread_even_if_model_fits_one_device() -> None:
    report = OllamaProvider().compatibility(
        context(
            worker=multi_worker(),
            execution=demand(
                estimated_size_mb=8000,
                topology=GPUTopology.MULTI_GPU,
            ),
        )
    )

    assert report.compatible
    reason = next(
        reason
        for reason in report.reasons
        if reason.code == "ollama-multi-gpu-spread"
    )
    assert not reason.blocking


def test_ollama_multi_gpu_setup_intent_requests_spread() -> None:
    ctx = context(
        worker=multi_worker(),
        execution=demand(
            estimated_size_mb=18000,
            topology=GPUTopology.MULTI_GPU,
        ),
    )
    intent = OllamaProvider().setup_intent(ctx)

    assert intent.configuration["gpu_uuids"] == [
        "GPU-example-a",
        "GPU-example-b",
    ]
    assert intent.configuration["sched_spread"] is True
    assert intent.configuration["no_cloud"] is True


def test_ollama_cpu_gpu_hybrid_is_automatic_and_advisory() -> None:
    report = OllamaProvider().compatibility(
        context(
            execution=demand(
                residency=ResidencyPolicy.CPU_GPU_HYBRID,
            )
        )
    )

    assert report.compatible
    reason = next(
        reason
        for reason in report.reasons
        if reason.code == "ollama-offload-automatic"
    )
    assert not reason.blocking


def test_ollama_setup_intent_uses_reviewed_download_and_exact_gpu_identity() -> None:
    provider = OllamaProvider(
        OllamaProviderConfig(
            base_url="http://127.0.0.1:11434",
            executable="/opt/ollama/bin/ollama",
            package_reference="ollama-runtime",
            keep_alive="15m",
        )
    )

    intent = provider.setup_intent(context())

    assert intent.provider_id == "ollama"
    assert intent.package_references == ("ollama-runtime",)
    assert intent.model_preparation.value == "download"
    assert intent.model_ref == "qwen3:8b"
    assert intent.requires_privilege
    assert intent.configuration["gpu_uuids"] == ["GPU-example-one"]
    assert intent.configuration["num_parallel"] == 1
    assert intent.configuration["max_loaded_models"] == 1
    assert intent.configuration["no_cloud"] is True
    assert intent.configuration["sched_spread"] is False
    assert intent.configuration["capabilities"] == [
        "llm.chat",
        "text.generate",
    ]


def test_ollama_setup_intent_builds_shared_reviewable_setup_plan() -> None:
    ctx = context()
    provider = OllamaProvider()
    plan = build_runtime_setup_plan(
        catalog=RuntimeCatalog([provider]),
        context=ctx,
        selection=RuntimeSelection(
            mode=RuntimeSelectionMode.EXPLICIT,
            provider_id="ollama",
        ),
        snapshot=SetupHostSnapshot(
            runtime_host=ctx.host,
            deployment_path=DeploymentPath.NIXOS,
            os_id="nixos",
            os_version="26.11",
            service_manager="systemd",
            package_manager="nix",
            available_commands=frozenset(
                {"nix", "systemctl", "nvidia-smi"}
            ),
            privilege_mode=PrivilegeMode.SUDO,
        ),
    )

    assert plan.provider_id == "ollama"
    assert plan.requires_privilege
    assert plan.requires_network
    assert plan.requires_confirmation
    assert any(
        action.kind is SetupActionKind.ENSURE_PACKAGE
        and action.payload["package_reference"] == "ollama"
        for action in plan.actions
    )
    assert any(
        action.kind is SetupActionKind.DOWNLOAD_MODEL
        and action.payload["model_ref"] == "qwen3:8b"
        for action in plan.actions
    )



@pytest.mark.asyncio
async def test_ollama_executor_chat_generate_metrics_and_model_binding() -> None:
    api = FakeApi()
    executor = OllamaExecutor(
        api=api,
        model="qwen3:8b",
        keep_alive="5m",
    )

    chat = await executor.execute(
        JobRequest(
            job_id="chat-1",
            capability="llm.chat",
            payload={
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    )
    assert chat.text == "hello"
    assert chat.metrics["total_duration"] == 100
    assert api.chat_payloads[0]["model"] == "qwen3:8b"
    assert api.chat_payloads[0]["stream"] is False
    assert api.chat_payloads[0]["keep_alive"] == "5m"

    generated = await executor.execute(
        JobRequest(
            job_id="generate-1",
            capability="text.generate",
            payload={"prompt": "hello"},
        )
    )
    assert generated.text == "generated"
    assert generated.metrics["prompt_eval_count"] == 3
    assert generated.metadata["runtime_provider"] == "ollama"

    with pytest.raises(ValueError, match="does not match"):
        await executor.execute(
            JobRequest(
                job_id="wrong-model",
                capability="text.generate",
                payload={"model": "other:latest", "prompt": "hello"},
            )
        )


@pytest.mark.asyncio
async def test_ollama_executor_cancel_aborts_inflight_request() -> None:
    api = FakeApi()
    api.block_chat = True
    executor = OllamaExecutor(
        api=api,
        model="qwen3:8b",
        keep_alive="5m",
    )
    job = JobRequest(
        job_id="chat-cancel",
        capability="llm.chat",
        payload={"messages": [{"role": "user", "content": "wait"}]},
    )

    execution = asyncio.create_task(executor.execute(job))
    await api.chat_started.wait()

    await executor.cancel(job.job_id)

    with pytest.raises(asyncio.CancelledError):
        await execution


@pytest.mark.asyncio
async def test_ollama_residency_maps_api_ps_size_vram() -> None:
    api = FakeApi(
        model_size=10_000,
        size_vram=8_000,
    )
    api.loaded = True
    executor = OllamaExecutor(
        api=api,
        model="qwen3:8b",
        keep_alive="5m",
    )

    report = await executor.residency()

    assert len(report.items) == 1
    assert report.items[0].accelerator_memory_bytes == 8_000
    assert report.items[0].metadata["size_bytes"] == 10_000
    assert report.items[0].metadata["context_length"] == 8192


@pytest.mark.asyncio
async def test_managed_runtime_starts_loads_verifies_and_stops_owned_server() -> None:
    api = FakeApi(reachable=False)
    process = FakeProcess(api)
    runtime = OllamaManagedRuntime(
        api=api,
        process=process,
        context=context(
            execution=demand(
                residency=ResidencyPolicy.VRAM_ONLY,
                estimated_size_mb=8000,
            )
        ),
        model="qwen3:8b",
        keep_alive="5m",
        startup_timeout_seconds=1.0,
    )

    await runtime.start()

    assert process.starts == 1
    assert api.loaded
    assert (await runtime.health()).ready
    assert runtime.executor().capabilities == frozenset(
        {"llm.chat", "text.generate"}
    )

    await runtime.stop()
    assert process.stops == 1
    assert api.unloaded == ["qwen3:8b"]


@pytest.mark.asyncio
async def test_managed_runtime_vram_only_fails_when_api_ps_shows_cpu_offload() -> None:
    api = FakeApi(
        reachable=False,
        model_size=10 * 1024 * 1024 * 1024,
        size_vram=6 * 1024 * 1024 * 1024,
    )
    process = FakeProcess(api)
    runtime = OllamaManagedRuntime(
        api=api,
        process=process,
        context=context(
            execution=demand(
                residency=ResidencyPolicy.VRAM_ONLY,
                estimated_size_mb=8000,
            )
        ),
        model="qwen3:8b",
        keep_alive="5m",
        startup_timeout_seconds=1.0,
    )

    with pytest.raises(RuntimeError, match="outside VRAM"):
        await runtime.start()

    assert process.stops == 1
    assert api.unloaded == ["qwen3:8b"]


@pytest.mark.asyncio
async def test_managed_runtime_multi_gpu_does_not_invent_per_device_proof() -> None:
    api = FakeApi(
        reachable=False,
        model_size=18 * 1024 * 1024 * 1024,
        size_vram=10 * 1024 * 1024 * 1024,
    )
    process = FakeProcess(api)
    runtime = OllamaManagedRuntime(
        api=api,
        process=process,
        context=context(
            worker=multi_worker(),
            execution=demand(
                estimated_size_mb=18000,
                topology=GPUTopology.MULTI_GPU,
            ),
        ),
        model="qwen3:8b",
        keep_alive="5m",
        startup_timeout_seconds=1.0,
    )

    await runtime.start()

    assert (await runtime.health()).ready
    await runtime.stop()


@pytest.mark.asyncio
async def test_managed_runtime_rejects_unowned_external_ollama_server() -> None:
    api = FakeApi(reachable=True)
    process = FakeProcess(api, running=False)
    runtime = OllamaManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model="qwen3:8b",
        keep_alive="5m",
        startup_timeout_seconds=1.0,
    )

    health = await runtime.health()
    assert health.state is RuntimeHealthState.FAILED
    assert not health.ready

    with pytest.raises(RuntimeError, match="external Ollama"):
        await runtime.start()


@pytest.mark.asyncio
async def test_managed_runtime_release_unloads_and_closes_once() -> None:
    api = FakeApi(reachable=True)
    api.loaded = True
    process = FakeProcess(api, running=True)
    runtime = OllamaManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model="qwen3:8b",
        keep_alive="5m",
        startup_timeout_seconds=1.0,
    )

    await runtime.release()
    await runtime.release()

    assert api.unloaded == ["qwen3:8b"]
    assert api.closed


@pytest.mark.asyncio
async def test_http_ollama_api_uses_documented_local_endpoints() -> None:
    requests: list[tuple[str, str, Mapping | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = None
        if request.content:
            body = __import__("json").loads(request.content)
        requests.append((request.method, request.url.path, body))
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "qwen3:8b", "model": "qwen3:8b"}]},
            )
        if request.url.path == "/api/ps":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "qwen3:8b",
                            "model": "qwen3:8b",
                            "size": 100,
                            "size_vram": 100,
                        }
                    ]
                },
            )
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "ok"}},
            )
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"response": "ok"})
        return httpx.Response(404)

    api = HttpOllamaApi(
        "http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
    )
    try:
        await api.list_models()
        await api.running_models()
        await api.chat(
            {
                "model": "qwen3:8b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }
        )
        await api.unload("qwen3:8b")
    finally:
        await api.close()

    assert requests[0] == ("GET", "/api/tags", None)
    assert requests[1] == ("GET", "/api/ps", None)
    assert requests[2][1] == "/api/chat"
    assert requests[3][1] == "/api/generate"
    assert requests[3][2]["keep_alive"] == 0



@pytest.mark.asyncio
async def test_ollama_latest_tag_alias_is_accepted_for_health_and_residency() -> None:
    api = FakeApi(model="qwen3:latest", reachable=False)
    process = FakeProcess(api)
    runtime = OllamaManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model="qwen3",
        keep_alive="5m",
        startup_timeout_seconds=1.0,
    )

    await runtime.start()

    assert (await runtime.health()).ready
    residency = await runtime.residency()
    assert len(residency.items) == 1
    assert residency.items[0].name == "qwen3"


@pytest.mark.asyncio
async def test_ollama_subprocess_environment_pins_uuid_set_and_spread(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class Process:
        returncode = None

        def terminate(self) -> None:
            self.returncode = 0

        async def wait(self) -> int:
            return 0

        def kill(self) -> None:
            self.returncode = -9

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        captured["env"] = dict(kwargs["env"])
        return Process()

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    controller = OllamaSubprocessController(
        executable="/opt/ollama/bin/ollama",
        base_url="http://127.0.0.1:11434",
        gpu_uuids=("GPU-a", "GPU-b"),
        keep_alive="15m",
        no_cloud=True,
        sched_spread=True,
    )
    await controller.start()

    assert captured["args"] == ("/opt/ollama/bin/ollama", "serve")
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"
    assert env["OLLAMA_SCHED_SPREAD"] == "true"
    assert env["OLLAMA_KEEP_ALIVE"] == "15m"
    assert env["OLLAMA_NO_CLOUD"] == "true"
    assert env["OLLAMA_NUM_PARALLEL"] == "1"
    assert env["OLLAMA_MAX_LOADED_MODELS"] == "1"

    await controller.stop()
