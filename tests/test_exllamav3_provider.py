from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace

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
    RuntimeHealthState,
    RuntimeHostFacts,
    RuntimeSelection,
    RuntimeSelectionMode,
    resolve_runtime,
)
from astrumweaver.runtime.providers.exllamav3 import (
    ExLlamaMultiGpuMode,
    ExLlamaV3Executor,
    ExLlamaV3LaunchPolicy,
    ExLlamaV3ManagedRuntime,
    ExLlamaV3Provider,
    ExLlamaV3ProviderConfig,
    ExLlamaV3SubprocessController,
    HttpExLlamaV3Api,
)
from astrumweaver.setup import (
    DeploymentPath,
    PrivilegeMode,
    SetupActionKind,
    SetupHostSnapshot,
    build_runtime_setup_plan,
)


def host(ram_mb: int = 131072, architecture: str = "x86_64") -> RuntimeHostFacts:
    return RuntimeHostFacts(
        cpu_count=32,
        host_ram_mb=ram_mb,
        architecture=architecture,
    )


def single_worker(
    vram_mb: int = 24576,
    *,
    labels: Mapping[str, str] | None = None,
) -> WorkerSpec:
    return WorkerSpec(
        worker_id="exl3-single",
        worker_class="consumer-single",
        resources=ResourceShape(
            gpu_count=1,
            total_vram_mb=vram_mb,
            max_single_gpu_vram_mb=vram_mb,
        ),
        gpu_uuids=("GPU-exl3-one",),
        capabilities=frozenset({"llm.chat", "text.generate"}),
        labels=labels or {},
    )


def multi_worker() -> WorkerSpec:
    return WorkerSpec(
        worker_id="exl3-multi",
        worker_class="consumer-multi",
        resources=ResourceShape(
            gpu_count=2,
            total_vram_mb=36864,
            max_single_gpu_vram_mb=24576,
        ),
        gpu_uuids=("GPU-exl3-large", "GPU-exl3-small"),
        capabilities=frozenset({"llm.chat", "text.generate"}),
    )


def demand(
    *,
    model_ref: str = "upiscium/Qwen3-14B-EXL3",
    model_format: str = "exl3",
    topology: GPUTopology = GPUTopology.SINGLE_GPU,
    model_topology: ModelTopology = ModelTopology.DENSE,
    residency: ResidencyPolicy = ResidencyPolicy.PREFER_VRAM,
    estimated_size_mb: int | None = 12000,
) -> ExecutionDemand:
    return ExecutionDemand(
        model=ModelDemand(
            model_ref=model_ref,
            model_format=model_format,
            topology=model_topology,
            estimated_size_mb=estimated_size_mb,
        ),
        residency_policy=residency,
        gpu_topology=topology,
        min_gpu_count=2 if topology is GPUTopology.MULTI_GPU else 0,
    )


def context(
    *,
    worker: WorkerSpec | None = None,
    execution: ExecutionDemand | None = None,
    host_facts: RuntimeHostFacts | None = None,
) -> RuntimeCompatibilityContext:
    return RuntimeCompatibilityContext(
        worker=worker or single_worker(),
        host=host_facts or host(),
        demand=execution or demand(),
    )


def snapshot(ctx: RuntimeCompatibilityContext) -> SetupHostSnapshot:
    return SetupHostSnapshot(
        runtime_host=ctx.host,
        deployment_path=DeploymentPath.NIXOS,
        os_id="nixos",
        os_version="26.11",
        service_manager="systemd",
        package_manager="nix",
        available_commands=frozenset({"nix", "systemctl", "nvidia-smi", "python"}),
        privilege_mode=PrivilegeMode.SUDO,
    )


def test_exl3_single_gpu_prefer_vram_is_compatible() -> None:
    report = ExLlamaV3Provider().compatibility(context())
    assert report.compatible


def test_exl3_vram_only_requires_model_size_evidence() -> None:
    report = ExLlamaV3Provider().compatibility(
        context(
            execution=demand(
                residency=ResidencyPolicy.VRAM_ONLY,
                estimated_size_mb=None,
            )
        )
    )
    assert not report.compatible
    assert "model-size-required-for-vram-only" in {
        reason.code for reason in report.reasons
    }


def test_exl3_rejects_model_larger_than_total_vram() -> None:
    report = ExLlamaV3Provider().compatibility(
        context(
            execution=demand(
                residency=ResidencyPolicy.VRAM_ONLY,
                estimated_size_mb=30000,
            )
        )
    )
    assert not report.compatible
    assert "model-exceeds-total-vram" in {
        reason.code for reason in report.reasons
    }


def test_non_exl3_format_is_rejected() -> None:
    report = ExLlamaV3Provider().compatibility(
        context(execution=demand(model_format="gguf"))
    )
    assert not report.compatible
    assert "model-format-unsupported" in {
        reason.code for reason in report.reasons
    }


def test_cpu_gpu_hybrid_is_outside_provider_scope() -> None:
    report = ExLlamaV3Provider().compatibility(
        context(execution=demand(residency=ResidencyPolicy.CPU_GPU_HYBRID))
    )
    assert not report.compatible
    assert "cpu-gpu-hybrid-outside-provider-scope" in {
        reason.code for reason in report.reasons
    }


def test_pascal_compute_capability_is_rejected() -> None:
    report = ExLlamaV3Provider().compatibility(
        context(
            worker=single_worker(
                labels={"gpu.compute_capability.min": "6.1"}
            )
        )
    )
    assert not report.compatible
    assert "compute-capability-unsupported" in {
        reason.code for reason in report.reasons
    }


def test_ampere_compute_capability_is_accepted() -> None:
    report = ExLlamaV3Provider().compatibility(
        context(
            worker=single_worker(
                labels={"gpu.compute_capability.min": "8.6"}
            )
        )
    )
    assert report.compatible
    assert "compute-capability-unverified" not in {
        reason.code for reason in report.reasons
    }


def test_current_linux_package_path_rejects_non_x86_64() -> None:
    report = ExLlamaV3Provider().compatibility(
        context(host_facts=host(architecture="aarch64"))
    )
    assert not report.compatible
    assert "architecture-unsupported" in {
        reason.code for reason in report.reasons
    }


def test_multi_gpu_autosplit_uses_all_owned_devices() -> None:
    ctx = context(
        worker=multi_worker(),
        execution=demand(topology=GPUTopology.MULTI_GPU, estimated_size_mb=30000),
    )
    provider = ExLlamaV3Provider()
    report = provider.compatibility(ctx)
    assert report.compatible
    intent = provider.setup_intent(ctx)
    assert intent.configuration["gpu_uuids"] == [
        "GPU-exl3-large",
        "GPU-exl3-small",
    ]
    assert intent.configuration["multi_gpu_mode"] == "autosplit"
    assert intent.configuration["tensor_parallel"] is False
    assert intent.configuration["gpu_split_auto"] is True
    assert intent.configuration["tabby_config"]["model"]["gpu_split_auto"] is True


def test_multi_gpu_tensor_parallel_mode_is_explicit() -> None:
    ctx = context(
        worker=multi_worker(),
        execution=demand(topology=GPUTopology.MULTI_GPU, estimated_size_mb=30000),
    )
    provider = ExLlamaV3Provider(
        ExLlamaV3ProviderConfig(
            multi_gpu_mode=ExLlamaMultiGpuMode.TENSOR_PARALLEL,
            tensor_parallel_backend="native",
            gpu_split=(20.0, 10.0),
        )
    )
    intent = provider.setup_intent(ctx)
    assert intent.configuration["tensor_parallel"] is True
    assert intent.configuration["tensor_parallel_backend"] == "native"
    assert intent.configuration["gpu_split"] == [20.0, 10.0]
    assert intent.configuration["tabby_config"]["model"]["tensor_parallel"] is True


def test_autosplit_reserve_must_match_worker_gpu_count() -> None:
    provider = ExLlamaV3Provider(
        ExLlamaV3ProviderConfig(autosplit_reserve_mb=(96,))
    )
    report = provider.compatibility(
        context(
            worker=multi_worker(),
            execution=demand(topology=GPUTopology.MULTI_GPU),
        )
    )
    assert not report.compatible
    assert "autosplit-reserve-length-mismatch" in {
        reason.code for reason in report.reasons
    }


def test_explicit_gpu_split_must_match_worker_gpu_count() -> None:
    provider = ExLlamaV3Provider(
        ExLlamaV3ProviderConfig(gpu_split=(20.0, 10.0, 5.0))
    )
    report = provider.compatibility(
        context(
            worker=multi_worker(),
            execution=demand(topology=GPUTopology.MULTI_GPU),
        )
    )
    assert not report.compatible
    assert "gpu-split-length-mismatch" in {
        reason.code for reason in report.reasons
    }


def test_setup_plan_preserves_explicit_exllamav3_choice() -> None:
    ctx = context()
    provider = ExLlamaV3Provider()
    selection = RuntimeSelection(
        mode=RuntimeSelectionMode.EXPLICIT,
        provider_id="exllamav3",
    )
    resolution = resolve_runtime(
        catalog=RuntimeCatalog([provider]),
        context=ctx,
        selection=selection,
    )
    assert resolution.selected_provider_id == "exllamav3"

    intent = provider.setup_intent(ctx)
    assert intent.package_references == ("tabbyAPI[cu12]", "exllamav3")
    assert intent.configuration["served_model_name"] == "Qwen3-14B-EXL3"
    assert intent.model_preparation.value == "download"
    assert intent.configuration["tabby_config"]["network"]["disable_auth"] is True
    assert intent.configuration["tabby_config"]["network"]["allowed_origins"] == []

    plan = build_runtime_setup_plan(
        catalog=RuntimeCatalog([provider]),
        context=ctx,
        selection=selection,
        snapshot=snapshot(ctx),
    )
    assert plan.provider_id == "exllamav3"
    assert any(
        action.kind is SetupActionKind.DOWNLOAD_MODEL
        and action.payload["model_ref"] == "upiscium/Qwen3-14B-EXL3"
        for action in plan.actions
    )
    assert any(
        action.kind is SetupActionKind.RENDER_CONFIG
        and action.payload["configuration"]["tabby_config"]["model"]["backend"]
        == "exllamav3"
        for action in plan.actions
    )


def test_local_model_reference_uses_reference_only_and_parent_dir() -> None:
    ctx = context(
        execution=demand(model_ref="/models/Qwen3-EXL3")
    )
    intent = ExLlamaV3Provider().setup_intent(ctx)
    assert intent.model_preparation.value == "reference_only"
    assert intent.configuration["model_dir"] == "/models"
    assert intent.configuration["served_model_name"] == "Qwen3-EXL3"


def test_create_runtime_rejects_tampered_gpu_ownership() -> None:
    provider = ExLlamaV3Provider()
    ctx = context()
    intent = provider.setup_intent(ctx)
    tampered = replace(
        intent,
        configuration={
            **dict(intent.configuration),
            "gpu_uuids": ["GPU-not-owned"],
        },
    )
    with pytest.raises(ValueError, match="GPU UUIDs"):
        provider.create_runtime(ctx, tampered)


def test_create_runtime_rejects_tampered_multi_gpu_mode() -> None:
    provider = ExLlamaV3Provider()
    ctx = context()
    intent = provider.setup_intent(ctx)
    tampered = replace(
        intent,
        configuration={
            **dict(intent.configuration),
            "multi_gpu_mode": "tensor_parallel",
        },
    )
    with pytest.raises(ValueError, match="multi-GPU mode"):
        provider.create_runtime(ctx, tampered)


def test_subprocess_command_uses_reviewed_config_path() -> None:
    controller = ExLlamaV3SubprocessController(
        executable="python",
        entrypoint="/opt/tabbyAPI/main.py",
        config_path="/run/astrumweaver/exl3.yml",
        gpu_uuids=("GPU-a", "GPU-b"),
    )
    assert controller.command() == (
        "python",
        "/opt/tabbyAPI/main.py",
        "--config",
        "/run/astrumweaver/exl3.yml",
    )


@pytest.mark.asyncio
async def test_subprocess_environment_pins_exact_worker_gpu_set(monkeypatch) -> None:
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
    controller = ExLlamaV3SubprocessController(
        executable="python",
        entrypoint="/opt/tabbyAPI/main.py",
        config_path="/run/astrumweaver/exl3.yml",
        gpu_uuids=("GPU-a", "GPU-b"),
    )
    await controller.start()
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"
    await controller.stop()


@pytest.mark.asyncio
async def test_http_api_uses_tabby_health_models_and_oai_endpoints() -> None:
    requests: list[tuple[str, str, Mapping | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, body))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy", "issues": []})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "Qwen3-14B-EXL3"}]},
            )
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "hello"}}
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                },
            )
        if request.url.path == "/v1/completions":
            return httpx.Response(
                200,
                json={
                    "choices": [{"text": "generated"}],
                    "usage": {"total_tokens": 4},
                },
            )
        return httpx.Response(404)

    api = HttpExLlamaV3Api(
        "http://127.0.0.1:5000",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert await api.health()
        assert await api.models() == ("Qwen3-14B-EXL3",)
        await api.chat(
            {
                "model": "Qwen3-14B-EXL3",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }
        )
        await api.completion(
            {
                "model": "Qwen3-14B-EXL3",
                "prompt": "hi",
                "stream": False,
            }
        )
    finally:
        await api.close()

    assert [path for _, path, _ in requests] == [
        "/health",
        "/v1/models",
        "/v1/chat/completions",
        "/v1/completions",
    ]


@pytest.mark.asyncio
async def test_http_health_rejects_unhealthy_status() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"status": "unhealthy", "issues": [{"message": "failed"}]},
        )

    api = HttpExLlamaV3Api(
        "http://127.0.0.1:5000",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert not await api.health()
    finally:
        await api.close()


class FakeApi:
    def __init__(
        self,
        *,
        reachable: bool = True,
        models: tuple[str, ...] = ("Qwen3-14B-EXL3",),
    ) -> None:
        self.reachable = reachable
        self.model_ids = models
        self.closed = False
        self.chat_started = asyncio.Event()
        self.block_chat = False
        self.chat_payloads: list[dict] = []

    async def health(self) -> bool:
        return self.reachable

    async def models(self) -> tuple[str, ...]:
        if not self.reachable:
            raise httpx.ConnectError("offline")
        return self.model_ids

    async def chat(self, payload: Mapping[str, object]):
        self.chat_payloads.append(dict(payload))
        self.chat_started.set()
        if self.block_chat:
            await asyncio.Event().wait()
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }

    async def completion(self, payload: Mapping[str, object]):
        return {
            "choices": [{"text": "generated"}],
            "usage": {"total_tokens": 4},
        }

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


def launch_policy() -> ExLlamaV3LaunchPolicy:
    return ExLlamaV3LaunchPolicy(
        gpu_uuids=("GPU-exl3-one",),
        multi_gpu_mode=ExLlamaMultiGpuMode.AUTOSPLIT,
        tensor_parallel=False,
        tensor_parallel_backend="native",
        gpu_split_auto=False,
        gpu_split=None,
        autosplit_reserve_mb=None,
        cache_mode="FP16",
        max_seq_len=None,
        max_batch_size=None,
    )


@pytest.mark.asyncio
async def test_executor_chat_metrics_model_binding_and_cancel() -> None:
    api = FakeApi()
    executor = ExLlamaV3Executor(
        api=api,
        model_ref="upiscium/Qwen3-14B-EXL3",
        served_model_name="Qwen3-14B-EXL3",
        residency_metadata={},
    )
    result = await executor.execute(
        JobRequest(
            job_id="chat",
            capability="llm.chat",
            payload={"messages": [{"role": "user", "content": "hi"}]},
        )
    )
    assert result.text == "hello"
    assert result.metrics["total_tokens"] == 5
    assert api.chat_payloads[0]["stream"] is False

    with pytest.raises(ValueError, match="does not match"):
        await executor.execute(
            JobRequest(
                job_id="bad-model",
                capability="llm.chat",
                payload={"model": "other", "messages": []},
            )
        )

    api.chat_started.clear()
    api.block_chat = True
    execution = asyncio.create_task(
        executor.execute(
            JobRequest(
                job_id="cancel",
                capability="llm.chat",
                payload={"messages": []},
            )
        )
    )
    await api.chat_started.wait()
    await executor.cancel("cancel")
    with pytest.raises(asyncio.CancelledError):
        await execution


@pytest.mark.asyncio
async def test_residency_reports_policy_without_inventing_memory_bytes() -> None:
    executor = ExLlamaV3Executor(
        api=FakeApi(),
        model_ref="upiscium/Qwen3-14B-EXL3",
        served_model_name="Qwen3-14B-EXL3",
        residency_metadata={
            "gpu_uuids": ["GPU-a", "GPU-b"],
            "multi_gpu_mode": "autosplit",
        },
    )
    report = await executor.residency()
    assert report.items[0].accelerator_memory_bytes is None
    assert report.items[0].metadata["gpu_uuids"] == ["GPU-a", "GPU-b"]


@pytest.mark.asyncio
async def test_managed_runtime_lifecycle_and_external_collision() -> None:
    api = FakeApi(reachable=False)
    process = FakeProcess(api)
    runtime = ExLlamaV3ManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model_ref="upiscium/Qwen3-14B-EXL3",
        served_model_name="Qwen3-14B-EXL3",
        startup_timeout_seconds=1.0,
        launch_policy=launch_policy(),
    )
    await runtime.start()
    assert process.starts == 1
    assert (await runtime.health()).ready
    await runtime.stop()
    assert process.stops == 1
    await runtime.release()
    await runtime.release()
    assert api.closed

    external_api = FakeApi(reachable=True)
    external = ExLlamaV3ManagedRuntime(
        api=external_api,
        process=FakeProcess(external_api, running=False),
        context=context(),
        model_ref="upiscium/Qwen3-14B-EXL3",
        served_model_name="Qwen3-14B-EXL3",
        startup_timeout_seconds=1.0,
        launch_policy=launch_policy(),
    )
    assert (await external.health()).state is RuntimeHealthState.FAILED
    with pytest.raises(RuntimeError, match="external TabbyAPI"):
        await external.start()


@pytest.mark.asyncio
async def test_failed_start_cleans_up_owned_process() -> None:
    api = FakeApi(reachable=False, models=("wrong-model",))
    process = FakeProcess(api)
    runtime = ExLlamaV3ManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model_ref="upiscium/Qwen3-14B-EXL3",
        served_model_name="Qwen3-14B-EXL3",
        startup_timeout_seconds=1.0,
        launch_policy=launch_policy(),
    )
    with pytest.raises(RuntimeError, match="configured ExLlamaV3 model"):
        await runtime.start()
    assert process.starts == 1
    assert process.stops == 1
