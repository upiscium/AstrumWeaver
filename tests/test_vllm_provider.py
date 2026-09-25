from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

import httpx
import pytest

from astrumweaver import AcceleratorDevice, ResourceShape, WorkerSpec
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
from astrumweaver.runtime.providers.vllm import (
    HttpVllmApi,
    VllmExecutor,
    VllmLaunchPolicy,
    VllmManagedRuntime,
    VllmProvider,
    VllmProviderConfig,
    VllmSubprocessController,
)
from astrumweaver.setup import (
    DeploymentPath,
    PrivilegeMode,
    SetupActionKind,
    SetupHostSnapshot,
    build_runtime_setup_plan,
)


def host(ram_mb: int = 131072) -> RuntimeHostFacts:
    return RuntimeHostFacts(
        cpu_count=32,
        host_ram_mb=ram_mb,
        architecture="x86_64",
    )


def single_worker(vram_mb: int = 24576) -> WorkerSpec:
    return WorkerSpec(
        worker_id="vllm-single",
        worker_class="modern-single",
        resources=ResourceShape(
            gpu_count=1,
            total_vram_mb=vram_mb,
            max_single_gpu_vram_mb=vram_mb,
        ),
        gpu_uuids=("GPU-example-one",),
        capabilities=frozenset({"llm.chat", "text.generate"}),
    )


def homogeneous_multi_worker() -> WorkerSpec:
    return WorkerSpec(
        worker_id="vllm-multi",
        worker_class="multi-gpu",
        resources=ResourceShape(
            gpu_count=2,
            total_vram_mb=49152,
            max_single_gpu_vram_mb=24576,
        ),
        gpu_uuids=("GPU-example-a", "GPU-example-b"),
        accelerators=(
            AcceleratorDevice(
                "GPU-example-a",
                24576,
                compute_capability="8.6",
                device_class="NVIDIA RTX 3090",
            ),
            AcceleratorDevice(
                "GPU-example-b",
                24576,
                compute_capability="8.6",
                device_class="NVIDIA RTX 3090",
            ),
        ),
        capabilities=frozenset({"llm.chat", "text.generate"}),
    )


def heterogeneous_multi_worker() -> WorkerSpec:
    return WorkerSpec(
        worker_id="vllm-hetero",
        worker_class="multi-gpu",
        resources=ResourceShape(
            gpu_count=2,
            total_vram_mb=36864,
            max_single_gpu_vram_mb=24576,
        ),
        gpu_uuids=("GPU-example-large", "GPU-example-small"),
        accelerators=(
            AcceleratorDevice(
                "GPU-example-large",
                24576,
                compute_capability="8.6",
                device_class="NVIDIA RTX 3090",
            ),
            AcceleratorDevice(
                "GPU-example-small",
                12288,
                compute_capability="8.6",
                device_class="NVIDIA RTX 3060",
            ),
        ),
        capabilities=frozenset({"llm.chat", "text.generate"}),
    )


def demand(
    *,
    model_ref: str = "Qwen/Qwen3-8B",
    model_format: str = "huggingface",
    topology: GPUTopology = GPUTopology.SINGLE_GPU,
    model_topology: ModelTopology = ModelTopology.DENSE,
    residency: ResidencyPolicy = ResidencyPolicy.PREFER_VRAM,
    estimated_size_mb: int | None = 16000,
    min_host_ram_mb: int = 0,
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
        min_gpu_count=(
            2 if topology is GPUTopology.MULTI_GPU else 0
        ),
        min_host_ram_mb=min_host_ram_mb,
        preferred_host_ram_mb=min_host_ram_mb,
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
        available_commands=frozenset(
            {"nix", "systemctl", "nvidia-smi"}
        ),
        privilege_mode=PrivilegeMode.SUDO,
    )


def test_vllm_provider_rejects_unsupported_model_format() -> None:
    report = VllmProvider().compatibility(
        context(execution=demand(model_format="gguf"))
    )

    assert not report.compatible
    assert "model-format-unsupported" in {
        reason.code for reason in report.reasons
    }


def test_vllm_explicit_selection_is_preserved() -> None:
    ctx = context()
    resolution = resolve_runtime(
        catalog=RuntimeCatalog([VllmProvider()]),
        context=ctx,
        selection=RuntimeSelection(
            mode=RuntimeSelectionMode.EXPLICIT,
            provider_id="vllm",
        ),
    )

    assert resolution.selected_provider_id == "vllm"


def test_single_gpu_derives_tp_one_and_exact_device_uuid() -> None:
    intent = VllmProvider().setup_intent(context())

    assert intent.configuration["tensor_parallel_size"] == 1
    assert intent.configuration["gpu_uuids"] == [
        "GPU-example-one"
    ]
    assert intent.configuration["enable_expert_parallel"] is False


def test_homogeneous_multi_gpu_derives_full_worker_tensor_parallel() -> None:
    ctx = context(
        worker=homogeneous_multi_worker(),
        execution=demand(topology=GPUTopology.MULTI_GPU),
    )
    provider = VllmProvider()

    report = provider.compatibility(ctx)
    assert report.compatible

    intent = provider.setup_intent(ctx)
    assert intent.configuration["tensor_parallel_size"] == 2
    assert intent.configuration["gpu_uuids"] == [
        "GPU-example-a",
        "GPU-example-b",
    ]


def test_multi_gpu_rejects_partial_tensor_parallel_size() -> None:
    provider = VllmProvider(
        VllmProviderConfig(tensor_parallel_size=1)
    )
    report = provider.compatibility(
        context(
            worker=homogeneous_multi_worker(),
            execution=demand(topology=GPUTopology.MULTI_GPU),
        )
    )

    assert not report.compatible
    assert "tensor-parallel-size-mismatch" in {
        reason.code for reason in report.reasons
    }


def test_multi_gpu_requires_per_device_accelerator_facts() -> None:
    worker = WorkerSpec(
        worker_id="vllm-unknown",
        worker_class="multi-gpu",
        resources=ResourceShape(
            gpu_count=2,
            total_vram_mb=49152,
            max_single_gpu_vram_mb=24576,
        ),
        gpu_uuids=("GPU-a", "GPU-b"),
        capabilities=frozenset({"llm.chat", "text.generate"}),
    )

    report = VllmProvider().compatibility(
        context(
            worker=worker,
            execution=demand(topology=GPUTopology.MULTI_GPU),
        )
    )

    assert not report.compatible
    assert "accelerator-facts-required" in {
        reason.code for reason in report.reasons
    }


def test_multi_gpu_rejects_mixed_device_class_even_with_equal_vram() -> None:
    worker = WorkerSpec(
        worker_id="vllm-mixed-class",
        worker_class="multi-gpu",
        resources=ResourceShape(
            gpu_count=2,
            total_vram_mb=49152,
            max_single_gpu_vram_mb=24576,
        ),
        gpu_uuids=("GPU-a", "GPU-b"),
        accelerators=(
            AcceleratorDevice(
                "GPU-a",
                24576,
                compute_capability="8.6",
                device_class="NVIDIA RTX 3090",
            ),
            AcceleratorDevice(
                "GPU-b",
                24576,
                compute_capability="8.9",
                device_class="NVIDIA RTX 4090",
            ),
        ),
        capabilities=frozenset({"llm.chat", "text.generate"}),
    )

    report = VllmProvider().compatibility(
        context(
            worker=worker,
            execution=demand(topology=GPUTopology.MULTI_GPU),
        )
    )

    assert not report.compatible
    codes = {reason.code for reason in report.reasons}
    assert "heterogeneous-device-class-unsupported" in codes
    assert "heterogeneous-compute-capability-unsupported" in codes


def test_multi_gpu_rejects_heterogeneous_vram_capacity() -> None:
    report = VllmProvider().compatibility(
        context(
            worker=heterogeneous_multi_worker(),
            execution=demand(topology=GPUTopology.MULTI_GPU),
        )
    )

    assert not report.compatible
    assert "heterogeneous-vram-unsupported" in {
        reason.code for reason in report.reasons
    }


def test_moe_multi_gpu_enables_expert_parallel_by_default() -> None:
    ctx = context(
        worker=homogeneous_multi_worker(),
        execution=demand(
            topology=GPUTopology.MULTI_GPU,
            model_topology=ModelTopology.MOE,
        ),
    )
    provider = VllmProvider()

    report = provider.compatibility(ctx)

    assert report.compatible
    reason = next(
        reason
        for reason in report.reasons
        if reason.code == "expert-parallel-enabled"
    )
    assert not reason.blocking
    assert provider.setup_intent(
        ctx
    ).configuration["enable_expert_parallel"] is True


def test_dense_model_rejects_forced_expert_parallel() -> None:
    provider = VllmProvider(
        VllmProviderConfig(enable_expert_parallel=True)
    )

    report = provider.compatibility(
        context(
            worker=homogeneous_multi_worker(),
            execution=demand(
                topology=GPUTopology.MULTI_GPU,
                model_topology=ModelTopology.DENSE,
            ),
        )
    )

    assert not report.compatible
    assert "expert-parallel-requires-moe" in {
        reason.code for reason in report.reasons
    }


def test_expert_parallel_rejects_single_gpu() -> None:
    provider = VllmProvider(
        VllmProviderConfig(enable_expert_parallel=True)
    )

    report = provider.compatibility(
        context(
            execution=demand(model_topology=ModelTopology.MOE)
        )
    )

    assert not report.compatible
    assert "expert-parallel-requires-multi-gpu" in {
        reason.code for reason in report.reasons
    }


def test_vram_only_requires_size_and_disallows_cpu_offload() -> None:
    missing = VllmProvider().compatibility(
        context(
            execution=demand(
                residency=ResidencyPolicy.VRAM_ONLY,
                estimated_size_mb=None,
            )
        )
    )
    assert not missing.compatible
    assert "model-size-required-for-vram-only" in {
        reason.code for reason in missing.reasons
    }

    offloaded = VllmProvider(
        VllmProviderConfig(cpu_offload_gb=4)
    ).compatibility(
        context(
            execution=demand(
                residency=ResidencyPolicy.VRAM_ONLY,
            )
        )
    )
    assert not offloaded.compatible
    assert "cpu-offload-conflicts-with-vram-only" in {
        reason.code for reason in offloaded.reasons
    }


def test_vram_only_uses_gpu_memory_utilization_budget() -> None:
    provider = VllmProvider(
        VllmProviderConfig(gpu_memory_utilization=0.5)
    )

    report = provider.compatibility(
        context(
            worker=single_worker(vram_mb=24576),
            execution=demand(
                residency=ResidencyPolicy.VRAM_ONLY,
                estimated_size_mb=16000,
            ),
        )
    )

    assert not report.compatible
    assert "model-exceeds-effective-memory-budget" in {
        reason.code for reason in report.reasons
    }


def test_prefer_vram_rejects_known_model_beyond_effective_budget() -> None:
    report = VllmProvider().compatibility(
        context(
            worker=single_worker(vram_mb=8192),
            execution=demand(
                residency=ResidencyPolicy.PREFER_VRAM,
                estimated_size_mb=12000,
            ),
        )
    )

    assert not report.compatible
    assert "model-exceeds-effective-memory-budget" in {
        reason.code for reason in report.reasons
    }


def test_cpu_gpu_hybrid_requires_explicit_offload_budget() -> None:
    report = VllmProvider().compatibility(
        context(
            execution=demand(
                residency=ResidencyPolicy.CPU_GPU_HYBRID,
            )
        )
    )

    assert not report.compatible
    assert "cpu-offload-not-configured" in {
        reason.code for reason in report.reasons
    }


def test_cpu_gpu_hybrid_accepts_explicit_offload_with_advisory() -> None:
    provider = VllmProvider(
        VllmProviderConfig(cpu_offload_gb=8)
    )
    ctx = context(
        worker=single_worker(vram_mb=12288),
        execution=demand(
            residency=ResidencyPolicy.CPU_GPU_HYBRID,
            estimated_size_mb=18000,
            min_host_ram_mb=16384,
        ),
    )

    report = provider.compatibility(ctx)

    assert report.compatible
    reason = next(
        reason
        for reason in report.reasons
        if reason.code == "cpu-offload-interconnect-sensitive"
    )
    assert not reason.blocking

    intent = provider.setup_intent(ctx)
    assert intent.configuration["cpu_offload_gb"] == 8


def test_cpu_offload_rejects_obvious_host_ram_overcommit() -> None:
    provider = VllmProvider(
        VllmProviderConfig(cpu_offload_gb=20)
    )

    report = provider.compatibility(
        context(
            worker=homogeneous_multi_worker(),
            host_facts=host(ram_mb=32768),
            execution=demand(
                topology=GPUTopology.MULTI_GPU,
                residency=ResidencyPolicy.CPU_GPU_HYBRID,
                estimated_size_mb=50000,
            ),
        )
    )

    assert not report.compatible
    assert "cpu-offload-exceeds-host-ram" in {
        reason.code for reason in report.reasons
    }


def test_remote_model_ref_requests_reviewed_download_setup_plan() -> None:
    ctx = context(
        execution=demand(model_ref="Qwen/Qwen3-8B")
    )
    provider = VllmProvider()

    intent = provider.setup_intent(ctx)
    assert intent.model_preparation.value == "download"

    plan = build_runtime_setup_plan(
        catalog=RuntimeCatalog([provider]),
        context=ctx,
        selection=RuntimeSelection(
            mode=RuntimeSelectionMode.EXPLICIT,
            provider_id="vllm",
        ),
        snapshot=snapshot(ctx),
    )

    assert any(
        action.kind is SetupActionKind.DOWNLOAD_MODEL
        and action.payload["model_ref"] == "Qwen/Qwen3-8B"
        and action.requires_network
        and action.requires_confirmation
        for action in plan.actions
    )


def test_local_model_ref_uses_reference_only_setup() -> None:
    ctx = context(
        execution=demand(model_ref="/models/qwen")
    )
    provider = VllmProvider()

    intent = provider.setup_intent(ctx)
    assert intent.model_preparation.value == "reference_only"

    plan = build_runtime_setup_plan(
        catalog=RuntimeCatalog([provider]),
        context=ctx,
        selection=RuntimeSelection(
            mode=RuntimeSelectionMode.EXPLICIT,
            provider_id="vllm",
        ),
        snapshot=snapshot(ctx),
    )

    assert any(
        action.kind is SetupActionKind.VERIFY_MODEL_REFERENCE
        and action.payload["model_ref"] == "/models/qwen"
        for action in plan.actions
    )
    assert not any(
        action.kind is SetupActionKind.DOWNLOAD_MODEL
        for action in plan.actions
    )


def test_single_gpu_command_uses_physical_uuid_device_ids() -> None:
    controller = VllmSubprocessController(
        executable="vllm",
        base_url="http://127.0.0.1:8000",
        model_ref="Qwen/Qwen3-8B",
        served_model_name="astrumweaver",
        launch_policy=VllmLaunchPolicy(
            tensor_parallel_size=1,
            enable_expert_parallel=False,
            gpu_memory_utilization=0.9,
            cpu_offload_gb=0,
            device_ids=("GPU-one",),
        ),
        generation_config="vllm",
        trust_remote_code=False,
        enforce_eager=False,
    )

    command = controller.command()

    assert command[:3] == (
        "vllm",
        "serve",
        "Qwen/Qwen3-8B",
    )
    assert command[command.index("--device-ids") + 1] == "GPU-one"
    assert "--tensor-parallel-size" not in command
    assert command[
        command.index("--gpu-memory-utilization") + 1
    ] == "0.9"
    assert command[
        command.index("--served-model-name") + 1
    ] == "astrumweaver"
    assert command[
        command.index("--generation-config") + 1
    ] == "vllm"


def test_multi_gpu_command_uses_uuid_set_mp_tp_and_expert_parallel() -> None:
    controller = VllmSubprocessController(
        executable="vllm",
        base_url="http://127.0.0.1:8000",
        model_ref="Qwen/Qwen3-30B-A3B",
        served_model_name="astrumweaver",
        launch_policy=VllmLaunchPolicy(
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            gpu_memory_utilization=0.92,
            cpu_offload_gb=4,
            device_ids=("GPU-a", "GPU-b"),
        ),
        generation_config="vllm",
        trust_remote_code=False,
        enforce_eager=True,
    )

    command = controller.command()

    assert command[command.index("--device-ids") + 1] == "GPU-a,GPU-b"
    assert command[
        command.index("--tensor-parallel-size") + 1
    ] == "2"
    assert command[
        command.index("--distributed-executor-backend") + 1
    ] == "mp"
    assert "--enable-expert-parallel" in command
    assert command[command.index("--cpu-offload-gb") + 1] == "4"
    assert "--enforce-eager" in command


@pytest.mark.asyncio
async def test_subprocess_invokes_command_without_cuda_visible_devices(
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
        captured["kwargs"] = dict(kwargs)
        return Process()

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    controller = VllmSubprocessController(
        executable="vllm",
        base_url="http://127.0.0.1:8000",
        model_ref="Qwen/Qwen3-8B",
        served_model_name="astrumweaver",
        launch_policy=VllmLaunchPolicy(
            tensor_parallel_size=1,
            enable_expert_parallel=False,
            gpu_memory_utilization=0.92,
            cpu_offload_gb=0,
            device_ids=("GPU-one",),
        ),
        generation_config="vllm",
        trust_remote_code=False,
        enforce_eager=False,
    )

    await controller.start()

    assert "--device-ids" in captured["args"]
    assert "env" not in captured["kwargs"]

    await controller.stop()


@pytest.mark.asyncio
async def test_http_api_uses_health_models_and_openai_endpoints() -> None:
    requests: list[tuple[str, str, Mapping | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = None
        if request.content:
            body = json.loads(request.content)
        requests.append((request.method, request.url.path, body))
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "astrumweaver",
                            "object": "model",
                        }
                    ],
                },
            )
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
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

    api = HttpVllmApi(
        "http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert await api.health()
        assert await api.models() == ("astrumweaver",)
        chat = await api.chat(
            {
                "model": "astrumweaver",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }
        )
        assert chat["choices"][0]["message"]["content"] == "hello"
        completion = await api.completion(
            {
                "model": "astrumweaver",
                "prompt": "hi",
                "stream": False,
            }
        )
        assert completion["choices"][0]["text"] == "generated"
    finally:
        await api.close()

    assert [path for _, path, _ in requests] == [
        "/health",
        "/v1/models",
        "/v1/chat/completions",
        "/v1/completions",
    ]


class FakeApi:
    def __init__(
        self,
        *,
        reachable: bool = True,
        models: tuple[str, ...] = ("astrumweaver",),
    ) -> None:
        self.reachable = reachable
        self.model_ids = models
        self.closed = False
        self.chat_started = asyncio.Event()
        self.block_chat = False
        self.chat_payloads: list[dict] = []
        self.completion_payloads: list[dict] = []

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
        self.completion_payloads.append(dict(payload))
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


@pytest.mark.asyncio
async def test_executor_chat_completion_metrics_and_model_binding() -> None:
    api = FakeApi()
    executor = VllmExecutor(
        api=api,
        model_ref="Qwen/Qwen3-8B",
        served_model_name="astrumweaver",
        residency_metadata={},
    )

    chat = await executor.execute(
        JobRequest(
            job_id="chat",
            capability="llm.chat",
            payload={
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    )
    assert chat.text == "hello"
    assert chat.metrics["total_tokens"] == 5
    assert api.chat_payloads[0]["model"] == "astrumweaver"
    assert api.chat_payloads[0]["stream"] is False

    completion = await executor.execute(
        JobRequest(
            job_id="completion",
            capability="text.generate",
            payload={"prompt": "hi"},
        )
    )
    assert completion.text == "generated"

    with pytest.raises(ValueError, match="does not match"):
        await executor.execute(
            JobRequest(
                job_id="other-model",
                capability="text.generate",
                payload={"model": "other", "prompt": "hi"},
            )
        )


@pytest.mark.asyncio
async def test_executor_cancellation_aborts_inflight_request() -> None:
    api = FakeApi()
    api.block_chat = True
    executor = VllmExecutor(
        api=api,
        model_ref="Qwen/Qwen3-8B",
        served_model_name="astrumweaver",
        residency_metadata={},
    )

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
    executor = VllmExecutor(
        api=FakeApi(),
        model_ref="Qwen/Qwen3-8B",
        served_model_name="astrumweaver",
        residency_metadata={
            "tensor_parallel_size": 2,
            "expert_parallel": False,
            "gpu_memory_utilization": 0.92,
        },
    )

    report = await executor.residency()

    assert len(report.items) == 1
    assert report.items[0].accelerator_memory_bytes is None
    assert report.items[0].metadata["tensor_parallel_size"] == 2
    assert report.items[0].metadata["gpu_memory_utilization"] == 0.92


@pytest.mark.asyncio
async def test_managed_runtime_starts_checks_alias_and_stops() -> None:
    api = FakeApi(reachable=False)
    process = FakeProcess(api)
    runtime = VllmManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model_ref="Qwen/Qwen3-8B",
        served_model_name="astrumweaver",
        startup_timeout_seconds=1.0,
        launch_policy=VllmLaunchPolicy(
            tensor_parallel_size=1,
            enable_expert_parallel=False,
            gpu_memory_utilization=0.92,
            cpu_offload_gb=0,
            device_ids=("GPU-example-one",),
        ),
    )

    await runtime.start()

    assert process.starts == 1
    assert (await runtime.health()).ready
    await runtime.stop()
    assert process.stops == 1


@pytest.mark.asyncio
async def test_managed_runtime_rejects_external_server() -> None:
    api = FakeApi(reachable=True)
    process = FakeProcess(api, running=False)
    runtime = VllmManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model_ref="Qwen/Qwen3-8B",
        served_model_name="astrumweaver",
        startup_timeout_seconds=1.0,
        launch_policy=VllmLaunchPolicy(
            tensor_parallel_size=1,
            enable_expert_parallel=False,
            gpu_memory_utilization=0.92,
            cpu_offload_gb=0,
            device_ids=("GPU-example-one",),
        ),
    )

    health = await runtime.health()
    assert health.state is RuntimeHealthState.FAILED

    with pytest.raises(RuntimeError, match="external vLLM"):
        await runtime.start()


@pytest.mark.asyncio
async def test_managed_runtime_cleans_up_failed_start() -> None:
    api = FakeApi(reachable=False, models=("wrong-alias",))
    process = FakeProcess(api)
    runtime = VllmManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model_ref="Qwen/Qwen3-8B",
        served_model_name="astrumweaver",
        startup_timeout_seconds=1.0,
        launch_policy=VllmLaunchPolicy(
            tensor_parallel_size=1,
            enable_expert_parallel=False,
            gpu_memory_utilization=0.92,
            cpu_offload_gb=0,
            device_ids=("GPU-example-one",),
        ),
    )

    with pytest.raises(RuntimeError, match="model alias"):
        await runtime.start()

    assert process.starts == 1
    assert process.stops == 1


@pytest.mark.asyncio
async def test_managed_runtime_release_closes_api_once() -> None:
    api = FakeApi()
    process = FakeProcess(api, running=True)
    runtime = VllmManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model_ref="Qwen/Qwen3-8B",
        served_model_name="astrumweaver",
        startup_timeout_seconds=1.0,
        launch_policy=VllmLaunchPolicy(
            tensor_parallel_size=1,
            enable_expert_parallel=False,
            gpu_memory_utilization=0.92,
            cpu_offload_gb=0,
            device_ids=("GPU-example-one",),
        ),
    )

    await runtime.release()
    await runtime.release()

    assert api.closed
