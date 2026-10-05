from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

import httpx
import pytest

from astrumweaver import ResourceShape, WorkerSpec
from astrumweaver.execution import JobExecutionError, JobRequest
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
    RuntimeSelectionError,
    RuntimeSelectionMode,
    resolve_runtime,
)
from astrumweaver.runtime.providers.llama_cpp import (
    HttpLlamaCppApi,
    LlamaCppExecutor,
    LlamaCppLaunchPolicy,
    LlamaCppManagedRuntime,
    LlamaCppProvider,
    LlamaCppProviderConfig,
    LlamaCppSplitMode,
    LlamaCppSubprocessController,
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
        worker_id="llama-single",
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
        worker_id="llama-multi",
        worker_class="multi-gpu",
        resources=ResourceShape(
            gpu_count=2,
            total_vram_mb=36864,
            max_single_gpu_vram_mb=24576,
        ),
        gpu_uuids=("GPU-example-large", "GPU-example-small"),
        capabilities=frozenset({"llm.chat", "text.generate"}),
    )


def cpu_worker() -> WorkerSpec:
    return WorkerSpec(
        worker_id="llama-cpu",
        worker_class="cpu",
        resources=ResourceShape(),
        gpu_uuids=(),
        capabilities=frozenset({"llm.chat", "text.generate"}),
    )


def demand(
    *,
    model_format: str = "gguf",
    topology: GPUTopology = GPUTopology.SINGLE_GPU,
    model_topology: ModelTopology = ModelTopology.DENSE,
    residency: ResidencyPolicy = ResidencyPolicy.PREFER_VRAM,
    estimated_size_mb: int | None = 18000,
    min_host_ram_mb: int = 0,
) -> ExecutionDemand:
    return ExecutionDemand(
        model=ModelDemand(
            model_ref="/models/qwen.gguf",
            model_format=model_format,
            topology=model_topology,
            estimated_size_mb=estimated_size_mb,
        ),
        residency_policy=residency,
        gpu_topology=topology,
        min_gpu_count=(
            0
            if topology is GPUTopology.NONE
            else 2
            if topology is GPUTopology.MULTI_GPU
            else 0
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


def test_llama_cpp_provider_rejects_non_gguf() -> None:
    report = LlamaCppProvider().compatibility(
        context(execution=demand(model_format="safetensors"))
    )

    assert not report.compatible
    assert "model-format-unsupported" in {
        reason.code for reason in report.reasons
    }


def test_llama_cpp_explicit_selection_is_preserved() -> None:
    ctx = context()

    resolution = resolve_runtime(
        catalog=RuntimeCatalog([LlamaCppProvider()]),
        context=ctx,
        selection=RuntimeSelection(
            mode=RuntimeSelectionMode.EXPLICIT,
            provider_id="llama-cpp",
        ),
    )

    assert resolution.selected_provider_id == "llama-cpp"


def test_vram_only_requires_size_and_forces_all_layers_without_fit() -> None:
    provider = LlamaCppProvider()

    missing = provider.compatibility(
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

    ctx = context(
        execution=demand(
            residency=ResidencyPolicy.VRAM_ONLY,
            estimated_size_mb=18000,
        )
    )
    intent = provider.setup_intent(ctx)

    assert intent.configuration["gpu_layers"] == "all"
    assert intent.configuration["fit"] is False
    assert intent.configuration["split_mode"] == "none"


def test_vram_only_rejects_model_larger_than_single_gpu() -> None:
    report = LlamaCppProvider().compatibility(
        context(
            worker=single_worker(vram_mb=12288),
            execution=demand(
                residency=ResidencyPolicy.VRAM_ONLY,
                estimated_size_mb=18000,
            ),
        )
    )

    assert not report.compatible
    assert "model-exceeds-single-gpu-vram" in {
        reason.code for reason in report.reasons
    }


def test_vram_only_rejects_cpu_offload_override() -> None:
    provider = LlamaCppProvider(
        LlamaCppProviderConfig(n_cpu_ffn=4)
    )

    report = provider.compatibility(
        context(
            execution=demand(
                residency=ResidencyPolicy.VRAM_ONLY,
            )
        )
    )

    assert not report.compatible
    assert "cpu-offload-conflicts-with-vram-only" in {
        reason.code for reason in report.reasons
    }


def test_prefer_vram_uses_auto_layers_and_fit() -> None:
    intent = LlamaCppProvider().setup_intent(
        context(
            execution=demand(
                residency=ResidencyPolicy.PREFER_VRAM,
            )
        )
    )

    assert intent.configuration["gpu_layers"] == "auto"
    assert intent.configuration["fit"] is True


def test_cpu_gpu_hybrid_uses_auto_fit_and_advisory() -> None:
    ctx = context(
        worker=single_worker(vram_mb=8192),
        execution=demand(
            residency=ResidencyPolicy.CPU_GPU_HYBRID,
            estimated_size_mb=32000,
            min_host_ram_mb=32768,
        ),
    )
    provider = LlamaCppProvider()

    report = provider.compatibility(ctx)

    assert report.compatible
    reason = next(
        reason
        for reason in report.reasons
        if reason.code == "hybrid-offload-auto"
    )
    assert not reason.blocking

    intent = provider.setup_intent(ctx)
    assert intent.configuration["gpu_layers"] == "auto"
    assert intent.configuration["fit"] is True


def test_cpu_gpu_hybrid_rejects_model_beyond_combined_memory() -> None:
    ctx = context(
        worker=single_worker(vram_mb=8192),
        host_facts=host(ram_mb=16384),
        execution=demand(
            residency=ResidencyPolicy.CPU_GPU_HYBRID,
            estimated_size_mb=30000,
        ),
    )

    report = LlamaCppProvider().compatibility(ctx)

    assert not report.compatible
    assert "model-exceeds-combined-memory" in {
        reason.code for reason in report.reasons
    }


def test_dense_model_accepts_explicit_cpu_ffn_offload() -> None:
    provider = LlamaCppProvider(
        LlamaCppProviderConfig(n_cpu_ffn=8)
    )
    ctx = context(
        execution=demand(
            residency=ResidencyPolicy.CPU_GPU_HYBRID,
            model_topology=ModelTopology.DENSE,
        )
    )

    report = provider.compatibility(ctx)
    assert report.compatible
    assert provider.setup_intent(ctx).configuration["n_cpu_ffn"] == 8


def test_dense_model_rejects_moe_only_offload_options() -> None:
    provider = LlamaCppProvider(
        LlamaCppProviderConfig(cpu_moe=True)
    )

    report = provider.compatibility(
        context(
            execution=demand(
                residency=ResidencyPolicy.CPU_GPU_HYBRID,
                model_topology=ModelTopology.DENSE,
            )
        )
    )

    assert not report.compatible
    assert "moe-offload-on-dense-model" in {
        reason.code for reason in report.reasons
    }


def test_moe_model_accepts_cpu_moe_offload() -> None:
    provider = LlamaCppProvider(
        LlamaCppProviderConfig(cpu_moe=True)
    )
    ctx = context(
        execution=demand(
            residency=ResidencyPolicy.CPU_GPU_HYBRID,
            model_topology=ModelTopology.MOE,
        )
    )

    report = provider.compatibility(ctx)
    assert report.compatible
    assert provider.setup_intent(ctx).configuration["cpu_moe"] is True


def test_heterogeneous_multi_gpu_defaults_to_layer_auto_fit() -> None:
    ctx = context(
        worker=multi_worker(),
        execution=demand(
            topology=GPUTopology.MULTI_GPU,
            estimated_size_mb=30000,
        ),
    )
    provider = LlamaCppProvider()

    report = provider.compatibility(ctx)

    assert report.compatible
    assert "heterogeneous-auto-fit" in {
        reason.code for reason in report.reasons
    }

    intent = provider.setup_intent(ctx)
    assert intent.configuration["split_mode"] == "layer"
    assert intent.configuration["tensor_split"] is None
    assert intent.configuration["fit"] is True
    assert intent.configuration["gpu_uuids"] == [
        "GPU-example-large",
        "GPU-example-small",
    ]


def test_explicit_tensor_split_must_match_gpu_count() -> None:
    provider = LlamaCppProvider(
        LlamaCppProviderConfig(tensor_split=(2.0, 1.0, 1.0))
    )

    report = provider.compatibility(
        context(
            worker=multi_worker(),
            execution=demand(topology=GPUTopology.MULTI_GPU),
        )
    )

    assert not report.compatible
    assert "tensor-split-length-mismatch" in {
        reason.code for reason in report.reasons
    }


def test_explicit_tensor_split_is_preserved() -> None:
    provider = LlamaCppProvider(
        LlamaCppProviderConfig(tensor_split=(2.0, 1.0))
    )
    ctx = context(
        worker=multi_worker(),
        execution=demand(topology=GPUTopology.MULTI_GPU),
    )

    intent = provider.setup_intent(ctx)

    assert intent.configuration["tensor_split"] == [2.0, 1.0]


def test_tensor_parallel_mode_is_supported_as_experimental_advisory() -> None:
    provider = LlamaCppProvider(
        LlamaCppProviderConfig(
            split_mode=LlamaCppSplitMode.TENSOR,
        )
    )

    report = provider.compatibility(
        context(
            worker=multi_worker(),
            execution=demand(topology=GPUTopology.MULTI_GPU),
        )
    )

    assert report.compatible
    reason = next(
        reason
        for reason in report.reasons
        if reason.code == "tensor-split-experimental"
    )
    assert not reason.blocking


def test_single_gpu_rejects_multi_gpu_split_override() -> None:
    provider = LlamaCppProvider(
        LlamaCppProviderConfig(
            split_mode=LlamaCppSplitMode.LAYER,
        )
    )

    report = provider.compatibility(context())

    assert not report.compatible
    assert "single-gpu-split-mode-invalid" in {
        reason.code for reason in report.reasons
    }


def test_cpu_only_path_disables_gpu_layers() -> None:
    ctx = context(
        worker=cpu_worker(),
        execution=demand(
            topology=GPUTopology.NONE,
            residency=ResidencyPolicy.CPU_GPU_HYBRID,
            estimated_size_mb=12000,
        ),
    )

    intent = LlamaCppProvider().setup_intent(ctx)

    assert intent.configuration["gpu_layers"] == 0
    assert intent.configuration["split_mode"] == "none"
    assert intent.configuration["gpu_uuids"] == []


def test_setup_intent_uses_reference_only_gguf_and_builds_setup_plan() -> None:
    ctx = context()
    provider = LlamaCppProvider()

    intent = provider.setup_intent(ctx)

    assert intent.package_references == ("llama-cpp",)
    assert intent.model_preparation.value == "reference_only"
    assert intent.model_ref == "/models/qwen.gguf"
    assert intent.requires_privilege

    plan = build_runtime_setup_plan(
        catalog=RuntimeCatalog([provider]),
        context=ctx,
        selection=RuntimeSelection(
            mode=RuntimeSelectionMode.EXPLICIT,
            provider_id="llama-cpp",
        ),
        snapshot=snapshot(ctx),
    )

    assert plan.provider_id == "llama-cpp"
    assert any(
        action.kind is SetupActionKind.ENSURE_PACKAGE
        and action.payload["package_reference"] == "llama-cpp"
        for action in plan.actions
    )
    assert any(
        action.kind is SetupActionKind.VERIFY_MODEL_REFERENCE
        and action.payload["model_ref"] == "/models/qwen.gguf"
        for action in plan.actions
    )
    assert not any(
        action.kind is SetupActionKind.DOWNLOAD_MODEL
        for action in plan.actions
    )


def test_single_gpu_command_pins_uuid_and_uses_no_split() -> None:
    policy = LlamaCppLaunchPolicy(
        gpu_layers="all",
        split_mode=LlamaCppSplitMode.NONE,
        fit=False,
        tensor_split=None,
        fit_target_mb=None,
        main_gpu=0,
        cpu_moe=False,
        n_cpu_moe=None,
        n_cpu_ffn=None,
    )
    controller = LlamaCppSubprocessController(
        executable="llama-server",
        base_url="http://127.0.0.1:8080",
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        gpu_uuids=("GPU-one",),
        context_size=4096,
        launch_policy=policy,
        offline=True,
        no_webui=True,
    )

    command = controller.command()

    assert command[:1] == ("llama-server",)
    assert "--model" in command
    assert command[command.index("--n-gpu-layers") + 1] == "all"
    assert command[command.index("--fit") + 1] == "off"
    assert command[command.index("--device") + 1] == "CUDA0"
    assert command[command.index("--split-mode") + 1] == "none"
    assert command[command.index("--main-gpu") + 1] == "0"
    assert "--offline" in command
    assert "--no-webui" in command


def test_multi_gpu_command_uses_visible_devices_and_tensor_split() -> None:
    policy = LlamaCppLaunchPolicy(
        gpu_layers="auto",
        split_mode=LlamaCppSplitMode.LAYER,
        fit=True,
        tensor_split=(2.0, 1.0),
        fit_target_mb=(2048, 1024),
        main_gpu=0,
        cpu_moe=False,
        n_cpu_moe=None,
        n_cpu_ffn=6,
    )
    controller = LlamaCppSubprocessController(
        executable="llama-server",
        base_url="http://127.0.0.1:8080",
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        gpu_uuids=("GPU-large", "GPU-small"),
        context_size=8192,
        launch_policy=policy,
        offline=True,
        no_webui=True,
    )

    command = controller.command()

    assert command[command.index("--device") + 1] == "CUDA0,CUDA1"
    assert command[command.index("--split-mode") + 1] == "layer"
    assert command[command.index("--tensor-split") + 1] == "2.0,1.0"
    assert command[command.index("--fit-target") + 1] == "2048,1024"
    assert command[command.index("--n-cpu-ffn") + 1] == "6"


@pytest.mark.asyncio
async def test_subprocess_environment_pins_gpu_uuid_set(monkeypatch) -> None:
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
        captured["kwargs"] = dict(kwargs)
        return Process()

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    controller = LlamaCppSubprocessController(
        executable="llama-server",
        base_url="http://127.0.0.1:8080",
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        gpu_uuids=("GPU-large", "GPU-small"),
        context_size=4096,
        launch_policy=LlamaCppLaunchPolicy(
            gpu_layers="auto",
            split_mode=LlamaCppSplitMode.LAYER,
            fit=True,
            tensor_split=None,
            fit_target_mb=None,
            main_gpu=0,
            cpu_moe=False,
            n_cpu_moe=None,
            n_cpu_ffn=None,
        ),
        offline=True,
        no_webui=True,
    )

    await controller.start()

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-large,GPU-small"
    assert "--device" in captured["args"]
    assert "stdout" not in captured["kwargs"]
    assert "stderr" not in captured["kwargs"]

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
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "astrumweaver", "object": "model"}],
                },
            )
        if request.url.path == "/v1/chat/completions/input_tokens":
            return httpx.Response(
                200,
                json={"object": "response.input_tokens", "input_tokens": 3},
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

    api = HttpLlamaCppApi(
        "http://127.0.0.1:8080",
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
        assert await api.chat_input_tokens(
            {
                "model": "astrumweaver",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }
        ) == 3
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
        "/v1/chat/completions/input_tokens",
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
        self.chat_token_payloads: list[dict] = []
        self.chat_input_token_count = 3
        self.completion_payloads: list[dict] = []

    async def health(self) -> bool:
        return self.reachable

    async def models(self) -> tuple[str, ...]:
        if not self.reachable:
            raise httpx.ConnectError("offline")
        return self.model_ids

    async def chat_input_tokens(self, payload: Mapping[str, object]) -> int:
        self.chat_token_payloads.append(dict(payload))
        return self.chat_input_token_count

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
    def __init__(
        self,
        api: FakeApi,
        *,
        running: bool = False,
        exit_on_start: bool = False,
    ) -> None:
        self.api = api
        self._running = running
        self.exit_on_start = exit_on_start
        self.starts = 0
        self.stops = 0

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        self.starts += 1
        if self.exit_on_start:
            self._running = False
            self.api.reachable = False
            return
        self._running = True
        self.api.reachable = True

    async def stop(self) -> None:
        self.stops += 1
        self._running = False
        self.api.reachable = False


@pytest.mark.asyncio
async def test_executor_chat_completion_metrics_and_model_binding() -> None:
    api = FakeApi()
    executor = LlamaCppExecutor(
        api=api,
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        residency_metadata={"split_mode": "none"},
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
    executor = LlamaCppExecutor(
        api=api,
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
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
    executor = LlamaCppExecutor(
        api=FakeApi(),
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        residency_metadata={
            "residency_policy": "cpu_gpu_hybrid",
            "split_mode": "layer",
            "gpu_count": 2,
        },
    )

    report = await executor.residency()

    assert len(report.items) == 1
    assert report.items[0].accelerator_memory_bytes is None
    assert report.items[0].metadata["split_mode"] == "layer"
    assert report.items[0].metadata["gpu_count"] == 2


@pytest.mark.asyncio
async def test_managed_runtime_starts_checks_alias_and_stops() -> None:
    api = FakeApi(reachable=False)
    process = FakeProcess(api)
    runtime = LlamaCppManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        startup_timeout_seconds=1.0,
        launch_policy=LlamaCppLaunchPolicy(
            gpu_layers="auto",
            split_mode=LlamaCppSplitMode.NONE,
            fit=True,
            tensor_split=None,
            fit_target_mb=None,
            main_gpu=0,
            cpu_moe=False,
            n_cpu_moe=None,
            n_cpu_ffn=None,
        ),
    )

    await runtime.start()

    assert process.starts == 1
    assert (await runtime.health()).ready
    await runtime.stop()
    assert process.stops == 1


@pytest.mark.asyncio
async def test_managed_runtime_fails_fast_when_owned_process_exits_before_readiness() -> None:
    api = FakeApi(reachable=False)
    process = FakeProcess(api, exit_on_start=True)
    runtime = LlamaCppManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        startup_timeout_seconds=60.0,
        launch_policy=LlamaCppLaunchPolicy(
            gpu_layers="auto",
            split_mode=LlamaCppSplitMode.NONE,
            fit=True,
            tensor_split=None,
            fit_target_mb=None,
            main_gpu=0,
            cpu_moe=False,
            n_cpu_moe=None,
            n_cpu_ffn=None,
        ),
    )

    with pytest.raises(RuntimeError, match="owned process exited before readiness"):
        await runtime.start()

    assert process.starts == 1
    assert process.stops == 1


@pytest.mark.asyncio
async def test_managed_runtime_rejects_external_server() -> None:
    api = FakeApi(reachable=True)
    process = FakeProcess(api, running=False)
    runtime = LlamaCppManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        startup_timeout_seconds=1.0,
        launch_policy=LlamaCppLaunchPolicy(
            gpu_layers="auto",
            split_mode=LlamaCppSplitMode.NONE,
            fit=True,
            tensor_split=None,
            fit_target_mb=None,
            main_gpu=0,
            cpu_moe=False,
            n_cpu_moe=None,
            n_cpu_ffn=None,
        ),
    )

    health = await runtime.health()
    assert health.state is RuntimeHealthState.FAILED

    with pytest.raises(RuntimeError, match="external llama.cpp"):
        await runtime.start()


@pytest.mark.asyncio
async def test_managed_runtime_cleans_up_failed_start() -> None:
    api = FakeApi(reachable=False, models=("wrong-alias",))
    process = FakeProcess(api)
    runtime = LlamaCppManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        startup_timeout_seconds=1.0,
        launch_policy=LlamaCppLaunchPolicy(
            gpu_layers="all",
            split_mode=LlamaCppSplitMode.NONE,
            fit=False,
            tensor_split=None,
            fit_target_mb=None,
            main_gpu=0,
            cpu_moe=False,
            n_cpu_moe=None,
            n_cpu_ffn=None,
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
    runtime = LlamaCppManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        startup_timeout_seconds=1.0,
        launch_policy=LlamaCppLaunchPolicy(
            gpu_layers="auto",
            split_mode=LlamaCppSplitMode.NONE,
            fit=True,
            tensor_split=None,
            fit_target_mb=None,
            main_gpu=0,
            cpu_moe=False,
            n_cpu_moe=None,
            n_cpu_ffn=None,
        ),
    )

    await runtime.release()
    await runtime.release()

    assert api.closed



def test_single_gpu_rejects_tensor_split_override() -> None:
    provider = LlamaCppProvider(
        LlamaCppProviderConfig(tensor_split=(1.0,))
    )

    report = provider.compatibility(context())

    assert not report.compatible
    assert "tensor-split-requires-multi-gpu" in {
        reason.code for reason in report.reasons
    }


def test_main_gpu_must_be_inside_visible_set() -> None:
    provider = LlamaCppProvider(
        LlamaCppProviderConfig(main_gpu=1)
    )

    report = provider.compatibility(context())

    assert not report.compatible
    assert "main-gpu-outside-visible-set" in {
        reason.code for reason in report.reasons
    }


@pytest.mark.asyncio
async def test_stage_a_gateway_chat_counts_actual_template_before_inference() -> None:
    api = FakeApi()
    api.chat_input_token_count = 40
    executor = LlamaCppExecutor(
        api=api,
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        residency_metadata={},
        tools_enabled=True,
    )
    arguments = '{"path":"README.md"}'
    result = await executor.execute(
        JobRequest(
            job_id="gateway-chat",
            capability="llm.chat",
            payload={
                "schema_version": "chat-job-v1",
                "adapter_id": "llama-cpp-chat-v1",
                "request": {
                    "messages": [
                        {"role": "user", "content": "inspect"},
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    "tool_choice": "auto",
                    "max_tokens": 20,
                    "stream": False,
                },
                "limits": {
                    "input_tokens": 64,
                    "output_tokens": 32,
                    "total_tokens": 84,
                },
            },
        )
    )

    assert result.text == "hello"
    assert len(api.chat_token_payloads) == 1
    counted = api.chat_token_payloads[0]
    assert counted["model"] == "astrumweaver"
    assert counted["tools"][0]["function"]["name"] == "read_file"
    assert counted["stream"] is False
    assert api.chat_payloads == [counted]


@pytest.mark.asyncio
async def test_stage_a_gateway_chat_context_overflow_is_non_retryable() -> None:
    api = FakeApi()
    api.chat_input_token_count = 65
    executor = LlamaCppExecutor(
        api=api,
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        residency_metadata={},
    )

    with pytest.raises(JobExecutionError) as error:
        await executor.execute(
            JobRequest(
                job_id="overflow",
                capability="llm.chat",
                payload={
                    "schema_version": "chat-job-v1",
                    "adapter_id": "llama-cpp-chat-v1",
                    "request": {
                        "messages": [{"role": "user", "content": "large"}],
                        "max_tokens": 8,
                        "stream": False,
                    },
                    "limits": {
                        "input_tokens": 64,
                        "output_tokens": 16,
                        "total_tokens": 80,
                    },
                },
            )
        )

    assert error.value.code == "context_length_exceeded"
    assert not error.value.retryable
    assert api.chat_payloads == []


@pytest.mark.asyncio
async def test_stage_a_gateway_chat_total_context_reserves_output_tokens() -> None:
    api = FakeApi()
    api.chat_input_token_count = 60
    executor = LlamaCppExecutor(
        api=api,
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        residency_metadata={},
    )

    with pytest.raises(JobExecutionError) as error:
        await executor.execute(
            JobRequest(
                job_id="reservation-overflow",
                capability="llm.chat",
                payload={
                    "schema_version": "chat-job-v1",
                    "adapter_id": "llama-cpp-chat-v1",
                    "request": {
                        "messages": [{"role": "user", "content": "large"}],
                        "max_tokens": 21,
                        "stream": False,
                    },
                    "limits": {
                        "input_tokens": 64,
                        "output_tokens": 32,
                        "total_tokens": 80,
                    },
                },
            )
        )

    assert error.value.code == "context_length_exceeded"
    assert not error.value.retryable
    assert api.chat_payloads == []


@pytest.mark.asyncio
async def test_stage_a_gateway_chat_rejects_wrong_adapter_before_provider_call() -> None:
    api = FakeApi()
    executor = LlamaCppExecutor(
        api=api,
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        residency_metadata={},
    )

    with pytest.raises(JobExecutionError) as error:
        await executor.execute(
            JobRequest(
                job_id="wrong-adapter",
                capability="llm.chat",
                payload={
                    "schema_version": "chat-job-v1",
                    "adapter_id": "other-chat-v1",
                    "request": {
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8,
                        "stream": False,
                    },
                    "limits": {
                        "input_tokens": 64,
                        "output_tokens": 16,
                        "total_tokens": 80,
                    },
                },
            )
        )

    assert error.value.code == "unsupported_chat_adapter"
    assert not error.value.retryable
    assert api.chat_token_payloads == []
    assert api.chat_payloads == []



def test_llama_cpp_jinja_tool_mode_threads_into_setup_and_command() -> None:
    provider = LlamaCppProvider(
        LlamaCppProviderConfig(
            jinja=True,
            chat_template_file="/templates/tool-use.jinja",
        )
    )
    intent = provider.setup_intent(context())
    assert intent.configuration["jinja"] is True
    assert (
        intent.configuration["chat_template_file"]
        == "/templates/tool-use.jinja"
    )

    controller = LlamaCppSubprocessController(
        executable="llama-server",
        base_url="http://127.0.0.1:8080",
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        gpu_uuids=("GPU-one",),
        context_size=4096,
        launch_policy=LlamaCppLaunchPolicy(
            gpu_layers="all",
            split_mode=LlamaCppSplitMode.NONE,
            fit=False,
            tensor_split=None,
            fit_target_mb=None,
            main_gpu=0,
            cpu_moe=False,
            n_cpu_moe=None,
            n_cpu_ffn=None,
        ),
        offline=True,
        no_webui=True,
        jinja=True,
        chat_template_file="/templates/tool-use.jinja",
    )
    command = controller.command()
    assert "--jinja" in command
    assert command[command.index("--chat-template-file") + 1] == (
        "/templates/tool-use.jinja"
    )


def test_llama_cpp_template_file_requires_jinja_mode() -> None:
    with pytest.raises(ValueError, match="requires jinja"):
        LlamaCppProviderConfig(
            jinja=False,
            chat_template_file="/templates/tool-use.jinja",
        )


@pytest.mark.asyncio
async def test_stage_a_tool_request_requires_jinja_runtime_evidence() -> None:
    api = FakeApi()
    executor = LlamaCppExecutor(
        api=api,
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        residency_metadata={},
        tools_enabled=False,
    )

    with pytest.raises(JobExecutionError) as error:
        await executor.execute(
            JobRequest(
                job_id="tool-without-jinja",
                capability="llm.chat",
                payload={
                    "schema_version": "chat-job-v1",
                    "adapter_id": "llama-cpp-chat-v1",
                    "request": {
                        "messages": [{"role": "user", "content": "inspect"}],
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "parameters": {"type": "object"},
                                },
                            }
                        ],
                        "tool_choice": "auto",
                        "max_tokens": 8,
                        "stream": False,
                    },
                    "limits": {
                        "input_tokens": 64,
                        "output_tokens": 16,
                        "total_tokens": 80,
                    },
                },
            )
        )

    assert error.value.code == "unsupported_feature"
    assert not error.value.retryable
    assert api.chat_token_payloads == []
    assert api.chat_payloads == []


def test_llama_cpp_executor_advertises_tools_only_with_jinja() -> None:
    api = FakeApi()
    plain = LlamaCppExecutor(
        api=api,
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        residency_metadata={},
        tools_enabled=False,
    )
    tools = LlamaCppExecutor(
        api=api,
        model_ref="/models/qwen.gguf",
        model_alias="astrumweaver",
        residency_metadata={},
        tools_enabled=True,
    )

    assert plain.serving_features["llm.chat"] == frozenset()
    assert tools.serving_features["llm.chat"] == frozenset({"tools"})
