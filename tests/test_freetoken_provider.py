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
    RuntimeSelectionError,
    RuntimeSelectionMode,
    resolve_runtime,
)
from astrumweaver.runtime.providers.freetoken import (
    FreeTokenExecutor,
    FreeTokenLaunchPolicy,
    FreeTokenManagedRuntime,
    FreeTokenMoeStrategy,
    FreeTokenProvider,
    FreeTokenProviderConfig,
    FreeTokenSubprocessController,
    HttpFreeTokenApi,
)
from astrumweaver.setup import (
    DeploymentPath,
    PrivilegeMode,
    SetupActionKind,
    SetupHostSnapshot,
    build_runtime_setup_plan,
)


def host(
    ram_mb: int = 131072,
    architecture: str = "x86_64",
) -> RuntimeHostFacts:
    return RuntimeHostFacts(
        cpu_count=32,
        host_ram_mb=ram_mb,
        architecture=architecture,
    )


def single_worker(vram_mb: int = 12288) -> WorkerSpec:
    return WorkerSpec(
        worker_id="freetoken-single",
        worker_class="legacy-single",
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
        worker_id="freetoken-multi",
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
    model_ref: str = "Qwen/Qwen3-30B-A3B",
    model_format: str = "huggingface",
    model_topology: ModelTopology = ModelTopology.MOE,
    gpu_topology: GPUTopology = GPUTopology.SINGLE_GPU,
    residency: ResidencyPolicy = ResidencyPolicy.CPU_GPU_HYBRID,
    estimated_size_mb: int | None = 60000,
    min_host_ram_mb: int = 0,
    preferred_host_ram_mb: int = 0,
) -> ExecutionDemand:
    return ExecutionDemand(
        model=ModelDemand(
            model_ref=model_ref,
            model_format=model_format,
            topology=model_topology,
            estimated_size_mb=estimated_size_mb,
        ),
        residency_policy=residency,
        gpu_topology=gpu_topology,
        min_gpu_count=2 if gpu_topology is GPUTopology.MULTI_GPU else 0,
        min_host_ram_mb=min_host_ram_mb,
        preferred_host_ram_mb=max(
            preferred_host_ram_mb,
            min_host_ram_mb,
        ),
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


def test_moe_single_gpu_hybrid_is_compatible() -> None:
    report = FreeTokenProvider().compatibility(context())

    assert report.compatible
    assert "cpu-gpu-hybrid-supported" in {
        reason.code for reason in report.reasons
    }
    assert "host-ram-interconnect-sensitive" in {
        reason.code for reason in report.reasons
    }


def test_dense_is_rejected_as_astrumweaver_scope_not_upstream_limit() -> None:
    report = FreeTokenProvider().compatibility(
        context(
            execution=demand(model_topology=ModelTopology.DENSE)
        )
    )

    assert not report.compatible
    reason = next(
        reason
        for reason in report.reasons
        if reason.code == "model-topology-outside-provider-scope"
    )
    assert "AstrumWeaver v0.x" in reason.message
    assert "upstream FreeToken" in reason.message


def test_gguf_is_rejected() -> None:
    report = FreeTokenProvider().compatibility(
        context(execution=demand(model_format="gguf"))
    )

    assert not report.compatible
    assert "model-format-unsupported" in {
        reason.code for reason in report.reasons
    }


def test_ftw_requires_local_reference() -> None:
    report = FreeTokenProvider().compatibility(
        context(
            execution=demand(
                model_ref="org/checkpoint.ftw",
                model_format="ftw",
            )
        )
    )

    assert not report.compatible
    assert "ftw-reference-must-be-local" in {
        reason.code for reason in report.reasons
    }


def test_multi_gpu_worker_is_rejected_instead_of_partial_use() -> None:
    report = FreeTokenProvider().compatibility(
        context(
            worker=multi_worker(),
            execution=demand(gpu_topology=GPUTopology.MULTI_GPU),
        )
    )

    assert not report.compatible
    assert "single-gpu-required" in {
        reason.code for reason in report.reasons
    }


def test_vram_only_is_explicitly_outside_v0_provider_scope() -> None:
    report = FreeTokenProvider().compatibility(
        context(execution=demand(residency=ResidencyPolicy.VRAM_ONLY))
    )

    assert not report.compatible
    assert "vram-only-unsupported" in {
        reason.code for reason in report.reasons
    }


def test_current_provider_rejects_non_x86_64_host() -> None:
    report = FreeTokenProvider().compatibility(
        context(host_facts=host(architecture="aarch64"))
    )

    assert not report.compatible
    assert "architecture-unsupported" in {
        reason.code for reason in report.reasons
    }


def test_explicit_selection_is_preserved_and_setup_plan_is_deterministic() -> None:
    ctx = context()
    provider = FreeTokenProvider()
    selection = RuntimeSelection(
        mode=RuntimeSelectionMode.EXPLICIT,
        provider_id="freetoken",
    )

    resolution = resolve_runtime(
        catalog=RuntimeCatalog([provider]),
        context=ctx,
        selection=selection,
    )
    assert resolution.selected_provider_id == "freetoken"

    intent = provider.setup_intent(ctx)
    assert intent.package_references == ("freetoken[accel]",)
    assert intent.configuration["gpu_uuid"] == "GPU-example-one"
    assert intent.configuration["moe_strategy"] == "auto"
    assert intent.configuration["memory_ratio"] == 0.9
    assert intent.model_preparation.value == "download"

    plan = build_runtime_setup_plan(
        catalog=RuntimeCatalog([provider]),
        context=ctx,
        selection=selection,
        snapshot=snapshot(ctx),
    )

    assert plan.provider_id == "freetoken"
    assert any(
        action.kind is SetupActionKind.ENSURE_PACKAGE
        and action.payload["package_reference"] == "freetoken[accel]"
        for action in plan.actions
    )
    assert any(
        action.kind is SetupActionKind.DOWNLOAD_MODEL
        and action.payload["model_ref"] == "Qwen/Qwen3-30B-A3B"
        for action in plan.actions
    )


def test_local_model_reference_uses_reference_verification() -> None:
    ctx = context(
        execution=demand(
            model_ref="/models/qwen-ftw",
            model_format="ftw",
        )
    )
    provider = FreeTokenProvider()
    intent = provider.setup_intent(ctx)

    assert intent.model_preparation.value == "reference_only"

    plan = build_runtime_setup_plan(
        catalog=RuntimeCatalog([provider]),
        context=ctx,
        selection=RuntimeSelection(
            mode=RuntimeSelectionMode.EXPLICIT,
            provider_id="freetoken",
        ),
        snapshot=snapshot(ctx),
    )
    assert any(
        action.kind is SetupActionKind.VERIFY_MODEL_REFERENCE
        and action.payload["model_ref"] == "/models/qwen-ftw"
        for action in plan.actions
    )


def test_generic_host_ram_requirement_remains_blocking() -> None:
    ctx = context(
        host_facts=host(ram_mb=32768),
        execution=demand(
            min_host_ram_mb=65536,
            preferred_host_ram_mb=98304,
        ),
    )

    with pytest.raises(RuntimeSelectionError) as exc_info:
        resolve_runtime(
            catalog=RuntimeCatalog([FreeTokenProvider()]),
            context=ctx,
            selection=RuntimeSelection(
                mode=RuntimeSelectionMode.EXPLICIT,
                provider_id="freetoken",
            ),
        )

    assert "insufficient-host-ram" in {
        reason.code for reason in exc_info.value.reasons
    }


def test_provider_does_not_invent_model_size_to_ram_formula() -> None:
    report = FreeTokenProvider().compatibility(
        context(
            host_facts=host(ram_mb=16384),
            execution=demand(
                estimated_size_mb=120000,
                min_host_ram_mb=0,
            ),
        )
    )

    assert report.compatible
    assert not any(
        reason.code == "model-exceeds-host-ram"
        for reason in report.reasons
    )


def test_incompatible_context_cannot_create_setup_or_runtime() -> None:
    provider = FreeTokenProvider()
    ctx = context(
        execution=demand(model_topology=ModelTopology.DENSE)
    )

    with pytest.raises(RuntimeError, match="incompatible"):
        provider.setup_intent(ctx)

    compatible_intent = FreeTokenProvider().setup_intent(context())
    with pytest.raises(RuntimeError, match="incompatible"):
        provider.create_runtime(ctx, compatible_intent)


def test_create_runtime_rejects_tampered_gpu_uuid() -> None:
    provider = FreeTokenProvider()
    ctx = context()
    intent = provider.setup_intent(ctx)
    tampered = replace(
        intent,
        configuration={
            **dict(intent.configuration),
            "gpu_uuid": "GPU-not-owned-by-worker",
        },
    )

    with pytest.raises(ValueError, match="GPU UUID"):
        provider.create_runtime(ctx, tampered)


def test_single_gpu_command_uses_exact_worker_uuid_and_free_token_flags() -> None:
    controller = FreeTokenSubprocessController(
        executable="ft",
        base_url="http://127.0.0.1:1919",
        model_ref="Qwen/Qwen3-30B-A3B",
        served_model_name="astrumweaver",
        launch_policy=FreeTokenLaunchPolicy(
            gpu_uuid="GPU-exact-uuid",
            moe_strategy=FreeTokenMoeStrategy.HYBRID,
            memory_ratio=0.85,
            moe_cache_size=96,
            moe_cache_rate=None,
            moe_cpu_threads=16,
            moe_cpu_layers="0.5",
            moe_hybrid_max_fetch=4,
            max_running_requests=1,
        ),
    )

    command = controller.command()

    assert command[:3] == ("ft", "serve", "--model")
    assert command[command.index("--gpu") + 1] == "GPU-exact-uuid"
    assert command[command.index("--moe-strategy") + 1] == "hybrid"
    assert command[command.index("--memory-ratio") + 1] == "0.85"
    assert command[command.index("--moe-cache-size") + 1] == "96"
    assert command[command.index("--moe-cpu-threads") + 1] == "16"
    assert command[command.index("--moe-cpu-layers") + 1] == "0.5"
    assert command[command.index("--moe-hybrid-max-fetch") + 1] == "4"


@pytest.mark.asyncio
async def test_subprocess_does_not_rewrite_gpu_identity(monkeypatch) -> None:
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

    controller = FreeTokenSubprocessController(
        executable="ft",
        base_url="http://127.0.0.1:1919",
        model_ref="/models/qwen",
        served_model_name="astrumweaver",
        launch_policy=FreeTokenLaunchPolicy(
            gpu_uuid="GPU-exact-uuid",
            moe_strategy=FreeTokenMoeStrategy.AUTO,
            memory_ratio=0.9,
            moe_cache_size=None,
            moe_cache_rate=None,
            moe_cpu_threads=None,
            moe_cpu_layers=None,
            moe_hybrid_max_fetch=None,
            max_running_requests=1,
        ),
    )

    await controller.start()

    assert captured["args"][
        captured["args"].index("--gpu") + 1
    ] == "GPU-exact-uuid"
    assert "env" not in captured["kwargs"]

    await controller.stop()


@pytest.mark.asyncio
async def test_http_api_uses_current_freetoken_control_and_openai_endpoints() -> None:
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
        if request.url.path == "/v1/stats":
            return httpx.Response(
                200,
                json={
                    "model": {"id": "astrumweaver", "moe": True},
                    "vram_bytes": 123456,
                },
            )
        return httpx.Response(404)

    api = HttpFreeTokenApi(
        "http://127.0.0.1:1919",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert await api.health()
        assert await api.models() == ("astrumweaver",)
        await api.chat(
            {
                "model": "astrumweaver",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }
        )
        await api.completion(
            {
                "model": "astrumweaver",
                "prompt": "hi",
                "stream": False,
            }
        )
        assert (await api.stats())["vram_bytes"] == 123456
    finally:
        await api.close()

    assert [path for _, path, _ in requests] == [
        "/health",
        "/v1/models",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/stats",
    ]


@pytest.mark.asyncio
async def test_http_health_requires_ready_status_not_loading() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "loading", "model": "astrumweaver"},
        )

    api = HttpFreeTokenApi(
        "http://127.0.0.1:1919",
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
        models: tuple[str, ...] = ("astrumweaver",),
        stats: Mapping[str, object] | None = None,
    ) -> None:
        self.reachable = reachable
        self.model_ids = models
        self.stats_doc = dict(stats or {})
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

    async def stats(self):
        return dict(self.stats_doc)

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


def launch_policy() -> FreeTokenLaunchPolicy:
    return FreeTokenLaunchPolicy(
        gpu_uuid="GPU-example-one",
        moe_strategy=FreeTokenMoeStrategy.AUTO,
        memory_ratio=0.9,
        moe_cache_size=None,
        moe_cache_rate=None,
        moe_cpu_threads=None,
        moe_cpu_layers=None,
        moe_hybrid_max_fetch=None,
        max_running_requests=1,
    )


@pytest.mark.asyncio
async def test_executor_chat_completion_metrics_and_model_binding() -> None:
    api = FakeApi()
    executor = FreeTokenExecutor(
        api=api,
        model_ref="Qwen/Qwen3-30B-A3B",
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
    assert api.completion_payloads[0]["stream"] is False

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
    executor = FreeTokenExecutor(
        api=api,
        model_ref="Qwen/Qwen3-30B-A3B",
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
async def test_residency_does_not_invent_memory_bytes() -> None:
    executor = FreeTokenExecutor(
        api=FakeApi(stats={"model": {"id": "astrumweaver", "moe": True}}),
        model_ref="Qwen/Qwen3-30B-A3B",
        served_model_name="astrumweaver",
        residency_metadata={
            "moe_strategy": "auto",
            "host_ram_mb": 131072,
        },
    )

    report = await executor.residency()

    assert len(report.items) == 1
    assert report.items[0].accelerator_memory_bytes is None
    assert report.items[0].metadata["observed_moe"] is True
    assert report.items[0].metadata["host_ram_mb"] == 131072


@pytest.mark.asyncio
async def test_residency_treats_zero_vram_as_unobserved() -> None:
    executor = FreeTokenExecutor(
        api=FakeApi(stats={"vram_bytes": 0}),
        model_ref="Qwen/Qwen3-30B-A3B",
        served_model_name="astrumweaver",
        residency_metadata={},
    )

    report = await executor.residency()

    assert report.items[0].accelerator_memory_bytes is None


@pytest.mark.asyncio
async def test_residency_uses_measured_vram_when_stats_reports_it() -> None:
    executor = FreeTokenExecutor(
        api=FakeApi(
            stats={
                "model": {"id": "astrumweaver", "moe": True},
                "vram_bytes": 123456,
            }
        ),
        model_ref="Qwen/Qwen3-30B-A3B",
        served_model_name="astrumweaver",
        residency_metadata={},
    )

    report = await executor.residency()

    assert report.items[0].accelerator_memory_bytes == 123456


@pytest.mark.asyncio
async def test_managed_runtime_starts_checks_alias_and_stops() -> None:
    api = FakeApi(reachable=False)
    process = FakeProcess(api)
    runtime = FreeTokenManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model_ref="Qwen/Qwen3-30B-A3B",
        served_model_name="astrumweaver",
        startup_timeout_seconds=1.0,
        launch_policy=launch_policy(),
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
    runtime = FreeTokenManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model_ref="Qwen/Qwen3-30B-A3B",
        served_model_name="astrumweaver",
        startup_timeout_seconds=1.0,
        launch_policy=launch_policy(),
    )

    health = await runtime.health()
    assert health.state is RuntimeHealthState.FAILED

    with pytest.raises(RuntimeError, match="external FreeToken"):
        await runtime.start()


@pytest.mark.asyncio
async def test_managed_runtime_cleans_up_failed_start() -> None:
    api = FakeApi(reachable=False, models=("wrong-alias",))
    process = FakeProcess(api)
    runtime = FreeTokenManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model_ref="Qwen/Qwen3-30B-A3B",
        served_model_name="astrumweaver",
        startup_timeout_seconds=1.0,
        launch_policy=launch_policy(),
    )

    with pytest.raises(RuntimeError, match="model alias"):
        await runtime.start()

    assert process.starts == 1
    assert process.stops == 1


@pytest.mark.asyncio
async def test_managed_runtime_release_closes_api_once() -> None:
    api = FakeApi()
    process = FakeProcess(api, running=True)
    runtime = FreeTokenManagedRuntime(
        api=api,
        process=process,
        context=context(),
        model_ref="Qwen/Qwen3-30B-A3B",
        served_model_name="astrumweaver",
        startup_timeout_seconds=1.0,
        launch_policy=launch_policy(),
    )

    await runtime.release()
    await runtime.release()

    assert api.closed
