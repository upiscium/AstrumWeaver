from __future__ import annotations

import subprocess
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from astrumweaver.execution import JobExecutor, JobRequest, JobResult, ResidencyReport
from astrumweaver.runtime import (
    CompatibilityReason,
    ManagedRuntime,
    RuntimeCatalog,
    RuntimeCompatibility,
    RuntimeCompatibilityContext,
    RuntimeHealth,
    RuntimeHealthState,
    RuntimeProviderInfo,
    RuntimeSetupIntent,
)
from astrumweaver.runtime.providers import VllmProvider
from astrumweaver.setup import (
    ActionInspection,
    ActionReceipt,
    DeploymentPath,
    DiscoveredGpu,
    PrivilegeMode,
    SetupActionState,
    SetupHostSnapshot,
    discover_local_gpus,
)
from astrumweaver.runtime import ModelPreparationPolicy, RuntimeHostFacts
from astrumweaver.setup.tui import (
    TuiRunStatus,
    build_worker_spec,
    configure_provider,
    run_setup_tui,
)


Response = str | Callable[[str], str]


class ScriptedIO:
    def __init__(self, responses: list[Response]) -> None:
        self.responses = deque(responses)
        self.output: list[str] = []
        self.prompts: list[str] = []
        self.cleared = 0

    def write(self, text: str = "") -> None:
        self.output.append(text)

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError(f"unexpected prompt: {prompt}")
        response = self.responses.popleft()
        return response(prompt) if callable(response) else response

    def clear(self) -> None:
        self.cleared += 1


class FakeExecutor(JobExecutor):
    capabilities = frozenset({"llm.chat", "text.generate"})

    async def execute(self, job: JobRequest) -> JobResult:
        return JobResult(outputs={"job_id": job.job_id})

    async def cancel(self, job_id: str) -> None:
        return None

    async def residency(self) -> ResidencyReport:
        return ResidencyReport()


class FakeRuntime(ManagedRuntime):
    provider_id = "fake"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def health(self) -> RuntimeHealth:
        return RuntimeHealth(state=RuntimeHealthState.READY, ready=True)

    async def residency(self) -> ResidencyReport:
        return ResidencyReport()

    def executor(self) -> JobExecutor:
        return FakeExecutor()

    async def release(self) -> None:
        return None


@dataclass
class FakeProvider:
    provider_id: str = "fake"
    compatible: bool = True

    @property
    def info(self) -> RuntimeProviderInfo:
        return RuntimeProviderInfo(
            provider_id=self.provider_id,
            display_name=self.provider_id.upper(),
        )

    def compatibility(
        self,
        context: RuntimeCompatibilityContext,
    ) -> RuntimeCompatibility:
        if self.compatible:
            return RuntimeCompatibility(provider_id=self.provider_id)
        return RuntimeCompatibility(
            provider_id=self.provider_id,
            reasons=(
                CompatibilityReason(
                    code="blocked-for-test",
                    message="provider is intentionally incompatible",
                ),
            ),
        )

    def setup_intent(
        self,
        context: RuntimeCompatibilityContext,
    ) -> RuntimeSetupIntent:
        return RuntimeSetupIntent(
            provider_id=self.provider_id,
            model_preparation=ModelPreparationPolicy.REFERENCE_ONLY,
            model_ref=context.demand.model.model_ref,
            configuration={"test": True},
        )

    def create_runtime(
        self,
        context: RuntimeCompatibilityContext,
        setup: RuntimeSetupIntent,
    ) -> ManagedRuntime:
        return FakeRuntime()


class FakeDriver:
    def __init__(self) -> None:
        self.applied: list[str] = []

    def inspect(self, action):
        return ActionInspection(state=SetupActionState.NEEDS_APPLY)

    def apply(self, action):
        self.applied.append(action.action_id)
        return ActionReceipt(changed=True, detail="applied")

    def rollback(self, action, receipt):
        return ActionReceipt(changed=True, detail="rolled back")


def snapshot() -> SetupHostSnapshot:
    return SetupHostSnapshot(
        runtime_host=RuntimeHostFacts(
            cpu_count=16,
            host_ram_mb=65536,
            architecture="x86_64",
        ),
        deployment_path=DeploymentPath.NIXOS,
        os_id="nixos",
        os_version="26.11",
        service_manager="systemd",
        package_manager="nix",
        available_commands=frozenset({"nix", "systemctl", "nvidia-smi"}),
        privilege_mode=PrivilegeMode.SUDO,
    )


def one_gpu() -> tuple[DiscoveredGpu, ...]:
    return (
        DiscoveredGpu(
            uuid="GPU-one",
            memory_mb=24576,
            compute_capability="8.6",
        ),
    )


def wizard_responses(*, runtime: str = "fake") -> list[Response]:
    return [
        "",  # all GPUs
        "",  # worker id
        "",  # worker class
        "org/model",
        "fake",
        "",  # dense
        "",  # prefer_vram
        "",  # unknown model size
        "",  # min total VRAM
        "",  # min single VRAM
        "",  # min host RAM
        "",  # preferred host RAM
        runtime,
    ]


def test_discover_local_gpus_reads_uuid_vram_and_compute_capability() -> None:
    def run(args, **kwargs):
        assert "--query-gpu=uuid,memory.total,compute_cap" in args
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="GPU-a, 24576, 8.6\nGPU-b, 12288, 8.6\n",
            stderr="",
        )

    gpus = discover_local_gpus(run=run)

    assert [gpu.uuid for gpu in gpus] == ["GPU-a", "GPU-b"]
    assert [gpu.memory_mb for gpu in gpus] == [24576, 12288]
    assert [gpu.compute_capability for gpu in gpus] == ["8.6", "8.6"]


def test_discover_local_gpus_falls_back_when_compute_query_is_unavailable() -> None:
    calls = 0

    def run(args, **kwargs):
        nonlocal calls
        calls += 1
        if "compute_cap" in args[1]:
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="GPU-a, 24576\n",
            stderr="",
        )

    gpus = discover_local_gpus(run=run)

    assert calls == 2
    assert gpus == (DiscoveredGpu(uuid="GPU-a", memory_mb=24576),)


def test_worker_shape_preserves_selected_gpu_order_and_min_compute_capability() -> None:
    gpus = (
        DiscoveredGpu("GPU-new", 24576, "8.6"),
        DiscoveredGpu("GPU-old", 12288, "8.0"),
    )

    worker = build_worker_spec(
        worker_id="worker",
        worker_class="mixed",
        gpus=gpus,
    )

    assert worker.gpu_uuids == ("GPU-new", "GPU-old")
    assert worker.resources.total_vram_mb == 36864
    assert worker.resources.max_single_gpu_vram_mb == 24576
    assert worker.labels["gpu.compute_capability.min"] == "8.0"


def test_runtime_specific_option_editor_rebuilds_selected_provider_only() -> None:
    io = ScriptedIO(
        [
            "y",
            "0.80",
            "2",
            "true",
            "4.5",
            "true",
        ]
    )

    configured = configure_provider(io, VllmProvider())

    assert configured.config.gpu_memory_utilization == 0.80
    assert configured.config.tensor_parallel_size == 2
    assert configured.config.enable_expert_parallel is True
    assert configured.config.cpu_offload_gb == 4.5
    assert configured.config.enforce_eager is True


def test_planning_only_tui_shows_all_candidates_and_preserves_explicit_choice() -> None:
    io = ScriptedIO(wizard_responses())
    catalog = RuntimeCatalog(
        (
            FakeProvider(provider_id="bad", compatible=False),
            FakeProvider(provider_id="fake", compatible=True),
        )
    )

    result = run_setup_tui(
        io=io,
        snapshot=snapshot(),
        gpus=one_gpu(),
        catalog=catalog,
        driver=None,
    )

    assert result.status is TuiRunStatus.PLANNED
    assert result.provider_id == "fake"
    assert result.plan_digest is not None
    output = "\n".join(io.output)
    assert "blocked-for-test" in output
    assert "[compatible] fake" in output
    assert "Planning-only mode" in output
    assert "No secret values" in output


def test_tui_requires_exact_plan_confirmation_and_streams_apply_progress() -> None:
    def exact_apply(prompt: str) -> str:
        marker = "Type '"
        assert marker in prompt
        return prompt.split(marker, 1)[1].split("'", 1)[0]

    io = ScriptedIO(wizard_responses() + [exact_apply])
    driver = FakeDriver()

    result = run_setup_tui(
        io=io,
        snapshot=snapshot(),
        gpus=one_gpu(),
        catalog=RuntimeCatalog((FakeProvider(),)),
        driver=driver,
    )

    assert result.status is TuiRunStatus.APPLIED
    assert result.apply_result is not None
    assert result.apply_result.succeeded
    assert driver.applied
    output = "\n".join(io.output)
    assert "Applying reviewed plan" in output
    assert ": applied" in output
    assert "Setup completed successfully." in output


def test_tui_cancelled_when_exact_apply_token_is_not_entered() -> None:
    io = ScriptedIO(wizard_responses() + ["no"])
    driver = FakeDriver()

    result = run_setup_tui(
        io=io,
        snapshot=snapshot(),
        gpus=one_gpu(),
        catalog=RuntimeCatalog((FakeProvider(),)),
        driver=driver,
    )

    assert result.status is TuiRunStatus.CANCELLED
    assert driver.applied == []
