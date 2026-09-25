from __future__ import annotations

from pathlib import Path

import pytest

from astrumweaver import ResourceShape, WorkerSpec
from astrumweaver.execution import JobRequest, JobResult, ResidencyReport
from astrumweaver.runtime import (
    ExecutionDemand,
    GPUTopology,
    ModelDemand,
    ModelPreparationPolicy,
    ModelTopology,
    ResidencyPolicy,
    RuntimeCatalog,
    RuntimeCompatibility,
    RuntimeCompatibilityContext,
    RuntimeHealth,
    RuntimeHealthState,
    RuntimeHostFacts,
    RuntimeProviderInfo,
    RuntimeSelection,
    RuntimeSelectionMode,
    RuntimeSetupIntent,
    RuntimeLifecycleManager,
)
from astrumweaver.setup import (
    ActionInspection,
    ActionReceipt,
    DeploymentPath,
    PrivilegeMode,
    SecretReference,
    SetupActionKind,
    SetupActionResultStatus,
    SetupActionState,
    SetupApplyStatus,
    SetupApproval,
    SetupApprovalError,
    SetupHostSnapshot,
    SetupPlan,
    SetupPlanningError,
    apply_plan,
    build_runtime_release_plan,
    build_runtime_setup_plan,
    discover_local_host,
    dry_run_plan,
    explain_plan,
)


class FakeExecutor:
    capabilities = frozenset({"llm.chat"})

    async def execute(self, job: JobRequest) -> JobResult:
        return JobResult(outputs={"job_id": job.job_id})

    async def cancel(self, job_id: str) -> None:
        return None

    async def residency(self) -> ResidencyReport:
        return ResidencyReport()


class FakeManagedRuntime:
    provider_id = "fake"

    def __init__(self) -> None:
        self.state = RuntimeHealthState.STOPPED
        self.start_count = 0
        self.stop_count = 0
        self.release_count = 0
        self._executor = FakeExecutor()

    async def start(self) -> None:
        self.start_count += 1
        self.state = RuntimeHealthState.READY

    async def stop(self) -> None:
        self.stop_count += 1
        self.state = RuntimeHealthState.STOPPED

    async def health(self) -> RuntimeHealth:
        return RuntimeHealth(
            state=self.state,
            ready=self.state is RuntimeHealthState.READY,
        )

    async def residency(self) -> ResidencyReport:
        return ResidencyReport()

    def executor(self) -> FakeExecutor:
        return self._executor

    async def release(self) -> None:
        self.release_count += 1


class FakeProvider:
    @property
    def info(self) -> RuntimeProviderInfo:
        return RuntimeProviderInfo(
            provider_id="fake",
            display_name="Fake Runtime",
        )

    def compatibility(
        self,
        context: RuntimeCompatibilityContext,
    ) -> RuntimeCompatibility:
        return RuntimeCompatibility(provider_id="fake")

    def setup_intent(
        self,
        context: RuntimeCompatibilityContext,
    ) -> RuntimeSetupIntent:
        return RuntimeSetupIntent(
            provider_id="fake",
            package_references=("runtime-z", "runtime-a"),
            configuration={
                "endpoint": "127.0.0.1:9999",
                "api_token": SecretReference("RUNTIME_API_TOKEN"),
            },
            model_preparation=ModelPreparationPolicy.DOWNLOAD,
            model_ref=context.demand.model.model_ref,
            requires_privilege=True,
        )

    def create_runtime(self, context, setup) -> FakeManagedRuntime:
        return FakeManagedRuntime()


def context() -> RuntimeCompatibilityContext:
    return RuntimeCompatibilityContext(
        worker=WorkerSpec(
            worker_id="worker-test",
            worker_class="modern-single",
            resources=ResourceShape(
                gpu_count=1,
                total_vram_mb=24576,
                max_single_gpu_vram_mb=24576,
            ),
            gpu_uuids=("GPU-example-one",),
            capabilities=frozenset({"llm.chat"}),
        ),
        host=RuntimeHostFacts(
            cpu_count=8,
            host_ram_mb=65536,
            architecture="x86_64",
        ),
        demand=ExecutionDemand(
            model=ModelDemand(
                model_ref="example/model",
                model_format="safetensors",
                topology=ModelTopology.DENSE,
            ),
            residency_policy=ResidencyPolicy.VRAM_ONLY,
            gpu_topology=GPUTopology.SINGLE_GPU,
            min_single_gpu_vram_mb=16000,
        ),
    )


def snapshot() -> SetupHostSnapshot:
    return SetupHostSnapshot(
        runtime_host=context().host,
        deployment_path=DeploymentPath.NIXOS,
        os_id="nixos",
        os_version="26.11",
        service_manager="systemd",
        package_manager="nix",
        available_commands=frozenset({"nix", "systemctl", "nvidia-smi"}),
        privilege_mode=PrivilegeMode.SUDO,
    )


def setup_plan() -> SetupPlan:
    return build_runtime_setup_plan(
        catalog=RuntimeCatalog([FakeProvider()]),
        context=context(),
        selection=RuntimeSelection(
            mode=RuntimeSelectionMode.EXPLICIT,
            provider_id="fake",
        ),
        snapshot=snapshot(),
    )


def test_setup_plan_is_deterministic_serializable_and_secret_safe() -> None:
    first = setup_plan()
    second = setup_plan()

    assert first.digest == second.digest
    assert first.to_json() == second.to_json()
    assert first.provider_id == "fake"
    assert first.requires_privilege
    assert first.requires_network
    assert first.requires_confirmation

    package_actions = [
        action
        for action in first.actions
        if action.kind is SetupActionKind.ENSURE_PACKAGE
    ]
    assert [
        action.payload["package_reference"]
        for action in package_actions
    ] == ["runtime-a", "runtime-z"]

    encoded = first.to_json()
    assert "RUNTIME_API_TOKEN" in encoded
    assert '"$secret_ref"' in encoded
    assert "actual-secret-value" not in encoded

    round_trip = SetupPlan.from_json(encoded)
    assert round_trip.digest == first.digest
    assert round_trip.to_dict() == first.to_dict()


def test_setup_requires_explicit_runtime_choice() -> None:
    with pytest.raises(SetupPlanningError, match="explicit user-selected"):
        build_runtime_setup_plan(
            catalog=RuntimeCatalog([FakeProvider()]),
            context=context(),
            selection=RuntimeSelection(
                mode=RuntimeSelectionMode.RECOMMEND,
            ),
            snapshot=snapshot(),
        )


def test_release_plan_is_deterministic_and_provider_scoped() -> None:
    first = build_runtime_release_plan(
        provider_id="fake",
        snapshot=snapshot(),
        requires_privilege=True,
    )
    second = build_runtime_release_plan(
        provider_id="fake",
        snapshot=snapshot(),
        requires_privilege=True,
    )

    assert first.digest == second.digest
    assert [action.kind for action in first.actions] == [
        SetupActionKind.RUNTIME_STOP,
        SetupActionKind.RUNTIME_RELEASE,
    ]
    assert first.requires_privilege


class FakeDriver:
    def __init__(
        self,
        *,
        fail_kind: SetupActionKind | None = None,
    ) -> None:
        self.satisfied: set[str] = set()
        self.apply_calls: list[str] = []
        self.rollback_calls: list[str] = []
        self.fail_kind = fail_kind

    def inspect(self, action):
        return ActionInspection(
            state=(
                SetupActionState.SATISFIED
                if action.action_id in self.satisfied
                else SetupActionState.NEEDS_APPLY
            ),
            detail=(
                "already satisfied"
                if action.action_id in self.satisfied
                else "change required"
            ),
        )

    def apply(self, action):
        if action.kind is self.fail_kind:
            raise RuntimeError("synthetic apply failure")
        self.apply_calls.append(action.action_id)
        self.satisfied.add(action.action_id)
        return ActionReceipt(
            changed=True,
            detail="applied",
            evidence={
                "action_kind": action.kind.value,
                "private_value_omitted": True,
            },
            rollback_data={"action_id": action.action_id},
        )

    def rollback(self, action, receipt):
        self.rollback_calls.append(action.action_id)
        self.satisfied.discard(action.action_id)
        return ActionReceipt(
            changed=True,
            detail="rolled back",
            evidence={"rolled_back": action.action_id},
        )


def full_approval(plan: SetupPlan) -> SetupApproval:
    return SetupApproval(
        plan_digest=plan.digest,
        allow_privileged=True,
        allow_network=True,
        allow_model_download=True,
        allow_model_convert=True,
        allow_confirmation_actions=True,
    )


def test_dry_run_is_non_mutating_and_exposes_privilege_network_flags() -> None:
    plan = setup_plan()
    driver = FakeDriver()

    preview = dry_run_plan(plan, driver)

    assert preview.plan_digest == plan.digest
    assert preview.changes_required
    assert not preview.blocked
    assert driver.apply_calls == []
    assert any(action.requires_privilege for action in preview.actions)
    assert any(action.requires_network for action in preview.actions)

    explanation = explain_plan(plan)
    assert plan.digest in explanation
    assert "privileged" in explanation
    assert "network" in explanation
    assert "confirmation" in explanation


def test_apply_requires_exact_reviewed_digest_before_mutation() -> None:
    plan = setup_plan()
    driver = FakeDriver()
    other = build_runtime_release_plan(
        provider_id="fake",
        snapshot=snapshot(),
        requires_privilege=True,
    )

    with pytest.raises(SetupApprovalError, match="digest"):
        apply_plan(
            plan,
            driver,
            approval=SetupApproval(
                plan_digest=other.digest,
                allow_privileged=True,
                allow_network=True,
                allow_model_download=True,
                allow_confirmation_actions=True,
            ),
        )

    assert driver.apply_calls == []


def test_apply_requires_explicit_model_download_and_privilege_approval() -> None:
    plan = setup_plan()
    driver = FakeDriver()

    with pytest.raises(SetupApprovalError, match="privilege"):
        apply_plan(
            plan,
            driver,
            approval=SetupApproval(plan_digest=plan.digest),
        )

    with pytest.raises(SetupApprovalError, match="model download"):
        apply_plan(
            plan,
            driver,
            approval=SetupApproval(
                plan_digest=plan.digest,
                allow_privileged=True,
                allow_network=True,
                allow_confirmation_actions=True,
            ),
        )

    assert driver.apply_calls == []


def test_apply_is_idempotent_and_returns_structured_evidence() -> None:
    plan = setup_plan()
    driver = FakeDriver()
    approval = full_approval(plan)

    first = apply_plan(plan, driver, approval=approval)
    assert first.status is SetupApplyStatus.SUCCEEDED
    assert first.succeeded
    assert all(
        result.status is SetupActionResultStatus.APPLIED
        for result in first.actions
    )
    assert first.actions[0].evidence["private_value_omitted"] is True

    first_apply_count = len(driver.apply_calls)
    second = apply_plan(plan, driver, approval=approval)

    assert second.succeeded
    assert len(driver.apply_calls) == first_apply_count
    assert all(
        result.status is SetupActionResultStatus.SKIPPED
        for result in second.actions
    )


def test_progress_callback_failure_cannot_change_successful_apply_outcome() -> None:
    plan = setup_plan()
    baseline_driver = FakeDriver()
    observed_driver = FakeDriver()

    baseline = apply_plan(
        plan,
        baseline_driver,
        approval=full_approval(plan),
    )

    def broken_observer(_result) -> None:
        raise RuntimeError("observer failed")

    observed = apply_plan(
        plan,
        observed_driver,
        approval=full_approval(plan),
        on_result=broken_observer,
    )

    assert observed == baseline
    assert observed_driver.apply_calls == baseline_driver.apply_calls
    assert observed_driver.rollback_calls == baseline_driver.rollback_calls


def test_progress_callback_failure_cannot_interrupt_rollback() -> None:
    plan = setup_plan()
    baseline_driver = FakeDriver(fail_kind=SetupActionKind.HEALTH_CHECK)
    observed_driver = FakeDriver(fail_kind=SetupActionKind.HEALTH_CHECK)

    baseline = apply_plan(
        plan,
        baseline_driver,
        approval=full_approval(plan),
    )

    def broken_observer(_result) -> None:
        raise RuntimeError("observer failed")

    observed = apply_plan(
        plan,
        observed_driver,
        approval=full_approval(plan),
        on_result=broken_observer,
    )

    assert observed == baseline
    assert observed_driver.apply_calls == baseline_driver.apply_calls
    assert observed_driver.rollback_calls == baseline_driver.rollback_calls
    assert observed_driver.rollback_calls


def test_progress_callback_failure_cannot_change_skipped_outcome() -> None:
    plan = setup_plan()
    baseline_driver = FakeDriver()
    observed_driver = FakeDriver()
    baseline_driver.satisfied.update(action.action_id for action in plan.actions)
    observed_driver.satisfied.update(action.action_id for action in plan.actions)

    baseline = apply_plan(
        plan,
        baseline_driver,
        approval=full_approval(plan),
    )
    observed = apply_plan(
        plan,
        observed_driver,
        approval=full_approval(plan),
        on_result=lambda _result: (_ for _ in ()).throw(
            RuntimeError("observer failed")
        ),
    )

    assert observed == baseline
    assert all(
        result.status is SetupActionResultStatus.SKIPPED
        for result in observed.actions
    )
    assert observed_driver.apply_calls == []


def test_progress_callback_failure_cannot_change_blocked_or_inspect_failure() -> None:
    plan = setup_plan()
    target_id = plan.actions[1].action_id

    class BlockedDriver(FakeDriver):
        def inspect(self, action):
            if action.action_id == target_id:
                return ActionInspection(
                    state=SetupActionState.BLOCKED,
                    detail="synthetic block",
                )
            return super().inspect(action)

    class InspectFailDriver(FakeDriver):
        def inspect(self, action):
            if action.action_id == target_id:
                raise RuntimeError("synthetic inspect failure")
            return super().inspect(action)

    def broken_observer(_result) -> None:
        raise RuntimeError("observer failed")

    for driver_type in (BlockedDriver, InspectFailDriver):
        baseline_driver = driver_type()
        observed_driver = driver_type()
        baseline = apply_plan(
            plan,
            baseline_driver,
            approval=full_approval(plan),
        )
        observed = apply_plan(
            plan,
            observed_driver,
            approval=full_approval(plan),
            on_result=broken_observer,
        )

        assert observed == baseline
        assert observed_driver.apply_calls == baseline_driver.apply_calls
        assert observed_driver.rollback_calls == baseline_driver.rollback_calls


def test_progress_callback_failure_cannot_interrupt_rollback_failures() -> None:
    plan = setup_plan()

    class RollbackFailDriver(FakeDriver):
        def rollback(self, action, receipt):
            self.rollback_calls.append(action.action_id)
            raise RuntimeError("synthetic rollback failure")

    def broken_observer(_result) -> None:
        raise RuntimeError("observer failed")

    baseline_driver = RollbackFailDriver(
        fail_kind=SetupActionKind.HEALTH_CHECK
    )
    observed_driver = RollbackFailDriver(
        fail_kind=SetupActionKind.HEALTH_CHECK
    )
    baseline = apply_plan(
        plan,
        baseline_driver,
        approval=full_approval(plan),
    )
    observed = apply_plan(
        plan,
        observed_driver,
        approval=full_approval(plan),
        on_result=broken_observer,
    )

    assert observed == baseline
    assert observed_driver.rollback_calls == baseline_driver.rollback_calls
    assert len(observed.rollback_actions) == len(baseline.rollback_actions)
    assert all(
        result.status is SetupActionResultStatus.ROLLBACK_FAILED
        for result in observed.rollback_actions
    )


def test_apply_rolls_back_reversible_actions_in_reverse_order() -> None:
    plan = setup_plan()
    driver = FakeDriver(fail_kind=SetupActionKind.HEALTH_CHECK)

    result = apply_plan(
        plan,
        driver,
        approval=full_approval(plan),
    )

    assert result.status is SetupApplyStatus.FAILED
    assert not result.succeeded

    reversible_applied = [
        action.action_id
        for action in plan.actions
        if action.reversible and action.kind is not SetupActionKind.HEALTH_CHECK
    ]
    assert driver.rollback_calls == list(reversed(reversible_applied))
    assert [
        item.status for item in result.rollback_actions
    ] == [
        SetupActionResultStatus.ROLLED_BACK
        for _ in reversible_applied
    ]
    assert result.rollback_actions[0].evidence


def test_host_discovery_reads_only_generic_local_facts(tmp_path: Path) -> None:
    os_release = tmp_path / "os-release"
    meminfo = tmp_path / "meminfo"
    nixos_marker = tmp_path / "NIXOS"

    os_release.write_text(
        'ID=nixos\nVERSION_ID="26.11"\n',
        encoding="utf-8",
    )
    meminfo.write_text(
        "MemTotal:       67108864 kB\n",
        encoding="utf-8",
    )
    nixos_marker.write_text("", encoding="utf-8")

    available = {
        "systemctl": "/run/current-system/sw/bin/systemctl",
        "nix": "/run/current-system/sw/bin/nix",
        "nvidia-smi": "/run/current-system/sw/bin/nvidia-smi",
        "sudo": "/run/wrappers/bin/sudo",
    }

    discovered = discover_local_host(
        os_release_path=os_release,
        meminfo_path=meminfo,
        nixos_marker_path=nixos_marker,
        which=lambda command: available.get(command),
        cpu_count=lambda: 16,
        geteuid=lambda: 1000,
        machine=lambda: "x86_64",
    )

    assert discovered.deployment_path is DeploymentPath.NIXOS
    assert discovered.runtime_host.cpu_count == 16
    assert discovered.runtime_host.host_ram_mb == 65536
    assert discovered.runtime_host.architecture == "x86_64"
    assert discovered.privilege_mode is PrivilegeMode.SUDO
    assert discovered.metadata == {}
    serialized = discovered.to_dict()
    assert "hostname" not in serialized
    assert "ip" not in serialized


@pytest.mark.asyncio
async def test_runtime_lifecycle_is_idempotent_and_releases_once() -> None:
    runtime = FakeManagedRuntime()
    lifecycle = RuntimeLifecycleManager(
        runtime,
        poll_interval_seconds=0.01,
    )

    ready = await lifecycle.ensure_ready(timeout_seconds=1.0)
    assert ready.ready
    assert runtime.start_count == 1

    await lifecycle.ensure_ready(timeout_seconds=1.0)
    assert runtime.start_count == 1

    stopped = await lifecycle.ensure_stopped(
        timeout_seconds=1.0,
        release=True,
    )
    assert stopped.state is RuntimeHealthState.STOPPED
    assert runtime.stop_count == 1
    assert runtime.release_count == 1

    await lifecycle.ensure_stopped(
        timeout_seconds=1.0,
        release=True,
    )
    assert runtime.stop_count == 1
    assert runtime.release_count == 1

    await lifecycle.restart(timeout_seconds=1.0)
    assert runtime.start_count == 2



def test_setup_plan_rejects_tampered_digest() -> None:
    plan = setup_plan()
    encoded = plan.to_dict()
    encoded["digest"] = "0" * 64

    with pytest.raises(ValueError, match="digest"):
        SetupPlan.from_dict(encoded)


class LeakyDriver(FakeDriver):
    def inspect(self, action):
        raise RuntimeError("token=super-secret-value")


def test_unexpected_driver_errors_do_not_leak_exception_text() -> None:
    plan = setup_plan()
    driver = LeakyDriver()

    preview = dry_run_plan(plan, driver)
    assert preview.blocked
    assert "super-secret-value" not in preview.actions[0].detail
    assert "RuntimeError" in preview.actions[0].detail

    result = apply_plan(
        plan,
        driver,
        approval=full_approval(plan),
    )
    assert not result.succeeded
    assert "super-secret-value" not in result.actions[0].detail
    assert "RuntimeError" in result.actions[0].detail
