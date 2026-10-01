from __future__ import annotations

import json
import os
import stat
import tomllib
from pathlib import Path

import pytest

from astrumweaver import AcceleratorDevice, ResourceShape, WorkerSpec
from astrumweaver.runtime import (
    ExecutionDemand,
    GPUTopology,
    ModelDemand,
    ModelTopology,
    ResidencyPolicy,
    RuntimeCatalog,
    RuntimeCompatibilityContext,
    RuntimeHostFacts,
    RuntimeSelection,
    RuntimeSelectionMode,
)
from astrumweaver.runtime.providers import LlamaCppProvider
from astrumweaver.setup import (
    DeploymentPath,
    PrivilegeMode,
    SetupAction,
    SetupActionKind,
    SetupActionState,
    SetupHostSnapshot,
    build_runtime_setup_plan,
)
from astrumweaver.setup.first_run import (
    FirstRunExecutionMode,
    render_worker_toml,
)
from astrumweaver.setup.migration import (
    DEFAULT_RUNTIME_MANIFEST,
    parse_installed_worker_toml,
    parse_installed_worker_unit,
    render_runtime_worker_toml,
    render_runtime_worker_unit,
)
from astrumweaver.setup.systemd import SystemdSetupDriver


def _smoke_worker() -> WorkerSpec:
    return WorkerSpec(
        worker_id="legacy-single",
        worker_class="gpu-single",
        resources=ResourceShape(
            gpu_count=1,
            total_vram_mb=24576,
            max_single_gpu_vram_mb=24576,
        ),
        gpu_uuids=("GPU-example",),
        accelerators=(
            AcceleratorDevice(
                uuid="GPU-example",
                memory_mb=24576,
                compute_capability="8.6",
                device_class="NVIDIA RTX 3090",
            ),
        ),
        capabilities=frozenset({"debug.echo"}),
        labels={"site": "test"},
    )


def _smoke_toml() -> str:
    return render_worker_toml(
        _smoke_worker(),
        control_url="https://control.example",
        execution_mode=FirstRunExecutionMode.SMOKE,
    )


def _worker_unit(
    *,
    executable: str = (
        "/nix/var/nix/profiles/astrumweaver-installer/bin/"
        "astrumweaver-worker"
    ),
    runtime: bool = False,
) -> str:
    runtime_arg = (
        " --runtime-manifest /etc/astrumweaver/runtime-deployment.json"
        if runtime
        else ""
    )
    return f"""[Unit]
Description=AstrumWeaver GPU Worker
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=astrumweaver
Group=astrumweaver
SupplementaryGroups=astrumweaver-config
EnvironmentFile=-/etc/astrumweaver/worker.env
ExecStartPre=+/usr/local/libexec/astrumweaver/gpu-preflight /etc/astrumweaver/gpu-uuids
ExecStart={executable} --config /etc/astrumweaver/worker.toml{runtime_arg}
Restart=on-failure
RestartSec=5s
RuntimeDirectory=astrumweaver-worker
RuntimeDirectoryMode=0750
StateDirectory=astrumweaver
StateDirectoryMode=0750
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6

[Install]
WantedBy=multi-user.target
"""


def _reconcile_action() -> SetupAction:
    return SetupAction(
        action_id="01-reconcile-worker-execution",
        kind=SetupActionKind.RECONCILE_WORKER_EXECUTION,
        description="Migrate smoke Worker to runtime execution",
        payload={
            "provider_id": "llama-cpp",
            "source_execution": "smoke",
            "desired_execution": "runtime",
            "capabilities": ["llm.chat", "text.generate"],
            "runtime_manifest": DEFAULT_RUNTIME_MANIFEST,
            "runtime_deployment": {"provider_id": "llama-cpp"},
            "startup_timeout_seconds": 600,
            "shutdown_timeout_seconds": 60,
        },
        requires_privilege=True,
        reversible=True,
    )


def _write_staged_worker(root: Path) -> tuple[Path, Path, Path]:
    worker_path = root / "etc/astrumweaver/worker.toml"
    unit_path = root / "etc/systemd/system/astrumweaver-worker.service"
    manifest_path = root / "etc/astrumweaver/runtime-deployment.json"
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    worker_path.write_text(_smoke_toml(), encoding="utf-8")
    unit_path.write_text(_worker_unit(), encoding="utf-8")
    manifest_path.write_text(
        json.dumps({"provider_id": "llama-cpp"}) + "\n",
        encoding="utf-8",
    )
    os.chmod(worker_path, 0o640)
    os.chmod(unit_path, 0o644)
    return worker_path, unit_path, manifest_path


def test_worker_toml_migration_preserves_installed_contract() -> None:
    source = _smoke_toml()
    before = parse_installed_worker_toml(source)

    rendered = render_runtime_worker_toml(source)
    after = parse_installed_worker_toml(rendered)
    parsed = tomllib.loads(rendered)

    assert before.execution_mode == "smoke"
    assert after.execution_mode == "runtime"
    assert after.spec.worker_id == before.spec.worker_id
    assert after.spec.worker_class == before.spec.worker_class
    assert after.spec.gpu_uuids == before.spec.gpu_uuids
    assert after.spec.resources == before.spec.resources
    assert after.spec.accelerators == before.spec.accelerators
    assert after.spec.labels == before.spec.labels
    assert after.control_url == before.control_url
    assert after.max_concurrency == before.max_concurrency
    assert after.gpu_preflight == before.gpu_preflight
    assert after.health_host == before.health_host
    assert after.health_port == before.health_port
    assert parsed["worker"]["capabilities"] == [
        "llm.chat",
        "text.generate",
    ]
    assert "executor" not in parsed
    assert parsed["runtime"]["manifest"] == DEFAULT_RUNTIME_MANIFEST


def test_worker_migration_accepts_documented_optional_defaults() -> None:
    source = _smoke_toml().replace("gpu_preflight = true\n", "", 1)
    source = source.replace(
        "health_port = 9100\n",
        (
            "health_port = 9100\n"
            "poll_interval_seconds = 1.0\n"
            "heartbeat_interval_seconds = 5.0\n"
        ),
        1,
    )

    before = parse_installed_worker_toml(source)
    rendered = render_runtime_worker_toml(source)
    after = parse_installed_worker_toml(rendered)

    assert before.gpu_preflight is True
    assert after.gpu_preflight is True
    assert "poll_interval_seconds = 1.0" in rendered
    assert "heartbeat_interval_seconds = 5.0" in rendered


def test_worker_unit_migration_preserves_stable_executable() -> None:
    source = _worker_unit()
    before = parse_installed_worker_unit(source)

    rendered = render_runtime_worker_unit(source)
    after = parse_installed_worker_unit(rendered)

    assert before.execution_mode == "smoke"
    assert after.execution_mode == "runtime"
    assert after.executable == before.executable
    assert rendered.count("--runtime-manifest") == 1


@pytest.mark.parametrize(
    "text",
    (
        _smoke_toml() + "\n[unrelated]\nvalue = true\n",
        _smoke_toml().replace(
            "health_port = 9100\n",
            'health_port = 9100\nunknown_field = "drift"\n',
            1,
        ),
        _smoke_toml().replace(
            "astrumweaver.executors.structured_echo:create_executor",
            "example.custom:create_executor",
        ),
    ),
)
def test_worker_migration_rejects_unrecognized_worker_toml(text: str) -> None:
    with pytest.raises(RuntimeError):
        parse_installed_worker_toml(text)


def test_worker_migration_rejects_modified_unit() -> None:
    modified = _worker_unit().replace(
        "Restart=on-failure\n",
        "Environment=UNREVIEWED=1\nRestart=on-failure\n",
    )

    with pytest.raises(RuntimeError):
        parse_installed_worker_unit(modified)


def test_loading_installed_worker_never_reads_worker_env(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    _write_staged_worker(root)
    env_path = root / "etc/astrumweaver/worker.env"
    env_path.symlink_to(root / "definitely-missing-secret")

    contract = SystemdSetupDriver(root=root).load_installed_worker_contract()

    assert contract.spec.worker_id == "legacy-single"
    assert contract.execution_mode == "smoke"


def test_already_runtime_worker_is_noop_only_for_exact_deployment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    worker_path, unit_path, manifest_path = _write_staged_worker(root)
    worker_path.write_text(
        render_runtime_worker_toml(_smoke_toml()),
        encoding="utf-8",
    )
    unit_path.write_text(_worker_unit(runtime=True), encoding="utf-8")

    driver = SystemdSetupDriver(root=root)
    reconcile = _reconcile_action()
    stop = SetupAction(
        action_id="01-worker-stop",
        kind=SetupActionKind.WORKER_STOP,
        description="Stop Worker before migration",
        payload={
            "provider_id": "llama-cpp",
            "source_execution": "smoke",
            "desired_execution": "runtime",
            "runtime_deployment": {"provider_id": "llama-cpp"},
        },
        requires_privilege=True,
        reversible=True,
    )

    assert driver.inspect(stop).state is SetupActionState.SATISFIED
    assert driver.inspect(reconcile).state is SetupActionState.SATISFIED

    manifest_path.write_text(
        json.dumps(
            {"provider_id": "llama-cpp", "unexpected": "drift"}
        )
        + "\n",
        encoding="utf-8",
    )

    assert driver.inspect(stop).state is SetupActionState.BLOCKED
    assert driver.inspect(reconcile).state is SetupActionState.BLOCKED


def test_systemd_driver_reconcile_is_idempotent_and_reversible(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    worker_path, unit_path, _ = _write_staged_worker(root)
    worker_mode = stat.S_IMODE(worker_path.stat().st_mode)
    unit_mode = stat.S_IMODE(unit_path.stat().st_mode)

    driver = SystemdSetupDriver(root=root)
    action = _reconcile_action()

    assert driver.inspect(action).state is SetupActionState.NEEDS_APPLY
    receipt = driver.apply(action)
    assert receipt.changed
    assert driver.inspect(action).state is SetupActionState.SATISFIED
    assert parse_installed_worker_toml(
        worker_path.read_text(encoding="utf-8")
    ).execution_mode == "runtime"
    assert parse_installed_worker_unit(
        unit_path.read_text(encoding="utf-8")
    ).execution_mode == "runtime"
    assert stat.S_IMODE(worker_path.stat().st_mode) == worker_mode
    assert stat.S_IMODE(unit_path.stat().st_mode) == unit_mode

    second = driver.apply(action)
    assert not second.changed

    rollback = driver.rollback(action, receipt)
    assert rollback.changed
    assert parse_installed_worker_toml(
        worker_path.read_text(encoding="utf-8")
    ).execution_mode == "smoke"
    assert parse_installed_worker_unit(
        unit_path.read_text(encoding="utf-8")
    ).execution_mode == "smoke"


def test_systemd_driver_blocks_mixed_worker_execution_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    worker_path, unit_path, _ = _write_staged_worker(root)
    worker_path.write_text(
        render_runtime_worker_toml(_smoke_toml()),
        encoding="utf-8",
    )
    unit_path.write_text(_worker_unit(runtime=False), encoding="utf-8")

    driver = SystemdSetupDriver(root=root)
    inspection = driver.inspect(_reconcile_action())

    assert inspection.state is SetupActionState.BLOCKED
    assert "mixed execution authority" in inspection.detail


def test_runtime_migration_plan_orders_authority_change_before_preflight() -> None:
    runtime_worker = WorkerSpec(
        worker_id="legacy-single",
        worker_class="gpu-single",
        resources=_smoke_worker().resources,
        gpu_uuids=_smoke_worker().gpu_uuids,
        accelerators=_smoke_worker().accelerators,
        capabilities=frozenset({"llm.chat", "text.generate"}),
        labels=_smoke_worker().labels,
    )
    host = RuntimeHostFacts(
        cpu_count=16,
        host_ram_mb=65536,
        architecture="x86_64",
    )
    context = RuntimeCompatibilityContext(
        worker=runtime_worker,
        host=host,
        demand=ExecutionDemand(
            model=ModelDemand(
                model_ref="/models/qwen.gguf",
                model_format="gguf",
                topology=ModelTopology.DENSE,
                estimated_size_mb=18000,
            ),
            residency_policy=ResidencyPolicy.PREFER_VRAM,
            gpu_topology=GPUTopology.SINGLE_GPU,
        ),
    )
    snapshot = SetupHostSnapshot(
        runtime_host=host,
        deployment_path=DeploymentPath.SYSTEMD,
        os_id="debian",
        os_version="13",
        service_manager="systemd",
        package_manager="nix",
        available_commands=frozenset(
            {"nix", "systemctl", "nvidia-smi"}
        ),
        privilege_mode=PrivilegeMode.ROOT,
    )
    plan = build_runtime_setup_plan(
        catalog=RuntimeCatalog((LlamaCppProvider(),)),
        context=context,
        selection=RuntimeSelection(
            mode=RuntimeSelectionMode.EXPLICIT,
            provider_id="llama-cpp",
        ),
        snapshot=snapshot,
        reconcile_existing_worker=True,
    )

    kinds = [action.kind for action in plan.actions]
    assert kinds.index(SetupActionKind.RENDER_CONFIG) < kinds.index(
        SetupActionKind.WORKER_STOP
    )
    assert kinds.index(SetupActionKind.WORKER_STOP) < kinds.index(
        SetupActionKind.RECONCILE_WORKER_EXECUTION
    )
    assert kinds.index(
        SetupActionKind.RECONCILE_WORKER_EXECUTION
    ) < kinds.index(SetupActionKind.PREFLIGHT)
    assert kinds.index(SetupActionKind.PREFLIGHT) < kinds.index(
        SetupActionKind.RUNTIME_START
    )
