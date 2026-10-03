from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrumweaver.setup import (
    SetupAction,
    SetupActionKind,
    SetupActionState,
    SystemdSetupDriver,
)


def action(
    kind: SetupActionKind,
    *,
    payload: dict | None = None,
    action_id: str = "01-action",
) -> SetupAction:
    return SetupAction(
        action_id=action_id,
        kind=kind,
        description="test action",
        payload=payload or {},
    )


def fake_nvidia_smi(tmp_path: Path, output: str) -> str:
    executable = tmp_path / "nvidia-smi"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"--query-gpu=uuid\" ]]; then\n"
        "  cat <<'ASTRUM_GPU_EOF'\n"
        + output
        + "\nASTRUM_GPU_EOF\n"
        "  exit 0\n"
        "fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return str(executable)


def fake_systemctl_show(tmp_path: Path, properties: list[str]) -> str:
    executable = tmp_path / "systemctl"
    commands = "".join(
        f"printf '%s\\n' {property_value!r}\n"
        for property_value in properties
    )
    executable.write_text("#!/bin/sh\n" + commands, encoding="utf-8")
    executable.chmod(0o755)
    return str(executable)


def test_systemd_driver_renders_reviewed_runtime_manifest_under_staged_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    driver = SystemdSetupDriver(root=root)
    deployment = {
        "schema_version": "v1",
        "provider_id": "ollama",
        "provider_config": {"keep_alive": "5m"},
        "demand": {
            "model": {
                "model_ref": "qwen3:8b",
                "model_format": "ollama",
                "topology": "dense",
                "estimated_size_mb": None,
                "metadata": {},
            },
            "residency_policy": "prefer_vram",
            "gpu_topology": "single_gpu",
            "min_gpu_count": 0,
            "min_total_vram_mb": 0,
            "min_single_gpu_vram_mb": 0,
            "min_host_ram_mb": 0,
            "preferred_host_ram_mb": 0,
            "metadata": {},
        },
        "setup_intent": None,
    }
    render = action(
        SetupActionKind.RENDER_CONFIG,
        payload={
            "provider_id": "ollama",
            "configuration": {},
            "runtime_deployment": deployment,
        },
    )

    before = driver.inspect(render)
    receipt = driver.apply(render)
    after = driver.inspect(render)

    assert before.state is SetupActionState.NEEDS_APPLY
    assert receipt.changed
    assert after.state is SetupActionState.SATISFIED
    manifest = (
        root / "etc/astrumweaver/runtime-deployment.json"
    )
    assert json.loads(manifest.read_text(encoding="utf-8")) == deployment

    rollback = driver.rollback(render, receipt)
    assert rollback.changed
    assert not manifest.exists()


def test_systemd_driver_never_guesses_package_installer(
    tmp_path: Path,
) -> None:
    install = action(
        SetupActionKind.ENSURE_PACKAGE,
        payload={
            "provider_id": "custom-provider",
            "package_reference": "custom-runtime",
        },
    )
    driver = SystemdSetupDriver(root=tmp_path / "root")

    inspection = driver.inspect(install)

    assert inspection.state is SetupActionState.BLOCKED
    assert "no explicit installer" in inspection.detail


def test_systemd_driver_runs_only_explicit_package_installer(
    tmp_path: Path,
) -> None:
    log = tmp_path / "installer.log"
    installer = tmp_path / "install.sh"
    installer.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s' \"$1\" > {str(log)!r}\n",
        encoding="utf-8",
    )
    installer.chmod(0o755)
    driver = SystemdSetupDriver(
        root=tmp_path / "root",
        installers={
            "custom-provider": (str(installer),),
        },
    )
    install = action(
        SetupActionKind.ENSURE_PACKAGE,
        payload={
            "provider_id": "custom-provider",
            "package_reference": "custom-runtime",
        },
    )

    assert driver.inspect(install).state is SetupActionState.NEEDS_APPLY
    receipt = driver.apply(install)

    assert receipt.changed
    assert log.read_text(encoding="utf-8") == "custom-runtime"
    assert driver.inspect(install).state is SetupActionState.SATISFIED


def test_systemd_driver_materializes_narrow_nvidia_driver_bridge(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    runtime_state = root / "etc/astrumweaver/runtime"
    runtime_state.mkdir(parents=True)

    host_driver = tmp_path / "host/libcuda.so.1"
    host_driver.parent.mkdir()
    host_driver.write_bytes(b"host-driver")

    driver = SystemdSetupDriver(
        root=root,
        nvidia_driver_library=host_driver,
    )
    bridge = action(
        SetupActionKind.ENSURE_NVIDIA_DRIVER_BRIDGE,
        payload={
            "provider_id": "llama-cpp",
            "soname": "libcuda.so.1",
            "bridge_directory": (
                "/etc/astrumweaver/runtime/nvidia-driver"
            ),
        },
    )

    assert driver.inspect(bridge).state is SetupActionState.NEEDS_APPLY
    receipt = driver.apply(bridge)

    target = (
        runtime_state / "nvidia-driver/libcuda.so.1"
    )
    assert receipt.changed
    assert target.is_symlink()
    assert target.readlink() == host_driver
    assert target.parent.stat().st_mode & 0o777 == 0o755
    assert driver.inspect(bridge).state is SetupActionState.SATISFIED

    rollback = driver.rollback(bridge, receipt)

    assert rollback.changed
    assert not target.exists()
    assert not target.is_symlink()


def test_systemd_driver_blocks_unmanaged_nvidia_bridge_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    bridge_dir = root / "etc/astrumweaver/runtime/nvidia-driver"
    bridge_dir.mkdir(parents=True)
    (bridge_dir / "libcuda.so.1").write_bytes(b"operator-file")

    host_driver = tmp_path / "host/libcuda.so.1"
    host_driver.parent.mkdir()
    host_driver.write_bytes(b"host-driver")

    driver = SystemdSetupDriver(
        root=root,
        nvidia_driver_library=host_driver,
    )
    bridge = action(
        SetupActionKind.ENSURE_NVIDIA_DRIVER_BRIDGE,
        payload={
            "provider_id": "llama-cpp",
            "soname": "libcuda.so.1",
            "bridge_directory": (
                "/etc/astrumweaver/runtime/nvidia-driver"
            ),
        },
    )

    inspection = driver.inspect(bridge)

    assert inspection.state is SetupActionState.BLOCKED
    assert "not a managed symlink" in inspection.detail


def test_exllamav3_packages_are_not_satisfied_by_python_alone(
    tmp_path: Path,
) -> None:
    install = action(
        SetupActionKind.ENSURE_PACKAGE,
        payload={
            "provider_id": "exllamav3",
            "package_reference": "exllamav3",
        },
    )
    driver = SystemdSetupDriver(root=tmp_path / "root")

    inspection = driver.inspect(install)

    assert inspection.state is SetupActionState.BLOCKED
    assert "no explicit installer" in inspection.detail


def test_systemd_model_download_requires_explicit_preparer(
    tmp_path: Path,
) -> None:
    download = action(
        SetupActionKind.DOWNLOAD_MODEL,
        payload={
            "provider_id": "exllamav3",
            "model_ref": "org/model-exl3",
        },
    )
    driver = SystemdSetupDriver(root=tmp_path / "root")

    inspection = driver.inspect(download)

    assert inspection.state is SetupActionState.BLOCKED
    assert "explicit reviewed model command" in inspection.detail


def test_systemd_gpu_preflight_accepts_exact_set_and_rejects_unisolated_superset(
    tmp_path: Path,
) -> None:
    worker_config = tmp_path / "worker.toml"
    worker_config.write_text(
        '[worker]\ngpu_uuids = ["GPU-a"]\n',
        encoding="utf-8",
    )
    manifest = tmp_path / "runtime.json"
    manifest.write_text("{}\n", encoding="utf-8")
    preflight = action(
        SetupActionKind.PREFLIGHT,
        payload={"provider_id": "ollama"},
    )

    exact_dir = tmp_path / "exact"
    exact_dir.mkdir()
    exact = SystemdSetupDriver(
        worker_config_path=worker_config,
        runtime_manifest_path=manifest,
        nvidia_smi=fake_nvidia_smi(exact_dir, "GPU-a"),
    )
    exact_result = exact.inspect(preflight)

    superset_dir = tmp_path / "superset"
    superset_dir.mkdir()
    superset = SystemdSetupDriver(
        worker_config_path=worker_config,
        runtime_manifest_path=manifest,
        nvidia_smi=fake_nvidia_smi(
            superset_dir,
            "GPU-a\nGPU-b",
        ),
    )
    superset_result = superset.inspect(preflight)

    assert exact_result.state is SetupActionState.SATISFIED
    assert superset_result.state is SetupActionState.BLOCKED
    assert "no verified service device isolation" in superset_result.detail


@pytest.mark.parametrize(
    ("effective_policy", "effective_exec_start_pre", "should_verify"),
    (
        (
            "closed",
            "{ path=/usr/local/libexec/astrumweaver/gpu-preflight ; "
            "argv[]=/usr/local/libexec/astrumweaver/gpu-preflight "
            "/etc/astrumweaver/gpu-uuids ; ignore_errors=no ; }",
            True,
        ),
        (
            "auto",
            "{ path=/usr/local/libexec/astrumweaver/gpu-preflight ; "
            "argv[]=/usr/local/libexec/astrumweaver/gpu-preflight "
            "/etc/astrumweaver/gpu-uuids ; ignore_errors=no ; }",
            False,
        ),
        (
            "closed",
            "{ path=/bin/false ; argv[]=/bin/false gpu-preflight ; "
            "ignore_errors=no ; }",
            False,
        ),
    ),
)
def test_systemd_gpu_preflight_uses_effective_device_isolation_properties(
    tmp_path: Path,
    effective_policy: str,
    effective_exec_start_pre: str,
    should_verify: bool,
) -> None:
    worker_config = tmp_path / "worker.toml"
    worker_config.write_text(
        '[worker]\ngpu_uuids = ["GPU-a"]\n',
        encoding="utf-8",
    )
    manifest = tmp_path / "runtime.json"
    manifest.write_text("{}\n", encoding="utf-8")
    expected = tmp_path / "gpu-uuids"
    expected.write_text("GPU-a\n", encoding="utf-8")
    device_map = tmp_path / "gpu-device-map"
    device_map.write_text("GPU-a=/dev/nvidia0\n", encoding="utf-8")
    dropin = tmp_path / "10-gpu-isolation.conf"
    dropin.write_text(
        "[Service]\n"
        "DevicePolicy=closed\n"
        "DeviceAllow=/dev/nvidia0 rw\n"
        "DeviceAllow=/dev/nvidiactl rw\n"
        "Environment=CUDA_VISIBLE_DEVICES=GPU-a\n"
        "Environment=ASTRUMWEAVER_GPU_PREFLIGHT_MODE=isolated-access\n"
        "Environment=ASTRUMWEAVER_GPU_DEVICE_MAP=/etc/astrumweaver/gpu-device-map\n"
        "Environment=ASTRUMWEAVER_GPU_WORKER_CONFIG=/etc/astrumweaver/worker.toml\n"
        "Environment=ASTRUMWEAVER_GPU_DEVICE_MAP_COMMAND=/usr/local/libexec/astrumweaver/gpu-device-map\n"
        "Environment=ASTRUMWEAVER_NVIDIA_SMI=/usr/bin/nvidia-smi\n",
        encoding="utf-8",
    )
    verifier = tmp_path / "gpu-device-map-verify"
    verifier_args = tmp_path / "gpu-device-map-verify.args"
    verifier.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$@\" > {str(verifier_args)!r}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    verifier.chmod(0o755)

    superset_dir = tmp_path / "isolated-superset"
    superset_dir.mkdir()
    nvidia_smi = fake_nvidia_smi(superset_dir, "GPU-a\nGPU-b")
    systemctl = fake_systemctl_show(
        tmp_path,
        [
            f"DevicePolicy={effective_policy}",
            "DeviceAllow=/dev/nvidia0 rw",
            "DeviceAllow=/dev/nvidiactl rw",
            "Environment=CUDA_VISIBLE_DEVICES=GPU-a",
            "Environment=ASTRUMWEAVER_GPU_PREFLIGHT_MODE=isolated-access",
            "Environment=ASTRUMWEAVER_GPU_DEVICE_MAP=/etc/astrumweaver/gpu-device-map",
            "Environment=ASTRUMWEAVER_GPU_WORKER_CONFIG=/etc/astrumweaver/worker.toml",
            "Environment=ASTRUMWEAVER_GPU_DEVICE_MAP_COMMAND=/usr/local/libexec/astrumweaver/gpu-device-map",
            "Environment=ASTRUMWEAVER_NVIDIA_SMI=/usr/bin/nvidia-smi",
            "EnvironmentFiles=",
            "UnsetEnvironment=",
            f"ExecStartPre={effective_exec_start_pre}",
            "Requires=astrumweaver-worker-gpu-isolation-preflight.service",
            "After=network-online.target astrumweaver-worker-gpu-isolation-preflight.service",
        ],
    )
    driver = SystemdSetupDriver(
        worker_config_path=worker_config,
        runtime_manifest_path=manifest,
        gpu_uuid_file_path=expected,
        gpu_device_map_path=device_map,
        gpu_isolation_dropin_path=dropin,
        gpu_device_map_command=str(verifier),
        nvidia_smi=nvidia_smi,
        systemctl=systemctl,
    )

    result = driver.inspect(
        action(
            SetupActionKind.PREFLIGHT,
            payload={"provider_id": "ollama"},
        )
    )

    if should_verify:
        assert result.state is SetupActionState.SATISFIED
        assert "reviewed systemd isolation contract is configured" in result.detail
        assert "ExecStartPre must still prove isolated-access" in result.detail
        assert verifier_args.read_text(encoding="utf-8").splitlines()[:2] == [
            "--nvidia-smi",
            nvidia_smi,
        ]
    else:
        assert result.state is SetupActionState.BLOCKED
        assert "no verified service device isolation" in result.detail
        assert not verifier_args.exists()
