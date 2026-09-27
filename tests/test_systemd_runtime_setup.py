from __future__ import annotations

import json
from pathlib import Path

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


def test_systemd_gpu_preflight_accepts_host_superset_with_verified_device_isolation(
    tmp_path: Path,
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
        "Environment=CUDA_VISIBLE_DEVICES=GPU-a\n",
        encoding="utf-8",
    )
    verifier = tmp_path / "gpu-device-map-verify"
    verifier.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    verifier.chmod(0o755)

    superset_dir = tmp_path / "isolated-superset"
    superset_dir.mkdir()
    driver = SystemdSetupDriver(
        worker_config_path=worker_config,
        runtime_manifest_path=manifest,
        gpu_uuid_file_path=expected,
        gpu_device_map_path=device_map,
        gpu_isolation_dropin_path=dropin,
        gpu_device_map_command=str(verifier),
        nvidia_smi=fake_nvidia_smi(
            superset_dir,
            "GPU-a\nGPU-b",
        ),
    )

    result = driver.inspect(
        action(
            SetupActionKind.PREFLIGHT,
            payload={"provider_id": "ollama"},
        )
    )

    assert result.state is SetupActionState.SATISFIED
    assert "device-cgroup isolation is verified" in result.detail
    assert "ExecStartPre must still prove the exact set" in result.detail
