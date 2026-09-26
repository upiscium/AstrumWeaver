from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup"
PREFLIGHT = ROOT / "libexec" / "gpu-preflight"


def run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_control_setup_stages_existing_node_idempotently(tmp_path: Path) -> None:
    config = tmp_path / "control.toml"
    config.write_text("[control]\nlisten = \"127.0.0.1:9000\"\n", encoding="utf-8")
    staged = tmp_path / "root"

    command = [
        "bash",
        str(SETUP / "setup-control-plane.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-control",
        "--root",
        str(staged),
    ]

    first = run(*command)
    second = run(*command)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert (
        staged / "etc/astrumweaver/control.toml"
    ).read_text(encoding="utf-8") == config.read_text(encoding="utf-8")
    unit = (staged / "etc/systemd/system/astrumweaver-control.service").read_text(
        encoding="utf-8"
    )
    assert "ExecStart=/usr/local/bin/astrumweaver-control --config /etc/astrumweaver/control.toml" in unit


def test_control_setup_refuses_unreviewed_config_overwrite(tmp_path: Path) -> None:
    config = tmp_path / "control.toml"
    config.write_text("[control]\nvalue = 1\n", encoding="utf-8")
    staged = tmp_path / "root"
    command = [
        "bash",
        str(SETUP / "setup-control-plane.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-control",
        "--root",
        str(staged),
    ]
    assert run(*command).returncode == 0

    config.write_text("[control]\nvalue = 2\n", encoding="utf-8")
    changed = run(*command)

    assert changed.returncode != 0
    assert "refusing overwrite" in changed.stderr


def test_worker_setup_stages_exact_gpu_identity_and_is_idempotent(tmp_path: Path) -> None:
    config = tmp_path / "worker.toml"
    config.write_text("[worker]\nclass = \"modern-single\"\n", encoding="utf-8")
    staged = tmp_path / "root"
    command = [
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-worker",
        "--gpu-uuid",
        "GPU-example-b",
        "--gpu-uuid",
        "GPU-example-a",
        "--root",
        str(staged),
    ]

    first = run(*command)
    second = run(*command)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert (staged / "etc/astrumweaver/gpu-uuids").read_text(
        encoding="utf-8"
    ) == "GPU-example-a\nGPU-example-b\n"
    assert (staged / "usr/local/libexec/astrumweaver/gpu-preflight").exists()
    unit = (staged / "etc/systemd/system/astrumweaver-worker.service").read_text(
        encoding="utf-8"
    )
    assert "gpu-preflight /etc/astrumweaver/gpu-uuids" in unit
    assert "ExecStart=/usr/local/bin/astrumweaver-worker --config /etc/astrumweaver/worker.toml" in unit


def test_worker_setup_stages_runtime_manifest_and_wires_service(
    tmp_path: Path,
) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\nid = "worker-runtime"\nclass = "gpu-single"\n',
        encoding="utf-8",
    )
    runtime_manifest = tmp_path / "runtime.json"
    runtime_manifest.write_text(
        '{"schema_version":"v1","provider_id":"ollama"}\n',
        encoding="utf-8",
    )
    staged = tmp_path / "root"

    result = run(
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
        "--runtime-manifest",
        str(runtime_manifest),
        "--executable",
        "/usr/local/bin/astrumweaver-worker",
        "--gpu-uuid",
        "GPU-example-a",
        "--root",
        str(staged),
    )

    assert result.returncode == 0, result.stderr
    deployed = (
        staged / "etc/astrumweaver/runtime-deployment.json"
    )
    assert deployed.read_text(encoding="utf-8") == runtime_manifest.read_text(
        encoding="utf-8"
    )
    unit = (
        staged
        / "etc/systemd/system/astrumweaver-worker.service"
    ).read_text(encoding="utf-8")
    assert (
        "--runtime-manifest /etc/astrumweaver/runtime-deployment.json"
        in unit
    )


def test_worker_setup_stages_gpu_device_cgroup_isolation(
    tmp_path: Path,
) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\nid = "worker-isolated"\nclass = "gpu-single"\n',
        encoding="utf-8",
    )
    staged = tmp_path / "root"

    result = run(
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-worker",
        "--gpu-uuid",
        "GPU-example-a",
        "--gpu-isolation",
        "on",
        "--gpu-device",
        "GPU-example-a=/dev/nvidia3",
        "--root",
        str(staged),
    )

    assert result.returncode == 0, result.stderr
    assert (
        staged / "etc/astrumweaver/gpu-device-map"
    ).read_text(encoding="utf-8") == "GPU-example-a=/dev/nvidia3\n"

    dropin = (
        staged
        / "etc/systemd/system/astrumweaver-worker.service.d/10-gpu-isolation.conf"
    ).read_text(encoding="utf-8")
    assert "DevicePolicy=closed" in dropin
    assert "DeviceAllow=/dev/nvidia3 rw" in dropin
    assert "Environment=CUDA_VISIBLE_DEVICES=GPU-example-a" in dropin
    assert (
        "Requires=astrumweaver-worker-gpu-isolation-preflight.service"
        in dropin
    )

    verifier_unit = (
        staged
        / "etc/systemd/system/astrumweaver-worker-gpu-isolation-preflight.service"
    ).read_text(encoding="utf-8")
    assert "gpu-device-map verify" in verifier_unit


def test_staged_gpu_isolation_requires_explicit_uuid_device_mapping(
    tmp_path: Path,
) -> None:
    config = tmp_path / "worker.toml"
    config.write_text("[worker]\n", encoding="utf-8")

    result = run(
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-worker",
        "--gpu-uuid",
        "GPU-example-a",
        "--gpu-isolation",
        "on",
        "--root",
        str(tmp_path / "root"),
    )

    assert result.returncode != 0
    assert "staged GPU isolation requires --gpu-device" in result.stderr


def test_worker_setup_rejects_duplicate_gpu_identity(tmp_path: Path) -> None:
    config = tmp_path / "worker.toml"
    config.write_text("[worker]\n", encoding="utf-8")

    result = run(
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-worker",
        "--gpu-uuid",
        "GPU-same",
        "--gpu-uuid",
        "GPU-same",
        "--root",
        str(tmp_path / "root"),
    )

    assert result.returncode != 0
    assert "duplicate" in result.stderr


def fake_nvidia_smi(tmp_path: Path, output: str) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "nvidia-smi"
    executable.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' " + " ".join(repr(line) for line in output.splitlines()) + "\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    return env


def test_gpu_device_map_discovers_selected_uuid_to_minor_mapping(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected"
    expected.write_text("GPU-b\n", encoding="utf-8")

    bin_dir = tmp_path / "map-bin"
    bin_dir.mkdir()
    executable = bin_dir / "nvidia-smi"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"--query-gpu=uuid,minor_number\" ]]; then\n"
        "  printf 'GPU-a, 2\\nGPU-b, 7\\n'\n"
        "  exit 0\n"
        "fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"

    result = run(
        "bash",
        str(ROOT / "libexec" / "gpu-device-map"),
        "discover",
        str(expected),
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "GPU-b=/dev/nvidia7\n"


def test_gpu_preflight_requires_exact_set_equality(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    expected.write_text("GPU-a\nGPU-b\n", encoding="utf-8")
    env = fake_nvidia_smi(tmp_path, "GPU-b\nGPU-a")

    matched = run("bash", str(PREFLIGHT), str(expected), env=env)

    assert matched.returncode == 0, matched.stderr
    assert "ok count=2" in matched.stdout

    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()
    extra_env = fake_nvidia_smi(extra_dir, "GPU-a\nGPU-b\nGPU-c")
    mismatched = run("bash", str(PREFLIGHT), str(expected), env=extra_env)

    assert mismatched.returncode != 0
    assert "GPU UUID set mismatch" in mismatched.stderr


def test_setup_layer_contains_no_proxmox_creation_commands() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            SETUP / "setup-control-plane.sh",
            SETUP / "setup-gpu-worker.sh",
        )
    )

    assert "qm create" not in text
    assert "pct create" not in text
