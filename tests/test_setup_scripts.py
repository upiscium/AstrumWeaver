from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup"
PREFLIGHT = ROOT / "libexec" / "gpu-preflight"


def assert_no_trailing_whitespace(text: str) -> None:
    for line_number, line in enumerate(text.splitlines(), start=1):
        assert line == line.rstrip(" \t"), (
            f"generated unit line {line_number} has trailing whitespace: {line!r}"
        )


def exec_start_line(unit: str) -> str:
    return next(line for line in unit.splitlines() if line.startswith("ExecStart="))


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
    assert exec_start_line(unit) == (
        "ExecStart=/usr/local/bin/astrumweaver-control "
        "--config /etc/astrumweaver/control.toml"
    )
    assert_no_trailing_whitespace(unit)


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


@pytest.mark.parametrize(
    ("script", "daemon", "config_name", "config_text", "unit_name"),
    (
        (
            "setup-control-plane.sh",
            "astrumweaver-control",
            "control.toml",
            '[control]\nlisten = "127.0.0.1:9000"\n',
            "astrumweaver-control.service",
        ),
        (
            "setup-gpu-worker.sh",
            "astrumweaver-worker",
            "worker.toml",
            '[worker]\ngpu_uuids = ["GPU-legacy"]\n',
            "astrumweaver-worker.service",
        ),
    ),
)
def test_setup_rejects_legacy_store_pinned_unit_without_mutation(
    tmp_path: Path,
    script: str,
    daemon: str,
    config_name: str,
    config_text: str,
    unit_name: str,
) -> None:
    config = tmp_path / config_name
    config.write_text(config_text, encoding="utf-8")
    staged = tmp_path / "root"
    legacy_executable = f"/nix/store/legacy-astrumweaver/bin/{daemon}"
    stable_executable = str(tmp_path / "profile" / "bin" / daemon)

    first = run(
        "bash",
        str(SETUP / script),
        "--config",
        str(config),
        "--executable",
        legacy_executable,
        "--root",
        str(staged),
    )
    assert first.returncode == 0, first.stderr

    unit = staged / "etc/systemd/system" / unit_name
    unit_before = unit.read_bytes()
    config_dest = staged / "etc/astrumweaver" / config_name
    config_before = config_dest.read_bytes()

    changed = run(
        "bash",
        str(SETUP / script),
        "--config",
        str(config),
        "--executable",
        stable_executable,
        "--root",
        str(staged),
    )

    assert changed.returncode != 0
    assert "legacy store-pinned" in changed.stderr
    assert unit.read_bytes() == unit_before
    assert config_dest.read_bytes() == config_before


def test_worker_setup_stages_exact_gpu_identity_and_is_idempotent(tmp_path: Path) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        "[worker]\n"
        "class = \"modern-single\"\n"
        "gpu_uuids = [\"GPU-example-b\", \"GPU-example-a\"]\n",
        encoding="utf-8",
    )
    staged = tmp_path / "root"
    command = [
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-worker",
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
    assert exec_start_line(unit) == (
        "ExecStart=/usr/local/bin/astrumweaver-worker "
        "--config /etc/astrumweaver/worker.toml"
    )
    assert_no_trailing_whitespace(unit)


def test_worker_setup_stages_runtime_manifest_and_wires_service(
    tmp_path: Path,
) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\nid = "worker-runtime"\nclass = "gpu-single"\n'
        'gpu_uuids = ["GPU-example-a"]\n',
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
    assert exec_start_line(unit) == (
        "ExecStart=/usr/local/bin/astrumweaver-worker "
        "--config /etc/astrumweaver/worker.toml "
        "--runtime-manifest /etc/astrumweaver/runtime-deployment.json"
    )
    assert_no_trailing_whitespace(unit)


@pytest.mark.parametrize("runtime", [False, True])
def test_worker_setup_converges_after_manual_store_to_profile_migration(
    tmp_path: Path, runtime: bool
) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\nid = "worker-migration"\nclass = "gpu-single"\n'
        'gpu_uuids = ["GPU-example-a"]\n',
        encoding="utf-8",
    )
    runtime_manifest = tmp_path / "runtime.json"
    runtime_manifest.write_text(
        '{"schema_version":"v1","provider_id":"test"}\n',
        encoding="utf-8",
    )
    staged = tmp_path / "root"
    legacy = "/nix/store/legacy-astrumweaver/bin/astrumweaver-worker"
    stable = "/nix/var/nix/profiles/astrumweaver-installer/bin/astrumweaver-worker"

    command = [
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
    ]
    if runtime:
        command.extend(["--runtime-manifest", str(runtime_manifest)])
    command.extend(["--executable", legacy, "--root", str(staged)])

    first = run(*command)
    assert first.returncode == 0, first.stderr
    unit_path = staged / "etc/systemd/system/astrumweaver-worker.service"
    legacy_unit = unit_path.read_text(encoding="utf-8")
    assert_no_trailing_whitespace(legacy_unit)

    migrated_unit = legacy_unit.replace(
        f"ExecStart={legacy}", f"ExecStart={stable}", 1
    )
    assert migrated_unit != legacy_unit
    assert migrated_unit.replace(stable, legacy, 1) == legacy_unit
    unit_path.write_text(migrated_unit, encoding="utf-8")

    rerun = list(command)
    executable_index = rerun.index("--executable") + 1
    rerun[executable_index] = stable
    converged = run(*rerun)

    assert converged.returncode == 0, converged.stderr
    assert unit_path.read_text(encoding="utf-8") == migrated_unit
    assert_no_trailing_whitespace(migrated_unit)


def test_worker_setup_stages_gpu_device_cgroup_isolation(
    tmp_path: Path,
) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\nid = "worker-isolated"\nclass = "gpu-single"\n'
        'gpu_uuids = ["GPU-example-a"]\n',
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
    assert_no_trailing_whitespace(
        (
            staged / "etc/systemd/system/astrumweaver-worker.service"
        ).read_text(encoding="utf-8")
    )
    assert_no_trailing_whitespace(dropin)
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
    assert_no_trailing_whitespace(verifier_unit)


def test_staged_gpu_isolation_requires_explicit_uuid_device_mapping(
    tmp_path: Path,
) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\ngpu_uuids = ["GPU-example-a"]\n',
        encoding="utf-8",
    )

    result = run(
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-worker",
        "--gpu-isolation",
        "on",
        "--root",
        str(tmp_path / "root"),
    )

    assert result.returncode != 0
    assert "staged GPU isolation requires --gpu-device" in result.stderr


def test_worker_setup_rejects_duplicate_gpu_identity(tmp_path: Path) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\ngpu_uuids = ["GPU-same", "GPU-same"]\n',
        encoding="utf-8",
    )

    result = run(
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-worker",
        "--root",
        str(tmp_path / "root"),
    )

    assert result.returncode != 0
    assert "duplicate" in result.stderr


def test_worker_setup_rejects_legacy_gpu_uuid_cli_authority(tmp_path: Path) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\ngpu_uuids = ["GPU-config"]\n',
        encoding="utf-8",
    )

    result = run(
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-worker",
        "--gpu-uuid",
        "GPU-other",
        "--root",
        str(tmp_path / "root"),
    )

    assert result.returncode != 0
    assert "unknown option: --gpu-uuid" in result.stderr


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


def test_gpu_device_map_verify_reads_reviewed_map_argument(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected"
    expected.write_text("GPU-b\n", encoding="utf-8")
    reviewed = tmp_path / "reviewed-map"
    reviewed.write_text("GPU-b=/dev/nvidia9\n", encoding="utf-8")

    bin_dir = tmp_path / "verify-bin"
    bin_dir.mkdir()
    executable = bin_dir / "nvidia-smi"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"--query-gpu=uuid,minor_number\" ]]; then\n"
        "  printf 'GPU-b, 7\\n'\n"
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
        "verify",
        str(expected),
        str(reviewed),
        env=env,
    )

    assert result.returncode != 0
    assert "GPU UUID to device-node mapping changed" in result.stderr
    assert "GPU device map file is missing" not in result.stderr


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
