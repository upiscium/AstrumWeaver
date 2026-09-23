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
