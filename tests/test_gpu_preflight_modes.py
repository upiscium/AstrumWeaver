from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "libexec" / "gpu-preflight"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_shell_preflight_invokes_isolated_access_mapper(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    reviewed = tmp_path / "map"
    config = tmp_path / "worker.toml"
    mapper = tmp_path / "gpu-device-map"
    nvidia_smi = tmp_path / "nvidia-smi"
    args_file = tmp_path / "mapper-args"

    expected.write_text("GPU-selected\n", encoding="utf-8")
    reviewed.write_text("GPU-selected=/dev/nvidia0\n", encoding="utf-8")
    config.write_text(
        '[worker]\ngpu_uuids = ["GPU-selected"]\n',
        encoding="utf-8",
    )
    _write_executable(
        mapper,
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" >\"$ARGS_FILE\"\nexit 0\n",
    )
    _write_executable(nvidia_smi, "#!/usr/bin/env bash\nexit 0\n")

    env = dict(os.environ)
    env.update(
        {
            "ASTRUMWEAVER_GPU_PREFLIGHT_MODE": "isolated-access",
            "ASTRUMWEAVER_GPU_DEVICE_MAP": str(reviewed),
            "ASTRUMWEAVER_GPU_WORKER_CONFIG": str(config),
            "ASTRUMWEAVER_GPU_DEVICE_MAP_COMMAND": str(mapper),
            "ASTRUMWEAVER_NVIDIA_SMI": str(nvidia_smi),
            "CUDA_VISIBLE_DEVICES": "GPU-selected",
            "ARGS_FILE": str(args_file),
        }
    )

    result = subprocess.run(
        ["bash", str(PREFLIGHT), str(expected)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "isolated-access ok count=1" in result.stdout
    assert args_file.read_text(encoding="utf-8").splitlines() == [
        "--nvidia-smi",
        str(nvidia_smi),
        "verify-isolated-access",
        str(expected),
        str(reviewed),
        str(config),
    ]


def test_shell_preflight_isolated_access_requires_explicit_map(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected"
    expected.write_text("GPU-selected\n", encoding="utf-8")
    env = dict(os.environ)
    env["ASTRUMWEAVER_GPU_PREFLIGHT_MODE"] = "isolated-access"
    env.pop("ASTRUMWEAVER_GPU_DEVICE_MAP", None)

    result = subprocess.run(
        ["bash", str(PREFLIGHT), str(expected)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "reviewed GPU map is unavailable" in result.stderr


def test_shell_preflight_rejects_unknown_mode(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    expected.write_text("GPU-selected\n", encoding="utf-8")
    nvidia_smi = tmp_path / "nvidia-smi"
    _write_executable(nvidia_smi, "#!/usr/bin/env bash\nexit 0\n")
    env = dict(os.environ)
    env["ASTRUMWEAVER_GPU_PREFLIGHT_MODE"] = "unknown"
    env["ASTRUMWEAVER_NVIDIA_SMI"] = str(nvidia_smi)

    result = subprocess.run(
        ["bash", str(PREFLIGHT), str(expected)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "exact-visible or isolated-access" in result.stderr
