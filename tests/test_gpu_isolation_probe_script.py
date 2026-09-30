from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "libexec" / "gpu-isolation-probe"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    expected = tmp_path / "expected"
    reviewed = tmp_path / "map"
    config = tmp_path / "worker.toml"
    expected.write_text("GPU-selected\n", encoding="utf-8")
    reviewed.write_text("GPU-selected=/dev/nvidia0\n", encoding="utf-8")
    config.write_text(
        '[worker]\ngpu_uuids = ["GPU-selected"]\n',
        encoding="utf-8",
    )
    return expected, reviewed, config


def _environment(
    tmp_path: Path,
    *,
    systemd_run_body: str,
) -> dict[str, str]:
    mapper = tmp_path / "gpu-device-map"
    nvidia_smi = tmp_path / "nvidia-smi"
    systemd_run = tmp_path / "systemd-run"

    _write_executable(
        mapper,
        """#!/usr/bin/env bash
set -eu
exit 0
""",
    )
    _write_executable(
        nvidia_smi,
        """#!/usr/bin/env bash
exit 0
""",
    )
    _write_executable(
        systemd_run,
        "#!/usr/bin/env bash\nset -eu\n" + systemd_run_body,
    )

    env = dict(os.environ)
    env.update(
        {
            "ASTRUMWEAVER_GPU_DEVICE_MAP_COMMAND": str(mapper),
            "ASTRUMWEAVER_NVIDIA_SMI": str(nvidia_smi),
            "ASTRUMWEAVER_SYSTEMD_RUN": str(systemd_run),
        }
    )
    return env


def test_probe_accepts_effective_device_denial(tmp_path: Path) -> None:
    expected, reviewed, config = _inputs(tmp_path)
    env = _environment(
        tmp_path,
        systemd_run_body="""
printf '%s\n' "$@" >"$PROBE_ARGS"
exit 0
""",
    )
    args_path = tmp_path / "args"
    env["PROBE_ARGS"] = str(args_path)

    result = subprocess.run(
        [
            "bash",
            str(PROBE),
            str(expected),
            str(reviewed),
            str(config),
            "GPU-selected",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "enforceable selected_count=1" in result.stdout
    args = args_path.read_text(encoding="utf-8")
    assert "--property=DevicePolicy=closed" in args
    assert "--property=DeviceAllow=/dev/nvidia0 rw" in args
    assert "--setenv=CUDA_VISIBLE_DEVICES=GPU-selected" in args
    assert "verify-isolated-access" in args


def test_probe_reports_isolation_unavailable_without_raw_private_output(
    tmp_path: Path,
) -> None:
    expected, reviewed, config = _inputs(tmp_path)
    env = _environment(
        tmp_path,
        systemd_run_body="""
echo '[astrumweaver-gpu-device-map] ISOLATION_UNAVAILABLE: internal detail' >&2
exit 1
""",
    )

    result = subprocess.run(
        [
            "bash",
            str(PROBE),
            str(expected),
            str(reviewed),
            str(config),
            "GPU-selected",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 3
    assert "ISOLATION_UNAVAILABLE" in result.stderr
    assert "narrow guest-visible GPU exposure externally" in result.stderr
    assert "internal detail" not in result.stderr
    assert "GPU-selected" not in result.stderr
    assert "/dev/nvidia0" not in result.stderr


def test_probe_keeps_unclassified_systemd_failure_fail_closed(
    tmp_path: Path,
) -> None:
    expected, reviewed, config = _inputs(tmp_path)
    env = _environment(
        tmp_path,
        systemd_run_body="""
echo 'unexpected runtime failure' >&2
exit 1
""",
    )

    result = subprocess.run(
        [
            "bash",
            str(PROBE),
            str(expected),
            str(reviewed),
            str(config),
            "GPU-selected",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 1
    assert "capability probe failed" in result.stderr
    assert "unexpected runtime failure" not in result.stderr
