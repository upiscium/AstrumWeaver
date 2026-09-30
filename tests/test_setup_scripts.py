from __future__ import annotations

import os
from pathlib import Path
import subprocess
import stat
from types import SimpleNamespace
from typing import Callable

import pytest

from astrumweaver.validation.gpu_mapping import (
    GpuMappingError,
    _load_reviewed_map,
    _select_mapping,
    discover_gpu_mapping,
)

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup"
PREFLIGHT = ROOT / "libexec" / "gpu-preflight"
LEGACY_GPU_DEVICE_MAP = ROOT / "tests/fixtures/gpu-device-map-v48"


def nix_patched_bash_script(source: Path) -> bytes:
    lines = source.read_bytes().splitlines(keepends=True)
    assert lines
    lines[0] = (
        b"#!/nix/store/"
        + b"a" * 32
        + b"-bash-5.2p37/bin/bash\n"
    )
    return b"".join(lines)


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
    assert (staged / "usr/local/libexec/astrumweaver/gpu-isolation-probe").exists()
    assert (staged / "usr/local/libexec/astrumweaver/gpu_mapping.py").exists()
    unit = (staged / "etc/systemd/system/astrumweaver-worker.service").read_text(
        encoding="utf-8"
    )
    assert "gpu-preflight /etc/astrumweaver/gpu-uuids" in unit
    assert exec_start_line(unit) == (
        "ExecStart=/usr/local/bin/astrumweaver-worker "
        "--config /etc/astrumweaver/worker.toml"
    )
    assert_no_trailing_whitespace(unit)


def test_worker_setup_upgrades_reviewed_v48_gpu_mapper_to_canonical_helper(
    tmp_path: Path,
) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\ngpu_uuids = ["GPU-legacy"]\n',
        encoding="utf-8",
    )
    staged = tmp_path / "root"
    helper = staged / "usr/local/libexec/astrumweaver/gpu-device-map"
    helper.parent.mkdir(parents=True)
    helper.write_bytes(LEGACY_GPU_DEVICE_MAP.read_bytes())

    result = run(
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-worker",
        "--root",
        str(staged),
    )

    assert result.returncode == 0, result.stderr
    assert helper.read_bytes() == (ROOT / "libexec/gpu-device-map").read_bytes()
    assert b"/nix/store/" not in helper.read_bytes()
    assert (staged / "usr/local/libexec/astrumweaver/gpu-preflight").exists()

    discover, _, _, _, _ = fake_gpu_mapping_host(
        tmp_path,
        visible=("GPU-legacy",),
        proc_entries=(),
        valid_minors=("0",),
        primary_entries=(("GPU-legacy", "0"),),
    )
    assert _select_mapping(discover(), ("GPU-legacy",)) == (
        ("GPU-legacy", "/dev/nvidia0"),
    )
    assert os.access(helper, os.X_OK)


@pytest.mark.parametrize(
    ("helper_name", "reviewed_source"),
    (
        ("gpu-preflight", PREFLIGHT),
        ("gpu-device-map", LEGACY_GPU_DEVICE_MAP),
    ),
)
def test_worker_setup_upgrades_nix_patched_reviewed_gpu_helper(
    tmp_path: Path,
    helper_name: str,
    reviewed_source: Path,
) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\ngpu_uuids = ["GPU-legacy"]\n',
        encoding="utf-8",
    )
    staged = tmp_path / "root"
    helper = staged / "usr/local/libexec/astrumweaver" / helper_name
    helper.parent.mkdir(parents=True)
    helper.write_bytes(nix_patched_bash_script(reviewed_source))

    result = run(
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-worker",
        "--root",
        str(staged),
    )

    assert result.returncode == 0, result.stderr
    assert helper.read_bytes() == (ROOT / "libexec" / helper_name).read_bytes()
    assert b"/nix/store/" not in helper.read_bytes()


@pytest.mark.parametrize(
    "helper_name",
    ("gpu-preflight", "gpu-device-map"),
)
def test_worker_setup_rejects_operator_modified_nix_patched_gpu_helper(
    tmp_path: Path,
    helper_name: str,
) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\ngpu_uuids = ["GPU-legacy"]\n',
        encoding="utf-8",
    )
    staged = tmp_path / "root"
    helper = staged / "usr/local/libexec/astrumweaver" / helper_name
    helper.parent.mkdir(parents=True)
    reviewed_source = (
        PREFLIGHT if helper_name == "gpu-preflight" else LEGACY_GPU_DEVICE_MAP
    )
    helper.write_bytes(
        nix_patched_bash_script(reviewed_source) + b"# operator change\n"
    )
    before = helper.read_bytes()

    result = run(
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-worker",
        "--root",
        str(staged),
    )

    assert result.returncode != 0
    assert "destination differs; refusing overwrite" in result.stderr
    assert helper.read_bytes() == before


@pytest.mark.parametrize(
    "shebang",
    (
        b"#!/usr/local/bin/bash\n",
        b"#!/nix/store/" + b"a" * 32 + b"-zsh-5.9/bin/zsh\n",
    ),
)
def test_worker_setup_rejects_unreviewed_gpu_helper_interpreter(
    tmp_path: Path,
    shebang: bytes,
) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\ngpu_uuids = ["GPU-legacy"]\n',
        encoding="utf-8",
    )
    staged = tmp_path / "root"
    helper = staged / "usr/local/libexec/astrumweaver/gpu-preflight"
    helper.parent.mkdir(parents=True)
    lines = PREFLIGHT.read_bytes().splitlines(keepends=True)
    lines[0] = shebang
    helper.write_bytes(b"".join(lines))
    before = helper.read_bytes()

    result = run(
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-worker",
        "--root",
        str(staged),
    )

    assert result.returncode != 0
    assert "destination differs; refusing overwrite" in result.stderr
    assert helper.read_bytes() == before


def test_worker_setup_rejects_modified_legacy_gpu_mapper(tmp_path: Path) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\ngpu_uuids = ["GPU-legacy"]\n',
        encoding="utf-8",
    )
    staged = tmp_path / "root"
    helper = staged / "usr/local/libexec/astrumweaver/gpu-device-map"
    helper.parent.mkdir(parents=True)
    helper.write_bytes(LEGACY_GPU_DEVICE_MAP.read_bytes() + b"# operator change\n")
    before = helper.read_bytes()

    result = run(
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-worker",
        "--root",
        str(staged),
    )

    assert result.returncode != 0
    assert "destination differs; refusing overwrite" in result.stderr
    assert helper.read_bytes() == before


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


def test_gpu_isolation_off_rejects_explicit_mapping_for_staged_installs(
    tmp_path: Path,
) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\ngpu_uuids = ["GPU-example-a"]\n',
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
        "off",
        "--gpu-device",
        "GPU-example-a=/dev/nvidia0",
        "--root",
        str(staged),
    )

    assert result.returncode != 0
    assert "--gpu-device cannot be used when --gpu-isolation off" in result.stderr
    assert not staged.exists()


def test_staged_gpu_isolation_off_rejects_existing_isolation_state(
    tmp_path: Path,
) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\ngpu_uuids = ["GPU-example-a"]\n',
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
    installed = run(
        *command,
        "--gpu-isolation",
        "on",
        "--gpu-device",
        "GPU-example-a=/dev/nvidia0",
    )
    assert installed.returncode == 0, installed.stderr

    map_file = staged / "etc/astrumweaver/gpu-device-map"
    isolation_unit = (
        staged
        / "etc/systemd/system/astrumweaver-worker-gpu-isolation-preflight.service"
    )
    isolation_dropin = (
        staged
        / "etc/systemd/system/astrumweaver-worker.service.d/10-gpu-isolation.conf"
    )
    before = tuple(
        path.read_bytes() for path in (map_file, isolation_unit, isolation_dropin)
    )

    disabled = run(*command, "--gpu-isolation", "off")

    assert disabled.returncode != 0
    assert "existing GPU isolation state requires reviewed removal" in disabled.stderr
    assert before == tuple(
        path.read_bytes() for path in (map_file, isolation_unit, isolation_dropin)
    )


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


def fake_gpu_mapping_host(
    tmp_path: Path,
    *,
    visible: tuple[str, ...],
    proc_entries: tuple[tuple[str, str], ...],
    valid_minors: tuple[str, ...],
    primary_entries: tuple[tuple[str, str], ...] | None = None,
    primary_failure: tuple[int, str] | None = None,
) -> tuple[
    Callable[[], tuple[tuple[str, str], ...]],
    Path,
    Path,
    Path,
    dict[str, object],
]:

    proc_root = tmp_path / "proc-gpus"
    for index, (uuid, minor) in enumerate(proc_entries):
        information_dir = proc_root / f"gpu{index}"
        information_dir.mkdir(parents=True)
        (information_dir / "information").write_text(
            f"GPU UUID : {uuid}\nDevice Minor : {minor}\n",
            encoding="utf-8",
        )

    device_root = tmp_path / "fake-dev"
    proc_devices = tmp_path / "proc-devices"
    proc_devices.write_text(
        "Character devices:\n 5 nvidia-frontend\nBlock devices:\n",
        encoding="utf-8",
    )
    device_metadata: dict[str, object] = {
        "major": 5,
        "minor_overrides": {},
    }

    def run_query(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        fields = args[1]
        if fields == "--query-gpu=uuid":
            return subprocess.CompletedProcess(
                args,
                0,
                "".join(f"{uuid}\n" for uuid in visible),
                "",
            )
        if fields == "--query-gpu=uuid,minor_number":
            if primary_entries is not None:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    "".join(f"{uuid}, {minor}\n" for uuid, minor in primary_entries),
                    "",
                )
            code, stderr = primary_failure or (
                2,
                "Unknown field: minor_number\n",
            )
            return subprocess.CompletedProcess(args, code, "", stderr)
        return subprocess.CompletedProcess(args, 2, "", "invalid query")

    def device_stat(path: Path) -> os.stat_result:
        minor_text = path.name.removeprefix("nvidia")
        if minor_text not in valid_minors:
            raise FileNotFoundError(path)
        minor_overrides = device_metadata["minor_overrides"]
        minor = minor_overrides.get(minor_text, int(minor_text))
        return SimpleNamespace(
            st_mode=stat.S_IFCHR | 0o660,
            st_rdev=os.makedev(device_metadata["major"], minor),
        )

    def discover() -> tuple[tuple[str, str], ...]:
        return discover_gpu_mapping(
            nvidia_smi="fake-nvidia-smi",
            proc_root=proc_root,
            proc_devices=proc_devices,
            device_root=device_root,
            run=run_query,
            device_stat=device_stat,
        )

    return discover, proc_root, device_root, proc_devices, device_metadata


def test_gpu_device_map_discovers_selected_uuid_to_proc_minor_mapping(
    tmp_path: Path,
) -> None:
    discover, _, _, _, _ = fake_gpu_mapping_host(
        tmp_path,
        visible=("GPU-a", "GPU-b"),
        proc_entries=(
            ("GPU-a", "0"),
            ("GPU-b", "2"),
            ("GPU-stale", "99"),
        ),
        valid_minors=("0", "2"),
        primary_failure=(2, "Unknown field: minor_number\n"),
    )

    assert _select_mapping(discover(), ("GPU-b",)) == (
        ("GPU-b", "/dev/nvidia2"),
    )


def test_gpu_device_map_uses_supported_minor_query_without_proc_fallback(
    tmp_path: Path,
) -> None:
    discover, proc_root, _, _, _ = fake_gpu_mapping_host(
        tmp_path,
        visible=("GPU-a", "GPU-b"),
        proc_entries=(),
        valid_minors=("0", "2"),
        primary_entries=(("GPU-a", "0"), ("GPU-b", "2")),
    )

    assert _select_mapping(discover(), ("GPU-b",)) == (
        ("GPU-b", "/dev/nvidia2"),
    )
    assert not proc_root.exists()


def test_gpu_device_map_fails_closed_on_primary_minor_query_error(
    tmp_path: Path,
) -> None:
    discover, _, _, _, _ = fake_gpu_mapping_host(
        tmp_path,
        visible=("GPU-b",),
        proc_entries=(("GPU-b", "2"),),
        valid_minors=("2",),
        primary_failure=(9, "Failed to initialize NVML\n"),
    )

    with pytest.raises(GpuMappingError, match="UUID/minor query failed"):
        discover()


def test_gpu_device_map_does_not_treat_silent_invalid_query_as_unsupported(
    tmp_path: Path,
) -> None:
    discover, _, _, _, _ = fake_gpu_mapping_host(
        tmp_path,
        visible=("GPU-b",),
        proc_entries=(("GPU-b", "2"),),
        valid_minors=("2",),
        primary_failure=(2, ""),
    )

    with pytest.raises(GpuMappingError, match="UUID/minor query failed"):
        discover()


def test_gpu_device_map_fails_when_visible_uuid_has_invalid_device_node(
    tmp_path: Path,
) -> None:
    discover, _, _, _, _ = fake_gpu_mapping_host(
        tmp_path,
        visible=("GPU-a", "GPU-b", "GPU-invalid"),
        proc_entries=(
            ("GPU-a", "0"),
            ("GPU-b", "2"),
            ("GPU-invalid", "99"),
        ),
        valid_minors=("0", "2"),
    )

    with pytest.raises(GpuMappingError, match="visible GPU device node is unavailable"):
        discover()


@pytest.mark.parametrize(
    ("registered_major", "actual_minor"),
    ((4, 0), (5, 2)),
)
def test_gpu_device_map_rejects_device_with_wrong_major_or_minor(
    tmp_path: Path,
    registered_major: int,
    actual_minor: int,
) -> None:
    discover, _, _, proc_devices, device_metadata = fake_gpu_mapping_host(
        tmp_path,
        visible=("GPU-a",),
        proc_entries=(("GPU-a", "0"),),
        valid_minors=("0",),
    )
    device_metadata["minor_overrides"] = {"0": actual_minor}
    proc_devices.write_text(
        "Character devices:\n"
        f" {registered_major} nvidia-frontend\n"
        "Block devices:\n",
        encoding="utf-8",
    )

    with pytest.raises(
        GpuMappingError,
        match="does not match the NVIDIA major/minor",
    ):
        discover()


def test_gpu_device_map_fails_when_visible_minors_are_duplicated(
    tmp_path: Path,
) -> None:
    discover, _, _, _, _ = fake_gpu_mapping_host(
        tmp_path,
        visible=("GPU-a", "GPU-b"),
        proc_entries=(("GPU-a", "0"), ("GPU-b", "0")),
        valid_minors=("0",),
    )

    with pytest.raises(GpuMappingError, match="reuse one device minor"):
        discover()


def test_gpu_device_map_fails_when_visible_uuid_has_duplicate_proc_records(
    tmp_path: Path,
) -> None:
    discover, _, _, _, _ = fake_gpu_mapping_host(
        tmp_path,
        visible=("GPU-a", "GPU-b"),
        proc_entries=(
            ("GPU-a", "0"),
            ("GPU-a", "2"),
            ("GPU-b", "2"),
        ),
        valid_minors=("0", "2"),
    )

    with pytest.raises(GpuMappingError, match="exactly one device mapping"):
        discover()


def test_gpu_device_map_fails_when_visible_proc_minor_is_malformed(
    tmp_path: Path,
) -> None:
    discover, _, _, _, _ = fake_gpu_mapping_host(
        tmp_path,
        visible=("GPU-a", "GPU-b"),
        proc_entries=(("GPU-a", "not-a-number"), ("GPU-b", "2")),
        valid_minors=("2",),
    )

    with pytest.raises(GpuMappingError, match="invalid device minor"):
        discover()


def test_gpu_device_map_fails_when_visible_uuid_lacks_proc_mapping(
    tmp_path: Path,
) -> None:
    discover, _, _, _, _ = fake_gpu_mapping_host(
        tmp_path,
        visible=("GPU-a", "GPU-b"),
        proc_entries=(("GPU-a", "0"),),
        valid_minors=("0",),
    )

    with pytest.raises(GpuMappingError, match="exactly one device mapping"):
        discover()


def test_gpu_device_map_verify_reads_reviewed_map_argument(
    tmp_path: Path,
) -> None:
    discover, _, _, _, _ = fake_gpu_mapping_host(
        tmp_path,
        visible=("GPU-b",),
        proc_entries=(("GPU-b", "2"),),
        valid_minors=("2",),
    )
    reviewed = tmp_path / "reviewed-map"
    reviewed.write_text("GPU-b=/dev/nvidia9\n", encoding="utf-8")
    selected = _select_mapping(discover(), ("GPU-b",))
    assert selected != _load_reviewed_map(reviewed)


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
