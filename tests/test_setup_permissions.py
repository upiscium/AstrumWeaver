from __future__ import annotations

from pathlib import Path
import stat
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup"


# These regressions use --root sandboxes so they never create users or touch
# the host's /etc.  They verify the complete permission model that is
# available for a staged install (modes, shared identity, and fail-before-
# mutation).  Live ownership is exercised by the scripts' root-only chown
# branch; this test module intentionally does not assert it by mutating a
# host account or filesystem.
def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*args],
        check=False,
        capture_output=True,
        text=True,
    )


def control_command(
    config: Path,
    staged: Path,
    *,
    user: str = "astrumweaver",
    environment: Path | None = None,
) -> list[str]:
    command = [
        "bash",
        str(SETUP / "setup-control-plane.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-control",
        "--user",
        user,
        "--root",
        str(staged),
    ]
    if environment is not None:
        command[4:4] = ["--environment-file", str(environment)]
    return command


def worker_command(
    config: Path,
    staged: Path,
    *,
    user: str = "astrumweaver",
    environment: Path | None = None,
    runtime_manifest: Path | None = None,
    isolation: bool = False,
) -> list[str]:
    command = [
        "bash",
        str(SETUP / "setup-gpu-worker.sh"),
        "--config",
        str(config),
        "--executable",
        "/usr/local/bin/astrumweaver-worker",
        "--user",
        user,
        "--root",
        str(staged),
    ]
    if environment is not None:
        command[4:4] = ["--environment-file", str(environment)]
    if runtime_manifest is not None:
        command[4:4] = ["--runtime-manifest", str(runtime_manifest)]
    if isolation:
        command[4:4] = [
            "--gpu-isolation",
            "on",
            "--gpu-device",
            "GPU-test=/dev/nvidia3",
        ]
    return command


def write_control_inputs(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "control.toml"
    config.write_text("[control]\nlisten = \"127.0.0.1:9000\"\n", encoding="utf-8")
    environment = tmp_path / "control.env"
    environment.write_text("ASTRUMWEAVER_TEST=control\n", encoding="utf-8")
    return config, environment


def write_worker_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = tmp_path / "worker.toml"
    config.write_text(
        '[worker]\nid = "worker-permissions"\n'
        'class = "gpu-single"\ngpu_uuids = ["GPU-test"]\n',
        encoding="utf-8",
    )
    environment = tmp_path / "worker.env"
    environment.write_text("ASTRUMWEAVER_TEST=worker\n", encoding="utf-8")
    manifest = tmp_path / "runtime-deployment.json"
    manifest.write_text('{"schema_version":"v1","provider_id":"test"}\n', encoding="utf-8")
    return config, environment, manifest


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_fresh_coexisting_roles_share_identity_and_protected_modes(
    tmp_path: Path,
) -> None:
    control_config, control_env = write_control_inputs(tmp_path)
    worker_config, worker_env, manifest = write_worker_inputs(tmp_path)
    staged = tmp_path / "root"

    control = run(
        *control_command(
            control_config,
            staged,
            user="shared-service",
            environment=control_env,
        )
    )
    worker = run(
        *worker_command(
            worker_config,
            staged,
            user="shared-service",
            environment=worker_env,
            runtime_manifest=manifest,
            isolation=True,
        )
    )

    assert control.returncode == 0, control.stderr
    assert worker.returncode == 0, worker.stderr

    etc = staged / "etc/astrumweaver"
    state = staged / "var/lib/astrumweaver"
    assert mode(etc) == 0o750
    assert mode(state) == 0o750
    for protected in (
        etc / "control.toml",
        etc / "control.env",
        etc / "worker.toml",
        etc / "worker.env",
        etc / "runtime-deployment.json",
        etc / "gpu-uuids",
        etc / "gpu-device-map",
    ):
        assert mode(protected) == 0o640

    for role in ("control", "worker"):
        unit = (
            staged / "etc/systemd/system" / f"astrumweaver-{role}.service"
        ).read_text(encoding="utf-8")
        assert "User=shared-service" in unit
        assert "Group=shared-service" in unit


@pytest.mark.parametrize("first_role", ["control", "worker"])
def test_coexisting_roles_accept_same_custom_user_in_either_order(
    tmp_path: Path,
    first_role: str,
) -> None:
    control_config, _ = write_control_inputs(tmp_path)
    worker_config, _, _ = write_worker_inputs(tmp_path)
    staged = tmp_path / "root"
    commands = {
        "control": control_command(control_config, staged, user="same-user"),
        "worker": worker_command(worker_config, staged, user="same-user"),
    }

    first = run(*commands[first_role])
    second_role = "worker" if first_role == "control" else "control"
    second = run(*commands[second_role])

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr


@pytest.mark.parametrize("first_role", ["control", "worker"])
def test_different_custom_users_fail_before_shared_path_mutation(
    tmp_path: Path,
    first_role: str,
) -> None:
    control_config, _ = write_control_inputs(tmp_path)
    worker_config, _, _ = write_worker_inputs(tmp_path)
    staged = tmp_path / "root"
    commands = {
        "control": control_command(control_config, staged, user="control-user"),
        "worker": worker_command(worker_config, staged, user="worker-user"),
    }

    first = run(*commands[first_role])
    assert first.returncode == 0, first.stderr

    etc = staged / "etc/astrumweaver"
    etc.chmod(0o700)
    before_mode = mode(etc)
    second_role = "worker" if first_role == "control" else "control"
    second = run(*commands[second_role])

    assert second.returncode != 0
    assert "conflicting existing role unit identity" in second.stderr
    assert mode(etc) == before_mode
    missing_destination = etc / (
        "worker.toml" if second_role == "worker" else "control.toml"
    )
    assert not missing_destination.exists()


def test_same_role_user_migration_is_rejected_before_repair(tmp_path: Path) -> None:
    config, _ = write_control_inputs(tmp_path)
    staged = tmp_path / "root"

    installed = run(
        *control_command(config, staged, user="old-user")
    )
    assert installed.returncode == 0, installed.stderr

    etc = staged / "etc/astrumweaver"
    etc.chmod(0o700)
    migrated = run(*control_command(config, staged, user="new-user"))

    assert migrated.returncode != 0
    assert "conflicting existing role unit identity" in migrated.stderr
    assert mode(etc) == 0o700
    unit = (
        staged / "etc/systemd/system/astrumweaver-control.service"
    ).read_text(encoding="utf-8")
    assert "User=old-user" in unit
    assert "User=new-user" not in unit


def test_ambiguous_existing_role_identity_fails_closed(tmp_path: Path) -> None:
    config, _ = write_control_inputs(tmp_path)
    staged = tmp_path / "root"
    unit_dir = staged / "etc/systemd/system"
    unit_dir.mkdir(parents=True)
    (unit_dir / "astrumweaver-worker.service").write_text(
        "[Service]\nUser=shared-service\n",
        encoding="utf-8",
    )

    result = run(*control_command(config, staged, user="shared-service"))

    assert result.returncode != 0
    assert "cannot safely determine service identity" in result.stderr
    assert not (staged / "etc/astrumweaver/control.toml").exists()


def test_identity_dropin_is_rejected_before_mutation(tmp_path: Path) -> None:
    config, _ = write_control_inputs(tmp_path)
    staged = tmp_path / "root"
    dropins = staged / "etc/systemd/system/astrumweaver-worker.service.d"
    dropins.mkdir(parents=True)
    (dropins / "identity.conf").write_text("[Service]\nUser=other-user\n")
    result = run(*control_command(config, staged))
    assert result.returncode != 0
    assert "identity drop-ins" in result.stderr
    assert not (staged / "etc/astrumweaver").exists()


def test_retry_repairs_shared_directory_and_retained_protected_file_modes(
    tmp_path: Path,
) -> None:
    worker_config, worker_env, manifest = write_worker_inputs(tmp_path)
    staged = tmp_path / "root"
    command = worker_command(
        worker_config,
        staged,
        user="retry-user",
        environment=worker_env,
        runtime_manifest=manifest,
        isolation=True,
    )

    first = run(*command)
    assert first.returncode == 0, first.stderr

    etc = staged / "etc/astrumweaver"
    state = staged / "var/lib/astrumweaver"
    protected = [
        etc / "worker.toml",
        etc / "worker.env",
        etc / "runtime-deployment.json",
        etc / "gpu-uuids",
        etc / "gpu-device-map",
    ]
    etc.chmod(0o777)
    state.chmod(0o700)
    for path in protected:
        path.chmod(0o600)

    second = run(*command)

    assert second.returncode == 0, second.stderr
    assert mode(etc) == 0o750
    assert mode(state) == 0o750
    assert all(mode(path) == 0o640 for path in protected)
