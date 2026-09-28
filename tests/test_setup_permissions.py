from __future__ import annotations

from pathlib import Path
import stat
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup"


# These regressions use --root sandboxes so they never create users or touch
# the host's /etc.  The NixOS acceptance test covers live ownership and
# account/group membership; this module covers the staged-install validation
# and fail-before-mutation paths.
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
    user: str = "astrumweaver-control",
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


def assert_control_layout(staged: Path, user: str) -> None:
    etc = staged / "etc/astrumweaver"
    state = staged / "var/lib/astrumweaver-control"
    assert mode(etc) == 0o710
    assert mode(state) == 0o750
    for path in (etc / "control.toml", etc / "control.env"):
        assert mode(path) == 0o640

    unit = (staged / "etc/systemd/system/astrumweaver-control.service").read_text(
        encoding="utf-8"
    )
    assert f"User={user}" in unit
    assert f"Group={user}" in unit
    assert "SupplementaryGroups=astrumweaver-config" in unit
    assert "StateDirectory=astrumweaver-control" in unit


def assert_worker_layout(staged: Path, user: str) -> None:
    etc = staged / "etc/astrumweaver"
    state = staged / "var/lib/astrumweaver"
    assert mode(etc) == 0o710
    assert mode(state) == 0o750
    for path in (
        etc / "worker.toml",
        etc / "worker.env",
        etc / "runtime-deployment.json",
        etc / "gpu-uuids",
        etc / "gpu-device-map",
    ):
        assert mode(path) == 0o640

    unit = (staged / "etc/systemd/system/astrumweaver-worker.service").read_text(
        encoding="utf-8"
    )
    assert f"User={user}" in unit
    assert f"Group={user}" in unit
    assert "SupplementaryGroups=astrumweaver-config" in unit
    assert "StateDirectory=astrumweaver" in unit


def assert_both_layout(staged: Path, control_user: str, worker_user: str) -> None:
    assert_control_layout(staged, control_user)
    assert_worker_layout(staged, worker_user)


def test_fresh_control_uses_isolated_default_identity_and_state(
    tmp_path: Path,
) -> None:
    config, environment = write_control_inputs(tmp_path)
    staged = tmp_path / "root"

    result = run(
        *control_command(config, staged, environment=environment),
    )

    assert result.returncode == 0, result.stderr
    assert_control_layout(staged, "astrumweaver-control")
    assert not (staged / "var/lib/astrumweaver").exists()
    assert not (
        staged / "etc/systemd/system/astrumweaver-worker.service"
    ).exists()


def test_fresh_worker_uses_isolated_default_identity_and_state(
    tmp_path: Path,
) -> None:
    config, environment, manifest = write_worker_inputs(tmp_path)
    staged = tmp_path / "root"

    result = run(
        *worker_command(
            config,
            staged,
            environment=environment,
            runtime_manifest=manifest,
            isolation=True,
        ),
    )

    assert result.returncode == 0, result.stderr
    assert_worker_layout(staged, "astrumweaver")
    assert not (staged / "var/lib/astrumweaver-control").exists()
    assert not (
        staged / "etc/systemd/system/astrumweaver-control.service"
    ).exists()


@pytest.mark.parametrize("first_role", ["control", "worker"])
def test_both_default_roles_install_in_either_order(
    tmp_path: Path,
    first_role: str,
) -> None:
    control_config, control_env = write_control_inputs(tmp_path)
    worker_config, worker_env, manifest = write_worker_inputs(tmp_path)
    staged = tmp_path / "root"
    commands = {
        "control": control_command(
            control_config,
            staged,
            environment=control_env,
        ),
        "worker": worker_command(
            worker_config,
            staged,
            environment=worker_env,
            runtime_manifest=manifest,
            isolation=True,
        ),
    }

    first = run(*commands[first_role])
    assert first.returncode == 0, first.stderr
    if first_role == "control":
        assert_control_layout(staged, "astrumweaver-control")
        assert not (staged / "var/lib/astrumweaver").exists()
    else:
        assert_worker_layout(staged, "astrumweaver")
        assert not (staged / "var/lib/astrumweaver-control").exists()

    second_role = "worker" if first_role == "control" else "control"
    second = run(*commands[second_role])
    assert second.returncode == 0, second.stderr
    assert_both_layout(staged, "astrumweaver-control", "astrumweaver")


@pytest.mark.parametrize("first_role", ["control", "worker"])
def test_distinct_custom_accounts_are_positive_in_either_order(
    tmp_path: Path,
    first_role: str,
) -> None:
    control_config, control_env = write_control_inputs(tmp_path)
    worker_config, worker_env, manifest = write_worker_inputs(tmp_path)
    staged = tmp_path / "root"
    commands = {
        "control": control_command(
            control_config,
            staged,
            user="custom-control",
            environment=control_env,
        ),
        "worker": worker_command(
            worker_config,
            staged,
            user="custom-worker",
            environment=worker_env,
            runtime_manifest=manifest,
            isolation=True,
        ),
    }

    first = run(*commands[first_role])
    second_role = "worker" if first_role == "control" else "control"
    second = run(*commands[second_role])

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert_both_layout(staged, "custom-control", "custom-worker")


@pytest.mark.parametrize("first_role", ["control", "worker"])
def test_same_account_is_rejected_before_second_role_mutation(
    tmp_path: Path,
    first_role: str,
) -> None:
    control_config, control_env = write_control_inputs(tmp_path)
    worker_config, worker_env, manifest = write_worker_inputs(tmp_path)
    staged = tmp_path / "root"
    commands = {
        "control": control_command(
            control_config,
            staged,
            user="same-service",
            environment=control_env,
        ),
        "worker": worker_command(
            worker_config,
            staged,
            user="same-service",
            environment=worker_env,
            runtime_manifest=manifest,
            isolation=True,
        ),
    }

    first = run(*commands[first_role])
    assert first.returncode == 0, first.stderr

    etc = staged / "etc/astrumweaver"
    etc.chmod(0o700)
    before_mode = mode(etc)
    second_role = "worker" if first_role == "control" else "control"
    second = run(*commands[second_role])

    assert second.returncode != 0
    assert "legacy/shared" in second.stderr
    assert mode(etc) == before_mode
    missing_destination = etc / (
        "worker.toml" if second_role == "worker" else "control.toml"
    )
    assert not missing_destination.exists()
    missing_state = staged / (
        "var/lib/astrumweaver"
        if second_role == "worker"
        else "var/lib/astrumweaver-control"
    )
    assert not missing_state.exists()
    assert not (
        staged
        / "etc/systemd/system"
        / f"astrumweaver-{second_role}.service"
    ).exists()


@pytest.mark.parametrize("fixture", ["shared-state", "shared-unit"])
def test_legacy_control_state_and_units_fail_before_mutation(
    tmp_path: Path,
    fixture: str,
) -> None:
    config, _ = write_control_inputs(tmp_path)
    staged = tmp_path / "root"
    etc = staged / "etc/astrumweaver"
    etc.mkdir(parents=True)
    etc.chmod(0o700)
    marker = etc / "unchanged"
    marker.write_text("sentinel\n", encoding="utf-8")

    legacy_state = staged / "var/lib/astrumweaver"
    unit = staged / "etc/systemd/system/astrumweaver-control.service"
    if fixture == "shared-state":
        legacy_state.mkdir(parents=True)
        legacy_state.chmod(0o701)
        before_state_mode = mode(legacy_state)
    else:
        unit.parent.mkdir(parents=True)
        unit.write_text(
            "[Service]\n"
            "User=astrumweaver\n"
            "Group=astrumweaver\n"
            "StateDirectory=astrumweaver\n",
            encoding="utf-8",
        )
        unit.chmod(0o600)
        before_unit = unit.read_text(encoding="utf-8")
        before_unit_mode = mode(unit)

    result = run(*control_command(config, staged))

    assert result.returncode != 0
    assert "legacy/shared" in result.stderr
    assert mode(etc) == 0o700
    assert marker.read_text(encoding="utf-8") == "sentinel\n"
    assert not (etc / "control.toml").exists()
    assert not (etc / "control.env").exists()
    assert not (staged / "var/lib/astrumweaver-control").exists()
    if fixture == "shared-state":
        assert mode(legacy_state) == before_state_mode
    else:
        assert unit.read_text(encoding="utf-8") == before_unit
        assert mode(unit) == before_unit_mode


@pytest.mark.parametrize("fixture", ["malformed", "group", "dropin"])
def test_malformed_group_and_dropin_overrides_fail_closed(
    tmp_path: Path,
    fixture: str,
) -> None:
    config, _ = write_control_inputs(tmp_path)
    staged = tmp_path / "root"
    etc = staged / "etc/astrumweaver"
    etc.mkdir(parents=True)
    etc.chmod(0o700)

    unit = staged / "etc/systemd/system/astrumweaver-control.service"
    dropin = (
        staged
        / "etc/systemd/system/astrumweaver-worker.service.d/override.conf"
    )
    if fixture == "malformed":
        unit.parent.mkdir(parents=True)
        unit.write_text(
            "[Service]\n"
            "User=astrumweaver-control\n"
            "Group=astrumweaver-control\n",
            encoding="utf-8",
        )
    elif fixture == "group":
        unit.parent.mkdir(parents=True)
        unit.write_text(
            "[Service]\n"
            "User=astrumweaver-control\n"
            "Group=other-service\n"
            "StateDirectory=astrumweaver-control\n",
            encoding="utf-8",
        )
    else:
        dropin.parent.mkdir(parents=True)
        dropin.write_text(
            "[Service]\nSupplementaryGroups=other-service\n",
            encoding="utf-8",
        )

    result = run(*control_command(config, staged))

    assert result.returncode != 0
    if fixture == "malformed":
        assert "exactly one" in result.stderr
    elif fixture == "group":
        assert "must name the same role-private identity" in result.stderr
    else:
        assert "identity drop-ins" in result.stderr
    assert mode(etc) == 0o700
    assert not (etc / "control.toml").exists()
    assert not (etc / "control.env").exists()
    assert not (staged / "var/lib/astrumweaver-control").exists()
    if fixture == "dropin":
        assert not unit.exists()
        assert dropin.read_text(encoding="utf-8") == (
            "[Service]\nSupplementaryGroups=other-service\n"
        )
    else:
        assert unit.exists()


def test_retry_repairs_permissions_and_repeats_cleanly(tmp_path: Path) -> None:
    config, environment, manifest = write_worker_inputs(tmp_path)
    staged = tmp_path / "root"
    command = worker_command(
        config,
        staged,
        user="retry-worker",
        environment=environment,
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

    for _ in range(2):
        etc.chmod(0o777)
        state.chmod(0o700)
        for path in protected:
            path.chmod(0o600)

        repaired = run(*command)

        assert repaired.returncode == 0, repaired.stderr
        assert mode(etc) == 0o710
        assert mode(state) == 0o750
        assert all(mode(path) == 0o640 for path in protected)
