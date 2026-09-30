from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

import pytest

from astrumweaver.validation.runtime_deployment import (
    RuntimeDeploymentAcceptanceError,
    RuntimeDeploymentAcceptanceRunner,
    SystemdRuntimeDeploymentHost,
    parse_systemd_show_properties,
    render_runtime_deployment_markdown,
    resolve_packaged_gpu_device_map,
)


ROOT = Path(__file__).resolve().parents[1]


def test_acceptance_resolves_absolute_profile_sibling_without_ambient_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_bin = tmp_path / "installer-profile" / "bin"
    profile_bin.mkdir(parents=True)
    sibling = profile_bin / "astrumweaver-gpu-device-map"
    sibling.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    sibling.chmod(0o755)

    ambient = tmp_path / "ambient"
    ambient.mkdir()
    ambient_mapper = ambient / "astrumweaver-gpu-device-map"
    ambient_mapper.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    ambient_mapper.chmod(0o755)
    monkeypatch.setenv("PATH", str(ambient))

    resolved = resolve_packaged_gpu_device_map(
        argv0=str(profile_bin / "astrumweaver-runtime-deployment-accept"),
        installed_path=tmp_path / "missing-installed-helper",
    )

    assert resolved == str(sibling)


def test_acceptance_prefers_reviewed_installed_mapper_over_profile_sibling(
    tmp_path: Path,
) -> None:
    installed = tmp_path / "usr/local/libexec/astrumweaver/gpu-device-map"
    installed.parent.mkdir(parents=True)
    installed.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    installed.chmod(0o755)
    profile_bin = tmp_path / "installer-profile" / "bin"
    profile_bin.mkdir(parents=True)
    sibling = profile_bin / "astrumweaver-gpu-device-map"
    sibling.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    sibling.chmod(0o755)

    resolved = resolve_packaged_gpu_device_map(
        argv0=str(profile_bin / "astrumweaver-runtime-deployment-accept"),
        installed_path=installed,
    )

    assert resolved == str(installed)


def test_acceptance_rejects_bare_mapper_override(tmp_path: Path) -> None:
    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="absolute executable path",
    ):
        SystemdRuntimeDeploymentHost(
            service="astrumweaver-worker.service",
            worker_config=tmp_path / "worker.toml",
            gpu_device_map="astrumweaver-gpu-device-map",
        )


def test_acceptance_rejects_missing_or_non_executable_mapper(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-mapper"
    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="existing executable file",
    ):
        SystemdRuntimeDeploymentHost(
            service="astrumweaver-worker.service",
            worker_config=tmp_path / "worker.toml",
            gpu_device_map=str(missing),
        )

    non_executable = tmp_path / "non-executable-mapper"
    non_executable.write_text("not executable\n", encoding="utf-8")
    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="existing executable file",
    ):
        SystemdRuntimeDeploymentHost(
            service="astrumweaver-worker.service",
            worker_config=tmp_path / "worker.toml",
            gpu_device_map=str(non_executable),
        )


def test_systemctl_show_properties_parse_effective_values() -> None:
    properties = parse_systemd_show_properties(
        "Description=Worker\n"
        "DevicePolicy=closed\n"
        "DeviceAllow=/dev/nvidia0 rw\n"
        "DeviceAllow=/dev/nvidiactl rw\n"
        "Environment=CUDA_VISIBLE_DEVICES=GPU-selected\n"
        "Environment=OTHER=value\n"
        "EnvironmentFiles=/etc/astrumweaver/worker.env (ignore_errors=yes)\n"
        "UnsetEnvironment=\n"
        "ExecStartPre={ path=/usr/local/libexec/astrumweaver/gpu-preflight ; "
        "argv[]=/usr/local/libexec/astrumweaver/gpu-preflight "
        "/etc/astrumweaver/gpu-uuids ; ignore_errors=no ; }\n"
        "Requires=astrumweaver-worker-gpu-isolation-preflight.service\n"
        "After=network-online.target astrumweaver-worker-gpu-isolation-preflight.service\n"
    )

    assert properties == {
        "DevicePolicy": "closed",
        "DeviceAllow": "/dev/nvidia0 rw /dev/nvidiactl rw",
        "Environment": "CUDA_VISIBLE_DEVICES=GPU-selected OTHER=value",
        "EnvironmentFiles": (
            "/etc/astrumweaver/worker.env (ignore_errors=yes)"
        ),
        "UnsetEnvironment": "",
        "ExecStartPre": (
            "{ path=/usr/local/libexec/astrumweaver/gpu-preflight ; "
            "argv[]=/usr/local/libexec/astrumweaver/gpu-preflight "
            "/etc/astrumweaver/gpu-uuids ; ignore_errors=no ; }"
        ),
        "Requires": "astrumweaver-worker-gpu-isolation-preflight.service",
        "After": (
            "network-online.target "
            "astrumweaver-worker-gpu-isolation-preflight.service"
        ),
    }


def test_systemd_host_reads_effective_properties_with_systemctl_show(
    tmp_path: Path,
) -> None:
    systemctl = tmp_path / "systemctl"
    args_file = tmp_path / "systemctl-args"
    systemctl.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {shlex.quote(str(args_file))}\n"
        "printf '%s\\n' \\\n"
        "  'DevicePolicy=closed' \\\n"
        "  'DeviceAllow=/dev/nvidia0 rw' \\\n"
        "  'DeviceAllow=/dev/nvidiactl rw' \\\n"
        "  'Environment=CUDA_VISIBLE_DEVICES=GPU-selected ASTRUMWEAVER_GPU_ISOLATION_VISIBLE_MAP=/run/astrumweaver-worker-gpu-isolation/gpu-visible-map' \\\n"
        "  'EnvironmentFiles=/etc/astrumweaver/worker.env (ignore_errors=yes)' \\\n"
        "  'UnsetEnvironment=' \\\n"
        "  'ExecStartPre={ path=/usr/local/libexec/astrumweaver/gpu-preflight ; "
        "argv[]=/usr/local/libexec/astrumweaver/gpu-preflight "
        "/etc/astrumweaver/gpu-uuids ; ignore_errors=no ; }' \\\n"
        "  'Requires=astrumweaver-worker-gpu-isolation-preflight.service' \\\n"
        "  'After=network-online.target astrumweaver-worker-gpu-isolation-preflight.service'\n",
        encoding="utf-8",
    )
    systemctl.chmod(0o755)
    mapper = tmp_path / "mapper"
    mapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    mapper.chmod(0o755)

    host = SystemdRuntimeDeploymentHost(
        service="astrumweaver-worker.service",
        worker_config=tmp_path / "worker.toml",
        systemctl=str(systemctl),
        gpu_device_map=str(mapper),
    )

    properties = host.worker_unit_properties()

    assert properties["DevicePolicy"] == "closed"
    assert args_file.read_text(encoding="utf-8").splitlines() == [
        "show",
        "--no-pager",
        "--all",
        "--property=DevicePolicy,DeviceAllow,Environment,EnvironmentFiles,UnsetEnvironment,ExecStartPre,Requires,After",
        "astrumweaver-worker.service",
    ]


@pytest.mark.parametrize(
    ("probe_rc", "expected"),
    ((0, True), (3, False)),
)
def test_systemd_host_probes_actual_subset_device_access(
    tmp_path: Path,
    probe_rc: int,
    expected: bool,
) -> None:
    args_file = tmp_path / "systemd-run-args"
    systemd_run = tmp_path / "systemd-run"
    systemd_run.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {shlex.quote(str(args_file))}\n"
        f"exit {probe_rc}\n",
        encoding="utf-8",
    )
    systemd_run.chmod(0o755)

    mapper = tmp_path / "gpu-device-map"
    mapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    mapper.chmod(0o755)

    host = SystemdRuntimeDeploymentHost(
        service="astrumweaver-worker.service",
        worker_config=tmp_path / "worker.toml",
        systemd_run=str(systemd_run),
        gpu_device_map=str(mapper),
    )

    enforced = host.probe_isolation_enforcement(
        {
            "GPU-selected": "/dev/nvidia0",
            "GPU-other": "/dev/nvidia1",
        },
        ("GPU-selected",),
    )

    assert enforced is expected
    args = args_file.read_text(encoding="utf-8").splitlines()
    assert "--property=DevicePolicy=closed" in args
    assert "--property=DeviceAllow=/dev/nvidia0 rw" in args
    assert "--property=DeviceAllow=/dev/nvidia1 rw" not in args
    assert str(mapper) in args
    assert "probe-access" in args


@pytest.mark.parametrize(
    ("status", "expected"),
    ((0, True), (3, False)),
)
def test_systemd_host_decodes_known_service_activity_statuses(
    tmp_path: Path,
    status: int,
    expected: bool,
) -> None:
    systemctl = tmp_path / "systemctl"
    systemctl.write_text(f"#!/bin/sh\nexit {status}\n", encoding="utf-8")
    systemctl.chmod(0o755)
    mapper = tmp_path / "mapper"
    mapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    mapper.chmod(0o755)
    host = SystemdRuntimeDeploymentHost(
        service="astrumweaver-worker.service",
        worker_config=tmp_path / "worker.toml",
        systemctl=str(systemctl),
        gpu_device_map=str(mapper),
    )

    assert host.service_active() is expected


def test_systemd_host_rejects_unknown_service_activity_status(
    tmp_path: Path,
) -> None:
    systemctl = tmp_path / "systemctl"
    systemctl.write_text("#!/bin/sh\nexit 4\n", encoding="utf-8")
    systemctl.chmod(0o755)
    mapper = tmp_path / "mapper"
    mapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    mapper.chmod(0o755)
    host = SystemdRuntimeDeploymentHost(
        service="astrumweaver-worker.service",
        worker_config=tmp_path / "worker.toml",
        systemctl=str(systemctl),
        gpu_device_map=str(mapper),
    )

    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="cannot determine Worker service state",
    ):
        host.service_active()


class FakeHost:
    def __init__(
        self,
        *,
        host_map: dict[str, str] | None = None,
        worker_uuids: tuple[str, ...] = ("GPU-selected",),
        gpu_preflight: bool = True,
        unit_properties: dict[str, str] | None = None,
        initially_active: bool = False,
        ready: bool = True,
        registered: bool = True,
        isolation_enforced: bool = True,
        start_error_after_activation: bool = False,
        stop_failures: int = 0,
    ) -> None:
        self._host_map = host_map or {
            "GPU-selected": "/dev/nvidia0",
            "GPU-other": "/dev/nvidia1",
        }
        self._worker_uuids = worker_uuids
        self._gpu_preflight = gpu_preflight
        self._unit_properties = unit_properties if unit_properties is not None else {
            "DevicePolicy": "closed",
            "DeviceAllow": "/dev/nvidia0 rw /dev/nvidiactl rw",
            "Environment": (
                "CUDA_VISIBLE_DEVICES=GPU-selected "
                "ASTRUMWEAVER_GPU_ISOLATION_VISIBLE_MAP="
                "/run/astrumweaver-worker-gpu-isolation/gpu-visible-map"
            ),
            "EnvironmentFiles": "",
            "UnsetEnvironment": "",
            "ExecStartPre": (
                "{ path=/usr/local/libexec/astrumweaver/gpu-device-map ; "
                "argv[]=/usr/local/libexec/astrumweaver/gpu-device-map "
                "probe-access /etc/astrumweaver/gpu-uuids "
                "/run/astrumweaver-worker-gpu-isolation/gpu-visible-map ; "
                "ignore_errors=no ; }"
            ),
            "Requires": "astrumweaver-worker-gpu-isolation-preflight.service",
            "After": (
                "network-online.target "
                "astrumweaver-worker-gpu-isolation-preflight.service"
            ),
        }
        self.active = initially_active
        self.ready = ready
        self.registered = registered
        self.isolation_enforced = isolation_enforced
        self.start_error_after_activation = start_error_after_activation
        self.stop_failures = stop_failures
        self.actions: list[str] = []

    def host_gpu_device_map(self):
        return dict(self._host_map)

    def worker_gpu_contract(self):
        return self._worker_uuids, self._gpu_preflight

    def worker_unit_properties(self):
        return dict(self._unit_properties)

    def probe_isolation_enforcement(self, host_map, expected_gpu_uuids):
        assert dict(host_map) == self._host_map
        assert tuple(expected_gpu_uuids) == ("GPU-selected",)
        self.actions.append("probe")
        return self.isolation_enforced

    def service_active(self):
        return self.active

    def start_service(self):
        self.actions.append("start")
        self.active = True
        if self.start_error_after_activation:
            raise RuntimeError("simulated start command failure")

    def stop_service(self):
        self.actions.append("stop")
        if self.stop_failures:
            self.stop_failures -= 1
            raise RuntimeError("simulated stop command failure")
        self.active = False

    def readiness(self):
        if not self.active:
            return None
        return {
            "ready": self.ready,
            "registered": self.registered,
        }


def test_systemd_host_uses_canonical_gpu_device_mapper(tmp_path: Path) -> None:
    mapper = tmp_path / "gpu-device-map"
    mapper.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'GPU-selected=/dev/nvidia0\\nGPU-other=/dev/nvidia2\\n'\n",
        encoding="utf-8",
    )
    mapper.chmod(0o755)

    host = SystemdRuntimeDeploymentHost(
        service="astrumweaver-worker.service",
        worker_config=tmp_path / "worker.toml",
        gpu_device_map=str(mapper),
    )

    assert host.host_gpu_device_map() == {
        "GPU-selected": "/dev/nvidia0",
        "GPU-other": "/dev/nvidia2",
    }


@dataclass
class FakeClock:
    value: float = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def make_runner(host: FakeHost):
    clock = FakeClock()
    return RuntimeDeploymentAcceptanceRunner(
        host=host,
        expected_gpu_uuids=("GPU-selected",),
        revision="deadbeef12345678",
        deployment_path="systemd",
        start_timeout_seconds=2.0,
        poll_interval_seconds=0.1,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )


def test_runtime_deployment_acceptance_proves_isolated_subset_start() -> None:
    host = FakeHost()
    evidence = make_runner(host).run()

    assert evidence.overall == "PASS"
    assert evidence.host_gpu_count == 2
    assert evidence.selected_gpu_count == 1
    assert evidence.host_gpu_superset == "PASS"
    assert evidence.selected_device_allow_exact == "PASS"
    assert evidence.worker_started_ready == "PASS"
    assert evidence.worker_registered == "PASS"
    assert evidence.isolation_enforcement == "PASS"
    assert evidence.fail_closed == "N/A"
    assert host.actions == ["probe", "start", "stop"]
    assert not host.active


def test_runtime_deployment_acceptance_records_fail_closed_unavailable_isolation() -> None:
    host = FakeHost(isolation_enforced=False)
    evidence = make_runner(host).run()

    assert evidence.overall == "PASS"
    assert evidence.isolation_enforcement == "UNAVAILABLE"
    assert evidence.fail_closed == "PASS"
    assert evidence.worker_started_ready == "NOT_RUN"
    assert evidence.worker_registered == "NOT_RUN"
    assert evidence.worker_contract_exact == "NOT_RUN"
    assert host.actions == ["probe"]
    assert not host.active

    markdown = render_runtime_deployment_markdown(evidence)
    assert "| Isolation enforcement | UNAVAILABLE |" in markdown
    assert "| Fail closed | PASS |" in markdown
    assert "| Overall | PASS |" in markdown


def test_runtime_deployment_acceptance_rejects_extra_physical_device_allow() -> None:
    host = FakeHost(
        unit_properties={
            "DevicePolicy": "closed",
            "DeviceAllow": "/dev/nvidia0 rw /dev/nvidia1 rw",
            "Environment": (
                "CUDA_VISIBLE_DEVICES=GPU-selected "
                "ASTRUMWEAVER_GPU_ISOLATION_VISIBLE_MAP="
                "/run/astrumweaver-worker-gpu-isolation/gpu-visible-map"
            ),
            "EnvironmentFiles": "",
            "UnsetEnvironment": "",
            "ExecStartPre": "/usr/local/libexec/astrumweaver/gpu-preflight x",
        }
    )

    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="DeviceAllow physical GPU set is not exact",
    ):
        make_runner(host).run()

    assert host.actions == ["probe"]


@pytest.mark.parametrize(
    "device_allow",
    (
        "/dev/nvidia0 r /dev/nvidiactl rw",
        "/dev/nvidia0 rw /dev/nvidia* rw",
        "char-* rwm /dev/nvidia0 rw",
    ),
)
def test_runtime_deployment_rejects_broad_or_incomplete_device_grants(
    device_allow: str,
) -> None:
    host = FakeHost(
        unit_properties={
            "DevicePolicy": "closed",
            "DeviceAllow": device_allow,
            "Environment": (
                "CUDA_VISIBLE_DEVICES=GPU-selected "
                "ASTRUMWEAVER_GPU_ISOLATION_VISIBLE_MAP="
                "/run/astrumweaver-worker-gpu-isolation/gpu-visible-map"
            ),
            "ExecStartPre": "path=/usr/local/libexec/astrumweaver/gpu-preflight ;",
        }
    )

    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="broad or unsupported device grant|physical GPU set is not exact",
    ):
        make_runner(host).run()


def test_runtime_deployment_rejects_effective_policy_override() -> None:
    host = FakeHost(
        unit_properties={
            "DevicePolicy": "auto",
            "DeviceAllow": "/dev/nvidia0 rw /dev/nvidiactl rw",
            "Environment": (
                "CUDA_VISIBLE_DEVICES=GPU-selected "
                "ASTRUMWEAVER_GPU_ISOLATION_VISIBLE_MAP="
                "/run/astrumweaver-worker-gpu-isolation/gpu-visible-map"
            ),
            "ExecStartPre": "/usr/local/libexec/astrumweaver/gpu-preflight x",
        }
    )

    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="does not use DevicePolicy=closed",
    ):
        make_runner(host).run()


def test_runtime_deployment_requires_effective_gpu_preflight_executable() -> None:
    host = FakeHost(
        unit_properties={
            "DevicePolicy": "closed",
            "DeviceAllow": "/dev/nvidia0 rw /dev/nvidiactl rw",
            "Environment": (
                "CUDA_VISIBLE_DEVICES=GPU-selected "
                "ASTRUMWEAVER_GPU_ISOLATION_VISIBLE_MAP="
                "/run/astrumweaver-worker-gpu-isolation/gpu-visible-map"
            ),
            "ExecStartPre": (
                "{ path=/bin/false ; argv[]=/bin/false gpu-preflight ; }"
            ),
        }
    )

    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="lacks in-cgroup GPU access preflight",
    ):
        make_runner(host).run()


def test_runtime_deployment_rejects_cuda_override_from_environment_file(
    tmp_path: Path,
) -> None:
    environment_file = tmp_path / "worker.env"
    environment_file.write_text(
        "CUDA_VISIBLE_DEVICES=GPU-other\n",
        encoding="utf-8",
    )
    properties = FakeHost().worker_unit_properties()
    properties["EnvironmentFiles"] = f"{environment_file} (ignore_errors=no)"

    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="may be overridden by an EnvironmentFile",
    ):
        make_runner(FakeHost(unit_properties=properties)).run()


@pytest.mark.parametrize(
    "unset_environment",
    ("CUDA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES=GPU-other"),
)
def test_runtime_deployment_rejects_effective_unset_of_cuda_visibility(
    unset_environment: str,
) -> None:
    properties = FakeHost().worker_unit_properties()
    properties["UnsetEnvironment"] = unset_environment

    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="is unset by effective systemd policy",
    ):
        make_runner(FakeHost(unit_properties=properties)).run()


def test_runtime_deployment_accepts_reviewed_nixos_access_preflight() -> None:
    mapper = (
        "/nix/store/"
        + "a" * 32
        + "-astrumweaver-gpu-device-map/bin/astrumweaver-gpu-device-map"
    )
    expected_uuids = "/nix/store/" + "b" * 32 + "-astrumweaver-gpu-uuids"
    properties = FakeHost().worker_unit_properties()
    properties["ExecStartPre"] = (
        f"{{ path={mapper} ; argv[]={mapper} probe-access "
        f"{expected_uuids} "
        "/run/astrumweaver-worker-gpu-isolation/gpu-visible-map ; "
        "ignore_errors=no ; }"
    )
    host = FakeHost(unit_properties=properties)

    assert make_runner(host).run().overall == "PASS"


def test_runtime_deployment_rejects_legacy_raw_exact_preflight_for_subset() -> None:
    properties = FakeHost().worker_unit_properties()
    properties["ExecStartPre"] = (
        "{ path=/usr/local/libexec/astrumweaver/gpu-preflight ; "
        "argv[]=/usr/local/libexec/astrumweaver/gpu-preflight "
        "/etc/astrumweaver/gpu-uuids ; ignore_errors=no ; }"
    )

    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="lacks in-cgroup GPU access preflight",
    ):
        make_runner(FakeHost(unit_properties=properties)).run()




def test_runtime_deployment_acceptance_requires_real_host_superset() -> None:
    host = FakeHost(
        host_map={"GPU-selected": "/dev/nvidia0"}
    )

    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="host-visible GPU superset",
    ):
        make_runner(host).run()


def test_runtime_deployment_acceptance_requires_exact_set_gate() -> None:
    host = FakeHost(gpu_preflight=False)

    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="exact GPU preflight is disabled",
    ):
        make_runner(host).run()


def test_runtime_deployment_acceptance_stops_service_on_readiness_failure() -> None:
    host = FakeHost(ready=False)

    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="did not become ready",
    ):
        make_runner(host).run()

    assert host.actions == ["probe", "start", "stop"]
    assert not host.active


def test_runtime_deployment_cleans_up_when_start_command_raises_after_start() -> None:
    host = FakeHost(start_error_after_activation=True)

    with pytest.raises(RuntimeError, match="simulated start command failure"):
        make_runner(host).run()

    assert host.actions == ["probe", "start", "stop"]
    assert not host.active


def test_runtime_deployment_retries_transient_stop_failure() -> None:
    host = FakeHost(stop_failures=1)

    evidence = make_runner(host).run()

    assert evidence.overall == "PASS"
    assert host.actions == ["probe", "start", "stop", "stop"]
    assert not host.active


def test_runtime_deployment_fails_if_cleanup_cannot_stop_service() -> None:
    host = FakeHost(stop_failures=2)

    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="remained active after acceptance cleanup",
    ):
        make_runner(host).run()

    assert host.actions == ["probe", "start", "stop", "stop"]
    assert host.active


def test_runtime_deployment_evidence_is_private_safe() -> None:
    host = FakeHost()
    evidence = make_runner(host).run()

    markdown = render_runtime_deployment_markdown(evidence)

    for private_value in (
        "GPU-selected",
        "GPU-other",
        "/dev/nvidia0",
        "worker-private",
        "192.168.1.20",
        "http://control.private",
    ):
        assert private_value not in markdown

    assert "| Host-visible GPU count | 2 |" in markdown
    assert "| Selected GPU count | 1 |" in markdown
    assert "| Overall | PASS |" in markdown


def test_runtime_deployment_evidence_schema_cannot_carry_private_identity() -> None:
    evidence = make_runner(FakeHost()).run()
    fields = set(evidence.__dataclass_fields__)

    assert "gpu_uuid" not in fields
    assert "gpu_uuids" not in fields
    assert "device_path" not in fields
    assert "hostname" not in fields
    assert "worker_id" not in fields
    assert "control_url" not in fields
