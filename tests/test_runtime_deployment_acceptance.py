from __future__ import annotations

from dataclasses import dataclass

import pytest

from astrumweaver.validation.runtime_deployment import (
    RuntimeDeploymentAcceptanceError,
    RuntimeDeploymentAcceptanceRunner,
    render_runtime_deployment_markdown,
)


class FakeHost:
    def __init__(
        self,
        *,
        host_map: dict[str, str] | None = None,
        worker_uuids: tuple[str, ...] = ("GPU-selected",),
        gpu_preflight: bool = True,
        unit_text: str | None = None,
        initially_active: bool = False,
        ready: bool = True,
        registered: bool = True,
    ) -> None:
        self._host_map = host_map or {
            "GPU-selected": "/dev/nvidia0",
            "GPU-other": "/dev/nvidia1",
        }
        self._worker_uuids = worker_uuids
        self._gpu_preflight = gpu_preflight
        self._unit_text = unit_text or (
            "[Service]\n"
            "DevicePolicy=closed\n"
            "DeviceAllow=/dev/nvidia0 rw\n"
            "DeviceAllow=/dev/nvidiactl rw\n"
            "Environment=CUDA_VISIBLE_DEVICES=GPU-selected\n"
            "ExecStartPre=+/usr/local/libexec/astrumweaver/gpu-preflight "
            "/etc/astrumweaver/gpu-uuids\n"
        )
        self.active = initially_active
        self.ready = ready
        self.registered = registered
        self.actions: list[str] = []

    def host_gpu_device_map(self):
        return dict(self._host_map)

    def worker_gpu_contract(self):
        return self._worker_uuids, self._gpu_preflight

    def worker_unit_text(self):
        return self._unit_text

    def service_active(self):
        return self.active

    def start_service(self):
        self.actions.append("start")
        self.active = True

    def stop_service(self):
        self.actions.append("stop")
        self.active = False

    def readiness(self):
        if not self.active:
            return None
        return {
            "ready": self.ready,
            "registered": self.registered,
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
    assert host.actions == ["start", "stop"]
    assert not host.active


def test_runtime_deployment_acceptance_rejects_extra_physical_device_allow() -> None:
    host = FakeHost(
        unit_text=(
            "[Service]\n"
            "DevicePolicy=closed\n"
            "DeviceAllow=/dev/nvidia0 rw\n"
            "DeviceAllow=/dev/nvidia1 rw\n"
            "Environment=CUDA_VISIBLE_DEVICES=GPU-selected\n"
            "ExecStartPre=/usr/local/libexec/astrumweaver/gpu-preflight x\n"
        )
    )

    with pytest.raises(
        RuntimeDeploymentAcceptanceError,
        match="DeviceAllow physical GPU set is not exact",
    ):
        make_runner(host).run()

    assert host.actions == []


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

    assert host.actions == ["start", "stop"]
    assert not host.active


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
