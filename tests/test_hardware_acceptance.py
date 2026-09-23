from __future__ import annotations

from dataclasses import dataclass

import pytest

from astrumweaver.validation.hardware import (
    HardwareAcceptanceError,
    HardwareAcceptanceRunner,
    render_markdown,
)
from astrumweaver.worker.mode import GPUProcess


class FakeControl:
    def __init__(self, *, probe_claimed: bool = False) -> None:
        self.probe_claimed = probe_claimed
        self.submissions: list[tuple[str, tuple[str, ...], str]] = []
        self.cancelled: list[str] = []
        self._counter = 0

    def submit_pinned_job(
        self,
        *,
        capability: str,
        gpu_uuids: tuple[str, ...],
        marker: str,
    ) -> str:
        self._counter += 1
        job_id = f"job-{self._counter}"
        self.submissions.append((capability, gpu_uuids, marker))
        return job_id

    def job_status(self, job_id: str) -> str:
        if job_id == "job-2":
            return "running" if self.probe_claimed else "queued"
        return "succeeded"

    def cancel_job(self, job_id: str) -> None:
        self.cancelled.append(job_id)


class FakeHealth:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = {
            "status": "ok",
            "registered": True,
            "ready": True,
            "draining": False,
            "active_job_id": None,
        }

    def snapshot(self):
        return None if self.payload is None else dict(self.payload)


class FakeService:
    def __init__(self, health: FakeHealth) -> None:
        self.health = health
        self.active = True
        self.actions: list[str] = []

    def is_active(self) -> bool:
        return self.active

    def request_drain(self) -> None:
        self.actions.append("drain")
        assert self.health.payload is not None
        self.health.payload["draining"] = True
        self.health.payload["ready"] = False
        self.health.payload["active_job_id"] = None

    def stop(self) -> None:
        self.actions.append("stop")
        self.active = False
        self.health.payload = None

    def start(self) -> None:
        self.actions.append("start")
        self.active = True
        self.health.payload = {
            "status": "ok",
            "registered": True,
            "ready": True,
            "draining": False,
            "active_job_id": None,
        }


class FakeGPU:
    def __init__(self) -> None:
        self.identity_checks = 0
        self.processes: tuple[GPUProcess, ...] = ()

    def require_exact_identity(self) -> None:
        self.identity_checks += 1

    def active_processes(self) -> tuple[GPUProcess, ...]:
        return self.processes


@dataclass
class FakeClock:
    value: float = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def make_runner(*, probe_claimed: bool = False):
    control = FakeControl(probe_claimed=probe_claimed)
    health = FakeHealth()
    service = FakeService(health)
    gpu = FakeGPU()
    clock = FakeClock()
    runner = HardwareAcceptanceRunner(
        control=control,
        service=service,
        health=health,
        gpu=gpu,
        expected_gpu_uuids=("GPU-private-real-value",),
        revision="deadbeef1234567890",
        deployment_path="nixos",
        profile_class="modern-single",
        capability="debug.echo",
        poll_interval_seconds=0.1,
        job_timeout_seconds=5.0,
        drain_probe_seconds=0.2,
        mode_start_timeout_seconds=5.0,
        mode_drain_timeout_seconds=5.0,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    return runner, control, health, service, gpu


def test_hardware_acceptance_runner_covers_full_real_node_contract() -> None:
    runner, control, health, service, gpu = make_runner()

    evidence = runner.run()

    assert evidence.overall == "PASS"
    assert evidence.exact_uuid_preflight == "PASS"
    assert evidence.worker_registration == "PASS"
    assert evidence.first_job_round_trip == "PASS"
    assert evidence.drain_state_observed == "PASS"
    assert evidence.drain_no_new_claims == "PASS"
    assert evidence.service_stopped_after_drain == "PASS"
    assert evidence.gpu_processes_after_release == 0
    assert evidence.development_mode == "PASS"
    assert evidence.return_online == "PASS"
    assert evidence.second_job_round_trip == "PASS"

    assert len(control.submissions) == 3
    assert all(
        gpu_uuids == ("GPU-private-real-value",)
        for _, gpu_uuids, _ in control.submissions
    )
    assert control.cancelled == ["job-2"]

    assert service.actions == ["drain", "drain", "stop", "start"]
    assert service.active
    assert health.payload is not None
    assert health.payload["ready"] is True
    assert gpu.identity_checks >= 3


def test_hardware_acceptance_rejects_new_claim_while_draining() -> None:
    runner, control, _, service, _ = make_runner(probe_claimed=True)

    with pytest.raises(
        HardwareAcceptanceError,
        match="claimed while Worker was draining",
    ):
        runner.run()

    assert control.cancelled == ["job-2"]
    assert service.active
    assert "stop" not in service.actions


def test_redacted_evidence_does_not_contain_private_runtime_values() -> None:
    runner, _, _, _, _ = make_runner()
    evidence = runner.run()

    markdown = render_markdown(evidence)

    for secret_value in (
        "GPU-private-real-value",
        "private-control.internal",
        "worker-private-id",
        "client-secret-token",
        "192.168.1.20",
        "vmid=1234",
    ):
        assert secret_value not in markdown

    assert "deadbeef1234567890" in markdown
    assert "| GPU count | 1 |" in markdown
    assert "| Private values omitted | true |" in markdown
    assert "| Overall | PASS |" in markdown


def test_evidence_schema_cannot_carry_uuid_hostname_or_control_url() -> None:
    runner, _, _, _, _ = make_runner()
    evidence = runner.run()

    fields = set(evidence.__dataclass_fields__)

    assert "gpu_uuid" not in fields
    assert "gpu_uuids" not in fields
    assert "hostname" not in fields
    assert "control_url" not in fields
    assert "worker_id" not in fields
    assert "gpu_model" not in fields


@pytest.mark.parametrize(
    ("revision", "profile_class"),
    [
        ("not-a-sha", "modern-single"),
        ("deadbeef1234567", "private-lab-profile"),
    ],
)
def test_public_evidence_metadata_rejects_freeform_private_values(
    revision: str,
    profile_class: str,
) -> None:
    control = FakeControl()
    health = FakeHealth()
    service = FakeService(health)
    gpu = FakeGPU()

    with pytest.raises(ValueError):
        HardwareAcceptanceRunner(
            control=control,
            service=service,
            health=health,
            gpu=gpu,
            expected_gpu_uuids=("GPU-private-real-value",),
            revision=revision,
            deployment_path="nixos",
            profile_class=profile_class,
            capability="debug.echo",
        )
