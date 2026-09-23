from __future__ import annotations

from dataclasses import dataclass

import pytest

from astrumweaver.worker.mode import (
    BorrowableWorkerController,
    GPUProcess,
    ModeTransitionError,
    NvidiaGPUProbe,
)


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
    def __init__(self, health: FakeHealth, *, active: bool = True) -> None:
        self.health = health
        self.active = active
        self.actions: list[str] = []

    def is_active(self) -> bool:
        return self.active

    def request_drain(self) -> None:
        self.actions.append("drain")
        if self.health.payload is not None:
            self.health.payload["draining"] = True
            self.health.payload["ready"] = False

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
    def __init__(self, processes: tuple[GPUProcess, ...] = ()) -> None:
        self.processes = processes
        self.identity_checks = 0

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


def test_development_transition_waits_for_active_job_then_stops() -> None:
    health = FakeHealth()
    health.payload["active_job_id"] = "job-1"
    service = FakeService(health)
    gpu = FakeGPU()
    clock = FakeClock()
    sleeps = 0

    def sleep(seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        clock.sleep(seconds)
        if sleeps == 1 and health.payload is not None:
            health.payload["active_job_id"] = None

    controller = BorrowableWorkerController(
        service,
        health,
        gpu,
        poll_interval_seconds=0.1,
        monotonic=clock.monotonic,
        sleep=sleep,
    )

    report = controller.to_development(drain_timeout_seconds=5.0)

    assert service.actions == ["drain", "stop"]
    assert gpu.identity_checks == 1
    assert report.mode == "development"
    assert not report.service_active
    assert report.gpu_processes == ()


def test_drain_timeout_never_forces_worker_stop() -> None:
    health = FakeHealth()
    health.payload["active_job_id"] = "long-job"
    service = FakeService(health)
    gpu = FakeGPU()
    clock = FakeClock()
    controller = BorrowableWorkerController(
        service,
        health,
        gpu,
        poll_interval_seconds=0.5,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    with pytest.raises(ModeTransitionError, match="timed out waiting for Worker drain"):
        controller.to_development(drain_timeout_seconds=1.0)

    assert service.actions == ["drain"]
    assert service.active
    assert gpu.identity_checks == 0


def test_development_transition_fails_if_gpu_context_remains_after_stop() -> None:
    health = FakeHealth()
    service = FakeService(health)
    process = GPUProcess(
        pid=123,
        gpu_uuid="GPU-a",
        process_type="C",
        process_name="leftover-worker",
    )
    gpu = FakeGPU((process,))
    controller = BorrowableWorkerController(service, health, gpu)

    with pytest.raises(ModeTransitionError, match="still owned"):
        controller.to_development(drain_timeout_seconds=1.0)

    assert service.actions == ["drain", "stop"]
    assert gpu.identity_checks == 1
    assert not service.active


def test_astrumweaver_transition_rejects_development_gpu_process() -> None:
    health = FakeHealth()
    health.payload = None
    service = FakeService(health, active=False)
    process = GPUProcess(
        pid=777,
        gpu_uuid="GPU-a",
        process_type="C+G",
        process_name="developer-app",
    )
    gpu = FakeGPU((process,))
    controller = BorrowableWorkerController(service, health, gpu)

    with pytest.raises(ModeTransitionError, match="development or unrelated"):
        controller.to_astrumweaver()

    assert service.actions == []
    assert gpu.identity_checks == 0
    assert not service.active


def test_astrumweaver_transition_checks_identity_then_starts_and_waits_ready() -> None:
    health = FakeHealth()
    health.payload = None
    service = FakeService(health, active=False)
    gpu = FakeGPU()
    controller = BorrowableWorkerController(service, health, gpu)

    report = controller.to_astrumweaver(start_timeout_seconds=1.0)

    assert gpu.identity_checks == 1
    assert service.actions == ["start"]
    assert report.mode == "astrumweaver"
    assert report.service_active
    assert report.worker_ready


def test_status_distinguishes_draining_from_astrumweaver() -> None:
    health = FakeHealth()
    service = FakeService(health)
    gpu = FakeGPU()
    controller = BorrowableWorkerController(service, health, gpu)

    assert controller.status().mode == "astrumweaver"

    health.payload["draining"] = True
    health.payload["ready"] = False
    assert controller.status().mode == "draining"


def test_nvidia_pmon_maps_processes_back_to_expected_gpu_uuid(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    class Result:
        def __init__(self, returncode: int, stdout: str) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    def fake_run(arguments, **kwargs):
        calls.append(tuple(arguments))
        if "--query-gpu=index,uuid" in arguments:
            return Result(0, "0, GPU-a\n1, GPU-b\n")
        if "pmon" in arguments:
            return Result(
                0,
                "# gpu pid type sm mem enc dec command\n"
                "0 123 C 10 20 0 0 python\n"
                "1 456 G 0 10 0 0 compositor\n"
                "0 - - - - - - -\n",
            )
        raise AssertionError(arguments)

    monkeypatch.setattr("astrumweaver.worker.mode.subprocess.run", fake_run)

    probe = NvidiaGPUProbe(("GPU-a",))
    processes = probe.active_processes()

    assert processes == (
        GPUProcess(
            pid=123,
            gpu_uuid="GPU-a",
            process_type="C",
            process_name="python",
        ),
    )
    assert any("pmon" in call for call in calls)


def test_nvidia_process_inspection_fails_closed(monkeypatch) -> None:
    class Result:
        returncode = 1
        stdout = ""
        stderr = "unsupported"

    monkeypatch.setattr(
        "astrumweaver.worker.mode.subprocess.run",
        lambda *args, **kwargs: Result(),
    )

    probe = NvidiaGPUProbe(("GPU-a",))

    with pytest.raises(ModeTransitionError, match="identity"):
        probe.active_processes()
