"""Local ownership transitions for borrowable GPU Workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol, Sequence

import httpx

from .runtime import require_exact_gpu_set


class ModeTransitionError(RuntimeError):
    """Borrowable Worker ownership transition failed safely."""


@dataclass(frozen=True, slots=True)
class GPUProcess:
    pid: int
    gpu_uuid: str
    process_name: str


@dataclass(frozen=True, slots=True)
class ModeReport:
    mode: str
    service_active: bool
    worker_ready: bool
    worker_draining: bool
    active_job_id: str | None
    gpu_processes: tuple[GPUProcess, ...]


class ServiceManager(Protocol):
    def is_active(self) -> bool: ...
    def request_drain(self) -> None: ...
    def stop(self) -> None: ...
    def start(self) -> None: ...


class HealthProbe(Protocol):
    def snapshot(self) -> dict[str, Any] | None: ...


class GPUProbe(Protocol):
    def require_exact_identity(self) -> None: ...
    def active_processes(self) -> tuple[GPUProcess, ...]: ...


class SystemdServiceManager:
    def __init__(
        self,
        service_name: str,
        *,
        systemctl: str = "systemctl",
    ) -> None:
        if not service_name:
            raise ValueError("service_name is required")
        self.service_name = service_name
        self.systemctl = systemctl

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [self.systemctl, *arguments],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise ModeTransitionError("systemctl is unavailable") from exc

    def is_active(self) -> bool:
        result = self._run("is-active", "--quiet", self.service_name)
        if result.returncode == 0:
            return True
        if result.returncode == 3:
            return False
        raise ModeTransitionError(
            f"cannot determine service state for {self.service_name}"
        )

    def request_drain(self) -> None:
        result = self._run(
            "kill",
            "--kill-whom=main",
            "--signal=SIGUSR1",
            self.service_name,
        )
        if result.returncode != 0:
            raise ModeTransitionError("failed to request Worker drain")

    def stop(self) -> None:
        result = self._run("stop", self.service_name)
        if result.returncode != 0:
            raise ModeTransitionError("failed to stop Worker service")

    def start(self) -> None:
        result = self._run("start", self.service_name)
        if result.returncode != 0:
            raise ModeTransitionError("failed to start Worker service")


class HTTPHealthProbe:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 2.0,
    ) -> None:
        if not base_url:
            raise ValueError("health base_url is required")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def snapshot(self) -> dict[str, Any] | None:
        try:
            response = httpx.get(
                f"{self.base_url}/health",
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        return payload if isinstance(payload, dict) else None

    def ready(self) -> bool:
        try:
            response = httpx.get(
                f"{self.base_url}/ready",
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError:
            return False
        if response.status_code != 200:
            return False
        try:
            payload = response.json()
        except ValueError:
            return False
        return isinstance(payload, dict) and payload.get("ready") is True


class NvidiaGPUProbe:
    def __init__(
        self,
        expected_gpu_uuids: Sequence[str],
        *,
        nvidia_smi: str = "nvidia-smi",
    ) -> None:
        expected = tuple(sorted(str(value).strip() for value in expected_gpu_uuids))
        if not expected or any(not value for value in expected):
            raise ValueError("borrowable GPU mode requires expected GPU UUIDs")
        if len(set(expected)) != len(expected):
            raise ValueError("expected GPU UUIDs must be unique")
        self.expected_gpu_uuids = expected
        self.nvidia_smi = nvidia_smi

    def require_exact_identity(self) -> None:
        require_exact_gpu_set(
            self.expected_gpu_uuids,
            command=self.nvidia_smi,
        )

    def active_processes(self) -> tuple[GPUProcess, ...]:
        try:
            result = subprocess.run(
                [
                    self.nvidia_smi,
                    "--query-compute-apps=pid,gpu_uuid,process_name",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise ModeTransitionError("nvidia-smi is unavailable") from exc
        if result.returncode != 0:
            raise ModeTransitionError("cannot inspect NVIDIA GPU processes")

        expected = set(self.expected_gpu_uuids)
        processes: list[GPUProcess] = []
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",", 2)]
            if len(parts) != 3:
                raise ModeTransitionError("unexpected nvidia-smi process output")
            raw_pid, gpu_uuid, process_name = parts
            if gpu_uuid not in expected:
                continue
            try:
                pid = int(raw_pid)
            except ValueError as exc:
                raise ModeTransitionError(
                    "unexpected nvidia-smi process PID"
                ) from exc
            processes.append(
                GPUProcess(
                    pid=pid,
                    gpu_uuid=gpu_uuid,
                    process_name=process_name,
                )
            )
        return tuple(processes)


class BorrowableWorkerController:
    def __init__(
        self,
        service: ServiceManager,
        health: HealthProbe,
        gpu: GPUProbe,
        *,
        poll_interval_seconds: float = 0.5,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self.service = service
        self.health = health
        self.gpu = gpu
        self.poll_interval_seconds = poll_interval_seconds
        self._monotonic = monotonic
        self._sleep = sleep

    def _wait(
        self,
        predicate: Callable[[], bool],
        *,
        timeout_seconds: float | None,
        description: str,
    ) -> None:
        deadline = (
            None
            if timeout_seconds is None
            else self._monotonic() + timeout_seconds
        )
        while not predicate():
            if deadline is not None and self._monotonic() >= deadline:
                raise ModeTransitionError(
                    f"timed out waiting for {description}"
                )
            self._sleep(self.poll_interval_seconds)

    def _worker_snapshot(self) -> dict[str, Any] | None:
        return self.health.snapshot()

    def _report(self) -> ModeReport:
        active = self.service.is_active()
        snapshot = self._worker_snapshot() if active else None
        ready = bool(snapshot and snapshot.get("status") == "ok" and not snapshot.get("draining"))
        draining = bool(snapshot and snapshot.get("draining"))
        active_job_id = (
            str(snapshot["active_job_id"])
            if snapshot and snapshot.get("active_job_id") is not None
            else None
        )
        gpu_processes = self.gpu.active_processes()

        if active and draining:
            mode = "draining"
        elif active and ready:
            mode = "astrumweaver"
        elif active:
            mode = "transitioning"
        else:
            mode = "development"

        return ModeReport(
            mode=mode,
            service_active=active,
            worker_ready=ready,
            worker_draining=draining,
            active_job_id=active_job_id,
            gpu_processes=gpu_processes,
        )

    def status(self) -> ModeReport:
        return self._report()

    def to_development(
        self,
        *,
        drain_timeout_seconds: float | None = None,
    ) -> ModeReport:
        if self.service.is_active():
            self.service.request_drain()

            def drained() -> bool:
                if not self.service.is_active():
                    return True
                snapshot = self._worker_snapshot()
                return bool(
                    snapshot
                    and snapshot.get("draining") is True
                    and snapshot.get("active_job_id") is None
                )

            self._wait(
                drained,
                timeout_seconds=drain_timeout_seconds,
                description="Worker drain",
            )

            if self.service.is_active():
                self.service.stop()

        if self.service.is_active():
            raise ModeTransitionError("Worker service is still active")

        busy = self.gpu.active_processes()
        if busy:
            raise ModeTransitionError(
                "GPU is still owned by one or more compute processes"
            )
        return self._report()

    def to_astrumweaver(
        self,
        *,
        start_timeout_seconds: float = 60.0,
    ) -> ModeReport:
        if start_timeout_seconds <= 0:
            raise ValueError("start_timeout_seconds must be positive")

        if not self.service.is_active():
            busy = self.gpu.active_processes()
            if busy:
                raise ModeTransitionError(
                    "GPU is in use by a development or unrelated compute process"
                )
            self.gpu.require_exact_identity()
            self.service.start()

        self._wait(
            self.service.is_active,
            timeout_seconds=start_timeout_seconds,
            description="Worker service activation",
        )

        def ready() -> bool:
            snapshot = self._worker_snapshot()
            return bool(
                snapshot
                and snapshot.get("status") == "ok"
                and snapshot.get("draining") is not True
            )

        self._wait(
            ready,
            timeout_seconds=start_timeout_seconds,
            description="Worker readiness",
        )
        return self._report()


def _report_json(report: ModeReport) -> str:
    payload = asdict(report)
    payload["gpu_processes"] = [
        asdict(process) for process in report.gpu_processes
    ]
    return json.dumps(payload, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="astrumweaver-worker-mode")
    parser.add_argument(
        "mode",
        choices=("development", "astrumweaver", "status"),
    )
    parser.add_argument(
        "--service",
        default="astrumweaver-worker.service",
    )
    parser.add_argument(
        "--gpu-uuid",
        action="append",
        dest="gpu_uuids",
        default=[],
    )
    parser.add_argument(
        "--health-url",
        default="http://127.0.0.1:9100",
    )
    parser.add_argument(
        "--systemctl",
        default="systemctl",
    )
    parser.add_argument(
        "--nvidia-smi",
        default="nvidia-smi",
    )
    parser.add_argument(
        "--drain-timeout-seconds",
        type=float,
        default=0.0,
        help="0 waits indefinitely; positive values fail without forcing the job.",
    )
    parser.add_argument(
        "--start-timeout-seconds",
        type=float,
        default=60.0,
    )
    args = parser.parse_args()

    service = SystemdServiceManager(
        args.service,
        systemctl=args.systemctl,
    )
    health = HTTPHealthProbe(args.health_url)
    gpu = NvidiaGPUProbe(
        args.gpu_uuids,
        nvidia_smi=args.nvidia_smi,
    )
    controller = BorrowableWorkerController(service, health, gpu)

    try:
        if args.mode == "development":
            timeout = (
                None
                if args.drain_timeout_seconds == 0
                else args.drain_timeout_seconds
            )
            report = controller.to_development(
                drain_timeout_seconds=timeout
            )
        elif args.mode == "astrumweaver":
            report = controller.to_astrumweaver(
                start_timeout_seconds=args.start_timeout_seconds
            )
        else:
            report = controller.status()
    except (ModeTransitionError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"astrumweaver-worker-mode: {exc}\n")

    print(_report_json(report))


if __name__ == "__main__":
    main()
