"""Redacted real-node hardware acceptance for AstrumWeaver v0.1."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

import httpx

from ..transport import PROTOCOL_VERSION
from ..worker.mode import (
    BorrowableWorkerController,
    GPUProbe,
    HealthProbe,
    ModeTransitionError,
    NvidiaGPUProbe,
    ServiceManager,
    SystemdServiceManager,
    HTTPHealthProbe,
)


class HardwareAcceptanceError(RuntimeError):
    """Real-node acceptance failed without publishing private context."""


PUBLIC_PROFILE_CLASSES = frozenset({
    "modern-single",
    "multi-gpu-large",
    "legacy-single",
})
REVISION_PATTERN = re.compile(r"^[0-9a-fA-F]{7,64}$")


class AcceptanceControl(Protocol):
    def submit_pinned_job(
        self,
        *,
        capability: str,
        gpu_uuids: tuple[str, ...],
        marker: str,
        delay_seconds: float = 0.0,
    ) -> str: ...

    def job_status(self, job_id: str) -> str: ...

    def cancel_job(self, job_id: str) -> None: ...


class HTTPAcceptanceControl:
    def __init__(
        self,
        base_url: str,
        client_token: str,
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not base_url:
            raise ValueError("control base_url is required")
        if not client_token:
            raise ValueError("client token is required")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"authorization": f"Bearer {client_token}"},
            timeout=timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise HardwareAcceptanceError("Control API is unavailable") from exc
        if response.status_code >= 400:
            raise HardwareAcceptanceError(
                f"Control API rejected acceptance request ({response.status_code})"
            )
        return response

    def submit_pinned_job(
        self,
        *,
        capability: str,
        gpu_uuids: tuple[str, ...],
        marker: str,
        delay_seconds: float = 0.0,
    ) -> str:
        payload = {
            "acceptance": "v0.1-hardware-e2e",
            "marker": marker,
        }
        if delay_seconds:
            payload["_debug_delay_seconds"] = delay_seconds

        response = self._request(
            "POST",
            "/v1/jobs",
            json={
                "protocol_version": PROTOCOL_VERSION,
                "capability": capability,
                "payload": payload,
                "requirements": {
                    "required_gpu_uuids": list(gpu_uuids),
                },
                "priority": 100,
                "max_attempts": 1,
                "idempotency_key": f"hardware-e2e-{marker}",
            },
        )
        body = response.json()
        job_id = body.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise HardwareAcceptanceError("Control returned an invalid job ID")
        return job_id

    def job_status(self, job_id: str) -> str:
        response = self._request("GET", f"/v1/jobs/{job_id}")
        body = response.json()
        status = body.get("status")
        if not isinstance(status, str):
            raise HardwareAcceptanceError("Control returned an invalid job status")
        return status

    def cancel_job(self, job_id: str) -> None:
        self._request("POST", f"/v1/jobs/{job_id}/cancel")


@dataclass(frozen=True, slots=True)
class HardwareAcceptanceEvidence:
    evidence_version: str
    date_utc: str
    astrumweaver_revision: str
    deployment_path: str
    profile_class: str
    gpu_count: int
    private_values_omitted: bool
    exact_uuid_preflight: str
    worker_registration: str
    first_job_round_trip: str
    drain_state_observed: str
    active_job_completed_while_draining: str
    drain_no_new_claims: str
    service_stopped_after_drain: str
    gpu_processes_after_release: int
    development_mode: str
    return_online: str
    second_job_round_trip: str
    overall: str


class HardwareAcceptanceRunner:
    def __init__(
        self,
        *,
        control: AcceptanceControl,
        service: ServiceManager,
        health: HealthProbe,
        gpu: GPUProbe,
        expected_gpu_uuids: tuple[str, ...],
        revision: str,
        deployment_path: str,
        profile_class: str,
        capability: str,
        poll_interval_seconds: float = 0.5,
        job_timeout_seconds: float = 60.0,
        drain_anchor_seconds: float = 3.0,
        drain_probe_seconds: float = 2.0,
        mode_start_timeout_seconds: float = 60.0,
        mode_drain_timeout_seconds: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not expected_gpu_uuids:
            raise ValueError("at least one expected GPU UUID is required")
        if len(set(expected_gpu_uuids)) != len(expected_gpu_uuids):
            raise ValueError("expected GPU UUIDs must be unique")
        if deployment_path not in {"nixos", "systemd"}:
            raise ValueError("deployment_path must be nixos or systemd")
        revision = revision.strip()
        profile_class = profile_class.strip()
        if not REVISION_PATTERN.fullmatch(revision):
            raise ValueError("revision must be a public git commit SHA")
        if profile_class not in PUBLIC_PROFILE_CLASSES:
            raise ValueError(
                "profile_class must be one of the public v0.1 GPU profiles"
            )
        if not capability.strip():
            raise ValueError("capability is required")
        for value, name in (
            (poll_interval_seconds, "poll_interval_seconds"),
            (job_timeout_seconds, "job_timeout_seconds"),
            (drain_anchor_seconds, "drain_anchor_seconds"),
            (drain_probe_seconds, "drain_probe_seconds"),
            (mode_start_timeout_seconds, "mode_start_timeout_seconds"),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.control = control
        self.service = service
        self.health = health
        self.gpu = gpu
        self.expected_gpu_uuids = expected_gpu_uuids
        self.revision = revision
        self.deployment_path = deployment_path
        self.profile_class = profile_class
        self.capability = capability.strip()
        if drain_anchor_seconds > 30:
            raise ValueError("drain_anchor_seconds must not exceed 30")

        self.poll_interval_seconds = poll_interval_seconds
        self.job_timeout_seconds = job_timeout_seconds
        self.drain_anchor_seconds = drain_anchor_seconds
        self.drain_probe_seconds = drain_probe_seconds
        self.mode_start_timeout_seconds = mode_start_timeout_seconds
        self.mode_drain_timeout_seconds = mode_drain_timeout_seconds
        self._sleep = sleep
        self._monotonic = monotonic

    def _wait_for_worker_ready(self) -> None:
        deadline = self._monotonic() + self.mode_start_timeout_seconds
        while True:
            snapshot = self.health.snapshot()
            if snapshot and snapshot.get("ready") is True:
                return
            if self._monotonic() >= deadline:
                raise HardwareAcceptanceError("Worker did not become locally ready")
            self._sleep(self.poll_interval_seconds)

    def _wait_job_terminal(self, job_id: str) -> str:
        deadline = self._monotonic() + self.job_timeout_seconds
        while True:
            status = self.control.job_status(job_id)
            if status in {"succeeded", "failed", "cancelled"}:
                return status
            if self._monotonic() >= deadline:
                raise HardwareAcceptanceError("validation job did not reach terminal state")
            self._sleep(self.poll_interval_seconds)

    def _wait_job_running(self, job_id: str) -> None:
        deadline = self._monotonic() + self.job_timeout_seconds
        while True:
            status = self.control.job_status(job_id)
            if status == "running":
                return
            if status in {"succeeded", "failed", "cancelled"}:
                raise HardwareAcceptanceError(
                    "drain-anchor job reached terminal state before RUNNING was observed"
                )
            if self._monotonic() >= deadline:
                raise HardwareAcceptanceError(
                    "drain-anchor job did not become RUNNING"
                )
            self._sleep(self.poll_interval_seconds)

    def _submit_round_trip(self) -> None:
        marker = uuid4().hex
        job_id = self.control.submit_pinned_job(
            capability=self.capability,
            gpu_uuids=self.expected_gpu_uuids,
            marker=marker,
        )
        status = self._wait_job_terminal(job_id)
        if status != "succeeded":
            raise HardwareAcceptanceError(
                f"validation job terminal state was {status}"
            )

    def _wait_drain_state(self) -> None:
        deadline = self._monotonic() + self.job_timeout_seconds
        while True:
            snapshot = self.health.snapshot()
            if snapshot and snapshot.get("draining") is True:
                return
            if self._monotonic() >= deadline:
                raise HardwareAcceptanceError("Worker did not expose DRAINING locally")
            self._sleep(self.poll_interval_seconds)

    def run(self) -> HardwareAcceptanceEvidence:
        self.gpu.require_exact_identity()

        controller = BorrowableWorkerController(
            self.service,
            self.health,
            self.gpu,
            poll_interval_seconds=self.poll_interval_seconds,
            monotonic=self._monotonic,
            sleep=self._sleep,
        )

        # Bring the node to a known ONLINE/ready starting point. If a local
        # development process currently owns the GPU, this fails closed.
        online = controller.to_astrumweaver(
            start_timeout_seconds=self.mode_start_timeout_seconds,
            drain_timeout_seconds=self.mode_drain_timeout_seconds,
        )
        if not online.worker_ready:
            raise HardwareAcceptanceError("Worker is not ready after enrollment")
        self._wait_for_worker_ready()

        self._submit_round_trip()

        # Put a real job in RUNNING first, then request DRAINING. This proves
        # the current attempt can finish normally while no later job is claimed.
        anchor_job = self.control.submit_pinned_job(
            capability=self.capability,
            gpu_uuids=self.expected_gpu_uuids,
            marker=uuid4().hex,
            delay_seconds=self.drain_anchor_seconds,
        )
        self._wait_job_running(anchor_job)

        self.service.request_drain()
        self._wait_drain_state()

        probe_job = self.control.submit_pinned_job(
            capability=self.capability,
            gpu_uuids=self.expected_gpu_uuids,
            marker=uuid4().hex,
        )
        try:
            self._sleep(self.drain_probe_seconds)
            if self.control.job_status(probe_job) != "queued":
                raise HardwareAcceptanceError(
                    "a new pinned job was claimed while Worker was draining"
                )

            anchor_status = self._wait_job_terminal(anchor_job)
            if anchor_status != "succeeded":
                raise HardwareAcceptanceError(
                    f"active drain-anchor job terminal state was {anchor_status}"
                )
        finally:
            self.control.cancel_job(probe_job)

        released = controller.to_development(
            drain_timeout_seconds=self.mode_drain_timeout_seconds,
        )
        if released.service_active:
            raise HardwareAcceptanceError("Worker service remained active after release")
        if released.gpu_processes:
            raise HardwareAcceptanceError("GPU process contexts remained after release")
        if released.mode != "development":
            raise HardwareAcceptanceError("local mode did not become development")

        returned = controller.to_astrumweaver(
            start_timeout_seconds=self.mode_start_timeout_seconds,
            drain_timeout_seconds=self.mode_drain_timeout_seconds,
        )
        if not returned.worker_ready or returned.mode != "astrumweaver":
            raise HardwareAcceptanceError("Worker did not return ONLINE/ready")

        self._submit_round_trip()

        return HardwareAcceptanceEvidence(
            evidence_version="v0.1",
            date_utc=datetime.now(UTC).date().isoformat(),
            astrumweaver_revision=self.revision,
            deployment_path=self.deployment_path,
            profile_class=self.profile_class,
            gpu_count=len(self.expected_gpu_uuids),
            private_values_omitted=True,
            exact_uuid_preflight="PASS",
            worker_registration="PASS",
            first_job_round_trip="PASS",
            drain_state_observed="PASS",
            active_job_completed_while_draining="PASS",
            drain_no_new_claims="PASS",
            service_stopped_after_drain="PASS",
            gpu_processes_after_release=0,
            development_mode="PASS",
            return_online="PASS",
            second_job_round_trip="PASS",
            overall="PASS",
        )


def render_markdown(evidence: HardwareAcceptanceEvidence) -> str:
    # Intentionally render only redacted/public-safe fields. Control URL,
    # Worker ID, tokens, hostnames, actual GPU UUIDs and GPU model are never
    # represented in HardwareAcceptanceEvidence.
    fields = [
        ("Evidence version", evidence.evidence_version),
        ("Date (UTC)", evidence.date_utc),
        ("AstrumWeaver revision", evidence.astrumweaver_revision),
        ("Deployment path", evidence.deployment_path),
        ("Profile class", evidence.profile_class),
        ("GPU count", str(evidence.gpu_count)),
        ("Private values omitted", str(evidence.private_values_omitted).lower()),
        ("Exact UUID preflight", evidence.exact_uuid_preflight),
        ("Worker registration / readiness", evidence.worker_registration),
        ("First generic job round-trip", evidence.first_job_round_trip),
        ("DRAINING observed", evidence.drain_state_observed),
        (
            "Active job completed while DRAINING",
            evidence.active_job_completed_while_draining,
        ),
        ("No new claim while DRAINING", evidence.drain_no_new_claims),
        ("Worker service stopped after drain", evidence.service_stopped_after_drain),
        ("GPU process contexts after release", str(evidence.gpu_processes_after_release)),
        ("Development ownership reached", evidence.development_mode),
        ("Worker returned ONLINE", evidence.return_online),
        ("Second generic job round-trip", evidence.second_job_round_trip),
        ("Overall", evidence.overall),
    ]
    rows = "\n".join(f"| {key} | {value} |" for key, value in fields)
    return (
        "# AstrumWeaver v0.1 Hardware E2E Evidence\n\n"
        "This file is intentionally redacted. It contains no hostname, IP address, "
        "VM/LXC identifier, GPU UUID, credential, private URL, or GPU model.\n\n"
        "| Field | Result |\n"
        "| --- | --- |\n"
        f"{rows}\n"
    )


def write_evidence(path: Path, evidence: HardwareAcceptanceEvidence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(evidence), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="astrumweaver-hardware-accept")
    parser.add_argument("--control-url", required=True)
    parser.add_argument("--gpu-uuid", action="append", default=[], dest="gpu_uuids")
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--deployment-path",
        choices=("nixos", "systemd"),
        required=True,
    )
    parser.add_argument(
        "--profile-class",
        choices=tuple(sorted(PUBLIC_PROFILE_CLASSES)),
        required=True,
    )
    parser.add_argument("--capability", default="debug.echo")
    parser.add_argument(
        "--health-url",
        default="http://127.0.0.1:9100",
    )
    parser.add_argument(
        "--service",
        default="astrumweaver-worker.service",
    )
    parser.add_argument("--systemctl", default="systemctl")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--job-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--drain-anchor-seconds", type=float, default=3.0)
    parser.add_argument("--drain-probe-seconds", type=float, default=2.0)
    parser.add_argument("--drain-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--start-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("validation/v0.1/hardware-e2e.md"),
    )
    args = parser.parse_args()

    client_token = os.environ.get("ASTRUMWEAVER_CLIENT_TOKEN", "")
    if not client_token:
        parser.exit(
            2,
            "astrumweaver-hardware-accept: ASTRUMWEAVER_CLIENT_TOKEN is required\n",
        )

    gpu_uuids = tuple(str(value).strip() for value in args.gpu_uuids)
    if not gpu_uuids or any(not value for value in gpu_uuids):
        parser.exit(
            2,
            "astrumweaver-hardware-accept: at least one --gpu-uuid is required\n",
        )

    control = HTTPAcceptanceControl(args.control_url, client_token)
    try:
        service = SystemdServiceManager(
            args.service,
            systemctl=args.systemctl,
        )
        health = HTTPHealthProbe(args.health_url)
        gpu = NvidiaGPUProbe(
            gpu_uuids,
            nvidia_smi=args.nvidia_smi,
        )
        runner = HardwareAcceptanceRunner(
            control=control,
            service=service,
            health=health,
            gpu=gpu,
            expected_gpu_uuids=gpu_uuids,
            revision=args.revision,
            deployment_path=args.deployment_path,
            profile_class=args.profile_class,
            capability=args.capability,
            job_timeout_seconds=args.job_timeout_seconds,
            drain_anchor_seconds=args.drain_anchor_seconds,
            drain_probe_seconds=args.drain_probe_seconds,
            mode_start_timeout_seconds=args.start_timeout_seconds,
            mode_drain_timeout_seconds=(
                None
                if args.drain_timeout_seconds == 0
                else args.drain_timeout_seconds
            ),
        )
        evidence = runner.run()
        write_evidence(args.evidence, evidence)
    except (HardwareAcceptanceError, ModeTransitionError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"astrumweaver-hardware-accept: {exc}\n")
    finally:
        control.close()

    print(json.dumps(asdict(evidence), sort_keys=True))


if __name__ == "__main__":
    main()
