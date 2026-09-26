"""Private-safe real-host acceptance for RuntimeProvider GPU subset isolation."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping, Protocol

import httpx

from .hardware import REVISION_PATTERN


class RuntimeDeploymentAcceptanceError(RuntimeError):
    pass


class RuntimeDeploymentHost(Protocol):
    def host_gpu_device_map(self) -> Mapping[str, str]: ...

    def worker_gpu_contract(self) -> tuple[tuple[str, ...], bool]: ...

    def worker_unit_text(self) -> str: ...

    def service_active(self) -> bool: ...

    def start_service(self) -> None: ...

    def stop_service(self) -> None: ...

    def readiness(self) -> Mapping[str, object] | None: ...


class SystemdRuntimeDeploymentHost:
    def __init__(
        self,
        *,
        service: str,
        worker_config: Path,
        systemctl: str = "systemctl",
        nvidia_smi: str = "nvidia-smi",
        health_url: str = "http://127.0.0.1:9100/health",
    ) -> None:
        self.service = service
        self.worker_config = worker_config
        self.systemctl = systemctl
        self.nvidia_smi = nvidia_smi
        self.health_url = health_url.rstrip("/")

    def _run(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                args,
                check=check,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeDeploymentAcceptanceError(
                "required host command failed"
            ) from exc

    def host_gpu_device_map(self) -> Mapping[str, str]:
        completed = self._run(
            [
                self.nvidia_smi,
                "--query-gpu=uuid,minor_number",
                "--format=csv,noheader,nounits",
            ]
        )
        result: dict[str, str] = {}
        for raw_line in completed.stdout.splitlines():
            fields = [item.strip() for item in raw_line.split(",")]
            if len(fields) != 2 or not fields[0] or not fields[1].isdigit():
                raise RuntimeDeploymentAcceptanceError(
                    "host GPU UUID/minor mapping is unavailable"
                )
            uuid, minor = fields
            if uuid in result:
                raise RuntimeDeploymentAcceptanceError(
                    "host reported duplicate GPU UUID"
                )
            result[uuid] = f"/dev/nvidia{minor}"
        if not result:
            raise RuntimeDeploymentAcceptanceError(
                "host reported no NVIDIA GPUs"
            )
        return result

    def worker_gpu_contract(self) -> tuple[tuple[str, ...], bool]:
        try:
            with self.worker_config.open("rb") as handle:
                config = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise RuntimeDeploymentAcceptanceError(
                "Worker configuration is unavailable"
            ) from exc
        worker = dict(config.get("worker") or {})
        uuids = tuple(str(item) for item in worker.get("gpu_uuids", ()))
        preflight = bool(worker.get("gpu_preflight", True))
        return uuids, preflight

    def worker_unit_text(self) -> str:
        return self._run(
            [self.systemctl, "cat", self.service]
        ).stdout

    def service_active(self) -> bool:
        completed = self._run(
            [self.systemctl, "is-active", "--quiet", self.service],
            check=False,
        )
        return completed.returncode == 0

    def start_service(self) -> None:
        self._run([self.systemctl, "start", self.service])

    def stop_service(self) -> None:
        self._run([self.systemctl, "stop", self.service])

    def readiness(self) -> Mapping[str, object] | None:
        try:
            response = httpx.get(self.health_url, timeout=3.0)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        try:
            body = response.json()
        except ValueError:
            return None
        return body if isinstance(body, dict) else None


@dataclass(frozen=True, slots=True)
class RuntimeDeploymentAcceptanceEvidence:
    evidence_version: str
    date_utc: str
    astrumweaver_revision: str
    deployment_path: str
    host_gpu_count: int
    selected_gpu_count: int
    private_values_omitted: bool
    host_gpu_superset: str
    worker_contract_exact: str
    worker_exact_set_preflight_enabled: str
    uuid_device_mapping_verified: str
    device_policy_closed: str
    selected_device_allow_exact: str
    cuda_visible_devices_exact: str
    in_service_exact_set_gate_present: str
    worker_started_ready: str
    worker_registered: str
    service_stopped_after_acceptance: str
    overall: str


_DEVICE_ALLOW_RE = re.compile(
    r"^DeviceAllow=(/dev/nvidia[0-9]+)\s+[rwm]+\s*$",
    re.MULTILINE,
)


class RuntimeDeploymentAcceptanceRunner:
    def __init__(
        self,
        *,
        host: RuntimeDeploymentHost,
        expected_gpu_uuids: tuple[str, ...],
        revision: str,
        deployment_path: str,
        start_timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not expected_gpu_uuids:
            raise ValueError("at least one expected GPU UUID is required")
        if len(set(expected_gpu_uuids)) != len(expected_gpu_uuids):
            raise ValueError("expected GPU UUIDs must be unique")
        if not REVISION_PATTERN.fullmatch(revision.strip()):
            raise ValueError("revision must be a public git commit SHA")
        if deployment_path not in {"nixos", "systemd"}:
            raise ValueError("deployment_path must be nixos or systemd")
        if start_timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("acceptance timeouts must be positive")
        self.host = host
        self.expected_gpu_uuids = expected_gpu_uuids
        self.revision = revision.strip()
        self.deployment_path = deployment_path
        self.start_timeout_seconds = start_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._sleep = sleep
        self._monotonic = monotonic

    def _wait_ready(self) -> Mapping[str, object]:
        deadline = self._monotonic() + self.start_timeout_seconds
        while True:
            payload = self.host.readiness()
            if (
                payload is not None
                and payload.get("ready") is True
                and payload.get("registered") is not False
            ):
                return payload
            if self._monotonic() >= deadline:
                raise RuntimeDeploymentAcceptanceError(
                    "isolated Worker did not become ready"
                )
            self._sleep(self.poll_interval_seconds)

    def run(self) -> RuntimeDeploymentAcceptanceEvidence:
        host_map = dict(self.host.host_gpu_device_map())
        expected = self.expected_gpu_uuids
        if len(host_map) <= len(expected):
            raise RuntimeDeploymentAcceptanceError(
                "acceptance requires a host-visible GPU superset"
            )
        if any(uuid not in host_map for uuid in expected):
            raise RuntimeDeploymentAcceptanceError(
                "selected GPU UUID is not visible on the host"
            )

        configured_uuids, gpu_preflight = self.host.worker_gpu_contract()
        if configured_uuids != expected:
            raise RuntimeDeploymentAcceptanceError(
                "Worker GPU UUID order/set does not match acceptance selection"
            )
        if not gpu_preflight:
            raise RuntimeDeploymentAcceptanceError(
                "Worker exact GPU preflight is disabled"
            )

        unit_text = self.host.worker_unit_text()
        if "DevicePolicy=closed" not in unit_text:
            raise RuntimeDeploymentAcceptanceError(
                "Worker service does not use DevicePolicy=closed"
            )

        expected_paths = {host_map[uuid] for uuid in expected}
        allowed_physical = set(_DEVICE_ALLOW_RE.findall(unit_text))
        if allowed_physical != expected_paths:
            raise RuntimeDeploymentAcceptanceError(
                "Worker DeviceAllow physical GPU set is not exact"
            )

        expected_visible = ",".join(expected)
        if (
            f"Environment=CUDA_VISIBLE_DEVICES={expected_visible}"
            not in unit_text
        ):
            raise RuntimeDeploymentAcceptanceError(
                "Worker CUDA_VISIBLE_DEVICES does not preserve selected GPU order"
            )
        if "gpu-preflight" not in unit_text:
            raise RuntimeDeploymentAcceptanceError(
                "Worker service lacks in-cgroup exact-set preflight"
            )

        if self.host.service_active():
            raise RuntimeDeploymentAcceptanceError(
                "acceptance requires the Worker service to start from inactive state"
            )

        started = False
        try:
            self.host.start_service()
            started = True
            ready = self._wait_ready()
            if ready.get("ready") is not True:
                raise RuntimeDeploymentAcceptanceError(
                    "Worker readiness did not prove exact-set startup"
                )
            if ready.get("registered") is not True:
                raise RuntimeDeploymentAcceptanceError(
                    "Worker did not register after isolated startup"
                )
        finally:
            if started:
                self.host.stop_service()

        if self.host.service_active():
            raise RuntimeDeploymentAcceptanceError(
                "Worker service remained active after acceptance"
            )

        return RuntimeDeploymentAcceptanceEvidence(
            evidence_version="runtime-deployment-v1",
            date_utc=datetime.now(UTC).date().isoformat(),
            astrumweaver_revision=self.revision,
            deployment_path=self.deployment_path,
            host_gpu_count=len(host_map),
            selected_gpu_count=len(expected),
            private_values_omitted=True,
            host_gpu_superset="PASS",
            worker_contract_exact="PASS",
            worker_exact_set_preflight_enabled="PASS",
            uuid_device_mapping_verified="PASS",
            device_policy_closed="PASS",
            selected_device_allow_exact="PASS",
            cuda_visible_devices_exact="PASS",
            in_service_exact_set_gate_present="PASS",
            worker_started_ready="PASS",
            worker_registered="PASS",
            service_stopped_after_acceptance="PASS",
            overall="PASS",
        )


def render_runtime_deployment_markdown(
    evidence: RuntimeDeploymentAcceptanceEvidence,
) -> str:
    fields = [
        ("Evidence version", evidence.evidence_version),
        ("Date (UTC)", evidence.date_utc),
        ("AstrumWeaver revision", evidence.astrumweaver_revision),
        ("Deployment path", evidence.deployment_path),
        ("Host-visible GPU count", str(evidence.host_gpu_count)),
        ("Selected GPU count", str(evidence.selected_gpu_count)),
        ("Private values omitted", str(evidence.private_values_omitted).lower()),
        ("Host-visible GPU superset", evidence.host_gpu_superset),
        ("Worker GPU contract exact", evidence.worker_contract_exact),
        (
            "Worker exact-set preflight enabled",
            evidence.worker_exact_set_preflight_enabled,
        ),
        ("UUID/device mapping verified", evidence.uuid_device_mapping_verified),
        ("DevicePolicy closed", evidence.device_policy_closed),
        ("Selected DeviceAllow exact", evidence.selected_device_allow_exact),
        ("CUDA visible-device order exact", evidence.cuda_visible_devices_exact),
        (
            "In-service exact-set gate present",
            evidence.in_service_exact_set_gate_present,
        ),
        ("Worker started ready", evidence.worker_started_ready),
        ("Worker registered", evidence.worker_registered),
        (
            "Service stopped after acceptance",
            evidence.service_stopped_after_acceptance,
        ),
        ("Overall", evidence.overall),
    ]
    rows = "\n".join(f"| {key} | {value} |" for key, value in fields)
    return (
        "# AstrumWeaver Runtime Deployment GPU Isolation Evidence\n\n"
        "This file is intentionally redacted. It contains no hostname, IP "
        "address, Worker ID, GPU UUID, device ordinal, credential, private "
        "URL, or GPU model.\n\n"
        "| Field | Result |\n"
        "| --- | --- |\n"
        f"{rows}\n"
    )


def write_runtime_deployment_evidence(
    path: Path,
    evidence: RuntimeDeploymentAcceptanceEvidence,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_runtime_deployment_markdown(evidence),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="astrumweaver-runtime-deployment-accept"
    )
    parser.add_argument(
        "--gpu-uuid",
        action="append",
        default=[],
        dest="gpu_uuids",
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--deployment-path",
        choices=("nixos", "systemd"),
        required=True,
    )
    parser.add_argument(
        "--worker-config",
        type=Path,
        default=Path("/etc/astrumweaver/worker.toml"),
    )
    parser.add_argument(
        "--service",
        default="astrumweaver-worker.service",
    )
    parser.add_argument("--systemctl", default="systemctl")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument(
        "--health-url",
        default="http://127.0.0.1:9100/health",
    )
    parser.add_argument(
        "--start-timeout-seconds",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path(
            "validation/runtime-deployment/gpu-subset-e2e.md"
        ),
    )
    args = parser.parse_args()

    gpu_uuids = tuple(str(value).strip() for value in args.gpu_uuids)
    if not gpu_uuids or any(not value for value in gpu_uuids):
        parser.exit(
            2,
            "astrumweaver-runtime-deployment-accept: "
            "at least one --gpu-uuid is required\n",
        )

    host = SystemdRuntimeDeploymentHost(
        service=args.service,
        worker_config=args.worker_config,
        systemctl=args.systemctl,
        nvidia_smi=args.nvidia_smi,
        health_url=args.health_url,
    )
    runner = RuntimeDeploymentAcceptanceRunner(
        host=host,
        expected_gpu_uuids=gpu_uuids,
        revision=args.revision,
        deployment_path=args.deployment_path,
        start_timeout_seconds=args.start_timeout_seconds,
    )

    try:
        evidence = runner.run()
        write_runtime_deployment_evidence(args.evidence, evidence)
    except (
        RuntimeDeploymentAcceptanceError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.exit(
            1,
            f"astrumweaver-runtime-deployment-accept: {exc}\n",
        )

    print(json.dumps(asdict(evidence), sort_keys=True))


if __name__ == "__main__":
    main()
