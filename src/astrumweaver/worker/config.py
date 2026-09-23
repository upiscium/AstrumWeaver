"""Worker daemon configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from ..config import (
    ConfigurationError,
    load_toml,
    positive_float,
    positive_int,
    require_string,
    table,
)
from ..contracts import ResourceShape, WorkerSpec
from ..control.models import WorkerRegistration


@dataclass(frozen=True, slots=True)
class WorkerRuntimeConfig:
    registration: WorkerRegistration
    control_url: str
    executor_factory: str
    executor_settings: Mapping[str, Any]
    poll_interval_seconds: float = 1.0
    heartbeat_interval_seconds: float = 10.0
    request_timeout_seconds: float = 15.0
    status_file: Path | None = Path("/run/astrumweaver-worker/status.json")
    allow_insecure_http: bool = False

    @classmethod
    def from_file(cls, path: str) -> "WorkerRuntimeConfig":
        config = load_toml(path)
        worker = table(config, "worker", required=True)
        resources = table(worker, "resources", required=True)
        control = table(config, "control", required=True)
        executor = table(config, "executor", required=True)

        worker_id = require_string(worker, "id", context="worker")
        worker_class = require_string(worker, "class", context="worker")

        capabilities_raw = worker.get("capabilities", [])
        if not isinstance(capabilities_raw, list):
            raise ConfigurationError("worker.capabilities must be an array")
        capabilities = frozenset(str(item) for item in capabilities_raw)

        gpu_raw = worker.get("gpu_uuids", [])
        if not isinstance(gpu_raw, list):
            raise ConfigurationError("worker.gpu_uuids must be an array")
        gpu_uuids = tuple(str(item) for item in gpu_raw)

        labels_raw = worker.get("labels", {})
        if not isinstance(labels_raw, dict):
            raise ConfigurationError("worker.labels must be a table")

        metadata_raw = worker.get("metadata", {})
        if not isinstance(metadata_raw, dict):
            raise ConfigurationError("worker.metadata must be a table")

        spec = WorkerSpec(
            worker_id=worker_id,
            worker_class=worker_class,
            resources=ResourceShape(
                gpu_count=int(resources.get("gpu_count", len(gpu_uuids))),
                total_vram_mb=int(resources.get("total_vram_mb", 0)),
                max_single_gpu_vram_mb=int(
                    resources.get("max_single_gpu_vram_mb", 0)
                ),
            ),
            gpu_uuids=gpu_uuids,
            capabilities=capabilities,
            labels={str(key): str(value) for key, value in labels_raw.items()},
        )

        control_url = require_string(control, "url", context="control").rstrip("/")
        parsed = urlparse(control_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigurationError("control.url must be an absolute HTTP(S) URL")

        allow_insecure = bool(control.get("allow_insecure_http", False))
        local_hosts = {"127.0.0.1", "localhost", "::1"}
        if parsed.scheme == "http" and parsed.hostname not in local_hosts and not allow_insecure:
            raise ConfigurationError(
                "plain HTTP Control URLs require control.allow_insecure_http=true"
            )

        executor_factory = require_string(executor, "factory", context="executor")
        executor_settings = executor.get("settings", {})
        if not isinstance(executor_settings, dict):
            raise ConfigurationError("executor.settings must be a table")

        raw_status_file = worker.get(
            "status_file",
            "/run/astrumweaver-worker/status.json",
        )
        if raw_status_file is None:
            status_file = None
        elif isinstance(raw_status_file, str) and raw_status_file:
            status_file = Path(raw_status_file)
        else:
            raise ConfigurationError("worker.status_file must be a path string or null")

        return cls(
            registration=WorkerRegistration(
                spec=spec,
                max_concurrency=positive_int(
                    worker,
                    "max_concurrency",
                    context="worker",
                    default=1,
                ),
                metadata=metadata_raw,
            ),
            control_url=control_url,
            executor_factory=executor_factory,
            executor_settings=executor_settings,
            poll_interval_seconds=positive_float(
                worker,
                "poll_interval_seconds",
                context="worker",
                default=1.0,
            ),
            heartbeat_interval_seconds=positive_float(
                worker,
                "heartbeat_interval_seconds",
                context="worker",
                default=10.0,
            ),
            request_timeout_seconds=positive_float(
                control,
                "request_timeout_seconds",
                context="control",
                default=15.0,
            ),
            status_file=status_file,
            allow_insecure_http=allow_insecure,
        )
