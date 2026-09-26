"""Generic systemd SetupActionDriver for RuntimeProvider deployment."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from ..worker.runtime import require_exact_gpu_set
from .contracts import (
    ActionInspection,
    ActionReceipt,
    SetupAction,
    SetupActionKind,
    SetupActionState,
)


_PROVIDER_EXECUTABLES: Mapping[str, str] = {
    "ollama": "ollama",
    "llama-cpp": "llama-server",
    "vllm": "vllm",
    "freetoken": "ft",
    "exllamav3": "python",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    ) + "\n"


def _action_marker(action: SetupAction) -> str:
    payload = _canonical_json(action.to_dict()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SystemdSetupDriver:
    """Materialize reviewed runtime setup on an existing systemd Worker host.

    Runtime package mutation is allowed only through an explicit operator-
    configured argv prefix. No shell is involved and no installer is guessed.
    """

    def __init__(
        self,
        *,
        root: Path = Path("/"),
        worker_config_path: Path = Path("/etc/astrumweaver/worker.toml"),
        runtime_manifest_path: Path = Path(
            "/etc/astrumweaver/runtime-deployment.json"
        ),
        service_name: str = "astrumweaver-worker.service",
        systemctl: str = "systemctl",
        nvidia_smi: str = "nvidia-smi",
        ready_url: str = "http://127.0.0.1:9100/ready",
        installers: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.root = root
        self.worker_config_path = worker_config_path
        self.runtime_manifest_path = runtime_manifest_path
        self.service_name = service_name
        self.systemctl = systemctl
        self.nvidia_smi = nvidia_smi
        self.ready_url = ready_url
        self.installers = {
            str(key): tuple(str(item) for item in value)
            for key, value in dict(installers or {}).items()
        }

    def _target(self, path: Path) -> Path:
        if self.root == Path("/"):
            return path
        return self.root / path.relative_to("/")

    def _runtime_dir(self, provider_id: str, logical_name: str) -> Path:
        if logical_name == "config":
            return self._target(Path(f"/etc/astrumweaver/runtime/{provider_id}"))
        if logical_name == "cache":
            return self._target(
                Path(f"/var/cache/astrumweaver/runtime/{provider_id}")
            )
        return self._target(
            Path(f"/var/lib/astrumweaver/runtime/{provider_id}")
        )

    def _package_marker(self, provider_id: str, package_reference: str) -> Path:
        digest = hashlib.sha256(package_reference.encode("utf-8")).hexdigest()[:20]
        return self._target(
            Path(
                f"/var/lib/astrumweaver/runtime/{provider_id}/"
                f"packages/{digest}.installed"
            )
        )

    def _provider_executable_available(self, provider_id: str) -> bool:
        executable = _PROVIDER_EXECUTABLES.get(provider_id)
        return bool(executable and shutil.which(executable))

    def _installer(self, provider_id: str) -> tuple[str, ...] | None:
        value = self.installers.get(provider_id)
        return value if value else None

    def _read_worker_gpu_uuids(self) -> tuple[str, ...]:
        path = self._target(self.worker_config_path)
        try:
            with path.open("rb") as handle:
                config = tomllib.load(handle)
        except OSError as exc:
            raise RuntimeError(
                f"Worker config is unavailable: {path}"
            ) from exc
        worker = dict(config.get("worker") or {})
        return tuple(str(item) for item in worker.get("gpu_uuids", ()))

    def _service_active(self) -> bool:
        if self.root != Path("/"):
            return False
        completed = subprocess.run(
            [self.systemctl, "is-active", "--quiet", self.service_name],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.returncode == 0

    def _ready(self) -> bool:
        if self.root != Path("/"):
            return False
        try:
            with urllib.request.urlopen(self.ready_url, timeout=3.0) as response:
                if response.status != 200:
                    return False
                body = json.loads(response.read().decode("utf-8"))
        except (
            OSError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ):
            return False
        return isinstance(body, dict) and body.get("ready") is True

    def _render_targets(
        self,
        action: SetupAction,
    ) -> tuple[tuple[Path, str], ...]:
        deployment = action.payload.get("runtime_deployment")
        if not isinstance(deployment, Mapping):
            raise RuntimeError(
                "render_config action lacks runtime_deployment manifest"
            )

        targets: list[tuple[Path, str]] = [
            (
                self._target(self.runtime_manifest_path),
                _canonical_json(deployment),
            )
        ]

        configuration = action.payload.get("configuration")
        if isinstance(configuration, Mapping):
            config_path = configuration.get("config_path")
            tabby_config = configuration.get("tabby_config")
            if (
                isinstance(config_path, str)
                and config_path.startswith("/")
                and isinstance(tabby_config, Mapping)
            ):
                # JSON is valid YAML and avoids a deployment-only YAML
                # dependency while remaining consumable by TabbyAPI.
                targets.append(
                    (
                        self._target(Path(config_path)),
                        _canonical_json(tabby_config),
                    )
                )
        return tuple(targets)

    def inspect(self, action: SetupAction) -> ActionInspection:
        kind = action.kind
        provider_id = str(action.payload.get("provider_id") or "").strip()

        if kind is SetupActionKind.ENSURE_DIRECTORY:
            path = self._runtime_dir(
                provider_id,
                str(action.payload["logical_name"]),
            )
            return ActionInspection(
                state=(
                    SetupActionState.SATISFIED
                    if path.is_dir()
                    else SetupActionState.NEEDS_APPLY
                ),
                detail=str(path),
            )

        if kind is SetupActionKind.ENSURE_PACKAGE:
            package_reference = str(action.payload["package_reference"])
            marker = self._package_marker(provider_id, package_reference)
            if marker.is_file() or self._provider_executable_available(provider_id):
                return ActionInspection(
                    SetupActionState.SATISFIED,
                    "runtime package/executable is available",
                )
            if self._installer(provider_id) is None:
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    (
                        "runtime package is unavailable and no explicit "
                        f"installer is configured for {provider_id}"
                    ),
                )
            return ActionInspection(
                SetupActionState.NEEDS_APPLY,
                "runtime package requires explicit installer",
            )

        if kind is SetupActionKind.RENDER_CONFIG:
            try:
                targets = self._render_targets(action)
            except RuntimeError as exc:
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    str(exc),
                )
            satisfied = all(
                path.is_file()
                and path.read_text(encoding="utf-8") == content
                for path, content in targets
            )
            return ActionInspection(
                SetupActionState.SATISFIED
                if satisfied
                else SetupActionState.NEEDS_APPLY,
                "runtime deployment configuration",
            )

        if kind is SetupActionKind.VERIFY_MODEL_REFERENCE:
            model_ref = str(action.payload.get("model_ref") or "")
            if model_ref.startswith("/"):
                path = self._target(Path(model_ref))
                return ActionInspection(
                    SetupActionState.SATISFIED
                    if path.exists()
                    else SetupActionState.BLOCKED,
                    (
                        "local model reference exists"
                        if path.exists()
                        else "local model reference does not exist"
                    ),
                )
            return ActionInspection(
                SetupActionState.SATISFIED,
                "provider-native model reference will be resolved by the runtime",
            )

        if kind in {
            SetupActionKind.DOWNLOAD_MODEL,
            SetupActionKind.CONVERT_MODEL,
        }:
            marker = self._target(
                Path(
                    f"/var/lib/astrumweaver/runtime/{provider_id}/"
                    f"authorizations/{_action_marker(action)}"
                )
            )
            return ActionInspection(
                SetupActionState.SATISFIED
                if marker.is_file()
                else SetupActionState.NEEDS_APPLY,
                (
                    "model preparation authorization recorded"
                    if marker.is_file()
                    else "model preparation will execute in the Worker-owned runtime"
                ),
            )

        if kind is SetupActionKind.PREFLIGHT:
            if self.root != Path("/"):
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    "runtime preflight requires the live target host",
                )
            try:
                expected = self._read_worker_gpu_uuids()
                if expected:
                    require_exact_gpu_set(
                        expected,
                        command=self.nvidia_smi,
                    )
            except Exception as exc:
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    f"GPU visibility preflight failed: {type(exc).__name__}",
                )
            if not self._target(self.runtime_manifest_path).is_file():
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    "runtime deployment manifest is missing",
                )
            return ActionInspection(
                SetupActionState.SATISFIED,
                "runtime and exact GPU visibility preflight passed",
            )

        if kind is SetupActionKind.RUNTIME_START:
            if self.root != Path("/"):
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    "service start requires the live target host",
                )
            return ActionInspection(
                SetupActionState.SATISFIED
                if self._service_active()
                else SetupActionState.NEEDS_APPLY,
                "Worker service owns the selected RuntimeProvider lifecycle",
            )

        if kind is SetupActionKind.HEALTH_CHECK:
            return ActionInspection(
                SetupActionState.SATISFIED
                if self._ready()
                else SetupActionState.NEEDS_APPLY,
                "Worker readiness includes RuntimeProvider readiness",
            )

        if kind in {
            SetupActionKind.RUNTIME_STOP,
            SetupActionKind.RUNTIME_RELEASE,
        }:
            return ActionInspection(
                SetupActionState.NEEDS_APPLY
                if self._service_active()
                else SetupActionState.SATISFIED,
                "Worker service lifecycle owns runtime stop/release",
            )

        return ActionInspection(
            SetupActionState.BLOCKED,
            f"unsupported systemd setup action: {kind.value}",
        )

    def apply(self, action: SetupAction) -> ActionReceipt:
        kind = action.kind
        provider_id = str(action.payload.get("provider_id") or "").strip()

        if kind is SetupActionKind.ENSURE_DIRECTORY:
            path = self._runtime_dir(
                provider_id,
                str(action.payload["logical_name"]),
            )
            existed = path.is_dir()
            path.mkdir(parents=True, exist_ok=True)
            return ActionReceipt(
                changed=not existed,
                detail=f"ensured {path}",
                rollback_data={"path": str(path), "created": not existed},
            )

        if kind is SetupActionKind.ENSURE_PACKAGE:
            package_reference = str(action.payload["package_reference"])
            if self._provider_executable_available(provider_id):
                return ActionReceipt(
                    changed=False,
                    detail="runtime executable already available",
                )
            installer = self._installer(provider_id)
            if installer is None:
                raise RuntimeError(
                    f"no explicit runtime installer configured for {provider_id}"
                )
            subprocess.run(
                [*installer, package_reference],
                check=True,
            )
            marker = self._package_marker(provider_id, package_reference)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(package_reference + "\n", encoding="utf-8")
            return ActionReceipt(
                changed=True,
                detail="runtime package installer completed",
                evidence={"package_reference": package_reference},
            )

        if kind is SetupActionKind.RENDER_CONFIG:
            rollback: dict[str, Any] = {"files": []}
            changed = False
            for path, content in self._render_targets(action):
                previous = (
                    path.read_text(encoding="utf-8")
                    if path.is_file()
                    else None
                )
                if previous == content:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + ".tmp")
                temporary.write_text(content, encoding="utf-8")
                os.chmod(temporary, 0o640)
                os.replace(temporary, path)
                changed = True
                rollback["files"].append(
                    {"path": str(path), "previous": previous}
                )
            return ActionReceipt(
                changed=changed,
                detail="rendered runtime deployment configuration",
                rollback_data=rollback,
            )

        if kind in {
            SetupActionKind.DOWNLOAD_MODEL,
            SetupActionKind.CONVERT_MODEL,
        }:
            marker = self._target(
                Path(
                    f"/var/lib/astrumweaver/runtime/{provider_id}/"
                    f"authorizations/{_action_marker(action)}"
                )
            )
            marker.parent.mkdir(parents=True, exist_ok=True)
            existed = marker.exists()
            marker.write_text(
                _canonical_json(action.to_dict()),
                encoding="utf-8",
            )
            return ActionReceipt(
                changed=not existed,
                detail=(
                    "authorized provider-owned model preparation at runtime start"
                ),
                evidence={"deferred_to_worker_runtime": True},
            )

        if kind is SetupActionKind.RUNTIME_START:
            subprocess.run(
                [self.systemctl, "enable", "--now", self.service_name],
                check=True,
            )
            return ActionReceipt(
                changed=True,
                detail="started Worker-owned runtime service",
                rollback_data={"service_started": True},
            )

        if kind is SetupActionKind.HEALTH_CHECK:
            if not self._ready():
                raise RuntimeError(
                    "Worker/RuntimeProvider readiness endpoint is not ready"
                )
            return ActionReceipt(
                changed=False,
                detail="Worker and selected runtime are ready",
            )

        if kind in {
            SetupActionKind.PREFLIGHT,
            SetupActionKind.VERIFY_MODEL_REFERENCE,
        }:
            inspection = self.inspect(action)
            if inspection.state is not SetupActionState.SATISFIED:
                raise RuntimeError(inspection.detail or "preflight failed")
            return ActionReceipt(changed=False, detail=inspection.detail)

        if kind in {
            SetupActionKind.RUNTIME_STOP,
            SetupActionKind.RUNTIME_RELEASE,
        }:
            if not self._service_active():
                return ActionReceipt(
                    changed=False,
                    detail="Worker-owned runtime is already stopped",
                )
            subprocess.run(
                [self.systemctl, "stop", self.service_name],
                check=True,
            )
            return ActionReceipt(
                changed=True,
                detail="stopped Worker-owned runtime service",
            )

        raise RuntimeError(
            f"unsupported systemd setup action: {kind.value}"
        )

    def rollback(
        self,
        action: SetupAction,
        receipt: ActionReceipt,
    ) -> ActionReceipt:
        if not receipt.changed:
            return ActionReceipt(changed=False, detail="no rollback required")

        if action.kind is SetupActionKind.ENSURE_DIRECTORY:
            path = Path(str(receipt.rollback_data.get("path") or ""))
            created = bool(receipt.rollback_data.get("created"))
            if created and path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    return ActionReceipt(
                        changed=False,
                        detail="directory retained because it is not empty",
                    )
                return ActionReceipt(
                    changed=True,
                    detail="removed empty created directory",
                )

        if action.kind is SetupActionKind.RENDER_CONFIG:
            changed = False
            for item in reversed(
                list(receipt.rollback_data.get("files") or ())
            ):
                path = Path(str(item["path"]))
                previous = item.get("previous")
                if previous is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_text(str(previous), encoding="utf-8")
                    os.chmod(path, 0o640)
                changed = True
            return ActionReceipt(
                changed=changed,
                detail="restored previous runtime configuration",
            )

        if action.kind is SetupActionKind.RUNTIME_START:
            subprocess.run(
                [self.systemctl, "stop", self.service_name],
                check=True,
            )
            return ActionReceipt(
                changed=True,
                detail="stopped Worker service started by this plan",
            )

        return ActionReceipt(
            changed=False,
            detail="action is not reversibly managed by systemd driver",
        )


def create_systemd_driver() -> SystemdSetupDriver:
    """Factory usable directly by astrumweaver-setup-tui --driver."""

    root = Path(os.environ.get("ASTRUMWEAVER_SETUP_ROOT", "/"))
    raw_installers = os.environ.get(
        "ASTRUMWEAVER_RUNTIME_INSTALLERS_JSON",
        "{}",
    )
    try:
        parsed = json.loads(raw_installers)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "ASTRUMWEAVER_RUNTIME_INSTALLERS_JSON must be valid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(
            "ASTRUMWEAVER_RUNTIME_INSTALLERS_JSON must contain an object"
        )
    installers: dict[str, tuple[str, ...]] = {}
    for provider_id, argv in parsed.items():
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) and item for item in argv)
        ):
            raise RuntimeError(
                "runtime installer entries must be non-empty argv arrays"
            )
        installers[str(provider_id)] = tuple(argv)

    return SystemdSetupDriver(
        root=root,
        worker_config_path=Path(
            os.environ.get(
                "ASTRUMWEAVER_WORKER_CONFIG",
                "/etc/astrumweaver/worker.toml",
            )
        ),
        runtime_manifest_path=Path(
            os.environ.get(
                "ASTRUMWEAVER_RUNTIME_MANIFEST",
                "/etc/astrumweaver/runtime-deployment.json",
            )
        ),
        service_name=os.environ.get(
            "ASTRUMWEAVER_WORKER_SERVICE",
            "astrumweaver-worker.service",
        ),
        systemctl=os.environ.get(
            "ASTRUMWEAVER_SYSTEMCTL",
            "systemctl",
        ),
        nvidia_smi=os.environ.get(
            "ASTRUMWEAVER_NVIDIA_SMI",
            "nvidia-smi",
        ),
        ready_url=os.environ.get(
            "ASTRUMWEAVER_WORKER_READY_URL",
            "http://127.0.0.1:9100/ready",
        ),
        installers=installers,
    )


__all__ = ["SystemdSetupDriver", "create_systemd_driver"]
