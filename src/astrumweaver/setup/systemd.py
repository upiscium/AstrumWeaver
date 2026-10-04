"""Generic systemd SetupActionDriver for RuntimeProvider deployment."""

from __future__ import annotations

import hashlib
import json
import grp
import os
import pwd
import re
import shutil
import stat
import subprocess
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from ..contracts import WorkerSpec
from ..worker.runtime import require_exact_gpu_set
from ..validation.runtime_deployment import (
    RuntimeDeploymentAcceptanceError,
    parse_systemd_show_properties,
    validate_effective_worker_gpu_isolation,
)
from .contracts import (
    ActionInspection,
    ActionReceipt,
    SetupAction,
    SetupActionKind,
    SetupActionState,
    thaw_json,
)
from .filesystem import SetupFilesystem, UnsafeSetupPath
from .migration import (
    DEFAULT_RUNTIME_MANIFEST,
    NVIDIA_DRIVER_BRIDGE_DIRECTORY,
    InstalledWorkerContract,
    InstalledWorkerUnit,
    parse_installed_worker_toml,
    parse_installed_worker_unit,
    render_runtime_worker_toml,
    render_runtime_worker_unit,
)


GENERIC_SYSTEMD_RUNTIME_PROFILE_MANAGER = (
    "/nix/var/nix/profiles/astrumweaver-installer/bin/"
    "astrumweaver-runtime-profile"
)
GENERIC_SYSTEMD_LLAMA_CPP_EXECUTABLE = (
    "/nix/var/nix/profiles/astrumweaver-runtime-llama-cpp/bin/llama-server"
)

_PROVIDER_EXECUTABLES: Mapping[str, str] = {
    "ollama": "ollama",
    "llama-cpp": "llama-server",
    "vllm": "vllm",
    "freetoken": "ft",
}
_GPU_DEVICE_PATH_RE = re.compile(r"^/dev/nvidia[0-9]+$")
_NVIDIA_DRIVER_SONAME = "libcuda.so.1"
_NVIDIA_DRIVER_BRIDGE_DIR = Path(NVIDIA_DRIVER_BRIDGE_DIRECTORY)
_SETUP_STATE_DIR = Path("/var/lib/astrumweaver-setup")
_PROVIDER_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        thaw_json(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    ) + "\n"


def _action_marker(action: SetupAction) -> str:
    payload = _canonical_json(action.to_dict()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SystemdSetupDriver:
    """Materialize reviewed runtime setup on an existing systemd Worker host.

    The standard generic-systemd llama.cpp path is reconciled through the
    packaged AstrumWeaver Nix runtime-profile manager. Other package/model
    mutation is allowed only through explicit operator-configured argv prefixes.
    No shell is involved and no distribution package manager is guessed.
    """

    def __init__(
        self,
        *,
        root: Path = Path("/"),
        worker_config_path: Path = Path("/etc/astrumweaver/worker.toml"),
        worker_unit_path: Path = Path(
            "/etc/systemd/system/astrumweaver-worker.service"
        ),
        runtime_manifest_path: Path = Path(
            "/etc/astrumweaver/runtime-deployment.json"
        ),
        gpu_uuid_file_path: Path = Path(
            "/etc/astrumweaver/gpu-uuids"
        ),
        gpu_device_map_path: Path = Path(
            "/etc/astrumweaver/gpu-device-map"
        ),
        gpu_isolation_dropin_path: Path = Path(
            "/etc/systemd/system/astrumweaver-worker.service.d/10-gpu-isolation.conf"
        ),
        gpu_device_map_command: str = "/usr/local/libexec/astrumweaver/gpu-device-map",
        service_name: str = "astrumweaver-worker.service",
        systemctl: str = "systemctl",
        nvidia_smi: str = "nvidia-smi",
        ldconfig: str = "ldconfig",
        nvidia_driver_library: Path | None = None,
        ready_url: str = "http://127.0.0.1:9100/ready",
        ready_poll_interval_seconds: float = 0.25,
        installers: Mapping[str, tuple[str, ...]] | None = None,
        downloaders: Mapping[str, tuple[str, ...]] | None = None,
        converters: Mapping[str, tuple[str, ...]] | None = None,
        verifiers: Mapping[str, tuple[str, ...]] | None = None,
        service_user: str = "astrumweaver",
        service_group: str = "astrumweaver",
    ) -> None:
        self.root = root
        self.filesystem = SetupFilesystem(root)
        self.worker_config_path = worker_config_path
        self.worker_unit_path = worker_unit_path
        self.runtime_manifest_path = runtime_manifest_path
        self.gpu_uuid_file_path = gpu_uuid_file_path
        self.gpu_device_map_path = gpu_device_map_path
        self.gpu_isolation_dropin_path = gpu_isolation_dropin_path
        self.gpu_device_map_command = gpu_device_map_command
        self.service_name = service_name
        self.systemctl = systemctl
        self.nvidia_smi = nvidia_smi
        self.ldconfig = ldconfig
        self.nvidia_driver_library = nvidia_driver_library
        self.ready_url = ready_url
        if ready_poll_interval_seconds <= 0:
            raise ValueError("ready_poll_interval_seconds must be positive")
        self.ready_poll_interval_seconds = ready_poll_interval_seconds
        self.installers = {
            str(key): tuple(str(item) for item in value)
            for key, value in dict(installers or {}).items()
        }
        self.downloaders = {
            str(key): tuple(str(item) for item in value)
            for key, value in dict(downloaders or {}).items()
        }
        self.converters = {
            str(key): tuple(str(item) for item in value)
            for key, value in dict(converters or {}).items()
        }
        self.verifiers = {
            str(key): tuple(str(item) for item in value)
            for key, value in dict(verifiers or {}).items()
        }
        self.service_user = service_user
        self.service_group = service_group
        # Live Worker restart is allowed only while the systemd manager is
        # known to reflect the on-disk execution authority. A failed
        # daemon-reload during migration/rollback makes restart fail closed.
        self._worker_restart_safe = True

    def _target(self, path: Path) -> Path:
        if self.root == Path("/"):
            return path
        return self.root / path.relative_to("/")

    @staticmethod
    def _validate_provider_id(provider_id: str) -> None:
        if not _PROVIDER_COMPONENT.fullmatch(provider_id):
            raise UnsafeSetupPath("provider identity must be a single safe path component")

    def _runtime_dir(self, provider_id: str, logical_name: str) -> Path:
        self._validate_provider_id(provider_id)
        bases = {
            "config": "/etc/astrumweaver/runtime",
            "state": "/var/lib/astrumweaver/runtime",
            "cache": "/var/cache/astrumweaver/runtime",
        }
        if logical_name not in bases:
            raise UnsafeSetupPath("unknown runtime directory role")
        return self._target(Path(bases[logical_name]) / provider_id)

    def _receipt_path(self, action: SetupAction) -> Path:
        provider_id = str(action.payload.get("provider_id") or "")
        self._validate_provider_id(provider_id)
        return self._target(
            _SETUP_STATE_DIR / provider_id / f"{_action_marker(action)}.json"
        )

    def _validate_receipt_path(self, action: SetupAction) -> None:
        # Validate any existing authoritative object but never use a receipt
        # (including legacy Worker-writable markers) as prerequisite evidence.
        try:
            with self.filesystem.directory(
                self._target(_SETUP_STATE_DIR), mode=0o700,
                uid=self.filesystem.owner,
            ):
                pass
        except FileNotFoundError:
            return
        self.filesystem.read_text(self._receipt_path(action))

    def _write_receipt(self, action: SetupAction) -> None:
        with self.filesystem.directory(
            self._target(_SETUP_STATE_DIR), create=True, mode=0o700,
            uid=self.filesystem.owner,
        ):
            pass
        self.filesystem.write_text(
            self._receipt_path(action),
            _canonical_json({"version": 1, "action_digest": _action_marker(action)}),
        )

    def _prerequisite_available(self, action: SetupAction) -> bool | None:
        provider_id = str(action.payload.get("provider_id") or "")
        self._validate_provider_id(provider_id)
        package = action.kind is SetupActionKind.ENSURE_PACKAGE
        reference = str(action.payload.get("package_reference" if package else "model_ref") or "")
        if not reference:
            raise RuntimeError("prerequisite reference must not be empty")
        verifier = self.verifiers.get(provider_id)
        if verifier:
            try:
                checked = subprocess.run(
                    [*verifier, "package" if package else "model", reference],
                    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=10.0,
                )
            except (OSError, subprocess.TimeoutExpired):
                raise RuntimeError("runtime prerequisite verifier failed") from None
            if checked.returncode not in (0, 1):
                raise RuntimeError("runtime prerequisite verifier returned an invalid status")
            return checked.returncode == 0
        if package:
            if provider_id not in _PROVIDER_EXECUTABLES:
                return None
            return self._provider_executable_available(provider_id, reference)
        if action.kind is SetupActionKind.DOWNLOAD_MODEL and reference.startswith("/"):
            # A model may legitimately be service-owned. It is read-only input
            # here; no privileged write follows its links or alters metadata.
            target = self._target(Path(reference))
            return target.is_file() or target.is_dir()
        return None

    def _discover_nvidia_driver_library(self) -> Path:
        if self.nvidia_driver_library is not None:
            candidate = self.nvidia_driver_library
        else:
            if self.root != Path("/"):
                raise RuntimeError(
                    "staged NVIDIA driver bridge requires an explicit "
                    "host driver-library path"
                )
            try:
                completed = subprocess.run(
                    [self.ldconfig, "-p"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except OSError as exc:
                raise RuntimeError(
                    "cannot execute ldconfig to discover libcuda.so.1"
                ) from exc
            if completed.returncode != 0:
                raise RuntimeError(
                    "ldconfig failed while discovering libcuda.so.1"
                )

            candidate = None
            for raw_line in completed.stdout.splitlines():
                line = raw_line.strip()
                if not line.startswith(_NVIDIA_DRIVER_SONAME + " "):
                    continue
                _, separator, raw_path = line.rpartition("=>")
                if not separator:
                    continue
                value = raw_path.strip()
                if value:
                    candidate = Path(value)
                    break
            if candidate is None:
                raise RuntimeError(
                    "host NVIDIA CUDA driver library libcuda.so.1 is unavailable"
                )

        if not candidate.is_absolute():
            raise RuntimeError(
                "host NVIDIA CUDA driver-library path must be absolute"
            )
        try:
            metadata = candidate.lstat()
            resolved = candidate.resolve(strict=True)
            resolved_metadata = resolved.stat()
        except OSError as exc:
            raise RuntimeError(
                "host NVIDIA CUDA driver library is unavailable"
            ) from exc
        if not (
            stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
        ) or not stat.S_ISREG(resolved_metadata.st_mode):
            raise RuntimeError(
                "host NVIDIA CUDA driver library is not a regular file"
            )
        if not os.access(resolved, os.R_OK):
            raise RuntimeError(
                "host NVIDIA CUDA driver library is not readable"
            )
        return candidate

    def _nvidia_driver_bridge_paths(
        self,
        action: SetupAction,
    ) -> tuple[Path, Path]:
        soname = str(action.payload.get("soname") or "").strip()
        bridge_directory = str(
            action.payload.get("bridge_directory") or ""
        ).strip()
        if soname != _NVIDIA_DRIVER_SONAME:
            raise RuntimeError(
                "NVIDIA driver bridge must use the reviewed libcuda.so.1 soname"
            )
        if bridge_directory != str(_NVIDIA_DRIVER_BRIDGE_DIR):
            raise RuntimeError(
                "NVIDIA driver bridge directory is not canonical"
            )
        source = self._discover_nvidia_driver_library()
        bridge_dir = self._target(_NVIDIA_DRIVER_BRIDGE_DIR)
        return source, bridge_dir / _NVIDIA_DRIVER_SONAME

    def _inspect_nvidia_driver_bridge(
        self,
        action: SetupAction,
    ) -> ActionInspection:
        try:
            source, target = self._nvidia_driver_bridge_paths(action)
        except RuntimeError as exc:
            return ActionInspection(SetupActionState.BLOCKED, str(exc))

        try:
            with self.filesystem.directory(
                target.parent, mode=0o755, uid=self.filesystem.owner,
            ):
                pass
            current = self.filesystem.read_link(target)
        except FileNotFoundError:
            current = None
        except UnsafeSetupPath as exc:
            return ActionInspection(SetupActionState.BLOCKED, str(exc))
        if current == str(source):
            return ActionInspection(
                SetupActionState.SATISFIED,
                "narrow libcuda.so.1 driver bridge is installed",
            )
        return ActionInspection(
            SetupActionState.NEEDS_APPLY,
            "host libcuda.so.1 requires a narrow runtime driver bridge",
        )

    def _provider_executable_available(
        self,
        provider_id: str,
        package_reference: str,
    ) -> bool:
        executable = _PROVIDER_EXECUTABLES.get(provider_id)
        if executable is None:
            return False
        return bool(shutil.which(executable))

    def _installer(self, provider_id: str) -> tuple[str, ...] | None:
        value = self.installers.get(provider_id)
        return value if value else None

    def _model_command(
        self,
        provider_id: str,
        kind: SetupActionKind,
    ) -> tuple[str, ...] | None:
        source = (
            self.downloaders
            if kind is SetupActionKind.DOWNLOAD_MODEL
            else self.converters
        )
        value = source.get(provider_id)
        return value if value else None

    def _service_ids(self) -> tuple[int, int]:
        try:
            uid = pwd.getpwnam(self.service_user).pw_uid
            gid = grp.getgrnam(self.service_group).gr_gid
        except KeyError as exc:
            raise RuntimeError(
                "AstrumWeaver service user/group must exist before runtime setup"
            ) from exc
        return uid, gid

    def _runtime_directory_policy(self, logical_name: str) -> dict[str, int]:
        if logical_name == "config":
            return {"mode": 0o755, "uid": self.filesystem.owner}
        uid, gid = self._service_ids() if self.root == Path("/") else (os.geteuid(), os.getegid())
        return {"mode": 0o750, "uid": uid, "gid": gid, "data_owner": uid}

    def _prepare_runtime_directory(
        self, provider_id: str, logical_name: str,
    ) -> tuple[Path, bool]:
        path = self._runtime_dir(provider_id, logical_name)
        policy = self._runtime_directory_policy(logical_name)
        try:
            with self.filesystem.directory(path, **policy):
                pass
            return path, True
        except FileNotFoundError:
            pass
        with self.filesystem.directory(path, create=True, **policy):
            pass
        return path, False

    def _write_nonsecret_config(self, path: Path, content: str) -> None:
        gid = self._service_ids()[1] if self.root == Path("/") else None
        self.filesystem.write_text(path, content, mode=0o640, gid=gid)

    def _read_required_text(self, path: Path, label: str) -> str:
        content = self.filesystem.read_text(self._target(path))
        if content is None:
            raise RuntimeError(f"{label} is unavailable")
        return content

    def _replace_preserving_metadata(self, path: Path, content: str) -> None:
        target = self._target(path)
        if self.filesystem.read_text(target) is None:
            raise RuntimeError("managed file is unavailable")
        self.filesystem.write_text(target, content, preserve_metadata=True)

    def _load_installed_worker_unit(
        self,
    ) -> tuple[str, InstalledWorkerUnit]:
        unit_text = self._read_required_text(
            self.worker_unit_path,
            "Worker unit",
        )
        unit = parse_installed_worker_unit(unit_text)
        if unit.service_user != self.service_user:
            raise RuntimeError(
                "installed Worker unit service identity differs from setup authority"
            )
        return unit_text, unit

    def load_installed_worker_contract(self) -> InstalledWorkerContract:
        worker_text = self._read_required_text(
            self.worker_config_path,
            "Worker config",
        )
        unit_text, unit = self._load_installed_worker_unit()
        contract = parse_installed_worker_toml(worker_text)
        if contract.execution_mode != unit.execution_mode:
            raise RuntimeError(
                "installed Worker config and unit use mixed execution authority"
            )
        return contract

    def load_installed_worker_spec(self) -> WorkerSpec:
        return self.load_installed_worker_contract().spec

    def _runtime_manifest_provider_id(self) -> str:
        path = self._target(self.runtime_manifest_path)
        try:
            value = json.loads(self.filesystem.read_text(path) or "")
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "runtime deployment manifest is unavailable or invalid"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(
                "runtime deployment manifest must contain an object"
            )
        provider_id = str(value.get("provider_id") or "").strip()
        if not provider_id:
            raise RuntimeError(
                "runtime deployment manifest lacks provider identity"
            )
        return provider_id

    def _expected_runtime_deployment_text(
        self,
        action: SetupAction,
    ) -> str:
        deployment = action.payload.get("runtime_deployment")
        if not isinstance(deployment, Mapping):
            raise RuntimeError(
                "Worker migration action lacks the reviewed runtime deployment"
            )
        return _canonical_json(deployment)

    def _runtime_manifest_matches(self, action: SetupAction) -> bool:
        expected = self._expected_runtime_deployment_text(action)
        path = self._target(self.runtime_manifest_path)
        try:
            value = json.loads(self.filesystem.read_text(path) or "")
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(value, dict):
            return False
        return _canonical_json(value) == expected

    def _inspect_worker_stop(
        self,
        action: SetupAction,
    ) -> ActionInspection:
        if (
            action.payload.get("source_execution")
            != "canonical_smoke_or_runtime"
        ):
            return ActionInspection(
                SetupActionState.BLOCKED,
                "Worker stop source contract is not the reviewed canonical state",
            )
        if action.payload.get("desired_execution") != "runtime":
            return ActionInspection(
                SetupActionState.BLOCKED,
                "Worker stop target contract is not runtime execution",
            )
        try:
            contract = self.load_installed_worker_contract()
            self._expected_runtime_deployment_text(action)
        except RuntimeError as exc:
            return ActionInspection(SetupActionState.BLOCKED, str(exc))

        if contract.execution_mode == "runtime":
            if not self._runtime_manifest_matches(action):
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    "installed runtime deployment differs from the reviewed desired state",
                )
            try:
                _unit_text, unit = self._load_installed_worker_unit()
            except RuntimeError as exc:
                return ActionInspection(SetupActionState.BLOCKED, str(exc))
            if unit.nvidia_driver_bridge:
                return ActionInspection(
                    SetupActionState.SATISFIED,
                    "current runtime Worker needs no execution-authority stop",
                )
            if self.root != Path("/"):
                return ActionInspection(
                    SetupActionState.SATISFIED,
                    "staged root has no live Worker service to stop",
                )
            return ActionInspection(
                SetupActionState.NEEDS_APPLY
                if self._service_active()
                else SetupActionState.SATISFIED,
                "prior runtime Worker must be stopped before driver-bridge reconciliation",
            )

        if self.root != Path("/"):
            return ActionInspection(
                SetupActionState.SATISFIED,
                "staged root has no live Worker service to stop",
            )
        return ActionInspection(
            SetupActionState.NEEDS_APPLY
            if self._service_active()
            else SetupActionState.SATISFIED,
            "existing smoke Worker must be stopped before execution reconciliation",
        )

    def _inspect_worker_execution(
        self,
        action: SetupAction,
    ) -> ActionInspection:
        provider_id = str(action.payload.get("provider_id") or "").strip()
        try:
            self._expected_runtime_deployment_text(action)
        except RuntimeError as exc:
            return ActionInspection(SetupActionState.BLOCKED, str(exc))
        if not provider_id:
            return ActionInspection(
                SetupActionState.BLOCKED,
                "Worker execution reconciliation lacks provider identity",
            )
        if (
            action.payload.get("source_execution")
            != "canonical_smoke_or_runtime"
        ):
            return ActionInspection(
                SetupActionState.BLOCKED,
                "Worker execution source contract is not the reviewed canonical state",
            )
        if action.payload.get("desired_execution") != "runtime":
            return ActionInspection(
                SetupActionState.BLOCKED,
                "Worker execution target contract is not runtime execution",
            )
        if (
            str(action.payload.get("runtime_manifest") or "").strip()
            != DEFAULT_RUNTIME_MANIFEST
        ):
            return ActionInspection(
                SetupActionState.BLOCKED,
                "Worker execution target uses a non-canonical runtime manifest",
            )

        try:
            contract = self.load_installed_worker_contract()
        except RuntimeError as exc:
            return ActionInspection(SetupActionState.BLOCKED, str(exc))

        if contract.execution_mode == "smoke":
            return ActionInspection(
                SetupActionState.NEEDS_APPLY,
                "canonical smoke Worker requires reviewed runtime reconciliation",
            )

        try:
            manifest_provider = self._runtime_manifest_provider_id()
        except RuntimeError as exc:
            return ActionInspection(SetupActionState.BLOCKED, str(exc))
        if manifest_provider != provider_id:
            return ActionInspection(
                SetupActionState.BLOCKED,
                "installed runtime provider differs from the reviewed provider",
            )
        if not self._runtime_manifest_matches(action):
            return ActionInspection(
                SetupActionState.BLOCKED,
                "installed runtime deployment differs from the reviewed desired state",
            )
        try:
            unit_text, _unit = self._load_installed_worker_unit()
            desired_unit = render_runtime_worker_unit(unit_text)
        except RuntimeError as exc:
            return ActionInspection(SetupActionState.BLOCKED, str(exc))
        if unit_text != desired_unit:
            return ActionInspection(
                SetupActionState.NEEDS_APPLY,
                "prior canonical runtime Worker requires reviewed driver-bridge reconciliation",
            )
        return ActionInspection(
            SetupActionState.SATISFIED,
            "Worker config and unit already own the exact reviewed RuntimeProvider state",
        )

    def _validate_render_migration_source(
        self,
        action: SetupAction,
    ) -> ActionInspection | None:
        if not bool(action.payload.get("existing_worker_migration")):
            return None
        provider_id = str(action.payload.get("provider_id") or "").strip()
        try:
            contract = self.load_installed_worker_contract()
        except RuntimeError as exc:
            return ActionInspection(SetupActionState.BLOCKED, str(exc))
        if contract.execution_mode == "runtime":
            try:
                current_provider = self._runtime_manifest_provider_id()
                targets = self._render_targets(action)
            except RuntimeError as exc:
                return ActionInspection(SetupActionState.BLOCKED, str(exc))
            if current_provider != provider_id:
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    "installed runtime provider differs from the reviewed provider",
                )
            if any(
                self.filesystem.read_text(path) != content
                for path, content in targets
            ):
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    "already-migrated Worker differs from the reviewed desired runtime state",
                )
        return None

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

    def _gpu_isolation_verified(
        self,
        expected: tuple[str, ...],
    ) -> bool:
        expected_file = self._target(self.gpu_uuid_file_path)
        map_file = self._target(self.gpu_device_map_path)
        dropin = self._target(self.gpu_isolation_dropin_path)

        if not expected_file.is_file() or not map_file.is_file() or not dropin.is_file():
            return False

        configured_expected = tuple(
            sorted(
                line.strip()
                for line in expected_file.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
        )
        if configured_expected != tuple(sorted(expected)):
            return False

        mapping: dict[str, str] = {}
        for raw_line in map_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            uuid, path = line.split("=", 1)
            uuid = uuid.strip()
            path = path.strip()
            if (
                not uuid
                or not _GPU_DEVICE_PATH_RE.fullmatch(path)
                or uuid in mapping
            ):
                return False
            mapping[uuid] = path

        if tuple(sorted(mapping)) != tuple(sorted(expected)):
            return False
        if len(set(mapping.values())) != len(mapping):
            return False

        if self.root != Path("/"):
            return False

        try:
            properties_result = subprocess.run(
                [
                    self.systemctl,
                    "show",
                    "--no-pager",
                    "--all",
                    "--property=DevicePolicy,DeviceAllow,Environment,EnvironmentFiles,UnsetEnvironment,ExecStartPre,Requires,After",
                    self.service_name,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if properties_result.returncode != 0:
                return False
            properties = parse_systemd_show_properties(
                properties_result.stdout
            )
            validate_effective_worker_gpu_isolation(
                properties,
                expected_gpu_uuids=expected,
                expected_device_paths=set(mapping.values()),
            )
        except (OSError, RuntimeDeploymentAcceptanceError):
            return False

        completed = subprocess.run(
            [
                self.gpu_device_map_command,
                "--nvidia-smi",
                self.nvidia_smi,
                "verify",
                str(expected_file),
                str(map_file),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.returncode == 0

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

    def _service_failed(self) -> bool:
        if self.root != Path("/"):
            return False
        try:
            completed = subprocess.run(
                [self.systemctl, "is-failed", "--quiet", self.service_name],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return False
        return completed.returncode == 0

    @staticmethod
    def _health_timeout_seconds(action: SetupAction) -> float:
        value = action.payload.get("timeout_seconds")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(
                "health check lacks an explicit reviewed startup timeout"
            )
        timeout_seconds = float(value)
        if timeout_seconds <= 0:
            raise RuntimeError(
                "health check startup timeout must be positive"
            )
        return timeout_seconds

    def _wait_ready(self, *, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while True:
            if self._ready():
                return True
            if self._service_failed():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(
                min(self.ready_poll_interval_seconds, remaining)
            )

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
        try:
            return self._inspect(action)
        except UnsafeSetupPath as exc:
            return ActionInspection(SetupActionState.BLOCKED, str(exc))

    def _inspect(self, action: SetupAction) -> ActionInspection:
        kind = action.kind
        provider_id = str(action.payload.get("provider_id") or "").strip()

        if kind is SetupActionKind.ENSURE_DIRECTORY:
            logical_name = str(action.payload["logical_name"])
            path = self._runtime_dir(provider_id, logical_name)
            try:
                with self.filesystem.directory(path, **self._runtime_directory_policy(logical_name)):
                    pass
            except FileNotFoundError:
                return ActionInspection(SetupActionState.NEEDS_APPLY, "runtime directory is missing")
            return ActionInspection(SetupActionState.SATISFIED, "runtime directory has safe ownership and mode")

        if kind is SetupActionKind.ENSURE_PACKAGE:
            self._validate_receipt_path(action)
            available = self._prerequisite_available(action)
            if available is True:
                return ActionInspection(SetupActionState.SATISFIED, "runtime prerequisite is currently available")
            if self._installer(provider_id) is None:
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    "runtime package is unavailable and no explicit installer is configured",
                )
            if available is None:
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    "runtime package requires an explicit reviewed read-only verifier",
                )
            return ActionInspection(SetupActionState.NEEDS_APPLY, "runtime package requires explicit installer")

        if kind is SetupActionKind.ENSURE_NVIDIA_DRIVER_BRIDGE:
            return self._inspect_nvidia_driver_bridge(action)

        if kind is SetupActionKind.RENDER_CONFIG:
            migration_block = self._validate_render_migration_source(action)
            if migration_block is not None:
                return migration_block
            try:
                targets = self._render_targets(action)
            except RuntimeError as exc:
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    str(exc),
                )
            satisfied = all(
                self.filesystem.read_text(path) == content
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

        if kind in {SetupActionKind.DOWNLOAD_MODEL, SetupActionKind.CONVERT_MODEL}:
            self._validate_receipt_path(action)
            available = self._prerequisite_available(action)
            if available is True:
                return ActionInspection(SetupActionState.SATISFIED, "model prerequisite is currently available")
            if self._model_command(provider_id, kind) is None:
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    f"{kind.value} requires an explicit reviewed model command",
                )
            if available is None:
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    "model preparation requires an explicit reviewed read-only verifier",
                )
            return ActionInspection(SetupActionState.NEEDS_APPLY, "model preparation command is configured")

        if kind is SetupActionKind.PREFLIGHT:
            if self.root != Path("/"):
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    "runtime preflight requires the live target host",
                )

            expected = self._read_worker_gpu_uuids()
            visibility_detail = "no GPU visibility requirement"
            if expected:
                try:
                    require_exact_gpu_set(
                        expected,
                        command=self.nvidia_smi,
                    )
                except Exception as exc:
                    if not self._gpu_isolation_verified(expected):
                        return ActionInspection(
                            SetupActionState.BLOCKED,
                            (
                                "GPU visibility preflight failed and no "
                                "verified service device isolation exists: "
                                f"{type(exc).__name__}"
                            ),
                        )
                    visibility_detail = (
                        "host GPU superset accepted only because the reviewed "
                        "systemd isolation contract is configured; service "
                        "ExecStartPre must still prove isolated-access inside "
                        "the final Worker cgroup"
                    )
                else:
                    visibility_detail = "host-visible GPU set is already exact"

            if not self._target(self.runtime_manifest_path).is_file():
                return ActionInspection(
                    SetupActionState.NEEDS_APPLY,
                    "runtime deployment manifest will be materialized by the reviewed plan",
                )
            return ActionInspection(
                SetupActionState.SATISFIED,
                "runtime preflight passed; " + visibility_detail,
            )

        if kind is SetupActionKind.WORKER_STOP:
            return self._inspect_worker_stop(action)

        if kind is SetupActionKind.RECONCILE_WORKER_EXECUTION:
            return self._inspect_worker_execution(action)

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
            try:
                timeout_seconds = self._health_timeout_seconds(action)
            except RuntimeError as exc:
                return ActionInspection(
                    SetupActionState.BLOCKED,
                    str(exc),
                )
            return ActionInspection(
                SetupActionState.SATISFIED
                if self._ready()
                else SetupActionState.NEEDS_APPLY,
                (
                    "Worker readiness includes RuntimeProvider readiness; "
                    f"apply may wait up to {timeout_seconds:g} seconds"
                ),
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
            path, existed = self._prepare_runtime_directory(
                provider_id,
                str(action.payload["logical_name"]),
            )
            return ActionReceipt(
                changed=not existed,
                detail=f"ensured {path}",
                rollback_data={"path": str(path), "created": not existed},
            )

        if kind is SetupActionKind.ENSURE_PACKAGE:
            inspection = self.inspect(action)
            if inspection.state is SetupActionState.BLOCKED:
                raise RuntimeError(inspection.detail)
            if inspection.state is SetupActionState.SATISFIED:
                return ActionReceipt(changed=False, detail=inspection.detail)
            installer = self._installer(provider_id)
            assert installer is not None
            package_reference = str(action.payload["package_reference"])
            subprocess.run([*installer, package_reference], check=True)
            if self._prerequisite_available(action) is not True:
                raise RuntimeError("runtime prerequisite is still unavailable after installer completed")
            self._write_receipt(action)
            return ActionReceipt(changed=True, detail="runtime package installer completed and verified")

        if kind is SetupActionKind.ENSURE_NVIDIA_DRIVER_BRIDGE:
            inspection = self._inspect_nvidia_driver_bridge(action)
            if inspection.state is SetupActionState.SATISFIED:
                return ActionReceipt(changed=False, detail=inspection.detail)
            if inspection.state is SetupActionState.BLOCKED:
                raise RuntimeError(
                    inspection.detail or "NVIDIA driver bridge is blocked"
                )

            source, target = self._nvidia_driver_bridge_paths(action)
            created_dir = False
            try:
                with self.filesystem.directory(target.parent, mode=0o755, uid=self.filesystem.owner):
                    pass
            except FileNotFoundError:
                created_dir = True
                with self.filesystem.directory(target.parent, create=True, mode=0o755, uid=self.filesystem.owner):
                    pass
            previous_target = self.filesystem.read_link(target)
            self.filesystem.replace_link(target, str(source), expected=previous_target)
            return ActionReceipt(
                changed=True,
                detail="installed narrow host libcuda.so.1 runtime bridge",
                rollback_data={
                    "target": str(target), "installed_target": str(source),
                    "previous_target": previous_target, "created_dir": created_dir,
                },
                evidence={"soname": _NVIDIA_DRIVER_SONAME},
            )

        if kind is SetupActionKind.RENDER_CONFIG:
            migration_block = self._validate_render_migration_source(action)
            if migration_block is not None:
                raise RuntimeError(
                    migration_block.detail
                    or "runtime configuration migration is blocked"
                )
            rollback: dict[str, Any] = {"files": []}
            # Validate every target before the first write; restore earlier
            # writes when one action fails before it can return its receipt.
            targets = tuple(
                (path, content, self.filesystem.read_text(path))
                for path, content in self._render_targets(action)
            )
            try:
                for path, content, previous in targets:
                    if previous == content:
                        continue
                    self._write_nonsecret_config(path, content)
                    rollback["files"].append(
                        {"path": str(path), "previous": previous, "installed": content}
                    )
            except Exception:
                self._restore_rendered_files(rollback["files"])
                raise
            return ActionReceipt(
                changed=bool(rollback["files"]),
                detail="rendered runtime deployment configuration", rollback_data=rollback,
            )

        if kind in {SetupActionKind.DOWNLOAD_MODEL, SetupActionKind.CONVERT_MODEL}:
            inspection = self.inspect(action)
            if inspection.state is SetupActionState.BLOCKED:
                raise RuntimeError(inspection.detail)
            if inspection.state is SetupActionState.SATISFIED:
                return ActionReceipt(changed=False, detail=inspection.detail)
            command = self._model_command(provider_id, kind)
            assert command is not None
            model_ref = str(action.payload["model_ref"])
            subprocess.run([*command, model_ref], check=True)
            if self._prerequisite_available(action) is not True:
                raise RuntimeError("model prerequisite is still unavailable after preparation")
            self._write_receipt(action)
            return ActionReceipt(changed=True, detail="model preparation completed and verified")

        if kind is SetupActionKind.WORKER_STOP:
            inspection = self._inspect_worker_stop(action)
            if inspection.state is SetupActionState.SATISFIED:
                return ActionReceipt(changed=False, detail=inspection.detail)
            if inspection.state is SetupActionState.BLOCKED:
                raise RuntimeError(
                    inspection.detail or "Worker stop is blocked"
                )
            subprocess.run(
                [self.systemctl, "stop", self.service_name],
                check=True,
            )
            return ActionReceipt(
                changed=True,
                detail="stopped existing Worker before execution reconciliation",
                rollback_data={"was_active": True},
            )

        if kind is SetupActionKind.RECONCILE_WORKER_EXECUTION:
            inspection = self._inspect_worker_execution(action)
            if inspection.state is SetupActionState.SATISFIED:
                return ActionReceipt(
                    changed=False,
                    detail=inspection.detail,
                )
            if inspection.state is SetupActionState.BLOCKED:
                raise RuntimeError(
                    inspection.detail or "Worker execution reconciliation is blocked"
                )
            if self.root == Path("/") and self._service_active():
                raise RuntimeError(
                    "Worker service is still active; refusing to change execution authority"
                )

            worker_path = self._target(self.worker_config_path)
            unit_path = self._target(self.worker_unit_path)
            worker_previous = self._read_required_text(
                self.worker_config_path,
                "Worker config",
            )
            unit_previous = self._read_required_text(
                self.worker_unit_path,
                "Worker unit",
            )
            worker_desired = render_runtime_worker_toml(worker_previous)
            unit_desired = render_runtime_worker_unit(unit_previous)

            worker_changed = worker_previous != worker_desired
            unit_changed = unit_previous != unit_desired
            if self.root == Path("/") and (worker_changed or unit_changed):
                self._worker_restart_safe = False
            try:
                if worker_changed:
                    self._replace_preserving_metadata(
                        self.worker_config_path,
                        worker_desired,
                    )
                if unit_changed:
                    self._replace_preserving_metadata(
                        self.worker_unit_path,
                        unit_desired,
                    )
                if self.root == Path("/") and unit_changed:
                    subprocess.run(
                        [self.systemctl, "daemon-reload"],
                        check=True,
                    )
                if self.root == Path("/"):
                    self._worker_restart_safe = True
            except Exception:
                try:
                    if worker_changed and worker_path.is_file():
                        self._replace_preserving_metadata(
                            self.worker_config_path,
                            worker_previous,
                        )
                    if unit_changed and unit_path.is_file():
                        self._replace_preserving_metadata(
                            self.worker_unit_path,
                            unit_previous,
                        )
                    if self.root == Path("/") and unit_changed:
                        subprocess.run(
                            [self.systemctl, "daemon-reload"],
                            check=True,
                        )
                except Exception:
                    if self.root == Path("/"):
                        self._worker_restart_safe = False
                else:
                    if self.root == Path("/"):
                        self._worker_restart_safe = True
                raise

            return ActionReceipt(
                changed=worker_changed or unit_changed,
                detail=(
                    "reconciled Worker config and unit to reviewed runtime execution"
                ),
                rollback_data={
                    "worker_previous": worker_previous,
                    "unit_previous": unit_previous,
                },
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
            timeout_seconds = self._health_timeout_seconds(action)
            if not self._wait_ready(timeout_seconds=timeout_seconds):
                if self._service_failed():
                    raise RuntimeError(
                        "Worker service entered failed state before readiness"
                    )
                raise RuntimeError(
                    "Worker/RuntimeProvider readiness endpoint did not become "
                    "ready before the reviewed startup timeout"
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

    def _restore_rendered_files(self, items: list[dict[str, Any]]) -> None:
        for item in reversed(items):
            path = Path(str(item["path"]))
            current = self.filesystem.read_text(path)
            if current != item["installed"]:
                raise UnsafeSetupPath("runtime configuration changed after apply; refusing rollback")
            if item["previous"] is None:
                self.filesystem.remove_file(path, expected=item["installed"])
            else:
                self._write_nonsecret_config(path, str(item["previous"]))

    def rollback(
        self,
        action: SetupAction,
        receipt: ActionReceipt,
    ) -> ActionReceipt:
        if not receipt.changed:
            return ActionReceipt(changed=False, detail="no rollback required")

        if action.kind is SetupActionKind.ENSURE_DIRECTORY:
            provider_id = str(action.payload.get("provider_id") or "")
            logical_name = str(action.payload["logical_name"])
            path = self._runtime_dir(provider_id, logical_name)
            created = bool(receipt.rollback_data.get("created"))
            removed = created and self.filesystem.remove_directory(
                path, data_owner=self._runtime_directory_policy(logical_name).get("data_owner"),
            )
            return ActionReceipt(changed=removed, detail="removed empty directory" if removed else "directory retained")

        if action.kind is SetupActionKind.ENSURE_NVIDIA_DRIVER_BRIDGE:
            target = self._target(_NVIDIA_DRIVER_BRIDGE_DIR) / _NVIDIA_DRIVER_SONAME
            installed = str(receipt.rollback_data.get("installed_target") or "")
            previous = receipt.rollback_data.get("previous_target")
            if previous is None:
                self.filesystem.remove_link(target, expected=installed)
            elif isinstance(previous, str) and previous:
                self.filesystem.replace_link(target, previous, expected=installed)
            else:
                raise RuntimeError("NVIDIA driver bridge rollback receipt is invalid")
            if receipt.rollback_data.get("created_dir"):
                self.filesystem.remove_directory(target.parent)
            return ActionReceipt(changed=True, detail="restored previous NVIDIA driver bridge state")

        if action.kind is SetupActionKind.RENDER_CONFIG:
            items = list(receipt.rollback_data.get("files") or ())
            self._restore_rendered_files(items)
            return ActionReceipt(changed=bool(items), detail="restored previous runtime configuration")

        if action.kind is SetupActionKind.RECONCILE_WORKER_EXECUTION:
            worker_previous = receipt.rollback_data.get("worker_previous")
            unit_previous = receipt.rollback_data.get("unit_previous")
            if not isinstance(worker_previous, str) or not isinstance(
                unit_previous, str
            ):
                raise RuntimeError(
                    "Worker execution rollback receipt is incomplete"
                )
            if self.root == Path("/"):
                self._worker_restart_safe = False
                if self._service_active():
                    subprocess.run(
                        [self.systemctl, "stop", self.service_name],
                        check=True,
                    )
            self._replace_preserving_metadata(
                self.worker_config_path,
                worker_previous,
            )
            self._replace_preserving_metadata(
                self.worker_unit_path,
                unit_previous,
            )
            if self.root == Path("/"):
                subprocess.run(
                    [self.systemctl, "daemon-reload"],
                    check=True,
                )
                self._worker_restart_safe = True
            return ActionReceipt(
                changed=True,
                detail="restored previous Worker execution configuration",
            )

        if action.kind is SetupActionKind.WORKER_STOP:
            if not bool(receipt.rollback_data.get("was_active")):
                return ActionReceipt(
                    changed=False,
                    detail="Worker was already stopped before migration",
                )
            if self.root != Path("/"):
                return ActionReceipt(
                    changed=False,
                    detail="staged root has no live Worker service to restore",
                )
            if not self._worker_restart_safe:
                return ActionReceipt(
                    changed=False,
                    detail=(
                        "Worker restart refused because systemd execution "
                        "authority reload was not proven safe"
                    ),
                )
            try:
                restored = self.load_installed_worker_contract()
            except RuntimeError:
                return ActionReceipt(
                    changed=False,
                    detail=(
                        "Worker left stopped because rollback did not restore "
                        "a recognizable execution contract"
                    ),
                )
            if restored.execution_mode != "smoke":
                return ActionReceipt(
                    changed=False,
                    detail=(
                        "Worker left stopped because smoke execution authority "
                        "was not restored"
                    ),
                )
            subprocess.run(
                [self.systemctl, "start", self.service_name],
                check=True,
            )
            return ActionReceipt(
                changed=True,
                detail="restored previously active smoke Worker service",
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


def _argv_map_from_env(name: str) -> dict[str, tuple[str, ...]]:
    raw = os.environ.get(name, "{}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{name} must contain an object")
    values: dict[str, tuple[str, ...]] = {}
    for provider_id, argv in parsed.items():
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) and item for item in argv)
        ):
            raise RuntimeError(
                f"{name} entries must be non-empty argv arrays"
            )
        values[str(provider_id)] = tuple(argv)
    return values


def _argv_map_with_defaults(
    name: str,
    defaults: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    values = {
        str(provider_id): tuple(argv)
        for provider_id, argv in defaults.items()
    }
    values.update(_argv_map_from_env(name))
    return values


def create_systemd_driver() -> SystemdSetupDriver:
    """Factory usable directly by astrumweaver-setup-tui --driver."""

    root = Path(os.environ.get("ASTRUMWEAVER_SETUP_ROOT", "/"))
    runtime_profile_manager = os.environ.get(
        "ASTRUMWEAVER_RUNTIME_PROFILE_MANAGER",
        GENERIC_SYSTEMD_RUNTIME_PROFILE_MANAGER,
    ).strip()
    if not runtime_profile_manager or not Path(runtime_profile_manager).is_absolute():
        raise RuntimeError(
            "ASTRUMWEAVER_RUNTIME_PROFILE_MANAGER must be an absolute path"
        )
    standard_installers = {
        "llama-cpp": (runtime_profile_manager, "ensure"),
    }
    standard_verifiers = {
        "llama-cpp": (runtime_profile_manager, "verify"),
    }
    return SystemdSetupDriver(
        root=root,
        worker_config_path=Path(
            os.environ.get(
                "ASTRUMWEAVER_WORKER_CONFIG",
                "/etc/astrumweaver/worker.toml",
            )
        ),
        worker_unit_path=Path(
            os.environ.get(
                "ASTRUMWEAVER_WORKER_UNIT",
                "/etc/systemd/system/astrumweaver-worker.service",
            )
        ),
        runtime_manifest_path=Path(
            os.environ.get(
                "ASTRUMWEAVER_RUNTIME_MANIFEST",
                "/etc/astrumweaver/runtime-deployment.json",
            )
        ),
        gpu_uuid_file_path=Path(
            os.environ.get(
                "ASTRUMWEAVER_GPU_UUID_FILE",
                "/etc/astrumweaver/gpu-uuids",
            )
        ),
        gpu_device_map_path=Path(
            os.environ.get(
                "ASTRUMWEAVER_GPU_DEVICE_MAP",
                "/etc/astrumweaver/gpu-device-map",
            )
        ),
        gpu_isolation_dropin_path=Path(
            os.environ.get(
                "ASTRUMWEAVER_GPU_ISOLATION_DROPIN",
                "/etc/systemd/system/astrumweaver-worker.service.d/10-gpu-isolation.conf",
            )
        ),
        gpu_device_map_command=os.environ.get(
            "ASTRUMWEAVER_GPU_DEVICE_MAP_COMMAND",
            "/usr/local/libexec/astrumweaver/gpu-device-map",
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
        ldconfig=os.environ.get(
            "ASTRUMWEAVER_LDCONFIG",
            "ldconfig",
        ),
        nvidia_driver_library=(
            Path(value)
            if (value := os.environ.get(
                "ASTRUMWEAVER_NVIDIA_DRIVER_LIBRARY",
                "",
            ).strip())
            else None
        ),
        ready_url=os.environ.get(
            "ASTRUMWEAVER_WORKER_READY_URL",
            "http://127.0.0.1:9100/ready",
        ),
        ready_poll_interval_seconds=float(
            os.environ.get(
                "ASTRUMWEAVER_WORKER_READY_POLL_INTERVAL_SECONDS",
                "0.25",
            )
        ),
        installers=_argv_map_with_defaults(
            "ASTRUMWEAVER_RUNTIME_INSTALLERS_JSON",
            standard_installers,
        ),
        downloaders=_argv_map_from_env(
            "ASTRUMWEAVER_RUNTIME_DOWNLOADERS_JSON"
        ),
        converters=_argv_map_from_env(
            "ASTRUMWEAVER_RUNTIME_CONVERTERS_JSON"
        ),
        verifiers=_argv_map_with_defaults(
            "ASTRUMWEAVER_RUNTIME_VERIFIERS_JSON",
            standard_verifiers,
        ),
        service_user=os.environ.get(
            "ASTRUMWEAVER_WORKER_USER",
            "astrumweaver",
        ),
        service_group=os.environ.get(
            "ASTRUMWEAVER_WORKER_GROUP",
            "astrumweaver",
        ),
    )


__all__ = [
    "GENERIC_SYSTEMD_LLAMA_CPP_EXECUTABLE",
    "GENERIC_SYSTEMD_RUNTIME_PROFILE_MANAGER",
    "SystemdSetupDriver",
    "create_systemd_driver",
]
