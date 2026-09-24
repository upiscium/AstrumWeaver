"""Deterministic setup-plan contracts shared by CLI automation and TUI."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from ..runtime import RuntimeHostFacts


_ACTION_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _nonblank(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


@dataclass(frozen=True, slots=True)
class SecretReference:
    """Reference to protected secret material; never stores the secret value."""

    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonblank(self.name, "name"))

    def to_dict(self) -> dict[str, str]:
        return {"$secret_ref": self.name}


def _freeze_json(value: Any) -> Any:
    if isinstance(value, SecretReference):
        return MappingProxyType(value.to_dict())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


class DeploymentPath(StrEnum):
    NIXOS = "nixos"
    SYSTEMD = "systemd"


class PrivilegeMode(StrEnum):
    ROOT = "root"
    SUDO = "sudo"
    UNAVAILABLE = "unavailable"


class SetupPlanGoal(StrEnum):
    ACTIVATE = "activate"
    RELEASE = "release"


class SetupActionKind(StrEnum):
    ENSURE_DIRECTORY = "ensure_directory"
    ENSURE_PACKAGE = "ensure_package"
    RENDER_CONFIG = "render_config"
    VERIFY_MODEL_REFERENCE = "verify_model_reference"
    DOWNLOAD_MODEL = "download_model"
    CONVERT_MODEL = "convert_model"
    PREFLIGHT = "preflight"
    RUNTIME_START = "runtime_start"
    HEALTH_CHECK = "health_check"
    RUNTIME_STOP = "runtime_stop"
    RUNTIME_RELEASE = "runtime_release"


class SetupActionState(StrEnum):
    SATISFIED = "satisfied"
    NEEDS_APPLY = "needs_apply"
    BLOCKED = "blocked"


class SetupActionResultStatus(StrEnum):
    SKIPPED = "skipped"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ROLLBACK_FAILED = "rollback_failed"
    BLOCKED = "blocked"


class SetupApplyStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SetupHostSnapshot:
    """Host discovery snapshot consumed by deterministic planning.

    It intentionally excludes hostname, IP addresses and site topology.
    """

    runtime_host: RuntimeHostFacts
    deployment_path: DeploymentPath
    os_id: str
    os_version: str = ""
    service_manager: str | None = None
    package_manager: str | None = None
    available_commands: frozenset[str] = field(default_factory=frozenset)
    privilege_mode: PrivilegeMode = PrivilegeMode.UNAVAILABLE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_host, RuntimeHostFacts):
            raise TypeError("runtime_host must be RuntimeHostFacts")
        object.__setattr__(
            self,
            "deployment_path",
            DeploymentPath(self.deployment_path),
        )
        object.__setattr__(self, "os_id", _nonblank(self.os_id, "os_id").lower())
        object.__setattr__(self, "os_version", str(self.os_version).strip())
        if self.service_manager is not None:
            object.__setattr__(
                self,
                "service_manager",
                _nonblank(self.service_manager, "service_manager"),
            )
        if self.package_manager is not None:
            object.__setattr__(
                self,
                "package_manager",
                _nonblank(self.package_manager, "package_manager"),
            )
        commands = frozenset(
            _nonblank(value, "available command")
            for value in self.available_commands
        )
        object.__setattr__(self, "available_commands", commands)
        object.__setattr__(
            self,
            "privilege_mode",
            PrivilegeMode(self.privilege_mode),
        )
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_host": {
                "cpu_count": self.runtime_host.cpu_count,
                "host_ram_mb": self.runtime_host.host_ram_mb,
                "architecture": self.runtime_host.architecture,
                "labels": dict(self.runtime_host.labels),
            },
            "deployment_path": self.deployment_path.value,
            "os_id": self.os_id,
            "os_version": self.os_version,
            "service_manager": self.service_manager,
            "package_manager": self.package_manager,
            "available_commands": sorted(self.available_commands),
            "privilege_mode": self.privilege_mode.value,
            "metadata": thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SetupAction:
    action_id: str
    kind: SetupActionKind
    description: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    requires_privilege: bool = False
    requires_network: bool = False
    requires_confirmation: bool = False
    reversible: bool = False

    def __post_init__(self) -> None:
        action_id = _nonblank(self.action_id, "action_id")
        if not _ACTION_ID.fullmatch(action_id):
            raise ValueError("action_id contains unsupported characters")
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "kind", SetupActionKind(self.kind))
        object.__setattr__(
            self,
            "description",
            _nonblank(self.description, "description"),
        )
        object.__setattr__(self, "payload", _freeze_json(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "description": self.description,
            "payload": thaw_json(self.payload),
            "requires_privilege": self.requires_privilege,
            "requires_network": self.requires_network,
            "requires_confirmation": self.requires_confirmation,
            "reversible": self.reversible,
        }


    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SetupAction":
        return cls(
            action_id=str(value["action_id"]),
            kind=SetupActionKind(str(value["kind"])),
            description=str(value["description"]),
            payload=dict(value.get("payload") or {}),
            requires_privilege=bool(value.get("requires_privilege", False)),
            requires_network=bool(value.get("requires_network", False)),
            requires_confirmation=bool(
                value.get("requires_confirmation", False)
            ),
            reversible=bool(value.get("reversible", False)),
        )


@dataclass(frozen=True, slots=True)
class SetupPlan:
    provider_id: str
    deployment_path: DeploymentPath
    goal: SetupPlanGoal
    actions: tuple[SetupAction, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_id",
            _nonblank(self.provider_id, "provider_id"),
        )
        object.__setattr__(
            self,
            "deployment_path",
            DeploymentPath(self.deployment_path),
        )
        object.__setattr__(self, "goal", SetupPlanGoal(self.goal))
        actions = tuple(self.actions)
        if not actions:
            raise ValueError("SetupPlan must contain at least one action")
        if not all(isinstance(action, SetupAction) for action in actions):
            raise TypeError("actions must contain SetupAction values")
        ids = [action.action_id for action in actions]
        if len(ids) != len(set(ids)):
            raise ValueError("SetupPlan action IDs must be unique")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))
        object.__setattr__(
            self,
            "schema_version",
            _nonblank(self.schema_version, "schema_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "deployment_path": self.deployment_path.value,
            "goal": self.goal.value,
            "actions": [action.to_dict() for action in self.actions],
            "metadata": thaw_json(self.metadata),
        }

    def to_json(self, *, pretty: bool = False) -> str:
        if pretty:
            return json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SetupPlan":
        plan = cls(
            schema_version=str(value.get("schema_version", "v1")),
            provider_id=str(value["provider_id"]),
            deployment_path=DeploymentPath(str(value["deployment_path"])),
            goal=SetupPlanGoal(str(value["goal"])),
            actions=tuple(
                SetupAction.from_dict(action)
                for action in value.get("actions") or ()
            ),
            metadata=dict(value.get("metadata") or {}),
        )
        supplied_digest = value.get("digest")
        if supplied_digest is not None and str(supplied_digest) != plan.digest:
            raise ValueError("SetupPlan digest does not match content")
        return plan

    @classmethod
    def from_json(cls, payload: str) -> "SetupPlan":
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("SetupPlan JSON must contain an object")
        return cls.from_dict(value)

    @property
    def requires_privilege(self) -> bool:
        return any(action.requires_privilege for action in self.actions)

    @property
    def requires_network(self) -> bool:
        return any(action.requires_network for action in self.actions)

    @property
    def requires_confirmation(self) -> bool:
        return any(action.requires_confirmation for action in self.actions)


@dataclass(frozen=True, slots=True)
class ActionInspection:
    state: SetupActionState
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", SetupActionState(self.state))
        object.__setattr__(self, "detail", str(self.detail).strip())


@dataclass(frozen=True, slots=True)
class ActionReceipt:
    changed: bool
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    rollback_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", str(self.detail).strip())
        object.__setattr__(
            self,
            "evidence",
            _freeze_json(self.evidence),
        )
        object.__setattr__(
            self,
            "rollback_data",
            _freeze_json(self.rollback_data),
        )


@dataclass(frozen=True, slots=True)
class SetupActionPreview:
    action_id: str
    kind: SetupActionKind
    state: SetupActionState
    description: str
    requires_privilege: bool
    requires_network: bool
    requires_confirmation: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "state": self.state.value,
            "description": self.description,
            "requires_privilege": self.requires_privilege,
            "requires_network": self.requires_network,
            "requires_confirmation": self.requires_confirmation,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class SetupPreview:
    plan_digest: str
    actions: tuple[SetupActionPreview, ...]

    @property
    def blocked(self) -> bool:
        return any(
            action.state is SetupActionState.BLOCKED
            for action in self.actions
        )

    @property
    def changes_required(self) -> bool:
        return any(
            action.state is SetupActionState.NEEDS_APPLY
            for action in self.actions
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_digest": self.plan_digest,
            "blocked": self.blocked,
            "changes_required": self.changes_required,
            "actions": [action.to_dict() for action in self.actions],
        }


@dataclass(frozen=True, slots=True)
class SetupActionResult:
    action_id: str
    kind: SetupActionKind
    status: SetupActionResultStatus
    changed: bool
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", SetupActionKind(self.kind))
        object.__setattr__(
            self,
            "status",
            SetupActionResultStatus(self.status),
        )
        object.__setattr__(self, "detail", str(self.detail).strip())
        object.__setattr__(self, "evidence", _freeze_json(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "changed": self.changed,
            "detail": self.detail,
            "evidence": thaw_json(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class SetupApplyResult:
    plan_digest: str
    status: SetupApplyStatus
    actions: tuple[SetupActionResult, ...]
    rollback_actions: tuple[SetupActionResult, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SetupApplyStatus(self.status))
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(
            self,
            "rollback_actions",
            tuple(self.rollback_actions),
        )

    @property
    def succeeded(self) -> bool:
        return self.status is SetupApplyStatus.SUCCEEDED

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_digest": self.plan_digest,
            "status": self.status.value,
            "actions": [action.to_dict() for action in self.actions],
            "rollback_actions": [
                action.to_dict() for action in self.rollback_actions
            ],
        }


@dataclass(frozen=True, slots=True)
class SetupApproval:
    """Explicit authorization for applying one exact reviewed plan."""

    plan_digest: str
    allow_privileged: bool = False
    allow_network: bool = False
    allow_model_download: bool = False
    allow_model_convert: bool = False
    allow_confirmation_actions: bool = False

    def __post_init__(self) -> None:
        digest = _nonblank(self.plan_digest, "plan_digest").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("plan_digest must be a SHA-256 hex digest")
        object.__setattr__(self, "plan_digest", digest)
