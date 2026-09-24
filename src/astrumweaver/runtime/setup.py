"""Deterministic runtime setup planning and apply engine."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from .contracts import ModelPreparationPolicy, RuntimeSetupIntent


_PROVIDER_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_ACTION_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _nonblank(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_compatible(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, StrEnum):
        return value.value
    raise TypeError(
        f"setup plan values must be JSON-compatible, got {type(value).__name__}"
    )


def _immutable_json_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    normalized = _json_compatible(dict(value or {}))
    assert isinstance(normalized, dict)
    return MappingProxyType(normalized)


class SetupActionKind(StrEnum):
    STOP_RUNTIME = "stop_runtime"
    INSTALL_PACKAGE = "install_package"
    ENSURE_DIRECTORY = "ensure_directory"
    WRITE_CONFIGURATION = "write_configuration"
    PREPARE_MODEL = "prepare_model"
    PREFLIGHT_RUNTIME = "preflight_runtime"
    START_RUNTIME = "start_runtime"


class SetupExecutionMode(StrEnum):
    DRY_RUN = "dry_run"
    APPLY = "apply"


class SetupActionStatus(StrEnum):
    PLANNED = "planned"
    SKIPPED = "skipped"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ROLLBACK_FAILED = "rollback_failed"


@dataclass(frozen=True, slots=True)
class SecretReference:
    """A secret name/reference, never the secret value itself."""

    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonblank(self.name, "name"))

    def to_dict(self) -> dict[str, str]:
        return {"$secret_ref": self.name}


@dataclass(frozen=True, slots=True)
class SetupLayout:
    config_root: str = "/etc/astrumweaver/runtime"
    state_root: str = "/var/lib/astrumweaver/runtime"
    cache_root: str = "/var/cache/astrumweaver/runtime"

    def __post_init__(self) -> None:
        for name in ("config_root", "state_root", "cache_root"):
            value = _nonblank(getattr(self, name), name)
            if not value.startswith("/"):
                raise ValueError(f"{name} must be an absolute path")
            object.__setattr__(self, name, value.rstrip("/") or "/")


@dataclass(frozen=True, slots=True)
class SetupAction:
    action_id: str
    kind: SetupActionKind
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    requires_privilege: bool = False
    requires_model_transfer: bool = False
    rollbackable: bool = False

    def __post_init__(self) -> None:
        action_id = _nonblank(self.action_id, "action_id")
        if not _ACTION_ID.fullmatch(action_id):
            raise ValueError("action_id must use lowercase slug syntax")
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "kind", SetupActionKind(self.kind))
        object.__setattr__(
            self,
            "description",
            _nonblank(self.description, "description"),
        )
        object.__setattr__(
            self,
            "parameters",
            _immutable_json_mapping(self.parameters),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "description": self.description,
            "parameters": _json_compatible(self.parameters),
            "requires_privilege": self.requires_privilege,
            "requires_model_transfer": self.requires_model_transfer,
            "rollbackable": self.rollbackable,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SetupAction":
        return cls(
            action_id=str(value["action_id"]),
            kind=SetupActionKind(str(value["kind"])),
            description=str(value["description"]),
            parameters=dict(value.get("parameters") or {}),
            requires_privilege=bool(value.get("requires_privilege", False)),
            requires_model_transfer=bool(
                value.get("requires_model_transfer", False)
            ),
            rollbackable=bool(value.get("rollbackable", False)),
        )


@dataclass(frozen=True, slots=True)
class SetupPlan:
    provider_id: str
    actions: tuple[SetupAction, ...]
    schema_version: str = "setup-plan-v1"

    def __post_init__(self) -> None:
        provider_id = _nonblank(self.provider_id, "provider_id")
        if not _PROVIDER_ID.fullmatch(provider_id):
            raise ValueError("provider_id must use lowercase slug syntax")
        object.__setattr__(self, "provider_id", provider_id)

        actions = tuple(self.actions)
        if not actions:
            raise ValueError("setup plan must contain at least one action")
        if not all(isinstance(action, SetupAction) for action in actions):
            raise TypeError("actions must contain SetupAction values")
        ids = [action.action_id for action in actions]
        if len(ids) != len(set(ids)):
            raise ValueError("setup action IDs must be unique")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(
            self,
            "schema_version",
            _nonblank(self.schema_version, "schema_version"),
        )

    @property
    def requires_privilege(self) -> bool:
        return any(action.requires_privilege for action in self.actions)

    @property
    def requires_model_transfer(self) -> bool:
        return any(action.requires_model_transfer for action in self.actions)

    @property
    def digest(self) -> str:
        payload = self._payload()
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "actions": [action.to_dict() for action in self.actions],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["digest"] = self.digest
        payload["requires_privilege"] = self.requires_privilege
        payload["requires_model_transfer"] = self.requires_model_transfer
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SetupPlan":
        plan = cls(
            schema_version=str(value.get("schema_version", "setup-plan-v1")),
            provider_id=str(value["provider_id"]),
            actions=tuple(
                SetupAction.from_dict(action)
                for action in value.get("actions") or ()
            ),
        )
        supplied_digest = value.get("digest")
        if supplied_digest is not None and str(supplied_digest) != plan.digest:
            raise ValueError("setup plan digest does not match content")
        return plan

    def explain(self) -> str:
        lines = [
            f"provider: {self.provider_id}",
            f"digest: {self.digest}",
            f"privileged: {'yes' if self.requires_privilege else 'no'}",
            (
                "model-transfer: yes"
                if self.requires_model_transfer
                else "model-transfer: no"
            ),
            "actions:",
        ]
        for index, action in enumerate(self.actions, 1):
            flags: list[str] = []
            if action.requires_privilege:
                flags.append("privileged")
            if action.requires_model_transfer:
                flags.append("model-transfer")
            if action.rollbackable:
                flags.append("rollbackable")
            suffix = f" [{' '.join(flags)}]" if flags else ""
            lines.append(
                f"  {index:02d}. {action.action_id}: "
                f"{action.description}{suffix}"
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SetupApplyPolicy:
    allow_privileged: bool = False
    allow_model_transfer: bool = False


@dataclass(frozen=True, slots=True)
class SetupProbe:
    satisfied: bool
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence",
            _immutable_json_mapping(self.evidence),
        )


@dataclass(frozen=True, slots=True)
class SetupActionResult:
    action_id: str
    status: SetupActionStatus
    changed: bool = False
    evidence: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_id",
            _nonblank(self.action_id, "action_id"),
        )
        object.__setattr__(self, "status", SetupActionStatus(self.status))
        object.__setattr__(
            self,
            "evidence",
            _immutable_json_mapping(self.evidence),
        )
        if self.error_code is not None:
            object.__setattr__(
                self,
                "error_code",
                _nonblank(self.error_code, "error_code"),
            )
        if self.error_message is not None:
            object.__setattr__(
                self,
                "error_message",
                _nonblank(self.error_message, "error_message"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "status": self.status.value,
            "changed": self.changed,
            "evidence": _json_compatible(self.evidence),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


@dataclass(frozen=True, slots=True)
class SetupRunReport:
    plan_digest: str
    mode: SetupExecutionMode
    success: bool
    results: tuple[SetupActionResult, ...]
    rollback_results: tuple[SetupActionResult, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_digest",
            _nonblank(self.plan_digest, "plan_digest"),
        )
        object.__setattr__(self, "mode", SetupExecutionMode(self.mode))
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(
            self,
            "rollback_results",
            tuple(self.rollback_results),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_digest": self.plan_digest,
            "mode": self.mode.value,
            "success": self.success,
            "results": [result.to_dict() for result in self.results],
            "rollback_results": [
                result.to_dict() for result in self.rollback_results
            ],
        }


class SetupPolicyError(RuntimeError):
    pass


class SetupTargetError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = _nonblank(code, "code")
        self.public_message = _nonblank(message, "message")


@runtime_checkable
class SetupTarget(Protocol):
    """Deployment adapter used by the shared apply engine."""

    def probe(self, action: SetupAction) -> SetupProbe: ...

    def apply(self, action: SetupAction) -> Mapping[str, Any]: ...

    def rollback(
        self,
        action: SetupAction,
        evidence: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _provider_paths(
    provider_id: str,
    layout: SetupLayout,
) -> tuple[str, str, str]:
    if not _PROVIDER_ID.fullmatch(provider_id):
        raise ValueError("provider_id must use lowercase slug syntax")
    return (
        f"{layout.config_root}/{provider_id}.json",
        f"{layout.state_root}/{provider_id}",
        f"{layout.cache_root}/{provider_id}",
    )


def build_setup_plan(
    intent: RuntimeSetupIntent,
    *,
    layout: SetupLayout | None = None,
) -> SetupPlan:
    """Translate one provider setup intent into a deterministic shared plan."""

    if not isinstance(intent, RuntimeSetupIntent):
        raise TypeError("intent must be RuntimeSetupIntent")
    layout = layout or SetupLayout()
    provider_id = intent.provider_id
    config_path, state_path, cache_path = _provider_paths(
        provider_id,
        layout,
    )
    privileged = intent.requires_privilege
    actions: list[SetupAction] = []

    def add(
        suffix: str,
        kind: SetupActionKind,
        description: str,
        *,
        parameters: Mapping[str, Any] | None = None,
        requires_privilege: bool = False,
        requires_model_transfer: bool = False,
        rollbackable: bool = False,
    ) -> None:
        actions.append(
            SetupAction(
                action_id=f"{len(actions) + 1:03d}-{suffix}",
                kind=kind,
                description=description,
                parameters=parameters or {},
                requires_privilege=requires_privilege,
                requires_model_transfer=requires_model_transfer,
                rollbackable=rollbackable,
            )
        )

    add(
        "stop-runtime",
        SetupActionKind.STOP_RUNTIME,
        f"Stop {provider_id} runtime before reconfiguration",
        parameters={"provider_id": provider_id},
        requires_privilege=privileged,
        rollbackable=True,
    )

    for package_reference in sorted(intent.package_references):
        add(
            "install-package",
            SetupActionKind.INSTALL_PACKAGE,
            f"Ensure runtime package {package_reference}",
            parameters={
                "provider_id": provider_id,
                "package_reference": package_reference,
            },
            requires_privilege=privileged,
            rollbackable=False,
        )

    add(
        "state-directory",
        SetupActionKind.ENSURE_DIRECTORY,
        f"Ensure state directory for {provider_id}",
        parameters={
            "provider_id": provider_id,
            "path": state_path,
            "purpose": "state",
        },
        requires_privilege=privileged,
        rollbackable=True,
    )
    add(
        "cache-directory",
        SetupActionKind.ENSURE_DIRECTORY,
        f"Ensure cache directory for {provider_id}",
        parameters={
            "provider_id": provider_id,
            "path": cache_path,
            "purpose": "cache",
        },
        requires_privilege=privileged,
        rollbackable=True,
    )

    if intent.configuration:
        rendered = json.dumps(
            _json_compatible(intent.configuration),
            sort_keys=True,
            separators=(",", ":"),
        )
        add(
            "write-configuration",
            SetupActionKind.WRITE_CONFIGURATION,
            f"Render non-secret configuration for {provider_id}",
            parameters={
                "provider_id": provider_id,
                "path": config_path,
                "format": "json",
                "content": rendered,
            },
            requires_privilege=privileged,
            rollbackable=True,
        )

    if intent.model_ref is not None:
        transfer = intent.model_preparation in {
            ModelPreparationPolicy.DOWNLOAD,
            ModelPreparationPolicy.CONVERT,
        }
        add(
            "prepare-model",
            SetupActionKind.PREPARE_MODEL,
            f"Prepare model reference for {provider_id}",
            parameters={
                "provider_id": provider_id,
                "policy": intent.model_preparation.value,
                "model_ref": intent.model_ref,
                "cache_path": cache_path,
            },
            requires_privilege=False,
            requires_model_transfer=transfer,
            rollbackable=False,
        )
    elif intent.model_preparation is not ModelPreparationPolicy.REFERENCE_ONLY:
        raise ValueError(
            "download/convert model preparation requires model_ref"
        )

    add(
        "preflight",
        SetupActionKind.PREFLIGHT_RUNTIME,
        f"Run {provider_id} runtime preflight",
        parameters={
            "provider_id": provider_id,
            "config_path": config_path,
            "state_path": state_path,
            "cache_path": cache_path,
        },
        requires_privilege=False,
        rollbackable=False,
    )
    add(
        "start-runtime",
        SetupActionKind.START_RUNTIME,
        f"Start {provider_id} runtime",
        parameters={"provider_id": provider_id},
        requires_privilege=privileged,
        rollbackable=True,
    )

    return SetupPlan(provider_id=provider_id, actions=tuple(actions))


class SetupEngine:
    def _check_policy(
        self,
        plan: SetupPlan,
        policy: SetupApplyPolicy,
    ) -> None:
        if plan.requires_privilege and not policy.allow_privileged:
            raise SetupPolicyError(
                "setup plan contains privileged actions; explicit authorization is required"
            )
        if (
            plan.requires_model_transfer
            and not policy.allow_model_transfer
        ):
            raise SetupPolicyError(
                "setup plan would download/convert model data; explicit authorization is required"
            )

    def dry_run(self, plan: SetupPlan) -> SetupRunReport:
        results = tuple(
            SetupActionResult(
                action_id=action.action_id,
                status=SetupActionStatus.PLANNED,
                changed=False,
                evidence={
                    "kind": action.kind.value,
                    "requires_privilege": action.requires_privilege,
                    "requires_model_transfer": (
                        action.requires_model_transfer
                    ),
                    "rollbackable": action.rollbackable,
                },
            )
            for action in plan.actions
        )
        return SetupRunReport(
            plan_digest=plan.digest,
            mode=SetupExecutionMode.DRY_RUN,
            success=True,
            results=results,
        )

    def apply(
        self,
        plan: SetupPlan,
        *,
        target: SetupTarget,
        policy: SetupApplyPolicy,
    ) -> SetupRunReport:
        self._check_policy(plan, policy)

        results: list[SetupActionResult] = []
        changed: list[tuple[SetupAction, SetupActionResult]] = []

        for action in plan.actions:
            try:
                probe = target.probe(action)
                if probe.satisfied:
                    result = SetupActionResult(
                        action_id=action.action_id,
                        status=SetupActionStatus.SKIPPED,
                        changed=False,
                        evidence=probe.evidence,
                    )
                    results.append(result)
                    continue

                evidence = target.apply(action)
                result = SetupActionResult(
                    action_id=action.action_id,
                    status=SetupActionStatus.APPLIED,
                    changed=True,
                    evidence=evidence,
                )
                results.append(result)
                changed.append((action, result))
            except SetupTargetError as exc:
                results.append(
                    SetupActionResult(
                        action_id=action.action_id,
                        status=SetupActionStatus.FAILED,
                        changed=False,
                        error_code=exc.code,
                        error_message=exc.public_message,
                    )
                )
                return SetupRunReport(
                    plan_digest=plan.digest,
                    mode=SetupExecutionMode.APPLY,
                    success=False,
                    results=tuple(results),
                    rollback_results=self._rollback(
                        target,
                        changed,
                    ),
                )
            except Exception:
                results.append(
                    SetupActionResult(
                        action_id=action.action_id,
                        status=SetupActionStatus.FAILED,
                        changed=False,
                        error_code="unexpected-target-error",
                        error_message=(
                            "setup target failed unexpectedly; "
                            "inspect local protected logs"
                        ),
                    )
                )
                return SetupRunReport(
                    plan_digest=plan.digest,
                    mode=SetupExecutionMode.APPLY,
                    success=False,
                    results=tuple(results),
                    rollback_results=self._rollback(
                        target,
                        changed,
                    ),
                )

        return SetupRunReport(
            plan_digest=plan.digest,
            mode=SetupExecutionMode.APPLY,
            success=True,
            results=tuple(results),
        )

    def _rollback(
        self,
        target: SetupTarget,
        changed: list[tuple[SetupAction, SetupActionResult]],
    ) -> tuple[SetupActionResult, ...]:
        rollback_results: list[SetupActionResult] = []
        for action, result in reversed(changed):
            if not action.rollbackable:
                continue
            try:
                evidence = target.rollback(
                    action,
                    result.evidence,
                )
                rollback_results.append(
                    SetupActionResult(
                        action_id=action.action_id,
                        status=SetupActionStatus.ROLLED_BACK,
                        changed=True,
                        evidence=evidence,
                    )
                )
            except SetupTargetError as exc:
                rollback_results.append(
                    SetupActionResult(
                        action_id=action.action_id,
                        status=SetupActionStatus.ROLLBACK_FAILED,
                        changed=True,
                        error_code=exc.code,
                        error_message=exc.public_message,
                    )
                )
            except Exception:
                rollback_results.append(
                    SetupActionResult(
                        action_id=action.action_id,
                        status=SetupActionStatus.ROLLBACK_FAILED,
                        changed=True,
                        error_code="unexpected-rollback-error",
                        error_message=(
                            "rollback failed unexpectedly; "
                            "inspect local protected logs"
                        ),
                    )
                )
        return tuple(rollback_results)
