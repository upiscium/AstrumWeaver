"""Isolated serving identity/profile values; not runtime admission or attestation.

This module deliberately has no Control, Worker, provider, transport or storage
imports. Callers must validate provenance and later recheck these bindings at
atomic claim/execution. Successful resolution grants no execution authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import re
from types import MappingProxyType
from uuid import UUID


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_QUANTITY = 2**63 - 1


class ServingContractError(ValueError):
    """Structured, value-redacted validation error for operator/client handling."""

    def __init__(self, code: str, field_name: str) -> None:
        self.code = code
        self.field_name = field_name
        super().__init__(f"{code}: {field_name}")


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ServingContractError("invalid-identifier", name)


def _digest(value: object, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ServingContractError("invalid-digest", name)


def _version(value: object, expected: str) -> None:
    if type(value) is not str or value != expected:
        raise ServingContractError("unsupported-schema", "schema_version")


def _features(values: object, name: str) -> frozenset[str]:
    if not isinstance(values, (tuple, list, set, frozenset)):
        raise ServingContractError("invalid-features", name)
    result: set[str] = set()
    for value in values:
        _identifier(value, name)
        if value in result:
            raise ServingContractError("duplicate-feature", name)
        result.add(value)
    return frozenset(result)


def _quantities(values: object, name: str, *, minimum: int = 1) -> Mapping[str, int]:
    if not isinstance(values, Mapping):
        raise ServingContractError("invalid-quantities", name)
    result: dict[str, int] = {}
    for key, value in values.items():
        _identifier(key, name)
        if type(value) is not int or not minimum <= value <= _MAX_QUANTITY:
            raise ServingContractError("invalid-quantity", name)
        result[key] = value
    return MappingProxyType(result)


def _revision(payload: dict[str, object]) -> str:
    # Each caller includes its distinct schema_version as a domain separator.
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class DeploymentIdentity:
    """Content identity companion to RuntimeDeploymentSpec, not another owner.

    Digests are caller-supplied provenance references, not verified artifacts.
    execution_config_sha256 must exclude secrets and deployment-local paths.
    """

    provider_id: str
    runtime_artifact_sha256: str
    adapter_artifact_sha256: str
    model_artifact_sha256: str
    execution_config_sha256: str
    quantization: str
    tokenizer_artifact_sha256: str | None = None
    template_artifact_sha256: str | None = None
    schema_version: str = "deployment-identity-v1"

    def __post_init__(self) -> None:
        _version(self.schema_version, "deployment-identity-v1")
        _identifier(self.provider_id, "provider_id")
        _identifier(self.quantization, "quantization")
        for name in (
            "runtime_artifact_sha256", "adapter_artifact_sha256",
            "model_artifact_sha256", "execution_config_sha256",
        ):
            _digest(getattr(self, name), name)
        for name in ("tokenizer_artifact_sha256", "template_artifact_sha256"):
            value = getattr(self, name)
            if value is not None:
                _digest(value, name)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "runtime_artifact_sha256": self.runtime_artifact_sha256,
            "adapter_artifact_sha256": self.adapter_artifact_sha256,
            "model_artifact_sha256": self.model_artifact_sha256,
            "execution_config_sha256": self.execution_config_sha256,
            "quantization": self.quantization,
            "tokenizer_artifact_sha256": self.tokenizer_artifact_sha256,
            "template_artifact_sha256": self.template_artifact_sha256,
        }

    @property
    def revision(self) -> str:
        return _revision(self.to_dict())


@dataclass(frozen=True, slots=True)
class RuntimeInstance:
    """Per-start identity supplied by the lifecycle owner; no readiness claim."""

    deployment_revision: str
    epoch: str
    schema_version: str = "runtime-instance-v1"

    def __post_init__(self) -> None:
        _version(self.schema_version, "runtime-instance-v1")
        _digest(self.deployment_revision, "deployment_revision")
        if type(self.epoch) is not str:
            raise ServingContractError("invalid-epoch", "epoch")
        try:
            parsed = UUID(self.epoch)
        except ValueError:
            raise ServingContractError("invalid-epoch", "epoch") from None
        if str(parsed) != self.epoch or parsed.int == 0:
            raise ServingContractError("invalid-epoch", "epoch")


@dataclass(frozen=True, slots=True)
class ServingContract:
    """One operation's declared evidence/features and generic upper bounds.

    Operation adapters, not this module, validate semantic consistency and
    whether validation_evidence_sha256 actually establishes these claims.
    """

    deployment_revision: str
    capability: str
    operation_schema: str
    validation_evidence_sha256: str
    features: frozenset[str] = field(default_factory=frozenset)
    limits: Mapping[str, int] = field(default_factory=dict)
    semantic_revision: str | None = None
    schema_version: str = "serving-contract-v1"

    def __post_init__(self) -> None:
        _version(self.schema_version, "serving-contract-v1")
        _digest(self.deployment_revision, "deployment_revision")
        _digest(self.validation_evidence_sha256, "validation_evidence_sha256")
        _identifier(self.capability, "capability")
        _identifier(self.operation_schema, "operation_schema")
        if self.semantic_revision is not None:
            _digest(self.semantic_revision, "semantic_revision")
        object.__setattr__(self, "features", _features(self.features, "features"))
        object.__setattr__(self, "limits", _quantities(self.limits, "limits"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "deployment_revision": self.deployment_revision,
            "capability": self.capability,
            "operation_schema": self.operation_schema,
            "validation_evidence_sha256": self.validation_evidence_sha256,
            "features": sorted(self.features),
            "limits": dict(self.limits),
            "semantic_revision": self.semantic_revision,
        }

    @property
    def revision(self) -> str:
        return _revision(self.to_dict())


@dataclass(frozen=True, slots=True)
class LogicalServingProfile:
    """Exact operator-selected binding; distinct from host-sizing profiles."""

    profile_id: str
    deployment_revision: str
    serving_contract_revision: str
    capability: str
    operation_schema: str
    required_features: frozenset[str] = field(default_factory=frozenset)
    limits: Mapping[str, int] = field(default_factory=dict)
    schema_version: str = "serving-profile-v1"

    def __post_init__(self) -> None:
        _version(self.schema_version, "serving-profile-v1")
        for name in ("profile_id", "capability", "operation_schema"):
            _identifier(getattr(self, name), name)
        _digest(self.deployment_revision, "deployment_revision")
        _digest(self.serving_contract_revision, "serving_contract_revision")
        object.__setattr__(
            self, "required_features", _features(self.required_features, "required_features"),
        )
        object.__setattr__(self, "limits", _quantities(self.limits, "limits"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "deployment_revision": self.deployment_revision,
            "serving_contract_revision": self.serving_contract_revision,
            "capability": self.capability,
            "operation_schema": self.operation_schema,
            "required_features": sorted(self.required_features),
            "limits": dict(self.limits),
        }

    @property
    def revision(self) -> str:
        return _revision(self.to_dict())


@dataclass(frozen=True, slots=True)
class ResolvedServingProfile:
    """Detached immutable configuration snapshot, never a reservation/lease."""

    profile_revision: str
    deployment: DeploymentIdentity
    contract: ServingContract
    profile: LogicalServingProfile
    effective_limits: Mapping[str, int]

    def __post_init__(self) -> None:
        # Validate also on direct construction: frozen does not imply trusted.
        if not isinstance(self.deployment, DeploymentIdentity):
            raise ServingContractError("invalid-type", "deployment")
        if not isinstance(self.contract, ServingContract):
            raise ServingContractError("invalid-type", "contract")
        if not isinstance(self.profile, LogicalServingProfile):
            raise ServingContractError("invalid-type", "profile")
        if self.profile_revision != self.profile.revision:
            raise ServingContractError("profile-revision-mismatch", "profile_revision")
        expected = _validate_binding(self.profile, self.contract, self.deployment)
        actual = _quantities(self.effective_limits, "effective_limits")
        if dict(actual) != expected:
            raise ServingContractError("effective-limits-mismatch", "effective_limits")
        object.__setattr__(self, "effective_limits", actual)


def _validate_binding(
    profile: LogicalServingProfile,
    contract: ServingContract,
    deployment: DeploymentIdentity,
) -> dict[str, int]:
    if (
        profile.deployment_revision != deployment.revision
        or contract.deployment_revision != deployment.revision
    ):
        raise ServingContractError("deployment-revision-mismatch", "deployment_revision")
    if profile.serving_contract_revision != contract.revision:
        raise ServingContractError("serving-contract-mismatch", "serving_contract_revision")
    if profile.capability != contract.capability:
        raise ServingContractError("capability-mismatch", "capability")
    if profile.operation_schema != contract.operation_schema:
        raise ServingContractError("operation-schema-mismatch", "operation_schema")
    if not profile.required_features <= contract.features:
        raise ServingContractError("unsupported-feature", "required_features")
    effective = dict(contract.limits)
    for key, maximum in profile.limits.items():
        if key not in contract.limits:
            raise ServingContractError("unknown-limit", "limits")
        if maximum > contract.limits[key]:
            raise ServingContractError("profile-limit-exceeded", "limits")
        effective[key] = maximum
    return effective


def resolve_profile(
    profile: LogicalServingProfile,
    contract: ServingContract,
    deployment: DeploymentIdentity,
    *,
    required_features: frozenset[str] = frozenset(),
    requested_usage: Mapping[str, int] | None = None,
    explicit_provider_id: str | None = None,
    explicit_model_sha256: str | None = None,
) -> ResolvedServingProfile:
    """Resolve exactly one supplied profile; never search/substitute a model.

    Usage is already measured by the operation adapter (e.g. template-inclusive
    tokens), not untrusted client assertions. Missing usage does not certify any
    request size. Readiness, per-attempt epochs and atomic admission are deferred.
    """
    for value, expected, name in (
        (profile, LogicalServingProfile, "profile"),
        (contract, ServingContract, "contract"),
        (deployment, DeploymentIdentity, "deployment"),
    ):
        if not isinstance(value, expected):
            raise ServingContractError("invalid-type", name)
    effective = _validate_binding(profile, contract, deployment)
    features = _features(required_features, "required_features")
    if not features <= contract.features:
        raise ServingContractError("unsupported-feature", "required_features")
    usage = _quantities(
        {} if requested_usage is None else requested_usage, "requested_usage", minimum=0,
    )
    for key, amount in usage.items():
        if key not in effective:
            raise ServingContractError("unknown-limit", "requested_usage")
        if amount > effective[key]:
            raise ServingContractError("request-limit-exceeded", "requested_usage")
    if explicit_provider_id is not None:
        _identifier(explicit_provider_id, "explicit_provider_id")
        if explicit_provider_id != deployment.provider_id:
            raise ServingContractError("provider-selection-mismatch", "explicit_provider_id")
    if explicit_model_sha256 is not None:
        _digest(explicit_model_sha256, "explicit_model_sha256")
        if explicit_model_sha256 != deployment.model_artifact_sha256:
            raise ServingContractError("model-selection-mismatch", "explicit_model_sha256")
    return ResolvedServingProfile(
        profile.revision, deployment, contract, profile, effective,
    )
