"""Validated native System-One decision gateway contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..serving import (
    DeploymentIdentity,
    LogicalServingProfile,
    ResolvedServingProfile,
    ServingContract,
    ServingJobBinding,
    resolve_profile,
)


DECISION_CATALOG_SCHEMA = "decision-gateway-catalog-v1"
DECISION_JOB_SCHEMA = "decision-job-v1"
DECISION_OPERATION_SCHEMA = "astrumweaver-decisions-v1"
DECISION_SEMANTICS_SCHEMA = "decision-semantics-v1"
LLAMA_CPP_SYSTEM_ONE_ADAPTER = "llama-cpp-system-one-v1"
DECISION_SCORE_KIND = "choice_set_probability"
LLAMA_CPP_SCORE_SEMANTICS = "llama-cpp-systemone-temperature-softmax-v1"

_REQUIRED_LIMITS = frozenset(
    {
        "state_bytes",
        "question_bytes",
        "choice_count",
        "choice_id_bytes",
        "choice_label_bytes",
        "state_tokens",
        "question_tokens",
        "choice_label_tokens",
        "aggregate_tokens",
        "request_bytes",
    }
)
_ALLOWED_REQUEST_FIELDS = frozenset(
    {"profile", "profile_revision", "state", "question", "choices"}
)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class DecisionGatewayError(ValueError):
    """Public-safe decision validation/normalization failure."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(message)


def _nonblank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DecisionGatewayError(
            "invalid_request",
            f"{name} must be a non-blank string",
        )
    return value.strip()


def _identifier(value: object, name: str) -> str:
    result = _nonblank(value, name)
    if _ID.fullmatch(result) is None:
        raise DecisionGatewayError(
            "invalid_profile",
            f"{name} must be a stable identifier",
        )
    return result


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise DecisionGatewayError(
            "invalid_profile",
            f"{name} must be a sha256 digest",
        )
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DecisionGatewayError(
            "invalid_profile",
            f"{name} must be a positive integer",
        )
    return value


def _revision(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class DecisionSemanticsIdentity:
    deployment_revision: str
    adapter_id: str
    score_kind: str
    provider_score_semantics: str
    calibration_status: str
    calibration_reference_sha256: str | None
    abstain_below: float
    mode: str = "shadow"
    schema_version: str = DECISION_SEMANTICS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DECISION_SEMANTICS_SCHEMA:
            raise DecisionGatewayError(
                "invalid_profile",
                "unsupported decision-semantics schema",
            )
        _digest(self.deployment_revision, "deployment_revision")
        _identifier(self.adapter_id, "adapter_id")
        _identifier(self.score_kind, "score_kind")
        _identifier(self.provider_score_semantics, "provider_score_semantics")
        if self.score_kind != DECISION_SCORE_KIND:
            raise DecisionGatewayError(
                "invalid_profile",
                "unsupported decision score kind",
            )
        if self.provider_score_semantics != LLAMA_CPP_SCORE_SEMANTICS:
            raise DecisionGatewayError(
                "invalid_profile",
                "unsupported provider score semantics",
            )
        if self.calibration_status not in {"uncalibrated", "calibrated"}:
            raise DecisionGatewayError(
                "invalid_profile",
                "calibration_status must be uncalibrated or calibrated",
            )
        if self.calibration_status == "calibrated":
            if self.calibration_reference_sha256 is None:
                raise DecisionGatewayError(
                    "invalid_profile",
                    "calibrated decisions require a calibration reference",
                )
            _digest(
                self.calibration_reference_sha256,
                "calibration_reference_sha256",
            )
        elif self.calibration_reference_sha256 is not None:
            raise DecisionGatewayError(
                "invalid_profile",
                "uncalibrated decisions must not claim calibration evidence",
            )
        if (
            isinstance(self.abstain_below, bool)
            or not isinstance(self.abstain_below, (int, float))
            or not math.isfinite(float(self.abstain_below))
            or not 0.0 <= float(self.abstain_below) <= 1.0
        ):
            raise DecisionGatewayError(
                "invalid_profile",
                "abstain_below must be finite and between zero and one",
            )
        if self.mode != "shadow":
            raise DecisionGatewayError(
                "invalid_profile",
                "first decision adapter is shadow-mode only",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "deployment_revision": self.deployment_revision,
            "adapter_id": self.adapter_id,
            "score_kind": self.score_kind,
            "provider_score_semantics": self.provider_score_semantics,
            "calibration_status": self.calibration_status,
            "calibration_reference_sha256": self.calibration_reference_sha256,
            "abstain_below": float(self.abstain_below),
            "mode": self.mode,
        }

    @property
    def decision_semantics_id(self) -> str:
        return _revision(self.to_dict())


@dataclass(frozen=True, slots=True)
class DecisionGatewayProfile:
    resolved: ResolvedServingProfile
    semantics: DecisionSemanticsIdentity
    adapter_id: str
    request_timeout_seconds: float
    max_attempts: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.resolved, ResolvedServingProfile):
            raise TypeError("resolved must be ResolvedServingProfile")
        if not isinstance(self.semantics, DecisionSemanticsIdentity):
            raise TypeError("semantics must be DecisionSemanticsIdentity")
        adapter_id = _identifier(self.adapter_id, "adapter_id")
        if adapter_id != LLAMA_CPP_SYSTEM_ONE_ADAPTER:
            raise DecisionGatewayError(
                "unsupported_adapter",
                "decision gateway supports only the reviewed llama.cpp adapter",
            )
        if self.resolved.profile.capability != "decision.system_one":
            raise DecisionGatewayError(
                "invalid_profile",
                "decision profile must bind decision.system_one",
            )
        if self.resolved.profile.operation_schema != DECISION_OPERATION_SCHEMA:
            raise DecisionGatewayError(
                "invalid_profile",
                "decision profile uses an unsupported operation schema",
            )
        if self.resolved.deployment.provider_id not in {"llama-cpp", "llama-cpp-dual"}:
            raise DecisionGatewayError(
                "invalid_profile",
                "decision profile must bind the reviewed llama.cpp provider",
            )
        if self.semantics.deployment_revision != self.resolved.deployment.revision:
            raise DecisionGatewayError(
                "invalid_profile",
                "decision semantics do not match deployment identity",
            )
        if self.semantics.adapter_id != adapter_id:
            raise DecisionGatewayError(
                "invalid_profile",
                "decision semantics do not match adapter identity",
            )
        if (
            self.resolved.contract.semantic_revision
            != self.semantics.decision_semantics_id
        ):
            raise DecisionGatewayError(
                "invalid_profile",
                "serving contract does not bind decision semantics",
            )
        required_features = {
            "native-decision-head",
            "choice-probabilities",
            "multi-token-choice-labels",
            "provider-temperature-softmax",
        }
        if not required_features <= self.resolved.contract.features:
            raise DecisionGatewayError(
                "invalid_profile",
                "serving contract does not prove decision runtime semantics",
            )
        limits = dict(self.resolved.effective_limits)
        missing = _REQUIRED_LIMITS - limits.keys()
        if missing:
            raise DecisionGatewayError(
                "invalid_profile",
                "decision profile is missing required bounded limits",
            )
        for name in _REQUIRED_LIMITS:
            _positive_int(limits[name], name)
        if limits["choice_count"] < 2:
            raise DecisionGatewayError(
                "invalid_profile",
                "decision choice-count limit must allow at least two choices",
            )
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, (int, float))
            or not math.isfinite(float(self.request_timeout_seconds))
            or self.request_timeout_seconds <= 0
        ):
            raise DecisionGatewayError(
                "invalid_profile",
                "request_timeout_seconds must be finite and positive",
            )
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise DecisionGatewayError(
                "invalid_profile",
                "max_attempts must be a positive integer",
            )
        object.__setattr__(self, "adapter_id", adapter_id)

    @property
    def profile_id(self) -> str:
        return self.resolved.profile.profile_id

    @property
    def binding(self) -> ServingJobBinding:
        return ServingJobBinding.from_resolved(self.resolved)

    @property
    def decision_semantics_id(self) -> str:
        return self.semantics.decision_semantics_id

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DecisionGatewayProfile":
        data = dict(value)
        try:
            deployment = DeploymentIdentity(**dict(data["deployment"]))
            contract = ServingContract(**dict(data["contract"]))
            profile = LogicalServingProfile(**dict(data["profile"]))
            resolved = resolve_profile(profile, contract, deployment)
            semantics = DecisionSemanticsIdentity(**dict(data["semantics"]))
            return cls(
                resolved=resolved,
                semantics=semantics,
                adapter_id=data["adapter_id"],
                request_timeout_seconds=data.get("request_timeout_seconds", 30.0),
                max_attempts=data.get("max_attempts", 2),
            )
        except DecisionGatewayError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise DecisionGatewayError(
                "invalid_profile",
                "configured decision profile is invalid",
            ) from exc


class DecisionProfileCatalog:
    def __init__(self, profiles: tuple[DecisionGatewayProfile, ...]) -> None:
        values = tuple(profiles)
        if not values:
            raise DecisionGatewayError(
                "invalid_catalog",
                "decision catalog must contain at least one profile",
            )
        table: dict[str, DecisionGatewayProfile] = {}
        for profile in values:
            if not isinstance(profile, DecisionGatewayProfile):
                raise TypeError(
                    "profiles must contain DecisionGatewayProfile values"
                )
            if profile.profile_id in table:
                raise DecisionGatewayError(
                    "invalid_catalog",
                    "decision profile IDs must be unique",
                )
            table[profile.profile_id] = profile
        self._profiles = values
        self._table = MappingProxyType(table)

    @property
    def profiles(self) -> tuple[DecisionGatewayProfile, ...]:
        return self._profiles

    def get(self, profile_id: str) -> DecisionGatewayProfile:
        try:
            return self._table[profile_id]
        except KeyError as exc:
            raise DecisionGatewayError(
                "profile_not_found",
                "requested decision profile is not configured",
                status_code=404,
            ) from exc

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DecisionProfileCatalog":
        data = dict(value)
        if data.get("schema_version") != DECISION_CATALOG_SCHEMA:
            raise DecisionGatewayError(
                "invalid_catalog",
                "unsupported decision catalog schema",
            )
        raw = data.get("profiles")
        if not isinstance(raw, list) or not all(
            isinstance(item, Mapping) for item in raw
        ):
            raise DecisionGatewayError(
                "invalid_catalog",
                "decision catalog profiles must be an array of objects",
            )
        return cls(tuple(DecisionGatewayProfile.from_dict(item) for item in raw))


@dataclass(frozen=True, slots=True)
class CompiledDecisionRequest:
    profile: DecisionGatewayProfile
    choices: tuple[tuple[str, str], ...]
    provider_choice_keys: tuple[str, ...]
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.profile, DecisionGatewayProfile):
            raise TypeError("profile must be DecisionGatewayProfile")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


def compile_decision_request(
    value: Mapping[str, object],
    profile: DecisionGatewayProfile,
    *,
    request_size_bytes: int,
) -> CompiledDecisionRequest:
    data = dict(value)
    extra = set(data) - _ALLOWED_REQUEST_FIELDS
    if extra:
        raise DecisionGatewayError(
            "unsupported_field",
            "decision request uses unsupported fields",
        )
    if data.get("profile") != profile.profile_id:
        raise DecisionGatewayError(
            "profile_not_found",
            "request profile is not configured",
            status_code=404,
        )
    profile_revision = data.get("profile_revision")
    if (
        not isinstance(profile_revision, str)
        or _DIGEST.fullmatch(profile_revision) is None
    ):
        raise DecisionGatewayError(
            "invalid_request",
            "profile_revision must be a sha256 digest",
        )
    if profile_revision != profile.resolved.profile_revision:
        raise DecisionGatewayError(
            "profile_revision_mismatch",
            "requested decision profile revision is stale or incompatible",
            status_code=409,
        )

    state = _nonblank(data.get("state"), "state")
    question = _nonblank(data.get("question"), "question")
    raw_choices = data.get("choices")
    if not isinstance(raw_choices, list) or len(raw_choices) < 2:
        raise DecisionGatewayError(
            "invalid_request",
            "choices must contain at least two ordered choices",
        )

    limits = dict(profile.resolved.effective_limits)
    if request_size_bytes > limits["request_bytes"]:
        raise DecisionGatewayError(
            "request_too_large",
            "decision request exceeds the profile byte limit",
            status_code=413,
        )
    if len(raw_choices) > limits["choice_count"]:
        raise DecisionGatewayError(
            "too_many_choices",
            "decision request exceeds the profile choice-count limit",
        )
    if len(state.encode("utf-8")) > limits["state_bytes"]:
        raise DecisionGatewayError(
            "state_too_large",
            "decision state exceeds the profile byte limit",
        )
    if len(question.encode("utf-8")) > limits["question_bytes"]:
        raise DecisionGatewayError(
            "question_too_large",
            "decision question exceeds the profile byte limit",
        )

    choices: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw_choices:
        if not isinstance(item, Mapping) or set(item) != {"id", "label"}:
            raise DecisionGatewayError(
                "invalid_request",
                "each decision choice must contain exactly id and label",
            )
        choice_id = _identifier(item.get("id"), "choice.id")
        label = _nonblank(item.get("label"), "choice.label")
        if choice_id in seen:
            raise DecisionGatewayError(
                "invalid_request",
                "decision choice IDs must be unique",
            )
        seen.add(choice_id)
        if len(choice_id.encode("utf-8")) > limits["choice_id_bytes"]:
            raise DecisionGatewayError(
                "choice_too_large",
                "decision choice ID exceeds the profile byte limit",
            )
        if len(label.encode("utf-8")) > limits["choice_label_bytes"]:
            raise DecisionGatewayError(
                "choice_too_large",
                "decision choice label exceeds the profile byte limit",
            )
        choices.append((choice_id, label))

    provider_keys = tuple(f"{index:04d}" for index in range(len(choices)))
    criteria = {
        key: label
        for key, (_, label) in zip(provider_keys, choices, strict=True)
    }
    payload = {
        "schema_version": DECISION_JOB_SCHEMA,
        "adapter_id": profile.adapter_id,
        "decision_semantics_id": profile.decision_semantics_id,
        "choice_ids": [choice_id for choice_id, _ in choices],
        "provider_choice_keys": list(provider_keys),
        "request": {
            "state": state,
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": question,
                    "criteria": criteria,
                }
            },
        },
        "limits": {
            name: limits[name]
            for name in (
                "state_bytes",
                "question_bytes",
                "choice_count",
                "choice_id_bytes",
                "choice_label_bytes",
                "state_tokens",
                "question_tokens",
                "choice_label_tokens",
                "aggregate_tokens",
            )
        },
    }
    return CompiledDecisionRequest(
        profile=profile,
        choices=tuple(choices),
        provider_choice_keys=provider_keys,
        payload=payload,
    )


def validate_decision_provider_response(
    value: object,
    *,
    provider_choice_keys: tuple[str, ...],
) -> tuple[list[float], dict[str, int]]:
    if not isinstance(value, Mapping):
        raise DecisionGatewayError(
            "invalid_provider_response",
            "decision backend returned a non-object response",
            status_code=502,
        )
    answers = value.get("answers")
    if not isinstance(answers, Mapping) or set(answers) != {"decision"}:
        raise DecisionGatewayError(
            "invalid_provider_response",
            "decision backend returned an invalid answer set",
            status_code=502,
        )
    answer = answers.get("decision")
    if not isinstance(answer, Mapping) or answer.get("type") != "choice":
        raise DecisionGatewayError(
            "invalid_provider_response",
            "decision backend returned an invalid choice answer",
            status_code=502,
        )
    probabilities = answer.get("probabilities")
    if (
        not isinstance(probabilities, Mapping)
        or set(probabilities) != set(provider_choice_keys)
    ):
        raise DecisionGatewayError(
            "invalid_provider_response",
            "decision backend returned an incompatible probability set",
            status_code=502,
        )
    scores: list[float] = []
    for key in provider_choice_keys:
        raw = probabilities.get(key)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or not 0.0 <= float(raw) <= 1.0
        ):
            raise DecisionGatewayError(
                "invalid_provider_response",
                "decision backend returned a non-finite probability",
                status_code=502,
            )
        scores.append(float(raw))
    if abs(sum(scores) - 1.0) > 1e-4:
        raise DecisionGatewayError(
            "invalid_provider_response",
            "decision backend probabilities are not normalized",
            status_code=502,
        )
    provider_choice = answer.get("choice")
    if provider_choice not in provider_choice_keys:
        raise DecisionGatewayError(
            "invalid_provider_response",
            "decision backend selected an unknown choice",
            status_code=502,
        )
    best = max(scores)
    selected_index = provider_choice_keys.index(str(provider_choice))
    if scores[selected_index] < best - 1e-8:
        raise DecisionGatewayError(
            "invalid_provider_response",
            "decision backend selected a non-maximal choice",
            status_code=502,
        )
    confidence = answer.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise DecisionGatewayError(
            "invalid_provider_response",
            "decision backend returned invalid provider confidence",
            status_code=502,
        )
    usage_raw = value.get("usage")
    if not isinstance(usage_raw, Mapping):
        raise DecisionGatewayError(
            "invalid_provider_response",
            "decision backend omitted usage",
            status_code=502,
        )
    usage: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens"):
        raw = usage_raw.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise DecisionGatewayError(
                "invalid_provider_response",
                "decision backend returned invalid usage",
                status_code=502,
            )
        usage[key] = raw
    if usage["output_tokens"] != 0:
        raise DecisionGatewayError(
            "invalid_provider_response",
            "native decision backend unexpectedly generated output tokens",
            status_code=502,
        )
    return scores, usage


def normalize_decision_response(
    value: object,
    *,
    compiled: CompiledDecisionRequest,
) -> dict[str, object]:
    scores, usage = validate_decision_provider_response(
        value,
        provider_choice_keys=compiled.provider_choice_keys,
    )
    best_index = max(range(len(scores)), key=scores.__getitem__)
    best_score = scores[best_index]
    abstained = best_score < float(compiled.profile.semantics.abstain_below)
    choice_id = None if abstained else compiled.choices[best_index][0]
    semantics = compiled.profile.semantics
    return {
        "object": "astrumweaver.decision",
        "profile": compiled.profile.profile_id,
        "choice_id": choice_id,
        "abstained": abstained,
        "scores": [
            {"id": choice_id_value, "score": score}
            for (choice_id_value, _), score in zip(
                compiled.choices,
                scores,
                strict=True,
            )
        ],
        "score_kind": semantics.score_kind,
        "calibration": {
            "status": semantics.calibration_status,
            "reference_sha256": semantics.calibration_reference_sha256,
        },
        "usage": usage,
        "x_astrumweaver": {
            "decision_semantics_id": semantics.decision_semantics_id,
            "profile_revision": compiled.profile.resolved.profile_revision,
            "deployment_revision": compiled.profile.resolved.deployment.revision,
            "serving_contract_revision": (
                compiled.profile.resolved.contract.revision
            ),
            "provider_score_semantics": semantics.provider_score_semantics,
            "abstain_below": float(semantics.abstain_below),
            "mode": semantics.mode,
            "authority": "recommendation-only",
        },
    }


__all__ = [
    "DECISION_CATALOG_SCHEMA",
    "DECISION_JOB_SCHEMA",
    "DECISION_OPERATION_SCHEMA",
    "DECISION_SCORE_KIND",
    "DECISION_SEMANTICS_SCHEMA",
    "LLAMA_CPP_SCORE_SEMANTICS",
    "LLAMA_CPP_SYSTEM_ONE_ADAPTER",
    "CompiledDecisionRequest",
    "DecisionGatewayError",
    "DecisionGatewayProfile",
    "DecisionProfileCatalog",
    "DecisionSemanticsIdentity",
    "compile_decision_request",
    "normalize_decision_response",
    "validate_decision_provider_response",
]