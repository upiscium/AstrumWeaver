"""Validated embedding gateway contracts for immutable vector spaces."""

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


EMBEDDING_CATALOG_SCHEMA = "embedding-gateway-catalog-v1"
EMBEDDING_JOB_SCHEMA = "embedding-job-v1"
EMBEDDING_OPERATION_SCHEMA = "openai-embeddings-v1"
EMBEDDING_SPACE_SCHEMA = "embedding-space-v1"
LLAMA_CPP_EMBEDDING_ADAPTER = "llama-cpp-embedding-v1"

_REQUIRED_LIMITS = frozenset(
    {
        "item_tokens",
        "item_bytes",
        "batch_items",
        "batch_bytes",
        "aggregate_tokens",
        "request_bytes",
    }
)
_ALLOWED_REQUEST_FIELDS = frozenset(
    {
        "model",
        "input",
        "encoding_format",
        "x_astrumweaver_input_type",
        "x_astrumweaver_embedding_space_id",
    }
)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class EmbeddingGatewayError(ValueError):
    """Public-safe embedding validation/normalization failure."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(message)


def _nonblank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EmbeddingGatewayError(
            "invalid_request",
            f"{name} must be a non-blank string",
        )
    return value.strip()


def _identifier(value: object, name: str) -> str:
    result = _nonblank(value, name)
    if _ID.fullmatch(result) is None:
        raise EmbeddingGatewayError(
            "invalid_profile",
            f"{name} must be a stable identifier",
        )
    return result


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise EmbeddingGatewayError(
            "invalid_profile",
            f"{name} must be a sha256 digest",
        )
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EmbeddingGatewayError(
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


def text_policy_digest(prefix: str, suffix: str = "") -> str:
    if not isinstance(prefix, str) or not isinstance(suffix, str):
        raise TypeError("preprocessing prefix/suffix must be strings")
    encoded = json.dumps(
        {"prefix": prefix, "suffix": suffix},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class EmbeddingSpaceIdentity:
    deployment_revision: str
    model_artifact_sha256: str
    quantization: str
    tokenizer_artifact_sha256: str
    pooling: str
    normalization: str
    dimensions: int
    query_policy_id: str
    query_preprocess_sha256: str
    document_policy_id: str
    document_preprocess_sha256: str
    adapter_id: str
    schema_version: str = EMBEDDING_SPACE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != EMBEDDING_SPACE_SCHEMA:
            raise EmbeddingGatewayError(
                "invalid_profile",
                "unsupported embedding-space schema",
            )
        for name in (
            "deployment_revision",
            "model_artifact_sha256",
            "tokenizer_artifact_sha256",
            "query_preprocess_sha256",
            "document_preprocess_sha256",
        ):
            _digest(getattr(self, name), name)
        for name in (
            "quantization",
            "pooling",
            "normalization",
            "query_policy_id",
            "document_policy_id",
            "adapter_id",
        ):
            _identifier(getattr(self, name), name)
        _positive_int(self.dimensions, "dimensions")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "deployment_revision": self.deployment_revision,
            "model_artifact_sha256": self.model_artifact_sha256,
            "quantization": self.quantization,
            "tokenizer_artifact_sha256": self.tokenizer_artifact_sha256,
            "pooling": self.pooling,
            "normalization": self.normalization,
            "dimensions": self.dimensions,
            "query_policy_id": self.query_policy_id,
            "query_preprocess_sha256": self.query_preprocess_sha256,
            "document_policy_id": self.document_policy_id,
            "document_preprocess_sha256": self.document_preprocess_sha256,
            "adapter_id": self.adapter_id,
        }

    @property
    def embedding_space_id(self) -> str:
        return _revision(self.to_dict())


@dataclass(frozen=True, slots=True)
class EmbeddingGatewayProfile:
    resolved: ResolvedServingProfile
    space: EmbeddingSpaceIdentity
    adapter_id: str
    query_prefix: str
    query_suffix: str
    document_prefix: str
    document_suffix: str
    request_timeout_seconds: float
    max_attempts: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.resolved, ResolvedServingProfile):
            raise TypeError("resolved must be ResolvedServingProfile")
        if not isinstance(self.space, EmbeddingSpaceIdentity):
            raise TypeError("space must be EmbeddingSpaceIdentity")
        adapter_id = _identifier(self.adapter_id, "adapter_id")
        if adapter_id != LLAMA_CPP_EMBEDDING_ADAPTER:
            raise EmbeddingGatewayError(
                "unsupported_adapter",
                "embedding gateway supports only the reviewed llama.cpp adapter",
            )
        if self.resolved.profile.capability != "text.embed":
            raise EmbeddingGatewayError(
                "invalid_profile",
                "embedding profile must bind text.embed",
            )
        if self.resolved.profile.operation_schema != EMBEDDING_OPERATION_SCHEMA:
            raise EmbeddingGatewayError(
                "invalid_profile",
                "embedding profile uses an unsupported operation schema",
            )
        if self.resolved.deployment.provider_id != "llama-cpp":
            raise EmbeddingGatewayError(
                "invalid_profile",
                "embedding profile must bind the reviewed llama.cpp provider",
            )
        deployment = self.resolved.deployment
        if deployment.tokenizer_artifact_sha256 is None:
            raise EmbeddingGatewayError(
                "invalid_profile",
                "embedding deployment requires tokenizer identity",
            )
        expected = {
            "deployment_revision": deployment.revision,
            "model_artifact_sha256": deployment.model_artifact_sha256,
            "quantization": deployment.quantization,
            "tokenizer_artifact_sha256": deployment.tokenizer_artifact_sha256,
            "adapter_id": adapter_id,
        }
        for name, value in expected.items():
            if getattr(self.space, name) != value:
                raise EmbeddingGatewayError(
                    "invalid_profile",
                    "embedding space does not match deployment identity",
                )
        if (
            text_policy_digest(self.query_prefix, self.query_suffix)
            != self.space.query_preprocess_sha256
        ):
            raise EmbeddingGatewayError(
                "invalid_profile",
                "query preprocessing does not match embedding space",
            )
        if (
            text_policy_digest(self.document_prefix, self.document_suffix)
            != self.space.document_preprocess_sha256
        ):
            raise EmbeddingGatewayError(
                "invalid_profile",
                "document preprocessing does not match embedding space",
            )
        if self.space.normalization != "l2":
            raise EmbeddingGatewayError(
                "invalid_profile",
                "first embedding adapter requires l2 normalization",
            )
        if self.resolved.contract.semantic_revision != self.embedding_space_id:
            raise EmbeddingGatewayError(
                "invalid_profile",
                "serving contract does not bind the embedding space identity",
            )
        required_runtime_features = {
            "float",
            f"pooling-{self.space.pooling}",
            f"normalization-{self.space.normalization}",
        }
        if not required_runtime_features <= self.resolved.contract.features:
            raise EmbeddingGatewayError(
                "invalid_profile",
                "serving contract does not prove embedding runtime semantics",
            )

        limits = dict(self.resolved.effective_limits)
        if _REQUIRED_LIMITS - limits.keys():
            raise EmbeddingGatewayError(
                "invalid_profile",
                "embedding profile is missing required bounded limits",
            )
        if limits["item_bytes"] > limits["batch_bytes"]:
            raise EmbeddingGatewayError(
                "invalid_profile",
                "per-item byte limit exceeds aggregate batch byte limit",
            )
        if limits["item_tokens"] > limits["aggregate_tokens"]:
            raise EmbeddingGatewayError(
                "invalid_profile",
                "per-item token limit exceeds aggregate token limit",
            )
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, (int, float))
            or not math.isfinite(float(self.request_timeout_seconds))
            or self.request_timeout_seconds <= 0
        ):
            raise EmbeddingGatewayError(
                "invalid_profile",
                "request_timeout_seconds must be finite and positive",
            )
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise EmbeddingGatewayError(
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
    def embedding_space_id(self) -> str:
        return self.space.embedding_space_id

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EmbeddingGatewayProfile":
        data = dict(value)
        try:
            deployment = DeploymentIdentity(**dict(data["deployment"]))
            contract = ServingContract(**dict(data["contract"]))
            profile = LogicalServingProfile(**dict(data["profile"]))
            resolved = resolve_profile(profile, contract, deployment)
            space = EmbeddingSpaceIdentity(**dict(data["space"]))
            return cls(
                resolved=resolved,
                space=space,
                adapter_id=data["adapter_id"],
                query_prefix=data.get("query_prefix", ""),
                query_suffix=data.get("query_suffix", ""),
                document_prefix=data.get("document_prefix", ""),
                document_suffix=data.get("document_suffix", ""),
                request_timeout_seconds=data.get("request_timeout_seconds", 60.0),
                max_attempts=data.get("max_attempts", 2),
            )
        except EmbeddingGatewayError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingGatewayError(
                "invalid_profile",
                "configured embedding profile is invalid",
            ) from exc


class EmbeddingProfileCatalog:
    def __init__(self, profiles: tuple[EmbeddingGatewayProfile, ...]) -> None:
        values = tuple(profiles)
        if not values:
            raise EmbeddingGatewayError(
                "invalid_catalog",
                "embedding catalog must contain at least one profile",
            )
        table: dict[str, EmbeddingGatewayProfile] = {}
        for profile in values:
            if not isinstance(profile, EmbeddingGatewayProfile):
                raise TypeError(
                    "profiles must contain EmbeddingGatewayProfile values"
                )
            if profile.profile_id in table:
                raise EmbeddingGatewayError(
                    "invalid_catalog",
                    "embedding profile IDs must be unique",
                )
            table[profile.profile_id] = profile
        self._profiles = values
        self._table = MappingProxyType(table)

    @property
    def profiles(self) -> tuple[EmbeddingGatewayProfile, ...]:
        return self._profiles

    def get(self, profile_id: str) -> EmbeddingGatewayProfile:
        try:
            return self._table[profile_id]
        except KeyError as exc:
            raise EmbeddingGatewayError(
                "model_not_found",
                "requested embedding profile is not configured",
                status_code=404,
            ) from exc

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EmbeddingProfileCatalog":
        data = dict(value)
        if data.get("schema_version") != EMBEDDING_CATALOG_SCHEMA:
            raise EmbeddingGatewayError(
                "invalid_catalog",
                "unsupported embedding catalog schema",
            )
        raw = data.get("profiles")
        if not isinstance(raw, list) or not all(
            isinstance(item, Mapping) for item in raw
        ):
            raise EmbeddingGatewayError(
                "invalid_catalog",
                "embedding catalog profiles must be an array of objects",
            )
        return cls(tuple(EmbeddingGatewayProfile.from_dict(item) for item in raw))


@dataclass(frozen=True, slots=True)
class CompiledEmbeddingRequest:
    profile: EmbeddingGatewayProfile
    input_type: str
    item_count: int
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.profile, EmbeddingGatewayProfile):
            raise TypeError("profile must be EmbeddingGatewayProfile")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


def compile_embedding_request(
    value: Mapping[str, object],
    profile: EmbeddingGatewayProfile,
    *,
    request_size_bytes: int,
) -> CompiledEmbeddingRequest:
    data = dict(value)
    extra = set(data) - _ALLOWED_REQUEST_FIELDS
    if extra:
        raise EmbeddingGatewayError(
            "unsupported_field",
            "embedding request uses unsupported fields",
        )
    if data.get("model") != profile.profile_id:
        raise EmbeddingGatewayError(
            "model_not_found",
            "request model does not match the configured embedding profile",
            status_code=404,
        )
    requested_space = data.get("x_astrumweaver_embedding_space_id")
    if not isinstance(requested_space, str) or _DIGEST.fullmatch(requested_space) is None:
        raise EmbeddingGatewayError(
            "invalid_request",
            "x_astrumweaver_embedding_space_id must be a sha256 digest",
        )
    if requested_space != profile.embedding_space_id:
        raise EmbeddingGatewayError(
            "embedding_space_mismatch",
            "requested embedding space does not match the configured profile",
            status_code=409,
        )

    encoding = data.get("encoding_format", "float")
    if encoding != "float":
        raise EmbeddingGatewayError(
            "unsupported_encoding",
            "only float embedding output is supported",
        )
    input_type = data.get("x_astrumweaver_input_type")
    if input_type not in {"query", "document"}:
        raise EmbeddingGatewayError(
            "invalid_request",
            "x_astrumweaver_input_type must be query or document",
        )

    raw_input = data.get("input")
    if isinstance(raw_input, str):
        items = [raw_input]
    elif isinstance(raw_input, list) and raw_input and all(
        isinstance(item, str) for item in raw_input
    ):
        items = list(raw_input)
    else:
        raise EmbeddingGatewayError(
            "invalid_request",
            "input must be one string or a non-empty array of strings",
        )
    if any(not item for item in items):
        raise EmbeddingGatewayError(
            "invalid_request",
            "embedding input strings must not be empty",
        )

    limits = dict(profile.resolved.effective_limits)
    if request_size_bytes > limits["request_bytes"]:
        raise EmbeddingGatewayError(
            "request_too_large",
            "embedding request exceeds the profile byte limit",
            status_code=413,
        )
    if len(items) > limits["batch_items"]:
        raise EmbeddingGatewayError(
            "batch_too_large",
            "embedding request exceeds the profile batch-item limit",
        )
    if input_type == "query":
        prefix, suffix = profile.query_prefix, profile.query_suffix
    else:
        prefix, suffix = profile.document_prefix, profile.document_suffix
    processed = [prefix + item + suffix for item in items]
    sizes = [len(item.encode("utf-8")) for item in processed]
    if any(size > limits["item_bytes"] for size in sizes):
        raise EmbeddingGatewayError(
            "input_too_large",
            "embedding item exceeds the profile byte limit",
        )
    if sum(sizes) > limits["batch_bytes"]:
        raise EmbeddingGatewayError(
            "batch_too_large",
            "embedding batch exceeds the aggregate byte limit",
        )

    payload = {
        "schema_version": EMBEDDING_JOB_SCHEMA,
        "adapter_id": profile.adapter_id,
        "embedding_space_id": profile.embedding_space_id,
        "request": {
            "input": processed,
            "encoding_format": "float",
        },
        "limits": {
            "item_tokens": limits["item_tokens"],
            "item_bytes": limits["item_bytes"],
            "batch_items": limits["batch_items"],
            "batch_bytes": limits["batch_bytes"],
            "aggregate_tokens": limits["aggregate_tokens"],
            "dimensions": profile.space.dimensions,
        },
    }
    return CompiledEmbeddingRequest(
        profile=profile,
        input_type=str(input_type),
        item_count=len(items),
        payload=payload,
    )


def validate_embedding_response(
    value: object,
    *,
    item_count: int,
    dimensions: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    if not isinstance(value, Mapping):
        raise EmbeddingGatewayError(
            "invalid_provider_response",
            "embedding backend returned a non-object response",
            status_code=502,
        )
    data = value.get("data")
    if not isinstance(data, list) or len(data) != item_count:
        raise EmbeddingGatewayError(
            "invalid_provider_response",
            "embedding backend returned the wrong number of vectors",
            status_code=502,
        )
    normalized: list[dict[str, object]] = []
    for expected_index, item in enumerate(data):
        if not isinstance(item, Mapping) or item.get("index") != expected_index:
            raise EmbeddingGatewayError(
                "invalid_provider_response",
                "embedding backend changed vector ordering",
                status_code=502,
            )
        vector = item.get("embedding")
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise EmbeddingGatewayError(
                "invalid_provider_response",
                "embedding backend returned an invalid vector dimension",
                status_code=502,
            )
        converted: list[float] = []
        for component in vector:
            if (
                isinstance(component, bool)
                or not isinstance(component, (int, float))
                or not math.isfinite(float(component))
            ):
                raise EmbeddingGatewayError(
                    "invalid_provider_response",
                    "embedding backend returned a non-finite vector",
                    status_code=502,
                )
            converted.append(float(component))
        norm = math.sqrt(sum(component * component for component in converted))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
            raise EmbeddingGatewayError(
                "invalid_provider_response",
                "embedding backend returned a vector that is not l2 normalized",
                status_code=502,
            )
        normalized.append(
            {
                "object": "embedding",
                "embedding": converted,
                "index": expected_index,
            }
        )

    usage: dict[str, int] = {}
    raw_usage = value.get("usage")
    if raw_usage is not None:
        if not isinstance(raw_usage, Mapping):
            raise EmbeddingGatewayError(
                "invalid_provider_response",
                "embedding backend returned invalid usage metadata",
                status_code=502,
            )
        for key in ("prompt_tokens", "total_tokens"):
            raw = raw_usage.get(key)
            if raw is not None:
                if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                    raise EmbeddingGatewayError(
                        "invalid_provider_response",
                        "embedding backend returned invalid token usage",
                        status_code=502,
                    )
                usage[key] = raw
    return normalized, usage


def normalize_embedding_response(
    value: object,
    *,
    profile: EmbeddingGatewayProfile,
    input_type: str,
    item_count: int,
) -> dict[str, object]:
    data, usage = validate_embedding_response(
        value,
        item_count=item_count,
        dimensions=profile.space.dimensions,
    )
    result: dict[str, object] = {
        "object": "list",
        "data": data,
        "model": profile.profile_id,
        "usage": usage,
        "x_astrumweaver": {
            "embedding_space_id": profile.embedding_space_id,
            "profile_revision": profile.resolved.profile_revision,
            "deployment_revision": profile.resolved.deployment.revision,
            "serving_contract_revision": profile.resolved.contract.revision,
            "input_type": input_type,
            "dimensions": profile.space.dimensions,
            "normalization": profile.space.normalization,
        },
    }
    return result


__all__ = [
    "CompiledEmbeddingRequest",
    "EMBEDDING_CATALOG_SCHEMA",
    "EMBEDDING_JOB_SCHEMA",
    "EMBEDDING_OPERATION_SCHEMA",
    "EMBEDDING_SPACE_SCHEMA",
    "EmbeddingGatewayError",
    "EmbeddingGatewayProfile",
    "EmbeddingProfileCatalog",
    "EmbeddingSpaceIdentity",
    "LLAMA_CPP_EMBEDDING_ADAPTER",
    "compile_embedding_request",
    "normalize_embedding_response",
    "text_policy_digest",
    "validate_embedding_response",
]
