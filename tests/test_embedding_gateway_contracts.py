from __future__ import annotations

from dataclasses import replace
import math

import pytest

from astrumweaver.gateway.embedding import (
    EMBEDDING_CATALOG_SCHEMA,
    EMBEDDING_OPERATION_SCHEMA,
    LLAMA_CPP_EMBEDDING_ADAPTER,
    EmbeddingGatewayError,
    EmbeddingGatewayProfile,
    EmbeddingProfileCatalog,
    EmbeddingSpaceIdentity,
    compile_embedding_request,
    normalize_embedding_response,
    text_policy_digest,
    validate_embedding_response,
)
from astrumweaver.serving import (
    DeploymentIdentity,
    LogicalServingProfile,
    ServingContract,
    resolve_profile,
)


def digest(ch: str) -> str:
    return "sha256:" + ch * 64


QUERY_PREFIX = "Instruct: retrieve the relevant passage\nQuery: "
DOCUMENT_PREFIX = ""


def values():
    deployment = DeploymentIdentity(
        provider_id="llama-cpp",
        runtime_artifact_sha256=digest("1"),
        adapter_artifact_sha256=digest("2"),
        model_artifact_sha256=digest("3"),
        execution_config_sha256=digest("4"),
        quantization="Q8_0",
        tokenizer_artifact_sha256=digest("5"),
    )
    space = EmbeddingSpaceIdentity(
        deployment_revision=deployment.revision,
        model_artifact_sha256=deployment.model_artifact_sha256,
        quantization=deployment.quantization,
        tokenizer_artifact_sha256=deployment.tokenizer_artifact_sha256 or "",
        pooling="last",
        normalization="l2",
        dimensions=1024,
        query_policy_id="qwen3-retrieval-v1",
        query_preprocess_sha256=text_policy_digest(QUERY_PREFIX),
        document_policy_id="plain-document-v1",
        document_preprocess_sha256=text_policy_digest(DOCUMENT_PREFIX),
        adapter_id=LLAMA_CPP_EMBEDDING_ADAPTER,
    )
    contract = ServingContract(
        deployment_revision=deployment.revision,
        capability="text.embed",
        operation_schema=EMBEDDING_OPERATION_SCHEMA,
        validation_evidence_sha256=digest("6"),
        features=frozenset(
            {"float", "pooling-last", "normalization-l2"}
        ),
        limits={
            "item_tokens": 256,
            "item_bytes": 4096,
            "batch_items": 4,
            "batch_bytes": 8192,
            "aggregate_tokens": 512,
            "request_bytes": 16384,
        },
        semantic_revision=space.embedding_space_id,
    )
    logical = LogicalServingProfile(
        profile_id="notes-embed-v1",
        deployment_revision=deployment.revision,
        serving_contract_revision=contract.revision,
        capability="text.embed",
        operation_schema=EMBEDDING_OPERATION_SCHEMA,
        required_features=frozenset(
            {"float", "pooling-last", "normalization-l2"}
        ),
        limits={},
    )
    resolved = resolve_profile(logical, contract, deployment)
    gateway = EmbeddingGatewayProfile(
        resolved=resolved,
        space=space,
        adapter_id=LLAMA_CPP_EMBEDDING_ADAPTER,
        query_prefix=QUERY_PREFIX,
        query_suffix="",
        document_prefix=DOCUMENT_PREFIX,
        document_suffix="",
        request_timeout_seconds=30.0,
        max_attempts=2,
    )
    return deployment, contract, logical, gateway


def manifest(gateway: EmbeddingGatewayProfile) -> dict:
    return {
        "schema_version": EMBEDDING_CATALOG_SCHEMA,
        "profiles": [
            {
                "adapter_id": gateway.adapter_id,
                "request_timeout_seconds": gateway.request_timeout_seconds,
                "max_attempts": gateway.max_attempts,
                "query_prefix": gateway.query_prefix,
                "query_suffix": gateway.query_suffix,
                "document_prefix": gateway.document_prefix,
                "document_suffix": gateway.document_suffix,
                "deployment": gateway.resolved.deployment.to_dict(),
                "contract": gateway.resolved.contract.to_dict(),
                "profile": gateway.resolved.profile.to_dict(),
                "space": gateway.space.to_dict(),
            }
        ],
    }


def test_embedding_space_identity_is_semantic_and_catalog_round_trips():
    _, _, _, gateway = values()
    catalog = EmbeddingProfileCatalog.from_dict(manifest(gateway))
    loaded = catalog.get("notes-embed-v1")

    assert loaded.embedding_space_id == gateway.embedding_space_id
    assert loaded.binding == gateway.binding
    assert loaded.space.dimensions == 1024


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_artifact_sha256", digest("a")),
        ("quantization", "Q6_K"),
        ("tokenizer_artifact_sha256", digest("b")),
        ("pooling", "mean"),
        ("normalization", "none"),
        ("dimensions", 768),
        ("query_preprocess_sha256", digest("c")),
        ("document_preprocess_sha256", digest("d")),
        ("adapter_id", "other-embedding-v1"),
    ),
)
def test_space_defining_changes_change_identity(field, value):
    _, _, _, gateway = values()
    changed = replace(gateway.space, **{field: value})
    assert changed.embedding_space_id != gateway.embedding_space_id


def test_profile_rejects_contract_not_bound_to_space():
    deployment, contract, logical, gateway = values()
    bad = replace(contract, semantic_revision=digest("f"))
    bad_logical = replace(
        logical,
        serving_contract_revision=bad.revision,
    )
    resolved = resolve_profile(bad_logical, bad, deployment)

    with pytest.raises(EmbeddingGatewayError, match="space identity"):
        replace(gateway, resolved=resolved)


def test_compile_query_and_document_apply_exact_reviewed_policy():
    _, _, _, gateway = values()

    query = compile_embedding_request(
        {
            "model": gateway.profile_id,
            "input": ["alpha", "beta"],
            "encoding_format": "float",
            "x_astrumweaver_input_type": "query",
            "x_astrumweaver_embedding_space_id": gateway.embedding_space_id,
        },
        gateway,
        request_size_bytes=100,
    )
    assert query.item_count == 2
    assert query.payload["request"]["input"] == [
        QUERY_PREFIX + "alpha",
        QUERY_PREFIX + "beta",
    ]
    assert query.payload["embedding_space_id"] == gateway.embedding_space_id

    document = compile_embedding_request(
        {
            "model": gateway.profile_id,
            "input": "alpha",
            "x_astrumweaver_input_type": "document",
            "x_astrumweaver_embedding_space_id": gateway.embedding_space_id,
        },
        gateway,
        request_size_bytes=60,
    )
    assert document.payload["request"]["input"] == ["alpha"]


@pytest.mark.parametrize(
    "body",
    (
        {
            "model": "notes-embed-v1",
            "input": "x",
            "encoding_format": "base64",
            "x_astrumweaver_input_type": "document",
        },
        {
            "model": "notes-embed-v1",
            "input": [[1, 2, 3]],
            "x_astrumweaver_input_type": "document",
        },
        {
            "model": "notes-embed-v1",
            "input": "x",
            "dimensions": 768,
            "x_astrumweaver_input_type": "document",
        },
        {
            "model": "notes-embed-v1",
            "input": "x",
            "x_astrumweaver_input_type": "unknown",
        },
    ),
)
def test_compile_rejects_unsupported_embedding_shapes(body):
    _, _, _, gateway = values()
    pinned = {
        **body,
        "x_astrumweaver_embedding_space_id": gateway.embedding_space_id,
    }
    with pytest.raises(EmbeddingGatewayError):
        compile_embedding_request(pinned, gateway, request_size_bytes=100)


def test_compile_requires_exact_embedding_space_pin():
    _, _, _, gateway = values()
    body = {
        "model": gateway.profile_id,
        "input": "alpha",
        "x_astrumweaver_input_type": "document",
    }

    with pytest.raises(EmbeddingGatewayError, match="must be a sha256 digest"):
        compile_embedding_request(body, gateway, request_size_bytes=60)

    with pytest.raises(EmbeddingGatewayError) as mismatch:
        compile_embedding_request(
            {
                **body,
                "x_astrumweaver_embedding_space_id": digest("f"),
            },
            gateway,
            request_size_bytes=60,
        )
    assert mismatch.value.code == "embedding_space_mismatch"
    assert mismatch.value.status_code == 409


def test_compile_enforces_batch_and_byte_bounds():
    _, _, _, gateway = values()

    with pytest.raises(EmbeddingGatewayError, match="batch-item"):
        compile_embedding_request(
            {
                "model": gateway.profile_id,
                "input": ["a", "b", "c", "d", "e"],
                "x_astrumweaver_input_type": "document",
                "x_astrumweaver_embedding_space_id": gateway.embedding_space_id,
            },
            gateway,
            request_size_bytes=100,
        )

    with pytest.raises(EmbeddingGatewayError, match="item exceeds"):
        compile_embedding_request(
            {
                "model": gateway.profile_id,
                "input": "x" * 4097,
                "x_astrumweaver_input_type": "document",
                "x_astrumweaver_embedding_space_id": gateway.embedding_space_id,
            },
            gateway,
            request_size_bytes=5000,
        )


def response(dimensions=1024):
    first = [0.0] * dimensions
    first[0] = 1.0
    second = [1.0 / math.sqrt(dimensions)] * dimensions
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": first,
                "index": 0,
            },
            {
                "object": "embedding",
                "embedding": second,
                "index": 1,
            },
        ],
        "usage": {"prompt_tokens": 12, "total_tokens": 12},
    }


def test_embedding_response_preserves_order_dimension_and_space_extension():
    _, _, _, gateway = values()
    normalized = normalize_embedding_response(
        response(),
        profile=gateway,
        input_type="document",
        item_count=2,
    )
    assert normalized["model"] == "notes-embed-v1"
    assert [item["index"] for item in normalized["data"]] == [0, 1]
    assert normalized["x_astrumweaver"]["embedding_space_id"] == (
        gateway.embedding_space_id
    )
    assert normalized["x_astrumweaver"]["dimensions"] == 1024


def test_embedding_response_rejects_wrong_dimension_order_and_nonfinite():
    with pytest.raises(EmbeddingGatewayError, match="dimension"):
        validate_embedding_response(response(3), item_count=2, dimensions=1024)

    out_of_order = response()
    out_of_order["data"][0]["index"] = 1
    with pytest.raises(EmbeddingGatewayError, match="ordering"):
        validate_embedding_response(out_of_order, item_count=2, dimensions=1024)

    nonfinite = response()
    nonfinite["data"][0]["embedding"][0] = float("nan")
    with pytest.raises(EmbeddingGatewayError, match="non-finite"):
        validate_embedding_response(nonfinite, item_count=2, dimensions=1024)

    unnormalized = response()
    unnormalized["data"][0]["embedding"][0] = 2.0
    with pytest.raises(EmbeddingGatewayError, match="l2 normalized"):
        validate_embedding_response(unnormalized, item_count=2, dimensions=1024)
