from __future__ import annotations

import json

import pytest

from astrumweaver.control.daemon import build_app
from astrumweaver.gateway.embedding import (
    EMBEDDING_CATALOG_SCHEMA,
    EMBEDDING_OPERATION_SCHEMA,
    LLAMA_CPP_EMBEDDING_ADAPTER,
    EmbeddingSpaceIdentity,
    text_policy_digest,
)
from astrumweaver.serving import (
    DeploymentIdentity,
    LogicalServingProfile,
    ServingContract,
)


def digest(ch: str) -> str:
    return "sha256:" + ch * 64


def catalog_document() -> dict:
    deployment = DeploymentIdentity(
        provider_id="llama-cpp",
        runtime_artifact_sha256=digest("1"),
        adapter_artifact_sha256=digest("2"),
        model_artifact_sha256=digest("3"),
        execution_config_sha256=digest("4"),
        quantization="Q8_0",
        tokenizer_artifact_sha256=digest("5"),
    )
    query_prefix = "Instruct: retrieve relevant notes\nQuery: "
    space = EmbeddingSpaceIdentity(
        deployment_revision=deployment.revision,
        model_artifact_sha256=deployment.model_artifact_sha256,
        quantization=deployment.quantization,
        tokenizer_artifact_sha256=deployment.tokenizer_artifact_sha256 or "",
        pooling="last",
        normalization="l2",
        dimensions=1024,
        query_policy_id="notes-query-v1",
        query_preprocess_sha256=text_policy_digest(query_prefix),
        document_policy_id="notes-document-v1",
        document_preprocess_sha256=text_policy_digest(""),
        adapter_id=LLAMA_CPP_EMBEDDING_ADAPTER,
    )
    features = frozenset(
        {"float", "pooling-last", "normalization-l2"}
    )
    contract = ServingContract(
        deployment_revision=deployment.revision,
        capability="text.embed",
        operation_schema=EMBEDDING_OPERATION_SCHEMA,
        validation_evidence_sha256=digest("6"),
        features=features,
        limits={
            "item_tokens": 512,
            "item_bytes": 4096,
            "batch_items": 8,
            "batch_bytes": 16384,
            "aggregate_tokens": 2048,
            "request_bytes": 32768,
        },
        semantic_revision=space.embedding_space_id,
    )
    profile = LogicalServingProfile(
        profile_id="notes-embed-v1",
        deployment_revision=deployment.revision,
        serving_contract_revision=contract.revision,
        capability="text.embed",
        operation_schema=EMBEDDING_OPERATION_SCHEMA,
        required_features=features,
        limits={},
    )
    return {
        "schema_version": EMBEDDING_CATALOG_SCHEMA,
        "profiles": [
            {
                "adapter_id": LLAMA_CPP_EMBEDDING_ADAPTER,
                "request_timeout_seconds": 30,
                "max_attempts": 2,
                "query_prefix": query_prefix,
                "document_prefix": "",
                "deployment": deployment.to_dict(),
                "contract": contract.to_dict(),
                "profile": profile.to_dict(),
                "space": space.to_dict(),
            }
        ],
    }


def base_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "ASTRUMWEAVER_DATABASE_URL",
        "postgresql://example.invalid/astrumweaver",
    )
    monkeypatch.setenv("ASTRUMWEAVER_WORKER_TOKEN", "worker-secret")
    monkeypatch.delenv("ASTRUMWEAVER_CLIENT_TOKEN", raising=False)


def test_control_daemon_mounts_embedding_gateway_only_when_enabled(
    tmp_path,
    monkeypatch,
):
    base_environment(monkeypatch)
    catalog = tmp_path / "embedding-catalog.json"
    catalog.write_text(
        json.dumps(catalog_document()),
        encoding="utf-8",
    )
    config = tmp_path / "control.toml"
    config.write_text(
        "\n".join(
            [
                "[control]",
                'client_auth = "none"',
                "",
                "[embedding_gateway]",
                "enabled = true",
                f'catalog = "{catalog}"',
                "poll_interval_seconds = 0.01",
                "",
            ]
        ),
        encoding="utf-8",
    )

    app = build_app(str(config))
    paths = set(app.openapi()["paths"])
    assert "/v1/embeddings" in paths
    assert "/v1/embedding-spaces" in paths
    assert "/v1/chat/completions" not in paths

    disabled = tmp_path / "disabled.toml"
    disabled.write_text(
        '[control]\nclient_auth = "none"\n',
        encoding="utf-8",
    )
    app = build_app(str(disabled))
    paths = set(app.openapi()["paths"])
    assert "/v1/embeddings" not in paths
    assert "/v1/embedding-spaces" not in paths


def test_control_daemon_requires_valid_embedding_catalog(
    tmp_path,
    monkeypatch,
):
    base_environment(monkeypatch)
    missing = tmp_path / "missing.toml"
    missing.write_text(
        '[control]\nclient_auth = "none"\n'
        "[embedding_gateway]\nenabled = true\n",
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError,
        match="embedding_gateway.catalog",
    ):
        build_app(str(missing))

    invalid_catalog = tmp_path / "invalid.json"
    invalid_catalog.write_text("{}", encoding="utf-8")
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        '[control]\nclient_auth = "none"\n'
        "[embedding_gateway]\nenabled = true\n"
        f'catalog = "{invalid_catalog}"\n',
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError,
        match="profile catalog is invalid",
    ):
        build_app(str(invalid))
