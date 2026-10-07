from __future__ import annotations

import json

import pytest

from astrumweaver.control.daemon import build_app
from astrumweaver.gateway.decision import (
    DECISION_CATALOG_SCHEMA,
    DECISION_OPERATION_SCHEMA,
    DECISION_SCORE_KIND,
    LLAMA_CPP_SCORE_SEMANTICS,
    LLAMA_CPP_SYSTEM_ONE_ADAPTER,
    DecisionSemanticsIdentity,
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
        quantization="Q4_K_M",
        tokenizer_artifact_sha256=digest("5"),
    )
    semantics = DecisionSemanticsIdentity(
        deployment_revision=deployment.revision,
        adapter_id=LLAMA_CPP_SYSTEM_ONE_ADAPTER,
        score_kind=DECISION_SCORE_KIND,
        provider_score_semantics=LLAMA_CPP_SCORE_SEMANTICS,
        calibration_status="uncalibrated",
        calibration_reference_sha256=None,
        abstain_below=0.6,
    )
    features = frozenset(
        {
            "native-decision-head",
            "choice-probabilities",
            "multi-token-choice-labels",
            "provider-temperature-softmax",
        }
    )
    contract = ServingContract(
        deployment_revision=deployment.revision,
        capability="decision.system_one",
        operation_schema=DECISION_OPERATION_SCHEMA,
        validation_evidence_sha256=digest("6"),
        features=features,
        limits={
            "state_bytes": 4096,
            "question_bytes": 1024,
            "choice_count": 8,
            "choice_id_bytes": 64,
            "choice_label_bytes": 512,
            "state_tokens": 512,
            "question_tokens": 128,
            "choice_label_tokens": 64,
            "aggregate_tokens": 1024,
            "request_bytes": 16384,
        },
        semantic_revision=semantics.decision_semantics_id,
    )
    profile = LogicalServingProfile(
        profile_id="decision-local-v1",
        deployment_revision=deployment.revision,
        serving_contract_revision=contract.revision,
        capability="decision.system_one",
        operation_schema=DECISION_OPERATION_SCHEMA,
        required_features=features,
        limits={},
    )
    return {
        "schema_version": DECISION_CATALOG_SCHEMA,
        "profiles": [
            {
                "adapter_id": LLAMA_CPP_SYSTEM_ONE_ADAPTER,
                "request_timeout_seconds": 30,
                "max_attempts": 2,
                "deployment": deployment.to_dict(),
                "contract": contract.to_dict(),
                "profile": profile.to_dict(),
                "semantics": semantics.to_dict(),
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


def test_control_daemon_mounts_decision_gateway_only_when_enabled(
    tmp_path,
    monkeypatch,
):
    base_environment(monkeypatch)
    catalog = tmp_path / "decision-catalog.json"
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
                "[decision_gateway]",
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
    assert "/v1/decisions" in paths
    assert "/v1/decision-profiles" in paths
    assert "/v1/embeddings" not in paths
    assert "/v1/chat/completions" not in paths

    disabled = tmp_path / "disabled.toml"
    disabled.write_text(
        '[control]\nclient_auth = "none"\n',
        encoding="utf-8",
    )
    app = build_app(str(disabled))
    paths = set(app.openapi()["paths"])
    assert "/v1/decisions" not in paths
    assert "/v1/decision-profiles" not in paths


def test_control_daemon_requires_valid_decision_catalog(
    tmp_path,
    monkeypatch,
):
    base_environment(monkeypatch)
    missing = tmp_path / "missing.toml"
    missing.write_text(
        '[control]\nclient_auth = "none"\n'
        "[decision_gateway]\nenabled = true\n",
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError,
        match="decision_gateway.catalog",
    ):
        build_app(str(missing))

    invalid_catalog = tmp_path / "invalid.json"
    invalid_catalog.write_text("{}", encoding="utf-8")
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        '[control]\nclient_auth = "none"\n'
        "[decision_gateway]\nenabled = true\n"
        f'catalog = "{invalid_catalog}"\n',
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError,
        match="profile catalog is invalid",
    ):
        build_app(str(invalid))