from __future__ import annotations

from dataclasses import replace

import pytest

from astrumweaver.gateway.decision import (
    DECISION_CATALOG_SCHEMA,
    DECISION_OPERATION_SCHEMA,
    DECISION_SCORE_KIND,
    LLAMA_CPP_SCORE_SEMANTICS,
    LLAMA_CPP_SYSTEM_ONE_ADAPTER,
    DecisionGatewayError,
    DecisionGatewayProfile,
    DecisionProfileCatalog,
    DecisionSemanticsIdentity,
    compile_decision_request,
    normalize_decision_response,
    validate_decision_provider_response,
)
from astrumweaver.serving import (
    DeploymentIdentity,
    LogicalServingProfile,
    ServingContract,
    resolve_profile,
)


def digest(ch: str) -> str:
    return "sha256:" + ch * 64


def values(*, abstain_below: float = 0.6):
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
        abstain_below=abstain_below,
    )
    features = frozenset(
        {
            "native-decision-head",
            "choice-probabilities",
            "multi-token-choice-labels",
            "provider-temperature-softmax",
        }
    )
    limits = {
        "state_bytes": 1024,
        "question_bytes": 512,
        "choice_count": 8,
        "choice_id_bytes": 64,
        "choice_label_bytes": 256,
        "state_tokens": 128,
        "question_tokens": 64,
        "choice_label_tokens": 32,
        "aggregate_tokens": 256,
        "request_bytes": 4096,
    }
    contract = ServingContract(
        deployment_revision=deployment.revision,
        capability="decision.system_one",
        operation_schema=DECISION_OPERATION_SCHEMA,
        validation_evidence_sha256=digest("6"),
        features=features,
        limits=limits,
        semantic_revision=semantics.decision_semantics_id,
    )
    logical = LogicalServingProfile(
        profile_id="decision-local-v1",
        deployment_revision=deployment.revision,
        serving_contract_revision=contract.revision,
        capability="decision.system_one",
        operation_schema=DECISION_OPERATION_SCHEMA,
        required_features=features,
        limits={},
    )
    profile = DecisionGatewayProfile(
        resolved=resolve_profile(logical, contract, deployment),
        semantics=semantics,
        adapter_id=LLAMA_CPP_SYSTEM_ONE_ADAPTER,
        request_timeout_seconds=10,
        max_attempts=2,
    )
    return deployment, semantics, contract, logical, profile


def manifest(profile: DecisionGatewayProfile) -> dict:
    return {
        "schema_version": DECISION_CATALOG_SCHEMA,
        "profiles": [
            {
                "adapter_id": profile.adapter_id,
                "request_timeout_seconds": profile.request_timeout_seconds,
                "max_attempts": profile.max_attempts,
                "deployment": profile.resolved.deployment.to_dict(),
                "contract": profile.resolved.contract.to_dict(),
                "profile": profile.resolved.profile.to_dict(),
                "semantics": profile.semantics.to_dict(),
            }
        ],
    }


def request(profile: DecisionGatewayProfile) -> dict:
    return {
        "profile": profile.profile_id,
        "profile_revision": profile.resolved.profile_revision,
        "state": "A build failed after a dependency update.",
        "question": "What should the agent do next?",
        "choices": [
            {"id": "z-inspect", "label": "inspect the failure carefully"},
            {"id": "a-retry", "label": "retry without changing anything"},
            {"id": "m-escalate", "label": "ask a human for review"},
        ],
    }


def provider_response(
    probabilities=(0.7, 0.2, 0.1),
    *,
    choice="0000",
):
    return {
        "model": "astrumweaver",
        "answers": {
            "decision": {
                "type": "choice",
                "choice": choice,
                "probabilities": {
                    f"{index:04d}": value
                    for index, value in enumerate(probabilities)
                },
                "confidence": 0.55,
            }
        },
        "usage": {"input_tokens": 42, "output_tokens": 0},
    }


def test_catalog_round_trip_binds_semantics_revision():
    _, _, _, _, original = values()
    catalog = DecisionProfileCatalog.from_dict(manifest(original))
    loaded = catalog.get(original.profile_id)

    assert loaded.resolved == original.resolved
    assert loaded.semantics == original.semantics
    assert loaded.decision_semantics_id == original.decision_semantics_id
    assert loaded.binding == original.binding


def test_compile_preserves_order_with_positional_provider_keys():
    _, _, _, _, profile = values()
    compiled = compile_decision_request(
        request(profile),
        profile,
        request_size_bytes=700,
    )

    assert [choice_id for choice_id, _ in compiled.choices] == [
        "z-inspect",
        "a-retry",
        "m-escalate",
    ]
    assert compiled.provider_choice_keys == ("0000", "0001", "0002")
    criteria = compiled.payload["request"]["questions"]["decision"]["criteria"]
    assert list(criteria) == ["0000", "0001", "0002"]
    assert list(criteria.values()) == [
        "inspect the failure carefully",
        "retry without changing anything",
        "ask a human for review",
    ]


def test_multi_token_labels_are_not_reduced_to_output_tokens():
    _, _, _, _, profile = values()
    body = request(profile)
    body["choices"][0]["label"] = (
        "perform a careful multi step inspection before changing code"
    )

    compiled = compile_decision_request(
        body,
        profile,
        request_size_bytes=800,
    )

    assert compiled.choices[0][1].startswith("perform a careful multi")


@pytest.mark.parametrize(
    "mutator,match",
    (
        (
            lambda body: body["choices"].append(
                {"id": "z-inspect", "label": "duplicate"}
            ),
            "unique",
        ),
        (
            lambda body: body.update(
                {"profile_revision": digest("f")}
            ),
            "stale or incompatible",
        ),
        (
            lambda body: body.update({"state": "x" * 1025}),
            "state exceeds",
        ),
    ),
)
def test_request_fail_closed_validation(mutator, match):
    _, _, _, _, profile = values()
    body = request(profile)
    mutator(body)

    with pytest.raises(DecisionGatewayError, match=match):
        compile_decision_request(
            body,
            profile,
            request_size_bytes=1500,
        )


def test_provider_response_requires_complete_finite_normalized_scores():
    scores, usage = validate_decision_provider_response(
        provider_response(),
        provider_choice_keys=("0000", "0001", "0002"),
    )
    assert scores == [0.7, 0.2, 0.1]
    assert usage == {"input_tokens": 42, "output_tokens": 0}

    nonfinite = provider_response((float("nan"), 0.5, 0.5))
    with pytest.raises(DecisionGatewayError, match="non-finite"):
        validate_decision_provider_response(
            nonfinite,
            provider_choice_keys=("0000", "0001", "0002"),
        )

    incomplete = provider_response()
    del incomplete["answers"]["decision"]["probabilities"]["0002"]
    with pytest.raises(DecisionGatewayError, match="probability set"):
        validate_decision_provider_response(
            incomplete,
            provider_choice_keys=("0000", "0001", "0002"),
        )

    not_normalized = provider_response((0.6, 0.2, 0.1))
    with pytest.raises(DecisionGatewayError, match="not normalized"):
        validate_decision_provider_response(
            not_normalized,
            provider_choice_keys=("0000", "0001", "0002"),
        )

    wrong_choice = provider_response(choice="0001")
    with pytest.raises(DecisionGatewayError, match="non-maximal"):
        validate_decision_provider_response(
            wrong_choice,
            provider_choice_keys=("0000", "0001", "0002"),
        )


def test_normalized_response_maps_scores_and_abstains_without_failure():
    _, _, _, _, profile = values(abstain_below=0.6)
    compiled = compile_decision_request(
        request(profile),
        profile,
        request_size_bytes=700,
    )

    accepted = normalize_decision_response(
        provider_response((0.7, 0.2, 0.1)),
        compiled=compiled,
    )
    assert accepted["choice_id"] == "z-inspect"
    assert accepted["abstained"] is False
    assert [item["id"] for item in accepted["scores"]] == [
        "z-inspect",
        "a-retry",
        "m-escalate",
    ]
    assert accepted["score_kind"] == DECISION_SCORE_KIND
    assert accepted["calibration"] == {
        "status": "uncalibrated",
        "reference_sha256": None,
    }
    assert accepted["x_astrumweaver"]["mode"] == "shadow"
    assert accepted["x_astrumweaver"]["authority"] == "recommendation-only"

    abstained = normalize_decision_response(
        provider_response((0.4, 0.35, 0.25)),
        compiled=compiled,
    )
    assert abstained["choice_id"] is None
    assert abstained["abstained"] is True
    assert [item["score"] for item in abstained["scores"]] == [
        0.4,
        0.35,
        0.25,
    ]


def test_calibration_claim_requires_reference_and_changes_semantics():
    deployment, semantics, _, _, _ = values()

    with pytest.raises(DecisionGatewayError, match="require"):
        replace(
            semantics,
            calibration_status="calibrated",
        )

    calibrated = DecisionSemanticsIdentity(
        deployment_revision=deployment.revision,
        adapter_id=LLAMA_CPP_SYSTEM_ONE_ADAPTER,
        score_kind=DECISION_SCORE_KIND,
        provider_score_semantics=LLAMA_CPP_SCORE_SEMANTICS,
        calibration_status="calibrated",
        calibration_reference_sha256=digest("a"),
        abstain_below=0.6,
    )
    assert calibrated.decision_semantics_id != semantics.decision_semantics_id


def test_profile_rejects_missing_runtime_semantics_feature():
    deployment, semantics, contract, logical, _ = values()
    weakened = replace(
        contract,
        features=frozenset(
            feature
            for feature in contract.features
            if feature != "native-decision-head"
        ),
    )
    changed = replace(
        logical,
        serving_contract_revision=weakened.revision,
        required_features=weakened.features,
    )

    with pytest.raises(DecisionGatewayError, match="runtime semantics"):
        DecisionGatewayProfile(
            resolved=resolve_profile(changed, weakened, deployment),
            semantics=semantics,
            adapter_id=LLAMA_CPP_SYSTEM_ONE_ADAPTER,
            request_timeout_seconds=10,
        )