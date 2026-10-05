"""Pure serving-contract coverage; not live admission/compatibility evidence."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from astrumweaver.serving import (
    DeploymentIdentity,
    LogicalServingProfile,
    ResolvedServingProfile,
    RuntimeInstance,
    ServingContract,
    ServingContractError,
    resolve_profile,
)


def digest(character: str) -> str:
    return "sha256:" + character * 64


@pytest.fixture
def deployment():
    return DeploymentIdentity(
        provider_id="llama-cpp",
        runtime_artifact_sha256=digest("1"),
        adapter_artifact_sha256=digest("2"),
        model_artifact_sha256=digest("3"),
        execution_config_sha256=digest("4"),
        quantization="Q6_K",
        tokenizer_artifact_sha256=digest("5"),
        template_artifact_sha256=digest("6"),
    )


@pytest.fixture
def contract(deployment):
    return ServingContract(
        deployment_revision=deployment.revision,
        capability="llm.chat",
        operation_schema="chat-v1",
        validation_evidence_sha256=digest("7"),
        features=frozenset({"tools", "json"}),
        limits={"input_tokens": 8192, "output_tokens": 2048, "total_tokens": 10240},
    )


@pytest.fixture
def profile(deployment, contract):
    return LogicalServingProfile(
        profile_id="local-code-v1",
        deployment_revision=deployment.revision,
        serving_contract_revision=contract.revision,
        capability=contract.capability,
        operation_schema=contract.operation_schema,
        required_features=frozenset({"tools"}),
        limits={"input_tokens": 4096},
    )


def assert_error(code, fn):
    with pytest.raises(ServingContractError) as caught:
        fn()
    assert caught.value.code == code
    return caught.value


def test_canonical_wire_values_and_revision(deployment):
    payload = deployment.to_dict()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert deployment.revision == "sha256:" + hashlib.sha256(encoded).hexdigest()
    assert DeploymentIdentity(**json.loads(json.dumps(payload))) == deployment
    assert deployment.revision == "sha256:e73c33900e1c29f999f6563a3782d0c2ec55363793f72546412ced58e7c63335"


def test_order_independent_contract_revision(contract):
    reordered = replace(
        contract, features=["json", "tools"],
        limits=dict(reversed(list(contract.limits.items()))),
    )
    assert reordered.revision == contract.revision
    assert ServingContract(**contract.to_dict()) == contract


def test_profile_roundtrip_and_semantic_order(profile):
    assert LogicalServingProfile(**profile.to_dict()) == profile
    a = replace(profile, required_features=["tools", "json"])
    b = replace(profile, required_features=["json", "tools"])
    assert a.revision == b.revision


@pytest.mark.parametrize("name,value", [
    ("provider_id", "other-provider"), ("quantization", "Q4_K_M"),
    *[(name, digest("a")) for name in (
        "runtime_artifact_sha256", "adapter_artifact_sha256", "model_artifact_sha256",
        "execution_config_sha256", "tokenizer_artifact_sha256", "template_artifact_sha256",
    )],
    ("template_artifact_sha256", None), ("tokenizer_artifact_sha256", None),
])
def test_each_execution_input_changes_revision(deployment, name, value):
    assert replace(deployment, **{name: value}).revision != deployment.revision


@pytest.mark.parametrize("name,value", [
    ("deployment_revision", digest("a")), ("capability", "text.embed"),
    ("operation_schema", "chat-v2"), ("validation_evidence_sha256", digest("a")),
    ("features", {"tools"}), ("limits", {"input_tokens": 8191}),
    ("semantic_revision", digest("b")),
])
def test_each_contract_input_changes_revision(contract, name, value):
    assert replace(contract, **{name: value}).revision != contract.revision


@pytest.mark.parametrize("name,value", [
    ("profile_id", "local-code-v2"), ("deployment_revision", digest("a")),
    ("serving_contract_revision", digest("a")), ("capability", "text.embed"),
    ("operation_schema", "chat-v2"), ("required_features", {"json"}),
    ("limits", {"input_tokens": 4095}),
])
def test_each_profile_input_changes_revision(profile, name, value):
    assert replace(profile, **{name: value}).revision != profile.revision


@pytest.mark.parametrize("value", ["", " model", "model ", "model/path", "https://host", 1, None, "x" * 129])
def test_identifiers_are_not_silently_normalized(deployment, value):
    assert_error("invalid-identifier", lambda: replace(deployment, provider_id=value))


@pytest.mark.parametrize("value", ["", "a" * 64, digest("A"), digest("g"), digest("1") + "\n", 1, None])
def test_content_identity_requires_canonical_digest(deployment, value):
    assert_error("invalid-digest", lambda: replace(deployment, model_artifact_sha256=value))


@pytest.mark.parametrize("kind", ["deployment", "contract", "profile", "instance"])
@pytest.mark.parametrize("version", ["v2", "", 1, None])
def test_unknown_schema_is_rejected(deployment, contract, profile, kind, version):
    instance = RuntimeInstance(deployment.revision, "12345678-1234-4234-9234-123456789abc")
    value = {"deployment": deployment, "contract": contract, "profile": profile, "instance": instance}[kind]
    assert_error("unsupported-schema", lambda: replace(value, schema_version=version))


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, float("nan"), float("inf"), "4", None, 2**63])
def test_limits_require_bounded_positive_integers(contract, profile, value):
    for obj in (contract, profile):
        assert_error("invalid-quantity", lambda: replace(obj, limits={"input_tokens": value}))


@pytest.mark.parametrize("value", [None, [], "limits", 1])
def test_limits_require_a_mapping(contract, value):
    assert_error("invalid-quantities", lambda: replace(contract, limits=value))


@pytest.mark.parametrize("value,code", [
    ("tools", "invalid-features"), (None, "invalid-features"),
    ({"tools": True}, "invalid-features"), (["tools", "tools"], "duplicate-feature"),
    ([" tools"], "invalid-identifier"), ([None], "invalid-identifier"),
])
def test_malformed_feature_sets(contract, profile, value, code):
    assert_error(code, lambda: replace(contract, features=value))
    assert_error(code, lambda: replace(profile, required_features=value))


def test_defensive_copy_and_detached_serialization(contract, profile):
    limits = {"input_tokens": 20}
    features = ["tools"]
    c = replace(contract, limits=limits, features=features)
    p = replace(profile, limits=limits, required_features=features)
    revisions = (c.revision, p.revision)
    limits["input_tokens"] = 999
    features.append("streaming")
    for obj in (c, p):
        view = obj.to_dict()
        view["limits"]["input_tokens"] = 1000
        feature_key = "features" if isinstance(obj, ServingContract) else "required_features"
        view[feature_key].append("unknown")
        with pytest.raises(TypeError):
            obj.limits["input_tokens"] = 30
        with pytest.raises(FrozenInstanceError):
            obj.limits = {}
    assert revisions == (c.revision, p.revision)


def test_restart_epoch_does_not_change_deployment_revision(deployment):
    first = RuntimeInstance(deployment.revision, "12345678-1234-4234-9234-123456789abc")
    second = replace(first, epoch="22345678-1234-4234-9234-123456789abc")
    assert first != second
    assert first.deployment_revision == second.deployment_revision == deployment.revision


@pytest.mark.parametrize("epoch", [None, 1, "", "not-uuid", "12345678123442349234123456789abc", "12345678-1234-4234-9234-123456789ABC", "00000000-0000-0000-0000-000000000000"])
def test_noncanonical_epochs_rejected(deployment, epoch):
    assert_error("invalid-epoch", lambda: RuntimeInstance(deployment.revision, epoch))


def test_resolve_inherits_then_narrows_limits(deployment, contract, profile):
    result = resolve_profile(
        profile, contract, deployment, required_features={"json"},
        requested_usage={"input_tokens": 4096, "output_tokens": 0, "total_tokens": 4096},
        explicit_provider_id=deployment.provider_id,
        explicit_model_sha256=deployment.model_artifact_sha256,
    )
    assert result.effective_limits == {"input_tokens": 4096, "output_tokens": 2048, "total_tokens": 10240}
    assert result.profile_revision == profile.revision
    with pytest.raises(TypeError):
        result.effective_limits["input_tokens"] = 8192


@pytest.mark.parametrize("name,value,code", [
    ("deployment_revision", digest("a"), "deployment-revision-mismatch"),
    ("serving_contract_revision", digest("a"), "serving-contract-mismatch"),
    ("capability", "text.embed", "capability-mismatch"),
    ("operation_schema", "chat-v2", "operation-schema-mismatch"),
    ("required_features", {"streaming"}, "unsupported-feature"),
    ("limits", {"missing_limit": 1}, "unknown-limit"),
    ("limits", {"input_tokens": 8193}, "profile-limit-exceeded"),
])
def test_profile_never_weakens_or_changes_contract(deployment, contract, profile, name, value, code):
    assert_error(code, lambda: resolve_profile(replace(profile, **{name: value}), contract, deployment))


def test_contract_cannot_bind_another_deployment(deployment, contract, profile):
    other = replace(contract, deployment_revision=digest("a"))
    p = replace(profile, serving_contract_revision=other.revision)
    assert_error("deployment-revision-mismatch", lambda: resolve_profile(p, other, deployment))


@pytest.mark.parametrize("kwargs,code", [
    ({"explicit_provider_id": "ollama"}, "provider-selection-mismatch"),
    ({"explicit_model_sha256": digest("a")}, "model-selection-mismatch"),
    ({"required_features": {"streaming"}}, "unsupported-feature"),
    ({"required_features": "tools"}, "invalid-features"),
    ({"requested_usage": {"input_tokens": 4097}}, "request-limit-exceeded"),
    ({"requested_usage": {"unknown": 1}}, "unknown-limit"),
    ({"requested_usage": {"input_tokens": True}}, "invalid-quantity"),
    ({"requested_usage": {"input_tokens": -1}}, "invalid-quantity"),
    ({"requested_usage": []}, "invalid-quantities"),
    ({"explicit_provider_id": ""}, "invalid-identifier"),
    ({"explicit_model_sha256": ""}, "invalid-digest"),
])
def test_requested_constraints_fail_closed(deployment, contract, profile, kwargs, code):
    assert_error(code, lambda: resolve_profile(profile, contract, deployment, **kwargs))


def test_catalog_edit_does_not_retarget_existing_snapshot(deployment, contract, profile):
    catalog = {profile.profile_id: profile}
    result = resolve_profile(catalog[profile.profile_id], contract, deployment)
    catalog[profile.profile_id] = replace(profile, limits={"input_tokens": 2048})
    assert result.profile_revision == profile.revision
    assert result.profile_revision != catalog[profile.profile_id].revision
    assert result.effective_limits["input_tokens"] == 4096


def test_direct_snapshot_constructor_validates_bindings(deployment, contract, profile):
    result = resolve_profile(profile, contract, deployment)
    assert_error("profile-revision-mismatch", lambda: replace(result, profile_revision=digest("a")))
    assert_error("effective-limits-mismatch", lambda: replace(result, effective_limits={"input_tokens": 99999}))
    assert_error("deployment-revision-mismatch", lambda: replace(result, deployment=replace(deployment, quantization="Q4_K_M")))


@pytest.mark.parametrize("position", [0, 1, 2])
def test_wrong_value_types_rejected(deployment, contract, profile, position):
    args = [profile, contract, deployment]
    args[position] = {}
    assert_error("invalid-type", lambda: resolve_profile(*args))


def test_validation_errors_do_not_echo_supplied_values(deployment, contract, profile):
    private_value = "secret-at-private-host/path"
    error = assert_error("invalid-identifier", lambda: replace(deployment, provider_id=private_value))
    assert private_value not in str(error)
    error = assert_error("invalid-identifier", lambda: replace(contract, limits={private_value: 1}))
    assert private_value not in str(error)


def test_cross_operation_binding_is_not_a_quality_ranking(deployment, contract, profile):
    embedding = replace(contract, capability="text.embed", operation_schema="embedding-v1", semantic_revision=digest("e"))
    wrong = replace(profile, serving_contract_revision=embedding.revision)
    assert_error("capability-mismatch", lambda: resolve_profile(wrong, embedding, deployment))
    correct = replace(wrong, capability=embedding.capability, operation_schema=embedding.operation_schema)
    assert resolve_profile(correct, embedding, deployment).contract.semantic_revision == digest("e")


def test_no_evidence_reference_is_not_accepted(contract):
    assert_error("invalid-digest", lambda: replace(contract, validation_evidence_sha256=None))
