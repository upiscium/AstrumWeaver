from __future__ import annotations

import json

import pytest

from astrumweaver.control.daemon import build_app
from astrumweaver.gateway.chat import (
    CHAT_CATALOG_SCHEMA,
    CHAT_OPERATION_SCHEMA,
    LLAMA_CPP_CHAT_ADAPTER,
)
from astrumweaver.serving import (
    DeploymentIdentity,
    LogicalServingProfile,
    ServingContract,
)


def digest(character: str) -> str:
    return "sha256:" + character * 64


def catalog_document() -> dict:
    deployment = DeploymentIdentity(
        provider_id="llama-cpp",
        runtime_artifact_sha256=digest("1"),
        adapter_artifact_sha256=digest("2"),
        model_artifact_sha256=digest("3"),
        execution_config_sha256=digest("4"),
        quantization="Q6_K",
        tokenizer_artifact_sha256=digest("5"),
        template_artifact_sha256=digest("6"),
    )
    contract = ServingContract(
        deployment_revision=deployment.revision,
        capability="llm.chat",
        operation_schema=CHAT_OPERATION_SCHEMA,
        validation_evidence_sha256=digest("7"),
        features=frozenset({"tools"}),
        limits={
            "input_tokens": 4096,
            "output_tokens": 1024,
            "total_tokens": 5120,
            "request_bytes": 65536,
        },
    )
    profile = LogicalServingProfile(
        profile_id="local-code-v1",
        deployment_revision=deployment.revision,
        serving_contract_revision=contract.revision,
        capability="llm.chat",
        operation_schema=CHAT_OPERATION_SCHEMA,
        required_features=frozenset({"tools"}),
        limits={},
    )
    return {
        "schema_version": CHAT_CATALOG_SCHEMA,
        "profiles": [
            {
                "adapter_id": LLAMA_CPP_CHAT_ADAPTER,
                "request_timeout_seconds": 30,
                "max_attempts": 2,
                "deployment": deployment.to_dict(),
                "contract": contract.to_dict(),
                "profile": profile.to_dict(),
            }
        ],
    }


def route_paths(app) -> set[str]:
    result: set[str] = set()
    pending = list(app.routes)
    while pending:
        route = pending.pop()
        path = getattr(route, "path", None)
        if isinstance(path, str):
            result.add(path)
        nested = getattr(route, "routes", ())
        pending.extend(nested)
    return result


def base_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "ASTRUMWEAVER_DATABASE_URL",
        "postgresql://example.invalid/astrumweaver",
    )
    monkeypatch.setenv("ASTRUMWEAVER_WORKER_TOKEN", "worker-secret")
    monkeypatch.delenv("ASTRUMWEAVER_CLIENT_TOKEN", raising=False)


def test_control_daemon_mounts_chat_gateway_only_when_enabled(tmp_path, monkeypatch):
    base_environment(monkeypatch)
    catalog = tmp_path / "chat-catalog.json"
    catalog.write_text(json.dumps(catalog_document()), encoding="utf-8")
    config = tmp_path / "control.toml"
    config.write_text(
        "\n".join(
            [
                "[control]",
                'client_auth = "none"',
                "",
                "[chat_gateway]",
                "enabled = true",
                f'catalog = "{catalog}"',
                "poll_interval_seconds = 0.01",
                "",
            ]
        ),
        encoding="utf-8",
    )

    app = build_app(str(config))
    paths = route_paths(app)
    assert "/v1/models" in paths
    assert "/v1/chat/completions" in paths

    disabled = tmp_path / "disabled.toml"
    disabled.write_text(
        '[control]\nclient_auth = "none"\n',
        encoding="utf-8",
    )
    app = build_app(str(disabled))
    paths = {route.path for route in app.routes}
    assert "/v1/models" not in paths
    assert "/v1/chat/completions" not in paths


def test_control_daemon_requires_valid_catalog_when_chat_gateway_enabled(
    tmp_path,
    monkeypatch,
):
    base_environment(monkeypatch)
    missing = tmp_path / "missing.toml"
    missing.write_text(
        '[control]\nclient_auth = "none"\n'
        "[chat_gateway]\nenabled = true\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="chat_gateway.catalog"):
        build_app(str(missing))

    invalid_catalog = tmp_path / "invalid.json"
    invalid_catalog.write_text("{}", encoding="utf-8")
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        '[control]\nclient_auth = "none"\n'
        "[chat_gateway]\nenabled = true\n"
        f'catalog = "{invalid_catalog}"\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="profile catalog is invalid"):
        build_app(str(invalid))
