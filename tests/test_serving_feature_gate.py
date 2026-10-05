from __future__ import annotations

import pytest

from astrumweaver import JobResult, ResidencyReport
from astrumweaver.execution import JobRequest
from astrumweaver.serving import (
    DeploymentIdentity,
    ServingContract,
    ServingDeploymentDeclaration,
)
from astrumweaver.worker import require_executor_serving_features


def digest(character: str) -> str:
    return "sha256:" + character * 64


def declaration(*, features=frozenset({"tools"})) -> ServingDeploymentDeclaration:
    deployment = DeploymentIdentity(
        provider_id="llama-cpp",
        runtime_artifact_sha256=digest("1"),
        adapter_artifact_sha256=digest("2"),
        model_artifact_sha256=digest("3"),
        execution_config_sha256=digest("4"),
        quantization="Q6_K",
        template_artifact_sha256=digest("5"),
    )
    contract = ServingContract(
        deployment_revision=deployment.revision,
        capability="llm.chat",
        operation_schema="openai-chat-completions-v1",
        validation_evidence_sha256=digest("7"),
        features=features,
        limits={
            "input_tokens": 4096,
            "output_tokens": 1024,
            "total_tokens": 4096,
            "request_bytes": 65536,
        },
    )
    return ServingDeploymentDeclaration(
        deployment=deployment,
        contracts=(contract,),
    )


class Executor:
    capabilities = frozenset({"llm.chat"})

    def __init__(self, *, tools: bool) -> None:
        self.serving_features = {
            "llm.chat": frozenset({"tools"} if tools else set())
        }

    async def execute(self, _job: JobRequest) -> JobResult:
        return JobResult()

    async def cancel(self, _job_id: str) -> None:
        return None

    async def residency(self) -> ResidencyReport:
        return ResidencyReport()


def test_serving_feature_gate_accepts_reviewed_executor_feature():
    require_executor_serving_features(
        Executor(tools=True),
        declaration(),
    )


def test_serving_feature_gate_rejects_unproven_executor_feature():
    with pytest.raises(RuntimeError, match="tools"):
        require_executor_serving_features(
            Executor(tools=False),
            declaration(),
        )


def test_serving_feature_gate_keeps_featureless_legacy_contract_compatible():
    require_executor_serving_features(
        Executor(tools=False),
        declaration(features=frozenset()),
    )
