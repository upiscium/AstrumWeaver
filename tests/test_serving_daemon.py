from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from uuid import UUID

import pytest

from astrumweaver import ResourceShape, WorkerSpec
from astrumweaver.executors.structured_echo import StructuredEchoExecutor
from astrumweaver.serving import (
    DeploymentIdentity,
    ServingContract,
    ServingDeploymentDeclaration,
)
from astrumweaver.worker import daemon


def digest(character: str) -> str:
    return "sha256:" + character * 64


def declaration(*, provider_id: str = "custom") -> ServingDeploymentDeclaration:
    deployment = DeploymentIdentity(
        provider_id=provider_id,
        runtime_artifact_sha256=digest("1"),
        adapter_artifact_sha256=digest("2"),
        model_artifact_sha256=digest("3"),
        execution_config_sha256=digest("4"),
        quantization="Q6_K",
    )
    contract = ServingContract(
        deployment_revision=deployment.revision,
        capability="debug.echo",
        operation_schema="echo-v1",
        validation_evidence_sha256=digest("7"),
        features=frozenset(),
        limits={"input_bytes": 4096},
    )
    return ServingDeploymentDeclaration(
        deployment=deployment,
        contracts=(contract,),
    )


def worker_spec(*, capabilities=frozenset({"debug.echo"})) -> WorkerSpec:
    return WorkerSpec(
        worker_id="serving-worker",
        worker_class="cpu-test",
        resources=ResourceShape(),
        capabilities=capabilities,
    )


def test_serving_deployment_manifest_round_trip_and_fresh_epoch(tmp_path, monkeypatch):
    value = declaration()
    manifest = tmp_path / "serving.json"
    manifest.write_text(json.dumps(value.to_dict()), encoding="utf-8")

    loaded = daemon._load_serving_deployment(str(manifest))
    assert loaded == value

    epochs = iter(
        (
            UUID("12345678-1234-4234-9234-123456789abc"),
            UUID("22345678-1234-4234-9234-123456789abc"),
        )
    )
    monkeypatch.setattr(daemon, "uuid4", lambda: next(epochs))

    first = daemon._build_serving_advertisement(
        loaded,
        spec=worker_spec(),
        runtime_deployment=None,
    )
    second = daemon._build_serving_advertisement(
        loaded,
        spec=worker_spec(),
        runtime_deployment=None,
    )

    assert first.deployment_revision == value.deployment.revision
    assert first.contracts == value.contracts
    assert first.runtime_instance.epoch == "12345678-1234-4234-9234-123456789abc"
    assert second.runtime_instance.epoch == "22345678-1234-4234-9234-123456789abc"


def test_serving_deployment_must_match_worker_capability_and_runtime_provider():
    value = declaration(provider_id="llama-cpp")

    with pytest.raises(RuntimeError, match="capability"):
        daemon._build_serving_advertisement(
            value,
            spec=worker_spec(capabilities=frozenset({"other.capability"})),
            runtime_deployment=None,
        )

    with pytest.raises(RuntimeError, match="provider"):
        daemon._build_serving_advertisement(
            value,
            spec=worker_spec(),
            runtime_deployment=SimpleNamespace(provider_id="ollama"),
        )


@pytest.mark.asyncio
async def test_worker_daemon_wires_reviewed_serving_deployment(monkeypatch, tmp_path):
    serving = declaration()
    manifest = tmp_path / "serving.json"
    manifest.write_text(json.dumps(serving.to_dict()), encoding="utf-8")

    config = {
        "worker": {
            "id": "serving-worker",
            "class": "cpu-test",
            "control_url": "http://control",
            "capabilities": ["debug.echo"],
            "poll_interval_seconds": 0.01,
            "heartbeat_interval_seconds": 0.01,
        },
        "executor": {"factory": "test:factory"},
        "serving": {"manifest": str(manifest)},
    }

    captured = []
    closed = asyncio.Event()

    class FakeClient:
        async def aclose(self):
            closed.set()

    class FakeRuntime:
        registered = False
        active_job_id = None

        def __init__(self, **kwargs):
            captured.append(kwargs["serving"])

        async def run_forever(self):
            await asyncio.sleep(0)

        def request_stop(self):
            return None

        def request_drain(self):
            return None

        async def drain(self):
            return None

    class FakeHealthServer:
        def __init__(self, _config):
            self.should_exit = False

        async def serve(self):
            while not self.should_exit:
                await asyncio.sleep(0)

    monkeypatch.setenv("ASTRUMWEAVER_WORKER_TOKEN", "worker-token")
    monkeypatch.setattr(daemon, "_load_toml", lambda _: config)
    monkeypatch.setattr(
        daemon,
        "ControlClient",
        lambda *_args, **_kwargs: FakeClient(),
    )
    monkeypatch.setattr(
        daemon,
        "load_executor",
        lambda *_args, **_kwargs: StructuredEchoExecutor(),
    )
    monkeypatch.setattr(daemon, "WorkerRuntime", FakeRuntime)
    monkeypatch.setattr(daemon, "create_health_app", lambda _: object())
    monkeypatch.setattr(daemon.uvicorn, "Config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(daemon.uvicorn, "Server", FakeHealthServer)
    monkeypatch.setattr(
        asyncio.get_running_loop(),
        "add_signal_handler",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        daemon,
        "uuid4",
        lambda: UUID("32345678-1234-4234-9234-123456789abc"),
    )

    await asyncio.wait_for(daemon.run_worker("unused"), timeout=1)

    assert closed.is_set()
    assert len(captured) == 1
    advertisement = captured[0]
    assert advertisement is not None
    assert advertisement.deployment_revision == serving.deployment.revision
    assert advertisement.runtime_instance.epoch == (
        "32345678-1234-4234-9234-123456789abc"
    )
