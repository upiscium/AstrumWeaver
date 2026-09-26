from __future__ import annotations

from astrumweaver import ResourceShape, WorkerSpec
from astrumweaver.runtime import (
    ExecutionDemand,
    GPUTopology,
    ModelDemand,
    ModelTopology,
    ResidencyPolicy,
    RuntimeCompatibilityContext,
    RuntimeDeploymentSpec,
    RuntimeHostFacts,
    build_runtime_deployment_spec,
    managed_runtime_from_deployment,
    provider_from_deployment,
)
from astrumweaver.runtime.providers import (
    OllamaProvider,
    OllamaProviderConfig,
)


def context() -> RuntimeCompatibilityContext:
    return RuntimeCompatibilityContext(
        worker=WorkerSpec(
            worker_id="runtime-worker",
            worker_class="gpu-single",
            resources=ResourceShape(
                gpu_count=1,
                total_vram_mb=24576,
                max_single_gpu_vram_mb=24576,
            ),
            gpu_uuids=("GPU-runtime",),
            capabilities=frozenset({"llm.chat", "text.generate"}),
        ),
        host=RuntimeHostFacts(
            cpu_count=16,
            host_ram_mb=65536,
            architecture="x86_64",
        ),
        demand=ExecutionDemand(
            model=ModelDemand(
                model_ref="qwen3:8b",
                model_format="ollama",
                topology=ModelTopology.DENSE,
            ),
            residency_policy=ResidencyPolicy.PREFER_VRAM,
            gpu_topology=GPUTopology.SINGLE_GPU,
        ),
    )


def test_runtime_deployment_round_trip_preserves_provider_and_demand() -> None:
    provider = OllamaProvider(
        OllamaProviderConfig(
            keep_alive="10m",
            startup_timeout_seconds=45.0,
        )
    )
    deployment = build_runtime_deployment_spec(provider, context())

    restored = RuntimeDeploymentSpec.from_dict(deployment.to_dict())

    assert restored.provider_id == "ollama"
    assert restored.provider_config["keep_alive"] == "10m"
    assert restored.demand == deployment.demand
    assert restored.setup_intent is not None
    assert restored.setup_intent.provider_id == "ollama"
    assert restored.setup_intent.configuration["gpu_uuids"] == [
        "GPU-runtime"
    ]


def test_declarative_manifest_can_regenerate_provider_setup_intent() -> None:
    generated = build_runtime_deployment_spec(
        OllamaProvider(),
        context(),
    )
    declarative = RuntimeDeploymentSpec(
        provider_id=generated.provider_id,
        provider_config=generated.provider_config,
        demand=generated.demand,
        setup_intent=None,
    )

    provider = provider_from_deployment(declarative)
    managed = managed_runtime_from_deployment(
        declarative,
        worker=context().worker,
        host=context().host,
    )

    assert provider.info.provider_id == "ollama"
    assert managed.provider_id == "ollama"
    assert managed.executor().capabilities == frozenset(
        {"llm.chat", "text.generate"}
    )


def test_runtime_deployment_rejects_unknown_provider() -> None:
    generated = build_runtime_deployment_spec(
        OllamaProvider(),
        context(),
    )
    invalid = RuntimeDeploymentSpec(
        provider_id="unknown-runtime",
        provider_config={},
        demand=generated.demand,
        setup_intent=None,
    )

    try:
        provider_from_deployment(invalid)
    except ValueError as exc:
        assert "unsupported runtime provider" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown runtime provider was accepted")
