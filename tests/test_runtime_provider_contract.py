from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from astrumweaver import ResourceShape, WorkerSpec
from astrumweaver.runtime import (
    CompatibilityReason,
    ExecutionDemand,
    GPUTopology,
    ModelDemand,
    ModelPreparationPolicy,
    ModelTopology,
    ResidencyPolicy,
    RuntimeCatalog,
    RuntimeCompatibility,
    RuntimeCompatibilityContext,
    RuntimeHostFacts,
    RuntimeProviderInfo,
    RuntimeSelection,
    RuntimeSelectionError,
    RuntimeSelectionMode,
    RuntimeSetupIntent,
    resolve_runtime,
)


class FakeProvider:
    def __init__(
        self,
        provider_id: str,
        *,
        reasons: tuple[CompatibilityReason, ...] = (),
    ) -> None:
        self._info = RuntimeProviderInfo(
            provider_id=provider_id,
            display_name=provider_id.upper(),
        )
        self._reasons = reasons

    @property
    def info(self) -> RuntimeProviderInfo:
        return self._info

    def compatibility(
        self,
        context: RuntimeCompatibilityContext,
    ) -> RuntimeCompatibility:
        return RuntimeCompatibility(
            provider_id=self.info.provider_id,
            reasons=self._reasons,
        )

    def setup_intent(
        self,
        context: RuntimeCompatibilityContext,
    ) -> RuntimeSetupIntent:
        return RuntimeSetupIntent(
            provider_id=self.info.provider_id,
            package_references=(self.info.provider_id,),
            model_preparation=ModelPreparationPolicy.REFERENCE_ONLY,
            model_ref=context.demand.model.model_ref,
        )

    def create_runtime(self, context, setup):
        raise NotImplementedError


def one_gpu_worker() -> WorkerSpec:
    return WorkerSpec(
        worker_id="worker-one",
        worker_class="modern-single",
        resources=ResourceShape(
            gpu_count=1,
            total_vram_mb=24576,
            max_single_gpu_vram_mb=24576,
        ),
        gpu_uuids=("GPU-example-one",),
        capabilities=frozenset({"llm.chat"}),
    )


def two_gpu_worker() -> WorkerSpec:
    return WorkerSpec(
        worker_id="worker-two",
        worker_class="multi-gpu",
        resources=ResourceShape(
            gpu_count=2,
            total_vram_mb=24576,
            max_single_gpu_vram_mb=12288,
        ),
        gpu_uuids=("GPU-example-a", "GPU-example-b"),
        capabilities=frozenset({"llm.chat"}),
    )


def host(*, ram_mb: int = 65536) -> RuntimeHostFacts:
    return RuntimeHostFacts(
        cpu_count=16,
        host_ram_mb=ram_mb,
    )


def dense_single_demand() -> ExecutionDemand:
    return ExecutionDemand(
        model=ModelDemand(
            model_ref="example/model",
            model_format="safetensors",
            topology=ModelTopology.DENSE,
            estimated_size_mb=16000,
        ),
        residency_policy=ResidencyPolicy.VRAM_ONLY,
        gpu_topology=GPUTopology.SINGLE_GPU,
        min_single_gpu_vram_mb=18000,
    )


def test_explicit_runtime_selection_is_preserved_even_with_other_candidates() -> None:
    catalog = RuntimeCatalog(
        [
            FakeProvider("llama-cpp"),
            FakeProvider("vllm"),
            FakeProvider("ollama"),
        ]
    )
    context = RuntimeCompatibilityContext(
        worker=one_gpu_worker(),
        host=host(),
        demand=dense_single_demand(),
    )

    resolution = resolve_runtime(
        catalog=catalog,
        context=context,
        selection=RuntimeSelection(
            mode=RuntimeSelectionMode.EXPLICIT,
            provider_id="llama-cpp",
        ),
    )

    assert resolution.selected_provider_id == "llama-cpp"
    assert resolution.selection.provider_id == "llama-cpp"
    assert resolution.compatible_provider_ids == (
        "llama-cpp",
        "ollama",
        "vllm",
    )


def test_incompatible_explicit_runtime_fails_without_substitution() -> None:
    catalog = RuntimeCatalog(
        [
            FakeProvider(
                "freetoken",
                reasons=(
                    CompatibilityReason(
                        code="model-topology-unsupported",
                        message="FreeToken example provider requires MoE",
                    ),
                ),
            ),
            FakeProvider("llama-cpp"),
        ]
    )
    context = RuntimeCompatibilityContext(
        worker=one_gpu_worker(),
        host=host(),
        demand=dense_single_demand(),
    )

    with pytest.raises(RuntimeSelectionError) as raised:
        resolve_runtime(
            catalog=catalog,
            context=context,
            selection=RuntimeSelection(
                mode=RuntimeSelectionMode.EXPLICIT,
                provider_id="freetoken",
            ),
        )

    assert raised.value.provider_id == "freetoken"
    assert {reason.code for reason in raised.value.reasons} == {
        "model-topology-unsupported"
    }


def test_missing_explicit_runtime_does_not_fall_back() -> None:
    catalog = RuntimeCatalog([FakeProvider("llama-cpp")])
    context = RuntimeCompatibilityContext(
        worker=one_gpu_worker(),
        host=host(),
        demand=dense_single_demand(),
    )

    with pytest.raises(RuntimeSelectionError) as raised:
        resolve_runtime(
            catalog=catalog,
            context=context,
            selection=RuntimeSelection(
                mode=RuntimeSelectionMode.EXPLICIT,
                provider_id="vllm",
            ),
        )

    assert raised.value.provider_id == "vllm"
    assert raised.value.reasons[0].code == "provider-not-found"


def test_recommend_mode_lists_candidates_but_never_selects_one() -> None:
    catalog = RuntimeCatalog(
        [
            FakeProvider("vllm"),
            FakeProvider("llama-cpp"),
            FakeProvider(
                "freetoken",
                reasons=(
                    CompatibilityReason(
                        code="model-topology-unsupported",
                        message="requires MoE",
                    ),
                ),
            ),
        ]
    )
    context = RuntimeCompatibilityContext(
        worker=one_gpu_worker(),
        host=host(),
        demand=dense_single_demand(),
    )

    resolution = resolve_runtime(
        catalog=catalog,
        context=context,
        selection=RuntimeSelection(
            mode=RuntimeSelectionMode.RECOMMEND,
        ),
    )

    assert resolution.selected_provider_id is None
    assert resolution.compatible_provider_ids == ("llama-cpp", "vllm")
    assert not resolution.reports["freetoken"].compatible


def test_generic_planner_rejects_single_gpu_demand_on_multi_gpu_worker() -> None:
    catalog = RuntimeCatalog([FakeProvider("llama-cpp")])
    context = RuntimeCompatibilityContext(
        worker=two_gpu_worker(),
        host=host(),
        demand=dense_single_demand(),
    )

    with pytest.raises(RuntimeSelectionError) as raised:
        resolve_runtime(
            catalog=catalog,
            context=context,
            selection=RuntimeSelection(
                mode=RuntimeSelectionMode.EXPLICIT,
                provider_id="llama-cpp",
            ),
        )

    codes = {reason.code for reason in raised.value.reasons}
    assert "gpu-topology-mismatch" in codes


def test_multi_gpu_total_vram_does_not_replace_single_device_requirement() -> None:
    demand = ExecutionDemand(
        model=ModelDemand(
            model_ref="example/large",
            model_format="gguf",
            topology=ModelTopology.DENSE,
        ),
        residency_policy=ResidencyPolicy.PREFER_VRAM,
        gpu_topology=GPUTopology.MULTI_GPU,
        min_gpu_count=2,
        min_total_vram_mb=20000,
        min_single_gpu_vram_mb=20000,
    )
    context = RuntimeCompatibilityContext(
        worker=two_gpu_worker(),
        host=host(),
        demand=demand,
    )

    with pytest.raises(RuntimeSelectionError) as raised:
        resolve_runtime(
            catalog=RuntimeCatalog([FakeProvider("llama-cpp")]),
            context=context,
            selection=RuntimeSelection(
                mode=RuntimeSelectionMode.EXPLICIT,
                provider_id="llama-cpp",
            ),
        )

    assert {
        reason.code for reason in raised.value.reasons
    } == {"insufficient-single-gpu-vram"}


def test_host_ram_requirement_blocks_cpu_gpu_hybrid_when_too_small() -> None:
    demand = ExecutionDemand(
        model=ModelDemand(
            model_ref="example/offloaded",
            model_format="gguf",
            topology=ModelTopology.DENSE,
        ),
        residency_policy=ResidencyPolicy.CPU_GPU_HYBRID,
        gpu_topology=GPUTopology.SINGLE_GPU,
        min_single_gpu_vram_mb=4096,
        min_host_ram_mb=32768,
        preferred_host_ram_mb=65536,
    )
    context = RuntimeCompatibilityContext(
        worker=one_gpu_worker(),
        host=host(ram_mb=16384),
        demand=demand,
    )

    with pytest.raises(RuntimeSelectionError) as raised:
        resolve_runtime(
            catalog=RuntimeCatalog([FakeProvider("llama-cpp")]),
            context=context,
            selection=RuntimeSelection(
                mode=RuntimeSelectionMode.EXPLICIT,
                provider_id="llama-cpp",
            ),
        )

    assert "insufficient-host-ram" in {
        reason.code for reason in raised.value.reasons
    }


def test_below_preferred_host_ram_is_advisory_not_blocking() -> None:
    demand = ExecutionDemand(
        model=ModelDemand(
            model_ref="example/offloaded",
            model_format="gguf",
            topology=ModelTopology.DENSE,
        ),
        residency_policy=ResidencyPolicy.CPU_GPU_HYBRID,
        gpu_topology=GPUTopology.SINGLE_GPU,
        min_single_gpu_vram_mb=4096,
        min_host_ram_mb=16384,
        preferred_host_ram_mb=65536,
    )
    context = RuntimeCompatibilityContext(
        worker=one_gpu_worker(),
        host=host(ram_mb=32768),
        demand=demand,
    )

    resolution = resolve_runtime(
        catalog=RuntimeCatalog([FakeProvider("llama-cpp")]),
        context=context,
        selection=RuntimeSelection(
            mode=RuntimeSelectionMode.EXPLICIT,
            provider_id="llama-cpp",
        ),
    )

    report = resolution.reports["llama-cpp"]
    assert report.compatible
    assert report.reasons[0].code == "below-preferred-host-ram"
    assert not report.reasons[0].blocking


def test_runtime_contracts_are_immutable() -> None:
    selection = RuntimeSelection(
        mode=RuntimeSelectionMode.EXPLICIT,
        provider_id="vllm",
    )
    demand = dense_single_demand()

    with pytest.raises(FrozenInstanceError):
        selection.provider_id = "llama-cpp"  # type: ignore[misc]

    with pytest.raises(TypeError):
        demand.model.metadata["private"] = "mutation"  # type: ignore[index]


def test_runtime_setup_intent_is_declarative_only() -> None:
    intent = RuntimeSetupIntent(
        provider_id="llama-cpp",
        package_references=("llama.cpp",),
        configuration={"split_mode": "layer"},
        model_preparation=ModelPreparationPolicy.REFERENCE_ONLY,
        model_ref="/models/example.gguf",
        requires_privilege=True,
    )

    assert intent.provider_id == "llama-cpp"
    assert intent.package_references == ("llama.cpp",)
    assert intent.configuration["split_mode"] == "layer"
    with pytest.raises(TypeError):
        intent.configuration["split_mode"] = "tensor"  # type: ignore[index]


def test_runtime_selection_modes_reject_ambiguous_state() -> None:
    with pytest.raises(ValueError, match="requires provider_id"):
        RuntimeSelection(mode=RuntimeSelectionMode.EXPLICIT)

    with pytest.raises(ValueError, match="must not preselect"):
        RuntimeSelection(
            mode=RuntimeSelectionMode.RECOMMEND,
            provider_id="vllm",
        )
