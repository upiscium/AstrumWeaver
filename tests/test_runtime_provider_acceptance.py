from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tomllib

import pytest

from astrumweaver import AcceleratorDevice, ResourceShape, WorkerSpec
from astrumweaver.runtime import (
    ExecutionDemand,
    GPUTopology,
    ModelDemand,
    ModelTopology,
    ResidencyPolicy,
    RuntimeCatalog,
    RuntimeCompatibilityContext,
    RuntimeHostFacts,
    RuntimeSelection,
    RuntimeSelectionError,
    RuntimeSelectionMode,
    resolve_runtime,
)
from astrumweaver.runtime.providers import (
    ExLlamaV3Provider,
    FreeTokenProvider,
    LlamaCppProvider,
    OllamaProvider,
    VllmProvider,
    VllmProviderConfig,
)
from astrumweaver.setup import (
    DeploymentPath,
    PrivilegeMode,
    SetupHostSnapshot,
    build_runtime_setup_plan,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "acceptance/runtime-providers-v0.1.toml"
SUPPORT_STATUSES = {"validated", "supported_unvalidated", "unsupported"}
INTEGRATION_STATUSES = {"integrated", "not_integrated"}
MULTI_GPU_HARDWARE_STATUSES = {
    "heterogeneous_validated",
    "homogeneous_required",
    "supported_unvalidated",
    "unsupported",
}
PROVIDER_IDS = {"ollama", "llama-cpp", "vllm", "freetoken", "exllamav3"}


@dataclass(frozen=True)
class ProviderCase:
    provider_id: str
    provider: object
    model_ref: str
    model_format: str
    model_topology: ModelTopology
    residency: ResidencyPolicy


def load_matrix() -> dict:
    with MATRIX.open("rb") as handle:
        return tomllib.load(handle)


def provider_cases() -> dict[str, ProviderCase]:
    return {
        "ollama": ProviderCase(
            "ollama",
            OllamaProvider(),
            "qwen3:8b",
            "ollama",
            ModelTopology.DENSE,
            ResidencyPolicy.PREFER_VRAM,
        ),
        "llama-cpp": ProviderCase(
            "llama-cpp",
            LlamaCppProvider(),
            "/models/example.gguf",
            "gguf",
            ModelTopology.DENSE,
            ResidencyPolicy.PREFER_VRAM,
        ),
        "vllm": ProviderCase(
            "vllm",
            VllmProvider(),
            "example/model",
            "huggingface",
            ModelTopology.DENSE,
            ResidencyPolicy.PREFER_VRAM,
        ),
        "freetoken": ProviderCase(
            "freetoken",
            FreeTokenProvider(),
            "example/moe-model",
            "huggingface",
            ModelTopology.MOE,
            ResidencyPolicy.CPU_GPU_HYBRID,
        ),
        "exllamav3": ProviderCase(
            "exllamav3",
            ExLlamaV3Provider(),
            "example/model-exl3",
            "exl3",
            ModelTopology.DENSE,
            ResidencyPolicy.PREFER_VRAM,
        ),
    }


def host() -> RuntimeHostFacts:
    return RuntimeHostFacts(
        cpu_count=32,
        host_ram_mb=131072,
        architecture="x86_64",
    )


def single_worker() -> WorkerSpec:
    return WorkerSpec(
        worker_id="runtime-acceptance-single",
        worker_class="runtime-single",
        resources=ResourceShape(
            gpu_count=1,
            total_vram_mb=24576,
            max_single_gpu_vram_mb=24576,
        ),
        gpu_uuids=("GPU-example-one",),
        accelerators=(
            AcceleratorDevice(
                "GPU-example-one",
                24576,
                compute_capability="8.6",
                device_class="NVIDIA RTX 3090",
            ),
        ),
        capabilities=frozenset({"llm.chat", "text.generate"}),
    )


def multi_worker(*, heterogeneous: bool = False, facts: bool = True) -> WorkerSpec:
    if heterogeneous:
        devices = (
            AcceleratorDevice(
                "GPU-example-large",
                24576,
                compute_capability="8.6",
                device_class="NVIDIA RTX 3090",
            ),
            AcceleratorDevice(
                "GPU-example-small",
                12288,
                compute_capability="6.1",
                device_class="NVIDIA GTX 1080 Ti",
            ),
        )
    else:
        devices = (
            AcceleratorDevice(
                "GPU-example-a",
                12288,
                compute_capability="8.6",
                device_class="NVIDIA RTX 3060",
            ),
            AcceleratorDevice(
                "GPU-example-b",
                12288,
                compute_capability="8.6",
                device_class="NVIDIA RTX 3060",
            ),
        )

    resources = ResourceShape(
        gpu_count=2,
        total_vram_mb=sum(device.memory_mb for device in devices),
        max_single_gpu_vram_mb=max(device.memory_mb for device in devices),
    )
    return WorkerSpec(
        worker_id="runtime-acceptance-multi",
        worker_class="runtime-multi",
        resources=resources,
        gpu_uuids=tuple(device.uuid for device in devices),
        accelerators=devices if facts else (),
        capabilities=frozenset({"llm.chat", "text.generate"}),
    )


def demand_for(
    provider_id: str,
    *,
    model_format: str | None = None,
    model_topology: ModelTopology | None = None,
    residency: ResidencyPolicy | None = None,
    gpu_topology: GPUTopology = GPUTopology.SINGLE_GPU,
) -> ExecutionDemand:
    case = provider_cases()[provider_id]
    selected_format = model_format or case.model_format
    model_ref = case.model_ref
    if provider_id == "freetoken" and selected_format == "ftw":
        model_ref = "/models/example-ftw"

    return ExecutionDemand(
        model=ModelDemand(
            model_ref=model_ref,
            model_format=selected_format,
            topology=model_topology or case.model_topology,
            estimated_size_mb=(
                60000 if provider_id == "freetoken" else 8000
            ),
        ),
        residency_policy=residency or case.residency,
        gpu_topology=gpu_topology,
        min_gpu_count=2 if gpu_topology is GPUTopology.MULTI_GPU else 0,
    )


def context_for(
    provider_id: str,
    *,
    worker: WorkerSpec | None = None,
    model_format: str | None = None,
    model_topology: ModelTopology | None = None,
    residency: ResidencyPolicy | None = None,
    gpu_topology: GPUTopology = GPUTopology.SINGLE_GPU,
) -> RuntimeCompatibilityContext:
    return RuntimeCompatibilityContext(
        worker=worker or single_worker(),
        host=host(),
        demand=demand_for(
            provider_id,
            model_format=model_format,
            model_topology=model_topology,
            residency=residency,
            gpu_topology=gpu_topology,
        ),
    )


def provider_for_residency(provider_id: str, residency: ResidencyPolicy):
    if (
        provider_id == "vllm"
        and residency is ResidencyPolicy.CPU_GPU_HYBRID
    ):
        return VllmProvider(VllmProviderConfig(cpu_offload_gb=4.0))
    return provider_cases()[provider_id].provider


def snapshot(ctx: RuntimeCompatibilityContext) -> SetupHostSnapshot:
    return SetupHostSnapshot(
        runtime_host=ctx.host,
        deployment_path=DeploymentPath.SYSTEMD,
        os_id="debian",
        service_manager="systemd",
        package_manager="nix",
        available_commands=frozenset({"nix", "systemctl", "nvidia-smi"}),
        privilege_mode=PrivilegeMode.SUDO,
    )


def matrix_rows() -> dict[str, dict]:
    return {
        row["provider_id"]: row
        for row in load_matrix()["provider"]
    }


def test_runtime_provider_matrix_has_exact_first_class_provider_set() -> None:
    matrix = load_matrix()
    assert matrix["matrix_version"] == "v0.1"
    rows = matrix_rows()
    assert set(rows) == PROVIDER_IDS
    assert set(provider_cases()) == PROVIDER_IDS

    for provider_id, case in provider_cases().items():
        assert case.provider.info.provider_id == provider_id
        assert rows[provider_id]["capabilities"] == ["llm.chat", "text.generate"]


def test_runtime_provider_matrix_uses_only_reviewed_status_vocabulary() -> None:
    rows = matrix_rows()
    support_fields = (
        "single_gpu",
        "multi_gpu",
        "dense",
        "moe",
        "vram_only",
        "prefer_vram",
        "cpu_gpu_hybrid",
        "process_lifecycle",
        "cancellation",
        "residency_reporting",
    )

    for row in rows.values():
        for field in support_fields:
            assert row[field] in SUPPORT_STATUSES
        assert row["nixos_integration"] in INTEGRATION_STATUSES
        assert row["systemd_integration"] in INTEGRATION_STATUSES
        assert row["multi_gpu_hardware"] in MULTI_GPU_HARDWARE_STATUSES
        assert row["evidence"]


def test_matrix_evidence_references_existing_tests_or_flake_checks() -> None:
    for row in matrix_rows().values():
        for reference in row["evidence"]:
            if reference.startswith("flake.nix#"):
                path_text, _, check = reference.partition("#")
                source = (ROOT / path_text).read_text(encoding="utf-8")
                assert check.rsplit(".", 1)[-1] in source
                continue

            path_text, separator, test_name = reference.partition("::")
            assert separator == "::", reference
            source_path = ROOT / path_text
            assert source_path.is_file(), reference
            source = source_path.read_text(encoding="utf-8")
            assert re.search(
                rf"^(?:async\s+)?def\s+{re.escape(test_name)}\(",
                source,
                flags=re.MULTILINE,
            ), reference


@pytest.mark.parametrize("provider_id", sorted(PROVIDER_IDS))
def test_each_provider_has_one_compatible_contract_and_reviewable_setup_plan(
    provider_id: str,
) -> None:
    case = provider_cases()[provider_id]
    ctx = context_for(provider_id)

    report = case.provider.compatibility(ctx)
    assert report.provider_id == provider_id
    assert report.compatible

    selection = RuntimeSelection(
        mode=RuntimeSelectionMode.EXPLICIT,
        provider_id=provider_id,
    )
    resolution = resolve_runtime(
        catalog=RuntimeCatalog([case.provider]),
        context=ctx,
        selection=selection,
    )
    assert resolution.selected_provider_id == provider_id
    assert resolution.selection.provider_id == provider_id

    intent = case.provider.setup_intent(ctx)
    assert intent.provider_id == provider_id

    plan = build_runtime_setup_plan(
        catalog=RuntimeCatalog([case.provider]),
        context=ctx,
        selection=selection,
        snapshot=snapshot(ctx),
    )
    assert plan.provider_id == provider_id
    assert plan.actions


@pytest.mark.parametrize("provider_id", sorted(PROVIDER_IDS))
def test_each_provider_rejects_an_unlisted_model_format_without_substitution(
    provider_id: str,
) -> None:
    case = provider_cases()[provider_id]
    ctx = context_for(provider_id, model_format="unsupported-format")

    report = case.provider.compatibility(ctx)
    assert not report.compatible
    assert "model-format-unsupported" in {
        reason.code for reason in report.reasons
    }

    with pytest.raises(RuntimeSelectionError) as exc_info:
        resolve_runtime(
            catalog=RuntimeCatalog([case.provider]),
            context=ctx,
            selection=RuntimeSelection(
                mode=RuntimeSelectionMode.EXPLICIT,
                provider_id=provider_id,
            ),
        )
    assert exc_info.value.provider_id == provider_id


@pytest.mark.parametrize("provider_id", sorted(PROVIDER_IDS))
def test_every_listed_model_format_is_accepted_by_provider_scope(
    provider_id: str,
) -> None:
    row = matrix_rows()[provider_id]
    provider = provider_cases()[provider_id].provider

    for model_format in row["model_formats"]:
        report = provider.compatibility(
            context_for(provider_id, model_format=model_format)
        )
        assert report.compatible, (
            provider_id,
            model_format,
            tuple(reason.code for reason in report.reasons),
        )


@pytest.mark.parametrize("provider_id", sorted(PROVIDER_IDS))
def test_single_and_multi_gpu_claims_match_actual_compatibility(
    provider_id: str,
) -> None:
    row = matrix_rows()[provider_id]
    provider = provider_cases()[provider_id].provider

    single = provider.compatibility(context_for(provider_id))
    if row["single_gpu"] == "validated":
        assert single.compatible
    elif row["single_gpu"] == "unsupported":
        assert not single.compatible

    multi_ctx = context_for(
        provider_id,
        worker=(
            multi_worker(heterogeneous=True)
            if row["multi_gpu_hardware"] == "heterogeneous_validated"
            else multi_worker()
        ),
        gpu_topology=GPUTopology.MULTI_GPU,
    )
    multi = provider.compatibility(multi_ctx)
    if row["multi_gpu"] == "validated":
        assert multi.compatible, tuple(reason.code for reason in multi.reasons)
    elif row["multi_gpu"] == "unsupported":
        assert not multi.compatible


def test_vllm_homogeneous_claim_requires_auditable_per_device_facts() -> None:
    provider = VllmProvider()
    missing_facts = provider.compatibility(
        context_for(
            "vllm",
            worker=multi_worker(facts=False),
            gpu_topology=GPUTopology.MULTI_GPU,
        )
    )
    heterogeneous = provider.compatibility(
        context_for(
            "vllm",
            worker=multi_worker(heterogeneous=True),
            gpu_topology=GPUTopology.MULTI_GPU,
        )
    )

    assert not missing_facts.compatible
    assert "accelerator-facts-required" in {
        reason.code for reason in missing_facts.reasons
    }
    assert not heterogeneous.compatible
    hetero_codes = {reason.code for reason in heterogeneous.reasons}
    assert "heterogeneous-vram-unsupported" in hetero_codes
    assert "heterogeneous-compute-capability-unsupported" in hetero_codes
    assert "heterogeneous-device-class-unsupported" in hetero_codes


def test_only_llama_cpp_claims_validated_heterogeneous_multi_gpu() -> None:
    rows = matrix_rows()
    validated = {
        provider_id
        for provider_id, row in rows.items()
        if row["multi_gpu_hardware"] == "heterogeneous_validated"
    }
    assert validated == {"llama-cpp"}

    report = LlamaCppProvider().compatibility(
        context_for(
            "llama-cpp",
            worker=multi_worker(heterogeneous=True),
            gpu_topology=GPUTopology.MULTI_GPU,
        )
    )
    assert report.compatible


@pytest.mark.parametrize(
    ("field", "topology"),
    (
        ("dense", ModelTopology.DENSE),
        ("moe", ModelTopology.MOE),
    ),
)
def test_model_topology_claims_match_provider_scope(
    field: str,
    topology: ModelTopology,
) -> None:
    for provider_id, row in matrix_rows().items():
        status = row[field]
        if status == "supported_unvalidated":
            continue

        provider = provider_cases()[provider_id].provider
        report = provider.compatibility(
            context_for(provider_id, model_topology=topology)
        )
        if status == "validated":
            assert report.compatible, (
                provider_id,
                field,
                tuple(reason.code for reason in report.reasons),
            )
        elif status == "unsupported":
            assert not report.compatible


@pytest.mark.parametrize(
    ("field", "residency"),
    (
        ("vram_only", ResidencyPolicy.VRAM_ONLY),
        ("prefer_vram", ResidencyPolicy.PREFER_VRAM),
        ("cpu_gpu_hybrid", ResidencyPolicy.CPU_GPU_HYBRID),
    ),
)
def test_residency_claims_match_provider_scope(
    field: str,
    residency: ResidencyPolicy,
) -> None:
    for provider_id, row in matrix_rows().items():
        status = row[field]
        if status == "supported_unvalidated":
            continue

        provider = provider_for_residency(provider_id, residency)
        report = provider.compatibility(
            context_for(provider_id, residency=residency)
        )
        if status == "validated":
            assert report.compatible, (
                provider_id,
                field,
                tuple(reason.code for reason in report.reasons),
            )
        elif status == "unsupported":
            assert not report.compatible


def test_control_plane_contains_no_first_class_provider_branches() -> None:
    provider_tokens = (
        "ollama",
        "llama-cpp",
        "llama_cpp",
        "vllm",
        "freetoken",
        "exllamav3",
        "runtime.providers",
    )
    for path in (ROOT / "src/astrumweaver/control").rglob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        for token in provider_tokens:
            assert token not in source, (path, token)
