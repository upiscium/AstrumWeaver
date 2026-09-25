from __future__ import annotations

import pytest

from astrumweaver import (
    AcceleratorDevice,
    JobRequirements,
    ResourceShape,
    WorkerSpec,
    match_worker,
    worker_matches,
)
from astrumweaver.control.serde import worker_spec_from_dict, worker_spec_to_dict


def multi_gpu_worker() -> WorkerSpec:
    return WorkerSpec(
        worker_id="worker-multi",
        worker_class="multi-gpu",
        resources=ResourceShape(
            gpu_count=2,
            total_vram_mb=24_576,
            max_single_gpu_vram_mb=12_288,
        ),
        gpu_uuids=("GPU-example-a", "GPU-example-b"),
        capabilities=frozenset({"llm.chat", "code.review"}),
        labels={"runtime_family": "modern"},
    )


def single_large_gpu_worker() -> WorkerSpec:
    return WorkerSpec(
        worker_id="worker-single",
        worker_class="modern-single",
        resources=ResourceShape(
            gpu_count=1,
            total_vram_mb=24_576,
            max_single_gpu_vram_mb=24_576,
        ),
        gpu_uuids=("GPU-example-c",),
        capabilities=frozenset({"llm.chat", "image.generate"}),
        labels={"runtime_family": "modern"},
    )


def test_total_vram_requirement_matches_two_smaller_gpus() -> None:
    requirements = JobRequirements(min_total_vram_mb=20_000)

    assert worker_matches(multi_gpu_worker(), requirements)
    assert worker_matches(single_large_gpu_worker(), requirements)


def test_single_gpu_vram_requirement_does_not_treat_two_12gib_as_one_24gib() -> None:
    requirements = JobRequirements(min_single_gpu_vram_mb=20_000)

    result = match_worker(multi_gpu_worker(), requirements)

    assert not result.matched
    assert result.reasons == ("max_single_gpu_vram_mb",)
    assert worker_matches(single_large_gpu_worker(), requirements)


def test_gpu_count_requirement_can_select_multi_gpu_topology() -> None:
    requirements = JobRequirements(min_gpu_count=2)

    assert worker_matches(multi_gpu_worker(), requirements)
    assert not worker_matches(single_large_gpu_worker(), requirements)


def test_capability_matching_is_generic_and_independent_from_worker_class() -> None:
    worker = WorkerSpec(
        worker_id="worker-arbitrary",
        worker_class="legacy-single",
        resources=ResourceShape(
            gpu_count=1,
            total_vram_mb=11_264,
            max_single_gpu_vram_mb=11_264,
        ),
        gpu_uuids=("GPU-example-d",),
        capabilities=frozenset({"custom.example.capability"}),
    )

    assert worker_matches(
        worker,
        JobRequirements(required_capabilities=frozenset({"custom.example.capability"})),
    )
    assert not worker_matches(
        worker,
        JobRequirements(required_capabilities=frozenset({"some.other.capability"})),
    )


def test_worker_class_and_capability_are_separate_constraints() -> None:
    worker = single_large_gpu_worker()

    assert worker_matches(
        worker,
        JobRequirements(required_capabilities=frozenset({"llm.chat"})),
    )
    assert not worker_matches(
        worker,
        JobRequirements(
            worker_class="multi-gpu",
            required_capabilities=frozenset({"llm.chat"}),
        ),
    )


def test_labels_are_exact_key_value_constraints() -> None:
    worker = multi_gpu_worker()

    assert worker_matches(
        worker,
        JobRequirements(required_labels={"runtime_family": "modern"}),
    )
    result = match_worker(
        worker,
        JobRequirements(required_labels={"runtime_family": "legacy"}),
    )
    assert result.reasons == ("label:runtime_family",)


def test_required_gpu_uuid_is_identity_constraint_not_ordinal() -> None:
    worker = multi_gpu_worker()

    assert worker_matches(
        worker,
        JobRequirements(required_gpu_uuids=frozenset({"GPU-example-b"})),
    )
    assert not worker_matches(
        worker,
        JobRequirements(required_gpu_uuids=frozenset({"GPU-example-missing"})),
    )


def test_gpu_uuid_count_must_match_resource_shape() -> None:
    with pytest.raises(ValueError, match="gpu_uuids count"):
        WorkerSpec(
            worker_id="broken",
            worker_class="multi-gpu",
            resources=ResourceShape(
                gpu_count=2,
                total_vram_mb=24_576,
                max_single_gpu_vram_mb=12_288,
            ),
            gpu_uuids=("GPU-only-one",),
        )


def test_worker_per_device_facts_must_match_gpu_identity_and_memory_shape() -> None:
    with pytest.raises(ValueError, match="UUID order"):
        WorkerSpec(
            worker_id="wrong-order",
            worker_class="multi-gpu",
            resources=ResourceShape(
                gpu_count=2,
                total_vram_mb=24_576,
                max_single_gpu_vram_mb=12_288,
            ),
            gpu_uuids=("GPU-a", "GPU-b"),
            accelerators=(
                AcceleratorDevice("GPU-b", 12_288, "8.6", "RTX 3060"),
                AcceleratorDevice("GPU-a", 12_288, "8.6", "RTX 3060"),
            ),
        )

    with pytest.raises(ValueError, match="memory sum"):
        WorkerSpec(
            worker_id="wrong-memory",
            worker_class="multi-gpu",
            resources=ResourceShape(
                gpu_count=2,
                total_vram_mb=24_576,
                max_single_gpu_vram_mb=12_288,
            ),
            gpu_uuids=("GPU-a", "GPU-b"),
            accelerators=(
                AcceleratorDevice("GPU-a", 12_288, "8.6", "RTX 3060"),
                AcceleratorDevice("GPU-b", 8_192, "8.6", "RTX 3060"),
            ),
        )


def test_worker_per_device_facts_are_optional_for_legacy_workers() -> None:
    worker = multi_gpu_worker()
    assert worker.accelerators == ()


def test_cpu_only_worker_uses_zero_gpu_resource_shape() -> None:
    worker = WorkerSpec(
        worker_id="cpu-only",
        worker_class="cpu",
        capabilities=frozenset({"metadata.normalize"}),
    )

    assert worker_matches(
        worker,
        JobRequirements(required_capabilities=frozenset({"metadata.normalize"})),
    )
    assert not worker_matches(worker, JobRequirements(min_gpu_count=1))


def test_worker_per_device_facts_survive_control_serde_round_trip() -> None:
    worker = WorkerSpec(
        worker_id="durable-facts",
        worker_class="multi-gpu",
        resources=ResourceShape(
            gpu_count=2,
            total_vram_mb=36_864,
            max_single_gpu_vram_mb=24_576,
        ),
        gpu_uuids=("GPU-a", "GPU-b"),
        accelerators=(
            AcceleratorDevice(
                "GPU-a",
                24_576,
                compute_capability="8.6",
                device_class="NVIDIA RTX 3090",
            ),
            AcceleratorDevice(
                "GPU-b",
                12_288,
                compute_capability="8.6",
                device_class="NVIDIA RTX 3060",
            ),
        ),
        capabilities=frozenset({"llm.chat"}),
        labels={"availability": "borrowable"},
    )

    encoded = worker_spec_to_dict(worker)
    restored = worker_spec_from_dict(encoded)

    assert restored == worker
    assert restored.accelerators[0].uuid == "GPU-a"
    assert restored.accelerators[1].memory_mb == 12_288
    assert restored.accelerators[1].device_class == "NVIDIA RTX 3060"
