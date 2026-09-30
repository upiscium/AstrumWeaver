from __future__ import annotations

from pathlib import Path

import pytest

from astrumweaver import AcceleratorDevice, ResourceShape, WorkerSpec
from astrumweaver.worker import daemon


def gpu_spec() -> WorkerSpec:
    return WorkerSpec(
        worker_id="worker-test",
        worker_class="gpu-single",
        resources=ResourceShape(
            gpu_count=1,
            total_vram_mb=12288,
            max_single_gpu_vram_mb=12288,
        ),
        gpu_uuids=("GPU-selected",),
        accelerators=(
            AcceleratorDevice(
                uuid="GPU-selected",
                memory_mb=12288,
            ),
        ),
        capabilities=frozenset({"debug.echo"}),
    )


def test_exact_visible_worker_preflight_uses_raw_uuid_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_calls: list[tuple[tuple[str, ...], str]] = []
    isolated_calls: list[object] = []

    monkeypatch.setattr(
        daemon,
        "require_exact_gpu_set",
        lambda expected, *, command: exact_calls.append((expected, command)),
    )
    monkeypatch.setattr(
        daemon,
        "require_isolated_gpu_access",
        lambda *args, **kwargs: isolated_calls.append((args, kwargs)),
    )

    daemon._require_worker_gpu_preflight(
        {
            "gpu_preflight": True,
            "gpu_preflight_mode": "exact-visible",
            "nvidia_smi_command": "/usr/bin/nvidia-smi",
        },
        gpu_spec(),
    )

    assert exact_calls == [
        (("GPU-selected",), "/usr/bin/nvidia-smi")
    ]
    assert isolated_calls == []


def test_isolated_access_worker_preflight_repeats_effective_access_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_calls: list[object] = []
    isolated_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-selected")
    monkeypatch.setattr(
        daemon,
        "require_exact_gpu_set",
        lambda *args, **kwargs: exact_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        daemon,
        "require_isolated_gpu_access",
        lambda *args, **kwargs: isolated_calls.append((args, kwargs)),
    )

    daemon._require_worker_gpu_preflight(
        {
            "gpu_preflight": True,
            "gpu_preflight_mode": "isolated-access",
            "gpu_device_map": "/etc/astrumweaver/gpu-device-map",
            "nvidia_smi_command": "/usr/bin/nvidia-smi",
        },
        gpu_spec(),
    )

    assert exact_calls == []
    assert len(isolated_calls) == 1
    args, kwargs = isolated_calls[0]
    assert args == (
        ("GPU-selected",),
        Path("/etc/astrumweaver/gpu-device-map"),
    )
    assert kwargs == {
        "worker_gpu_order": ("GPU-selected",),
        "cuda_visible_devices": "GPU-selected",
        "nvidia_smi": "/usr/bin/nvidia-smi",
    }


@pytest.mark.parametrize(
    ("worker_section", "message"),
    (
        (
            {
                "gpu_preflight_mode": "isolated-access",
                "gpu_preflight": False,
                "gpu_device_map": "/etc/astrumweaver/gpu-device-map",
            },
            "cannot be disabled",
        ),
        (
            {
                "gpu_preflight_mode": "isolated-access",
                "gpu_device_map": "relative-map",
            },
            "absolute gpu_device_map",
        ),
        (
            {
                "gpu_preflight_mode": "exact-visible",
                "gpu_device_map": "/etc/astrumweaver/gpu-device-map",
            },
            "valid only with isolated-access",
        ),
        (
            {
                "gpu_preflight_mode": "invalid",
            },
            "must be exact-visible or isolated-access",
        ),
    ),
)
def test_worker_gpu_preflight_rejects_invalid_mode_contract(
    worker_section: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        daemon._require_worker_gpu_preflight(worker_section, gpu_spec())
