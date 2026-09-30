from __future__ import annotations

from dataclasses import replace

import pytest

from astrumweaver.contracts import ResourceShape, WorkerSpec
from astrumweaver.worker import daemon


def _spec() -> WorkerSpec:
    return WorkerSpec(
        worker_id="worker-gpu-preflight",
        worker_class="gpu-single",
        resources=ResourceShape(
            gpu_count=1,
            total_vram_mb=12288,
            max_single_gpu_vram_mb=12288,
        ),
        gpu_uuids=("GPU-selected",),
        capabilities=frozenset({"debug.echo"}),
    )


def test_daemon_gpu_preflight_defaults_to_exact_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], str]] = []

    monkeypatch.delenv("ASTRUMWEAVER_GPU_PREFLIGHT_MODE", raising=False)
    monkeypatch.setenv("ASTRUMWEAVER_NVIDIA_SMI", "/test/nvidia-smi")
    monkeypatch.setattr(
        daemon,
        "require_exact_gpu_set",
        lambda expected, *, command: calls.append((expected, command)),
    )
    monkeypatch.setattr(
        daemon,
        "require_isolated_gpu_access",
        lambda *args, **kwargs: pytest.fail("isolated preflight called"),
    )

    daemon._run_gpu_preflight(_spec(), {"gpu_preflight": True})

    assert calls == [(("GPU-selected",), "/test/nvidia-smi")]


def test_daemon_gpu_preflight_uses_explicit_isolated_access_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    monkeypatch.setenv("ASTRUMWEAVER_GPU_PREFLIGHT_MODE", "isolated-access")
    monkeypatch.setenv(
        "ASTRUMWEAVER_GPU_DEVICE_MAP",
        "/etc/astrumweaver/gpu-device-map",
    )
    monkeypatch.setenv("ASTRUMWEAVER_NVIDIA_SMI", "/test/nvidia-smi")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-selected")
    monkeypatch.setattr(
        daemon,
        "require_exact_gpu_set",
        lambda *args, **kwargs: pytest.fail("exact preflight called"),
    )

    def isolated(expected, **kwargs):
        calls.append({"expected": expected, **kwargs})

    monkeypatch.setattr(
        daemon,
        "require_isolated_gpu_access",
        isolated,
    )

    daemon._run_gpu_preflight(_spec(), {"gpu_preflight": True})

    assert calls == [
        {
            "expected": ("GPU-selected",),
            "reviewed_map_path": "/etc/astrumweaver/gpu-device-map",
            "command": "/test/nvidia-smi",
            "cuda_visible_devices": "GPU-selected",
        }
    ]


def test_daemon_isolated_access_requires_reviewed_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASTRUMWEAVER_GPU_PREFLIGHT_MODE", "isolated-access")
    monkeypatch.delenv("ASTRUMWEAVER_GPU_DEVICE_MAP", raising=False)

    with pytest.raises(RuntimeError, match="ASTRUMWEAVER_GPU_DEVICE_MAP"):
        daemon._run_gpu_preflight(_spec(), {"gpu_preflight": True})


def test_daemon_rejects_unknown_gpu_preflight_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASTRUMWEAVER_GPU_PREFLIGHT_MODE", "unknown-mode")

    with pytest.raises(RuntimeError, match="exact-visible or isolated-access"):
        daemon._run_gpu_preflight(_spec(), {"gpu_preflight": True})


def test_daemon_respects_explicit_gpu_preflight_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        daemon,
        "require_exact_gpu_set",
        lambda *args, **kwargs: pytest.fail("preflight called"),
    )
    monkeypatch.setattr(
        daemon,
        "require_isolated_gpu_access",
        lambda *args, **kwargs: pytest.fail("preflight called"),
    )

    daemon._run_gpu_preflight(_spec(), {"gpu_preflight": False})


def test_daemon_skips_gpu_preflight_for_cpu_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu = WorkerSpec(
        worker_id="worker-cpu",
        worker_class="cpu",
        resources=ResourceShape(),
        gpu_uuids=(),
        capabilities=frozenset({"debug.echo"}),
    )
    monkeypatch.setattr(
        daemon,
        "require_exact_gpu_set",
        lambda *args, **kwargs: pytest.fail("preflight called"),
    )

    daemon._run_gpu_preflight(cpu, {"gpu_preflight": True})
