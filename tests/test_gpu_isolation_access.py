from __future__ import annotations

import errno
from pathlib import Path

import pytest

from astrumweaver.validation.gpu_mapping import (
    GpuIsolationUnavailableError,
    GpuMappingError,
    verify_isolated_gpu_access,
)


MAPPING = (
    ("GPU-selected", "/dev/nvidia0"),
    ("GPU-unselected", "/dev/nvidia1"),
)
REVIEWED = (("GPU-selected", "/dev/nvidia0"),)


def _opener(
    *,
    selected_ok: bool = True,
    unselected_denied: bool = True,
):
    opened: list[str] = []

    def open_device(path: Path, flags: int) -> int:
        del flags
        opened.append(path.name)
        if path.name == "nvidia0":
            if selected_ok:
                return 10
            raise OSError(errno.EACCES, "denied")
        if path.name == "nvidia1":
            if unselected_denied:
                raise OSError(errno.EPERM, "denied")
            return 11
        raise AssertionError(path)

    return opened, open_device


def test_isolated_access_requires_selected_open_and_unselected_denied() -> None:
    opened, open_device = _opener()

    verify_isolated_gpu_access(
        MAPPING,
        ("GPU-selected",),
        REVIEWED,
        worker_gpu_order=("GPU-selected",),
        cuda_visible_devices="GPU-selected",
        open_device=open_device,
        close_device=lambda _fd: None,
    )

    assert opened == ["nvidia0", "nvidia1"]


def test_isolated_access_fails_closed_when_unselected_device_opens() -> None:
    _, open_device = _opener(unselected_denied=False)

    with pytest.raises(
        GpuIsolationUnavailableError,
        match="not enforceable",
    ):
        verify_isolated_gpu_access(
            MAPPING,
            ("GPU-selected",),
            REVIEWED,
            worker_gpu_order=("GPU-selected",),
            cuda_visible_devices="GPU-selected",
            open_device=open_device,
            close_device=lambda _fd: None,
        )


def test_isolated_access_fails_when_selected_device_is_denied() -> None:
    _, open_device = _opener(selected_ok=False)

    with pytest.raises(GpuMappingError, match="selected GPU device access"):
        verify_isolated_gpu_access(
            MAPPING,
            ("GPU-selected",),
            REVIEWED,
            worker_gpu_order=("GPU-selected",),
            cuda_visible_devices="GPU-selected",
            open_device=open_device,
            close_device=lambda _fd: None,
        )


def test_isolated_access_rejects_cuda_order_mismatch() -> None:
    _, open_device = _opener()

    with pytest.raises(GpuMappingError, match="CUDA visible-device order"):
        verify_isolated_gpu_access(
            MAPPING,
            ("GPU-selected",),
            REVIEWED,
            worker_gpu_order=("GPU-selected",),
            cuda_visible_devices="GPU-other",
            open_device=open_device,
            close_device=lambda _fd: None,
        )


def test_isolated_access_rejects_reviewed_map_drift() -> None:
    _, open_device = _opener()

    with pytest.raises(GpuMappingError, match="reviewed GPU device map changed"):
        verify_isolated_gpu_access(
            MAPPING,
            ("GPU-selected",),
            (("GPU-selected", "/dev/nvidia1"),),
            worker_gpu_order=("GPU-selected",),
            cuda_visible_devices="GPU-selected",
            open_device=open_device,
            close_device=lambda _fd: None,
        )


def test_isolated_access_allows_already_exact_visible_set() -> None:
    opened, open_device = _opener()

    verify_isolated_gpu_access(
        REVIEWED,
        ("GPU-selected",),
        REVIEWED,
        worker_gpu_order=("GPU-selected",),
        cuda_visible_devices="GPU-selected",
        open_device=open_device,
        close_device=lambda _fd: None,
    )

    assert opened == ["nvidia0"]


def test_isolated_access_rejects_ambiguous_unselected_open_failure() -> None:
    def open_device(path: Path, flags: int) -> int:
        del flags
        if path.name == "nvidia0":
            return 10
        raise OSError(errno.ENODEV, "not a permission denial")

    with pytest.raises(GpuMappingError, match="failed ambiguously"):
        verify_isolated_gpu_access(
            MAPPING,
            ("GPU-selected",),
            REVIEWED,
            worker_gpu_order=("GPU-selected",),
            cuda_visible_devices="GPU-selected",
            open_device=open_device,
            close_device=lambda _fd: None,
        )
