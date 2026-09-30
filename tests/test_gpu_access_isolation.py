from __future__ import annotations

import errno
import os

import pytest

from astrumweaver.validation.gpu_mapping import (
    GpuIsolationUnavailable,
    GpuMappingError,
    probe_gpu_access_isolation,
)


MAPPING = (
    ("GPU-selected", "/dev/nvidia0"),
    ("GPU-other", "/dev/nvidia1"),
)


def test_gpu_access_probe_requires_selected_open_and_unselected_denied() -> None:
    opened: list[str] = []
    closed: list[int] = []

    def opener(path: str, flags: int) -> int:
        assert flags == os.O_RDWR | os.O_CLOEXEC
        opened.append(path)
        if path == "/dev/nvidia1":
            raise OSError(errno.EPERM, "blocked")
        return 17

    probe_gpu_access_isolation(
        ("GPU-selected",),
        MAPPING,
        opener=opener,
        closer=closed.append,
    )

    assert opened == ["/dev/nvidia1", "/dev/nvidia0"] or opened == [
        "/dev/nvidia0",
        "/dev/nvidia1",
    ]
    assert closed == [17]


def test_gpu_access_probe_reports_unavailable_when_unselected_opens() -> None:
    def opener(path: str, flags: int) -> int:
        return 11 if path.endswith("0") else 12

    closed: list[int] = []
    with pytest.raises(GpuIsolationUnavailable, match="not enforced"):
        probe_gpu_access_isolation(
            ("GPU-selected",),
            MAPPING,
            opener=opener,
            closer=closed.append,
        )

    assert closed


def test_gpu_access_probe_rejects_selected_inaccessible() -> None:
    def opener(path: str, flags: int) -> int:
        if path == "/dev/nvidia0":
            raise OSError(errno.EACCES, "blocked")
        raise OSError(errno.EPERM, "blocked")

    with pytest.raises(GpuMappingError, match="selected GPU device"):
        probe_gpu_access_isolation(
            ("GPU-selected",),
            MAPPING,
            opener=opener,
            closer=lambda _fd: None,
        )


def test_gpu_access_probe_rejects_non_access_denial_for_unselected() -> None:
    def opener(path: str, flags: int) -> int:
        if path == "/dev/nvidia1":
            raise OSError(errno.ENOENT, "missing")
        return 10

    with pytest.raises(GpuMappingError, match="could not be proven"):
        probe_gpu_access_isolation(
            ("GPU-selected",),
            MAPPING,
            opener=opener,
            closer=lambda _fd: None,
        )


def test_gpu_access_probe_requires_real_superset() -> None:
    with pytest.raises(GpuMappingError, match="visible superset"):
        probe_gpu_access_isolation(
            ("GPU-selected",),
            (("GPU-selected", "/dev/nvidia0"),),
            opener=lambda _path, _flags: 1,
            closer=lambda _fd: None,
        )
