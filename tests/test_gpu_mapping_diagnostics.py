from __future__ import annotations

import subprocess

import pytest

from astrumweaver.validation.gpu_mapping import (
    GpuMappingError,
    _minor_query_is_explicitly_unsupported,
    _query_primary_mapping,
)


def _completed(
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 2,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        ["nvidia-smi"],
        returncode,
        stdout,
        stderr,
    )


@pytest.mark.parametrize("diagnostic_stream", ("stdout", "stderr"))
def test_real_host_not_valid_field_diagnostic_allows_proc_fallback(
    diagnostic_stream: str,
) -> None:
    diagnostic = 'Field "minor_number" is not a valid field to query.\n'
    completed = _completed(
        stdout=diagnostic if diagnostic_stream == "stdout" else "",
        stderr=diagnostic if diagnostic_stream == "stderr" else "",
    )

    assert _minor_query_is_explicitly_unsupported(completed)
    assert _query_primary_mapping(
        "nvidia-smi",
        run=lambda *_args, **_kwargs: completed,
    ) is None


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    (
        ("", ""),
        ('Field "minor_number" is invalid.\n', ""),
        ('GPU-example, <INVALID>\nField "minor_number" is not a valid field to query.\n', ""),
        ("", "Failed to initialize NVML: Unknown Error\n"),
        ("", "Permission denied while querying minor_number field\n"),
        ("", "Could not communicate with the NVIDIA driver while querying minor_number field\n"),
    ),
)
def test_non_explicit_or_runtime_failures_do_not_fallback(
    stdout: str,
    stderr: str,
) -> None:
    completed = _completed(stdout=stdout, stderr=stderr)

    assert not _minor_query_is_explicitly_unsupported(completed)
    with pytest.raises(GpuMappingError, match="UUID/minor query failed"):
        _query_primary_mapping(
            "nvidia-smi",
            run=lambda *_args, **_kwargs: completed,
        )


@pytest.mark.parametrize(
    "diagnostic",
    (
        'Unknown field: minor_number\n',
        'Field minor_number is unsupported\n',
        'Field minor_number is not supported\n',
        'Unrecognized field minor_number\n',
        'Field minor number is not recognized\n',
    ),
)
def test_existing_explicit_field_rejections_still_fallback(
    diagnostic: str,
) -> None:
    completed = _completed(stderr=diagnostic)

    assert _minor_query_is_explicitly_unsupported(completed)
