"""Worker-local accelerator identity preflight."""

from __future__ import annotations

import subprocess

from ..contracts import WorkerSpec


class PreflightError(RuntimeError):
    """Local worker resource identity does not match configuration."""


def observed_nvidia_gpu_uuids() -> tuple[str, ...]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PreflightError("nvidia-smi is not available") from exc
    except subprocess.CalledProcessError as exc:
        raise PreflightError("nvidia-smi failed") from exc

    values = tuple(
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    )
    if len(values) != len(set(values)):
        raise PreflightError("observed GPU UUID set contains duplicates")
    return values


def validate_worker_resources(spec: WorkerSpec) -> None:
    expected = tuple(spec.gpu_uuids)
    if not expected:
        if spec.resources.gpu_count != 0:
            raise PreflightError("GPU resource shape requires configured GPU UUIDs")
        return

    observed = observed_nvidia_gpu_uuids()
    if set(expected) != set(observed):
        raise PreflightError(
            "GPU UUID set mismatch "
            f"(expected_count={len(expected)} observed_count={len(observed)})"
        )
