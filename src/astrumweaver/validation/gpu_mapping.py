"""Canonical physical NVIDIA GPU UUID to device-node mapping."""

from __future__ import annotations

import argparse
import errno
import os
import re
import stat
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Callable, Mapping, Sequence


class GpuMappingError(RuntimeError):
    """A physical GPU mapping could not be proven safely."""


class GpuIsolationUnavailableError(GpuMappingError):
    """The requested GPU subset cannot be enforced in this service context."""


_GPU_UUID_RE = re.compile(r"^GPU-[^\s,=]+$")
_MINOR_RE = re.compile(r"^[0-9]+$")
_DEVICE_PATH_RE = re.compile(r"^/dev/nvidia[0-9]+$")
_NVIDIA_DEVICE_NAMES = frozenset({"nvidia-frontend", "nvidia"})
_Runner = Callable[..., subprocess.CompletedProcess[str]]
_MappingRows = Mapping[str, Sequence[object]]


def _validate_uuid(value: str) -> str:
    normalized = value.strip()
    if not _GPU_UUID_RE.fullmatch(normalized):
        raise GpuMappingError("GPU UUID data is invalid")
    return normalized


def _run_nvidia_query(
    command: str,
    fields: str,
    *,
    run: _Runner,
) -> subprocess.CompletedProcess[str]:
    try:
        return run(
            [
                command,
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits"
                if fields == "uuid,minor_number"
                else "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise GpuMappingError("NVIDIA visibility command is unavailable") from exc


def _parse_visible_uuids(output: str) -> tuple[str, ...]:
    values: list[str] = []
    for raw_line in output.splitlines():
        value = raw_line.strip()
        if not value:
            continue
        values.append(_validate_uuid(value))

    if not values:
        raise GpuMappingError("no visible NVIDIA GPU UUIDs were reported")
    if len(set(values)) != len(values):
        raise GpuMappingError("visible NVIDIA GPU UUIDs are not unique")
    return tuple(sorted(values))


def _query_visible_uuids(
    command: str,
    *,
    run: _Runner,
) -> tuple[str, ...]:
    completed = _run_nvidia_query(command, "uuid", run=run)
    if completed.returncode != 0:
        raise GpuMappingError("NVIDIA GPU UUID visibility query failed")
    return _parse_visible_uuids(completed.stdout)


def _parse_uuid_minor_rows(output: str) -> dict[str, int]:
    rows: dict[str, int] = {}
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        fields = [item.strip() for item in raw_line.split(",")]
        if len(fields) != 2:
            raise GpuMappingError("NVIDIA GPU UUID/minor data is malformed")
        uuid = _validate_uuid(fields[0])
        if not _MINOR_RE.fullmatch(fields[1]):
            raise GpuMappingError("NVIDIA GPU minor data is malformed")
        if uuid in rows:
            raise GpuMappingError("NVIDIA GPU UUID/minor data is ambiguous")
        rows[uuid] = int(fields[1], 10)

    if not rows:
        raise GpuMappingError("NVIDIA GPU UUID/minor data is empty")
    return rows


def _stdout_contains_gpu_data(output: str) -> bool:
    """Return whether failed-query stdout contains GPU-shaped result data."""

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        first_field = line.split(",", 1)[0].strip()
        if _GPU_UUID_RE.fullmatch(first_field):
            return True
    return False


def _minor_query_is_explicitly_unsupported(
    completed: subprocess.CompletedProcess[str],
) -> bool:
    """Recognize only explicit query-field rejection diagnostics.

    NVIDIA CLI versions differ on whether query validation errors are emitted
    on stdout or stderr. A compatibility fallback is safe only when the
    diagnostic itself proves that minor_number is not a supported query
    field. Partial GPU data, empty diagnostics, and runtime/driver failures
    remain fail-closed.
    """

    if completed.returncode == 0:
        return False

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if _stdout_contains_gpu_data(stdout):
        return False

    diagnostic = "\n".join(
        part for part in (stdout, stderr) if part
    ).casefold()
    if not diagnostic.strip():
        return False

    runtime_failure_markers = (
        "failed to initialize nvml",
        "driver/library version mismatch",
        "couldn\'t communicate with the nvidia driver",
        "could not communicate with the nvidia driver",
        "failed to communicate with the nvidia driver",
        "permission denied",
        "not permitted",
        "insufficient permission",
    )
    if any(marker in diagnostic for marker in runtime_failure_markers):
        return False

    mentions_minor = (
        "minor_number" in diagnostic or "minor number" in diagnostic
    )
    if not mentions_minor:
        return False

    if "not a valid field to query" in diagnostic:
        return True

    if "field" not in diagnostic:
        return False
    return any(
        marker in diagnostic
        for marker in (
            "unknown",
            "unsupported",
            "not supported",
            "unrecognized",
            "not recognized",
        )
    )


def _query_primary_mapping(
    command: str,
    *,
    run: _Runner,
) -> dict[str, int] | None:
    completed = _run_nvidia_query(command, "uuid,minor_number", run=run)
    if completed.returncode != 0:
        # Older NVIDIA utilities reject the minor_number field. Only an
        # explicit field-query rejection may fall back to the kernel
        # information files; driver, permission, partial-data, and transient
        # failures remain fail-closed.
        if _minor_query_is_explicitly_unsupported(completed):
            return None
        raise GpuMappingError("NVIDIA GPU UUID/minor query failed")
    return _parse_uuid_minor_rows(completed.stdout)

def _proc_mapping_rows(proc_root: Path) -> dict[str, list[object]]:
    information_files = sorted(proc_root.glob("*/information"))
    if not information_files:
        raise GpuMappingError("NVIDIA GPU information entries are unavailable")

    rows: dict[str, list[object]] = {}
    for information in information_files:
        try:
            text = information.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            # An unreadable unrelated entry must not override the nvidia-smi
            # visibility authority. A visible UUID with no readable record
            # still fails below because it has zero matching rows.
            continue

        uuids: list[str] = []
        minors: list[str] = []
        for raw_line in text.splitlines():
            key, separator, raw_value = raw_line.partition(":")
            if not separator:
                continue
            value = raw_value.strip()
            if key.strip() == "GPU UUID":
                uuids.append(value)
            elif key.strip() == "Device Minor":
                minors.append(value)

        if not uuids:
            continue

        uuid = uuids[0]
        # Multiple copies of either field in one information file are
        # ambiguous for a visible UUID, but remain ignorable when unrelated.
        value: object = minors[0] if len(uuids) == 1 and len(minors) == 1 else None
        rows.setdefault(uuid, []).append(value)
    return rows


def _validate_mapping(
    visible_uuids: Sequence[str],
    rows: _MappingRows,
    *,
    device_root: Path,
    nvidia_major: int,
    device_stat: Callable[[Path], os.stat_result],
) -> tuple[tuple[str, str], ...]:
    mapping: dict[str, str] = {}
    for uuid in visible_uuids:
        candidates = rows.get(uuid, ())
        if len(candidates) != 1:
            raise GpuMappingError("a visible GPU lacks exactly one device mapping")

        raw_minor = candidates[0]
        if isinstance(raw_minor, bool):
            raise GpuMappingError("a visible GPU has an invalid device minor")
        if isinstance(raw_minor, int):
            minor = raw_minor
        elif isinstance(raw_minor, str) and _MINOR_RE.fullmatch(raw_minor):
            minor = int(raw_minor, 10)
        else:
            raise GpuMappingError("a visible GPU has an invalid device minor")
        if minor < 0:
            raise GpuMappingError("a visible GPU has an invalid device minor")

        device_path = f"/dev/nvidia{minor}"
        if not _DEVICE_PATH_RE.fullmatch(device_path):
            raise GpuMappingError("a visible GPU resolved to an invalid device path")
        try:
            node_stat = device_stat(device_root / f"nvidia{minor}")
        except OSError as exc:
            raise GpuMappingError(
                "a visible GPU device node is unavailable"
            ) from exc
        if not stat.S_ISCHR(node_stat.st_mode):
            raise GpuMappingError(
                "a visible GPU device node is not a character device"
            )
        if (
            os.major(node_stat.st_rdev) != nvidia_major
            or os.minor(node_stat.st_rdev) != minor
        ):
            raise GpuMappingError(
                "a visible GPU device node does not match the NVIDIA major/minor"
            )
        if device_path in mapping.values():
            raise GpuMappingError("visible GPUs reuse one device minor")
        mapping[uuid] = device_path

    return tuple(sorted(mapping.items()))


def _nvidia_character_device_major(proc_devices: Path) -> int:
    try:
        lines = proc_devices.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise GpuMappingError("NVIDIA character device registry is unavailable") from exc

    in_character_devices = False
    majors: set[int] = set()
    for line in lines:
        stripped = line.strip()
        if stripped == "Character devices:":
            in_character_devices = True
            continue
        if stripped == "Block devices:":
            break
        if not in_character_devices:
            continue
        fields = stripped.split(None, 1)
        if len(fields) != 2 or fields[1] not in _NVIDIA_DEVICE_NAMES:
            continue
        try:
            majors.add(int(fields[0], 10))
        except ValueError as exc:
            raise GpuMappingError(
                "NVIDIA character device registry is malformed"
            ) from exc

    if len(majors) != 1:
        raise GpuMappingError(
            "NVIDIA character device major is unavailable or ambiguous"
        )
    return next(iter(majors))


def discover_gpu_mapping(
    *,
    nvidia_smi: str = "nvidia-smi",
    proc_root: Path = Path("/proc/driver/nvidia/gpus"),
    proc_devices: Path = Path("/proc/devices"),
    device_root: Path = Path("/dev"),
    run: _Runner = subprocess.run,
    device_stat: Callable[[Path], os.stat_result] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return validated mappings; ``device_stat`` is only a test seam."""

    visible_uuids = _query_visible_uuids(nvidia_smi, run=run)
    primary = _query_primary_mapping(nvidia_smi, run=run)
    if primary is None:
        rows: _MappingRows = _proc_mapping_rows(proc_root)
    else:
        rows = {uuid: [minor] for uuid, minor in primary.items()}
    nvidia_major = _nvidia_character_device_major(proc_devices)
    return _validate_mapping(
        visible_uuids,
        rows,
        device_root=device_root,
        nvidia_major=nvidia_major,
        device_stat=Path.stat if device_stat is None else device_stat,
    )


def _load_expected(path: Path) -> tuple[str, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise GpuMappingError("expected GPU UUID file is unavailable") from exc

    values: list[str] = []
    for line in lines:
        if line.strip():
            values.append(_validate_uuid(line))
    if not values:
        raise GpuMappingError("expected GPU UUID set is empty")
    if len(set(values)) != len(values):
        raise GpuMappingError("expected GPU UUID set is not unique")
    return tuple(sorted(values))


def _load_worker_gpu_order(path: Path) -> tuple[str, ...]:
    try:
        with path.open("rb") as handle:
            config = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise GpuMappingError("Worker GPU configuration is unavailable") from exc

    worker = dict(config.get("worker") or {})
    raw_values = worker.get("gpu_uuids", ())
    if (
        isinstance(raw_values, (str, bytes))
        or not isinstance(raw_values, list)
    ):
        raise GpuMappingError("Worker GPU UUID order is invalid")
    values = tuple(_validate_uuid(str(value)) for value in raw_values)
    if not values:
        raise GpuMappingError("Worker GPU UUID order is empty")
    if len(set(values)) != len(values):
        raise GpuMappingError("Worker GPU UUID order is not unique")
    return values


def verify_isolated_gpu_access(
    mapping: Sequence[tuple[str, str]],
    expected: Sequence[str],
    reviewed_mapping: Sequence[tuple[str, str]],
    *,
    worker_gpu_order: Sequence[str],
    cuda_visible_devices: str | None,
    device_root: Path = Path("/dev"),
    open_device: Callable[[Path, int], int] = os.open,
    close_device: Callable[[int], None] = os.close,
) -> None:
    """Prove that only the selected physical GPUs are openable.

    This is intentionally stronger than checking systemd configuration text.
    A host/container may report DevicePolicy=closed while the delegated
    cgroup does not actually deny device opens. Such environments must fail
    closed instead of claiming subset isolation.
    """

    selected = _select_mapping(mapping, expected)
    reviewed = tuple(sorted(reviewed_mapping))
    if selected != reviewed:
        raise GpuMappingError("reviewed GPU device map changed")
    if len(mapping) <= len(selected):
        raise GpuMappingError("isolated-access requires a host GPU superset")

    ordered = tuple(_validate_uuid(str(value)) for value in worker_gpu_order)
    if len(set(ordered)) != len(ordered):
        raise GpuMappingError("Worker GPU UUID order is not unique")
    if tuple(sorted(ordered)) != tuple(sorted(expected)):
        raise GpuMappingError("Worker GPU UUID order does not match selected set")

    expected_cuda = ",".join(ordered)
    if cuda_visible_devices != expected_cuda:
        raise GpuMappingError("CUDA visible-device order does not match Worker GPU order")

    selected_paths = {path for _, path in selected}
    all_paths = {path for _, path in mapping}

    def actual_path(logical_path: str) -> Path:
        return device_root / Path(logical_path).name

    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    for logical_path in sorted(selected_paths):
        try:
            fd = open_device(actual_path(logical_path), flags)
        except OSError as exc:
            raise GpuMappingError("selected GPU device access is unavailable") from exc
        close_device(fd)

    denied_errnos = {errno.EACCES, errno.EPERM}
    for logical_path in sorted(all_paths - selected_paths):
        try:
            fd = open_device(actual_path(logical_path), flags)
        except OSError as exc:
            if exc.errno in denied_errnos:
                continue
            raise GpuMappingError(
                "unselected GPU device access probe failed ambiguously"
            ) from exc
        else:
            close_device(fd)
            raise GpuIsolationUnavailableError(
                "GPU subset isolation is not enforceable in this environment"
            )


def _select_mapping(
    mapping: Sequence[tuple[str, str]],
    expected: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    available = dict(mapping)
    selected: dict[str, str] = {}
    for uuid in expected:
        if uuid not in available:
            raise GpuMappingError("a configured GPU UUID is not visible")
        selected[uuid] = available[uuid]
    return tuple(sorted(selected.items()))


def _load_reviewed_map(path: Path) -> tuple[tuple[str, str], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise GpuMappingError("reviewed GPU device map is unavailable") from exc

    mapping: dict[str, str] = {}
    for line in lines:
        if not line.strip() or "=" not in line:
            raise GpuMappingError("reviewed GPU device map is malformed")
        uuid_raw, device_path = (item.strip() for item in line.split("=", 1))
        uuid = _validate_uuid(uuid_raw)
        if not _DEVICE_PATH_RE.fullmatch(device_path):
            raise GpuMappingError("reviewed GPU device map is malformed")
        if uuid in mapping:
            raise GpuMappingError("reviewed GPU device map is ambiguous")
        mapping[uuid] = device_path
    if not mapping:
        raise GpuMappingError("reviewed GPU device map is empty")
    return tuple(sorted(mapping.items()))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrumweaver-gpu-device-map")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument(
        "--proc-root",
        type=Path,
        default=Path(os.environ.get("ASTRUMWEAVER_GPU_PROC_ROOT", "/proc/driver/nvidia/gpus")),
    )
    parser.add_argument(
        "--device-root",
        type=Path,
        default=Path(os.environ.get("ASTRUMWEAVER_GPU_DEVICE_ROOT", "/dev")),
    )
    parser.add_argument(
        "--proc-devices",
        type=Path,
        default=Path(os.environ.get("ASTRUMWEAVER_GPU_PROC_DEVICES", "/proc/devices")),
    )
    parser.add_argument(
        "mode",
        choices=("discover", "discover-visible", "verify", "verify-isolated-access"),
    )
    parser.add_argument("paths", nargs="*")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        mapping = discover_gpu_mapping(
            nvidia_smi=args.nvidia_smi,
            proc_root=args.proc_root,
            proc_devices=args.proc_devices,
            device_root=args.device_root,
        )
        if args.mode == "discover-visible":
            if args.paths:
                raise GpuMappingError("discover-visible arguments are invalid")
            selected = mapping
        else:
            if args.mode == "discover" and len(args.paths) != 1:
                raise GpuMappingError("discover arguments are invalid")
            if args.mode == "verify" and len(args.paths) != 2:
                raise GpuMappingError("mapping arguments are invalid")
            if args.mode == "verify-isolated-access" and len(args.paths) != 3:
                raise GpuMappingError(
                    "isolated-access arguments are invalid"
                )
            expected = _load_expected(Path(args.paths[0]))
            selected = _select_mapping(mapping, expected)
            if args.mode == "verify":
                if selected != _load_reviewed_map(Path(args.paths[1])):
                    raise GpuMappingError("reviewed GPU device map changed")
                print(f"[astrumweaver-gpu-device-map] ok count={len(selected)}")
                return 0
            if args.mode == "verify-isolated-access":
                worker_order = _load_worker_gpu_order(Path(args.paths[2]))
                verify_isolated_gpu_access(
                    mapping,
                    expected,
                    _load_reviewed_map(Path(args.paths[1])),
                    worker_gpu_order=worker_order,
                    cuda_visible_devices=os.environ.get(
                        "CUDA_VISIBLE_DEVICES"
                    ),
                    device_root=args.device_root,
                )
                print(
                    "[astrumweaver-gpu-device-map] isolated-access ok "
                    f"selected_count={len(selected)} host_count={len(mapping)}"
                )
                return 0

        for uuid, device_path in selected:
            print(f"{uuid}={device_path}")
        return 0
    except GpuIsolationUnavailableError as exc:
        print(
            f"[astrumweaver-gpu-device-map] ISOLATION_UNAVAILABLE: {exc}",
            file=sys.stderr,
        )
        return 3
    except GpuMappingError as exc:
        print(f"[astrumweaver-gpu-device-map] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
