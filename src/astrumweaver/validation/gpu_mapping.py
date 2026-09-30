"""Compatibility imports for the canonical GPU mapping module.

New runtime code must import :mod:`astrumweaver.gpu_mapping` directly so
Worker code does not depend on the eager validation package initializer.
"""

from ..gpu_mapping import (
    GpuIsolationUnavailableError,
    GpuMappingError,
    _load_expected,
    _load_reviewed_map,
    _load_worker_gpu_order,
    _minor_query_is_explicitly_unsupported,
    _parse_uuid_minor_rows,
    _parse_visible_uuids,
    _proc_mapping_rows,
    _query_primary_mapping,
    _query_visible_uuids,
    _select_mapping,
    _stdout_contains_gpu_data,
    _validate_mapping,
    _validate_uuid,
    discover_gpu_mapping,
    load_reviewed_gpu_map,
    main,
    verify_isolated_gpu_access,
)

__all__ = [
    "GpuIsolationUnavailableError",
    "GpuMappingError",
    "discover_gpu_mapping",
    "load_reviewed_gpu_map",
    "verify_isolated_gpu_access",
    "main",
]
