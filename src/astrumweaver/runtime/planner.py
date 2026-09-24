"""Deterministic runtime compatibility planning.

The planner never silently replaces an explicitly selected runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from .contracts import (
    CompatibilityReason,
    ExecutionDemand,
    GPUTopology,
    RuntimeCompatibility,
    RuntimeCompatibilityContext,
    RuntimeProvider,
    RuntimeSelection,
    RuntimeSelectionMode,
)


class RuntimeSelectionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provider_id: str | None = None,
        reasons: tuple[CompatibilityReason, ...] = (),
    ) -> None:
        super().__init__(message)
        self.provider_id = provider_id
        self.reasons = reasons


@dataclass(frozen=True, slots=True)
class RuntimeResolution:
    selection: RuntimeSelection
    selected_provider_id: str | None
    compatible_provider_ids: tuple[str, ...]
    reports: Mapping[str, RuntimeCompatibility]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reports", MappingProxyType(dict(self.reports)))

        if self.selection.mode is RuntimeSelectionMode.EXPLICIT:
            if self.selected_provider_id != self.selection.provider_id:
                raise ValueError(
                    "explicit runtime resolution must preserve provider_id"
                )
        elif self.selected_provider_id is not None:
            raise ValueError(
                "recommend mode must not finalize a runtime selection"
            )


class RuntimeCatalog:
    def __init__(self, providers: Iterable[RuntimeProvider]) -> None:
        indexed: dict[str, RuntimeProvider] = {}
        for provider in providers:
            provider_id = provider.info.provider_id
            if provider_id in indexed:
                raise ValueError(f"duplicate runtime provider: {provider_id}")
            indexed[provider_id] = provider
        self._providers = MappingProxyType(indexed)

    def get(self, provider_id: str) -> RuntimeProvider | None:
        return self._providers.get(provider_id)

    def providers(self) -> tuple[RuntimeProvider, ...]:
        return tuple(
            self._providers[provider_id]
            for provider_id in sorted(self._providers)
        )


def _generic_reasons(
    context: RuntimeCompatibilityContext,
) -> tuple[CompatibilityReason, ...]:
    worker = context.worker
    host = context.host
    demand = context.demand
    resources = worker.resources
    reasons: list[CompatibilityReason] = []

    if demand.gpu_topology is GPUTopology.NONE:
        if resources.gpu_count != 0:
            reasons.append(
                CompatibilityReason(
                    code="gpu-topology-mismatch",
                    message=(
                        "execution demand requests no GPU but the selected "
                        "Worker is a GPU-owned compute unit"
                    ),
                )
            )
    elif demand.gpu_topology is GPUTopology.SINGLE_GPU:
        if resources.gpu_count != 1:
            reasons.append(
                CompatibilityReason(
                    code="gpu-topology-mismatch",
                    message=(
                        "single_gpu execution requires a one-GPU Worker"
                    ),
                )
            )
    else:
        if resources.gpu_count < 2:
            reasons.append(
                CompatibilityReason(
                    code="gpu-topology-mismatch",
                    message=(
                        "multi_gpu execution requires a Worker with at least "
                        "two GPUs"
                    ),
                )
            )

    if resources.gpu_count < demand.effective_min_gpu_count:
        reasons.append(
            CompatibilityReason(
                code="insufficient-gpu-count",
                message=(
                    f"Worker exposes {resources.gpu_count} GPUs but demand "
                    f"requires at least {demand.effective_min_gpu_count}"
                ),
            )
        )

    if resources.total_vram_mb < demand.min_total_vram_mb:
        reasons.append(
            CompatibilityReason(
                code="insufficient-total-vram",
                message=(
                    f"Worker total VRAM {resources.total_vram_mb} MiB is below "
                    f"required {demand.min_total_vram_mb} MiB"
                ),
            )
        )

    if resources.max_single_gpu_vram_mb < demand.min_single_gpu_vram_mb:
        reasons.append(
            CompatibilityReason(
                code="insufficient-single-gpu-vram",
                message=(
                    "Worker maximum single-device VRAM "
                    f"{resources.max_single_gpu_vram_mb} MiB is below required "
                    f"{demand.min_single_gpu_vram_mb} MiB"
                ),
            )
        )

    if host.host_ram_mb < demand.min_host_ram_mb:
        reasons.append(
            CompatibilityReason(
                code="insufficient-host-ram",
                message=(
                    f"Host RAM {host.host_ram_mb} MiB is below required "
                    f"{demand.min_host_ram_mb} MiB"
                ),
            )
        )
    elif (
        demand.preferred_host_ram_mb
        and host.host_ram_mb < demand.preferred_host_ram_mb
    ):
        reasons.append(
            CompatibilityReason(
                code="below-preferred-host-ram",
                message=(
                    f"Host RAM {host.host_ram_mb} MiB is below preferred "
                    f"{demand.preferred_host_ram_mb} MiB"
                ),
                blocking=False,
            )
        )

    return tuple(reasons)


def evaluate_provider(
    provider: RuntimeProvider,
    context: RuntimeCompatibilityContext,
) -> RuntimeCompatibility:
    generic = _generic_reasons(context)
    provider_report = provider.compatibility(context)
    if provider_report.provider_id != provider.info.provider_id:
        raise ValueError(
            "runtime provider returned compatibility for a different provider_id"
        )
    return RuntimeCompatibility(
        provider_id=provider.info.provider_id,
        reasons=generic + provider_report.reasons,
    )


def evaluate_catalog(
    catalog: RuntimeCatalog,
    context: RuntimeCompatibilityContext,
) -> Mapping[str, RuntimeCompatibility]:
    reports = {
        provider.info.provider_id: evaluate_provider(provider, context)
        for provider in catalog.providers()
    }
    return MappingProxyType(reports)


def resolve_runtime(
    *,
    catalog: RuntimeCatalog,
    context: RuntimeCompatibilityContext,
    selection: RuntimeSelection,
) -> RuntimeResolution:
    reports = evaluate_catalog(catalog, context)
    compatible_ids = tuple(
        provider_id
        for provider_id in sorted(reports)
        if reports[provider_id].compatible
    )

    if selection.mode is RuntimeSelectionMode.RECOMMEND:
        return RuntimeResolution(
            selection=selection,
            selected_provider_id=None,
            compatible_provider_ids=compatible_ids,
            reports=reports,
        )

    assert selection.provider_id is not None
    provider = catalog.get(selection.provider_id)
    if provider is None:
        raise RuntimeSelectionError(
            f"explicit runtime provider is not installed: {selection.provider_id}",
            provider_id=selection.provider_id,
            reasons=(
                CompatibilityReason(
                    code="provider-not-found",
                    message="selected runtime provider is not present in the catalog",
                ),
            ),
        )

    report = reports[selection.provider_id]
    if not report.compatible:
        raise RuntimeSelectionError(
            (
                "explicit runtime selection is incompatible: "
                f"{selection.provider_id}"
            ),
            provider_id=selection.provider_id,
            reasons=report.reasons,
        )

    return RuntimeResolution(
        selection=selection,
        selected_provider_id=selection.provider_id,
        compatible_provider_ids=compatible_ids,
        reports=reports,
    )
