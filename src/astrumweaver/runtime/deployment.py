"""Serializable RuntimeProvider deployment contract for Worker services."""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..contracts import WorkerSpec
from .contracts import (
    ExecutionDemand,
    GPUTopology,
    ModelDemand,
    ModelPreparationPolicy,
    ModelTopology,
    ResidencyPolicy,
    RuntimeCompatibilityContext,
    RuntimeHostFacts,
    RuntimeProvider,
    RuntimeSetupIntent,
)
from .providers import (
    EXLLAMAV3_PROVIDER_ID,
    FREETOKEN_PROVIDER_ID,
    LLAMA_CPP_PROVIDER_ID,
    OLLAMA_PROVIDER_ID,
    VLLM_PROVIDER_ID,
    ExLlamaV3Provider,
    ExLlamaV3ProviderConfig,
    FreeTokenProvider,
    FreeTokenProviderConfig,
    LlamaCppProvider,
    LlamaCppProviderConfig,
    OllamaProvider,
    OllamaProviderConfig,
    VllmProvider,
    VllmProviderConfig,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"runtime deployment value is not JSON-compatible: {type(value).__name__}"
    )


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class RuntimeDeploymentSpec:
    """Exact provider+demand+setup state consumed by one Worker service."""

    provider_id: str
    provider_config: Mapping[str, Any]
    demand: ExecutionDemand
    setup_intent: RuntimeSetupIntent
    schema_version: str = "v1"

    def __post_init__(self) -> None:
        provider_id = str(self.provider_id).strip()
        if not provider_id:
            raise ValueError("provider_id must not be blank")
        if self.setup_intent.provider_id != provider_id:
            raise ValueError(
                "runtime deployment setup intent changed provider identity"
            )
        if not isinstance(self.demand, ExecutionDemand):
            raise TypeError("demand must be ExecutionDemand")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(
            self,
            "provider_config",
            _mapping(self.provider_config),
        )
        schema_version = str(self.schema_version).strip()
        if schema_version != "v1":
            raise ValueError(
                f"unsupported runtime deployment schema: {schema_version}"
            )
        object.__setattr__(self, "schema_version", schema_version)

    def to_dict(self) -> dict[str, Any]:
        model = self.demand.model
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "provider_config": _json_value(self.provider_config),
            "demand": {
                "model": {
                    "model_ref": model.model_ref,
                    "model_format": model.model_format,
                    "topology": model.topology.value,
                    "estimated_size_mb": model.estimated_size_mb,
                    "metadata": _json_value(model.metadata),
                },
                "residency_policy": self.demand.residency_policy.value,
                "gpu_topology": self.demand.gpu_topology.value,
                "min_gpu_count": self.demand.min_gpu_count,
                "min_total_vram_mb": self.demand.min_total_vram_mb,
                "min_single_gpu_vram_mb": (
                    self.demand.min_single_gpu_vram_mb
                ),
                "min_host_ram_mb": self.demand.min_host_ram_mb,
                "preferred_host_ram_mb": (
                    self.demand.preferred_host_ram_mb
                ),
                "metadata": _json_value(self.demand.metadata),
            },
            "setup_intent": {
                "provider_id": self.setup_intent.provider_id,
                "package_references": list(
                    self.setup_intent.package_references
                ),
                "configuration": _json_value(
                    self.setup_intent.configuration
                ),
                "model_preparation": (
                    self.setup_intent.model_preparation.value
                ),
                "model_ref": self.setup_intent.model_ref,
                "requires_privilege": (
                    self.setup_intent.requires_privilege
                ),
                "metadata": _json_value(self.setup_intent.metadata),
            },
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "RuntimeDeploymentSpec":
        data = dict(value)
        demand_data = dict(data["demand"])
        model_data = dict(demand_data["model"])
        intent_data = dict(data["setup_intent"])
        return cls(
            schema_version=str(data.get("schema_version", "v1")),
            provider_id=str(data["provider_id"]),
            provider_config=dict(data.get("provider_config") or {}),
            demand=ExecutionDemand(
                model=ModelDemand(
                    model_ref=str(model_data["model_ref"]),
                    model_format=str(model_data["model_format"]),
                    topology=ModelTopology(str(model_data["topology"])),
                    estimated_size_mb=model_data.get("estimated_size_mb"),
                    metadata=dict(model_data.get("metadata") or {}),
                ),
                residency_policy=ResidencyPolicy(
                    str(demand_data["residency_policy"])
                ),
                gpu_topology=GPUTopology(
                    str(demand_data["gpu_topology"])
                ),
                min_gpu_count=int(
                    demand_data.get("min_gpu_count") or 0
                ),
                min_total_vram_mb=int(
                    demand_data.get("min_total_vram_mb") or 0
                ),
                min_single_gpu_vram_mb=int(
                    demand_data.get("min_single_gpu_vram_mb") or 0
                ),
                min_host_ram_mb=int(
                    demand_data.get("min_host_ram_mb") or 0
                ),
                preferred_host_ram_mb=int(
                    demand_data.get("preferred_host_ram_mb") or 0
                ),
                metadata=dict(demand_data.get("metadata") or {}),
            ),
            setup_intent=RuntimeSetupIntent(
                provider_id=str(intent_data["provider_id"]),
                package_references=tuple(
                    str(item)
                    for item in intent_data.get(
                        "package_references", ()
                    )
                ),
                configuration=dict(
                    intent_data.get("configuration") or {}
                ),
                model_preparation=ModelPreparationPolicy(
                    str(intent_data.get(
                        "model_preparation",
                        ModelPreparationPolicy.REFERENCE_ONLY.value,
                    ))
                ),
                model_ref=intent_data.get("model_ref"),
                requires_privilege=bool(
                    intent_data.get("requires_privilege", False)
                ),
                metadata=dict(intent_data.get("metadata") or {}),
            ),
        )


_PROVIDER_TYPES: Mapping[
    str,
    tuple[type[Any], type[Any]],
] = {
    OLLAMA_PROVIDER_ID: (OllamaProvider, OllamaProviderConfig),
    LLAMA_CPP_PROVIDER_ID: (
        LlamaCppProvider,
        LlamaCppProviderConfig,
    ),
    VLLM_PROVIDER_ID: (VllmProvider, VllmProviderConfig),
    FREETOKEN_PROVIDER_ID: (
        FreeTokenProvider,
        FreeTokenProviderConfig,
    ),
    EXLLAMAV3_PROVIDER_ID: (
        ExLlamaV3Provider,
        ExLlamaV3ProviderConfig,
    ),
}


def discover_runtime_host_facts(
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
) -> RuntimeHostFacts:
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count <= 0:
        raise RuntimeError("runtime host CPU count is unavailable")

    try:
        meminfo = meminfo_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError("runtime host RAM information is unavailable") from exc

    host_ram_mb: int | None = None
    for raw_line in meminfo.splitlines():
        if not raw_line.startswith("MemTotal:"):
            continue
        fields = raw_line.split()
        if len(fields) < 2:
            break
        try:
            host_ram_mb = int(fields[1]) // 1024
        except ValueError as exc:
            raise RuntimeError("runtime host RAM information is invalid") from exc
        break
    if host_ram_mb is None or host_ram_mb <= 0:
        raise RuntimeError("runtime host RAM information is invalid")

    architecture = platform.machine().strip()
    if not architecture:
        raise RuntimeError("runtime host architecture is unavailable")

    return RuntimeHostFacts(
        cpu_count=cpu_count,
        host_ram_mb=host_ram_mb,
        architecture=architecture,
    )


def provider_from_deployment(
    deployment: RuntimeDeploymentSpec,
) -> RuntimeProvider:
    entry = _PROVIDER_TYPES.get(deployment.provider_id)
    if entry is None:
        raise ValueError(
            f"unsupported runtime provider: {deployment.provider_id}"
        )
    provider_type, config_type = entry
    config = config_type(**dict(deployment.provider_config))
    return provider_type(config)


def build_runtime_deployment_spec(
    provider: RuntimeProvider,
    context: RuntimeCompatibilityContext,
) -> RuntimeDeploymentSpec:
    """Freeze one reviewed provider configuration and setup intent."""
    setup = provider.setup_intent(context)
    config = getattr(provider, "config", None)
    provider_config = (
        _json_value(config)
        if config is not None
        else {}
    )
    if not isinstance(provider_config, Mapping):
        raise TypeError(
            "runtime provider config must serialize to a mapping"
        )
    return RuntimeDeploymentSpec(
        provider_id=provider.info.provider_id,
        provider_config=provider_config,
        demand=context.demand,
        setup_intent=setup,
    )


def managed_runtime_from_deployment(
    deployment: RuntimeDeploymentSpec,
    *,
    worker: WorkerSpec,
    host: RuntimeHostFacts,
):
    provider = provider_from_deployment(deployment)
    context = RuntimeCompatibilityContext(
        worker=worker,
        host=host,
        demand=deployment.demand,
    )
    report = provider.compatibility(context)
    if not report.compatible:
        detail = "; ".join(
            f"{reason.code}: {reason.message}"
            for reason in report.reasons
            if reason.blocking
        )
        raise RuntimeError(
            "persisted runtime deployment is incompatible with current "
            f"Worker/host facts: {detail}"
        )
    return provider.create_runtime(
        context,
        deployment.setup_intent,
    )


__all__ = [
    "RuntimeDeploymentSpec",
    "build_runtime_deployment_spec",
    "discover_runtime_host_facts",
    "managed_runtime_from_deployment",
    "provider_from_deployment",
]
