"""Provider-neutral runtime selection and lifecycle contracts.

Runtime providers manage model-serving runtimes locally on a Worker. The
Control Plane remains unaware of provider brands and implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from ..contracts import WorkerSpec
from ..execution import JobExecutor, ResidencyReport


def _nonblank(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _immutable_mapping(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _nonnegative(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative")
    return value


class RuntimeSelectionMode(StrEnum):
    EXPLICIT = "explicit"
    RECOMMEND = "recommend"


class ResidencyPolicy(StrEnum):
    VRAM_ONLY = "vram_only"
    PREFER_VRAM = "prefer_vram"
    CPU_GPU_HYBRID = "cpu_gpu_hybrid"


class GPUTopology(StrEnum):
    NONE = "none"
    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu"


class ModelTopology(StrEnum):
    DENSE = "dense"
    MOE = "moe"


class ModelPreparationPolicy(StrEnum):
    REFERENCE_ONLY = "reference_only"
    DOWNLOAD = "download"
    CONVERT = "convert"


class RuntimeHealthState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    """Operator intent for runtime choice.

    EXPLICIT is authoritative. RECOMMEND asks for compatible candidates but
    deliberately leaves selected_provider_id unresolved.
    """

    mode: RuntimeSelectionMode
    provider_id: str | None = None

    def __post_init__(self) -> None:
        mode = RuntimeSelectionMode(self.mode)
        object.__setattr__(self, "mode", mode)

        if mode is RuntimeSelectionMode.EXPLICIT:
            if self.provider_id is None:
                raise ValueError("explicit runtime selection requires provider_id")
            object.__setattr__(
                self,
                "provider_id",
                _nonblank(self.provider_id, "provider_id"),
            )
            return

        if self.provider_id is not None:
            raise ValueError(
                "recommend runtime selection must not preselect provider_id"
            )


@dataclass(frozen=True, slots=True)
class ModelDemand:
    """Model facts relevant to runtime compatibility."""

    model_ref: str
    model_format: str
    topology: ModelTopology
    estimated_size_mb: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_ref",
            _nonblank(self.model_ref, "model_ref"),
        )
        object.__setattr__(
            self,
            "model_format",
            _nonblank(self.model_format, "model_format").lower(),
        )
        object.__setattr__(self, "topology", ModelTopology(self.topology))
        if self.estimated_size_mb is not None:
            _nonnegative(self.estimated_size_mb, "estimated_size_mb")
            if self.estimated_size_mb == 0:
                raise ValueError("estimated_size_mb must be positive when set")
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ExecutionDemand:
    """Runtime-facing execution policy independent from provider brands."""

    model: ModelDemand
    residency_policy: ResidencyPolicy
    gpu_topology: GPUTopology
    min_gpu_count: int = 0
    min_total_vram_mb: int = 0
    min_single_gpu_vram_mb: int = 0
    min_host_ram_mb: int = 0
    preferred_host_ram_mb: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelDemand):
            raise TypeError("model must be ModelDemand")
        residency_policy = ResidencyPolicy(self.residency_policy)
        gpu_topology = GPUTopology(self.gpu_topology)
        object.__setattr__(self, "residency_policy", residency_policy)
        object.__setattr__(self, "gpu_topology", gpu_topology)

        for name in (
            "min_gpu_count",
            "min_total_vram_mb",
            "min_single_gpu_vram_mb",
            "min_host_ram_mb",
            "preferred_host_ram_mb",
        ):
            _nonnegative(getattr(self, name), name)

        if self.preferred_host_ram_mb < self.min_host_ram_mb:
            raise ValueError(
                "preferred_host_ram_mb must be >= min_host_ram_mb"
            )

        if gpu_topology is GPUTopology.NONE:
            if (
                self.min_gpu_count != 0
                or self.min_total_vram_mb != 0
                or self.min_single_gpu_vram_mb != 0
            ):
                raise ValueError(
                    "GPU requirements must be zero when gpu_topology=none"
                )
            if residency_policy is ResidencyPolicy.VRAM_ONLY:
                raise ValueError(
                    "vram_only residency requires a GPU topology"
                )
        elif gpu_topology is GPUTopology.SINGLE_GPU:
            if self.min_gpu_count not in (0, 1):
                raise ValueError(
                    "single_gpu demand cannot require more than one GPU"
                )
        elif gpu_topology is GPUTopology.MULTI_GPU:
            if self.min_gpu_count < 2:
                raise ValueError(
                    "multi_gpu demand requires min_gpu_count >= 2"
                )

        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    @property
    def effective_min_gpu_count(self) -> int:
        if self.gpu_topology is GPUTopology.NONE:
            return 0
        if self.gpu_topology is GPUTopology.SINGLE_GPU:
            return 1
        return self.min_gpu_count


@dataclass(frozen=True, slots=True)
class RuntimeHostFacts:
    """Host facts discovered outside Control and supplied to runtime planning."""

    cpu_count: int
    host_ram_mb: int
    architecture: str = "x86_64"
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonnegative(self.cpu_count, "cpu_count")
        _nonnegative(self.host_ram_mb, "host_ram_mb")
        if self.cpu_count == 0:
            raise ValueError("cpu_count must be positive")
        if self.host_ram_mb == 0:
            raise ValueError("host_ram_mb must be positive")
        object.__setattr__(
            self,
            "architecture",
            _nonblank(self.architecture, "architecture"),
        )
        normalized = {
            _nonblank(key, "label key"): _nonblank(value, "label value")
            for key, value in dict(self.labels).items()
        }
        object.__setattr__(self, "labels", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class RuntimeCompatibilityContext:
    worker: WorkerSpec
    host: RuntimeHostFacts
    demand: ExecutionDemand

    def __post_init__(self) -> None:
        if not isinstance(self.worker, WorkerSpec):
            raise TypeError("worker must be WorkerSpec")
        if not isinstance(self.host, RuntimeHostFacts):
            raise TypeError("host must be RuntimeHostFacts")
        if not isinstance(self.demand, ExecutionDemand):
            raise TypeError("demand must be ExecutionDemand")


@dataclass(frozen=True, slots=True)
class RuntimeProviderInfo:
    provider_id: str
    display_name: str
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_id",
            _nonblank(self.provider_id, "provider_id"),
        )
        object.__setattr__(
            self,
            "display_name",
            _nonblank(self.display_name, "display_name"),
        )
        object.__setattr__(self, "description", str(self.description).strip())


@dataclass(frozen=True, slots=True)
class CompatibilityReason:
    code: str
    message: str
    blocking: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _nonblank(self.code, "code"))
        object.__setattr__(
            self,
            "message",
            _nonblank(self.message, "message"),
        )


@dataclass(frozen=True, slots=True)
class RuntimeCompatibility:
    provider_id: str
    reasons: tuple[CompatibilityReason, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_id",
            _nonblank(self.provider_id, "provider_id"),
        )
        reasons = tuple(self.reasons)
        if not all(isinstance(reason, CompatibilityReason) for reason in reasons):
            raise TypeError("reasons must contain CompatibilityReason values")
        object.__setattr__(self, "reasons", reasons)

    @property
    def compatible(self) -> bool:
        return not any(reason.blocking for reason in self.reasons)


@dataclass(frozen=True, slots=True)
class RuntimeSetupIntent:
    """Provider-authored setup intent consumed by the shared setup backend.

    Issue #24 turns this declarative intent into a deterministic SetupPlan.
    No action is executed merely by constructing this value.
    """

    provider_id: str
    package_references: tuple[str, ...] = ()
    configuration: Mapping[str, Any] = field(default_factory=dict)
    model_preparation: ModelPreparationPolicy = (
        ModelPreparationPolicy.REFERENCE_ONLY
    )
    model_ref: str | None = None
    requires_privilege: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_id",
            _nonblank(self.provider_id, "provider_id"),
        )
        packages = tuple(
            _nonblank(value, "package reference")
            for value in self.package_references
        )
        if len(set(packages)) != len(packages):
            raise ValueError("package_references must not contain duplicates")
        object.__setattr__(self, "package_references", packages)
        object.__setattr__(
            self,
            "configuration",
            _immutable_mapping(self.configuration),
        )
        object.__setattr__(
            self,
            "model_preparation",
            ModelPreparationPolicy(self.model_preparation),
        )
        if self.model_ref is not None:
            object.__setattr__(
                self,
                "model_ref",
                _nonblank(self.model_ref, "model_ref"),
            )
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class RuntimeHealth:
    state: RuntimeHealthState
    ready: bool
    detail: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", RuntimeHealthState(self.state))
        if self.detail is not None:
            object.__setattr__(self, "detail", str(self.detail).strip())
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@runtime_checkable
class ManagedRuntime(Protocol):
    """One locally materialized runtime instance owned by a Worker."""

    @property
    def provider_id(self) -> str: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def health(self) -> RuntimeHealth: ...

    async def residency(self) -> ResidencyReport: ...

    def executor(self) -> JobExecutor: ...

    async def release(self) -> None:
        """Release runtime/model resources after stop/drain."""
        ...


@runtime_checkable
class RuntimeProvider(Protocol):
    """Provider contract implemented by Ollama/llama.cpp/vLLM/etc."""

    @property
    def info(self) -> RuntimeProviderInfo: ...

    def compatibility(
        self,
        context: RuntimeCompatibilityContext,
    ) -> RuntimeCompatibility: ...

    def setup_intent(
        self,
        context: RuntimeCompatibilityContext,
    ) -> RuntimeSetupIntent: ...

    def create_runtime(
        self,
        context: RuntimeCompatibilityContext,
        setup: RuntimeSetupIntent,
    ) -> ManagedRuntime: ...
