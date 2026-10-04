"""Keyboard-first interactive setup wizard over the shared SetupPlan backend."""

from __future__ import annotations

import argparse
import getpass
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol, TypeVar, runtime_checkable

from ..contracts import AcceleratorDevice, ResourceShape, WorkerSpec
from ..control.auth import ClientAuthMode
from ..runtime import (
    ExecutionDemand,
    GPUTopology,
    ModelDemand,
    ModelTopology,
    ResidencyPolicy,
    RuntimeCatalog,
    RuntimeCompatibility,
    RuntimeCompatibilityContext,
    RuntimeProvider,
    RuntimeSelection,
    RuntimeSelectionMode,
    evaluate_catalog,
    evaluate_provider,
)
from ..runtime.providers import (
    ExLlamaMultiGpuMode,
    ExLlamaV3Provider,
    FreeTokenMoeStrategy,
    FreeTokenProvider,
    LlamaCppProvider,
    LlamaCppProviderConfig,
    LlamaCppSplitMode,
    OllamaProvider,
    VllmProvider,
)
from .apply import SetupActionDriver, apply_plan, dry_run_plan, explain_plan
from .contracts import (
    DeploymentPath,
    SetupActionKind,
    SetupActionResult,
    SetupActionState,
    SetupApplyResult,
    SetupApproval,
    SetupHostSnapshot,
    thaw_json,
)
from .discovery import DiscoveredGpu, discover_local_gpus, discover_local_host
from .migration import DEFAULT_RUNTIME_MANIFEST, InstalledWorkerContract
from .planner import build_runtime_setup_plan
from .systemd import (
    GENERIC_SYSTEMD_LLAMA_CPP_EXECUTABLE,
    SystemdSetupDriver,
    create_systemd_driver,
)
from .first_run import (
    ControlBootstrapSpec,
    control_url_for_bind_host,
    FirstRunExecutionMode,
    FirstRunRole,
    FirstRunSecrets,
    SystemdFirstRunInstaller,
    generate_authority_tokens,
    generate_worker_token,
    render_control_env,
    render_nixos_bootstrap_snippet,
    render_worker_env,
    render_worker_toml,
    validate_nixos_runtime_package_expression,
    write_protected_file,
)


@runtime_checkable
class TuiIO(Protocol):
    def write(self, text: str = "") -> None: ...

    def ask(self, prompt: str) -> str: ...

    def ask_secret(self, prompt: str) -> str: ...

    def clear(self) -> None: ...


class ConsoleIO:
    def __init__(self, *, clear_screen: bool = True) -> None:
        self.clear_screen = clear_screen

    def write(self, text: str = "") -> None:
        print(text)

    def ask(self, prompt: str) -> str:
        return input(prompt)

    def ask_secret(self, prompt: str) -> str:
        return getpass.getpass(prompt)

    def clear(self) -> None:
        if self.clear_screen and sys.stdout.isatty():
            print("\033[2J\033[H", end="")


class TuiRunStatus(StrEnum):
    PLANNED = "planned"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    APPLIED = "applied"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TuiRunResult:
    status: TuiRunStatus
    provider_id: str | None = None
    plan_digest: str | None = None
    apply_result: SetupApplyResult | None = None


@dataclass(frozen=True, slots=True)
class RuntimeTuiPlan:
    provider_id: str
    plan: object


@dataclass(frozen=True, slots=True)
class TuiOptionSpec:
    field_name: str
    label: str
    parser: Callable[[str], object]


E = TypeVar("E", bound=StrEnum)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("expected yes/no or true/false")


def _parse_optional_bool(value: str) -> bool | None:
    if value.strip().lower() in {"none", "auto", "default"}:
        return None
    return _parse_bool(value)


def _parse_int(value: str) -> int:
    return int(value.strip())


def _parse_optional_int(value: str) -> int | None:
    if value.strip().lower() in {"none", "auto", "default"}:
        return None
    return _parse_int(value)


def _parse_float(value: str) -> float:
    return float(value.strip())


def _parse_optional_float(value: str) -> float | None:
    if value.strip().lower() in {"none", "auto", "default"}:
        return None
    return _parse_float(value)


def _parse_float_tuple(value: str) -> tuple[float, ...] | None:
    normalized = value.strip()
    if normalized.lower() in {"none", "auto", "default"}:
        return None
    values = tuple(float(item.strip()) for item in normalized.split(",") if item.strip())
    if not values:
        raise ValueError("expected comma-separated numbers")
    return values


def _parse_int_tuple(value: str) -> tuple[int, ...] | None:
    normalized = value.strip()
    if normalized.lower() in {"none", "auto", "default"}:
        return None
    values = tuple(int(item.strip()) for item in normalized.split(",") if item.strip())
    if not values:
        raise ValueError("expected comma-separated integers")
    return values


def _parse_gpu_layers(value: str) -> int | str | None:
    normalized = value.strip().lower()
    if normalized in {"none", "default"}:
        return None
    if normalized in {"auto", "all"}:
        return normalized
    return int(normalized)


def _enum_parser(enum_type: type[E]) -> Callable[[str], E]:
    def parse(value: str) -> E:
        return enum_type(value.strip())

    return parse


def _format_option(value: object) -> str:
    if value is None:
        return "default"
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    if isinstance(value, StrEnum):
        return value.value
    return str(value)


_PROVIDER_OPTIONS: Mapping[str, tuple[TuiOptionSpec, ...]] = {
    "ollama": (
        TuiOptionSpec("keep_alive", "model keep-alive", str),
    ),
    "llama-cpp": (
        TuiOptionSpec("context_size", "context size", _parse_int),
        TuiOptionSpec("gpu_layers", "GPU layers (auto/all/int)", _parse_gpu_layers),
        TuiOptionSpec("split_mode", "multi-GPU split mode", _enum_parser(LlamaCppSplitMode)),
        TuiOptionSpec("tensor_split", "tensor split weights", _parse_float_tuple),
        TuiOptionSpec("fit", "automatic fit", _parse_optional_bool),
        TuiOptionSpec("cpu_moe", "CPU MoE", _parse_bool),
        TuiOptionSpec("n_cpu_moe", "CPU MoE layers", _parse_optional_int),
        TuiOptionSpec("n_cpu_ffn", "CPU FFN layers", _parse_optional_int),
    ),
    "vllm": (
        TuiOptionSpec("gpu_memory_utilization", "GPU memory utilization", _parse_float),
        TuiOptionSpec("tensor_parallel_size", "tensor parallel size", _parse_optional_int),
        TuiOptionSpec("enable_expert_parallel", "expert parallel", _parse_optional_bool),
        TuiOptionSpec("cpu_offload_gb", "CPU offload GiB per GPU", _parse_float),
        TuiOptionSpec("enforce_eager", "enforce eager execution", _parse_bool),
    ),
    "freetoken": (
        TuiOptionSpec("moe_strategy", "MoE strategy", _enum_parser(FreeTokenMoeStrategy)),
        TuiOptionSpec("memory_ratio", "GPU memory ratio", _parse_float),
        TuiOptionSpec("moe_cache_size", "MoE cache slots", _parse_optional_int),
        TuiOptionSpec("moe_cache_rate", "MoE cache fraction", _parse_optional_float),
        TuiOptionSpec("moe_cpu_threads", "MoE CPU threads", _parse_optional_int),
    ),
    "exllamav3": (
        TuiOptionSpec("multi_gpu_mode", "multi-GPU mode", _enum_parser(ExLlamaMultiGpuMode)),
        TuiOptionSpec("tensor_parallel_backend", "tensor-parallel backend", str),
        TuiOptionSpec("gpu_split", "GPU split GiB", _parse_float_tuple),
        TuiOptionSpec("autosplit_reserve_mb", "autosplit reserve MiB", _parse_int_tuple),
        TuiOptionSpec("cache_mode", "cache mode", str),
        TuiOptionSpec("max_seq_len", "maximum sequence length", _parse_optional_int),
        TuiOptionSpec("max_batch_size", "maximum batch size", _parse_optional_int),
    ),
}


def default_runtime_catalog(
    *,
    deployment_path: DeploymentPath | None = None,
) -> RuntimeCatalog:
    llama_cpp = LlamaCppProvider(
        LlamaCppProviderConfig(
            executable=GENERIC_SYSTEMD_LLAMA_CPP_EXECUTABLE,
        )
        if deployment_path is DeploymentPath.SYSTEMD
        else None
    )
    return RuntimeCatalog(
        (
            OllamaProvider(),
            llama_cpp,
            VllmProvider(),
            FreeTokenProvider(),
            ExLlamaV3Provider(),
        )
    )


def _confirm(io: TuiIO, prompt: str, *, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        raw = io.ask(prompt + suffix).strip()
        if not raw:
            return default
        try:
            return _parse_bool(raw)
        except ValueError:
            io.write("Enter yes/no (y/n).")


def _ask_nonblank(io: TuiIO, prompt: str, *, default: str | None = None) -> str:
    while True:
        suffix = f" [{default}]: " if default is not None else ": "
        raw = io.ask(prompt + suffix).strip()
        value = raw or (default or "")
        if value:
            return value
        io.write("Value must not be blank.")


def _ask_secret_nonblank(io: TuiIO, prompt: str) -> str:
    while True:
        raw = io.ask_secret(prompt + ": ").strip()
        if raw:
            return raw
        io.write("Secret value must not be blank.")


def _ask_int(io: TuiIO, prompt: str, *, default: int = 0) -> int:
    while True:
        raw = io.ask(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            io.write("Enter an integer.")
            continue
        if value < 0:
            io.write("Value must not be negative.")
            continue
        return value


def _ask_optional_int(io: TuiIO, prompt: str) -> int | None:
    while True:
        raw = io.ask(f"{prompt} [unknown]: ").strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            io.write("Enter an integer or leave blank.")
            continue
        if value <= 0:
            io.write("Value must be positive.")
            continue
        return value


def _select_gpus(io: TuiIO, gpus: tuple[DiscoveredGpu, ...]) -> tuple[DiscoveredGpu, ...]:
    io.write("Detected GPUs:")
    if not gpus:
        io.write("  none")
        return ()
    for index, gpu in enumerate(gpus, start=1):
        cap = gpu.compute_capability or "unknown"
        device_class = gpu.device_class or "unknown"
        io.write(
            f"  {index}. {gpu.uuid}  {gpu.memory_mb} MiB  "
            f"class={device_class}  compute={cap}"
        )

    while True:
        raw = io.ask("GPU selection [all; comma-separated indexes; none]: ").strip().lower()
        if not raw or raw == "all":
            return gpus
        if raw == "none":
            return ()
        try:
            indexes = [int(item.strip()) for item in raw.split(",")]
        except ValueError:
            io.write("Use GPU indexes such as 1 or 1,2.")
            continue
        if len(indexes) != len(set(indexes)):
            io.write("GPU selection must not contain duplicates.")
            continue
        if any(index < 1 or index > len(gpus) for index in indexes):
            io.write("GPU index is outside the detected range.")
            continue
        return tuple(gpus[index - 1] for index in indexes)


def _minimum_compute_capability(gpus: tuple[DiscoveredGpu, ...]) -> str | None:
    if not gpus or any(gpu.compute_capability is None for gpu in gpus):
        return None
    parsed: list[tuple[int, int, str]] = []
    for gpu in gpus:
        assert gpu.compute_capability is not None
        parts = gpu.compute_capability.split(".", 1)
        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
        except ValueError:
            return None
        parsed.append((major, minor, gpu.compute_capability))
    return min(parsed)[2]


def build_worker_spec(
    *,
    worker_id: str,
    worker_class: str,
    gpus: tuple[DiscoveredGpu, ...],
) -> WorkerSpec:
    total_vram_mb = sum(gpu.memory_mb for gpu in gpus)
    max_vram_mb = max((gpu.memory_mb for gpu in gpus), default=0)
    labels: dict[str, str] = {}
    compute_capability = _minimum_compute_capability(gpus)
    if compute_capability is not None:
        labels["gpu.compute_capability.min"] = compute_capability
    return WorkerSpec(
        worker_id=worker_id,
        worker_class=worker_class,
        resources=ResourceShape(
            gpu_count=len(gpus),
            total_vram_mb=total_vram_mb,
            max_single_gpu_vram_mb=max_vram_mb,
        ),
        gpu_uuids=tuple(gpu.uuid for gpu in gpus),
        accelerators=tuple(
            AcceleratorDevice(
                uuid=gpu.uuid,
                memory_mb=gpu.memory_mb,
                compute_capability=gpu.compute_capability,
                device_class=gpu.device_class,
            )
            for gpu in gpus
        ),
        capabilities=frozenset({"llm.chat", "text.generate"}),
        labels=labels,
    )


def _prompt_worker(io: TuiIO, gpus: tuple[DiscoveredGpu, ...]) -> WorkerSpec:
    default_class = (
        "cpu"
        if not gpus
        else "gpu-single"
        if len(gpus) == 1
        else "gpu-multi"
    )
    worker_id = _ask_nonblank(io, "Worker ID", default="worker-1")
    worker_class = _ask_nonblank(io, "Worker class", default=default_class)
    return build_worker_spec(
        worker_id=worker_id,
        worker_class=worker_class,
        gpus=gpus,
    )


def _prompt_demand(io: TuiIO, worker: WorkerSpec) -> ExecutionDemand:
    model_ref = _ask_nonblank(io, "Model reference")
    model_format = _ask_nonblank(
        io,
        "Model format (safetensors/gguf/exl3/ftw/provider-native)",
        default="safetensors",
    ).lower()

    while True:
        topology_raw = _ask_nonblank(
            io,
            "Model topology (dense/moe)",
            default="dense",
        ).lower()
        try:
            model_topology = ModelTopology(topology_raw)
            break
        except ValueError:
            io.write("Model topology must be dense or moe.")

    while True:
        residency_raw = _ask_nonblank(
            io,
            "Residency policy (vram_only/prefer_vram/cpu_gpu_hybrid)",
            default="prefer_vram",
        ).lower()
        try:
            residency = ResidencyPolicy(residency_raw)
            break
        except ValueError:
            io.write("Unknown residency policy.")

    if worker.resources.gpu_count == 0:
        gpu_topology = GPUTopology.NONE
    elif worker.resources.gpu_count == 1:
        gpu_topology = GPUTopology.SINGLE_GPU
    else:
        gpu_topology = GPUTopology.MULTI_GPU

    estimated_size_mb = _ask_optional_int(io, "Estimated model size MiB")
    min_total_vram_mb = _ask_int(io, "Minimum total VRAM MiB", default=0)
    min_single_gpu_vram_mb = _ask_int(io, "Minimum single-GPU VRAM MiB", default=0)
    min_host_ram_mb = _ask_int(io, "Minimum host RAM MiB", default=0)
    preferred_host_ram_mb = _ask_int(
        io,
        "Preferred host RAM MiB",
        default=min_host_ram_mb,
    )
    if preferred_host_ram_mb < min_host_ram_mb:
        preferred_host_ram_mb = min_host_ram_mb
        io.write("Preferred host RAM raised to the declared minimum.")

    return ExecutionDemand(
        model=ModelDemand(
            model_ref=model_ref,
            model_format=model_format,
            topology=model_topology,
            estimated_size_mb=estimated_size_mb,
        ),
        residency_policy=residency,
        gpu_topology=gpu_topology,
        min_gpu_count=2 if gpu_topology is GPUTopology.MULTI_GPU else 0,
        min_total_vram_mb=min_total_vram_mb,
        min_single_gpu_vram_mb=min_single_gpu_vram_mb,
        min_host_ram_mb=min_host_ram_mb,
        preferred_host_ram_mb=preferred_host_ram_mb,
    )


def _render_report(io: TuiIO, report: RuntimeCompatibility) -> None:
    state = "compatible" if report.compatible else "incompatible"
    io.write(f"  [{state}] {report.provider_id}")
    if not report.reasons:
        io.write("      no compatibility notes")
        return
    for reason in report.reasons:
        severity = "BLOCK" if reason.blocking else "note"
        io.write(f"      {severity}: {reason.code}: {reason.message}")


def _choose_provider(
    io: TuiIO,
    catalog: RuntimeCatalog,
    context: RuntimeCompatibilityContext,
) -> str:
    reports = evaluate_catalog(catalog, context)
    providers = catalog.providers()
    io.write("Runtime candidates:")
    for index, provider in enumerate(providers, start=1):
        io.write(f"{index}. {provider.info.display_name}")
        _render_report(io, reports[provider.info.provider_id])

    while True:
        raw = io.ask("Select runtime by number or provider ID: ").strip()
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(providers):
                return providers[index - 1].info.provider_id
        if catalog.get(raw) is not None:
            return raw
        io.write("Unknown runtime selection.")


def configure_provider(
    io: TuiIO,
    provider: RuntimeProvider,
) -> RuntimeProvider:
    config = getattr(provider, "config", None)
    specs = _PROVIDER_OPTIONS.get(provider.info.provider_id, ())
    if config is None or not specs:
        return provider
    if not _confirm(io, "Customize runtime-specific options?", default=False):
        return provider

    updates: dict[str, object] = {}
    for spec in specs:
        current = getattr(config, spec.field_name)
        while True:
            raw = io.ask(
                f"{spec.label} [{_format_option(current)}]: "
            ).strip()
            if not raw:
                break
            try:
                updates[spec.field_name] = spec.parser(raw)
            except (TypeError, ValueError) as exc:
                io.write(f"Invalid value: {exc}")
                continue
            break

    new_config = replace(config, **updates)
    return type(provider)(new_config)


def _replace_catalog_provider(
    catalog: RuntimeCatalog,
    replacement: RuntimeProvider,
) -> RuntimeCatalog:
    return RuntimeCatalog(
        replacement if provider.info.provider_id == replacement.info.provider_id else provider
        for provider in catalog.providers()
    )


def _render_preview(io: TuiIO, plan, driver: SetupActionDriver) -> bool:
    preview = dry_run_plan(plan, driver)
    io.write("Dry-run:")
    for action in preview.actions:
        detail = f" — {action.detail}" if action.detail else ""
        io.write(
            f"  {action.action_id} {action.state.value}: "
            f"{action.description}{detail}"
        )
    if (
        plan.deployment_path is DeploymentPath.SYSTEMD
        and plan.provider_id == "llama-cpp"
        and any(
            action.kind is SetupActionKind.ENSURE_PACKAGE
            and action.state is not SetupActionState.SATISFIED
            for action in preview.actions
        )
    ):
        io.write(
            "  RuntimeBackend recovery: "
            "docs/installation.md#runtimebackend-nix-profile"
        )
    return not preview.blocked


def _approval(io: TuiIO, plan) -> SetupApproval | None:
    expected = f"APPLY {plan.digest[:12]}"
    typed = io.ask(f"Type '{expected}' to apply this exact plan: ").strip()
    if typed != expected:
        io.write("Apply cancelled; plan was not mutated.")
        return None

    allow_privileged = (
        not plan.requires_privilege
        or _confirm(io, "Approve privileged actions?", default=False)
    )
    allow_network = (
        not plan.requires_network
        or _confirm(io, "Approve network access?", default=False)
    )
    has_download = any(
        action.kind is SetupActionKind.DOWNLOAD_MODEL for action in plan.actions
    )
    allow_model_download = (
        not has_download
        or _confirm(io, "Approve selected-model download?", default=False)
    )
    has_convert = any(
        action.kind is SetupActionKind.CONVERT_MODEL for action in plan.actions
    )
    allow_model_convert = (
        not has_convert
        or _confirm(io, "Approve selected-model conversion?", default=False)
    )
    allow_confirmation_actions = (
        not plan.requires_confirmation
        or _confirm(io, "Approve confirmation-gated actions?", default=False)
    )

    if not all(
        (
            allow_privileged,
            allow_network,
            allow_model_download,
            allow_model_convert,
            allow_confirmation_actions,
        )
    ):
        io.write("One or more required permissions were denied; nothing was applied.")
        return None

    return SetupApproval(
        plan_digest=plan.digest,
        allow_privileged=allow_privileged,
        allow_network=allow_network,
        allow_model_download=allow_model_download,
        allow_model_convert=allow_model_convert,
        allow_confirmation_actions=allow_confirmation_actions,
    )


def recovery_guidance(result: SetupApplyResult) -> tuple[str, ...]:
    if result.succeeded:
        return ("Setup completed successfully.",)
    if any(action.status.value == "rollback_failed" for action in result.rollback_actions):
        return (
            "Rollback was incomplete. Reconcile the reported action before retrying.",
            "Do not switch runtime providers implicitly; keep the explicit selection or edit it.",
        )
    if any(
        action.kind is SetupActionKind.ENSURE_PACKAGE
        and action.status.value in {"blocked", "failed"}
        for action in result.actions
    ):
        return (
            "Runtime package provisioning did not complete.",
            (
                "For generic-systemd use the standard Nix RuntimeBackend path in "
                "docs/installation.md#runtimebackend-nix-profile; custom installer "
                "argv is an advanced override."
            ),
            "No provider substitution was performed.",
        )
    if any(action.status.value == "blocked" for action in result.actions):
        return (
            "Resolve the blocked preflight/action and rerun the same reviewed setup flow.",
            "No provider substitution was performed.",
        )
    return (
        "Review the failed action, correct the host/runtime prerequisite, then rerun.",
        "Applied reversible actions were rolled back where the driver supported it.",
    )


def _render_result(io: TuiIO, result: SetupApplyResult) -> None:
    io.write(f"Apply result: {result.status.value}")
    for action in result.actions:
        detail = f" — {action.detail}" if action.detail else ""
        io.write(f"  {action.action_id} {action.status.value}{detail}")
    for action in result.rollback_actions:
        detail = f" — {action.detail}" if action.detail else ""
        io.write(f"  rollback {action.action_id} {action.status.value}{detail}")
    for line in recovery_guidance(result):
        io.write(line)


def plan_runtime_for_worker(
    *,
    io: TuiIO,
    snapshot: SetupHostSnapshot,
    worker: WorkerSpec,
    catalog: RuntimeCatalog | None = None,
    reconcile_existing_worker: bool = False,
) -> RuntimeTuiPlan | None:
    catalog = catalog or default_runtime_catalog(
        deployment_path=snapshot.deployment_path
    )
    demand = _prompt_demand(io, worker)

    while True:
        context = RuntimeCompatibilityContext(
            worker=worker,
            host=snapshot.runtime_host,
            demand=demand,
        )
        provider_id = _choose_provider(io, catalog, context)
        provider = catalog.get(provider_id)
        assert provider is not None
        configured = configure_provider(io, provider)
        catalog = _replace_catalog_provider(catalog, configured)
        report = evaluate_provider(configured, context)
        io.write("Selected runtime after options:")
        _render_report(io, report)

        if report.compatible:
            break

        choice = _ask_nonblank(
            io,
            "Incompatible: choose action (runtime/demand/options/quit)",
            default="runtime",
        ).lower()
        if choice in {"quit", "q"}:
            return None
        if choice in {"demand", "d"}:
            demand = _prompt_demand(io, worker)
            continue
        if choice in {"options", "o"}:
            configured = configure_provider(io, configured)
            catalog = _replace_catalog_provider(catalog, configured)
            report = evaluate_provider(configured, context)
            if report.compatible:
                provider_id = configured.info.provider_id
                break
            continue

    selection = RuntimeSelection(
        mode=RuntimeSelectionMode.EXPLICIT,
        provider_id=provider_id,
    )
    plan = build_runtime_setup_plan(
        catalog=catalog,
        context=context,
        selection=selection,
        snapshot=snapshot,
        reconcile_existing_worker=reconcile_existing_worker,
    )
    return RuntimeTuiPlan(provider_id=provider_id, plan=plan)


def apply_runtime_tui_plan(
    *,
    io: TuiIO,
    runtime_plan: RuntimeTuiPlan,
    driver: SetupActionDriver | None,
) -> TuiRunResult:
    provider_id = runtime_plan.provider_id
    plan = runtime_plan.plan

    io.write("")
    io.write(explain_plan(plan))
    io.write("No secret values are stored in or printed from the SetupPlan.")

    if driver is None:
        io.write("")
        io.write(
            "Planning-only mode: no deployment SetupActionDriver is connected."
        )
        io.write(
            "The exact plan above can be reviewed now; no install/config/start action ran."
        )
        return TuiRunResult(
            status=TuiRunStatus.PLANNED,
            provider_id=provider_id,
            plan_digest=plan.digest,
        )

    if not _render_preview(io, plan, driver):
        io.write("Dry-run is blocked. Resolve the displayed prerequisite and rerun.")
        return TuiRunResult(
            status=TuiRunStatus.BLOCKED,
            provider_id=provider_id,
            plan_digest=plan.digest,
        )

    approval = _approval(io, plan)
    if approval is None:
        return TuiRunResult(
            status=TuiRunStatus.CANCELLED,
            provider_id=provider_id,
            plan_digest=plan.digest,
        )

    io.write("Applying reviewed plan...")
    result = apply_plan(
        plan,
        driver,
        approval=approval,
        on_result=lambda action: io.write(
            f"  {action.action_id}: {action.status.value}"
        ),
    )
    _render_result(io, result)
    return TuiRunResult(
        status=TuiRunStatus.APPLIED if result.succeeded else TuiRunStatus.FAILED,
        provider_id=provider_id,
        plan_digest=plan.digest,
        apply_result=result,
    )


def _verify_installed_worker_gpus(
    worker: WorkerSpec,
    discovered: tuple[DiscoveredGpu, ...],
) -> None:
    by_uuid = {gpu.uuid: gpu for gpu in discovered}
    if len(by_uuid) != len(discovered):
        raise RuntimeError("local GPU discovery returned duplicate UUIDs")

    selected: list[DiscoveredGpu] = []
    for index, uuid in enumerate(worker.gpu_uuids):
        gpu = by_uuid.get(uuid)
        if gpu is None:
            raise RuntimeError(
                "installed Worker GPU ownership is not present on this host"
            )
        selected.append(gpu)
        if worker.accelerators:
            expected = worker.accelerators[index]
            if gpu.memory_mb != expected.memory_mb:
                raise RuntimeError(
                    "installed Worker per-device VRAM no longer matches discovery"
                )
            if (
                expected.compute_capability is not None
                and gpu.compute_capability != expected.compute_capability
            ):
                raise RuntimeError(
                    "installed Worker compute capability no longer matches discovery"
                )
            if (
                expected.device_class is not None
                and gpu.device_class != expected.device_class
            ):
                raise RuntimeError(
                    "installed Worker device class no longer matches discovery"
                )

    total_vram = sum(gpu.memory_mb for gpu in selected)
    max_vram = max((gpu.memory_mb for gpu in selected), default=0)
    if total_vram != worker.resources.total_vram_mb:
        raise RuntimeError(
            "installed Worker total VRAM no longer matches local GPU discovery"
        )
    if max_vram != worker.resources.max_single_gpu_vram_mb:
        raise RuntimeError(
            "installed Worker maximum single-GPU VRAM no longer matches discovery"
        )


def run_setup_tui(
    *,
    io: TuiIO,
    snapshot: SetupHostSnapshot,
    gpus: tuple[DiscoveredGpu, ...],
    catalog: RuntimeCatalog | None = None,
    driver: SetupActionDriver | None = None,
    existing_worker: InstalledWorkerContract | None = None,
    reconcile_existing_worker: bool = False,
) -> TuiRunResult:
    catalog = catalog or default_runtime_catalog(
        deployment_path=snapshot.deployment_path
    )
    io.clear()
    io.write("AstrumWeaver Worker/runtime setup")
    io.write("=" * 34)
    io.write(
        f"Host: {snapshot.os_id} {snapshot.os_version} | "
        f"{snapshot.runtime_host.cpu_count} CPUs | "
        f"{snapshot.runtime_host.host_ram_mb} MiB RAM | "
        f"{snapshot.deployment_path.value}"
    )

    if existing_worker is None:
        if reconcile_existing_worker:
            raise ValueError(
                "existing Worker reconciliation requires an installed Worker contract"
            )
        selected_gpus = _select_gpus(io, gpus)
        worker = _prompt_worker(io, selected_gpus)
    else:
        if not reconcile_existing_worker:
            raise ValueError(
                "installed Worker contract requires explicit reconciliation mode"
            )
        _verify_installed_worker_gpus(existing_worker.spec, gpus)
        worker = existing_worker.spec
        source_execution = (
            "smoke/debug.echo"
            if existing_worker.execution_mode == "smoke"
            else "runtime/current-provider"
        )
        io.write("")
        io.write(f"Existing Worker: {worker.worker_id}")
        io.write(
            "Execution transition: "
            f"{source_execution} -> runtime/selected-provider"
        )
        io.write("GPU ownership: preserved from installed Worker contract")
        io.write("Control URL: preserved from installed Worker contract")
        io.write("Worker token: preserved by reference; value not read")
        io.write(f"Runtime manifest: {DEFAULT_RUNTIME_MANIFEST}")

    runtime_plan = plan_runtime_for_worker(
        io=io,
        snapshot=snapshot,
        worker=worker,
        catalog=catalog,
        reconcile_existing_worker=reconcile_existing_worker,
    )
    if runtime_plan is None:
        return TuiRunResult(status=TuiRunStatus.CANCELLED)
    if existing_worker is not None:
        source_execution = (
            "smoke/debug.echo"
            if existing_worker.execution_mode == "smoke"
            else "runtime/current-provider"
        )
        io.write(
            "Reviewed execution transition: "
            f"{source_execution} -> runtime/{runtime_plan.provider_id}"
        )
    return apply_runtime_tui_plan(
        io=io,
        runtime_plan=runtime_plan,
        driver=driver,
    )


def _choose_first_run_role(io: TuiIO) -> FirstRunRole:
    io.write("First-run role:")
    io.write("  1. Control")
    io.write("  2. Worker")
    io.write("  3. Control + Worker")
    while True:
        raw = _ask_nonblank(io, "Role", default="3").lower()
        mapping = {
            "1": FirstRunRole.CONTROL,
            "control": FirstRunRole.CONTROL,
            "2": FirstRunRole.WORKER,
            "worker": FirstRunRole.WORKER,
            "3": FirstRunRole.BOTH,
            "both": FirstRunRole.BOTH,
        }
        role = mapping.get(raw)
        if role is not None:
            return role
        io.write("Choose 1/control, 2/worker, or 3/both.")


def _choose_execution_mode(io: TuiIO) -> FirstRunExecutionMode:
    io.write("Worker execution mode:")
    io.write("  1. smoke — built-in debug.echo")
    io.write("  2. runtime — choose a first-class RuntimeProvider")
    while True:
        raw = _ask_nonblank(io, "Execution mode", default="1").lower()
        if raw in {"1", "smoke", "debug", "echo"}:
            return FirstRunExecutionMode.SMOKE
        if raw in {"2", "runtime", "provider"}:
            return FirstRunExecutionMode.RUNTIME
        io.write("Choose 1/smoke or 2/runtime.")


def _prompt_control_spec(io: TuiIO) -> ControlBootstrapSpec:
    bind_host = _ask_nonblank(
        io,
        "Control bind host",
        default="127.0.0.1",
    )
    port = _ask_int(io, "Control port", default=9000)
    if port <= 0 or port > 65535:
        raise ValueError("Control port must be between 1 and 65535")

    while True:
        client_auth_raw = _ask_nonblank(
            io,
            "Client API auth (bearer/none)",
            default="bearer",
        ).lower()
        try:
            client_auth = ClientAuthMode(client_auth_raw)
            break
        except ValueError:
            io.write("Client API auth must be bearer or none.")

    if client_auth is ClientAuthMode.NONE:
        io.write(
            "Client API bearer auth is disabled. Protect job submit/read/cancel "
            "with the deployment network, VPN, reverse proxy, or upstream auth."
        )

    return ControlBootstrapSpec(
        bind_host=bind_host,
        port=port,
        client_auth=client_auth,
    )


def _prompt_control_secrets(
    io: TuiIO,
    *,
    client_auth: ClientAuthMode,
) -> FirstRunSecrets:
    database_url = _ask_secret_nonblank(
        io,
        "PostgreSQL URL (input hidden)",
    )

    if client_auth is ClientAuthMode.NONE:
        if _confirm(io, "Generate new Worker authority token?", default=True):
            worker_token = generate_worker_token()
            io.write(
                "Generated Worker authority token. No Client token is generated "
                "or stored because client_auth=none."
            )
        else:
            worker_token = _ask_secret_nonblank(
                io,
                "Worker authority token (input hidden)",
            )
        client_token = None
    elif _confirm(
        io,
        "Generate new client/Worker authority tokens?",
        default=True,
    ):
        client_token, worker_token = generate_authority_tokens()
        io.write(
            "Generated distinct authority tokens. Values will be written only "
            "to the protected Control environment file."
        )
    else:
        client_token = _ask_secret_nonblank(
            io,
            "Client authority token (input hidden)",
        )
        worker_token = _ask_secret_nonblank(
            io,
            "Worker authority token (input hidden)",
        )

    return FirstRunSecrets(
        database_url=database_url,
        client_token=client_token,
        worker_token=worker_token,
    )


def _runtime_deployment_dict(
    runtime_plan: RuntimeTuiPlan,
) -> dict[str, object]:
    for action in runtime_plan.plan.actions:
        if action.kind is not SetupActionKind.RENDER_CONFIG:
            continue
        deployment = action.payload.get("runtime_deployment")
        if deployment is None:
            continue
        value = thaw_json(deployment)
        if not isinstance(value, dict):
            raise RuntimeError(
                "Runtime SetupPlan deployment manifest is not an object"
            )
        return value
    raise RuntimeError("Runtime SetupPlan lacks runtime deployment manifest")


def _runtime_manifest_json(runtime_plan: RuntimeTuiPlan) -> str:
    return json.dumps(
        _runtime_deployment_dict(runtime_plan),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    ) + "\n"


def _first_run_review_token(
    *,
    role: FirstRunRole,
    control: ControlBootstrapSpec | None,
    worker: WorkerSpec | None,
    control_url: str | None,
    execution_mode: FirstRunExecutionMode | None,
    runtime_plan_digest: str | None = None,
    runtime_package_expression: str | None = None,
) -> str:
    import hashlib

    payload = {
        "role": role.value,
        "control": (
            None
            if control is None
            else {
                "bind_host": control.bind_host,
                "port": control.port,
                "client_auth": control.client_auth.value,
                "worker_ttl_seconds": control.worker_ttl_seconds,
                "lease_seconds": control.lease_seconds,
                "maintenance_interval_seconds": control.maintenance_interval_seconds,
                "access_log": control.access_log,
            }
        ),
        "worker": (
            None
            if worker is None
            else {
                "id": worker.worker_id,
                "class": worker.worker_class,
                "gpu_uuids": list(worker.gpu_uuids),
                "capabilities": sorted(worker.capabilities),
                "resources": {
                    "gpu_count": worker.resources.gpu_count,
                    "total_vram_mb": worker.resources.total_vram_mb,
                    "max_single_gpu_vram_mb": worker.resources.max_single_gpu_vram_mb,
                },
            }
        ),
        "control_url": control_url,
        "execution_mode": None if execution_mode is None else execution_mode.value,
        "runtime_plan_digest": runtime_plan_digest,
        "runtime_package_expression": runtime_package_expression,
        "secret_refs": {
            "database_url": role in {FirstRunRole.CONTROL, FirstRunRole.BOTH},
            "client_token": (
                role in {FirstRunRole.CONTROL, FirstRunRole.BOTH}
                and control is not None
                and control.client_auth is ClientAuthMode.BEARER
            ),
            "worker_token": True,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _render_first_run_review(
    io: TuiIO,
    *,
    deployment_path: DeploymentPath,
    role: FirstRunRole,
    control: ControlBootstrapSpec | None,
    worker: WorkerSpec | None,
    control_url: str | None,
    execution_mode: FirstRunExecutionMode | None,
    digest: str,
    runtime_plan: RuntimeTuiPlan | None = None,
    runtime_package_expression: str | None = None,
) -> None:
    io.write("")
    io.write("First-run review")
    io.write("=" * 16)
    io.write(f"Role: {role.value}")
    if control is not None:
        io.write(f"Control: {control.bind_host}:{control.port}")
        io.write(f"Client API auth: {control.client_auth.value}")
        if control.client_auth is ClientAuthMode.NONE:
            io.write(
                "Client API access control is delegated to the deployment/network boundary."
            )
        io.write("Control actions:")
        if deployment_path is DeploymentPath.NIXOS:
            io.write("  - render reviewed NixOS module snippet")
            io.write("  - optionally write protected /etc/astrumweaver/control.env")
            io.write("  - preserve nixos-rebuild as the mutation boundary")
        else:
            io.write("  - write /etc/astrumweaver/control.toml")
            io.write("  - write protected /etc/astrumweaver/control.env")
            io.write("  - run astrumweaver-migrate explicitly")
            io.write("  - enable/start astrumweaver-control.service")
            io.write("  - wait for /v1/ready")
    if worker is not None:
        io.write(
            f"Worker: {worker.worker_id} | {worker.worker_class} | "
            f"{len(worker.gpu_uuids)} GPU(s)"
        )
        io.write(f"Control URL: {control_url}")
        io.write(f"Execution: {execution_mode.value if execution_mode else 'none'}")
        io.write("Capabilities: " + ", ".join(sorted(worker.capabilities)))
        if runtime_plan is not None:
            io.write(f"RuntimeProvider: {runtime_plan.provider_id}")
            io.write(f"Runtime plan digest: {runtime_plan.plan.digest}")
        if runtime_package_expression is not None:
            io.write(f"Nix runtime package: {runtime_package_expression}")
        io.write("Worker actions:")
        if deployment_path is DeploymentPath.NIXOS:
            io.write("  - render reviewed NixOS Worker module snippet")
            io.write("  - optionally write protected /etc/astrumweaver/worker.env")
            if execution_mode is FirstRunExecutionMode.RUNTIME:
                io.write("  - embed reviewed RuntimeProvider settings with explicit package expression")
            io.write("  - preserve nixos-rebuild as the mutation boundary")
        else:
            io.write("  - write /etc/astrumweaver/worker.toml")
            io.write("  - write protected /etc/astrumweaver/worker.env")
            io.write("  - install systemd Worker integration")
            if execution_mode is FirstRunExecutionMode.SMOKE:
                io.write("  - start Worker and wait for registration/readiness")
            else:
                io.write("  - hand off to reviewed RuntimeProvider SetupPlan")
    io.write("Secret values are omitted from this review and digest.")
    io.write(f"First-run digest: {digest}")


def run_first_run_tui(
    *,
    io: TuiIO,
    snapshot: SetupHostSnapshot,
    gpus: tuple[DiscoveredGpu, ...],
    catalog: RuntimeCatalog | None = None,
    packaged_tool_dir: Path | None = None,
) -> TuiRunResult:
    io.clear()
    io.write("AstrumWeaver first-run setup")
    io.write("=" * 27)
    io.write(
        f"Host: {snapshot.os_id} {snapshot.os_version} | "
        f"{snapshot.runtime_host.cpu_count} CPUs | "
        f"{snapshot.runtime_host.host_ram_mb} MiB RAM | "
        f"{snapshot.deployment_path.value}"
    )

    role = _choose_first_run_role(io)
    control_spec: ControlBootstrapSpec | None = None
    secrets = FirstRunSecrets()
    if role in {FirstRunRole.CONTROL, FirstRunRole.BOTH}:
        control_spec = _prompt_control_spec(io)
        secrets = _prompt_control_secrets(
            io,
            client_auth=control_spec.client_auth,
        )

    worker: WorkerSpec | None = None
    control_url: str | None = None
    execution_mode: FirstRunExecutionMode | None = None
    runtime_plan: RuntimeTuiPlan | None = None
    nix_runtime_package_expression: str | None = None

    if role in {FirstRunRole.WORKER, FirstRunRole.BOTH}:
        selected_gpus = _select_gpus(io, gpus)
        if not selected_gpus and snapshot.deployment_path is DeploymentPath.SYSTEMD:
            io.write(
                "Generic systemd first-run Worker setup currently requires "
                "at least one NVIDIA GPU."
            )
            return TuiRunResult(status=TuiRunStatus.BLOCKED)
        worker = _prompt_worker(io, selected_gpus)
        execution_mode = _choose_execution_mode(io)
        if execution_mode is FirstRunExecutionMode.SMOKE:
            worker = replace(
                worker,
                capabilities=frozenset({"debug.echo"}),
            )

        if role is FirstRunRole.BOTH:
            assert control_spec is not None
            control_url = control_url_for_bind_host(
                control_spec.bind_host, control_spec.port
            )
            assert secrets.worker_token is not None
        else:
            control_url = _ask_nonblank(
                io,
                "Control URL",
                default="http://127.0.0.1:9000",
            )
            worker_token = _ask_secret_nonblank(
                io,
                "Worker authority token (input hidden)",
            )
            secrets = FirstRunSecrets(worker_token=worker_token)

        if execution_mode is FirstRunExecutionMode.RUNTIME:
            runtime_plan = plan_runtime_for_worker(
                io=io,
                snapshot=snapshot,
                worker=worker,
                catalog=catalog,
            )
            if runtime_plan is None:
                return TuiRunResult(status=TuiRunStatus.CANCELLED)
            if snapshot.deployment_path is DeploymentPath.NIXOS:
                while True:
                    expression = _ask_nonblank(
                        io,
                        "Nix runtime package attribute path (pkgs.ollama / pkgs.vllm)",
                    )
                    try:
                        nix_runtime_package_expression = (
                            validate_nixos_runtime_package_expression(expression)
                        )
                        break
                    except ValueError as exc:
                        io.write(str(exc))

    digest = _first_run_review_token(
        role=role,
        control=control_spec,
        worker=worker,
        control_url=control_url,
        execution_mode=execution_mode,
        runtime_plan_digest=(
            runtime_plan.plan.digest
            if runtime_plan is not None
            else None
        ),
        runtime_package_expression=nix_runtime_package_expression,
    )
    _render_first_run_review(
        io,
        deployment_path=snapshot.deployment_path,
        role=role,
        control=control_spec,
        worker=worker,
        control_url=control_url,
        execution_mode=execution_mode,
        digest=digest,
        runtime_plan=runtime_plan,
        runtime_package_expression=nix_runtime_package_expression,
    )

    if snapshot.deployment_path is DeploymentPath.NIXOS:
        runtime_deployment = (
            _runtime_deployment_dict(runtime_plan)
            if runtime_plan is not None
            else None
        )
        snippet = render_nixos_bootstrap_snippet(
            role=role,
            control=control_spec,
            worker=worker,
            control_url=control_url,
            execution_mode=execution_mode or FirstRunExecutionMode.SMOKE,
            runtime_deployment=runtime_deployment,
            runtime_package_expression=nix_runtime_package_expression,
        )
        output_path = Path(
            _ask_nonblank(
                io,
                "Write generated Nix module snippet to",
                default="./astrumweaver-first-run.nix",
            )
        )
        expected = f"WRITE {digest[:12]}"
        typed = io.ask(
            f"Type '{expected}' to write the reviewed snippet: "
        ).strip()
        if typed != expected:
            io.write("First-run snippet write cancelled.")
            return TuiRunResult(status=TuiRunStatus.CANCELLED)
        output_path.write_text(snippet, encoding="utf-8")
        io.write(f"Wrote Nix module snippet: {output_path}")
        io.write(
            "NixOS authority boundary preserved: import/review this file and "
            "run nixos-rebuild yourself."
        )
        can_write_env = not hasattr(os, "geteuid") or os.geteuid() == 0
        if _confirm(
            io,
            "Write protected environment files under /etc/astrumweaver now?",
            default=can_write_env,
        ):
            if not can_write_env:
                raise PermissionError(
                    "writing /etc/astrumweaver secret files requires root"
                )
            if role in {FirstRunRole.CONTROL, FirstRunRole.BOTH}:
                write_protected_file(
                    Path("/etc/astrumweaver/control.env"),
                    render_control_env(
                        secrets,
                        client_auth=control_spec.client_auth,
                    ),
                )
                io.write("Wrote protected /etc/astrumweaver/control.env")
            if role in {FirstRunRole.WORKER, FirstRunRole.BOTH}:
                assert secrets.worker_token is not None
                write_protected_file(
                    Path("/etc/astrumweaver/worker.env"),
                    render_worker_env(secrets.worker_token),
                )
                io.write("Wrote protected /etc/astrumweaver/worker.env")
        else:
            io.write(
                "Secret environment files were not written. Create the "
                "environmentFile paths from the generated snippet before rebuild."
            )
        return TuiRunResult(
            status=TuiRunStatus.PLANNED,
            provider_id=(
                runtime_plan.provider_id
                if runtime_plan is not None
                else None
            ),
            plan_digest=digest,
        )

    if snapshot.deployment_path is not DeploymentPath.SYSTEMD:
        io.write(
            "First-run apply is not implemented for this deployment path."
        )
        return TuiRunResult(
            status=TuiRunStatus.BLOCKED,
            plan_digest=digest,
        )

    expected = f"APPLY {digest[:12]}"
    typed = io.ask(
        f"Type '{expected}' to apply this reviewed first-run setup: "
    ).strip()
    if typed != expected:
        io.write("First-run apply cancelled; no host mutation was started.")
        return TuiRunResult(
            status=TuiRunStatus.CANCELLED,
            plan_digest=digest,
        )

    installer = (
        SystemdFirstRunInstaller()
        if packaged_tool_dir is None
        else SystemdFirstRunInstaller(tool_dir=packaged_tool_dir)
    )
    if control_spec is not None:
        io.write("Installing Control...")
        installer.install_control(
            spec=control_spec,
            secrets=secrets,
        )
        io.write("Control migration applied and /v1/ready is healthy.")

    if worker is not None:
        assert control_url is not None
        assert execution_mode is not None
        assert secrets.worker_token is not None
        manifest_json = (
            None
            if runtime_plan is None
            else _runtime_manifest_json(runtime_plan)
        )
        worker_toml = render_worker_toml(
            worker,
            control_url=control_url,
            execution_mode=execution_mode,
        )
        io.write("Installing Worker base configuration...")
        installer.install_worker(
            worker_toml=worker_toml,
            worker_token=secrets.worker_token,
            runtime_manifest_json=manifest_json,
            start=execution_mode is FirstRunExecutionMode.SMOKE,
        )

        if execution_mode is FirstRunExecutionMode.SMOKE:
            io.write("Worker is registered and ready.")
        else:
            assert runtime_plan is not None
            from .systemd import create_systemd_driver

            io.write("Base Worker installed; entering RuntimeProvider apply phase.")
            runtime_result = apply_runtime_tui_plan(
                io=io,
                runtime_plan=runtime_plan,
                driver=create_systemd_driver(),
            )
            if runtime_result.status is not TuiRunStatus.APPLIED:
                return runtime_result

    io.write("")
    io.write("First-run setup completed.")
    if control_spec is not None:
        io.write(
            "Control: systemctl status astrumweaver-control.service"
        )
    if worker is not None:
        io.write(
            "Worker: curl -fsS http://127.0.0.1:9100/health"
        )
    return TuiRunResult(
        status=TuiRunStatus.APPLIED,
        provider_id=(
            runtime_plan.provider_id
            if runtime_plan is not None
            else None
        ),
        plan_digest=digest,
    )


def _load_driver(specifier: str) -> SetupActionDriver:
    module_name, separator, attribute_name = specifier.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("driver must use module:attribute syntax")
    module = importlib.import_module(module_name)
    target = getattr(module, attribute_name)
    if isinstance(target, type):
        driver = target()
    elif isinstance(target, SetupActionDriver):
        driver = target
    elif callable(target):
        driver = target()
    else:
        driver = target
    if not isinstance(driver, SetupActionDriver):
        raise TypeError("driver target does not implement SetupActionDriver")
    return driver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive AstrumWeaver first-run and RuntimeProvider setup wizard"
    )
    parser.add_argument(
        "--mode",
        choices=("first-run", "runtime"),
        default="first-run",
        help=(
            "first-run configures Control/Worker bootstrap; runtime migrates "
            "an installed generic-systemd smoke Worker or plans a runtime on "
            "other deployment paths"
        ),
    )
    parser.add_argument(
        "--driver",
        help=(
            "optional deployment SetupActionDriver as module:attribute; "
            "without it the TUI runs in deterministic planning-only mode"
        ),
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="do not clear the terminal at startup",
    )
    parser.add_argument(
        "--packaged-tool-dir",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--check-packaging",
        action="store_true",
        help=(
            "validate the packaged first-run tool authority without starting "
            "the interactive wizard"
        ),
    )
    return parser


def _check_packaging(tool_dir: Path | None) -> int:
    if tool_dir is None:
        raise ValueError(
            "packaging check requires the package-provided tool authority"
        )
    installer = SystemdFirstRunInstaller(tool_dir=tool_dir)
    names = (
        "astrumweaver-setup-control-plane",
        "astrumweaver-setup-gpu-worker",
        "astrumweaver-control",
        "astrumweaver-worker",
        "astrumweaver-migrate",
        "astrumweaver-runtime-profile",
    )
    print(f"packaged tool authority: {tool_dir}")
    for name in names:
        try:
            resolved = Path(installer._resolve_tool(name))
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} (authority={tool_dir})") from None
        if resolved.parent != tool_dir:
            raise RuntimeError(
                f"packaged command escaped explicit tool authority: {name}"
            )
        subprocess.run(
            [str(resolved), "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"{name}: {resolved}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.check_packaging:
            return _check_packaging(args.packaged_tool_dir)
        snapshot = discover_local_host()
        gpus = discover_local_gpus()
        io = ConsoleIO(clear_screen=not args.no_clear)
        if args.mode == "runtime":
            driver = _load_driver(args.driver) if args.driver else None
            if snapshot.deployment_path is DeploymentPath.SYSTEMD:
                if driver is not None and not isinstance(
                    driver, SystemdSetupDriver
                ):
                    raise TypeError(
                        "generic-systemd runtime migration requires "
                        "SystemdSetupDriver"
                    )
                loader = (
                    driver
                    if isinstance(driver, SystemdSetupDriver)
                    else create_systemd_driver()
                )
                existing_worker = loader.load_installed_worker_contract()
                result = run_setup_tui(
                    io=io,
                    snapshot=snapshot,
                    gpus=gpus,
                    driver=driver,
                    existing_worker=existing_worker,
                    reconcile_existing_worker=True,
                )
            else:
                result = run_setup_tui(
                    io=io,
                    snapshot=snapshot,
                    gpus=gpus,
                    driver=driver,
                )
        else:
            if args.driver:
                raise ValueError(
                    "--driver applies only to --mode runtime; first-run "
                    "selects its deployment integration automatically"
                )
            result = run_first_run_tui(
                io=io,
                snapshot=snapshot,
                gpus=gpus,
                packaged_tool_dir=args.packaged_tool_dir,
            )
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if result.status in {TuiRunStatus.PLANNED, TuiRunStatus.APPLIED}:
        return 0
    if result.status is TuiRunStatus.CANCELLED:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
