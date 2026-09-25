"""Keyboard-first interactive setup wizard over the shared SetupPlan backend."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol, TypeVar, runtime_checkable

from ..contracts import ResourceShape, WorkerSpec
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
    LlamaCppSplitMode,
    OllamaProvider,
    VllmProvider,
)
from .apply import SetupActionDriver, apply_plan, dry_run_plan, explain_plan
from .contracts import (
    SetupActionKind,
    SetupActionResult,
    SetupApplyResult,
    SetupApproval,
    SetupHostSnapshot,
)
from .discovery import DiscoveredGpu, discover_local_gpus, discover_local_host
from .planner import build_runtime_setup_plan


@runtime_checkable
class TuiIO(Protocol):
    def write(self, text: str = "") -> None: ...

    def ask(self, prompt: str) -> str: ...

    def clear(self) -> None: ...


class ConsoleIO:
    def __init__(self, *, clear_screen: bool = True) -> None:
        self.clear_screen = clear_screen

    def write(self, text: str = "") -> None:
        print(text)

    def ask(self, prompt: str) -> str:
        return input(prompt)

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


def default_runtime_catalog() -> RuntimeCatalog:
    return RuntimeCatalog(
        (
            OllamaProvider(),
            LlamaCppProvider(),
            VllmProvider(),
            FreeTokenProvider(),
            ExLlamaV3Provider(),
        )
    )


def _confirm(io: TuiIO, prompt: str, *, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    raw = io.ask(prompt + suffix).strip()
    if not raw:
        return default
    return _parse_bool(raw)


def _ask_nonblank(io: TuiIO, prompt: str, *, default: str | None = None) -> str:
    while True:
        suffix = f" [{default}]: " if default is not None else ": "
        raw = io.ask(prompt + suffix).strip()
        value = raw or (default or "")
        if value:
            return value
        io.write("Value must not be blank.")


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
        io.write(
            f"  {index}. {gpu.uuid}  {gpu.memory_mb} MiB  compute={cap}"
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


def run_setup_tui(
    *,
    io: TuiIO,
    snapshot: SetupHostSnapshot,
    gpus: tuple[DiscoveredGpu, ...],
    catalog: RuntimeCatalog | None = None,
    driver: SetupActionDriver | None = None,
) -> TuiRunResult:
    catalog = catalog or default_runtime_catalog()
    io.clear()
    io.write("AstrumWeaver Worker/runtime setup")
    io.write("=" * 34)
    io.write(
        f"Host: {snapshot.os_id} {snapshot.os_version} | "
        f"{snapshot.runtime_host.cpu_count} CPUs | "
        f"{snapshot.runtime_host.host_ram_mb} MiB RAM | "
        f"{snapshot.deployment_path.value}"
    )

    selected_gpus = _select_gpus(io, gpus)
    worker = _prompt_worker(io, selected_gpus)
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
            return TuiRunResult(status=TuiRunStatus.CANCELLED)
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
        # default: return to chooser

    selection = RuntimeSelection(
        mode=RuntimeSelectionMode.EXPLICIT,
        provider_id=provider_id,
    )
    plan = build_runtime_setup_plan(
        catalog=catalog,
        context=context,
        selection=selection,
        snapshot=snapshot,
    )

    io.write("")
    io.write(explain_plan(plan))
    io.write("No secret values are stored in or printed from the SetupPlan.")

    if driver is None:
        io.write("")
        io.write(
            "Planning-only mode: no deployment SetupActionDriver is connected. "
            "Issue #31 supplies the NixOS/systemd mutation driver."
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
        description="Interactive AstrumWeaver Worker/runtime setup wizard"
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        snapshot = discover_local_host()
        gpus = discover_local_gpus()
        driver = _load_driver(args.driver) if args.driver else None
        result = run_setup_tui(
            io=ConsoleIO(clear_screen=not args.no_clear),
            snapshot=snapshot,
            gpus=gpus,
            driver=driver,
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
