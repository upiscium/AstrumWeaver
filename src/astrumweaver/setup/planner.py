"""Deterministic RuntimeSetupIntent -> SetupPlan planning."""

from __future__ import annotations

from dataclasses import dataclass

from ..runtime import (
    ModelPreparationPolicy,
    RuntimeCatalog,
    RuntimeCompatibilityContext,
    RuntimeSelection,
    RuntimeSelectionMode,
    build_runtime_deployment_spec,
    resolve_runtime,
)
from .contracts import (
    SetupAction,
    SetupActionKind,
    SetupHostSnapshot,
    SetupPlan,
    SetupPlanGoal,
)


class SetupPlanningError(RuntimeError):
    pass


@dataclass(slots=True)
class _ActionBuilder:
    actions: list[SetupAction]

    def add(
        self,
        kind: SetupActionKind,
        description: str,
        *,
        payload: dict | None = None,
        requires_privilege: bool = False,
        requires_network: bool = False,
        requires_confirmation: bool = False,
        reversible: bool = False,
    ) -> None:
        index = len(self.actions) + 1
        self.actions.append(
            SetupAction(
                action_id=f"{index:02d}-{kind.value.replace('_', '-')}",
                kind=kind,
                description=description,
                payload=payload or {},
                requires_privilege=requires_privilege,
                requires_network=requires_network,
                requires_confirmation=requires_confirmation,
                reversible=reversible,
            )
        )


def _require_snapshot_matches_context(
    snapshot: SetupHostSnapshot,
    context: RuntimeCompatibilityContext,
) -> None:
    if snapshot.runtime_host != context.host:
        raise SetupPlanningError(
            "host discovery snapshot does not match compatibility context"
        )


def build_runtime_setup_plan(
    *,
    catalog: RuntimeCatalog,
    context: RuntimeCompatibilityContext,
    selection: RuntimeSelection,
    snapshot: SetupHostSnapshot,
) -> SetupPlan:
    """Build an activation plan after the user has made a runtime choice."""

    _require_snapshot_matches_context(snapshot, context)

    if selection.mode is not RuntimeSelectionMode.EXPLICIT:
        raise SetupPlanningError(
            "runtime setup requires an explicit user-selected provider"
        )

    resolution = resolve_runtime(
        catalog=catalog,
        context=context,
        selection=selection,
    )
    provider_id = resolution.selected_provider_id
    if provider_id is None:
        raise SetupPlanningError("runtime provider selection is unresolved")

    provider = catalog.get(provider_id)
    if provider is None:
        raise SetupPlanningError(
            f"selected runtime provider is not installed: {provider_id}"
        )

    intent = provider.setup_intent(context)
    if intent.provider_id != provider_id:
        raise SetupPlanningError(
            "runtime provider setup intent changed provider identity"
        )
    deployment = build_runtime_deployment_spec(
        provider,
        context,
        setup_intent=intent,
    )

    actions: list[SetupAction] = []
    builder = _ActionBuilder(actions)

    for logical_name in ("config", "state", "cache"):
        builder.add(
            SetupActionKind.ENSURE_DIRECTORY,
            f"Ensure runtime {logical_name} directory exists",
            payload={
                "provider_id": provider_id,
                "logical_name": logical_name,
            },
            requires_privilege=intent.requires_privilege,
        )

    for package_reference in sorted(intent.package_references):
        builder.add(
            SetupActionKind.ENSURE_PACKAGE,
            f"Ensure runtime package is installed: {package_reference}",
            payload={
                "provider_id": provider_id,
                "package_reference": package_reference,
            },
            requires_privilege=intent.requires_privilege,
            requires_network=True,
        )

    if intent.configuration:
        builder.add(
            SetupActionKind.RENDER_CONFIG,
            "Render selected runtime configuration",
            payload={
                "provider_id": provider_id,
                "configuration": dict(intent.configuration),
                "runtime_deployment": deployment.to_dict(),
            },
            requires_privilege=intent.requires_privilege,
            reversible=True,
        )

    model_ref = intent.model_ref or context.demand.model.model_ref
    if intent.model_preparation is ModelPreparationPolicy.REFERENCE_ONLY:
        builder.add(
            SetupActionKind.VERIFY_MODEL_REFERENCE,
            "Verify the selected model reference is available",
            payload={
                "provider_id": provider_id,
                "model_ref": model_ref,
                "model_format": context.demand.model.model_format,
            },
        )
    elif intent.model_preparation is ModelPreparationPolicy.DOWNLOAD:
        builder.add(
            SetupActionKind.DOWNLOAD_MODEL,
            "Download the explicitly selected model",
            payload={
                "provider_id": provider_id,
                "model_ref": model_ref,
                "model_format": context.demand.model.model_format,
            },
            requires_network=True,
            requires_confirmation=True,
        )
    elif intent.model_preparation is ModelPreparationPolicy.CONVERT:
        builder.add(
            SetupActionKind.CONVERT_MODEL,
            "Convert the explicitly selected model for this runtime",
            payload={
                "provider_id": provider_id,
                "model_ref": model_ref,
                "model_format": context.demand.model.model_format,
            },
            requires_confirmation=True,
        )
    else:  # pragma: no cover
        raise SetupPlanningError(
            f"unsupported model preparation policy: {intent.model_preparation}"
        )

    builder.add(
        SetupActionKind.PREFLIGHT,
        "Run runtime and model compatibility preflight",
        payload={
            "provider_id": provider_id,
            "residency_policy": context.demand.residency_policy.value,
            "gpu_topology": context.demand.gpu_topology.value,
            "model_topology": context.demand.model.topology.value,
        },
    )
    builder.add(
        SetupActionKind.RUNTIME_START,
        "Start or reconcile the selected runtime",
        payload={"provider_id": provider_id},
        requires_privilege=intent.requires_privilege,
        reversible=True,
    )
    builder.add(
        SetupActionKind.HEALTH_CHECK,
        "Verify the selected runtime is ready",
        payload={"provider_id": provider_id},
    )

    return SetupPlan(
        provider_id=provider_id,
        deployment_path=snapshot.deployment_path,
        goal=SetupPlanGoal.ACTIVATE,
        actions=tuple(actions),
        metadata={
            "selection_mode": selection.mode.value,
            "model_format": context.demand.model.model_format,
            "model_topology": context.demand.model.topology.value,
            "residency_policy": context.demand.residency_policy.value,
            "gpu_topology": context.demand.gpu_topology.value,
            "worker_class": context.worker.worker_class,
            "gpu_count": context.worker.resources.gpu_count,
            "host_privilege_mode": snapshot.privilege_mode.value,
        },
    )


def build_runtime_release_plan(
    *,
    provider_id: str,
    snapshot: SetupHostSnapshot,
    requires_privilege: bool,
) -> SetupPlan:
    """Build a deterministic runtime stop/release plan."""

    actions: list[SetupAction] = []
    builder = _ActionBuilder(actions)
    builder.add(
        SetupActionKind.RUNTIME_STOP,
        "Stop the selected runtime",
        payload={"provider_id": provider_id},
        requires_privilege=requires_privilege,
        reversible=True,
    )
    builder.add(
        SetupActionKind.RUNTIME_RELEASE,
        "Release runtime-local model and accelerator residency",
        payload={"provider_id": provider_id},
        requires_privilege=requires_privilege,
    )

    return SetupPlan(
        provider_id=provider_id,
        deployment_path=snapshot.deployment_path,
        goal=SetupPlanGoal.RELEASE,
        actions=tuple(actions),
        metadata={
            "host_privilege_mode": snapshot.privilege_mode.value,
        },
    )
