"""Dry-run, approval and apply coordination for SetupPlan."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from .contracts import (
    ActionInspection,
    ActionReceipt,
    SetupAction,
    SetupActionKind,
    SetupActionPreview,
    SetupActionResult,
    SetupActionResultStatus,
    SetupActionState,
    SetupApplyResult,
    SetupApplyStatus,
    SetupApproval,
    SetupPlan,
    SetupPreview,
)


class SetupApprovalError(RuntimeError):
    pass


@runtime_checkable
class SetupActionDriver(Protocol):
    """Deployment-specific action implementation supplied by #31."""

    def inspect(self, action: SetupAction) -> ActionInspection:
        """Return whether this desired action is already satisfied."""
        ...

    def apply(self, action: SetupAction) -> ActionReceipt:
        """Apply one action. Must be idempotent with inspect()."""
        ...

    def rollback(
        self,
        action: SetupAction,
        receipt: ActionReceipt,
    ) -> ActionReceipt:
        """Best-effort rollback using opaque receipt.rollback_data."""
        ...


def preview_plan(
    plan: SetupPlan,
    driver: SetupActionDriver,
) -> SetupPreview:
    previews: list[SetupActionPreview] = []
    for action in plan.actions:
        try:
            inspection = driver.inspect(action)
        except Exception as exc:
            inspection = ActionInspection(
                state=SetupActionState.BLOCKED,
                detail=f"inspection failed: {type(exc).__name__}",
            )
        previews.append(
            SetupActionPreview(
                action_id=action.action_id,
                kind=action.kind,
                state=inspection.state,
                description=action.description,
                requires_privilege=action.requires_privilege,
                requires_network=action.requires_network,
                requires_confirmation=action.requires_confirmation,
                detail=inspection.detail,
            )
        )

    return SetupPreview(
        plan_digest=plan.digest,
        actions=tuple(previews),
    )


def dry_run_plan(
    plan: SetupPlan,
    driver: SetupActionDriver,
) -> SetupPreview:
    """Inspect the exact plan without mutating the target."""
    return preview_plan(plan, driver)


def explain_plan(plan: SetupPlan) -> str:
    lines = [
        f"SetupPlan {plan.digest}",
        f"provider: {plan.provider_id}",
        f"deployment: {plan.deployment_path.value}",
        f"goal: {plan.goal.value}",
        "actions:",
    ]
    for action in plan.actions:
        flags: list[str] = []
        if action.requires_privilege:
            flags.append("privileged")
        if action.requires_network:
            flags.append("network")
        if action.requires_confirmation:
            flags.append("confirmation")
        if action.reversible:
            flags.append("reversible")
        suffix = f" [{' '.join(flags)}]" if flags else ""
        lines.append(
            f"  {action.action_id}: {action.kind.value} - "
            f"{action.description}{suffix}"
        )
    return "\n".join(lines)


def _validate_approval(
    plan: SetupPlan,
    approval: SetupApproval,
) -> None:
    if approval.plan_digest != plan.digest:
        raise SetupApprovalError(
            "approved plan digest does not match the plan being applied"
        )

    if plan.requires_privilege and not approval.allow_privileged:
        raise SetupApprovalError(
            "plan contains privileged actions but privilege was not approved"
        )
    if plan.requires_network and not approval.allow_network:
        raise SetupApprovalError(
            "plan contains network actions but network access was not approved"
        )
    if plan.requires_confirmation and not approval.allow_confirmation_actions:
        raise SetupApprovalError(
            "plan contains explicit-confirmation actions that were not approved"
        )

    for action in plan.actions:
        if (
            action.kind is SetupActionKind.DOWNLOAD_MODEL
            and not approval.allow_model_download
        ):
            raise SetupApprovalError(
                "model download action was not explicitly approved"
            )
        if (
            action.kind is SetupActionKind.CONVERT_MODEL
            and not approval.allow_model_convert
        ):
            raise SetupApprovalError(
                "model conversion action was not explicitly approved"
            )


def _rollback(
    driver: SetupActionDriver,
    applied: list[tuple[SetupAction, ActionReceipt]],
    *,
    on_result: Callable[[SetupActionResult], None] | None = None,
) -> tuple[SetupActionResult, ...]:
    results: list[SetupActionResult] = []
    for action, receipt in reversed(applied):
        if not action.reversible or not receipt.changed:
            continue
        try:
            rollback_receipt = driver.rollback(action, receipt)
        except Exception as exc:
            result = SetupActionResult(
                action_id=action.action_id,
                kind=action.kind,
                status=SetupActionResultStatus.ROLLBACK_FAILED,
                changed=False,
                detail=f"rollback failed: {type(exc).__name__}",
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
            continue

        result = SetupActionResult(
            action_id=action.action_id,
            kind=action.kind,
            status=SetupActionResultStatus.ROLLED_BACK,
            changed=rollback_receipt.changed,
            detail=rollback_receipt.detail,
            evidence=rollback_receipt.evidence,
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
    return tuple(results)


def apply_plan(
    plan: SetupPlan,
    driver: SetupActionDriver,
    *,
    approval: SetupApproval,
    rollback_on_failure: bool = True,
    on_result: Callable[[SetupActionResult], None] | None = None,
) -> SetupApplyResult:
    """Apply one exact reviewed plan.

    Approval is validated before any mutation occurs. Driver state is then
    inspected immediately before each action so already-satisfied actions are
    skipped on repeat application.
    """

    _validate_approval(plan, approval)

    action_results: list[SetupActionResult] = []
    applied: list[tuple[SetupAction, ActionReceipt]] = []

    for action in plan.actions:
        try:
            inspection = driver.inspect(action)
        except Exception as exc:
            result = SetupActionResult(
                action_id=action.action_id,
                kind=action.kind,
                status=SetupActionResultStatus.FAILED,
                changed=False,
                detail=f"inspection failed: {type(exc).__name__}",
            )
            action_results.append(result)
            if on_result is not None:
                on_result(result)
            rollback = (
                _rollback(driver, applied, on_result=on_result)
                if rollback_on_failure
                else ()
            )
            return SetupApplyResult(
                plan_digest=plan.digest,
                status=SetupApplyStatus.FAILED,
                actions=tuple(action_results),
                rollback_actions=rollback,
            )

        if inspection.state is SetupActionState.SATISFIED:
            result = SetupActionResult(
                action_id=action.action_id,
                kind=action.kind,
                status=SetupActionResultStatus.SKIPPED,
                changed=False,
                detail=inspection.detail or "already satisfied",
            )
            action_results.append(result)
            if on_result is not None:
                on_result(result)
            continue

        if inspection.state is SetupActionState.BLOCKED:
            result = SetupActionResult(
                action_id=action.action_id,
                kind=action.kind,
                status=SetupActionResultStatus.BLOCKED,
                changed=False,
                detail=inspection.detail or "action is blocked",
            )
            action_results.append(result)
            if on_result is not None:
                on_result(result)
            rollback = (
                _rollback(driver, applied, on_result=on_result)
                if rollback_on_failure
                else ()
            )
            return SetupApplyResult(
                plan_digest=plan.digest,
                status=SetupApplyStatus.FAILED,
                actions=tuple(action_results),
                rollback_actions=rollback,
            )

        try:
            receipt = driver.apply(action)
        except Exception as exc:
            result = SetupActionResult(
                action_id=action.action_id,
                kind=action.kind,
                status=SetupActionResultStatus.FAILED,
                changed=False,
                detail=f"apply failed: {type(exc).__name__}",
            )
            action_results.append(result)
            if on_result is not None:
                on_result(result)
            rollback = (
                _rollback(driver, applied, on_result=on_result)
                if rollback_on_failure
                else ()
            )
            return SetupApplyResult(
                plan_digest=plan.digest,
                status=SetupApplyStatus.FAILED,
                actions=tuple(action_results),
                rollback_actions=rollback,
            )

        if receipt.changed:
            status = SetupActionResultStatus.APPLIED
            applied.append((action, receipt))
        else:
            status = SetupActionResultStatus.SKIPPED

        result = SetupActionResult(
            action_id=action.action_id,
            kind=action.kind,
            status=status,
            changed=receipt.changed,
            detail=receipt.detail,
            evidence=receipt.evidence,
        )
        action_results.append(result)
        if on_result is not None:
            on_result(result)

    return SetupApplyResult(
        plan_digest=plan.digest,
        status=SetupApplyStatus.SUCCEEDED,
        actions=tuple(action_results),
        rollback_actions=(),
    )
