"""Shared deterministic runtime setup backend.

Both non-interactive automation and the interactive TUI consume this API.
Deployment-specific mutation is supplied later through SetupActionDriver.
"""

from .apply import (
    SetupActionDriver,
    SetupApprovalError,
    apply_plan,
    dry_run_plan,
    explain_plan,
    preview_plan,
)
from .contracts import (
    ActionInspection,
    ActionReceipt,
    DeploymentPath,
    PrivilegeMode,
    SecretReference,
    SetupAction,
    SetupActionKind,
    SetupActionPreview,
    SetupActionResult,
    SetupActionResultStatus,
    SetupActionState,
    SetupApplyResult,
    SetupApplyStatus,
    SetupApproval,
    SetupHostSnapshot,
    SetupPlan,
    SetupPlanGoal,
    SetupPreview,
)
from .discovery import HostDiscoveryError, discover_local_host
from .planner import (
    SetupPlanningError,
    build_runtime_release_plan,
    build_runtime_setup_plan,
)

__all__ = [
    "ActionInspection",
    "ActionReceipt",
    "DeploymentPath",
    "HostDiscoveryError",
    "PrivilegeMode",
    "SecretReference",
    "SetupAction",
    "SetupActionDriver",
    "SetupActionKind",
    "SetupActionPreview",
    "SetupActionResult",
    "SetupActionResultStatus",
    "SetupActionState",
    "SetupApplyResult",
    "SetupApplyStatus",
    "SetupApproval",
    "SetupApprovalError",
    "SetupHostSnapshot",
    "SetupPlan",
    "SetupPlanGoal",
    "SetupPlanningError",
    "SetupPreview",
    "apply_plan",
    "build_runtime_release_plan",
    "build_runtime_setup_plan",
    "discover_local_host",
    "dry_run_plan",
    "explain_plan",
    "preview_plan",
]
