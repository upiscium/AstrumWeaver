"""Private-safe bounded coding-pilot accounting for #97."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hardware import REVISION_PATTERN


SCHEMA_VERSION = "coding-pilot-v1"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ALLOWED_TASK_KINDS = frozenset({"discovery", "test_addition", "scoped_fix"})
_ALLOWED_LANES = frozenset({"direct_remote", "delegated_local"})
_ALLOWED_STATUSES = frozenset(
    {"accepted", "rejected", "failed", "escalated", "not_run"}
)
_ALLOWED_RESULTS = frozenset({"PASS", "FAIL", "NOT_RUN"})
_ALLOWED_DISPOSITIONS = frozenset({"accepted", "experimental", "rejected"})
_ALLOWED_GAPS = frozenset(
    {
        "remote_usage",
        "remote_rework",
        "queue_wait",
        "ttft",
        "generation_time",
        "electrical_cost",
    }
)


class CodingPilotError(RuntimeError):
    """Pilot input is incomplete, unsafe, or internally inconsistent."""


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CodingPilotError(f"{name} must be an object")
    return dict(value)


def _exact_keys(
    data: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str] = set(),
    name: str,
) -> None:
    actual = set(data)
    missing = required - actual
    extra = actual - required - optional
    if missing:
        raise CodingPilotError(
            f"{name} is missing required fields: {', '.join(sorted(missing))}"
        )
    if extra:
        raise CodingPilotError(
            f"{name} contains unsupported fields: {', '.join(sorted(extra))}"
        )


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise CodingPilotError(f"{name} is invalid")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise CodingPilotError(f"{name} must be a sha256 digest")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CodingPilotError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CodingPilotError(f"{name} must be a non-negative integer")
    return value


def _optional_metric(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CodingPilotError(f"{name} must be numeric or null")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise CodingPilotError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class CodingPilotScope:
    astrumweaver_revision: str
    opencode_version: str
    provider_adapter: str
    runtime_revision: str
    model_artifact_sha256: str
    quantization: str
    tokenizer_artifact_sha256: str
    template_artifact_sha256: str
    deployment_revision: str
    profile_revision: str
    serving_contract_revision: str
    context_tokens: int
    output_tokens: int
    request_timeout_seconds: int
    concurrency: int

    @classmethod
    def from_dict(cls, value: object) -> "CodingPilotScope":
        data = _mapping(value, "scope")
        fields = {
            "astrumweaver_revision",
            "opencode_version",
            "provider_adapter",
            "runtime_revision",
            "model_artifact_sha256",
            "quantization",
            "tokenizer_artifact_sha256",
            "template_artifact_sha256",
            "deployment_revision",
            "profile_revision",
            "serving_contract_revision",
            "context_tokens",
            "output_tokens",
            "request_timeout_seconds",
            "concurrency",
        }
        _exact_keys(data, required=fields, name="scope")
        revision = data["astrumweaver_revision"]
        if not isinstance(revision, str) or not REVISION_PATTERN.fullmatch(revision):
            raise CodingPilotError("scope.astrumweaver_revision is invalid")
        runtime_revision = data["runtime_revision"]
        if (
            not isinstance(runtime_revision, str)
            or not REVISION_PATTERN.fullmatch(runtime_revision)
        ):
            raise CodingPilotError("scope.runtime_revision is invalid")
        opencode = _identifier(data["opencode_version"], "scope.opencode_version")
        adapter = _identifier(data["provider_adapter"], "scope.provider_adapter")
        quantization = _identifier(data["quantization"], "scope.quantization")
        context = _positive_int(data["context_tokens"], "scope.context_tokens")
        output = _positive_int(data["output_tokens"], "scope.output_tokens")
        if output > context:
            raise CodingPilotError("scope.output_tokens exceeds context_tokens")
        concurrency = _positive_int(data["concurrency"], "scope.concurrency")
        if concurrency != 1:
            raise CodingPilotError("coding pilot v1 requires concurrency=1")
        return cls(
            astrumweaver_revision=revision,
            opencode_version=opencode,
            provider_adapter=adapter,
            runtime_revision=runtime_revision,
            model_artifact_sha256=_digest(
                data["model_artifact_sha256"], "scope.model_artifact_sha256"
            ),
            quantization=quantization,
            tokenizer_artifact_sha256=_digest(
                data["tokenizer_artifact_sha256"],
                "scope.tokenizer_artifact_sha256",
            ),
            template_artifact_sha256=_digest(
                data["template_artifact_sha256"],
                "scope.template_artifact_sha256",
            ),
            deployment_revision=_digest(
                data["deployment_revision"], "scope.deployment_revision"
            ),
            profile_revision=_digest(
                data["profile_revision"], "scope.profile_revision"
            ),
            serving_contract_revision=_digest(
                data["serving_contract_revision"],
                "scope.serving_contract_revision",
            ),
            context_tokens=context,
            output_tokens=output,
            request_timeout_seconds=_positive_int(
                data["request_timeout_seconds"],
                "scope.request_timeout_seconds",
            ),
            concurrency=concurrency,
        )


@dataclass(frozen=True, slots=True)
class CodingPilotTask:
    task_id: str
    kind: str
    baseline_digest: str
    acceptance_digest: str
    write_task: bool
    max_paths: int
    max_tool_calls: int

    @classmethod
    def from_dict(cls, value: object) -> "CodingPilotTask":
        data = _mapping(value, "task")
        _exact_keys(
            data,
            required={
                "task_id",
                "kind",
                "baseline_digest",
                "acceptance_digest",
                "write_task",
                "max_paths",
                "max_tool_calls",
            },
            name="task",
        )
        kind = data["kind"]
        if not isinstance(kind, str) or kind not in _ALLOWED_TASK_KINDS:
            raise CodingPilotError("task.kind is unsupported")
        write_task = data["write_task"]
        if type(write_task) is not bool:
            raise CodingPilotError("task.write_task must be boolean")
        if kind == "discovery" and write_task:
            raise CodingPilotError("discovery task must be read-only")
        if kind in {"test_addition", "scoped_fix"} and not write_task:
            raise CodingPilotError(f"{kind} must be a write task")
        return cls(
            task_id=_identifier(data["task_id"], "task.task_id"),
            kind=kind,
            baseline_digest=_digest(
                data["baseline_digest"], "task.baseline_digest"
            ),
            acceptance_digest=_digest(
                data["acceptance_digest"], "task.acceptance_digest"
            ),
            write_task=write_task,
            max_paths=_positive_int(data["max_paths"], "task.max_paths"),
            max_tool_calls=_positive_int(
                data["max_tool_calls"], "task.max_tool_calls"
            ),
        )


@dataclass(frozen=True, slots=True)
class CodingPilotRun:
    task_id: str
    lane: str
    status: str
    attempts: int
    client_tool_completion: str
    correctness: str
    wall_time_ms: float | None
    queue_wait_ms: float | None
    ttft_ms: float | None
    generation_time_ms: float | None
    remote_planning_events: int | None
    remote_review_events: int | None
    remote_repair_events: int | None
    escalations: int
    observed_remote_usage_units: float | None
    measurement_gaps: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object) -> "CodingPilotRun":
        data = _mapping(value, "run")
        _exact_keys(
            data,
            required={
                "task_id",
                "lane",
                "status",
                "attempts",
                "client_tool_completion",
                "correctness",
                "wall_time_ms",
                "queue_wait_ms",
                "ttft_ms",
                "generation_time_ms",
                "remote_planning_events",
                "remote_review_events",
                "remote_repair_events",
                "escalations",
                "observed_remote_usage_units",
                "measurement_gaps",
            },
            name="run",
        )
        lane = data["lane"]
        status = data["status"]
        tool = data["client_tool_completion"]
        correctness = data["correctness"]
        if not isinstance(lane, str) or lane not in _ALLOWED_LANES:
            raise CodingPilotError("run.lane is unsupported")
        if not isinstance(status, str) or status not in _ALLOWED_STATUSES:
            raise CodingPilotError("run.status is unsupported")
        for name, result in (
            ("client_tool_completion", tool),
            ("correctness", correctness),
        ):
            if not isinstance(result, str) or result not in _ALLOWED_RESULTS:
                raise CodingPilotError(f"run.{name} is unsupported")
        attempts = _nonnegative_int(data["attempts"], "run.attempts")
        escalations = _nonnegative_int(data["escalations"], "run.escalations")
        raw_gaps = data["measurement_gaps"]
        if (
            not isinstance(raw_gaps, list)
            or any(not isinstance(item, str) for item in raw_gaps)
        ):
            raise CodingPilotError("run.measurement_gaps must be an array of strings")
        gaps = tuple(sorted(set(raw_gaps)))
        if not set(gaps) <= _ALLOWED_GAPS:
            raise CodingPilotError("run.measurement_gaps contains unsupported values")

        optional_counts: dict[str, int | None] = {}
        for field in (
            "remote_planning_events",
            "remote_review_events",
            "remote_repair_events",
        ):
            raw = data[field]
            optional_counts[field] = (
                None if raw is None else _nonnegative_int(raw, f"run.{field}")
            )

        metrics = {
            field: _optional_metric(data[field], f"run.{field}")
            for field in (
                "wall_time_ms",
                "queue_wait_ms",
                "ttft_ms",
                "generation_time_ms",
                "observed_remote_usage_units",
            )
        }

        if status == "not_run":
            if attempts != 0 or tool != "NOT_RUN" or correctness != "NOT_RUN":
                raise CodingPilotError(
                    "not_run requires attempts=0 and NOT_RUN results"
                )
            if any(value is not None for value in metrics.values()):
                raise CodingPilotError("not_run cannot contain measured metrics")
            if any(value is not None for value in optional_counts.values()):
                raise CodingPilotError(
                    "not_run cannot contain remote event observations"
                )
            if escalations != 0:
                raise CodingPilotError("not_run cannot contain escalations")
        else:
            if attempts < 1 or metrics["wall_time_ms"] is None:
                raise CodingPilotError(
                    "observed run requires attempts and wall_time_ms"
                )
        if status == "accepted" and (
            tool != "PASS" or correctness != "PASS"
        ):
            raise CodingPilotError(
                "accepted run requires tool completion and correctness PASS"
            )
        if status == "rejected" and correctness != "FAIL":
            raise CodingPilotError("rejected run requires correctness FAIL")
        if status == "escalated" and escalations < 1:
            raise CodingPilotError("escalated run requires escalations > 0")

        return cls(
            task_id=_identifier(data["task_id"], "run.task_id"),
            lane=lane,
            status=status,
            attempts=attempts,
            client_tool_completion=str(tool),
            correctness=str(correctness),
            wall_time_ms=metrics["wall_time_ms"],
            queue_wait_ms=metrics["queue_wait_ms"],
            ttft_ms=metrics["ttft_ms"],
            generation_time_ms=metrics["generation_time_ms"],
            remote_planning_events=optional_counts["remote_planning_events"],
            remote_review_events=optional_counts["remote_review_events"],
            remote_repair_events=optional_counts["remote_repair_events"],
            escalations=escalations,
            observed_remote_usage_units=metrics["observed_remote_usage_units"],
            measurement_gaps=gaps,
        )


@dataclass(frozen=True, slots=True)
class CodingPilotLaneSummary:
    lane: str
    observed_runs: int
    not_run: int
    accepted: int
    rejected: int
    failed: int
    escalated: int
    accepted_rate: float | None
    median_wall_time_ms: float | None
    median_queue_wait_ms: float | None
    median_ttft_ms: float | None
    median_generation_time_ms: float | None
    remote_rework_events_observed: int | None
    remote_rework_unknown_runs: int
    escalation_events: int
    observed_remote_usage_total: float | None
    remote_usage_unknown_runs: int
    measurement_gaps: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CodingPilotEvidence:
    evidence_version: str
    scope: CodingPilotScope
    tasks: tuple[CodingPilotTask, ...]
    direct_remote: CodingPilotLaneSummary
    delegated_local: CodingPilotLaneSummary
    disposition: str
    private_values_omitted: bool


def _median(values: Sequence[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    return None if not observed else float(statistics.median(observed))


def _lane_summary(lane: str, runs: Sequence[CodingPilotRun]) -> CodingPilotLaneSummary:
    selected = [run for run in runs if run.lane == lane]
    observed = [run for run in selected if run.status != "not_run"]
    counts = {status: sum(run.status == status for run in selected) for status in _ALLOWED_STATUSES}
    remote_rework_values: list[int] = []
    remote_rework_unknown = 0
    remote_usage_values: list[float] = []
    remote_usage_unknown = 0
    gaps: set[str] = set()
    for run in selected:
        gaps.update(run.measurement_gaps)
        if run.status == "not_run":
            continue
        parts = (
            run.remote_planning_events,
            run.remote_review_events,
            run.remote_repair_events,
        )
        if any(part is None for part in parts):
            remote_rework_unknown += 1
        else:
            remote_rework_values.append(sum(int(part) for part in parts))
        if run.observed_remote_usage_units is None:
            remote_usage_unknown += 1
        else:
            remote_usage_values.append(run.observed_remote_usage_units)
    return CodingPilotLaneSummary(
        lane=lane,
        observed_runs=len(observed),
        not_run=counts["not_run"],
        accepted=counts["accepted"],
        rejected=counts["rejected"],
        failed=counts["failed"],
        escalated=counts["escalated"],
        accepted_rate=(
            None
            if not observed
            else counts["accepted"] / len(observed)
        ),
        median_wall_time_ms=_median([run.wall_time_ms for run in observed]),
        median_queue_wait_ms=_median([run.queue_wait_ms for run in observed]),
        median_ttft_ms=_median([run.ttft_ms for run in observed]),
        median_generation_time_ms=_median(
            [run.generation_time_ms for run in observed]
        ),
        remote_rework_events_observed=(
            None if not remote_rework_values else sum(remote_rework_values)
        ),
        remote_rework_unknown_runs=remote_rework_unknown,
        escalation_events=sum(run.escalations for run in observed),
        observed_remote_usage_total=(
            None if not remote_usage_values else sum(remote_usage_values)
        ),
        remote_usage_unknown_runs=remote_usage_unknown,
        measurement_gaps=tuple(sorted(gaps)),
    )


def build_evidence(value: object) -> CodingPilotEvidence:
    data = _mapping(value, "pilot")
    _exact_keys(
        data,
        required={"schema_version", "scope", "tasks", "runs", "disposition"},
        name="pilot",
    )
    if data["schema_version"] != SCHEMA_VERSION:
        raise CodingPilotError("unsupported coding-pilot schema version")
    disposition = data["disposition"]
    if (
        not isinstance(disposition, str)
        or disposition not in _ALLOWED_DISPOSITIONS
    ):
        raise CodingPilotError("pilot.disposition is unsupported")
    raw_tasks = data["tasks"]
    raw_runs = data["runs"]
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise CodingPilotError("pilot.tasks must be a non-empty array")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise CodingPilotError("pilot.runs must be a non-empty array")
    tasks = tuple(CodingPilotTask.from_dict(item) for item in raw_tasks)
    runs = tuple(CodingPilotRun.from_dict(item) for item in raw_runs)
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise CodingPilotError("pilot task IDs must be unique")
    task_set = set(task_ids)
    if any(run.task_id not in task_set for run in runs):
        raise CodingPilotError("pilot run references an unknown task")
    for task_id in task_ids:
        for lane in _ALLOWED_LANES:
            if not any(run.task_id == task_id and run.lane == lane for run in runs):
                raise CodingPilotError(
                    f"task {task_id} is missing an explicit {lane} record"
                )

    direct = _lane_summary("direct_remote", runs)
    local = _lane_summary("delegated_local", runs)
    if disposition == "accepted":
        if direct.not_run or local.not_run:
            raise CodingPilotError(
                "accepted disposition requires both lanes to be fully observed"
            )
        if direct.observed_runs == 0 or local.observed_runs == 0:
            raise CodingPilotError(
                "accepted disposition requires observations in both lanes"
            )
    return CodingPilotEvidence(
        evidence_version=SCHEMA_VERSION,
        scope=CodingPilotScope.from_dict(data["scope"]),
        tasks=tasks,
        direct_remote=direct,
        delegated_local=local,
        disposition=disposition,
        private_values_omitted=True,
    )


def _fmt_metric(value: float | None) -> str:
    return "UNKNOWN" if value is None else f"{value:.2f}"


def _fmt_rate(value: float | None) -> str:
    return "NOT_RUN" if value is None else f"{value:.3f}"


def render_coding_pilot_markdown(evidence: CodingPilotEvidence) -> str:
    scope = evidence.scope
    scope_rows = [
        ("Evidence version", evidence.evidence_version),
        ("AstrumWeaver revision", scope.astrumweaver_revision),
        ("OpenCode version", scope.opencode_version),
        ("Provider adapter", scope.provider_adapter),
        ("Runtime revision", scope.runtime_revision),
        ("Model artifact", scope.model_artifact_sha256),
        ("Quantization", scope.quantization),
        ("Tokenizer artifact", scope.tokenizer_artifact_sha256),
        ("Template artifact", scope.template_artifact_sha256),
        ("Deployment revision", scope.deployment_revision),
        ("Logical profile revision", scope.profile_revision),
        ("Serving contract revision", scope.serving_contract_revision),
        ("Context tokens", str(scope.context_tokens)),
        ("Output tokens", str(scope.output_tokens)),
        ("Request timeout seconds", str(scope.request_timeout_seconds)),
        ("Concurrency", str(scope.concurrency)),
        ("Disposition", evidence.disposition),
        ("Private values omitted", "true"),
    ]
    task_rows = "\n".join(
        "| "
        + " | ".join(
            (
                task.task_id,
                task.kind,
                task.baseline_digest,
                task.acceptance_digest,
                "yes" if task.write_task else "no",
                str(task.max_paths),
                str(task.max_tool_calls),
            )
        )
        + " |"
        for task in evidence.tasks
    )

    def lane_row(summary: CodingPilotLaneSummary) -> str:
        gaps = ",".join(summary.measurement_gaps) or "none"
        rework = (
            "UNKNOWN"
            if summary.remote_rework_events_observed is None
            else str(summary.remote_rework_events_observed)
        )
        usage = _fmt_metric(summary.observed_remote_usage_total)
        return (
            f"| {summary.lane} | {summary.observed_runs} | {summary.not_run} | "
            f"{summary.accepted} | {summary.rejected} | {summary.failed} | "
            f"{summary.escalated} | {_fmt_rate(summary.accepted_rate)} | "
            f"{_fmt_metric(summary.median_wall_time_ms)} | "
            f"{_fmt_metric(summary.median_queue_wait_ms)} | "
            f"{_fmt_metric(summary.median_ttft_ms)} | "
            f"{_fmt_metric(summary.median_generation_time_ms)} | "
            f"{rework} | {summary.remote_rework_unknown_runs} | "
            f"{summary.escalation_events} | {usage} | "
            f"{summary.remote_usage_unknown_runs} | {gaps} |"
        )

    scope_table = "\n".join(f"| {k} | {v} |" for k, v in scope_rows)
    return (
        "# AstrumWeaver Coding Pilot Evidence\n\n"
        "This evidence is intentionally metadata-only. It contains no repository "
        "name, source text, prompt, credential, worktree path, hostname, Worker "
        "identity, GPU identity, or raw tool input/output.\n\n"
        "## Support scope\n\n"
        "| Field | Result |\n| --- | --- |\n"
        f"{scope_table}\n\n"
        "## Bounded tasks\n\n"
        "| Task | Kind | Baseline | Acceptance | Write | Max paths | Max tool calls |\n"
        "| --- | --- | --- | --- | --- | ---: | ---: |\n"
        f"{task_rows}\n\n"
        "## Lane aggregates\n\n"
        "| Lane | Observed | NOT_RUN | Accepted | Rejected | Failed | Escalated | "
        "Accepted rate | Median wall ms | Median queue ms | Median TTFT ms | "
        "Median generation ms | Remote rework events | Rework unknown runs | "
        "Escalation events | Observed remote usage | Usage unknown runs | Gaps |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        f"{lane_row(evidence.direct_remote)}\n"
        f"{lane_row(evidence.delegated_local)}\n\n"
        "Accepted task rate counts only observed runs. NOT_RUN is never treated "
        "as a failure or a success. UNKNOWN metrics remain unknown rather than "
        "being inferred from local token counts.\n"
    )


def write_coding_pilot_evidence(path: Path, evidence: CodingPilotEvidence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_coding_pilot_markdown(evidence), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="astrumweaver-coding-pilot")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("validation/coding-pilot/coding-v1.md"),
    )
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        evidence = build_evidence(payload)
        write_coding_pilot_evidence(args.evidence, evidence)
    except (OSError, json.JSONDecodeError, CodingPilotError, ValueError) as exc:
        parser.exit(1, f"astrumweaver-coding-pilot: {exc}\n")
    print(json.dumps(asdict(evidence), sort_keys=True))


if __name__ == "__main__":
    main()