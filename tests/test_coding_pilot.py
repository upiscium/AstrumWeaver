from __future__ import annotations

from dataclasses import asdict

import pytest

from astrumweaver.validation.coding_pilot import (
    CodingPilotError,
    build_evidence,
    render_coding_pilot_markdown,
)


D = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64
D3 = "sha256:" + "3" * 64
D4 = "sha256:" + "4" * 64
D5 = "sha256:" + "5" * 64
D6 = "sha256:" + "6" * 64


def scope():
    return {
        "astrumweaver_revision": "deadbeef12345678",
        "opencode_version": "1.18.30",
        "provider_adapter": "llama-cpp-chat-v1",
        "runtime_revision": "cafebabe12345678",
        "model_artifact_sha256": D,
        "quantization": "Q4_K_M",
        "tokenizer_artifact_sha256": D2,
        "template_artifact_sha256": D3,
        "deployment_revision": D4,
        "profile_revision": D5,
        "serving_contract_revision": D6,
        "context_tokens": 12288,
        "output_tokens": 1024,
        "request_timeout_seconds": 240,
        "concurrency": 1,
    }


def task(task_id="task-01", kind="discovery"):
    write = kind != "discovery"
    return {
        "task_id": task_id,
        "kind": kind,
        "baseline_digest": D,
        "acceptance_digest": D2,
        "write_task": write,
        "max_paths": 4,
        "max_tool_calls": 12,
    }


def observed(task_id, lane, *, status="accepted"):
    return {
        "task_id": task_id,
        "lane": lane,
        "status": status,
        "attempts": 1,
        "client_tool_completion": "PASS",
        "correctness": "PASS" if status == "accepted" else "NOT_RUN",
        "wall_time_ms": 1000,
        "queue_wait_ms": 10,
        "ttft_ms": None,
        "generation_time_ms": 800,
        "remote_planning_events": None,
        "remote_review_events": None,
        "remote_repair_events": None,
        "escalations": 0,
        "observed_remote_usage_units": None,
        "measurement_gaps": ["ttft", "remote_rework", "remote_usage"],
    }


def not_run(task_id, lane):
    return {
        "task_id": task_id,
        "lane": lane,
        "status": "not_run",
        "attempts": 0,
        "client_tool_completion": "NOT_RUN",
        "correctness": "NOT_RUN",
        "wall_time_ms": None,
        "queue_wait_ms": None,
        "ttft_ms": None,
        "generation_time_ms": None,
        "remote_planning_events": None,
        "remote_review_events": None,
        "remote_repair_events": None,
        "escalations": 0,
        "observed_remote_usage_units": None,
        "measurement_gaps": ["remote_usage", "remote_rework"],
    }


def pilot(*, disposition="experimental"):
    tasks = [
        task("task-01", "discovery"),
        task("task-02", "test_addition"),
        task("task-03", "scoped_fix"),
    ]
    runs = []
    for item in tasks:
        runs.append(not_run(item["task_id"], "direct_remote"))
        runs.append(observed(item["task_id"], "delegated_local"))
    return {
        "schema_version": "coding-pilot-v1",
        "scope": scope(),
        "tasks": tasks,
        "runs": runs,
        "disposition": disposition,
    }


def test_build_evidence_preserves_not_run_and_unknown():
    evidence = build_evidence(pilot())

    assert evidence.disposition == "experimental"
    assert evidence.direct_remote.observed_runs == 0
    assert evidence.direct_remote.not_run == 3
    assert evidence.direct_remote.accepted_rate is None
    assert evidence.delegated_local.observed_runs == 3
    assert evidence.delegated_local.accepted == 3
    assert evidence.delegated_local.accepted_rate == 1
    assert evidence.delegated_local.remote_rework_events_observed is None
    assert evidence.delegated_local.remote_rework_unknown_runs == 3
    assert evidence.private_values_omitted


def test_accepted_disposition_requires_both_lanes_observed():
    with pytest.raises(CodingPilotError, match="both lanes"):
        build_evidence(pilot(disposition="accepted"))


def test_unknown_fields_fail_closed_to_avoid_private_payload_leak():
    value = pilot()
    value["runs"][0]["prompt"] = "private source prompt"

    with pytest.raises(CodingPilotError, match="unsupported fields"):
        build_evidence(value)


def test_accepted_run_requires_correctness_not_only_tool_completion():
    value = pilot()
    local = next(
        run for run in value["runs"] if run["lane"] == "delegated_local"
    )
    local["correctness"] = "FAIL"

    with pytest.raises(CodingPilotError, match="correctness PASS"):
        build_evidence(value)


def test_rendered_evidence_is_metadata_only_and_explicit_about_unknowns():
    evidence = build_evidence(pilot())
    markdown = render_coding_pilot_markdown(evidence)

    assert "private source prompt" not in markdown
    assert "/tmp/" not in markdown
    assert "task-01" in markdown
    assert "direct_remote" in markdown
    assert "delegated_local" in markdown
    assert "NOT_RUN" in markdown
    assert "UNKNOWN" in markdown
    assert "| Disposition | experimental |" in markdown
    assert "remote_usage" in markdown


def test_asdict_output_remains_json_serializable_shape():
    evidence = build_evidence(pilot())
    data = asdict(evidence)
    assert data["scope"]["concurrency"] == 1
    assert data["direct_remote"]["accepted_rate"] is None