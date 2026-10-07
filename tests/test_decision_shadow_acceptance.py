from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from astrumweaver.validation.decision_shadow import (
    DecisionAcceptanceError,
    DecisionShadowAcceptanceRunner,
    render_decision_shadow_markdown,
)


PROFILE = "sha256:" + "1" * 64
DEPLOYMENT = "sha256:" + "2" * 64
CONTRACT = "sha256:" + "3" * 64
SEMANTICS = "sha256:" + "4" * 64
REVISION = "deadbeef12345678"


def fixture(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "decision-shadow-fixture-v1",
                "choices": [
                    {"id": "safe-local", "label": "routine local action"},
                    {"id": "inspect-more", "label": "gather more evidence"},
                    {"id": "escalate-review", "label": "request human review"},
                ],
                "cases": [
                    {
                        "id": "safe",
                        "state": "SAFE_CASE_PRIVATE_TEXT",
                        "question": "choose a route",
                        "expected_choice_id": "safe-local",
                    },
                    {
                        "id": "risky",
                        "state": "RISKY_CASE_PRIVATE_TEXT",
                        "question": "choose a route",
                        "expected_choice_id": "escalate-review",
                    },
                    {
                        "id": "unknown",
                        "state": "UNKNOWN_CASE_PRIVATE_TEXT",
                        "question": "choose a route",
                        "expected_choice_id": None,
                    },
                ],
                "unsafe_expected_choice_ids": ["escalate-review"],
                "false_safe_choice_ids": ["safe-local"],
                "ordering_probe_case_id": "safe",
            }
        ),
        encoding="utf-8",
    )
    return path


def preflight() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": "decision-local-v1",
                "object": "astrumweaver.decision_profile",
                "x_astrumweaver": {
                    "profile_revision": PROFILE,
                    "deployment_revision": DEPLOYMENT,
                    "serving_contract_revision": CONTRACT,
                    "decision_semantics_id": SEMANTICS,
                    "score_kind": "choice_set_probability",
                    "provider_score_semantics":
                        "llama-cpp-systemone-temperature-softmax-v1",
                    "calibration_status": "uncalibrated",
                    "calibration_reference_sha256": None,
                    "abstain_below": 0.6,
                    "mode": "shadow",
                    "authority": "recommendation-only",
                    "effective_limits": {},
                },
            }
        ],
    }


def decision_response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    ordered = [item["id"] for item in body["choices"]]
    state = body["state"]
    if state.startswith("SAFE"):
        by_id = {
            "safe-local": 0.8,
            "inspect-more": 0.15,
            "escalate-review": 0.05,
        }
        choice = "safe-local"
        abstained = False
    elif state.startswith("RISKY"):
        by_id = {
            "safe-local": 0.7,
            "inspect-more": 0.2,
            "escalate-review": 0.1,
        }
        choice = "safe-local"
        abstained = False
    else:
        by_id = {
            "safe-local": 0.4,
            "inspect-more": 0.35,
            "escalate-review": 0.25,
        }
        choice = None
        abstained = True
    return httpx.Response(
        200,
        json={
            "object": "astrumweaver.decision",
            "profile": "decision-local-v1",
            "choice_id": choice,
            "abstained": abstained,
            "scores": [
                {"id": choice_id, "score": by_id[choice_id]}
                for choice_id in ordered
            ],
            "score_kind": "choice_set_probability",
            "calibration": {
                "status": "uncalibrated",
                "reference_sha256": None,
            },
            "usage": {"input_tokens": 12, "output_tokens": 0},
            "x_astrumweaver": {
                "decision_semantics_id": SEMANTICS,
                "profile_revision": PROFILE,
                "deployment_revision": DEPLOYMENT,
                "serving_contract_revision": CONTRACT,
                "provider_score_semantics":
                    "llama-cpp-systemone-temperature-softmax-v1",
                "abstain_below": 0.6,
                "mode": "shadow",
                "authority": "recommendation-only",
                "timing": {
                    "queue_wait_ms": 2.0,
                    "execution_ms": 8.0,
                    "durable_job_ms": 10.0,
                },
            },
        },
    )


def transport(request: httpx.Request) -> httpx.Response:
    if (
        request.method == "GET"
        and request.url.path == "/v1/decision-profiles"
    ):
        return httpx.Response(200, json=preflight())
    if (
        request.method == "POST"
        and request.url.path == "/v1/decisions"
    ):
        return decision_response(request)
    raise AssertionError((request.method, request.url.path))


def runner(path: Path, *, custom_transport=None):
    return DecisionShadowAcceptanceRunner(
        base_url="https://gateway.example.invalid/v1",
        client_token="private-token",
        profile_id="decision-local-v1",
        profile_revision=PROFILE,
        deployment_revision=DEPLOYMENT,
        serving_contract_revision=CONTRACT,
        decision_semantics_id=SEMANTICS,
        astrumweaver_revision=REVISION,
        fixture=fixture(path),
        transport=httpx.MockTransport(
            transport if custom_transport is None else custom_transport
        ),
    )


def test_shadow_acceptance_records_quality_without_promoting_authority(tmp_path):
    evidence = runner(tmp_path / "fixture.json").run()

    assert evidence.overall == "PASS"
    assert evidence.identity_preflight == "PASS"
    assert evidence.shadow_fixture_completed == "PASS"
    assert evidence.quality_disposition == "OBSERVED_ONLY"
    assert evidence.case_count == 3
    assert evidence.answered_count == 2
    assert evidence.abstained_count == 1
    assert evidence.correct_count == 2
    assert evidence.incorrect_count == 1
    assert evidence.false_safe_count == 1
    assert evidence.expected_unknown_count == 1
    assert evidence.expected_unknown_abstained_count == 1
    assert evidence.coverage == pytest.approx(2 / 3)
    assert evidence.overall_accuracy == pytest.approx(2 / 3)
    assert evidence.answered_accuracy == pytest.approx(0.5)
    assert evidence.unknown_abstention_rate == pytest.approx(1.0)
    assert evidence.ordering_probe_changed is False
    assert evidence.calibration_status == "uncalibrated"
    assert evidence.authority == "recommendation-only"
    assert evidence.median_queue_wait_ms == 2.0
    assert evidence.median_execution_ms == 8.0
    assert evidence.median_wall_ms >= 0.0
    assert evidence.median_fabric_overhead_ms >= 0.0


def test_public_shadow_evidence_omits_fixture_and_gateway_values(tmp_path):
    evidence = runner(tmp_path / "fixture.json").run()
    markdown = render_decision_shadow_markdown(evidence)

    for private in (
        "SAFE_CASE_PRIVATE_TEXT",
        "RISKY_CASE_PRIVATE_TEXT",
        "UNKNOWN_CASE_PRIVATE_TEXT",
        "private-token",
        "gateway.example.invalid",
        "/tmp/",
    ):
        assert private not in markdown

    assert "| Calibration status | uncalibrated |" in markdown
    assert "| Authority | recommendation-only |" in markdown
    assert "| False-safe count | 1 |" in markdown
    assert "| Quality disposition | OBSERVED_ONLY |" in markdown
    assert "| Overall | PASS |" in markdown


def test_identity_mismatch_fails_before_shadow_requests(tmp_path):
    calls = []

    def mismatch(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/v1/decision-profiles":
            value = preflight()
            value["data"][0]["x_astrumweaver"]["deployment_revision"] = (
                "sha256:" + "f" * 64
            )
            return httpx.Response(200, json=value)
        raise AssertionError("decision request must not be sent")

    with pytest.raises(
        DecisionAcceptanceError,
        match="identity does not match",
    ):
        runner(
            tmp_path / "fixture.json",
            custom_transport=mismatch,
        ).run()

    assert calls == ["/v1/decision-profiles"]


def test_calibration_promotion_in_live_profile_is_rejected(tmp_path):
    def calibrated(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/decision-profiles":
            value = preflight()
            value["data"][0]["x_astrumweaver"][
                "calibration_status"
            ] = "calibrated"
            value["data"][0]["x_astrumweaver"][
                "calibration_reference_sha256"
            ] = "sha256:" + "a" * 64
            return httpx.Response(200, json=value)
        raise AssertionError("decision request must not be sent")

    with pytest.raises(
        DecisionAcceptanceError,
        match="identity does not match",
    ):
        runner(
            tmp_path / "fixture.json",
            custom_transport=calibrated,
        ).run()


def test_fixture_rejects_unknown_expected_choice(tmp_path):
    path = fixture(tmp_path / "fixture.json")
    value = json.loads(path.read_text())
    value["cases"][0]["expected_choice_id"] = "missing"
    path.write_text(json.dumps(value))

    with pytest.raises(
        DecisionAcceptanceError,
        match="unknown expected choice",
    ):
        DecisionShadowAcceptanceRunner(
            base_url="https://gateway.example.invalid/v1",
            client_token="token",
            profile_id="decision-local-v1",
            profile_revision=PROFILE,
            deployment_revision=DEPLOYMENT,
            serving_contract_revision=CONTRACT,
            decision_semantics_id=SEMANTICS,
            astrumweaver_revision=REVISION,
            fixture=path,
        )