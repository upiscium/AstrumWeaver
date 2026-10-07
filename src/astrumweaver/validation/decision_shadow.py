"""Private-safe real shadow acceptance for System-One decision serving."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import httpx

from .hardware import REVISION_PATTERN


_FIXTURE_SCHEMA = "decision-shadow-fixture-v1"
_EVIDENCE_SCHEMA = "decision-shadow-v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class DecisionAcceptanceError(RuntimeError):
    """Shadow acceptance failed without exposing private fixture values."""


@dataclass(frozen=True, slots=True)
class DecisionShadowEvidence:
    evidence_version: str
    date_utc: str
    astrumweaver_revision: str
    profile_id: str
    profile_revision: str
    deployment_revision: str
    serving_contract_revision: str
    decision_semantics_id: str
    score_kind: str
    provider_score_semantics: str
    calibration_status: str
    calibration_reference_sha256: str | None
    abstain_below: float
    mode: str
    authority: str
    fixture_sha256: str
    case_count: int
    answered_count: int
    abstained_count: int
    correct_count: int
    incorrect_count: int
    expected_unknown_count: int
    expected_unknown_abstained_count: int
    false_safe_count: int
    coverage: float
    overall_accuracy: float
    answered_accuracy: float | None
    unknown_abstention_rate: float | None
    ordering_probe_changed: bool
    median_wall_ms: float
    p95_wall_ms: float
    median_queue_wait_ms: float
    median_execution_ms: float
    median_fabric_overhead_ms: float
    identity_preflight: str
    shadow_fixture_completed: str
    private_values_omitted: bool
    quality_disposition: str
    overall: str


@dataclass(frozen=True, slots=True)
class _FixtureCase:
    case_id: str
    state: str
    question: str
    expected_choice_id: str | None


@dataclass(frozen=True, slots=True)
class _Fixture:
    choices: tuple[tuple[str, str], ...]
    cases: tuple[_FixtureCase, ...]
    unsafe_expected_choice_ids: frozenset[str]
    false_safe_choice_ids: frozenset[str]
    ordering_probe_case_id: str
    digest: str


@dataclass(frozen=True, slots=True)
class _DecisionResult:
    choice_id: str | None
    abstained: bool
    scores: tuple[tuple[str, float], ...]
    wall_ms: float
    queue_wait_ms: float
    execution_ms: float
    fabric_overhead_ms: float


def _validate_base_url(value: str) -> str:
    candidate = value.strip().rstrip("/")
    parsed = urlsplit(candidate)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith("/v1")
    ):
        raise DecisionAcceptanceError(
            "ASTRUMWEAVER_DECISION_BASE_URL must be an http(s) URL ending in /v1"
        )
    return candidate


def _require_digest(value: str, name: str) -> str:
    if _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a sha256 digest")
    return value


def _load_fixture(path: Path) -> _Fixture:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DecisionAcceptanceError(
            "private decision fixture could not be loaded"
        ) from exc
    expected_fields = {
        "schema_version",
        "choices",
        "cases",
        "unsafe_expected_choice_ids",
        "false_safe_choice_ids",
        "ordering_probe_case_id",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise DecisionAcceptanceError(
            "private decision fixture uses an unsupported schema"
        )
    if value.get("schema_version") != _FIXTURE_SCHEMA:
        raise DecisionAcceptanceError(
            "private decision fixture schema version is unsupported"
        )

    raw_choices = value.get("choices")
    if not isinstance(raw_choices, list) or len(raw_choices) < 2:
        raise DecisionAcceptanceError(
            "private decision fixture requires at least two choices"
        )
    choices: list[tuple[str, str]] = []
    choice_ids: set[str] = set()
    for item in raw_choices:
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "label"}
            or not isinstance(item.get("id"), str)
            or not item["id"].strip()
            or not isinstance(item.get("label"), str)
            or not item["label"].strip()
        ):
            raise DecisionAcceptanceError(
                "private decision fixture contains an invalid choice"
            )
        choice_id = item["id"].strip()
        if choice_id in choice_ids:
            raise DecisionAcceptanceError(
                "private decision fixture contains duplicate choice IDs"
            )
        choice_ids.add(choice_id)
        choices.append((choice_id, item["label"]))

    raw_cases = value.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) < 3:
        raise DecisionAcceptanceError(
            "private decision fixture requires at least three cases"
        )
    cases: list[_FixtureCase] = []
    case_ids: set[str] = set()
    for item in raw_cases:
        if (
            not isinstance(item, dict)
            or set(item)
            != {"id", "state", "question", "expected_choice_id"}
            or not isinstance(item.get("id"), str)
            or not item["id"].strip()
            or not isinstance(item.get("state"), str)
            or not item["state"].strip()
            or not isinstance(item.get("question"), str)
            or not item["question"].strip()
        ):
            raise DecisionAcceptanceError(
                "private decision fixture contains an invalid case"
            )
        case_id = item["id"].strip()
        if case_id in case_ids:
            raise DecisionAcceptanceError(
                "private decision fixture contains duplicate case IDs"
            )
        case_ids.add(case_id)
        expected = item.get("expected_choice_id")
        if expected is not None and (
            not isinstance(expected, str)
            or expected not in choice_ids
        ):
            raise DecisionAcceptanceError(
                "private decision fixture contains an unknown expected choice"
            )
        cases.append(
            _FixtureCase(
                case_id=case_id,
                state=item["state"],
                question=item["question"],
                expected_choice_id=expected,
            )
        )

    def id_set(field: str) -> frozenset[str]:
        raw = value.get(field)
        if (
            not isinstance(raw, list)
            or any(
                not isinstance(item, str) or item not in choice_ids
                for item in raw
            )
        ):
            raise DecisionAcceptanceError(
                "private decision fixture contains an invalid risk choice set"
            )
        return frozenset(raw)

    unsafe_expected = id_set("unsafe_expected_choice_ids")
    false_safe = id_set("false_safe_choice_ids")
    ordering_probe = value.get("ordering_probe_case_id")
    if not isinstance(ordering_probe, str) or ordering_probe not in case_ids:
        raise DecisionAcceptanceError(
            "private decision fixture ordering probe is invalid"
        )

    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _Fixture(
        choices=tuple(choices),
        cases=tuple(cases),
        unsafe_expected_choice_ids=unsafe_expected,
        false_safe_choice_ids=false_safe,
        ordering_probe_case_id=ordering_probe,
        digest="sha256:" + hashlib.sha256(canonical).hexdigest(),
    )


def _percentile95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("values must not be empty")
    index = math.ceil(0.95 * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


class DecisionShadowAcceptanceRunner:
    def __init__(
        self,
        *,
        base_url: str,
        client_token: str,
        profile_id: str,
        profile_revision: str,
        deployment_revision: str,
        serving_contract_revision: str,
        decision_semantics_id: str,
        astrumweaver_revision: str,
        fixture: Path,
        timeout_seconds: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not client_token:
            raise ValueError("client token must not be blank")
        if _PROFILE_ID.fullmatch(profile_id) is None:
            raise ValueError("profile_id is invalid")
        for name, value in (
            ("profile_revision", profile_revision),
            ("deployment_revision", deployment_revision),
            ("serving_contract_revision", serving_contract_revision),
            ("decision_semantics_id", decision_semantics_id),
        ):
            _require_digest(value, name)
        if REVISION_PATTERN.fullmatch(astrumweaver_revision) is None:
            raise ValueError("astrumweaver_revision is invalid")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")

        self.base_url = _validate_base_url(base_url)
        self.client_token = client_token
        self.profile_id = profile_id
        self.profile_revision = profile_revision
        self.deployment_revision = deployment_revision
        self.serving_contract_revision = serving_contract_revision
        self.decision_semantics_id = decision_semantics_id
        self.astrumweaver_revision = astrumweaver_revision
        self.fixture = _load_fixture(fixture)
        self.timeout_seconds = float(timeout_seconds)
        self._transport = transport

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url + "/",
            headers={
                "authorization": f"Bearer {self.client_token}",
            },
            timeout=self.timeout_seconds,
            transport=self._transport,
        )

    @staticmethod
    def _json(response: httpx.Response, label: str) -> Mapping[str, Any]:
        if response.status_code != 200:
            raise DecisionAcceptanceError(
                f"decision gateway {label} request was rejected"
            )
        try:
            value = response.json()
        except ValueError as exc:
            raise DecisionAcceptanceError(
                f"decision gateway {label} returned invalid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise DecisionAcceptanceError(
                f"decision gateway {label} returned an invalid object"
            )
        return value

    def _preflight(
        self,
        client: httpx.Client,
    ) -> Mapping[str, Any]:
        body = self._json(
            client.get("decision-profiles"),
            "identity preflight",
        )
        data = body.get("data")
        matches = (
            [
                item
                for item in data
                if isinstance(item, Mapping)
                and item.get("id") == self.profile_id
            ]
            if isinstance(data, list)
            else []
        )
        if len(matches) != 1:
            raise DecisionAcceptanceError(
                "decision acceptance profile is not uniquely configured"
            )
        extension = matches[0].get("x_astrumweaver")
        if not isinstance(extension, Mapping):
            raise DecisionAcceptanceError(
                "decision gateway omitted semantics metadata"
            )
        expected = {
            "profile_revision": self.profile_revision,
            "deployment_revision": self.deployment_revision,
            "serving_contract_revision": self.serving_contract_revision,
            "decision_semantics_id": self.decision_semantics_id,
            "score_kind": "choice_set_probability",
            "provider_score_semantics":
                "llama-cpp-systemone-temperature-softmax-v1",
            "calibration_status": "uncalibrated",
            "calibration_reference_sha256": None,
            "mode": "shadow",
            "authority": "recommendation-only",
        }
        for key, expected_value in expected.items():
            if extension.get(key) != expected_value:
                raise DecisionAcceptanceError(
                    "live decision identity does not match the acceptance target"
                )
        threshold = extension.get("abstain_below")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise DecisionAcceptanceError(
                "live decision abstention threshold is invalid"
            )
        return extension

    def _decide(
        self,
        client: httpx.Client,
        *,
        case: _FixtureCase,
        choices: Sequence[tuple[str, str]],
    ) -> _DecisionResult:
        started = time.monotonic()
        response = client.post(
            "decisions",
            json={
                "profile": self.profile_id,
                "profile_revision": self.profile_revision,
                "state": case.state,
                "question": case.question,
                "choices": [
                    {"id": choice_id, "label": label}
                    for choice_id, label in choices
                ],
            },
        )
        wall_ms = (time.monotonic() - started) * 1000.0
        body = self._json(response, "shadow decision")
        if body.get("profile") != self.profile_id:
            raise DecisionAcceptanceError(
                "decision response changed the profile identity"
            )
        if body.get("score_kind") != "choice_set_probability":
            raise DecisionAcceptanceError(
                "decision response changed score semantics"
            )
        abstained = body.get("abstained")
        choice_id = body.get("choice_id")
        if not isinstance(abstained, bool):
            raise DecisionAcceptanceError(
                "decision response omitted explicit abstention"
            )
        valid_ids = {choice for choice, _ in choices}
        if abstained:
            if choice_id is not None:
                raise DecisionAcceptanceError(
                    "abstained decision unexpectedly selected a choice"
                )
        elif not isinstance(choice_id, str) or choice_id not in valid_ids:
            raise DecisionAcceptanceError(
                "decision response selected an unknown choice"
            )

        scores_raw = body.get("scores")
        if not isinstance(scores_raw, list) or len(scores_raw) != len(choices):
            raise DecisionAcceptanceError(
                "decision response returned the wrong score count"
            )
        scores: list[tuple[str, float]] = []
        for expected, item in zip(choices, scores_raw, strict=True):
            if (
                not isinstance(item, Mapping)
                or item.get("id") != expected[0]
            ):
                raise DecisionAcceptanceError(
                    "decision response changed choice ordering"
                )
            raw_score = item.get("score")
            if (
                isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(float(raw_score))
                or not 0.0 <= float(raw_score) <= 1.0
            ):
                raise DecisionAcceptanceError(
                    "decision response returned a non-finite score"
                )
            scores.append((expected[0], float(raw_score)))
        if abs(sum(score for _, score in scores) - 1.0) > 1e-4:
            raise DecisionAcceptanceError(
                "decision response scores are not normalized"
            )

        calibration = body.get("calibration")
        if (
            not isinstance(calibration, Mapping)
            or calibration.get("status") != "uncalibrated"
            or calibration.get("reference_sha256") is not None
        ):
            raise DecisionAcceptanceError(
                "decision response invented calibration evidence"
            )
        extension = body.get("x_astrumweaver")
        if not isinstance(extension, Mapping):
            raise DecisionAcceptanceError(
                "decision response omitted AstrumWeaver metadata"
            )
        expected_extension = {
            "decision_semantics_id": self.decision_semantics_id,
            "profile_revision": self.profile_revision,
            "deployment_revision": self.deployment_revision,
            "serving_contract_revision": self.serving_contract_revision,
            "provider_score_semantics":
                "llama-cpp-systemone-temperature-softmax-v1",
            "mode": "shadow",
            "authority": "recommendation-only",
        }
        for key, expected_value in expected_extension.items():
            if extension.get(key) != expected_value:
                raise DecisionAcceptanceError(
                    "decision response changed immutable semantics metadata"
                )
        timing = extension.get("timing")
        if not isinstance(timing, Mapping):
            raise DecisionAcceptanceError(
                "decision response omitted fabric timing"
            )

        def duration(name: str) -> float:
            raw = timing.get(name)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or float(raw) < 0
            ):
                raise DecisionAcceptanceError(
                    "decision response returned invalid fabric timing"
                )
            return float(raw)

        queue_ms = duration("queue_wait_ms")
        execution_ms = duration("execution_ms")
        durable_ms = duration("durable_job_ms")
        if durable_ms + 1.0 < queue_ms + execution_ms:
            raise DecisionAcceptanceError(
                "decision response returned inconsistent fabric timing"
            )
        fabric_overhead_ms = max(0.0, wall_ms - execution_ms)
        return _DecisionResult(
            choice_id=choice_id if isinstance(choice_id, str) else None,
            abstained=abstained,
            scores=tuple(scores),
            wall_ms=wall_ms,
            queue_wait_ms=queue_ms,
            execution_ms=execution_ms,
            fabric_overhead_ms=fabric_overhead_ms,
        )

    def run(self) -> DecisionShadowEvidence:
        with self._client() as client:
            profile = self._preflight(client)
            results: list[tuple[_FixtureCase, _DecisionResult]] = []
            for case in self.fixture.cases:
                results.append(
                    (
                        case,
                        self._decide(
                            client,
                            case=case,
                            choices=self.fixture.choices,
                        ),
                    )
                )

            probe = next(
                case
                for case in self.fixture.cases
                if case.case_id == self.fixture.ordering_probe_case_id
            )
            original_result = next(
                result
                for case, result in results
                if case.case_id == probe.case_id
            )
            reversed_result = self._decide(
                client,
                case=probe,
                choices=tuple(reversed(self.fixture.choices)),
            )

        answered = sum(not result.abstained for _, result in results)
        abstained = len(results) - answered
        correct = 0
        expected_unknown = 0
        unknown_abstained = 0
        false_safe = 0
        answered_correct = 0
        for case, result in results:
            if case.expected_choice_id is None:
                expected_unknown += 1
                is_correct = result.abstained
                if result.abstained:
                    unknown_abstained += 1
            else:
                is_correct = (
                    not result.abstained
                    and result.choice_id == case.expected_choice_id
                )
            if is_correct:
                correct += 1
            if (
                not result.abstained
                and case.expected_choice_id is not None
                and result.choice_id == case.expected_choice_id
            ):
                answered_correct += 1
            if (
                case.expected_choice_id
                in self.fixture.unsafe_expected_choice_ids
                and result.choice_id in self.fixture.false_safe_choice_ids
            ):
                false_safe += 1

        walls = [result.wall_ms for _, result in results]
        queues = [result.queue_wait_ms for _, result in results]
        executions = [result.execution_ms for _, result in results]
        overheads = [result.fabric_overhead_ms for _, result in results]
        changed = (
            original_result.abstained != reversed_result.abstained
            or original_result.choice_id != reversed_result.choice_id
        )

        count = len(results)
        coverage = answered / count
        answered_accuracy = (
            answered_correct / answered
            if answered
            else None
        )
        unknown_rate = (
            unknown_abstained / expected_unknown
            if expected_unknown
            else None
        )
        return DecisionShadowEvidence(
            evidence_version=_EVIDENCE_SCHEMA,
            date_utc=datetime.now(UTC).date().isoformat(),
            astrumweaver_revision=self.astrumweaver_revision,
            profile_id=self.profile_id,
            profile_revision=self.profile_revision,
            deployment_revision=self.deployment_revision,
            serving_contract_revision=self.serving_contract_revision,
            decision_semantics_id=self.decision_semantics_id,
            score_kind=str(profile["score_kind"]),
            provider_score_semantics=str(
                profile["provider_score_semantics"]
            ),
            calibration_status=str(profile["calibration_status"]),
            calibration_reference_sha256=None,
            abstain_below=float(profile["abstain_below"]),
            mode=str(profile["mode"]),
            authority=str(profile["authority"]),
            fixture_sha256=self.fixture.digest,
            case_count=count,
            answered_count=answered,
            abstained_count=abstained,
            correct_count=correct,
            incorrect_count=count - correct,
            expected_unknown_count=expected_unknown,
            expected_unknown_abstained_count=unknown_abstained,
            false_safe_count=false_safe,
            coverage=coverage,
            overall_accuracy=correct / count,
            answered_accuracy=answered_accuracy,
            unknown_abstention_rate=unknown_rate,
            ordering_probe_changed=changed,
            median_wall_ms=statistics.median(walls),
            p95_wall_ms=_percentile95(walls),
            median_queue_wait_ms=statistics.median(queues),
            median_execution_ms=statistics.median(executions),
            median_fabric_overhead_ms=statistics.median(overheads),
            identity_preflight="PASS",
            shadow_fixture_completed="PASS",
            private_values_omitted=True,
            quality_disposition="OBSERVED_ONLY",
            overall="PASS",
        )


def render_decision_shadow_markdown(
    evidence: DecisionShadowEvidence,
) -> str:
    def optional(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.6f}"

    fields = [
        ("Evidence version", evidence.evidence_version),
        ("Date (UTC)", evidence.date_utc),
        ("AstrumWeaver revision", evidence.astrumweaver_revision),
        ("Logical profile", evidence.profile_id),
        ("Logical profile revision", evidence.profile_revision),
        ("Deployment revision", evidence.deployment_revision),
        ("Serving contract revision", evidence.serving_contract_revision),
        ("Decision semantics", evidence.decision_semantics_id),
        ("Score kind", evidence.score_kind),
        ("Provider score semantics", evidence.provider_score_semantics),
        ("Calibration status", evidence.calibration_status),
        (
            "Calibration reference",
            evidence.calibration_reference_sha256 or "none",
        ),
        ("Abstain below", f"{evidence.abstain_below:.6f}"),
        ("Mode", evidence.mode),
        ("Authority", evidence.authority),
        ("Fixture SHA-256", evidence.fixture_sha256),
        ("Case count", str(evidence.case_count)),
        ("Answered count", str(evidence.answered_count)),
        ("Abstained count", str(evidence.abstained_count)),
        ("Correct count", str(evidence.correct_count)),
        ("Incorrect count", str(evidence.incorrect_count)),
        ("Expected unknown count", str(evidence.expected_unknown_count)),
        (
            "Expected unknown abstained",
            str(evidence.expected_unknown_abstained_count),
        ),
        ("False-safe count", str(evidence.false_safe_count)),
        ("Coverage", f"{evidence.coverage:.6f}"),
        ("Overall accuracy", f"{evidence.overall_accuracy:.6f}"),
        ("Answered accuracy", optional(evidence.answered_accuracy)),
        (
            "Unknown abstention rate",
            optional(evidence.unknown_abstention_rate),
        ),
        (
            "Ordering probe changed decision",
            str(evidence.ordering_probe_changed).lower(),
        ),
        ("Median wall ms", f"{evidence.median_wall_ms:.3f}"),
        ("P95 wall ms", f"{evidence.p95_wall_ms:.3f}"),
        (
            "Median queue wait ms",
            f"{evidence.median_queue_wait_ms:.3f}",
        ),
        (
            "Median execution ms",
            f"{evidence.median_execution_ms:.3f}",
        ),
        (
            "Median fabric overhead ms",
            f"{evidence.median_fabric_overhead_ms:.3f}",
        ),
        ("Identity preflight", evidence.identity_preflight),
        ("Shadow fixture completed", evidence.shadow_fixture_completed),
        ("Quality disposition", evidence.quality_disposition),
        (
            "Private values omitted",
            str(evidence.private_values_omitted).lower(),
        ),
        ("Overall", evidence.overall),
    ]
    rows = "\n".join(f"| {key} | {value} |" for key, value in fields)
    return (
        "# AstrumWeaver System-One Shadow Acceptance Evidence\n\n"
        "This file is intentionally redacted. It contains no gateway URL, "
        "credential, fixture state/question/choice text, fixture case IDs, "
        "local path, hostname, Worker/GPU identity, or model artifact path. "
        "Scores are choice-set probabilities and are explicitly uncalibrated; "
        "this evidence grants no execution, permission, review, or release "
        "authority.\n\n"
        "| Field | Result |\n"
        "| --- | --- |\n"
        f"{rows}\n"
    )


def write_decision_shadow_evidence(
    path: Path,
    evidence: DecisionShadowEvidence,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_decision_shadow_markdown(evidence),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="astrumweaver-decision-shadow-accept"
    )
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--profile-revision", required=True)
    parser.add_argument("--deployment-revision", required=True)
    parser.add_argument("--serving-contract-revision", required=True)
    parser.add_argument("--decision-semantics-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("validation/decision/shadow-v1.md"),
    )
    args = parser.parse_args()

    base_url = os.environ.get("ASTRUMWEAVER_DECISION_BASE_URL", "")
    client_token = os.environ.get("ASTRUMWEAVER_CLIENT_TOKEN", "")
    if not base_url:
        parser.exit(
            2,
            "astrumweaver-decision-shadow-accept: "
            "ASTRUMWEAVER_DECISION_BASE_URL is required\n",
        )
    if not client_token:
        parser.exit(
            2,
            "astrumweaver-decision-shadow-accept: "
            "ASTRUMWEAVER_CLIENT_TOKEN is required; client_auth=none may use "
            "a non-secret placeholder value\n",
        )

    try:
        evidence = DecisionShadowAcceptanceRunner(
            base_url=base_url,
            client_token=client_token,
            profile_id=args.profile_id,
            profile_revision=args.profile_revision,
            deployment_revision=args.deployment_revision,
            serving_contract_revision=args.serving_contract_revision,
            decision_semantics_id=args.decision_semantics_id,
            astrumweaver_revision=args.revision,
            fixture=args.fixture,
            timeout_seconds=args.timeout_seconds,
        ).run()
        write_decision_shadow_evidence(args.evidence, evidence)
    except (DecisionAcceptanceError, RuntimeError, ValueError) as exc:
        parser.exit(
            1,
            f"astrumweaver-decision-shadow-accept: {exc}\n",
        )

    print(json.dumps(asdict(evidence), sort_keys=True))


if __name__ == "__main__":
    main()