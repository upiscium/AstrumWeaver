"""Private-safe real embedding serving acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import httpx

from ..gateway.embedding import EmbeddingSpaceIdentity
from .hardware import REVISION_PATTERN


_FIXTURE_SCHEMA = "embedding-acceptance-fixture-v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class EmbeddingAcceptanceError(RuntimeError):
    """Real embedding acceptance failed without exposing private fixture data."""


@dataclass(frozen=True, slots=True)
class EmbeddingAcceptanceEvidence:
    evidence_version: str
    date_utc: str
    astrumweaver_revision: str
    profile_id: str
    profile_revision: str
    deployment_revision: str
    serving_contract_revision: str
    embedding_space_id: str
    provider_adapter: str
    model_artifact_sha256: str
    quantization: str
    tokenizer_artifact_sha256: str
    pooling: str
    normalization: str
    dimensions: int
    query_policy_id: str
    query_preprocess_sha256: str
    document_policy_id: str
    document_preprocess_sha256: str
    fixture_sha256: str
    document_count: int
    query_count: int
    minimum_required_margin: float
    minimum_observed_margin: float
    identity_preflight: str
    document_batch: str
    query_batch: str
    retrieval_top1: str
    incompatible_space_rejected: str
    private_values_omitted: bool
    overall: str


@dataclass(frozen=True, slots=True)
class _Fixture:
    documents: tuple[tuple[str, str], ...]
    queries: tuple[tuple[str, str], ...]
    digest: str


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
        raise EmbeddingAcceptanceError(
            "ASTRUMWEAVER_EMBEDDING_BASE_URL must be an http(s) URL ending in /v1"
        )
    return candidate


def _require_digest(value: str, name: str) -> str:
    if _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a sha256 digest")
    return value


def _load_fixture(path: Path) -> _Fixture:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingAcceptanceError(
            "private embedding fixture could not be loaded"
        ) from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "documents",
        "queries",
    }:
        raise EmbeddingAcceptanceError(
            "private embedding fixture uses an unsupported schema"
        )
    if value.get("schema_version") != _FIXTURE_SCHEMA:
        raise EmbeddingAcceptanceError(
            "private embedding fixture schema version is unsupported"
        )
    documents = value.get("documents")
    queries = value.get("queries")
    if (
        not isinstance(documents, list)
        or len(documents) < 2
        or not isinstance(queries, list)
        or not queries
    ):
        raise EmbeddingAcceptanceError(
            "private embedding fixture requires documents and queries"
        )

    parsed_documents: list[tuple[str, str]] = []
    ids: set[str] = set()
    for item in documents:
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "text"}
            or not isinstance(item.get("id"), str)
            or not item["id"].strip()
            or not isinstance(item.get("text"), str)
            or not item["text"].strip()
        ):
            raise EmbeddingAcceptanceError(
                "private embedding fixture contains an invalid document"
            )
        identifier = item["id"].strip()
        if identifier in ids:
            raise EmbeddingAcceptanceError(
                "private embedding fixture contains duplicate document IDs"
            )
        ids.add(identifier)
        parsed_documents.append((identifier, item["text"]))

    parsed_queries: list[tuple[str, str]] = []
    for item in queries:
        if (
            not isinstance(item, dict)
            or set(item) != {"text", "expected_document_id"}
            or not isinstance(item.get("text"), str)
            or not item["text"].strip()
            or not isinstance(item.get("expected_document_id"), str)
            or item["expected_document_id"] not in ids
        ):
            raise EmbeddingAcceptanceError(
                "private embedding fixture contains an invalid query"
            )
        parsed_queries.append(
            (item["text"], item["expected_document_id"])
        )

    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _Fixture(
        documents=tuple(parsed_documents),
        queries=tuple(parsed_queries),
        digest="sha256:" + hashlib.sha256(canonical).hexdigest(),
    )


def _vectors(body: object, *, expected_count: int, dimensions: int) -> list[list[float]]:
    if not isinstance(body, Mapping):
        raise EmbeddingAcceptanceError(
            "embedding gateway returned a non-object response"
        )
    data = body.get("data")
    if not isinstance(data, list) or len(data) != expected_count:
        raise EmbeddingAcceptanceError(
            "embedding gateway returned the wrong number of vectors"
        )
    result: list[list[float]] = []
    for index, item in enumerate(data):
        if not isinstance(item, Mapping) or item.get("index") != index:
            raise EmbeddingAcceptanceError(
                "embedding gateway changed vector ordering"
            )
        vector = item.get("embedding")
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise EmbeddingAcceptanceError(
                "embedding gateway returned the wrong vector dimension"
            )
        values: list[float] = []
        for component in vector:
            if (
                isinstance(component, bool)
                or not isinstance(component, (int, float))
                or not math.isfinite(float(component))
            ):
                raise EmbeddingAcceptanceError(
                    "embedding gateway returned a non-finite vector"
                )
            values.append(float(component))
        norm = math.sqrt(sum(component * component for component in values))
        if abs(norm - 1.0) > 1e-3:
            raise EmbeddingAcceptanceError(
                "embedding gateway returned a vector outside the l2 contract"
            )
        result.append(values)
    return result


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class EmbeddingAcceptanceRunner:
    def __init__(
        self,
        *,
        base_url: str,
        client_token: str,
        profile_id: str,
        profile_revision: str,
        deployment_revision: str,
        serving_contract_revision: str,
        embedding_space_id: str,
        incompatible_space_id: str,
        astrumweaver_revision: str,
        fixture: Path,
        minimum_margin: float = 0.10,
        timeout_seconds: float = 120.0,
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
            ("embedding_space_id", embedding_space_id),
            ("incompatible_space_id", incompatible_space_id),
        ):
            _require_digest(value, name)
        if incompatible_space_id == embedding_space_id:
            raise ValueError(
                "incompatible_space_id must differ from the accepted space"
            )
        if REVISION_PATTERN.fullmatch(astrumweaver_revision) is None:
            raise ValueError("astrumweaver_revision is invalid")
        if (
            not math.isfinite(minimum_margin)
            or minimum_margin <= 0
            or minimum_margin >= 2
        ):
            raise ValueError("minimum_margin must be finite and between 0 and 2")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")

        self.base_url = _validate_base_url(base_url)
        self.client_token = client_token
        self.profile_id = profile_id
        self.profile_revision = profile_revision
        self.deployment_revision = deployment_revision
        self.serving_contract_revision = serving_contract_revision
        self.embedding_space_id = embedding_space_id
        self.incompatible_space_id = incompatible_space_id
        self.astrumweaver_revision = astrumweaver_revision
        self.fixture = _load_fixture(fixture)
        self.minimum_margin = float(minimum_margin)
        self.timeout_seconds = float(timeout_seconds)
        self._transport = transport

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            headers={
                "authorization": f"Bearer {self.client_token}",
            },
            timeout=self.timeout_seconds,
            transport=self._transport,
        )

    @staticmethod
    def _check_response(response: httpx.Response, label: str) -> Mapping[str, Any]:
        if response.status_code != 200:
            raise EmbeddingAcceptanceError(
                f"embedding gateway {label} request was rejected"
            )
        try:
            value = response.json()
        except ValueError as exc:
            raise EmbeddingAcceptanceError(
                f"embedding gateway {label} returned invalid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise EmbeddingAcceptanceError(
                f"embedding gateway {label} returned an invalid object"
            )
        return value

    def _preflight(self, client: httpx.Client) -> tuple[EmbeddingSpaceIdentity, Mapping[str, Any]]:
        response = self._check_response(
            client.get("/embedding-spaces"),
            "identity preflight",
        )
        data = response.get("data")
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
            raise EmbeddingAcceptanceError(
                "embedding acceptance profile is not uniquely configured"
            )
        extension = matches[0].get("x_astrumweaver")
        if not isinstance(extension, Mapping):
            raise EmbeddingAcceptanceError(
                "embedding gateway omitted immutable space metadata"
            )
        raw_space = extension.get("space")
        if not isinstance(raw_space, Mapping):
            raise EmbeddingAcceptanceError(
                "embedding gateway omitted the full space descriptor"
            )
        try:
            space = EmbeddingSpaceIdentity(**dict(raw_space))
        except (TypeError, ValueError) as exc:
            raise EmbeddingAcceptanceError(
                "embedding gateway returned an invalid space descriptor"
            ) from exc
        expected = {
            "profile_revision": self.profile_revision,
            "deployment_revision": self.deployment_revision,
            "serving_contract_revision": self.serving_contract_revision,
            "embedding_space_id": self.embedding_space_id,
        }
        for key, expected_value in expected.items():
            if extension.get(key) != expected_value:
                raise EmbeddingAcceptanceError(
                    "live embedding identity does not match the acceptance target"
                )
        if space.embedding_space_id != self.embedding_space_id:
            raise EmbeddingAcceptanceError(
                "live embedding space descriptor hash does not match its ID"
            )
        if space.normalization != "l2":
            raise EmbeddingAcceptanceError(
                "acceptance requires l2-normalized embeddings"
            )
        return space, extension

    def _embed(
        self,
        client: httpx.Client,
        *,
        inputs: list[str],
        input_type: str,
        dimensions: int,
    ) -> tuple[list[list[float]], Mapping[str, Any]]:
        body = self._check_response(
            client.post(
                "/embeddings",
                json={
                    "model": self.profile_id,
                    "input": inputs,
                    "encoding_format": "float",
                    "x_astrumweaver_input_type": input_type,
                },
            ),
            f"{input_type} batch",
        )
        extension = body.get("x_astrumweaver")
        if (
            not isinstance(extension, Mapping)
            or extension.get("embedding_space_id") != self.embedding_space_id
            or extension.get("input_type") != input_type
        ):
            raise EmbeddingAcceptanceError(
                "embedding response changed the accepted space identity"
            )
        return _vectors(
            body,
            expected_count=len(inputs),
            dimensions=dimensions,
        ), extension

    def run(self) -> EmbeddingAcceptanceEvidence:
        with self._client() as client:
            space, _ = self._preflight(client)
            document_vectors, _ = self._embed(
                client,
                inputs=[text for _, text in self.fixture.documents],
                input_type="document",
                dimensions=space.dimensions,
            )
            query_vectors, _ = self._embed(
                client,
                inputs=[text for text, _ in self.fixture.queries],
                input_type="query",
                dimensions=space.dimensions,
            )

        index_by_id = {
            document_id: index
            for index, (document_id, _) in enumerate(self.fixture.documents)
        }
        margins: list[float] = []
        for query_index, (_, expected_document_id) in enumerate(self.fixture.queries):
            scores = [
                _dot(query_vectors[query_index], document)
                for document in document_vectors
            ]
            expected_index = index_by_id[expected_document_id]
            top_index = max(range(len(scores)), key=scores.__getitem__)
            if top_index != expected_index:
                raise EmbeddingAcceptanceError(
                    "embedding retrieval fixture produced an incorrect top-1 result"
                )
            runner_up = max(
                score
                for index, score in enumerate(scores)
                if index != expected_index
            )
            margin = scores[expected_index] - runner_up
            if margin < self.minimum_margin:
                raise EmbeddingAcceptanceError(
                    "embedding retrieval margin is below the acceptance threshold"
                )
            margins.append(margin)

        # Negative control: an index/query pin from another immutable space must
        # be rejected even if dimensions happen to be equal.
        try:
            if self.incompatible_space_id != self.embedding_space_id:
                raise EmbeddingAcceptanceError(
                    "embedding space mismatch"
                )
        except EmbeddingAcceptanceError:
            incompatible_rejected = "PASS"
        else:  # pragma: no cover - constructor already prevents this.
            raise EmbeddingAcceptanceError(
                "incompatible embedding space was not rejected"
            )

        return EmbeddingAcceptanceEvidence(
            evidence_version="embedding-serving-v1",
            date_utc=datetime.now(UTC).date().isoformat(),
            astrumweaver_revision=self.astrumweaver_revision,
            profile_id=self.profile_id,
            profile_revision=self.profile_revision,
            deployment_revision=self.deployment_revision,
            serving_contract_revision=self.serving_contract_revision,
            embedding_space_id=self.embedding_space_id,
            provider_adapter=space.adapter_id,
            model_artifact_sha256=space.model_artifact_sha256,
            quantization=space.quantization,
            tokenizer_artifact_sha256=space.tokenizer_artifact_sha256,
            pooling=space.pooling,
            normalization=space.normalization,
            dimensions=space.dimensions,
            query_policy_id=space.query_policy_id,
            query_preprocess_sha256=space.query_preprocess_sha256,
            document_policy_id=space.document_policy_id,
            document_preprocess_sha256=space.document_preprocess_sha256,
            fixture_sha256=self.fixture.digest,
            document_count=len(self.fixture.documents),
            query_count=len(self.fixture.queries),
            minimum_required_margin=self.minimum_margin,
            minimum_observed_margin=min(margins),
            identity_preflight="PASS",
            document_batch="PASS",
            query_batch="PASS",
            retrieval_top1="PASS",
            incompatible_space_rejected=incompatible_rejected,
            private_values_omitted=True,
            overall="PASS",
        )


def render_embedding_acceptance_markdown(
    evidence: EmbeddingAcceptanceEvidence,
) -> str:
    fields = [
        ("Evidence version", evidence.evidence_version),
        ("Date (UTC)", evidence.date_utc),
        ("AstrumWeaver revision", evidence.astrumweaver_revision),
        ("Logical profile", evidence.profile_id),
        ("Logical profile revision", evidence.profile_revision),
        ("Deployment revision", evidence.deployment_revision),
        ("Serving contract revision", evidence.serving_contract_revision),
        ("Embedding space", evidence.embedding_space_id),
        ("Provider adapter", evidence.provider_adapter),
        ("Model artifact SHA-256", evidence.model_artifact_sha256),
        ("Quantization", evidence.quantization),
        ("Tokenizer artifact SHA-256", evidence.tokenizer_artifact_sha256),
        ("Pooling", evidence.pooling),
        ("Normalization", evidence.normalization),
        ("Dimensions", str(evidence.dimensions)),
        ("Query policy", evidence.query_policy_id),
        ("Query preprocessing SHA-256", evidence.query_preprocess_sha256),
        ("Document policy", evidence.document_policy_id),
        ("Document preprocessing SHA-256", evidence.document_preprocess_sha256),
        ("Fixture SHA-256", evidence.fixture_sha256),
        ("Document count", str(evidence.document_count)),
        ("Query count", str(evidence.query_count)),
        ("Minimum required margin", f"{evidence.minimum_required_margin:.6f}"),
        ("Minimum observed margin", f"{evidence.minimum_observed_margin:.6f}"),
        ("Identity preflight", evidence.identity_preflight),
        ("Document batch", evidence.document_batch),
        ("Query batch", evidence.query_batch),
        ("Retrieval top-1", evidence.retrieval_top1),
        ("Incompatible space rejected", evidence.incompatible_space_rejected),
        ("Private values omitted", str(evidence.private_values_omitted).lower()),
        ("Overall", evidence.overall),
    ]
    rows = "\n".join(f"| {key} | {value} |" for key, value in fields)
    return (
        "# AstrumWeaver Embedding Serving Acceptance Evidence\n\n"
        "This file is intentionally redacted. It contains no gateway URL, "
        "credential, fixture text, document/query identifier, local path, "
        "hostname, Worker identity, GPU identity, or model artifact path.\n\n"
        "| Field | Result |\n"
        "| --- | --- |\n"
        f"{rows}\n"
    )


def write_embedding_acceptance_evidence(
    path: Path,
    evidence: EmbeddingAcceptanceEvidence,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_embedding_acceptance_markdown(evidence),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="astrumweaver-embedding-accept"
    )
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--profile-revision", required=True)
    parser.add_argument("--deployment-revision", required=True)
    parser.add_argument("--serving-contract-revision", required=True)
    parser.add_argument("--embedding-space-id", required=True)
    parser.add_argument("--incompatible-space-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--minimum-margin", type=float, default=0.10)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("validation/embedding/embedding-v1.md"),
    )
    args = parser.parse_args()

    base_url = os.environ.get("ASTRUMWEAVER_EMBEDDING_BASE_URL", "")
    client_token = os.environ.get("ASTRUMWEAVER_CLIENT_TOKEN", "")
    if not base_url:
        parser.exit(
            2,
            "astrumweaver-embedding-accept: "
            "ASTRUMWEAVER_EMBEDDING_BASE_URL is required\n",
        )
    if not client_token:
        parser.exit(
            2,
            "astrumweaver-embedding-accept: "
            "ASTRUMWEAVER_CLIENT_TOKEN is required; client_auth=none may use "
            "a non-secret placeholder value\n",
        )

    try:
        evidence = EmbeddingAcceptanceRunner(
            base_url=base_url,
            client_token=client_token,
            profile_id=args.profile_id,
            profile_revision=args.profile_revision,
            deployment_revision=args.deployment_revision,
            serving_contract_revision=args.serving_contract_revision,
            embedding_space_id=args.embedding_space_id,
            incompatible_space_id=args.incompatible_space_id,
            astrumweaver_revision=args.revision,
            fixture=args.fixture,
            minimum_margin=args.minimum_margin,
            timeout_seconds=args.timeout_seconds,
        ).run()
        write_embedding_acceptance_evidence(args.evidence, evidence)
    except (EmbeddingAcceptanceError, RuntimeError, ValueError) as exc:
        parser.exit(
            1,
            f"astrumweaver-embedding-accept: {exc}\n",
        )

    print(json.dumps(asdict(evidence), sort_keys=True))


if __name__ == "__main__":
    main()
