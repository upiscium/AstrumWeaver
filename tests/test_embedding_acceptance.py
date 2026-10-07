from __future__ import annotations

import json

import httpx

from astrumweaver.gateway.embedding import (
    LLAMA_CPP_EMBEDDING_ADAPTER,
    EmbeddingSpaceIdentity,
    text_policy_digest,
)
from astrumweaver.validation.embedding import (
    EmbeddingAcceptanceError,
    EmbeddingAcceptanceRunner,
    render_embedding_acceptance_markdown,
)


def digest(ch: str) -> str:
    return "sha256:" + ch * 64


PROFILE_REVISION = digest("1")
DEPLOYMENT_REVISION = digest("2")
CONTRACT_REVISION = digest("3")
INCOMPATIBLE_SPACE = digest("f")


def space() -> EmbeddingSpaceIdentity:
    return EmbeddingSpaceIdentity(
        deployment_revision=DEPLOYMENT_REVISION,
        model_artifact_sha256=digest("4"),
        quantization="Q8_0",
        tokenizer_artifact_sha256=digest("5"),
        pooling="last",
        normalization="l2",
        dimensions=3,
        query_policy_id="query-v1",
        query_preprocess_sha256=text_policy_digest("query: "),
        document_policy_id="document-v1",
        document_preprocess_sha256=text_policy_digest(""),
        adapter_id=LLAMA_CPP_EMBEDDING_ADAPTER,
    )


def fixture(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "embedding-acceptance-fixture-v1",
                "documents": [
                    {"id": "private-document-alpha", "text": "alpha"},
                    {"id": "private-document-beta", "text": "beta"},
                ],
                "queries": [
                    {"text": "find alpha", "expected_document_id": "private-document-alpha"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def response(vectors):
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": vector,
                "index": index,
            }
            for index, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
        "x_astrumweaver": {
            "embedding_space_id": space().embedding_space_id,
        },
    }


def runner(tmp_path, handler):
    accepted = space()
    return EmbeddingAcceptanceRunner(
        base_url="https://gateway.example.invalid/v1",
        client_token="private-token",
        profile_id="notes-embed-v1",
        profile_revision=PROFILE_REVISION,
        deployment_revision=DEPLOYMENT_REVISION,
        serving_contract_revision=CONTRACT_REVISION,
        embedding_space_id=accepted.embedding_space_id,
        incompatible_space_id=INCOMPATIBLE_SPACE,
        astrumweaver_revision="deadbeef12345678",
        fixture=fixture(tmp_path),
        minimum_margin=0.10,
        timeout_seconds=30,
        transport=httpx.MockTransport(handler),
    )


def test_real_acceptance_runner_exercises_live_incompatible_space_rejection(tmp_path):
    accepted = space()
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["authorization"] == "Bearer private-token"

        if request.method == "GET" and request.url.path == "/v1/embedding-spaces":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "notes-embed-v1",
                            "object": "astrumweaver.embedding_space",
                            "x_astrumweaver": {
                                "embedding_space_id": accepted.embedding_space_id,
                                "profile_revision": PROFILE_REVISION,
                                "deployment_revision": DEPLOYMENT_REVISION,
                                "serving_contract_revision": CONTRACT_REVISION,
                                "dimensions": accepted.dimensions,
                                "pooling": accepted.pooling,
                                "normalization": accepted.normalization,
                                "input_types": ["query", "document"],
                                "space": accepted.to_dict(),
                                "effective_limits": {},
                            },
                        }
                    ],
                },
            )

        if request.method == "POST" and request.url.path == "/v1/embeddings":
            body = json.loads(request.content)
            pin = body["x_astrumweaver_embedding_space_id"]
            if pin == INCOMPATIBLE_SPACE:
                return httpx.Response(
                    409,
                    json={
                        "error": {
                            "code": "embedding_space_mismatch",
                            "message": "rejected",
                            "type": "invalid_request_error",
                        }
                    },
                )

            assert pin == accepted.embedding_space_id
            input_type = body["x_astrumweaver_input_type"]
            vectors = (
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                if input_type == "document"
                else [[1.0, 0.0, 0.0]]
            )
            value = response(vectors)
            value["x_astrumweaver"]["input_type"] = input_type
            return httpx.Response(200, json=value)

        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    evidence = runner(tmp_path, handler).run()

    assert evidence.overall == "PASS"
    assert evidence.identity_preflight == "PASS"
    assert evidence.document_batch == "PASS"
    assert evidence.query_batch == "PASS"
    assert evidence.retrieval_top1 == "PASS"
    assert evidence.incompatible_space_rejected == "PASS"
    assert len(calls) == 4

    negative_body = json.loads(calls[-1].content)
    assert negative_body["x_astrumweaver_embedding_space_id"] == INCOMPATIBLE_SPACE


def test_real_acceptance_runner_fails_if_gateway_accepts_incompatible_space(tmp_path):
    accepted = space()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "notes-embed-v1",
                            "x_astrumweaver": {
                                "embedding_space_id": accepted.embedding_space_id,
                                "profile_revision": PROFILE_REVISION,
                                "deployment_revision": DEPLOYMENT_REVISION,
                                "serving_contract_revision": CONTRACT_REVISION,
                                "space": accepted.to_dict(),
                            },
                        }
                    ],
                },
            )
        body = json.loads(request.content)
        input_type = body["x_astrumweaver_input_type"]
        vectors = (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            if len(body["input"]) == 2
            else [[1.0, 0.0, 0.0]]
        )
        value = response(vectors)
        value["x_astrumweaver"]["input_type"] = input_type
        return httpx.Response(200, json=value)

    try:
        runner(tmp_path, handler).run()
    except EmbeddingAcceptanceError as exc:
        assert "not rejected" in str(exc)
    else:
        raise AssertionError("acceptance must fail when an incompatible space is accepted")


def test_embedding_acceptance_public_evidence_omits_private_values(tmp_path):
    accepted = space()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "notes-embed-v1",
                            "x_astrumweaver": {
                                "embedding_space_id": accepted.embedding_space_id,
                                "profile_revision": PROFILE_REVISION,
                                "deployment_revision": DEPLOYMENT_REVISION,
                                "serving_contract_revision": CONTRACT_REVISION,
                                "space": accepted.to_dict(),
                            },
                        }
                    ],
                },
            )
        body = json.loads(request.content)
        if body["x_astrumweaver_embedding_space_id"] == INCOMPATIBLE_SPACE:
            return httpx.Response(
                409,
                json={"error": {"code": "embedding_space_mismatch"}},
            )
        input_type = body["x_astrumweaver_input_type"]
        vectors = (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            if len(body["input"]) == 2
            else [[1.0, 0.0, 0.0]]
        )
        value = response(vectors)
        value["x_astrumweaver"]["input_type"] = input_type
        return httpx.Response(200, json=value)

    markdown = render_embedding_acceptance_markdown(
        runner(tmp_path, handler).run()
    )
    for private_value in (
        "gateway.example.invalid",
        "private-token",
        "alpha",
        "beta",
        "find alpha",
        "private-document-alpha",
        "private-document-beta",
        str(tmp_path),
    ):
        assert private_value not in markdown
    assert "| Incompatible space rejected | PASS |" in markdown
    assert "| Overall | PASS |" in markdown
