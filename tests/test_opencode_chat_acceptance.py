from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from astrumweaver.validation.opencode_chat import (
    PINNED_OPENCODE_CHAT_BLOB,
    PINNED_OPENCODE_SESSION_BLOB,
    OpenCodeChatAcceptanceError,
    OpenCodeChatAcceptanceRunner,
    render_opencode_chat_markdown,
)


DEPLOYMENT = "sha256:" + "1" * 64
CONTRACT = "sha256:" + "2" * 64


def completed(args, *, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        args=args,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class SuccessfulOpenCode:
    def __init__(self):
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((list(args), dict(kwargs)))
        if args[-1] == "--version" or args[1:] == ["--version"]:
            return completed(args, stdout="1.18.30\n")
        if len(self.calls) == 2:
            return completed(
                args,
                stdout=json.dumps(
                    {
                        "type": "text",
                        "part": {"text": "ASTRUMWEAVER_CHAT_OK"},
                    }
                )
                + "\n",
            )

        cwd = Path(kwargs["cwd"])
        assert (cwd / "acceptance.txt").read_text(encoding="utf-8") == (
            "The acceptance marker is ASTRUMWEAVER_TOOL_OK.\n"
        )
        return completed(
            args,
            stdout="\n".join(
                (
                    json.dumps(
                        {
                            "type": "tool_use",
                            "part": {
                                "tool": "read",
                                "state": {"status": "completed"},
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "text",
                            "part": {"text": "ASTRUMWEAVER_TOOL_OK"},
                        }
                    ),
                )
            )
            + "\n",
        )


def runner(command_runner):
    return OpenCodeChatAcceptanceRunner(
        opencode="/opt/opencode/bin/opencode",
        base_url="https://gateway.example.invalid/v1",
        client_token="private-token",
        profile_id="local-code-v1",
        astrumweaver_revision="deadbeef12345678",
        deployment_revision=DEPLOYMENT,
        serving_contract_revision=CONTRACT,
        context_tokens=5120,
        output_tokens=1024,
        timeout_seconds=30,
        command_runner=command_runner,
    )


def test_real_client_runner_uses_isolated_pinned_opencode_shape():
    fake = SuccessfulOpenCode()

    evidence = runner(fake).run()

    assert evidence.overall == "PASS"
    assert evidence.opencode_version == "1.18.30"
    assert evidence.opencode_session_source_sha1 == PINNED_OPENCODE_SESSION_BLOB
    assert evidence.opencode_chat_source_sha1 == PINNED_OPENCODE_CHAT_BLOB
    assert evidence.plain_chat_stream == "PASS"
    assert evidence.structured_tool_round_trip == "PASS"
    assert len(fake.calls) == 3

    for index, (args, kwargs) in enumerate(fake.calls):
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        env = kwargs["env"]
        assert env["HOME"].startswith("/tmp/")
        assert env["XDG_DATA_HOME"].startswith(env["HOME"])
        assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
        assert env["OPENCODE_PURE"] == "1"
        assert env["OPENCODE_DISABLE_AUTOUPDATE"] == "1"
        assert env["OPENCODE_DISABLE_AUTOCOMPACT"] == "1"
        assert env["OPENCODE_DISABLE_MODELS_FETCH"] == "1"
        assert env["OPENCODE_AUTH_CONTENT"] == "{}"
        assert env["ASTRUMWEAVER_CLIENT_TOKEN"] == "private-token"
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        provider = config["provider"]["astrumweaver-acceptance"]
        assert provider["options"]["baseURL"] == (
            "https://gateway.example.invalid/v1"
        )
        assert provider["options"]["apiKey"] == (
            "{env:ASTRUMWEAVER_CLIENT_TOKEN}"
        )
        assert "private-token" not in env["OPENCODE_CONFIG_CONTENT"]

        if index == 0:
            assert args == ["/opt/opencode/bin/opencode", "--version"]
            continue
        assert args[1] == "run"
        assert "--format" in args
        assert args[args.index("--format") + 1] == "json"
        assert args[args.index("--model") + 1] == (
            "astrumweaver-acceptance/local-code-v1"
        )
        assert "--auto" not in args
        assert "--dangerously-skip-permissions" not in args


def test_public_evidence_omits_private_gateway_and_acceptance_payloads():
    fake = SuccessfulOpenCode()
    evidence = runner(fake).run()

    markdown = render_opencode_chat_markdown(evidence)

    for private_value in (
        "https://gateway.example.invalid/v1",
        "private-token",
        "acceptance.txt",
        "ASTRUMWEAVER_CHAT_OK",
        "ASTRUMWEAVER_TOOL_OK",
        "/opt/opencode/bin/opencode",
        "/tmp/",
    ):
        assert private_value not in markdown

    assert "| OpenCode version | 1.18.30 |" in markdown
    assert "| Logical profile | local-code-v1 |" in markdown
    assert "| Structured tool round trip | PASS |" in markdown
    assert "| Overall | PASS |" in markdown


def test_pinned_version_mismatch_fails_before_chat_request():
    calls = []

    def fake(args, **kwargs):
        calls.append((args, kwargs))
        return completed(args, stdout="1.18.31\n")

    with pytest.raises(OpenCodeChatAcceptanceError, match="pinned"):
        runner(fake).run()

    assert len(calls) == 1


def test_nonzero_client_failure_does_not_echo_subprocess_output():
    calls = 0

    def fake(args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return completed(args, stdout="1.18.30\n")
        return completed(
            args,
            returncode=1,
            stderr=(
                "https://private.gateway.invalid/v1 "
                "Authorization: Bearer private-token"
            ),
        )

    with pytest.raises(OpenCodeChatAcceptanceError) as error:
        runner(fake).run()

    message = str(error.value)
    assert "private.gateway.invalid" not in message
    assert "private-token" not in message


@pytest.mark.parametrize(
    "url",
    (
        "https://user:secret@example.invalid/v1",
        "https://example.invalid/api",
        "file:///tmp/v1",
        "https://example.invalid/v1?token=secret",
    ),
)
def test_gateway_url_rejects_embedded_credentials_and_non_v1_shapes(url):
    with pytest.raises(OpenCodeChatAcceptanceError):
        OpenCodeChatAcceptanceRunner(
            opencode="opencode",
            base_url=url,
            client_token="token",
            profile_id="local-code-v1",
            astrumweaver_revision="deadbeef12345678",
            deployment_revision=DEPLOYMENT,
            serving_contract_revision=CONTRACT,
            context_tokens=5120,
            output_tokens=1024,
            command_runner=lambda *_args, **_kwargs: completed([]),
        )


def test_tool_smoke_requires_completed_structured_tool_event():
    calls = 0

    def fake(args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return completed(args, stdout="1.18.30\n")
        marker = (
            "ASTRUMWEAVER_CHAT_OK"
            if calls == 2
            else "ASTRUMWEAVER_TOOL_OK"
        )
        return completed(
            args,
            stdout=json.dumps(
                {"type": "text", "part": {"text": marker}}
            )
            + "\n",
        )

    with pytest.raises(OpenCodeChatAcceptanceError, match="structured tool"):
        runner(fake).run()
