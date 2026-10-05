"""Private-safe real OpenCode chat compatibility acceptance."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from .hardware import REVISION_PATTERN


PINNED_OPENCODE_VERSION = "1.18.30"
PINNED_OPENCODE_SESSION_BLOB = "a99f8acff20c5d64d0b6cb90df480218bb1daddc"
PINNED_OPENCODE_CHAT_BLOB = "9ac85b07b139f2a7a87f1a62d829a274b9cfd1ca"
_PROVIDER_ID = "astrumweaver-acceptance"
_PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_PLAIN_MARKER = "ASTRUMWEAVER_CHAT_OK"
_TOOL_MARKER = "ASTRUMWEAVER_TOOL_OK"
_TOOL_FILENAME = "acceptance.txt"

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class OpenCodeChatAcceptanceError(RuntimeError):
    """Pinned-client acceptance failed without exposing private payloads."""


@dataclass(frozen=True, slots=True)
class OpenCodeChatAcceptanceEvidence:
    evidence_version: str
    date_utc: str
    astrumweaver_revision: str
    opencode_version: str
    opencode_session_source_sha1: str
    opencode_chat_source_sha1: str
    profile_id: str
    deployment_revision: str
    serving_contract_revision: str
    isolated_config: str
    plain_chat_stream: str
    structured_tool_round_trip: str
    private_values_omitted: bool
    overall: str


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
        raise OpenCodeChatAcceptanceError(
            "ASTRUMWEAVER_CHAT_BASE_URL must be an http(s) URL ending in /v1"
        )
    return candidate


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_events(stdout: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OpenCodeChatAcceptanceError(
                "OpenCode emitted non-JSON output in --format json mode"
            ) from exc
        if not isinstance(value, dict):
            raise OpenCodeChatAcceptanceError(
                "OpenCode emitted a non-object JSON event"
            )
        events.append(value)
    if not events:
        raise OpenCodeChatAcceptanceError("OpenCode emitted no acceptance events")
    return events


def _text_output(events: Sequence[Mapping[str, object]]) -> str:
    parts: list[str] = []
    for event in events:
        if event.get("type") != "text":
            continue
        part = event.get("part")
        if not isinstance(part, Mapping):
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()


def _reject_error_events(events: Sequence[Mapping[str, object]]) -> None:
    for event in events:
        if event.get("type") == "error":
            raise OpenCodeChatAcceptanceError(
                "OpenCode reported a session error during acceptance"
            )
        if event.get("type") != "tool_use":
            continue
        part = event.get("part")
        if not isinstance(part, Mapping):
            continue
        state = part.get("state")
        if isinstance(state, Mapping) and state.get("status") == "error":
            raise OpenCodeChatAcceptanceError(
                "OpenCode reported a failed tool during acceptance"
            )


def _has_completed_tool(events: Sequence[Mapping[str, object]]) -> bool:
    for event in events:
        if event.get("type") != "tool_use":
            continue
        part = event.get("part")
        if not isinstance(part, Mapping):
            continue
        state = part.get("state")
        if isinstance(state, Mapping) and state.get("status") == "completed":
            return True
    return False


class OpenCodeChatAcceptanceRunner:
    def __init__(
        self,
        *,
        opencode: str,
        base_url: str,
        client_token: str,
        profile_id: str,
        astrumweaver_revision: str,
        deployment_revision: str,
        serving_contract_revision: str,
        context_tokens: int,
        output_tokens: int,
        timeout_seconds: float = 180.0,
        command_runner: CommandRunner = subprocess.run,
    ) -> None:
        executable = str(opencode).strip()
        if not executable:
            raise ValueError("opencode executable must not be blank")
        if not client_token:
            raise ValueError("client token must not be blank")
        if not _PROFILE_ID.fullmatch(profile_id):
            raise ValueError("profile_id is invalid")
        if not REVISION_PATTERN.fullmatch(astrumweaver_revision):
            raise ValueError("astrumweaver_revision is invalid")
        if not _DIGEST.fullmatch(deployment_revision):
            raise ValueError("deployment_revision is invalid")
        if not _DIGEST.fullmatch(serving_contract_revision):
            raise ValueError("serving_contract_revision is invalid")
        context = _positive_integer(context_tokens, "context_tokens")
        output = _positive_integer(output_tokens, "output_tokens")
        if output > context:
            raise ValueError("output_tokens must not exceed context_tokens")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")

        self.opencode = executable
        self.base_url = _validate_base_url(base_url)
        self.client_token = client_token
        self.profile_id = profile_id
        self.astrumweaver_revision = astrumweaver_revision
        self.deployment_revision = deployment_revision
        self.serving_contract_revision = serving_contract_revision
        self.context_tokens = context
        self.output_tokens = output
        self.timeout_seconds = float(timeout_seconds)
        self._command_runner = command_runner

    def _config(self) -> dict[str, object]:
        return {
            "$schema": "https://opencode.ai/config.json",
            "formatter": False,
            "lsp": False,
            "provider": {
                _PROVIDER_ID: {
                    "name": "AstrumWeaver acceptance",
                    "id": _PROVIDER_ID,
                    "env": [],
                    "npm": "@ai-sdk/openai-compatible",
                    "models": {
                        self.profile_id: {
                            "id": self.profile_id,
                            "name": self.profile_id,
                            "attachment": False,
                            "reasoning": False,
                            "temperature": True,
                            "tool_call": True,
                            "release_date": "2026-10-05",
                            "limit": {
                                "context": self.context_tokens,
                                "output": self.output_tokens,
                            },
                            "cost": {"input": 0, "output": 0},
                            "options": {},
                        }
                    },
                    "options": {
                        "apiKey": "{env:ASTRUMWEAVER_CLIENT_TOKEN}",
                        "baseURL": self.base_url,
                    },
                }
            },
        }

    def _environment(self, home: Path) -> dict[str, str]:
        env: dict[str, str] = {}
        for name in (
            "PATH",
            "LANG",
            "LC_ALL",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "NODE_EXTRA_CA_CERTS",
        ):
            value = os.environ.get(name)
            if value:
                env[name] = value

        env.update(
            {
                "HOME": str(home),
                "XDG_CONFIG_HOME": str(home / ".config"),
                "XDG_DATA_HOME": str(home / ".local" / "share"),
                "XDG_STATE_HOME": str(home / ".local" / "state"),
                "XDG_CACHE_HOME": str(home / ".cache"),
                "OPENCODE_TEST_HOME": str(home),
                "OPENCODE_CONFIG_DIR": str(home / ".config" / "opencode"),
                "OPENCODE_CONFIG_CONTENT": json.dumps(
                    self._config(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
                "OPENCODE_PURE": "1",
                "OPENCODE_DISABLE_AUTOUPDATE": "1",
                "OPENCODE_DISABLE_AUTOCOMPACT": "1",
                "OPENCODE_DISABLE_MODELS_FETCH": "1",
                "OPENCODE_AUTH_CONTENT": "{}",
                "ASTRUMWEAVER_CLIENT_TOKEN": self.client_token,
            }
        )
        return env

    def _invoke(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        label: str,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = self._command_runner(
                list(command),
                cwd=cwd,
                env=dict(env),
                timeout=self.timeout_seconds,
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise OpenCodeChatAcceptanceError(
                f"OpenCode {label} timed out"
            ) from exc
        except OSError as exc:
            raise OpenCodeChatAcceptanceError(
                f"OpenCode {label} could not be executed"
            ) from exc
        if result.returncode != 0:
            raise OpenCodeChatAcceptanceError(
                f"OpenCode {label} exited unsuccessfully"
            )
        return result

    def _run_prompt(
        self,
        prompt: str,
        *,
        workdir: Path,
        env: Mapping[str, str],
        label: str,
    ) -> list[dict[str, object]]:
        result = self._invoke(
            (
                self.opencode,
                "run",
                prompt,
                "--model",
                f"{_PROVIDER_ID}/{self.profile_id}",
                "--format",
                "json",
                "--dir",
                str(workdir),
            ),
            cwd=workdir,
            env=env,
            label=label,
        )
        events = _parse_events(result.stdout)
        _reject_error_events(events)
        return events

    def run(self) -> OpenCodeChatAcceptanceEvidence:
        with tempfile.TemporaryDirectory(
            prefix="astrumweaver-opencode-accept-"
        ) as raw:
            root = Path(raw)
            home = root / "home"
            workdir = root / "work"
            home.mkdir()
            workdir.mkdir()
            env = self._environment(home)

            version_result = self._invoke(
                (self.opencode, "--version"),
                cwd=workdir,
                env=env,
                label="version check",
            )
            lines = [
                line.strip()
                for line in version_result.stdout.splitlines()
                if line.strip()
            ]
            if not lines:
                raise OpenCodeChatAcceptanceError(
                    "OpenCode version output is unavailable"
                )
            version = lines[-1].removeprefix("v")
            if version != PINNED_OPENCODE_VERSION:
                raise OpenCodeChatAcceptanceError(
                    "OpenCode version does not match the pinned compatibility target"
                )

            plain_events = self._run_prompt(
                "Reply with exactly ASTRUMWEAVER_CHAT_OK and no other text.",
                workdir=workdir,
                env=env,
                label="plain-chat smoke",
            )
            if _text_output(plain_events) != _PLAIN_MARKER:
                raise OpenCodeChatAcceptanceError(
                    "OpenCode plain-chat smoke did not return the expected marker"
                )

            (workdir / _TOOL_FILENAME).write_text(
                "The acceptance marker is ASTRUMWEAVER_TOOL_OK.\n",
                encoding="utf-8",
            )
            tool_events = self._run_prompt(
                "Use the read tool to read acceptance.txt in the current "
                "directory. After the tool completes, reply with exactly "
                "ASTRUMWEAVER_TOOL_OK and no other text.",
                workdir=workdir,
                env=env,
                label="tool round-trip smoke",
            )
            if not _has_completed_tool(tool_events):
                raise OpenCodeChatAcceptanceError(
                    "OpenCode tool smoke did not complete a structured tool call"
                )
            if _text_output(tool_events) != _TOOL_MARKER:
                raise OpenCodeChatAcceptanceError(
                    "OpenCode tool smoke did not return the expected final marker"
                )

        return OpenCodeChatAcceptanceEvidence(
            evidence_version="opencode-chat-v1",
            date_utc=datetime.now(UTC).date().isoformat(),
            astrumweaver_revision=self.astrumweaver_revision,
            opencode_version=PINNED_OPENCODE_VERSION,
            opencode_session_source_sha1=PINNED_OPENCODE_SESSION_BLOB,
            opencode_chat_source_sha1=PINNED_OPENCODE_CHAT_BLOB,
            profile_id=self.profile_id,
            deployment_revision=self.deployment_revision,
            serving_contract_revision=self.serving_contract_revision,
            isolated_config="PASS",
            plain_chat_stream="PASS",
            structured_tool_round_trip="PASS",
            private_values_omitted=True,
            overall="PASS",
        )


def render_opencode_chat_markdown(
    evidence: OpenCodeChatAcceptanceEvidence,
) -> str:
    fields = [
        ("Evidence version", evidence.evidence_version),
        ("Date (UTC)", evidence.date_utc),
        ("AstrumWeaver revision", evidence.astrumweaver_revision),
        ("OpenCode version", evidence.opencode_version),
        (
            "OpenCode session source SHA-1",
            evidence.opencode_session_source_sha1,
        ),
        (
            "OpenCode chat source SHA-1",
            evidence.opencode_chat_source_sha1,
        ),
        ("Logical profile", evidence.profile_id),
        ("Deployment revision", evidence.deployment_revision),
        ("Serving contract revision", evidence.serving_contract_revision),
        ("Isolated OpenCode config", evidence.isolated_config),
        ("Plain chat stream", evidence.plain_chat_stream),
        (
            "Structured tool round trip",
            evidence.structured_tool_round_trip,
        ),
        (
            "Private values omitted",
            str(evidence.private_values_omitted).lower(),
        ),
        ("Overall", evidence.overall),
    ]
    rows = "\n".join(f"| {key} | {value} |" for key, value in fields)
    return (
        "# AstrumWeaver OpenCode Chat Acceptance Evidence\n\n"
        "This file is intentionally redacted. It contains no gateway URL, "
        "credential, prompt, tool arguments/result, local path, source payload, "
        "hostname, Worker ID, GPU UUID, or model artifact path.\n\n"
        "| Field | Result |\n"
        "| --- | --- |\n"
        f"{rows}\n"
    )


def write_opencode_chat_evidence(
    path: Path,
    evidence: OpenCodeChatAcceptanceEvidence,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_opencode_chat_markdown(evidence),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="astrumweaver-opencode-chat-accept"
    )
    parser.add_argument("--opencode", default="opencode")
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--deployment-revision", required=True)
    parser.add_argument("--serving-contract-revision", required=True)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("validation/opencode/chat-v1.md"),
    )
    args = parser.parse_args()

    base_url = os.environ.get("ASTRUMWEAVER_CHAT_BASE_URL", "")
    client_token = os.environ.get("ASTRUMWEAVER_CLIENT_TOKEN", "")
    if not base_url:
        parser.exit(
            2,
            "astrumweaver-opencode-chat-accept: "
            "ASTRUMWEAVER_CHAT_BASE_URL is required\n",
        )
    if not client_token:
        parser.exit(
            2,
            "astrumweaver-opencode-chat-accept: "
            "ASTRUMWEAVER_CLIENT_TOKEN is required; client_auth=none may use "
            "a non-secret placeholder value\n",
        )

    try:
        runner = OpenCodeChatAcceptanceRunner(
            opencode=args.opencode,
            base_url=base_url,
            client_token=client_token,
            profile_id=args.profile_id,
            astrumweaver_revision=args.revision,
            deployment_revision=args.deployment_revision,
            serving_contract_revision=args.serving_contract_revision,
            context_tokens=args.context_tokens,
            output_tokens=args.output_tokens,
            timeout_seconds=args.timeout_seconds,
        )
        evidence = runner.run()
        write_opencode_chat_evidence(args.evidence, evidence)
    except (
        OpenCodeChatAcceptanceError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.exit(
            1,
            f"astrumweaver-opencode-chat-accept: {exc}\n",
        )

    print(json.dumps(asdict(evidence), sort_keys=True))


if __name__ == "__main__":
    main()
