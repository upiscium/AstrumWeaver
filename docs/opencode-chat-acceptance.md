# OpenCode chat acceptance

Tracking: #91 / #94.

This is the real-client acceptance layer for the AstrumWeaver chat gateway.
Schema/unit tests do not substitute for this run.

## Pinned client

The command accepts exactly:

- OpenCode `1.18.30`;
- OpenCode package manifest SHA-1
  `c7c467037d109b457884484af81d4518527816c5`;
- OpenCode `bun.lock` SHA-1
  `efe01bf957f3e825d822997161aed6cb3efe3728`;
- session source SHA-1
  `a99f8acff20c5d64d0b6cb90df480218bb1daddc`;
- AI SDK `6.0.168`;
- `@ai-sdk/openai-compatible` `2.0.41`;
- adapter stream source SHA-1
  `8c622db23c2d9a7373701f5a1b0c2ba109e24602`;
- adapter error-schema source SHA-1
  `f0ebb31de52b6484c9faa5ffd5eaed599c0c150e`.

The acceptance command does not install or upgrade OpenCode. Supply an existing
binary with `--opencode`; a version mismatch fails before any chat request.

## Isolation

The runner mirrors the pinned OpenCode subprocess-test isolation contract:

- temporary `HOME` and XDG config/data/state/cache roots;
- inline `OPENCODE_CONFIG_CONTENT`;
- `OPENCODE_DISABLE_PROJECT_CONFIG=1`;
- `OPENCODE_PURE=1`;
- automatic update, compaction and model-catalog fetch disabled;
- empty OpenCode auth content;
- a temporary working directory.

It does not load the user's ordinary OpenCode config, credentials, project
instructions, plugins or persisted sessions.

The gateway URL and client token are supplied only through environment
variables:

```text
ASTRUMWEAVER_CHAT_BASE_URL
ASTRUMWEAVER_CLIENT_TOKEN
```

`ASTRUMWEAVER_CHAT_BASE_URL` must be an HTTP(S) URL ending in `/v1` and
must not contain embedded credentials, a query string or a fragment. The token
is never placed in generated config bytes; OpenCode receives the ordinary
`{env:ASTRUMWEAVER_CLIENT_TOKEN}` reference. With
`control.client_auth = "none"`, use a non-secret placeholder token because
the OpenAI-compatible provider still expects an API-key value.

## Acceptance actions

Before OpenCode is started, the runner authenticates to the real
`GET /v1/models` endpoint and verifies the selected model's namespaced
`x_astrumweaver` metadata. The live gateway must exactly match the expected
logical-profile revision, deployment revision, serving-contract revision,
chat operation schema, structured-tools feature, total-context limit and
output-token limit. A mismatch fails before any OpenCode chat request, so the
recorded identity is not merely caller-supplied evidence.

The runner then performs two independent real OpenCode `run --format json`
sessions against the selected logical serving profile.

1. Plain chat requires the final text to be exactly
   `ASTRUMWEAVER_CHAT_OK`. The OpenCode transport therefore has to consume the
   real AstrumWeaver SSE stream to completion.
2. Tool round-trip creates one fixed marker file inside the temporary workdir,
   asks OpenCode to read it with a tool, requires at least one completed
   structured `tool_use` event, and then requires the final model response to
   be exactly `ASTRUMWEAVER_TOOL_OK`.

The runner deliberately does **not** pass `--auto`,
`--dangerously-skip-permissions`, or another broad permission bypass. A normal
read-tool round trip must succeed under the pinned client's ordinary
non-interactive permission policy.

The tool input/output, prompts and source marker never enter public evidence.

## Run

Use the exact AstrumWeaver revision, serving deployment revision and serving
contract revision under test. The OpenCode model limits must equal the selected
logical profile.

```sh
export ASTRUMWEAVER_CHAT_BASE_URL='https://gateway.example.invalid/v1'
export ASTRUMWEAVER_CLIENT_TOKEN='REDACTED'

astrumweaver-opencode-chat-accept \
  --opencode /absolute/path/to/opencode \
  --profile-id local-code-v1 \
  --profile-revision sha256:PROFILE_REVISION \
  --revision GIT_REVISION \
  --deployment-revision sha256:DEPLOYMENT_DIGEST \
  --serving-contract-revision sha256:CONTRACT_DIGEST \
  --context-tokens PROFILE_CONTEXT_LIMIT \
  --output-tokens PROFILE_OUTPUT_LIMIT
```

Default evidence path:

```text
validation/opencode/chat-v1.md
```

A successful command also prints the redacted evidence object as JSON.

## Public evidence boundary

The generated Markdown may contain only:

- evidence schema/date;
- AstrumWeaver Git revision;
- pinned OpenCode, AI SDK and OpenAI-compatible adapter identities;
- logical profile identifier and revision;
- deployment and serving-contract revisions;
- live gateway identity-preflight result;
- PASS/FAIL-scope fields represented by a successful evidence record.

It contains no gateway URL, credential, prompt, tool argument/result, local
path, source payload, hostname, Worker ID, GPU UUID or model artifact path.

## What PASS means

A PASS first proves that the live gateway profile identity/limits match the
explicit acceptance target, then proves the exact pinned OpenCode client can complete:

- one live streamed plain-chat exchange; and
- one structured tool call -> client tool execution/result -> final model
  response exchange

through the selected AstrumWeaver profile.

It does not certify model coding quality, latency, throughput, general tool
selection quality, GPU performance, or compatibility with another OpenCode
release. Those are separate evidence scopes.
