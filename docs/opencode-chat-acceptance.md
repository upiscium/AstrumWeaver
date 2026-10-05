# OpenCode chat acceptance

Tracking: #91 / #94.

This is the real-client acceptance layer for the AstrumWeaver chat gateway.
Schema/unit tests do not substitute for this run.

## Pinned client

The command accepts exactly:

- OpenCode `1.18.30`;
- session source SHA-1
  `a99f8acff20c5d64d0b6cb90df480218bb1daddc`;
- OpenAI-compatible chat source SHA-1
  `9ac85b07b139f2a7a87f1a62d829a274b9cfd1ca`.

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

The runner performs two independent real OpenCode `run --format json`
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
- pinned OpenCode version and source identities;
- logical profile identifier;
- deployment and serving-contract revisions;
- PASS/FAIL-scope fields represented by a successful evidence record.

It contains no gateway URL, credential, prompt, tool argument/result, local
path, source payload, hostname, Worker ID, GPU UUID or model artifact path.

## What PASS means

A PASS proves the exact pinned OpenCode client can complete:

- one live streamed plain-chat exchange; and
- one structured tool call -> client tool execution/result -> final model
  response exchange

through the selected AstrumWeaver profile.

It does not certify model coding quality, latency, throughput, general tool
selection quality, GPU performance, or compatibility with another OpenCode
release. Those are separate evidence scopes.
