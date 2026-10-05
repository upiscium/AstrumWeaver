# OpenCode chat compatibility target

Tracking: #91 / #94.

This file records client-side compatibility evidence separately from
AstrumWeaver schema/unit tests. It does not claim a successful real-client smoke
until one is explicitly recorded below.

## Pinned client

Initial target:

- OpenCode release: `v1.18.30`
- release date: 2026-09-09
- source tag: `anomalyco/opencode@v1.18.30`
- session transport source:
  `packages/opencode/src/session/llm.ts`
  (Git blob `a99f8acff20c5d64d0b6cb90df480218bb1daddc`)
- OpenAI-compatible Chat source:
  `packages/llm/src/protocols/openai-chat.ts`
  (Git blob `9ac85b07b139f2a7a87f1a62d829a274b9cfd1ca`)

At this pin, the ordinary session path invokes AI SDK `streamText(...)`. The
OpenAI-compatible Chat protocol sends `stream: true` and requests usage in
stream options. This means the non-streaming Stage A gateway is intentionally
not accepted as OpenCode-compatible.

## Provider configuration shape

OpenCode v1.18.30 documents custom OpenAI-compatible providers using
`@ai-sdk/openai-compatible` with a `baseURL` ending at the API's `/v1`
prefix and an explicitly enumerated model ID. AstrumWeaver model IDs are logical
serving profile IDs returned by `GET /v1/models`.

Illustrative configuration only:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "astrumweaver-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "AstrumWeaver local",
      "options": {
        "baseURL": "https://astrumweaver.example.invalid/v1",
        "apiKey": "{env:ASTRUMWEAVER_CLIENT_TOKEN}"
      },
      "models": {
        "local-code-v1": {
          "name": "AstrumWeaver local-code-v1",
          "limit": {
            "context": 5120,
            "output": 1024
          }
        }
      }
    }
  }
}
```

The example limits are placeholders and must match the selected
`LogicalServingProfile`; they are not model recommendations.

## Acceptance state

- Stage A schema/provider tests: implemented and integrated through #104.
- Stage A real OpenCode smoke: **NOT APPLICABLE / BLOCKED BY STREAMING**.
- Stage B fenced streaming protocol: implemented in #105; final CI/integration
  is tracked there.
- Plain-chat OpenCode v1.18.30 smoke: executable acceptance harness implemented;
  real runtime execution still required.
- Tool call -> client tool result -> final response smoke: executable acceptance
  harness implemented; real runtime execution still required.
- Real runtime/model/quantization/template identity: must be supplied by the
  acceptance run and recorded with its exact deployment/contract revisions.

The packaged `astrumweaver-opencode-chat-accept` command performs both real
client smokes with an isolated temporary OpenCode config/workdir and writes only
redacted evidence. See [OpenCode chat acceptance](../opencode-chat-acceptance.md).

A later compatibility update must preserve this historical pin rather than
silently replacing the client version used for acceptance.
