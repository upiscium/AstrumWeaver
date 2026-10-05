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
- OpenCode package manifest:
  `packages/opencode/package.json`
  (Git blob `c7c467037d109b457884484af81d4518527816c5`)
- session transport source:
  `packages/opencode/src/session/llm.ts`
  (Git blob `a99f8acff20c5d64d0b6cb90df480218bb1daddc`)
- AI SDK runtime: `ai@6.0.168`
- custom-provider adapter: `@ai-sdk/openai-compatible@2.0.41`
- adapter stream implementation:
  `packages/openai-compatible/src/chat/openai-compatible-chat-language-model.ts`
  at tag `@ai-sdk/openai-compatible@2.0.41`
  (Git blob `8c622db23c2d9a7373701f5a1b0c2ba109e24602`)
- adapter error schema:
  `packages/openai-compatible/src/openai-compatible-error.ts`
  at the same tag
  (Git blob `f0ebb31de52b6484c9faa5ffd5eaed599c0c150e`)

At this pin, the ordinary session path invokes AI SDK `streamText(...)` and
the configured `@ai-sdk/openai-compatible` adapter sends `stream: true`.
Its streaming schema is a union of ordinary chat chunks and the provider error
schema. AstrumWeaver's top-level `{"error": {"message": ...}}` SSE event on a
failed partial stream is therefore consumed as an explicit client-side stream
error rather than a successful chat chunk. The non-streaming Stage A gateway is
intentionally not accepted as OpenCode-compatible.

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
