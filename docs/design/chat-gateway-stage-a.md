# Stage A bounded chat gateway

Tracking: #91 / #94. Foundation: completed #93 integration
`738578f85e2960990f5fda81216140dc1c36d5a8`.

This document freezes the first non-streaming vertical slice. It is not a claim
of general OpenAI, model, runtime or OpenCode compatibility.

## Boundary

The gateway is an edge adapter over the existing durable Control Job path. It
does not own GPUs, claim work, execute tools, access repositories or proxy
directly to a runtime.

```text
OpenAI-compatible client
        |
GET /v1/models
POST /v1/chat/completions
        |
configured LogicalServingProfile
        |
validated chat-job-v1 envelope
        |
durable #93 Job + deadline + ServingJobBinding
        |
atomic Worker claim / lease / cancellation
        |
provider-local chat adapter
        |
llama.cpp server (first provider only)
```

Stage A is explicitly non-streaming. `stream=true` is rejected until Stage B
adds the fenced generic job-event channel.

## First provider

The first provider-local adapter is llama.cpp only. No other RuntimeProvider is
advertised as Stage A chat-compatible merely because its executor exposes
`llm.chat`.

The llama.cpp adapter:

1. receives the opaque `chat-job-v1` envelope after claim;
2. verifies its adapter/schema identifier;
3. renders/counts the actual deployed chat request through llama.cpp's
   chat-completion token-count API before generation;
4. enforces the effective input/output token limits frozen by the serving
   profile;
5. forces non-streaming inference;
6. preserves the provider response structure for gateway normalization.

Context rejection is non-retryable. Provider transport/runtime failure retains
the ordinary retry/fencing policy before external output is visible.

## Configured models

`GET /v1/models` exposes configured logical serving profile IDs. It never
discovers arbitrary Worker/runtime models.

Each configured chat profile binds:

- one already resolved #93 `LogicalServingProfile`;
- its immutable `DeploymentIdentity` and `ServingContract`;
- one provider-local adapter ID;
- a bounded request deadline;
- an optional operator-facing display owner/name.

A request's `model` selects exactly one configured profile ID. No provider,
model, quantization or profile fallback is performed.

## Stage A request subset

Accepted chat requests preserve message order and support the fields needed for
the initial coding/tool round trip:

- `model`
- `messages`
- `tools` using OpenAI function-tool objects
- `tool_choice`: `none`, `auto`, `required`, or one named function
- `max_tokens`
- bounded sampling fields explicitly accepted by the adapter

Roles initially accepted are `system`, `user`, `assistant` and `tool`.
Assistant tool calls retain call IDs and JSON argument strings. Tool-result
messages retain `tool_call_id`. Unsupported fields fail closed rather than
being silently discarded.

If tools/tool choice are used, the resolved serving contract must advertise the
`tools` feature.

## Internal Job envelope

The gateway submits capability `llm.chat` with an opaque payload shaped as:

```json
{
  "schema_version": "chat-job-v1",
  "adapter_id": "llama-cpp-chat-v1",
  "request": {
    "messages": [],
    "tools": []
  },
  "limits": {
    "input_tokens": 0,
    "output_tokens": 0
  }
}
```

The numeric example above is schematic only. Real limits are positive values
from the resolved profile/contract snapshot. Control does not inspect the
envelope.

## Completion and cancellation

The gateway owns the Job it submits and waits only until the request deadline.
It polls durable state rather than bypassing Control.

- success: normalize the provider result to the documented Chat Completions
  subset;
- terminal non-retryable request/context failure: return a bounded client error;
- no compatible deployment / overload / expired admission: preserve the #93
  bounded error class;
- client disconnect or gateway timeout: cancel only the owned Job;
- a late/stale Worker result remains fenced by the existing durable Job
  lifecycle.

Stage A never retries after externally visible output because it exposes no
partial output.

## Stage B boundary

Live streaming is deferred until the generic event channel binds every event to
Job, attempt, runtime epoch and monotonic sequence, with bounded buffering and
stale-attempt publication fencing. A completed response split into fake chunks
does not count as Stage B streaming.
