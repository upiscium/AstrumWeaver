# Chat gateway Stage B: fenced live streaming

Tracking: #91 / #94. Stacked on Stage A PR #104.

Stage B adds live streaming without changing the durable scheduling/ownership
model. Streaming is an optional generic Job event channel carried through
Control; the gateway never proxies a client directly to a runtime.

## Authority model

A published event belongs to exactly one currently running attempt and binds:

- durable Job ID;
- current attempt number;
- monotonic per-Job event sequence;
- assigned Worker ID;
- claimed runtime-instance epoch;
- event kind and opaque payload.

Worker publication must present the current Worker authority, lease token and
runtime-instance epoch. Control validates the same running-attempt fences used
for heartbeat and terminal writes. A cancelled, recovered, expired or replaced
attempt cannot append another event.

Control does not interpret chat chunks. It stores/forwards opaque event payloads;
the chat gateway and provider-local adapter own workload semantics.

## Buffering and backpressure

The first implementation uses a bounded durable event buffer in PostgreSQL and
the in-memory reference repository. Limits are explicit:

- maximum event payload bytes;
- maximum events retained per Job;
- bounded read page size.

An append that would exceed the buffer fails closed. The Worker treats failed
publication as loss of streaming authority and aborts the provider stream rather
than dropping, coalescing or bypassing Control.

Events are retained with the durable Job state in this stage. This does not
weaken the existing payload-retention warning: Control storage/backups remain
private request/output storage.

## Retry policy

Streaming chat requests use `max_attempts = 1`.

This is intentionally stronger than the minimum requirement. A Stage B request
is never automatically recovered onto a second attempt, so externally visible
output can never be concatenated with a retry. Non-streaming Stage A requests
retain their existing bounded retry policy.

## Executor boundary

A generic optional streaming executor protocol receives an async event sink.
Executors without that protocol keep the existing `JobExecutor.execute` path.

The Worker owns the event sink. Each emitted event is serialized with Control
RPCs and published with the active lease/epoch before the executor is allowed to
continue. Provider adapters therefore cannot expose output that Control rejected.

The first provider-local implementation is llama.cpp:

1. validate the same `chat-job-v1` envelope and token limits as Stage A;
2. send `stream: true` and request usage chunks;
3. parse upstream SSE incrementally;
4. validate each OpenAI-compatible chat chunk;
5. emit `chat.completion.chunk` events through the Worker sink;
6. complete the durable Job only after the upstream stream ends successfully.

## Gateway SSE contract

For `stream: true`, `POST /v1/chat/completions` submits one durable Job and
returns an OpenAI-compatible `text/event-stream`.

The gateway reads only events for that Job in sequence order, rewrites the
public `model` field to the logical serving profile ID, and emits:

```text
data: {json chunk}\n\n
...
data: [DONE]\n\n
```

`[DONE]` is emitted only after the durable Job reaches SUCCEEDED and every
persisted event has been delivered. A terminal failure/cancellation after any
visible chunk terminates the SSE stream without `[DONE]`; the gateway never
fabricates success or retries another attempt.

Client disconnect/cancellation cancels only the owned durable Job. Subsequent
Worker event publication is fenced by Control.

## Pinned OpenCode contract

Initial client target: OpenCode v1.18.30.

Exact source pin:

- session `streamText(...)`: Git blob
  `a99f8acff20c5d64d0b6cb90df480218bb1daddc`;
- OpenAI-compatible chat protocol: Git blob
  `9ac85b07b139f2a7a87f1a62d829a274b9cfd1ca`.

That parser expects SSE JSON chunks with:

- `choices[].delta.content`;
- incremental `choices[].delta.tool_calls[]` using stable indexes/IDs and
  string argument fragments;
- `choices[].finish_reason`;
- optional usage, including a final usage-only chunk;
- normal SSE termination.

The gateway deliberately emits only this verified subset.

## Acceptance

Automated tests must prove:

- ordered event sequence and bounded buffers;
- stale lease/epoch and cancelled-attempt publication rejection;
- real SSE framing/order and `[DONE]` only after durable success;
- text-delta streaming;
- incremental structured tool calls with no duplicate replay;
- usage/finish reason preservation;
- context overflow before upstream generation;
- disconnect/cancellation fencing;
- provider failure after partial output terminates without retry or `[DONE]`;
- non-streaming Stage A behavior remains unchanged.

A real OpenCode/model smoke remains a separate evidence level and requires an
exact runtime build/model/quantization/template identity.
