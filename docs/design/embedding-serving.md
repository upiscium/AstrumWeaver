# Embedding serving contract

Tracking: #91 / #95. Shared serving substrate: #93.

## First implementation scope

The first explicitly supported embedding backend is llama.cpp running a dedicated
embedding model with `--embeddings`. This does not claim that every AstrumWeaver
runtime provider supports embeddings.

The public edge surface is a documented subset of `POST /v1/embeddings`:

- `model` is a configured logical serving profile ID;
- `input` is one string or a non-empty array of strings;
- `encoding_format` is omitted or exactly `float`;
- token-array input, base64 output, arbitrary dimensions and undocumented fields
  fail closed;
- AstrumWeaver adds a namespaced response extension carrying the immutable
  embedding-space identity.

## Immutable embedding-space identity

`embedding_space_id` is a canonical SHA-256 revision over all values that can
change vector semantics:

- exact deployment revision;
- exact model artifact SHA-256;
- quantization;
- tokenizer artifact SHA-256;
- pooling policy;
- normalization policy;
- output dimensions;
- query preprocessing/instruction policy;
- document preprocessing/instruction policy;
- provider adapter identity.

Equal dimensions or a similar model name never imply compatibility. A profile
revision resolves exactly one approved space. Changing any field above requires
a new space identity and client-owned reindex/migration.

The gateway accepts an explicit namespaced input role, `query` or `document`,
so a retrieval model may use different reviewed preprocessing for query and
document inputs without guessing from request position. Mixed roles inside one
batch are not accepted in v1; callers split them into separate bounded requests.

## Bounded execution

The adapter enforces:

- per-item UTF-8 byte limit;
- per-item token limit using the active llama.cpp tokenizer;
- maximum batch items;
- aggregate input bytes;
- aggregate input tokens;
- request bytes;
- exact output dimension;
- finite vector values;
- input/output index ordering.

A batch is all-or-error. No partial vector set is exposed.

The request compiles to a durable `text.embed` Job with the exact #93 serving
binding and deadline. Control remains workload-agnostic; tokenization and vector
semantics stay at the gateway/provider adapter boundary.

## llama.cpp mode

Embedding mode is explicit runtime configuration. A llama.cpp executor started
for embeddings advertises `text.embed` rather than silently inheriting
`llm.chat`/`text.generate`.

The managed process uses `--embeddings` and an explicit pooling policy. Runtime
readiness still verifies the configured model alias. Provider response handling
validates the OpenAI-compatible embedding response before durable success.

## Acceptance boundary

Schema/repository/provider tests must cover incompatible space identity,
profile mutation after enqueue, unsupported input/encoding/dimensions, query vs
document preprocessing, vector dimension/finite/order checks, batch/token/byte
overflow, cancellation and stale serving identity.

A later real acceptance run uses a disposable retrieval fixture and an
intentionally incompatible-space negative control. It does not modify a
production index.
