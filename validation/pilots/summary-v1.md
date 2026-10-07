# AstrumWeaver Bounded Consumer Pilot Roll-up

Tracking: #91 / #97.

This report rolls up the canonical private-safe evidence already accepted for
the first single-Worker coding, embedding, and System-One decision pilots. It
does not replace the underlying evidence files and does not infer missing
remote-provider usage from local observations.

## Current adapter dispositions

| Adapter | Canonical evidence | Observed result | Disposition |
| --- | --- | --- | --- |
| coding | `validation/coding-pilot/coding-v1.md` | delegated-local 2/3 accepted; direct-remote 0 observed / 3 NOT_RUN | experimental |
| embedding | `validation/embedding/embedding-v1.md` | disposable retrieval 4/4 top-1 correct; incompatible space rejected | accepted |
| decision | `validation/decision/shadow-v1.md` | shadow protocol PASS; 2/3 correct; deliberate-unknown abstention 0/1; false-safe 0 | experimental |

Dispositions apply only to the exact versioned scopes in the referenced
evidence. They are not claims about arbitrary models, deployments, repositories,
indexes, clients, or future revisions.

## Coding pilot

The pinned delegated-local coding lane observed three bounded tasks at
concurrency 1:

- 2 accepted;
- 1 rejected by the independent correctness checker;
- accepted rate 0.667;
- median wall time 432355.98 ms;
- median queue wait 288.91 ms;
- median TTFT 3015.54 ms;
- median generation time 424721.63 ms.

The rejection is retained as a model-quality observation. Tool completion is
not treated as correctness.

The equivalent `direct_remote` lane remains explicitly NOT_RUN for all three
tasks. Therefore remote planning/review/repair counts, observed remote usage,
and any remote-usage reduction remain UNKNOWN rather than inferred from local
token or request counts.

## Embedding pilot

The pinned embedding-space pilot used a separate disposable retrieval fixture:

- 4 documents and 4 queries;
- 4/4 top-1 retrieval correct;
- minimum observed similarity margin 0.250874 against a required 0.100000;
- incompatible embedding-space request rejected before durable admission.

No production index was read, migrated, replaced, or reused. The accepted
disposition is limited to the exact embedding-space identity and preprocessing
contract recorded in the canonical evidence.

## Decision shadow pilot

The pinned native System-One shadow run observed:

- live identity and native provider-score validation PASS;
- 3 labelled cases, all answered;
- 2 correct / 1 incorrect;
- deliberate-unknown abstention 0/1;
- false-safe count 0;
- changed-order probe did not change the selected decision;
- median wall time 22521.894 ms;
- median Worker/provider execution 22462.674 ms;
- median queue wait 13.030 ms;
- median wall-minus-execution overhead 69.910 ms.

The decision disposition remains experimental and `OBSERVED_ONLY`.
Choice-set probabilities are explicitly uncalibrated and grant no execution,
permission, review, merge, release, billing, or final-acceptance authority.

## Remaining #97 closure dependency

The remaining evidence gap is the genuine coding `direct_remote` comparison
lane on the same three pinned task baselines and acceptance checkers.

Until that lane is actually observed:

- direct-remote accepted rate and latency remain NOT_RUN/UNKNOWN;
- remote planning/review/repair remains UNKNOWN;
- observed remote usage remains UNKNOWN;
- electrical/operational cost remains UNKNOWN;
- no remote-usage reduction percentage can be stated;
- no paid-plan downgrade conclusion is supported.

A simulated remote model, local token accounting, or inferred quota savings
must not fill this gap.

## Current technical conclusion

The serving layer has demonstrated bounded usefulness in all three consumer
classes, but with mixed quality:

- coding: useful but experimental;
- embedding: accepted for the pinned disposable-space scope;
- decision: protocol-accepted but model quality remains experimental.

This is sufficient to continue technical experimentation. It is not sufficient
to close #97 or justify a subscription change until the direct-remote coding
baseline is independently observed.
