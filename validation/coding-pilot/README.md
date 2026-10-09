# Coding pilot fixture

This directory contains the synthetic public fixture used by the first #97
coding pilot. It is deliberately unrelated to any private repository.

The six baseline source/test files define one immutable baseline. The three
`check_*.py` programs are the task-specific correctness authorities.

Do not add raw OpenCode prompts, JSON event streams, tool payloads, local paths,
credentials or private repository material here. Those remain private run data.

See `docs/coding-pilot.md` for the canonical baseline/checker digests, budgets,
lane semantics and evidence procedure.

The later genuinely observed direct-remote lane is recorded separately in
[`direct-remote-v1.md`](direct-remote-v1.md), without rewriting the historical
[`coding-v1.md`](coding-v1.md) snapshot.