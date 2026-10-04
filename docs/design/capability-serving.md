# Deployment-bound capability serving

Status: design in progress; not an implemented API or installation procedure.

Tracking: [Epic #91](https://github.com/upiscium/AstrumWeaver/issues/91).

Inspected source baseline: `118baa14e525eb933e12bc0d79751c39ae66fd7f`.

This post-v0.1 design extends the existing generic compute fabric for coding-agent inference, embedding inference and experimental decision inference. It does not replace the Worker, RuntimeProvider, JobExecutor or durable job ownership contracts.

## Scope boundary

Worker remains the exclusive scheduler-visible resource owner. A single-GPU LXC is a deployment policy, not a second Slot registry or a restriction on supported topology. Infrastructure provisioning and GPU exposure remain operator-owned.

The design will specify deployment revisions and running instance identity, executable capability descriptors, logical serving profiles, bounded admission, workload adapters and consumer acceptance. Agent orchestration, repository execution, permission decisions and index management stay outside AstrumWeaver.

## Release separation

This work is independent of release tracker #79, repair #89 and operator-led Real Smoke #80. It authorizes no production mutation and satisfies no release or hardware gate. Installation instructions and runtime behavior remain unchanged.

The detailed contract and dependency-linked implementation plan are being prepared in this documentation-only PR. The approved direction must not be represented as already implemented support.
