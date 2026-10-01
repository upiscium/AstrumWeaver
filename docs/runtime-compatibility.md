# Runtime Compatibility Matrix

This document is the human-readable view of the canonical machine-readable
matrix at [`acceptance/runtime-providers-v0.1.toml`](../acceptance/runtime-providers-v0.1.toml).

The matrix describes **AstrumWeaver v0.x provider contract behavior**. It is
not a universal certification of every upstream runtime version, GPU model,
driver, model architecture, quantization, or performance characteristic.

## Status vocabulary

- **validated** — AstrumWeaver has automated contract coverage for this claim.
- **supported, unvalidated** — the provider does not reject the case, but
  AstrumWeaver does not publish a validation claim for that axis.
- **unsupported** — the AstrumWeaver v0.x provider deliberately rejects the
  case.
- **integrated** — the provider participates in the reviewed NixOS or generic
  systemd RuntimeProvider deployment path. This does not mean every upstream
  binary/package combination has been hardware-certified.

The matrix intentionally separates ordinary multi-GPU support from
**heterogeneous GPU validation**. Aggregate VRAM is not enough to prove a
homogeneous or heterogeneous runtime claim.

## Summary

| Provider | Formats | GPU topology | Model topology | Residency | CPU offload | Multi-GPU hardware claim |
| --- | --- | --- | --- | --- | --- | --- |
| Ollama | `ollama` | single ✓, multi ✓ | dense ✓, MoE unvalidated | VRAM-only ✓, prefer-VRAM ✓, hybrid ✓ | automatic | multi-GPU validated; heterogeneous hardware unvalidated |
| llama.cpp | `gguf` | single ✓, multi ✓ | dense ✓, MoE ✓ | VRAM-only ✓, prefer-VRAM ✓, hybrid ✓ | explicit or automatic | **heterogeneous multi-GPU validated** |
| vLLM | `hf`, `huggingface`, `safetensors`, `vllm` | single ✓, multi ✓ | dense ✓, MoE ✓ | VRAM-only ✓, prefer-VRAM ✓, hybrid ✓ | explicit | **homogeneous multi-GPU required** |
| FreeToken | `ftw`, `hf`, `huggingface`, `safetensors` | single ✓, multi ✗ | dense ✗, MoE ✓ | VRAM-only ✗, prefer-VRAM ✓, hybrid ✓ | MoE policy | multi-GPU unsupported |
| ExLlamaV3 | `exl3`, `exllamav3` | single ✓, multi ✓ | dense ✓, MoE unvalidated | VRAM-only ✓, prefer-VRAM ✓, hybrid ✗ | unsupported | multi-GPU validated; heterogeneous hardware unvalidated |

All five first-class providers expose the generic `llm.chat` and
`text.generate` Worker capabilities.

## Provider notes

### Ollama

Ollama is the general/easy local-serving path. AstrumWeaver can validate
single-GPU execution, explicit multi-GPU spread, VRAM-only planning, and the
automatic CPU/GPU hybrid behavior exposed by Ollama.

AstrumWeaver does **not** infer that arbitrary heterogeneous GPU combinations
are validated merely because Ollama can be asked to spread across multiple
visible GPUs.

### llama.cpp

llama.cpp is the primary generic GGUF provider. AstrumWeaver validates:

- single-GPU execution;
- multi-GPU layer/row/tensor split policy;
- heterogeneous multi-GPU auto-fit behavior;
- dense and MoE provider scopes;
- VRAM-only, prefer-VRAM, and CPU/GPU hybrid execution.

The heterogeneous claim is allowed because the provider does not replace
per-device evidence with aggregate VRAM arithmetic and has dedicated
heterogeneous multi-GPU contract coverage.

`split_mode=tensor` remains an upstream-experimental path and is reported as
an advisory compatibility reason.

### vLLM

vLLM is the GPU-resident/high-throughput provider. Multi-GPU v0.x is
deliberately homogeneous-only.

For a multi-GPU claim, AstrumWeaver requires `WorkerSpec.accelerators` to
provide one auditable record per Worker-owned GPU and requires:

- equal per-device VRAM;
- the same known compute capability;
- the same known device class.

Missing per-device facts fail closed. Mixed VRAM, compute capability, or
device class also fail closed.

CPU/GPU hybrid mode requires an explicit `cpu_offload_gb` budget. This mode
is interconnect-sensitive and should not be interpreted as a generic
llama.cpp-style offload replacement.

### FreeToken

AstrumWeaver v0.x intentionally scopes FreeToken to **single-GPU MoE/offload**
workloads.

The provider accepts Hugging Face/safetensors references and local FTW
checkpoints. Dense-model, multi-GPU, and VRAM-only rejection are AstrumWeaver
provider-scope decisions; they are not claims that upstream FreeToken can
never support those cases.

### ExLlamaV3

ExLlamaV3 is scoped to EXL3/ExLlamaV3 quantized, VRAM-resident or
prefer-VRAM execution.

The current Linux package path targets x86_64. When compute-capability
evidence is known, the provider requires NVIDIA compute capability 8.0 or
newer.

Single-GPU and multi-GPU autosplit/tensor-parallel contracts are validated.
AstrumWeaver does not currently publish a heterogeneous-GPU validation claim
for ExLlamaV3.

## Lifecycle and deployment

For every first-class provider, the acceptance matrix records automated
evidence for:

- compatibility/incompatibility behavior;
- preservation of explicit operator selection;
- deterministic SetupPlan generation;
- executor cancellation;
- ManagedRuntime start/stop/release behavior;
- health/residency reporting;
- NixOS RuntimeProvider integration;
- generic systemd RuntimeProvider integration.

The cross-provider acceptance suite is
[`tests/test_runtime_provider_acceptance.py`](../tests/test_runtime_provider_acceptance.py).
It additionally proves that:

- every matrix provider exists in the real provider catalog surface;
- every listed model format is accepted by that provider scope;
- unsupported model formats fail without silent provider substitution;
- validated single/multi-GPU, model-topology, and residency claims match the
  provider's actual compatibility result;
- vLLM's homogeneous multi-GPU claim requires auditable per-device facts;
- only llama.cpp is marked as validated for heterogeneous multi-GPU in v0.x;
- every matrix evidence reference resolves to a real test/check;
- Control contains no branch on first-class runtime provider brands.

## Deployment boundary

NixOS and generic systemd integration means AstrumWeaver can persist and
activate the reviewed RuntimeProvider contract on an already GPU-ready node.

It does not install or replace the NVIDIA host driver, provision a VM/LXC,
change passthrough/IOMMU configuration, or certify runtime performance.
