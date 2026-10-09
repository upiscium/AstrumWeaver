"""Experimental, fail-closed joint ownership of two llama.cpp runtimes.

One Worker owns one GPU and one runtime instance, with two independently pinned
child processes. This provider never creates a second Worker/GPU owner or relaxes
generic Control claim, lease, or serving-contract fencing.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any, Mapping

from ...execution import JobExecutionError, JobExecutor, JobRequest, JobResult, ResidencyReport
from ..contracts import (
    CompatibilityReason, ExecutionDemand, GPUTopology, ManagedRuntime,
    ModelDemand, ModelTopology, ModelPreparationPolicy, ResidencyPolicy,
    RuntimeCompatibility, RuntimeCompatibilityContext, RuntimeHealth,
    RuntimeHealthState, RuntimeProviderInfo, RuntimeSetupIntent,
)
from .llama_cpp import LlamaCppProvider, LlamaCppProviderConfig, LlamaCppSplitMode


DUAL_LLAMA_CPP_PROVIDER_ID = "llama-cpp-dual"
_DUAL_CAPABILITIES = frozenset({"text.embed", "decision.system_one"})
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True, slots=True)
class DualModelPin:
    model_ref: str
    sha256: str
    estimated_size_mb: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_ref, str) or not Path(self.model_ref).is_absolute():
            raise ValueError("dual model ref must be an absolute prepared path")
        if not self.model_ref.endswith(".gguf"):
            raise ValueError("dual models must be prepared GGUF files")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValueError("dual model sha256 must be an exact lowercase digest")
        if type(self.estimated_size_mb) is not int or self.estimated_size_mb < 1:
            raise ValueError("dual model estimated_size_mb must be positive")


@dataclass(frozen=True, slots=True)
class DualLlamaCppProviderConfig:
    embedding_model: DualModelPin | Mapping[str, Any]
    decision_model: DualModelPin | Mapping[str, Any]
    embedding_provider: LlamaCppProviderConfig | Mapping[str, Any]
    decision_provider: LlamaCppProviderConfig | Mapping[str, Any]
    required_total_vram_mb: int

    def __post_init__(self) -> None:
        for attr in ("embedding_model", "decision_model"):
            value = getattr(self, attr)
            if isinstance(value, Mapping):
                value = DualModelPin(**dict(value))
            if not isinstance(value, DualModelPin):
                raise TypeError(f"{attr} must be a DualModelPin")
            object.__setattr__(self, attr, value)
        for attr in ("embedding_provider", "decision_provider"):
            value = getattr(self, attr)
            if isinstance(value, Mapping):
                value = LlamaCppProviderConfig(**dict(value))
            if not isinstance(value, LlamaCppProviderConfig):
                raise TypeError(f"{attr} must be LlamaCppProviderConfig")
            object.__setattr__(self, attr, value)
        if type(self.required_total_vram_mb) is not int or self.required_total_vram_mb <= 0:
            raise ValueError("combined VRAM reservation must be positive")
        emb = self.embedding_provider
        dec = self.decision_provider
        if not emb.embeddings or emb.decision or emb.pooling != "last":
            raise ValueError("first dual provider requires an embedding-only last-pooling child")
        if not dec.decision or dec.embeddings:
            raise ValueError("first dual provider requires a native decision-only child")
        endpoint_a, endpoint_b = urlsplit(emb.base_url), urlsplit(dec.base_url)
        # 'localhost' and '127.0.0.1' refer to the same GPU-local port.
        if endpoint_a.port == endpoint_b.port:
            raise ValueError("dual models require different owned loopback ports")
        for cfg, endpoint in ((emb, endpoint_a), (dec, endpoint_b)):
            if endpoint.username or endpoint.password or endpoint.path not in ("", "/") or endpoint.query or endpoint.fragment:
                raise ValueError("dual runtime endpoint must be a bare credential-free loopback URL")
            exe = Path(cfg.executable)
            package = Path(cfg.package_reference)
            if (not exe.is_absolute() or exe.name != "astrumweaver-llama-server"
                or not str(package).startswith("/nix/store/")
                or exe != package / "bin" / "astrumweaver-llama-server"):
                raise ValueError("dual runtime must use an exact Nix-packaged credential-isolating launcher")
        if self.embedding_model.model_ref == self.decision_model.model_ref:
            raise ValueError("dual models must use distinct prepared files")
        for cfg in (emb, dec):
            if not cfg.offline or not cfg.no_webui:
                raise ValueError("dual models must be offline with no web interface")
            if cfg.gpu_layers != "all" or cfg.fit is not False:
                raise ValueError("dual models require fully GPU resident offload with fit off")
            if cfg.split_mode not in (None, LlamaCppSplitMode.NONE) or cfg.main_gpu != 0:
                raise ValueError("dual models require one selected GPU with no split")
            if cfg.context_size < 1:
                raise ValueError("dual model context must be explicitly positive")
        min_weights = (
            self.embedding_model.estimated_size_mb + self.decision_model.estimated_size_mb
        )
        if self.required_total_vram_mb < min_weights + 512:
            raise ValueError("combined GPU reservation must include context/scratch headroom")

    @property
    def bundle_sha256(self) -> str:
        """Binding of both pinned model artifacts in fixed role order."""
        material = {
            "schema": "dual-llama-cpp-models-v1",
            "embedding_sha256": self.embedding_model.sha256,
            "decision_sha256": self.decision_model.sha256,
        }
        return hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def bundle_model_ref(self) -> str:
        return "dual-sha256:" + self.bundle_sha256


class DualLlamaCppExecutor(JobExecutor):
    """Capability-fenced dispatch; one Worker is the concurrency authority."""

    capabilities = _DUAL_CAPABILITIES

    def __init__(self, embedding: ManagedRuntime, decision: ManagedRuntime) -> None:
        self._executors = {
            "text.embed": embedding.executor(),
            "decision.system_one": decision.executor(),
        }
        for capability, executor in self._executors.items():
            if capability not in getattr(executor, "capabilities", ()):
                raise ValueError("dual child executor capability mismatch")
        self.serving_features = {
            capability: frozenset(getattr(executor, "serving_features", {}).get(capability, ()))
            for capability, executor in self._executors.items()
        }

    def _select(self, capability: str) -> JobExecutor:
        try:
            return self._executors[capability]
        except KeyError:
            raise JobExecutionError(
                "unsupported_dual_capability",
                "joint runtime cannot execute an unadvertised capability",
                retryable=False,
            ) from None

    async def execute(self, job: JobRequest) -> JobResult:
        return await self._select(job.capability).execute(job)

    async def cancel(self, job_id: str) -> None:
        # At most one of the two children has a Worker-owned active attempt.
        results = await asyncio.gather(
            *(executor.cancel(job_id) for executor in self._executors.values()),
            return_exceptions=True,
        )
        if any(isinstance(x, BaseException) for x in results):
            raise RuntimeError("dual runtime cancellation failed")

    async def residency(self) -> ResidencyReport:
        reports = await asyncio.gather(
            *(executor.residency() for executor in self._executors.values())
        )
        return ResidencyReport(
            items=tuple(item for report in reports for item in report.items),
            metadata={"runtime_provider": DUAL_LLAMA_CPP_PROVIDER_ID, "model_count": 2},
        )


class DualLlamaCppManagedRuntime(ManagedRuntime):
    provider_id = DUAL_LLAMA_CPP_PROVIDER_ID

    def __init__(
        self, embedding: ManagedRuntime, decision: ManagedRuntime,
        embedding_pin: DualModelPin, decision_pin: DualModelPin,
    ) -> None:
        self._embedding = embedding
        self._decision = decision
        self._pins = (embedding_pin, decision_pin)
        self._executor = DualLlamaCppExecutor(embedding, decision)
        self._released = False

    @staticmethod
    def _check_pinned_model(model: DualModelPin) -> None:
        path = Path(model.model_ref)
        if not path.is_file():
            raise RuntimeError("prepared dual model is missing")
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != model.sha256:
            raise RuntimeError("prepared dual model digest mismatch")

    async def start(self) -> None:
        if self._released:
            raise RuntimeError("released dual runtime cannot be restarted")
        try:
            for pin in self._pins:
                await asyncio.to_thread(self._check_pinned_model, pin)
            await self._embedding.start()
            await self._decision.start()
            if not (await self.health()).ready:
                raise RuntimeError("both resident runtimes did not become jointly ready")
        except BaseException:
            # A partially started child must not remain resident or advertised.
            await self._stop_both()
            raise

    async def _stop_both(self) -> None:
        results = await asyncio.gather(
            self._decision.stop(), self._embedding.stop(), return_exceptions=True,
        )
        if any(isinstance(x, BaseException) for x in results):
            raise RuntimeError("joint runtime cleanup failed")

    async def stop(self) -> None:
        await self._stop_both()

    async def health(self) -> RuntimeHealth:
        results = await asyncio.gather(
            self._embedding.health(), self._decision.health(), return_exceptions=True,
        )
        if any(isinstance(x, BaseException) for x in results):
            return RuntimeHealth(RuntimeHealthState.FAILED, False,
                                 detail="joint runtime child health probe failed")
        first, second = results
        if first.ready and second.ready:
            return RuntimeHealth(
                RuntimeHealthState.READY, True,
                metadata={"runtime_provider": self.provider_id, "model_count": 2},
            )
        if first.state is RuntimeHealthState.STOPPED and second.state is RuntimeHealthState.STOPPED:
            return RuntimeHealth(RuntimeHealthState.STOPPED, False)
        if any(x.state in {RuntimeHealthState.FAILED, RuntimeHealthState.STOPPED} for x in results):
            return RuntimeHealth(RuntimeHealthState.FAILED, False,
                                 detail="one resident runtime is unavailable")
        return RuntimeHealth(RuntimeHealthState.DEGRADED, False,
                             detail="joint runtime is not fully ready")

    async def residency(self) -> ResidencyReport:
        return await self._executor.residency()

    def executor(self) -> JobExecutor:
        return self._executor

    async def release(self) -> None:
        if self._released:
            return
        results = await asyncio.gather(
            self._embedding.release(), self._decision.release(), return_exceptions=True,
        )
        if any(isinstance(x, BaseException) for x in results):
            raise RuntimeError("joint runtime release failed")
        self._released = True


class DualLlamaCppProvider:
    def __init__(self, config: DualLlamaCppProviderConfig | None = None) -> None:
        if config is None:
            raise ValueError("dual llama.cpp provider requires exact pinned configuration")
        self.config = config
        self._info = RuntimeProviderInfo(
            provider_id=DUAL_LLAMA_CPP_PROVIDER_ID,
            display_name="llama.cpp resident embedding + System-One",
            description="Two bounded llama.cpp processes owned by one GPU Worker.",
        )

    @property
    def info(self) -> RuntimeProviderInfo:
        return self._info

    def _child(
        self, context: RuntimeCompatibilityContext, role: str,
    ) -> tuple[LlamaCppProvider, RuntimeCompatibilityContext]:
        model = (
            self.config.embedding_model if role == "embedding" else self.config.decision_model
        )
        provider = LlamaCppProvider(
            self.config.embedding_provider if role == "embedding"
            else self.config.decision_provider
        )
        demand = ExecutionDemand(
            model=ModelDemand(
                model_ref=model.model_ref, model_format="gguf",
                topology=ModelTopology.DENSE, estimated_size_mb=model.estimated_size_mb,
            ),
            residency_policy=ResidencyPolicy.VRAM_ONLY,
            gpu_topology=GPUTopology.SINGLE_GPU,
            min_gpu_count=1,
            min_total_vram_mb=model.estimated_size_mb,
            min_single_gpu_vram_mb=model.estimated_size_mb,
        )
        return provider, RuntimeCompatibilityContext(
            worker=context.worker, host=context.host, demand=demand,
        )

    def compatibility(self, context: RuntimeCompatibilityContext) -> RuntimeCompatibility:
        reasons: list[CompatibilityReason] = []
        def error(code: str, message: str):
            reasons.append(CompatibilityReason(code=code, message=message))
        if context.worker.resources.gpu_count != 1:
            error("dual-requires-one-worker-gpu", "joint runtime requires exactly one Worker GPU")
        if not _DUAL_CAPABILITIES <= context.worker.capabilities:
            error("dual-worker-capabilities", "Worker does not advertise both resident capabilities")
        if context.demand.gpu_topology is not GPUTopology.SINGLE_GPU or context.demand.residency_policy is not ResidencyPolicy.VRAM_ONLY:
            error("dual-topology", "joint runtime requires a single fully-resident GPU")
        if context.demand.model.model_ref != self.config.bundle_model_ref:
            error("dual-bundle-identity", "deployment demand does not match both pinned model digests")
        if context.demand.model.model_format != "gguf":
            error("dual-model-format", "joint runtime bundle requires GGUF models")
        if context.worker.resources.max_single_gpu_vram_mb < self.config.required_total_vram_mb:
            error("dual-vram-budget", "Worker GPU has insufficient reserved combined VRAM")
        if context.demand.min_single_gpu_vram_mb < self.config.required_total_vram_mb:
            error("dual-demand-budget", "deployment demand must reserve combined GPU VRAM")
        for role in ("embedding", "decision"):
            provider, subcontext = self._child(context, role)
            report = provider.compatibility(subcontext)
            for reason in report.reasons:
                if reason.blocking:
                    error("dual-" + role + "-" + reason.code, reason.message)
        return RuntimeCompatibility(provider_id=DUAL_LLAMA_CPP_PROVIDER_ID, reasons=tuple(reasons))

    def setup_intent(self, context: RuntimeCompatibilityContext) -> RuntimeSetupIntent:
        if not self.compatibility(context).compatible:
            raise RuntimeError("dual runtime cannot prepare an incompatible Worker")
        packages = tuple(dict.fromkeys((
            self.config.embedding_provider.package_reference,
            self.config.decision_provider.package_reference,
        )))
        return RuntimeSetupIntent(
            provider_id=self.info.provider_id, package_references=packages,
            configuration={
                "bundle_sha256": self.config.bundle_sha256,
                "embedding_sha256": self.config.embedding_model.sha256,
                "decision_sha256": self.config.decision_model.sha256,
                "required_total_vram_mb": self.config.required_total_vram_mb,
            },
            model_preparation=ModelPreparationPolicy.REFERENCE_ONLY,
            model_ref=self.config.bundle_model_ref, requires_privilege=True,
        )

    def create_runtime(
        self, context: RuntimeCompatibilityContext, setup: RuntimeSetupIntent,
    ) -> ManagedRuntime:
        if not self.compatibility(context).compatible:
            raise RuntimeError("dual runtime deployment is incompatible with this Worker")
        expected = self.setup_intent(context)
        if setup != expected:
            raise RuntimeError("dual runtime setup intent does not match frozen model identity")
        embedding_provider, embed_context = self._child(context, "embedding")
        decision_provider, decision_context = self._child(context, "decision")
        embedding_setup = embedding_provider.setup_intent(embed_context)
        decision_setup = decision_provider.setup_intent(decision_context)
        embedding_setup = replace(
            embedding_setup, configuration={
                **embedding_setup.configuration, "model_alias": "tsumgi-embed-v1",
            },
        )
        decision_setup = replace(
            decision_setup, configuration={
                **decision_setup.configuration, "model_alias": "tsumgi-decision-v1",
            },
        )
        embedding = embedding_provider.create_runtime(embed_context, embedding_setup)
        decision = decision_provider.create_runtime(decision_context, decision_setup)
        return DualLlamaCppManagedRuntime(
            embedding, decision,
            self.config.embedding_model, self.config.decision_model,
        )
