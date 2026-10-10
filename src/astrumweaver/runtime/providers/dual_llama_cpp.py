"""Experimental, fail-closed joint ownership of two llama.cpp runtimes.

One Worker owns one GPU and one runtime instance, with two independently pinned
child processes. This provider never creates a second Worker/GPU owner or relaxes
generic Control claim, lease, or serving-contract fencing.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import hashlib
import os
import secrets
import stat
import tempfile
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
from .llama_cpp import (
    HttpLlamaCppApi, LlamaCppProvider, LlamaCppProviderConfig,
    LlamaCppSplitMode, LlamaCppSubprocessController,
)


DUAL_LLAMA_CPP_PROVIDER_ID = "llama-cpp-dual"
_DUAL_CAPABILITIES = frozenset({"text.embed", "decision.system_one"})
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_NIX_STORE_OUTPUT = re.compile(r"^/nix/store/[a-z0-9]{32}-[A-Za-z0-9+._-]+$")


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
    embedding_space_id: str
    decision_semantics_id: str
    # Zero is the safe production default: installed model bytes are owned
    # by root, not the Worker/Downloader identity. Test-only fixtures may
    # explicitly name their own trusted deployment owner.
    trusted_model_owner_uid: int = 0

    def __post_init__(self) -> None:
        if (type(self.trusted_model_owner_uid) is not int
                or self.trusted_model_owner_uid < 0):
            raise ValueError("trusted_model_owner_uid must be a nonnegative integer")
        for name in ("embedding_space_id", "decision_semantics_id"):
            value = getattr(self, name)
            if type(value) is not str or not value.startswith("sha256:") or not _SHA256.fullmatch(value[7:]):
                raise ValueError("dual runtime requires exact pinned per-capability semantics")
        if self.embedding_space_id == self.decision_semantics_id:
            raise ValueError("dual role semantics must remain distinct")
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
                or _NIX_STORE_OUTPUT.fullmatch(str(package)) is None
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
        trusted_model_owner_uid: int = 0,
        private_auth_files: tempfile.TemporaryDirectory[str] | None = None,
    ) -> None:
        self._private_auth_files = private_auth_files
        self._embedding = embedding
        self._decision = decision
        self._pins = (embedding_pin, decision_pin)
        self._trusted_model_owner_uid = trusted_model_owner_uid
        self._executor = DualLlamaCppExecutor(embedding, decision)
        self._released = False
        # Public callers can overlap lifecycle operations. Keep composite
        # ownership serialized and invalidate pending starts on shutdown.
        self._lifecycle_lock = asyncio.Lock()
        self._lifecycle_epoch = 0
        self._starting_task: asyncio.Task[None] | None = None
        self._owned_stopped = True

    @staticmethod
    def _check_pinned_model(
        model: DualModelPin, *, trusted_owner_uid: int = 0,
    ) -> None:
        """Require a trusted, non-substitutable pathname as well as exact bytes.

        Checking SHA-256 on an fd before/after subprocess startup alone does
        not bind the fd to the child's later --model pathname. Instead require
        that no actor outside the reviewed file owner (root by default) can
        rename or mutate any entry on the path. A root-owned sticky directory
        such as /tmp is safe only for a trusted-owned child entry.
        """
        path = Path(model.model_ref)
        if ".." in path.parts:
            raise RuntimeError("prepared dual model path may not traverse parents")
        # lstat every component, rejecting symlinks and non-directory
        # ancestors. A statically trusted owner is a necessary precondition
        # before the child can safely resolve the path independently.
        current = Path(path.anchor)
        nodes = [current]
        for part in path.parts[1:]:
            current = current / part
            nodes.append(current)
        for idx, node in enumerate(nodes):
            try:
                info = node.lstat()
            except OSError as exc:
                raise RuntimeError("prepared dual model path cannot be inspected") from exc
            final = idx == len(nodes) - 1
            if info.st_uid not in (0, trusted_owner_uid):
                raise RuntimeError("prepared dual model path has untrusted owner")
            if final:
                if not stat.S_ISREG(info.st_mode):
                    raise RuntimeError("prepared dual model must be a regular file")
                if info.st_uid != trusted_owner_uid:
                    raise RuntimeError("prepared dual model file owner is not pinned")
            elif not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("prepared dual model ancestor must be a real directory")
            writable_by_others = info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            # Sticky, root-owned /tmp prevents another UID from renaming an
            # entry owned by the trusted uid. Nonsticky writable parents fail.
            trusted_sticky = (
                not final and info.st_uid == 0 and
                bool(info.st_mode & stat.S_ISVTX)
            )
            if writable_by_others and not trusted_sticky:
                raise RuntimeError("prepared dual model path permits untrusted writes")
        h = hashlib.sha256()
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise RuntimeError("prepared dual model cannot be opened safely") from exc
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if (not stat.S_ISREG(info.st_mode)
                    or info.st_uid != trusted_owner_uid
                    or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
                raise RuntimeError("prepared dual model inode is untrusted")
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != model.sha256:
            raise RuntimeError("prepared dual model digest mismatch")

    async def start(self) -> None:
        generation = self._lifecycle_epoch
        async with self._lifecycle_lock:
            if generation != self._lifecycle_epoch:
                raise RuntimeError("dual runtime start invalidated by concurrent shutdown")
            if self._released:
                raise RuntimeError("released dual runtime cannot be restarted")
            self._starting_task = asyncio.current_task()
            try:
                await self._start_serial()
            finally:
                self._starting_task = None

    def _invalidate_pending_start(self) -> None:
        # Invalidate start requests already queued before a stop/release. The
        # active start handles its own fail-closed child cleanup on cancellation.
        self._lifecycle_epoch += 1
        starting = self._starting_task
        current = asyncio.current_task()
        if starting is not None and starting is not current and not starting.done():
            starting.cancel()

    async def _start_serial(self) -> None:
        if self._released:
            raise RuntimeError("released dual runtime cannot be restarted")
        try:
            for pin in self._pins:
                await asyncio.to_thread(
                    self._check_pinned_model, pin,
                    trusted_owner_uid=self._trusted_model_owner_uid,
                )
            self._owned_stopped = False
            await self._embedding.start()
            await self._decision.start()
            if not (await self.health()).ready:
                raise RuntimeError("both resident runtimes did not become jointly ready")
            # Model files must not change between preflight and server load.
            # This closes ordinary operator/download races; privileged hostile
            # mutation of trusted model storage is outside Worker authority.
            for pin in self._pins:
                await asyncio.to_thread(
                    self._check_pinned_model, pin,
                    trusted_owner_uid=self._trusted_model_owner_uid,
                )
        except BaseException as startup_failure:
            # Preserve the primary failure/cancellation even if one owned
            # child also fails to stop. A failed cleanup still makes startup
            # fail closed; the note retains its failure type for diagnosis.
            try:
                await self._stop_both()
            except BaseException as cleanup_failure:
                startup_failure.add_note(
                    "dual runtime startup cleanup also failed: "
                    + type(cleanup_failure).__name__
                )
            raise

    async def _stop_both(self) -> None:
        results = await asyncio.gather(
            self._decision.stop(), self._embedding.stop(), return_exceptions=True,
        )
        if any(isinstance(x, BaseException) for x in results):
            raise RuntimeError("joint runtime cleanup failed")
        self._owned_stopped = True

    async def stop(self) -> None:
        self._invalidate_pending_start()
        async with self._lifecycle_lock:
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
        self._invalidate_pending_start()
        async with self._lifecycle_lock:
            if self._released:
                return
            # Direct or overlapping release must not close native API clients
            # while a child is still alive. Existing stop-then-release callers
            # avoid a duplicate stop through the owned-stopped latch.
            if not self._owned_stopped:
                await self._stop_both()
            results = await asyncio.gather(
                self._embedding.release(), self._decision.release(), return_exceptions=True,
            )
            if any(isinstance(x, BaseException) for x in results):
                raise RuntimeError("joint runtime release failed")
            if self._private_auth_files is not None:
                self._private_auth_files.cleanup()
                self._private_auth_files = None
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
        *, api_key: str | None = None, api_key_file: str | None = None,
    ) -> tuple[LlamaCppProvider, RuntimeCompatibilityContext]:
        model = (
            self.config.embedding_model if role == "embedding" else self.config.decision_model
        )
        child_config = (
            self.config.embedding_provider if role == "embedding"
            else self.config.decision_provider
        )
        if (api_key is None) != (api_key_file is None):
            raise ValueError("private inference authentication must be complete")
        if api_key is not None:
            # Never serialize ephemeral bearer secrets into the runtime manifest.
            provider = LlamaCppProvider(
                child_config,
                api_factory=lambda url: HttpLlamaCppApi(
                    url, timeout_seconds=child_config.request_timeout_seconds,
                    api_key=api_key,
                ),
                process_factory=lambda **kwargs: LlamaCppSubprocessController(
                    **kwargs, api_key_file=api_key_file,
                ),
            )
        else:
            # Compatibility and setup inspection do not materialize GPU servers.
            provider = LlamaCppProvider(child_config)
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
                "embedding_space_id": self.config.embedding_space_id,
                "decision_semantics_id": self.config.decision_semantics_id,
                "trusted_model_owner_uid": self.config.trusted_model_owner_uid,
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
        # Only the Worker service UID can enter this ephemeral 0700 directory.
        # Key values never appear in a subprocess command line, environment,
        # reviewed manifest, or the Control serving advertisement.
        owned_auth = tempfile.TemporaryDirectory(prefix="tsumgi-dual-llama-auth-")
        try:
            def key_for(role: str) -> tuple[str, str]:
                value = secrets.token_urlsafe(48)
                path = Path(owned_auth.name) / (role + ".key")
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(path, flags, 0o600)
                with os.fdopen(fd, "w", encoding="ascii") as handle:
                    handle.write(value + "\n")
                return value, str(path)

            embed_key, embed_file = key_for("embedding")
            decision_key, decision_file = key_for("decision")
            embedding_provider, embed_context = self._child(
                context, "embedding", api_key=embed_key, api_key_file=embed_file,
            )
            decision_provider, decision_context = self._child(
                context, "decision", api_key=decision_key, api_key_file=decision_file,
            )
            embedding_setup = embedding_provider.setup_intent(embed_context)
            decision_setup = decision_provider.setup_intent(decision_context)
        except BaseException:
            owned_auth.cleanup()
            raise
        try:
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
                trusted_model_owner_uid=self.config.trusted_model_owner_uid,
                private_auth_files=owned_auth,
            )
        except BaseException:
            owned_auth.cleanup()
            raise
