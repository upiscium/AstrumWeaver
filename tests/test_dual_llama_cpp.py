"""Fail-closed, single-GPU two-resident-model provider contracts (#122)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from astrumweaver import ResourceShape, WorkerSpec
from astrumweaver.execution import (
    JobExecutionError, JobRequest, JobResult, ResidencyItem, ResidencyReport,
)
from astrumweaver.runtime import (
    ExecutionDemand, GPUTopology, ModelDemand, ModelTopology, ResidencyPolicy,
    RuntimeCompatibilityContext, RuntimeDeploymentSpec, RuntimeHostFacts,
    managed_runtime_from_deployment, provider_from_deployment,
)
from astrumweaver.runtime.contracts import RuntimeHealth, RuntimeHealthState
from astrumweaver.runtime.providers import (
    DUAL_LLAMA_CPP_PROVIDER_ID, DualLlamaCppManagedRuntime,
    DualLlamaCppProvider, DualLlamaCppProviderConfig, DualModelPin,
)


def setup_pair(tmp_path: Path) -> tuple[DualLlamaCppProvider, RuntimeCompatibilityContext]:
    embed = tmp_path / "embed.gguf"
    decision = tmp_path / "decision.gguf"
    embed.write_bytes(b"pinned-embedding-blob")
    decision.write_bytes(b"pinned-decision-blob")
    package = "/nix/store/pinned-test-llama"
    executable = package + "/bin/astrumweaver-llama-server"
    config = DualLlamaCppProviderConfig(
        embedding_model={
            "model_ref": str(embed),
            "sha256": hashlib.sha256(embed.read_bytes()).hexdigest(),
            "estimated_size_mb": 610,
        },
        decision_model={
            "model_ref": str(decision),
            "sha256": hashlib.sha256(decision.read_bytes()).hexdigest(),
            "estimated_size_mb": 1597,
        },
        embedding_provider={
            "base_url": "http://127.0.0.1:18441",
            "executable": executable,
            "package_reference": package,
            "embeddings": True,
            "pooling": "last",
            "gpu_layers": "all", "fit": False, "split_mode": "none",
        },
        decision_provider={
            "base_url": "http://127.0.0.1:18442",
            "executable": executable,
            "package_reference": package,
            "decision": True,
            "gpu_layers": "all", "fit": False, "split_mode": "none",
        },
        required_total_vram_mb=4096,
    )
    provider = DualLlamaCppProvider(config)
    context = RuntimeCompatibilityContext(
        worker=WorkerSpec(
            worker_id="dual-worker",
            worker_class="gpu-single",
            resources=ResourceShape(
                gpu_count=1, total_vram_mb=11264,
                max_single_gpu_vram_mb=11264,
            ),
            gpu_uuids=("GPU-dual-example",),
            capabilities=frozenset({"text.embed", "decision.system_one"}),
        ),
        host=RuntimeHostFacts(cpu_count=4, host_ram_mb=16384),
        demand=ExecutionDemand(
            model=ModelDemand(
                model_ref=config.bundle_model_ref,
                model_format="gguf", topology=ModelTopology.DENSE,
                estimated_size_mb=2207,
            ),
            residency_policy=ResidencyPolicy.VRAM_ONLY,
            gpu_topology=GPUTopology.SINGLE_GPU,
            min_gpu_count=1, min_single_gpu_vram_mb=4096,
            min_total_vram_mb=4096,
        ),
    )
    return provider, context


def test_dual_provider_binding_and_runtime_manifest_round_trip(tmp_path):
    provider, ctx = setup_pair(tmp_path)
    assert provider.compatibility(ctx).compatible
    intent = provider.setup_intent(ctx)
    assert intent.provider_id == DUAL_LLAMA_CPP_PROVIDER_ID
    assert intent.model_ref == provider.config.bundle_model_ref
    assert intent.configuration["embedding_sha256"] != intent.configuration["decision_sha256"]
    assert intent.package_references == ("/nix/store/pinned-test-llama",)
    deployment = RuntimeDeploymentSpec(
        provider_id=provider.info.provider_id,
        provider_config=vars_config(provider.config),
        demand=ctx.demand,
        setup_intent=intent,
    )
    restored = RuntimeDeploymentSpec.from_dict(deployment.to_dict())
    reconstructed = provider_from_deployment(restored)
    assert reconstructed.config.bundle_sha256 == provider.config.bundle_sha256
    runtime = managed_runtime_from_deployment(
        restored, worker=ctx.worker, host=ctx.host,
    )
    assert runtime.provider_id == DUAL_LLAMA_CPP_PROVIDER_ID
    assert runtime.executor().capabilities == frozenset({
        "text.embed", "decision.system_one",
    })
    assert runtime.executor().serving_features["text.embed"] == frozenset({
        "float", "pooling-last", "normalization-l2",
    })
    assert "native-decision-head" in runtime.executor().serving_features["decision.system_one"]


def vars_config(config: DualLlamaCppProviderConfig):
    from dataclasses import asdict
    return asdict(config)


def test_bundle_identity_rejects_unrelated_model_ref_and_vram_budget(tmp_path):
    provider, ctx = setup_pair(tmp_path)
    wrong = RuntimeCompatibilityContext(
        worker=ctx.worker, host=ctx.host,
        demand=ExecutionDemand(
            model=ModelDemand(
                model_ref="dual-sha256:" + "0" * 64,
                model_format="gguf", topology=ModelTopology.DENSE,
            ),
            residency_policy=ResidencyPolicy.VRAM_ONLY,
            gpu_topology=GPUTopology.SINGLE_GPU,
            min_gpu_count=1, min_total_vram_mb=4096, min_single_gpu_vram_mb=4096,
        ),
    )
    report = provider.compatibility(wrong)
    assert not report.compatible
    assert any(reason.code == "dual-bundle-identity" for reason in report.reasons)
    with pytest.raises(RuntimeError, match="incompatible"):
        provider.setup_intent(wrong)

    from dataclasses import replace
    small = replace(ctx.worker.resources, total_vram_mb=3200,
                    max_single_gpu_vram_mb=3200)
    reduced_worker = replace(ctx.worker, resources=small)
    report = provider.compatibility(RuntimeCompatibilityContext(
        worker=reduced_worker, host=ctx.host, demand=ctx.demand,
    ))
    assert not report.compatible
    assert any(reason.code == "dual-vram-budget" for reason in report.reasons)


def test_two_models_cannot_share_the_same_endpoint_or_mode(tmp_path):
    provider, ctx = setup_pair(tmp_path)
    from dataclasses import replace
    with pytest.raises(ValueError, match="different owned loopback ports"):
        replace(provider.config, decision_provider=replace(
            provider.config.decision_provider,
            base_url=provider.config.embedding_provider.base_url,
        ))
    with pytest.raises(ValueError, match="embedding-only"):
        replace(provider.config, embedding_provider=replace(
            provider.config.embedding_provider, embeddings=False, pooling=None,
        ))
    with pytest.raises(ValueError, match="native decision-only"):
        replace(provider.config, decision_provider=replace(
            provider.config.decision_provider, decision=False,
        ))
    with pytest.raises(ValueError, match="fully GPU resident"):
        replace(provider.config, decision_provider=replace(
            provider.config.decision_provider, fit=True,
        ))
    with pytest.raises(ValueError, match="context/scratch"):
        replace(provider.config, required_total_vram_mb=2208)


def test_dual_setup_cannot_rebind_after_review(tmp_path):
    provider, ctx = setup_pair(tmp_path)
    from dataclasses import replace
    original = provider.setup_intent(ctx)
    changed = replace(original, model_ref="dual-sha256:" + "a" * 64)
    with pytest.raises(RuntimeError, match="does not match frozen"):
        provider.create_runtime(ctx, changed)


class FakeExecutor:
    def __init__(self, capability):
        self.capabilities = frozenset({capability})
        self.serving_features = {capability: frozenset({"native-decision-head"} if capability == "decision.system_one" else {"float"})}
        self.received = []
        self.cancelled = []

    async def execute(self, job):
        self.received.append(job)
        return JobResult(outputs={"capability": job.capability})

    async def cancel(self, job_id):
        self.cancelled.append(job_id)

    async def residency(self):
        return ResidencyReport(items=(ResidencyItem(name="test-"+list(self.capabilities)[0], kind="gguf-model"),))


class FakeRuntime:
    def __init__(self, capability):
        self.provider_id = "llama-cpp"
        self.executor_value = FakeExecutor(capability)
        self.health_state = RuntimeHealthState.STOPPED
        self.fail_start = False
        self.stops = 0
        self.releases = 0

    def executor(self):
        return self.executor_value

    async def start(self):
        if self.fail_start:
            raise RuntimeError("synthetic child failure")
        self.health_state = RuntimeHealthState.READY

    async def stop(self):
        self.stops += 1
        self.health_state = RuntimeHealthState.STOPPED

    async def health(self):
        return RuntimeHealth(self.health_state, self.health_state == RuntimeHealthState.READY)

    async def release(self):
        self.releases += 1


def managed_pair(tmp_path):
    provider, _ = setup_pair(tmp_path)
    embedding = FakeRuntime("text.embed")
    decision = FakeRuntime("decision.system_one")
    managed = DualLlamaCppManagedRuntime(
        embedding, decision, provider.config.embedding_model, provider.config.decision_model,
    )
    return managed, embedding, decision


@pytest.mark.asyncio
async def test_joint_lifecycle_and_capability_fenced_dispatch(tmp_path):
    managed, embedding, decision = managed_pair(tmp_path)
    await managed.start()
    assert (await managed.health()).ready is True
    embed_job = JobRequest(job_id="e1", capability="text.embed")
    dec_job = JobRequest(job_id="d1", capability="decision.system_one")
    assert (await managed.executor().execute(embed_job)).outputs["capability"] == "text.embed"
    assert (await managed.executor().execute(dec_job)).outputs["capability"] == "decision.system_one"
    assert [x.job_id for x in embedding.executor_value.received] == ["e1"]
    assert [x.job_id for x in decision.executor_value.received] == ["d1"]
    with pytest.raises(JobExecutionError, match="unadvertised"):
        await managed.executor().execute(JobRequest(job_id="x1", capability="llm.chat"))
    assert len((await managed.residency()).items) == 2
    await managed.executor().cancel("e1")
    assert embedding.executor_value.cancelled == ["e1"]
    assert decision.executor_value.cancelled == ["e1"]
    await managed.stop()
    assert not (await managed.health()).ready
    assert embedding.stops == 1 and decision.stops == 1
    await managed.release()
    assert embedding.releases == 1 and decision.releases == 1
    await managed.release()
    assert embedding.releases == 1 and decision.releases == 1
    with pytest.raises(RuntimeError, match="released"):
        await managed.start()


@pytest.mark.asyncio
async def test_partial_start_failure_stops_both_owned_children(tmp_path):
    managed, embedding, decision = managed_pair(tmp_path)
    decision.fail_start = True
    with pytest.raises(RuntimeError, match="synthetic child failure"):
        await managed.start()
    assert embedding.stops == 1 and decision.stops == 1
    assert not (await managed.health()).ready


@pytest.mark.asyncio
async def test_health_latches_incomplete_residency_fail_closed(tmp_path):
    managed, embedding, decision = managed_pair(tmp_path)
    await managed.start()
    decision.health_state = RuntimeHealthState.FAILED
    health = await managed.health()
    assert not health.ready
    assert health.state == RuntimeHealthState.FAILED
    decision.health_state = RuntimeHealthState.READY
    embedding.health_state = RuntimeHealthState.STOPPED
    health = await managed.health()
    assert not health.ready and health.state == RuntimeHealthState.FAILED


@pytest.mark.asyncio
async def test_bad_pinned_model_digest_prevents_any_start(tmp_path):
    managed, embedding, decision = managed_pair(tmp_path)
    damaged = tmp_path / "decision.gguf"
    damaged.write_bytes(b"tampered model")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        await managed.start()
    assert embedding.health_state == RuntimeHealthState.STOPPED
    assert embedding.stops == 1 and decision.stops == 1


def test_worker_advertises_only_exact_dual_bundle_model_identity(tmp_path):
    from dataclasses import replace
    from astrumweaver.serving import (
        DeploymentIdentity, ServingContract, ServingDeploymentDeclaration,
    )
    from astrumweaver.worker.daemon import (
        _build_serving_advertisement, _require_runtime_concurrency,
    )

    provider, ctx = setup_pair(tmp_path)
    deployment = RuntimeDeploymentSpec(
        provider_id=provider.info.provider_id,
        provider_config=vars_config(provider.config),
        demand=ctx.demand,
        setup_intent=provider.setup_intent(ctx),
    )
    identity = DeploymentIdentity(
        provider_id=provider.info.provider_id,
        runtime_artifact_sha256="sha256:" + "a" * 64,
        adapter_artifact_sha256="sha256:" + "b" * 64,
        model_artifact_sha256="sha256:" + provider.config.bundle_sha256,
        execution_config_sha256="sha256:" + "c" * 64,
        quantization="dual-pinned",
    )
    def declaration(ident):
        return ServingDeploymentDeclaration(
            deployment=ident,
            contracts=(
                ServingContract(
                    deployment_revision=ident.revision,
                    capability="text.embed",
                    operation_schema="embedding-request-v1",
                    validation_evidence_sha256="sha256:" + "d" * 64,
                    semantic_revision="sha256:" + "e" * 64,
                ),
                ServingContract(
                    deployment_revision=ident.revision,
                    capability="decision.system_one",
                    operation_schema="decision-request-v1",
                    validation_evidence_sha256="sha256:" + "f" * 64,
                    semantic_revision="sha256:" + "1" * 64,
                ),
            ),
        )
    active = _build_serving_advertisement(
        declaration(identity), spec=ctx.worker,
        runtime_deployment=deployment,
    )
    assert len(active.contracts) == 2
    assert active.deployment_revision == identity.revision
    assert active.runtime_instance.epoch

    wrong = replace(identity, model_artifact_sha256="sha256:" + "0" * 64)
    with pytest.raises(RuntimeError, match="does not bind both pinned models"):
        _build_serving_advertisement(
            declaration(wrong), spec=ctx.worker, runtime_deployment=deployment,
        )
    _require_runtime_concurrency(deployment, 1)
    for amount in (0, 2):
        with pytest.raises(RuntimeError, match="max_concurrency=1"):
            _require_runtime_concurrency(deployment, amount)


def test_invalid_endpoint_alias_and_untrusted_launcher_are_rejected(tmp_path):
    from dataclasses import replace
    provider, _ = setup_pair(tmp_path)
    cfg = provider.config
    with pytest.raises(ValueError, match="different owned loopback ports"):
        replace(cfg, decision_provider=replace(
            cfg.decision_provider,
            base_url="http://localhost:18441",
        ))
    with pytest.raises(ValueError, match="credential-free"):
        replace(cfg, decision_provider=replace(
            cfg.decision_provider,
            base_url="http://user:secret@127.0.0.1:18442",
        ))
    with pytest.raises(ValueError, match="credential-isolating launcher"):
        replace(cfg, decision_provider=replace(
            cfg.decision_provider,
            executable="/bin/true",
        ))


@pytest.mark.asyncio
async def test_dual_release_and_stop_both_attempted_on_child_failures(tmp_path):
    managed, embedding, decision = managed_pair(tmp_path)
    def broken_stop():
        embedding.stops += 1
        raise RuntimeError("synthetic stop error")
    async def raise_error():
        return broken_stop()
    embedding.stop = raise_error
    with pytest.raises(RuntimeError, match="joint runtime cleanup failed"):
        await managed.stop()
    assert decision.stops == 1
