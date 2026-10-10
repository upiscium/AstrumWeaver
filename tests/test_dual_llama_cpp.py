"""Fail-closed, single-GPU two-resident-model provider contracts (#122)."""
from __future__ import annotations

import asyncio
import hashlib
import os
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
    package = "/nix/store/" + "0" * 32 + "-pinned-test-llama"
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
        embedding_space_id="sha256:" + "e" * 64,
        decision_semantics_id="sha256:" + "d" * 64,
        trusted_model_owner_uid=os.getuid(),
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
    assert intent.package_references == (
        "/nix/store/" + "0" * 32 + "-pinned-test-llama",
    )
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
        trusted_model_owner_uid=os.getuid(),
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
                    operation_schema="openai-embeddings-v1",
                    validation_evidence_sha256="sha256:" + "d" * 64,
                    semantic_revision="sha256:" + "e" * 64,
                ),
                ServingContract(
                    deployment_revision=ident.revision,
                    capability="decision.system_one",
                    operation_schema="astrumweaver-decisions-v1",
                    validation_evidence_sha256="sha256:" + "f" * 64,
                    semantic_revision="sha256:" + "d" * 64,
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

    from astrumweaver.worker.daemon import _require_dual_serving_manifest
    with pytest.raises(RuntimeError, match="reviewed serving manifest"):
        _require_dual_serving_manifest(deployment, None)
    _require_dual_serving_manifest(deployment, declaration(identity))

    # The declaration permits different operation schemas on one capability,
    # but one dual Worker must never accept hidden/extra contracts.
    valid = declaration(identity)
    extra_contract = replace(
        valid.contracts[0],
        operation_schema="unexpected-embedding-operation-v2",
        semantic_revision="sha256:" + "9" * 64,
    )
    with pytest.raises(RuntimeError, match="pinned child semantics"):
        _build_serving_advertisement(
            ServingDeploymentDeclaration(
                deployment=identity, contracts=valid.contracts+(extra_contract,),
            ),
            spec=ctx.worker, runtime_deployment=deployment,
        )

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


def test_dual_gateway_catalogs_bind_distinct_child_semantics(tmp_path):
    from dataclasses import replace
    from astrumweaver.gateway.embedding import (
        EMBEDDING_OPERATION_SCHEMA, LLAMA_CPP_EMBEDDING_ADAPTER,
        EmbeddingGatewayProfile, EmbeddingSpaceIdentity, text_policy_digest,
    )
    from astrumweaver.gateway.decision import (
        DECISION_OPERATION_SCHEMA, DECISION_SCORE_KIND, LLAMA_CPP_SCORE_SEMANTICS,
        LLAMA_CPP_SYSTEM_ONE_ADAPTER, DecisionGatewayProfile, DecisionSemanticsIdentity,
    )
    from astrumweaver.serving import DeploymentIdentity, LogicalServingProfile, ServingContract

    provider, ctx = setup_pair(tmp_path)
    cfg = provider.config
    dep = DeploymentIdentity(
        provider_id=provider.info.provider_id,
        runtime_artifact_sha256="sha256:" + "a" * 64,
        adapter_artifact_sha256="sha256:" + "b" * 64,
        model_artifact_sha256="sha256:" + cfg.bundle_sha256,
        execution_config_sha256="sha256:" + "c" * 64,
        quantization="composite-q8-q4",
        tokenizer_artifact_sha256="sha256:" + "7" * 64,
    )
    space = EmbeddingSpaceIdentity(
        deployment_revision=dep.revision,
        model_artifact_sha256="sha256:" + cfg.embedding_model.sha256,
        quantization="Q8_0",  # the embedding child, NOT the bundle
        tokenizer_artifact_sha256=dep.tokenizer_artifact_sha256,
        pooling="last", normalization="l2", dimensions=1024,
        query_policy_id="synthetic-query-v1",
        query_preprocess_sha256=text_policy_digest(""),
        document_policy_id="synthetic-document-v1",
        document_preprocess_sha256=text_policy_digest(""),
        adapter_id=LLAMA_CPP_EMBEDDING_ADAPTER,
    )
    assert space.model_artifact_sha256 != dep.model_artifact_sha256
    semantics = DecisionSemanticsIdentity(
        deployment_revision=dep.revision,
        adapter_id=LLAMA_CPP_SYSTEM_ONE_ADAPTER,
        score_kind=DECISION_SCORE_KIND,
        provider_score_semantics=LLAMA_CPP_SCORE_SEMANTICS,
        calibration_status="uncalibrated",
        calibration_reference_sha256=None, abstain_below=0.6,
    )
    emb_features=frozenset({"float","pooling-last","normalization-l2"})
    dec_features=frozenset({"native-decision-head","choice-probabilities",
                            "multi-token-choice-labels","provider-temperature-softmax"})
    emb_contract=ServingContract(
        deployment_revision=dep.revision,
        capability="text.embed", operation_schema=EMBEDDING_OPERATION_SCHEMA,
        validation_evidence_sha256="sha256:" + "2"*64,
        features=emb_features,
        limits={"item_tokens":128,"item_bytes":1024,"batch_items":4,
                "batch_bytes":4096,"aggregate_tokens":256,"request_bytes":8192},
        semantic_revision=space.embedding_space_id,
    )
    dec_contract=ServingContract(
        deployment_revision=dep.revision,
        capability="decision.system_one", operation_schema=DECISION_OPERATION_SCHEMA,
        validation_evidence_sha256="sha256:" + "3"*64,
        features=dec_features,
        limits={"state_bytes":1024,"question_bytes":512,"choice_count":4,
                "choice_id_bytes":64,"choice_label_bytes":128,"state_tokens":128,
                "question_tokens":64,"choice_label_tokens":64,
                "aggregate_tokens":256,"request_bytes":4096},
        semantic_revision=semantics.decision_semantics_id,
    )
    eprofile=LogicalServingProfile(
        profile_id="dual-embed-v1",deployment_revision=dep.revision,
        serving_contract_revision=emb_contract.revision,
        capability="text.embed",operation_schema=EMBEDDING_OPERATION_SCHEMA,
        required_features=emb_features,
    )
    dprofile=LogicalServingProfile(
        profile_id="dual-systemone-v1",deployment_revision=dep.revision,
        serving_contract_revision=dec_contract.revision,
        capability="decision.system_one",operation_schema=DECISION_OPERATION_SCHEMA,
        required_features=dec_features,
    )
    embed = EmbeddingGatewayProfile.from_dict({
        "adapter_id":LLAMA_CPP_EMBEDDING_ADAPTER,
        "deployment":dep.to_dict(),"contract":emb_contract.to_dict(),
        "profile":eprofile.to_dict(),"space":space.to_dict(),
        "query_prefix":"","document_prefix":"",
    })
    dec = DecisionGatewayProfile.from_dict({
        "adapter_id":LLAMA_CPP_SYSTEM_ONE_ADAPTER,
        "deployment":dep.to_dict(),"contract":dec_contract.to_dict(),
        "profile":dprofile.to_dict(),"semantics":semantics.to_dict(),
    })
    assert embed.embedding_space_id == space.embedding_space_id
    assert dec.decision_semantics_id == semantics.decision_semantics_id
    composite = replace(cfg,
        embedding_space_id=space.embedding_space_id,
        decision_semantics_id=semantics.decision_semantics_id,
    )
    runtime = RuntimeDeploymentSpec(
        provider_id=provider.info.provider_id,
        provider_config=vars_config(composite),
        demand=ctx.demand,setup_intent=None,
    )
    from astrumweaver.serving import ServingDeploymentDeclaration
    from astrumweaver.worker.daemon import _build_serving_advertisement
    declaration = ServingDeploymentDeclaration(
        deployment=dep, contracts=(emb_contract,dec_contract),
    )
    advert = _build_serving_advertisement(
        declaration,spec=ctx.worker,runtime_deployment=runtime,
    )
    assert len(advert.contracts)==2

    # Changing one gateway semantic identity without changing the reviewed
    # runtime must not advertise a partially compatible pair.
    tampered=replace(emb_contract,semantic_revision="sha256:"+"f"*64)
    with pytest.raises(RuntimeError,match="pinned child semantics"):
        _build_serving_advertisement(
            ServingDeploymentDeclaration(
                deployment=dep,contracts=(tampered,dec_contract),
            ),spec=ctx.worker,runtime_deployment=runtime,
        )
    with pytest.raises(ValueError,match="pinned per-capability semantics"):
        replace(composite,decision_semantics_id="not-a-digest")

@pytest.mark.asyncio
async def test_pinned_model_replaced_during_child_start_fails_closed(tmp_path):
    managed, embedding, decision = managed_pair(tmp_path)
    source = tmp_path / "decision.gguf"
    original_start = decision.start
    async def replace_during_start():
        await original_start()
        source.write_bytes(b"modified while model starts")
    decision.start = replace_during_start
    with pytest.raises(RuntimeError, match="digest mismatch"):
        await managed.start()
    assert embedding.stops == decision.stops == 1
    assert not (await managed.health()).ready


@pytest.mark.asyncio
async def test_pinned_model_symlink_not_followed(tmp_path):
    managed, embedding, decision = managed_pair(tmp_path)
    model = tmp_path / "decision.gguf"
    alternate = tmp_path / "regular-duplicate.gguf"
    alternate.write_bytes(model.read_bytes())
    model.unlink()
    model.symlink_to(alternate)
    with pytest.raises(RuntimeError, match="regular file"):
        await managed.start()
    assert embedding.stops == decision.stops == 1
    assert not (await managed.health()).ready


def test_nix_output_with_traversal_not_admitted(tmp_path):
    provider, _ = setup_pair(tmp_path)
    from dataclasses import replace
    cfg = provider.config
    invalid = "/nix/store/"+"z"*32+"-loader/.."
    with pytest.raises(ValueError, match="credential-isolating launcher"):
        replace(cfg, decision_provider=replace(
            cfg.decision_provider,
            package_reference=invalid,
            executable=invalid+"/bin/astrumweaver-llama-server",
        ))


@pytest.mark.asyncio
async def test_start_failure_preserves_cause_when_both_cleanup_paths_fail(tmp_path):
    managed, embedding, decision = managed_pair(tmp_path)
    async def broken_start():
        raise RuntimeError("original-model-startup-error")
    async def broken_stop():
        raise RuntimeError("owned-child-cleanup-error")
    decision.start = broken_start
    embedding.stop = broken_stop
    decision.stop = broken_stop
    with pytest.raises(RuntimeError, match="original-model-startup-error") as observed:
        await managed.start()
    assert any("cleanup also failed" in note for note in observed.value.__notes__)
    assert any("RuntimeError" in note for note in observed.value.__notes__)
    assert not (await managed.health()).ready


@pytest.mark.asyncio
async def test_startup_cancellation_not_masked_by_cleanup_error(tmp_path):
    managed, embedding, decision = managed_pair(tmp_path)
    async def cancelled_start():
        raise asyncio.CancelledError("original-requested-cancel")
    async def failed_stop():
        raise RuntimeError("cleanup-still-failed")
    decision.start = cancelled_start
    embedding.stop = failed_stop
    with pytest.raises(asyncio.CancelledError, match="original-requested-cancel") as observed:
        await managed.start()
    assert any("cleanup also failed" in note for note in observed.value.__notes__)
    assert decision.stops == 1
    assert not (await managed.health()).ready


def test_model_file_must_not_be_group_or_world_writable(tmp_path):
    provider, _ = setup_pair(tmp_path)
    path = tmp_path / "embed.gguf"
    path.chmod(0o666)
    with pytest.raises(RuntimeError, match="untrusted writes"):
        DualLlamaCppManagedRuntime._check_pinned_model(
            provider.config.embedding_model,
            trusted_owner_uid=os.getuid(),
        )


def test_model_parent_must_not_allow_untrusted_rename(tmp_path):
    provider, _ = setup_pair(tmp_path)
    old_mode = tmp_path.stat().st_mode & 0o7777
    try:
        tmp_path.chmod(0o777)
        with pytest.raises(RuntimeError, match="untrusted writes"):
            DualLlamaCppManagedRuntime._check_pinned_model(
                provider.config.embedding_model,
                trusted_owner_uid=os.getuid(),
            )
    finally:
        tmp_path.chmod(old_mode)


def test_model_parent_symlink_or_unauthorized_owner_is_rejected(tmp_path):
    from dataclasses import replace
    provider, _ = setup_pair(tmp_path)
    model = provider.config.embedding_model
    subdir = tmp_path / "safe"
    subdir.mkdir(mode=0o700)
    renamed = subdir / "embed.gguf"
    renamed.write_bytes((tmp_path/"embed.gguf").read_bytes())
    link = tmp_path / "alias"
    link.symlink_to(subdir, target_is_directory=True)
    symlink_pin = replace(model, model_ref=str(link/"embed.gguf"))
    with pytest.raises(RuntimeError, match="real directory"):
        DualLlamaCppManagedRuntime._check_pinned_model(
            symlink_pin, trusted_owner_uid=os.getuid()
        )
    wrong_uid = os.getuid() + 10000
    with pytest.raises(RuntimeError, match="untrusted owner|owner is not pinned"):
        DualLlamaCppManagedRuntime._check_pinned_model(
            model, trusted_owner_uid=wrong_uid
        )


def test_model_trust_owner_is_explicit_and_roundtrips(tmp_path):
    from dataclasses import replace
    provider, _ = setup_pair(tmp_path)
    config = provider.config
    assert config.trusted_model_owner_uid == os.getuid()
    with pytest.raises(ValueError, match="trusted_model_owner_uid"):
        replace(config, trusted_model_owner_uid=True)
    with pytest.raises(ValueError, match="trusted_model_owner_uid"):
        replace(config, trusted_model_owner_uid=-1)
    DualLlamaCppManagedRuntime._check_pinned_model(
        config.embedding_model, trusted_owner_uid=config.trusted_model_owner_uid,
    )


@pytest.mark.asyncio
async def test_replaced_file_in_untrusted_mutable_directory_fails_before_load(tmp_path):
    managed, embedding, decision = managed_pair(tmp_path)
    old_mode = tmp_path.stat().st_mode & 0o7777
    try:
        tmp_path.chmod(0o777)
        with pytest.raises(RuntimeError, match="untrusted writes"):
            await managed.start()
        assert embedding.health_state is RuntimeHealthState.STOPPED
        assert decision.health_state is RuntimeHealthState.STOPPED
        assert embedding.stops == decision.stops == 1
    finally:
        tmp_path.chmod(old_mode)


@pytest.mark.asyncio
async def test_composite_subprocesses_require_distinct_private_bearer_files(tmp_path):
    import stat
    provider, ctx = setup_pair(tmp_path)
    deployment = RuntimeDeploymentSpec(
        provider_id=provider.info.provider_id,
        provider_config=vars_config(provider.config),
        demand=ctx.demand,
        setup_intent=provider.setup_intent(ctx),
    )
    runtime = managed_runtime_from_deployment(
        deployment, worker=ctx.worker, host=ctx.host,
    )
    children = (runtime._embedding, runtime._decision)
    keys = []
    secret_dir = runtime._private_auth_files.name
    assert stat.S_IMODE(Path(secret_dir).stat().st_mode) == 0o700
    for child in children:
        file = Path(child.process.api_key_file)
        assert file.is_file()
        assert file.parent == Path(secret_dir)
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
        token = file.read_text().strip()
        assert len(token) >= 48
        auth = child.api._client.headers["authorization"]
        assert auth == "Bearer " + token
        args = child.process.command()
        assert args[args.index("--api-key-file") + 1] == str(file)
        assert token not in " ".join(args)
        keys.append(token)
    assert keys[0] != keys[1]
    assert "private_auth_files" not in deployment.to_dict()
    await runtime.release()
    assert not Path(secret_dir).exists()
    assert all(not Path(child.process.api_key_file).exists() for child in children)


@pytest.mark.asyncio
async def test_native_http_client_bearer_header_is_scoped_to_private_child():
    import httpx
    from astrumweaver.runtime.providers.llama_cpp import HttpLlamaCppApi
    observed=[]
    def handler(req: httpx.Request):
        observed.append(req.headers.get("authorization"))
        return httpx.Response(200,json={"ready":True})
    api=HttpLlamaCppApi(
        "http://127.0.0.1:18700",
        api_key="opaque-test-token",
        transport=httpx.MockTransport(handler),
    )
    assert await api.health() is True
    assert observed==["Bearer opaque-test-token"]
    await api.close()
