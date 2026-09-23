from __future__ import annotations

import re
import tomllib
from pathlib import Path

from astrumweaver import ResourceShape, WorkerSpec
from astrumweaver.control import InMemoryControlRepository, WorkerRegistration


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = ROOT / "profiles" / "v0.1"
ACCEPTANCE = ROOT / "acceptance" / "v0.1.toml"

EXPECTED_PROFILES = {
    "control-plane",
    "modern-single",
    "multi-gpu-large",
    "legacy-single",
}

EXPECTED_CHECKS = {
    "single-gpu-worker-registration",
    "multi-gpu-worker-registration",
    "exact-gpu-uuid-preflight",
    "multi-vs-single-vram-shape",
    "capability-matching",
    "draining-offline-lifecycle",
    "existing-node-install",
    "nixos-module-deployment",
    "versioned-control-worker-transport",
    "borrowable-worker-handoff",
}

PRIVATE_IPV4 = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
)


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_v0_1_profiles_are_complete_and_documentation_only() -> None:
    profiles = {
        data["profile_id"]: data
        for path in PROFILE_ROOT.glob("*.toml")
        for data in (load_toml(path),)
    }

    assert set(profiles) == EXPECTED_PROFILES

    for profile_id, profile in profiles.items():
        assert profile["profile_version"] == "v0.1"
        assert profile["enforcement"] == "documentation-only"

        minimum = profile["minimum"]
        recommended = profile["recommended"]

        assert minimum["vcpu"] > 0
        assert minimum["ram_mib"] > 0
        assert minimum["disk_gib"] > 0
        assert recommended["vcpu"] >= minimum["vcpu"]
        assert recommended["ram_mib"] >= minimum["ram_mib"]
        assert recommended["disk_gib"] >= minimum["disk_gib"]

        validations = profile.get("validated") or []
        assert validations
        for validation in validations:
            assert validation["scope"] in {"ci-contract", "hardware-e2e"}
            assert validation["evidence"]
            assert validation["claim"]
            assert isinstance(validation["hardware_sizing_validated"], bool)

        if profile_id == "control-plane":
            assert minimum["gpu_count"] == 0
            assert "example_resource_shape" not in profile
            continue

        shape = profile["example_resource_shape"]
        assert shape["gpu_count"] == len(shape["gpu_uuids"])
        assert shape["gpu_count"] >= 1
        assert shape["total_vram_mb"] > 0
        assert shape["max_single_gpu_vram_mb"] > 0
        assert shape["max_single_gpu_vram_mb"] <= shape["total_vram_mb"]
        assert all(uuid.startswith("GPU-example-") for uuid in shape["gpu_uuids"])


def test_profile_examples_publish_no_private_site_inventory() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(PROFILE_ROOT.glob("*.toml"))
    )

    assert PRIVATE_IPV4.search(combined) is None
    assert "qm create" not in combined
    assert "pct create" not in combined
    assert "hostpci" not in combined
    assert "vmid" not in combined.lower()
    assert "ctid" not in combined.lower()


def test_v0_1_acceptance_manifest_has_every_required_check() -> None:
    manifest = load_toml(ACCEPTANCE)

    assert manifest["acceptance_version"] == "v0.1"
    assert manifest["scope"] == "software-contract"

    checks = {check["id"]: check for check in manifest["check"]}
    assert set(checks) == EXPECTED_CHECKS

    for check in checks.values():
        assert check["description"]
        assert check["evidence"]


def test_v0_1_acceptance_evidence_references_existing_tests_or_nix_check() -> None:
    manifest = load_toml(ACCEPTANCE)

    for check in manifest["check"]:
        for evidence in check["evidence"]:
            if "::" in evidence:
                relative_path, symbol = evidence.split("::", 1)
                path = ROOT / relative_path
                assert path.is_file(), evidence
                source = path.read_text(encoding="utf-8")
                assert f"def {symbol}(" in source or f"async def {symbol}(" in source, evidence
                continue

            if "#" in evidence:
                relative_path, fragment = evidence.split("#", 1)
                path = ROOT / relative_path
                assert path.is_file(), evidence
                source = path.read_text(encoding="utf-8")
                for component in fragment.split("."):
                    if component in {"checks", "x86_64-linux"}:
                        continue
                    assert component in source, evidence
                continue

            path = ROOT / evidence
            assert path.exists(), evidence


def test_v0_1_profiles_preserve_multi_gpu_single_device_distinction() -> None:
    profile = load_toml(PROFILE_ROOT / "multi-gpu-large.toml")
    shape = profile["example_resource_shape"]

    assert shape == {
        "gpu_count": 2,
        "total_vram_mb": 24576,
        "max_single_gpu_vram_mb": 12288,
        "gpu_uuids": ["GPU-example-multi-a", "GPU-example-multi-b"],
    }
    assert shape["total_vram_mb"] >= 20_000
    assert shape["max_single_gpu_vram_mb"] < 20_000


def _worker_from_profile(profile_name: str) -> WorkerSpec:
    profile = load_toml(PROFILE_ROOT / f"{profile_name}.toml")
    shape = profile["example_resource_shape"]
    return WorkerSpec(
        worker_id=f"acceptance-{profile_name}",
        worker_class=profile_name,
        resources=ResourceShape(
            gpu_count=shape["gpu_count"],
            total_vram_mb=shape["total_vram_mb"],
            max_single_gpu_vram_mb=shape["max_single_gpu_vram_mb"],
        ),
        gpu_uuids=tuple(shape["gpu_uuids"]),
        capabilities=frozenset({"acceptance.example"}),
        labels={"profile": profile_name},
    )


def test_v0_1_single_gpu_profile_registers_as_one_worker() -> None:
    repo = InMemoryControlRepository()
    worker = _worker_from_profile("modern-single")

    registered = repo.register_worker(WorkerRegistration(spec=worker))

    assert registered.worker_id == "acceptance-modern-single"
    assert registered.spec.resources.gpu_count == 1
    assert registered.spec.gpu_uuids == ("GPU-example-modern-single",)


def test_v0_1_multi_gpu_profile_registers_as_one_worker() -> None:
    repo = InMemoryControlRepository()
    worker = _worker_from_profile("multi-gpu-large")

    registered = repo.register_worker(WorkerRegistration(spec=worker))

    assert registered.worker_id == "acceptance-multi-gpu-large"
    assert registered.spec.resources.gpu_count == 2
    assert registered.spec.gpu_uuids == (
        "GPU-example-multi-a",
        "GPU-example-multi-b",
    )
