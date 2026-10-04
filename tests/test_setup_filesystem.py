from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from astrumweaver.setup import SetupAction, SetupActionKind, SetupActionState, SystemdSetupDriver
from astrumweaver.setup.filesystem import SetupFilesystem, UnsafeSetupPath


def action(kind=SetupActionKind.ENSURE_PACKAGE, **payload):
    return SetupAction(
        action_id="01-check", kind=kind, description="filesystem regression",
        payload={"provider_id": "llama-cpp", "package_reference": "llama-cpp", **payload},
    )


@pytest.fixture
def fs(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    return SetupFilesystem(root)


@pytest.mark.parametrize("dangling", [False, True])
@pytest.mark.parametrize("operation", ["read", "write", "remove"])
def test_authoritative_file_links_are_rejected_without_touching_target(fs, tmp_path, dangling, operation):
    outside = tmp_path / "outside"
    if not dangling:
        outside.write_text("sentinel")
    target = fs.root / "receipt"
    target.symlink_to(outside)
    with pytest.raises(UnsafeSetupPath):
        if operation == "read":
            fs.read_text(target)
        elif operation == "write":
            fs.write_text(target, "new")
        else:
            fs.remove_file(target)
    assert target.is_symlink()
    assert not outside.exists() if dangling else outside.read_text() == "sentinel"


@pytest.mark.parametrize("dangling", [False, True])
@pytest.mark.parametrize("operation", ["write", "directory", "read"])
def test_linked_ancestor_is_rejected(fs, tmp_path, dangling, operation):
    outside = tmp_path / "outside"
    if not dangling:
        outside.mkdir()
        (outside / "receipt").write_text("sentinel")
    (fs.root / "linked").symlink_to(outside, target_is_directory=True)
    target = fs.root / "linked/receipt"
    with pytest.raises(UnsafeSetupPath):
        if operation == "write":
            fs.write_text(target, "new")
        elif operation == "read":
            fs.read_text(target)
        else:
            with fs.directory(fs.root / "linked/child", create=True):
                pass
    assert not outside.exists() if dangling else (outside / "receipt").read_text() == "sentinel"
    assert not (outside / "child").exists()


def test_authoritative_hard_link_is_rejected(fs, tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("sentinel")
    target = fs.root / "receipt"
    os.link(outside, target)
    with pytest.raises(UnsafeSetupPath, match="multiply linked"):
        fs.write_text(target, "new")
    assert outside.read_text() == "sentinel"


@pytest.mark.parametrize("mode", [0o777, 0o775, 0o770])
def test_writable_ancestor_is_not_adopted(fs, mode):
    parent = fs.root / "state"
    parent.mkdir()
    parent.chmod(mode)
    with pytest.raises(UnsafeSetupPath, match="writable"):
        fs.write_text(parent / "receipt", "new")
    assert stat.S_IMODE(parent.stat().st_mode) == mode
    assert not (parent / "receipt").exists()


def test_unsafe_file_mode_is_not_adopted(fs):
    target = fs.root / "receipt"
    target.write_text("original")
    target.chmod(0o666)
    with pytest.raises(UnsafeSetupPath, match="permissions"):
        fs.write_text(target, "new")
    assert target.read_text() == "original"
    assert stat.S_IMODE(target.stat().st_mode) == 0o666


@pytest.mark.skipif(os.geteuid() != 0, reason="requires disposable file ownership changes")
def test_worker_owned_ancestor_and_file_are_not_trusted(fs):
    parent = fs.root / "state"
    parent.mkdir()
    os.chown(parent, 65534, 65534)
    with pytest.raises(UnsafeSetupPath, match="ownership"):
        fs.write_text(parent / "receipt", "new")
    assert parent.stat().st_uid == 65534
    target = fs.root / "config"
    target.write_text("original")
    os.chown(target, 65534, 65534)
    with pytest.raises(UnsafeSetupPath, match="ownership"):
        fs.write_text(target, "new")
    assert target.read_text() == "original"
    assert target.stat().st_uid == 65534


def test_exclusive_temp_creation_never_truncates_collision(fs, tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.write_text("sentinel")
    collision = fs.root / ".receipt.fixed.tmp"
    collision.symlink_to(outside)
    monkeypatch.setattr("astrumweaver.setup.filesystem.secrets.token_hex", lambda _: "fixed")
    with pytest.raises(FileExistsError):
        fs.write_text(fs.root / "receipt", "new")
    assert outside.read_text() == "sentinel"
    assert collision.is_symlink()


def test_parent_replacement_cannot_redirect_atomic_write(fs, tmp_path, monkeypatch):
    parent = fs.root / "managed"
    parent.mkdir()
    target = parent / "receipt"
    target.write_text("original")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "receipt").write_text("sentinel")
    original_replace = os.replace

    def replace_parent(src, dst, **kwargs):
        assert isinstance(kwargs["src_dir_fd"], int)
        assert kwargs["src_dir_fd"] == kwargs["dst_dir_fd"]
        parent.rename(fs.root / "pinned")
        parent.symlink_to(outside, target_is_directory=True)
        original_replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", replace_parent)
    with pytest.raises(UnsafeSetupPath):
        fs.write_text(target, "new")
    assert (outside / "receipt").read_text() == "sentinel"
    assert (fs.root / "pinned/receipt").read_text() == "new"


def test_late_target_link_is_replaced_not_followed(fs, tmp_path, monkeypatch):
    target = fs.root / "receipt"
    outside = tmp_path / "outside"
    outside.write_text("sentinel")
    original_replace = os.replace

    def replace_after_final_check(src, dst, **kwargs):
        target.symlink_to(outside)
        original_replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", replace_after_final_check)
    fs.write_text(target, "new")
    assert outside.read_text() == "sentinel"
    assert not target.is_symlink()
    assert target.read_text() == "new"


def test_normal_atomic_write_preserves_requested_permissions(fs):
    target = fs.root / "config/value"
    fs.write_text(target, "first", mode=0o640)
    inode = target.stat().st_ino
    fs.write_text(target, "second", preserve_metadata=True)
    assert target.stat().st_ino != inode
    assert fs.read_text(target) == "second"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    with pytest.raises(UnsafeSetupPath, match="changed file"):
        fs.remove_file(target, expected="first")
    fs.remove_file(target, expected="second")
    assert not target.exists()


@pytest.mark.parametrize("path", ["relative/file", "/etc/../outside"])
def test_invalid_path_is_rejected(fs, path):
    with pytest.raises(UnsafeSetupPath):
        fs.write_text(Path(path), "new")


def test_package_legacy_marker_is_neither_trusted_nor_written(tmp_path, monkeypatch):
    import hashlib
    root = tmp_path / "root"
    legacy = root / "var/lib/astrumweaver/runtime/llama-cpp/packages"
    legacy.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("sentinel")
    marker = legacy / (hashlib.sha256(b"llama-cpp").hexdigest()[:20] + ".installed")
    marker.symlink_to(outside)
    driver = SystemdSetupDriver(root=root, installers={"llama-cpp": ("reviewed-installer",)})
    state = {"available": False}
    monkeypatch.setattr(driver, "_provider_executable_available", lambda *_: state["available"])
    monkeypatch.setattr("astrumweaver.setup.systemd.subprocess.run", lambda *a, **k: state.update(available=True))
    task = action()
    assert driver.inspect(task).state is SetupActionState.NEEDS_APPLY
    assert driver.apply(task).changed
    receipt = driver._receipt_path(task)
    assert receipt.is_relative_to(root / "var/lib/astrumweaver-setup")
    assert json.loads(receipt.read_text())["version"] == 1
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    assert stat.S_IMODE((root / "var/lib/astrumweaver-setup").stat().st_mode) == 0o700
    assert marker.is_symlink() and outside.read_text() == "sentinel"
    state["available"] = False
    assert driver.inspect(task).state is SetupActionState.NEEDS_APPLY


@pytest.mark.parametrize("where", ["target", "ancestor"])
def test_poisoned_new_receipt_path_blocks_before_installer(tmp_path, monkeypatch, where):
    driver = SystemdSetupDriver(root=tmp_path / "root", installers={"llama-cpp": ("installer",)})
    task = action()
    target = driver._receipt_path(task)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_text("keep")
    target.parent.mkdir(parents=True)
    (driver.root / "var/lib/astrumweaver-setup").chmod(0o700)
    if where == "target":
        target.symlink_to(outside / "sentinel")
    else:
        target.parent.rmdir()
        target.parent.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(driver, "_provider_executable_available", lambda *_: False)
    monkeypatch.setattr("astrumweaver.setup.systemd.subprocess.run", lambda *a, **k: pytest.fail("installer must not run"))
    assert driver.inspect(task).state is SetupActionState.BLOCKED
    with pytest.raises(RuntimeError):
        driver.apply(task)
    assert (outside / "sentinel").read_text() == "keep"


def test_success_exit_without_prerequisite_does_not_write_receipt(tmp_path, monkeypatch):
    driver = SystemdSetupDriver(root=tmp_path / "root", installers={"llama-cpp": ("installer",)})
    monkeypatch.setattr(driver, "_provider_executable_available", lambda *_: False)
    monkeypatch.setattr("astrumweaver.setup.systemd.subprocess.run", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="still unavailable"):
        driver.apply(action())
    assert not driver._receipt_path(action()).exists()


def test_remote_model_receipt_never_substitutes_for_current_verification(tmp_path, monkeypatch):
    driver = SystemdSetupDriver(
        root=tmp_path / "root", downloaders={"llama-cpp": ("download",)},
        verifiers={"llama-cpp": ("check",)},
    )
    task = action(SetupActionKind.DOWNLOAD_MODEL, model_ref="org/model")
    state = {"present": False}
    calls = []

    def run(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[0] == "download":
            state["present"] = True
            return subprocess.CompletedProcess(argv, 0)
        assert argv == ["check", "model", "org/model"]
        assert kwargs["timeout"] == 10.0
        return subprocess.CompletedProcess(argv, 0 if state["present"] else 1)

    monkeypatch.setattr("astrumweaver.setup.systemd.subprocess.run", run)
    assert driver.apply(task).changed
    assert driver.inspect(task).state is SetupActionState.SATISFIED
    state["present"] = False
    assert driver.inspect(task).state is SetupActionState.NEEDS_APPLY
    assert driver._receipt_path(task).is_file()
    assert calls.count(("download", "org/model")) == 1


def test_conversion_cannot_use_source_existence_as_output_evidence(tmp_path):
    root = tmp_path / "root"
    model = root / "models/source"
    model.parent.mkdir(parents=True)
    model.write_text("source")
    driver = SystemdSetupDriver(root=root, converters={"llama-cpp": ("convert",)})
    task = action(SetupActionKind.CONVERT_MODEL, model_ref="/models/source")
    assert driver.inspect(task).state is SetupActionState.BLOCKED


def test_runtime_directory_link_cannot_change_target_permissions(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    target = root / "var/lib/astrumweaver/runtime/llama-cpp"
    target.parent.mkdir(parents=True)
    target.symlink_to(outside, target_is_directory=True)
    driver = SystemdSetupDriver(root=root)
    task = action(SetupActionKind.ENSURE_DIRECTORY, logical_name="state")
    assert driver.inspect(task).state is SetupActionState.BLOCKED
    with pytest.raises(UnsafeSetupPath):
        driver.apply(task)
    assert stat.S_IMODE(outside.stat().st_mode) == 0o700


def test_config_partial_failure_restores_earlier_write(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    driver = SystemdSetupDriver(root=root)
    first = root / "etc/astrumweaver/runtime-deployment.json"
    second = root / "etc/astrumweaver/runtime/exllamav3/config.yml"
    driver.filesystem.write_text(first, "previous", mode=0o640)
    task = action(SetupActionKind.RENDER_CONFIG, runtime_deployment={"provider_id": "exllamav3"}, configuration={})
    monkeypatch.setattr(driver, "_render_targets", lambda _: ((first, "desired"), (second, "config")))
    original_write = driver._write_nonsecret_config

    def fail_second(path, content):
        if path == second:
            raise OSError("simulated write failure")
        original_write(path, content)

    monkeypatch.setattr(driver, "_write_nonsecret_config", fail_second)
    with pytest.raises(OSError):
        driver.apply(task)
    assert first.read_text() == "previous"
    assert not second.exists()


def test_config_rollback_rejects_replaced_target(tmp_path):
    root = tmp_path / "root"
    driver = SystemdSetupDriver(root=root)
    task = action(SetupActionKind.RENDER_CONFIG, runtime_deployment={"provider_id": "llama-cpp"}, configuration={})
    receipt = driver.apply(task)
    target = root / "etc/astrumweaver/runtime-deployment.json"
    outside = tmp_path / "outside"
    outside.write_text("sentinel")
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(UnsafeSetupPath):
        driver.rollback(task, receipt)
    assert outside.read_text() == "sentinel"


def test_bridge_rejects_unsafe_ancestor(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    outside.mkdir()
    parent = root / "etc/astrumweaver"
    parent.mkdir(parents=True)
    (parent / "runtime").symlink_to(outside, target_is_directory=True)
    lib = tmp_path / "libcuda.so.1"
    lib.write_text("driver")
    driver = SystemdSetupDriver(root=root, nvidia_driver_library=lib)
    task = action(
        SetupActionKind.ENSURE_NVIDIA_DRIVER_BRIDGE,
        soname="libcuda.so.1", bridge_directory="/etc/astrumweaver/runtime/nvidia-driver",
    )
    assert driver.inspect(task).state is SetupActionState.BLOCKED
    with pytest.raises(RuntimeError):
        driver.apply(task)
    assert list(outside.iterdir()) == []


def test_staging_anchor_ancestor_link_is_not_followed(tmp_path):
    outside = tmp_path / "outside"
    (outside / "root").mkdir(parents=True)
    (outside / "root/receipt").write_text("sentinel")
    parent = tmp_path / "linked"
    parent.symlink_to(outside, target_is_directory=True)
    fs = SetupFilesystem(parent / "root")
    with pytest.raises(UnsafeSetupPath):
        fs.write_text(parent / "root/receipt", "new")
    assert (outside / "root/receipt").read_text() == "sentinel"


def test_staging_anchor_writable_parent_is_not_trusted(tmp_path):
    parent = tmp_path / "writable"
    parent.mkdir()
    parent.chmod(0o777)
    fs = SetupFilesystem(parent / "root")
    with pytest.raises(UnsafeSetupPath):
        fs.write_text(parent / "root/receipt", "new")
    assert not (parent / "root").exists()


def test_replaced_new_directory_is_not_adopted(fs, monkeypatch):
    target = fs.root / "runtime"
    replacement = fs.root / "protected"
    replacement.mkdir(mode=0o700)
    (replacement / "sentinel").write_text("keep")
    mkdir = os.mkdir

    def swap_new(path, mode=0o777, *, dir_fd=None):
        mkdir(path, mode, dir_fd=dir_fd)
        if path == "runtime":
            target.rename(fs.root / "displaced")
            replacement.rename(target)

    monkeypatch.setattr(os, "mkdir", swap_new)
    with pytest.raises(UnsafeSetupPath, match="replaced"):
        with fs.directory(target, create=True, mode=0o750, uid=os.geteuid()):
            pass
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert (target / "sentinel").read_text() == "keep"
