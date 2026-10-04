from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER_TUI = (
    "/nix/var/nix/profiles/astrumweaver-installer/bin/"
    "astrumweaver-setup-tui"
)
FIRST_RUN_DOCS = (
    "README.md",
    "docs/installation.md",
    "docs/getting-started.md",
    "docs/setup-tui.md",
    "docs/deployment.md",
)


def test_privileged_tui_examples_use_the_installer_profile_path() -> None:
    for relative in FIRST_RUN_DOCS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "sudo astrumweaver-setup-tui" not in text

    for relative in FIRST_RUN_DOCS[1:]:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert f"sudo {INSTALLER_TUI}" in text


def test_profile_commands_resolve_nix_before_using_sudo() -> None:
    text = (ROOT / "docs/installation.md").read_text(encoding="utf-8")

    assert "sudo nix profile" not in text
    assert text.count('NIX_BIN="$(command -v nix)"') == 6
    assert text.count('sudo "$NIX_BIN" profile add') == 3
    assert text.count('sudo "$NIX_BIN" profile upgrade') == 3


def test_bare_planning_tui_example_states_the_path_requirement() -> None:
    text = (ROOT / "docs/setup-tui.md").read_text(encoding="utf-8")

    assert (
        "profile's `bin` directory is already on `PATH`, running without a "
        "driver:"
    ) in text


def test_generic_systemd_llama_runtime_profile_is_documented_as_standard_path() -> None:
    install = (ROOT / "docs/installation.md").read_text(encoding="utf-8")
    setup_tui = (ROOT / "docs/setup-tui.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "### RuntimeBackend Nix profile" in install
    assert "#runtime-llama-cpp" in install
    assert "llama-cpp-cuda" in install
    assert "/nix/var/nix/profiles/astrumweaver-runtime-llama-cpp/bin/llama-server" in install
    assert "astrumweaver-runtime-profile \\" in install
    assert "upgrade llama-cpp" in install
    assert "rollback llama-cpp" in install
    assert "Advanced/custom package and model hooks" in setup_tui
    assert "RuntimeBackend Nix profile" in readme


def test_runtime_package_output_is_cuda_pinned_but_installer_remains_lazy() -> None:
    flake = (ROOT / "flake.nix").read_text(encoding="utf-8")
    helper = (ROOT / "nix/runtime-profile-support.nix").read_text(
        encoding="utf-8"
    )

    assert "runtimeLlamaCpp = runtimePkgs.llama-cpp-cuda;" in flake
    assert "runtime-llama-cpp = runtimeLlamaCpp;" in flake
    assert "runtimeLlamaCpp" not in helper
    assert "profile upgrade" in helper
    assert "--override-flake" in helper
    assert "refusing to mutate unmanaged or mixed runtime profile" in helper
