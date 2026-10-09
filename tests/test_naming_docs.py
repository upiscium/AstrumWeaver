"""Guard the TSUMGI project name without silently renaming runtime interfaces."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_and_docs_use_tsumgi_brand() -> None:
    readme = (ROOT / "README.md").read_text()
    naming = (ROOT / "docs" / "naming.md").read_text()
    assert readme.startswith("# TSUMGI\n")
    assert "[Naming and compatibility](docs/naming.md)" in readme
    assert "TSUMGI（紡ぎ）" in naming


def test_legacy_interfaces_remain_declared() -> None:
    project = (ROOT / "pyproject.toml").read_text()
    flake = (ROOT / "flake.nix").read_text()
    assert 'name = "astrumweaver"' in project
    assert 'astrumweaver-control = "astrumweaver.control.daemon:main"' in project
    assert "services.astrumweaver.control" in flake
    assert "services.astrumweaver.worker" in flake
