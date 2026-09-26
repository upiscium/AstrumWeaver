from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _relative_markdown_links(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    links: list[str] = []
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = target.strip()
        if (
            not target
            or target.startswith(("http://", "https://", "mailto:", "#"))
        ):
            continue
        links.append(target.split("#", 1)[0])
    return tuple(links)


def test_readme_has_prominent_installation_entrypoint() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Start here" in text
    assert "[Installation](docs/installation.md)" in text
    assert "[Getting Started](docs/getting-started.md)" in text
    assert text.index("## Start here") < text.index("## Scope")


def test_first_run_documentation_links_resolve() -> None:
    for relative in (
        "README.md",
        "docs/installation.md",
        "docs/getting-started.md",
    ):
        source = ROOT / relative
        for target in _relative_markdown_links(source):
            resolved = (source.parent / target).resolve()
            assert resolved.exists(), (
                f"{relative} links to missing path: {target}"
            )


def test_installation_guide_tracks_supported_package_surface() -> None:
    text = (ROOT / "docs/installation.md").read_text(encoding="utf-8")

    assert "github:upiscium/AstrumWeaver#control" in text
    assert "github:upiscium/AstrumWeaver#worker" in text
    assert "astrumweaver-setup-control-plane" in text
    assert "astrumweaver-setup-gpu-worker" in text
    assert "pip-only" in text
    assert "complete supported host deployment" in text


def test_getting_started_proves_real_control_worker_round_trip() -> None:
    text = (ROOT / "docs/getting-started.md").read_text(encoding="utf-8")

    for required in (
        "/v1/ready",
        "127.0.0.1:9100/health",
        "debug.echo",
        "astrumweaver.executors.structured_echo:create_executor",
        "/v1/jobs",
        "succeeded",
    ):
        assert required in text
