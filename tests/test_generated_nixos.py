"""Fixture/harness tests complement, but never substitute for Nix evaluation."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from astrumweaver.setup import first_run


_spec = importlib.util.spec_from_file_location(
    "generated_nixos_check", Path(__file__).parents[1] / "tools/check_generated_nixos.py"
)
assert _spec is not None and _spec.loader is not None
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


def test_matrix_uses_production_renderer_and_is_deterministic(tmp_path, monkeypatch):
    original = first_run.render_nixos_bootstrap_snippet
    calls = []

    def observed(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(first_run, "render_nixos_bootstrap_snippet", observed)
    cases = checker.generate_cases(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert len(calls) == len(cases) == 12
    checker.generate_cases(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert {c["name"] for c in cases} == {
        "control-default", "control-none", "smoke-gpu", "smoke-cpu", "combined-smoke-none",
        "runtime-ollama", "runtime-vllm", "runtime-llama-cpp", "combined-runtime-none",
        "combined-vllm-none", "runtime-overlay", "literal-data",
    }
    assert json.loads((tmp_path / "manifest.json").read_text()) == cases
    literal = (tmp_path / "literal-data.nix").read_text()
    assert r"\${not_a_nix_binding}" in literal
    assert all(Path(case["module"]).is_file() for case in cases)
    assert all("ASTRUMWEAVER_WORKER_TOKEN" not in p.read_text() for p in tmp_path.glob("*.nix"))


def test_evaluate_applies_source_and_manifest_before_json_conversion(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source with space"
    evaluator = source / "nix/tests/generated-first-run.nix"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("{ source, manifest }: [ source manifest ]\n")
    manifest = tmp_path / "manifest with space.json"
    manifest.write_text("[]")

    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, "[]", "")

    monkeypatch.setattr(checker.subprocess, "run", fake_run)
    checker.evaluate(source, manifest)

    argv = captured["argv"]
    assert "--file" not in argv
    assert "--argstr" not in argv
    expression = argv[argv.index("--expr") + 1]
    assert "let check = import (builtins.toPath " in expression
    assert json.dumps(str(evaluator)) in expression
    assert "source = " + json.dumps(str(source)) in expression
    assert "manifest = " + json.dumps(str(manifest)) in expression
    assert captured["kwargs"]["check"] is False


@pytest.mark.parametrize("expression", ["myPkgs.ollama", "pkgs", "inputs.runtime", "pkgs.9foo"])
def test_renderer_itself_rejects_unknown_package_scope(expression):
    from astrumweaver import ResourceShape, WorkerSpec
    worker = WorkerSpec(worker_id="test", worker_class="test", resources=ResourceShape(), capabilities=frozenset({"llm.chat"}))
    with pytest.raises(ValueError, match="rooted in pkgs"):
        first_run.render_nixos_bootstrap_snippet(
            role=first_run.FirstRunRole.WORKER, worker=worker,
            control_url="http://control.example.invalid:9000",
            execution_mode=first_run.FirstRunExecutionMode.RUNTIME,
            runtime_deployment={}, runtime_package_expression=expression,
        )
