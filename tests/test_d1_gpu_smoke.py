"""Contract tests; no CUDA hardware, model download, or host mutation required."""
from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrumweaver.validation import d1_gpu_smoke as smoke


def _package(root: Path, arch: str = "61", *, server: str = "#!/bin/sh\nexit 0\n") -> Path:
    runtime = root / "runtime"
    binpath = runtime / "bin" / "llama-server"
    metadata = runtime / "share" / "astrumweaver"
    metadata.mkdir(parents=True)
    binpath.parent.mkdir()
    binpath.write_text("#!/bin/sh\nexit 0\n")
    binpath.chmod(0o755)
    launcher = runtime / "bin" / "astrumweaver-llama-server"
    launcher.write_text(server)
    launcher.chmod(0o755)
    (metadata / "llama-cpp-revision").write_text(smoke.SOURCE_REVISION + "\n")
    (metadata / "cuda-sm").write_text(arch + "\n")
    (metadata / "cuda-toolkit").write_text("12.9\n")
    return runtime


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_exact_package_architecture_and_provenance(tmp_path):
    p = _package(tmp_path)
    assert smoke._package_binary(p, 1, ("6.1",)) == p / "bin/astrumweaver-llama-server"
    with pytest.raises(smoke.SmokeError, match="gpu_architecture_not_qualified"):
        smoke._package_binary(p, 1, ("8.6", "6.1"))
    with pytest.raises(smoke.SmokeError, match="gpu_topology_not_qualified"):
        smoke._package_binary(p, 2, ("6.1", "6.1"))
    (p / "share/astrumweaver/llama-cpp-revision").write_text("different\n")
    with pytest.raises(smoke.SmokeError, match="runtime_provenance_mismatch"):
        smoke._package_binary(p, 1, ("6.1",))


def test_multi_gpu_architecture_is_exact(tmp_path):
    p = _package(tmp_path, "86")
    assert smoke._package_binary(p, 2, ("8.6", "8.6")).is_file()
    with pytest.raises(smoke.SmokeError, match="gpu_topology_not_qualified"):
        smoke._package_binary(p, 3, ("8.6", "8.6", "8.6"))


def test_model_modality_requires_native_decision_head():
    good = {"data": [{"architecture": {
        "input_modalities": ["text"], "output_modalities": ["decisions"]}}]}
    smoke._validate_models(good)
    for bad in (
        {"data": []},
        {"data": [{"architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}}]},
        {"data": [{"architecture": {"input_modalities": ["image"], "output_modalities": ["decisions"]}}]},
    ):
        with pytest.raises(smoke.SmokeError, match="runtime_"):
            smoke._validate_models(bad)


def test_invalid_probability_or_output_tokens_rejected(monkeypatch):
    correct = {
        "answers": {"decision": {"type": "choice",
                                 "probabilities": {"0000": 0.8, "0001": 0.2},
                                 "choice": "0000"}},
        "model": "fake", "usage": {"output_tokens": 0},
    }
    monkeypatch.setattr(smoke, "_request_json", lambda *a: correct)
    chosen, duration = smoke._probe("http://127.0.0.1:1", "state", "question",
                                    (("a", "alpha"), ("b", "beta")))
    assert chosen == "a" and duration >= 0
    correct["answers"]["decision"]["probabilities"]["0000"] = float("nan")
    with pytest.raises(smoke.SmokeError, match="runtime_decision_protocol_invalid"):
        smoke._probe("http://127.0.0.1:1", "state", "question",
                     (("a", "alpha"), ("b", "beta")))
    correct["answers"]["decision"]["probabilities"]["0000"] = 0.8
    correct["usage"]["output_tokens"] = 1
    with pytest.raises(smoke.SmokeError, match="runtime_decision_protocol_invalid"):
        smoke._probe("http://127.0.0.1:1", "state", "question",
                     (("a", "alpha"), ("b", "beta")))


def test_port_is_not_overwritten():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        with pytest.raises(smoke.SmokeError, match="already_in_use"):
            smoke._check_unused_loopback_port(s.getsockname()[1])
    with pytest.raises(smoke.SmokeError, match="invalid_isolated_port"):
        smoke._check_unused_loopback_port(80)


def test_gpu_enumeration_rejects_malformed_snapshot(monkeypatch):
    monkeypatch.setattr(
        smoke.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="GPU-aaa, 6.1, 50, 10200\n"),
    )
    assert smoke._gpu_snapshot() == (("GPU-aaa", "6.1", 50, 10200),)
    monkeypatch.setattr(
        smoke.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="unexpected nvidia-smi output\n"),
    )
    with pytest.raises(smoke.SmokeError, match="gpu_preflight_unavailable"):
        smoke._gpu_snapshot()


def test_missing_model_digest_is_fail_closed(tmp_path, monkeypatch):
    package = _package(tmp_path)
    model = tmp_path / "wrong.gguf"
    model.write_bytes(b"not d1")
    monkeypatch.setattr(smoke, "_gpu_snapshot",
                        lambda: (("GPU-fake", "6.1", 0, 11000),))
    with pytest.raises(smoke.SmokeError, match="d1_model_digest_mismatch"):
        smoke.run_smoke(package=package, model=model, evidence=tmp_path / "result.json",
                        gpu_count=1, port=_port(), allow_shared_gpu=False,
                        startup_seconds=5)
    assert not (tmp_path / "result.json").exists()


def test_existing_gpu_load_requires_explicit_acknowledgement(tmp_path, monkeypatch):
    package = _package(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"placeholder")
    monkeypatch.setattr(smoke, "_gpu_snapshot",
                        lambda: (("GPU-fake", "6.1", 577, 10000),))
    monkeypatch.setattr(smoke, "_file_sha256", lambda *a: smoke.MODEL_SHA256)
    with pytest.raises(smoke.SmokeError, match="existing_gpu_load_requires_acknowledgement"):
        smoke.run_smoke(package=package, model=model, evidence=tmp_path / "result.json",
                        gpu_count=1, port=_port(), allow_shared_gpu=False,
                        startup_seconds=5)


_SERVER = '''#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port")+1])
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def reply(self, value):
        data=json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if self.path == "/health":
            self.reply({"status": "ok"})
        elif self.path == "/v1/models":
            output=["text"] if os.getenv("TEST_BAD_MODALITY") else ["decisions"]
            self.reply({"data":[{"architecture":{"input_modalities":["text"],
                                                   "output_modalities":output}}]})
        else: self.send_error(404)
    def do_POST(self):
        if self.path != "/v1/systemone": return self.send_error(404)
        size=int(self.headers.get("Content-Length","0"))
        value=json.loads(self.rfile.read(size))
        assert value["questions"]["decision"]["type"]=="choice"
        choices=value["questions"]["decision"]["criteria"]
        keys=list(choices)
        probs={k:(0.8 if i==0 else 0.1) for i,k in enumerate(keys)}
        self.reply({"answers":{"decision":{"type":"choice","choice":keys[0],
                                           "probabilities":probs}},
                    "model":"fake-decision","usage":{"output_tokens":0}})
HTTPServer(("127.0.0.1", port),Handler).serve_forever()
'''


@pytest.mark.parametrize("bad_modality", [False, True])
def test_owned_server_cleanup_and_no_public_model_data(tmp_path, monkeypatch, bad_modality):
    package = _package(tmp_path, server=_SERVER)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"placeholder")
    driver = tmp_path / "libcuda.so.1"
    driver.write_bytes(b"fake")
    calls = iter([
        (("GPU-fake", "6.1", 0, 11000),),
        (("GPU-fake", "6.1", 500, 10500),),
    ])
    monkeypatch.setattr(smoke, "_gpu_snapshot", lambda: next(calls))
    monkeypatch.setattr(smoke, "_file_sha256",
                        lambda p: smoke.MODEL_SHA256 if p == model else "fake-runtime-digest")
    if bad_modality:
        monkeypatch.setenv("TEST_BAD_MODALITY", "1")
    port = _port()
    evidence = tmp_path / "private.json"
    if bad_modality:
        with pytest.raises(smoke.SmokeError, match="runtime_not_native_text_decision"):
            smoke.run_smoke(package=package, model=model, evidence=evidence,
                            gpu_count=1, port=port, allow_shared_gpu=False,
                            startup_seconds=7, driver_library=driver)
        assert not evidence.exists()
    else:
        result = smoke.run_smoke(
            package=package, model=model, evidence=evidence, gpu_count=1,
            port=port, allow_shared_gpu=False, startup_seconds=7,
            driver_library=driver,
        )
        assert result["protocol"] == "PASS"
        assert result["owned_process_cleanup"] == "PASS"
        assert result["probes"] == 6
        assert result["vram_increment_mib_per_gpu"] == [500]
        data = json.loads(evidence.read_text())
        assert "GPU-fake" not in json.dumps(data)
        assert str(model) not in json.dumps(data)
        assert evidence.stat().st_mode & 0o777 == 0o600
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", port)) != 0
