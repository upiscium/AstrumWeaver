"""Contract tests; no CUDA hardware, model download, or host mutation required."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
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
        {"data": [{"architecture": {"input_modalities": ["text"], "output_modalities": None}}]},
        {"data": [{"architecture": {"input_modalities": ["text"], "output_modalities": 0}}]},
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
    correct["usage"]["output_tokens"] = False
    with pytest.raises(smoke.SmokeError, match="runtime_decision_protocol_invalid"):
        smoke._probe("http://127.0.0.1:1", "state", "question",
                     (("a", "alpha"), ("b", "beta")))
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
    monkeypatch.setattr(smoke, "_require_nix_store_package", lambda _: None)
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
    monkeypatch.setattr(smoke, "_require_nix_store_package", lambda _: None)
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
            if os.getenv("ASTRUMWEAVER_WORKER_TOKEN"):
                output=["text"]
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
    monkeypatch.setattr(smoke, "_require_nix_store_package", lambda _: None)
    # GitHub's container isolation can hide ss(8) listener PID metadata even
    # from the test owner. This stub asserts the guard is invoked; the
    # dedicated owner/PID tests below validate fail-closed security semantics.
    owner_checks = []
    monkeypatch.setattr(
        smoke, "_require_owned_loopback_listener",
        lambda port, group: owner_checks.append((port, group)),
    )
    # Encode provider behavior in executable test code, rather than passing
    # a hidden environment variable to the untrusted inference subprocess.
    fake = _SERVER.replace(
        'output=["text"] if os.getenv("TEST_BAD_MODALITY") else ["decisions"]',
        'output=["text"]' if bad_modality else 'output=["decisions"]',
    )
    package = _package(tmp_path, server=fake)
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
    # Runtime must NOT inherit secrets from this operator process.
    monkeypatch.setenv("ASTRUMWEAVER_WORKER_TOKEN", "test-secret-must-not-leak")
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
    assert owner_checks and all(p == port and isinstance(group, int) and group > 0
                                for p, group in owner_checks)
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", port)) != 0


def test_listener_must_belong_to_owned_process_group(monkeypatch):
    out = 'LISTEN 0 10 127.0.0.1:18311 0.0.0.0:* users:(("server",pid=421,fd=4))\\n'
    monkeypatch.setattr(smoke.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout=out))
    monkeypatch.setattr(smoke.os, "getpgid", lambda pid: 731)
    smoke._require_owned_loopback_listener(18311, 731)
    with pytest.raises(smoke.SmokeError, match="owned_by_other_process"):
        smoke._require_owned_loopback_listener(18311, 999)
    monkeypatch.setattr(smoke.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout=""))
    with pytest.raises(smoke.SmokeError, match="owned_listener_not_ready"):
        smoke._require_owned_loopback_listener(18311, 731)
    monkeypatch.setattr(
        smoke.subprocess, "run",
        lambda *a, **k: SimpleNamespace(
            stdout="LISTEN 0 10 127.0.0.1:18311 0.0.0.0:*\\n",
        ),
    )
    with pytest.raises(smoke.SmokeError, match="listener_ownership_unverified"):
        smoke._require_owned_loopback_listener(18311, 731)


def test_parent_exit_must_not_leave_child_listener_alive():
    # The wrapper's parent may exit abnormally while llama-server is alive.
    # Cleanup owns the entire new_session process group, not just Popen.poll.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    child = (
        "import socket,time; "
        "s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); "
        f"s.bind(('127.0.0.1',{port})); s.listen(); time.sleep(30)"
    )
    parent = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child!r}])"
    proc = subprocess.Popen(
        [sys.executable, "-c", parent],
        start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                bound = sock.connect_ex(("127.0.0.1", port)) == 0
            if bound and proc.poll() is not None:
                break
            time.sleep(0.03)
        else:
            pytest.fail("fake parent/child start preflight failed")
        smoke._stop_owned_server(proc)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) != 0:
                    break
            time.sleep(0.03)
        else:
            pytest.fail("owned subprocess group still listening")
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=3)


def test_only_real_nix_store_artifacts_admitted(tmp_path):
    package = _package(tmp_path)
    with pytest.raises(smoke.SmokeError, match="runtime_package_not_a_nix_store_output"):
        smoke._require_nix_store_package(package)
    with pytest.raises(smoke.SmokeError, match="runtime_package_not_accessible"):
        smoke._require_nix_store_package(tmp_path / "missing-artifact")


def test_pinned_package_real_store_path_is_accepted_when_provided():
    import os
    path = os.getenv("ASTRUMWEAVER_TEST_NIX_PACKAGE")
    if not path:
        pytest.skip("standalone Nix package requires authorized GPU test host")
    smoke._require_nix_store_package(Path(path))


def test_native_http_never_uses_ambient_proxy_or_redirects(monkeypatch):
    """A local probe must not be forwarded through user HTTP_PROXY or 302s."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/health")
                self.end_headers()
                return
            if self.path == "/oversize":
                body = b"x" * (smoke.MAX_NATIVE_HTTP_BODY_BYTES + 1)
            else:
                body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # With NO_PROXY cleared, urllib.request.urlopen() would connect to
        # this invalid user-configured HTTP proxy instead of to loopback.
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")
        base = f"http://127.0.0.1:{server.server_port}"
        assert smoke._request_json(base, "/health") == {"status": "ok"}
        with pytest.raises(smoke.SmokeError, match="runtime_http_redirect_rejected"):
            smoke._request_json(base, "/redirect")
        with pytest.raises(smoke.SmokeError, match="runtime_http_response_too_large"):
            smoke._request_json(base, "/oversize")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_native_http_listener_rejects_mixed_owners(monkeypatch):
    """SO_REUSEPORT-style multiple owners must all be in our owned process group."""
    output = (
        'LISTEN 0 10 127.0.0.1:18311 0.0.0.0:* users:(("owned",pid=421,fd=4))\n'
        'LISTEN 0 10 127.0.0.1:18311 0.0.0.0:* users:(("foreign",pid=422,fd=4))\n'
    )
    monkeypatch.setattr(
        smoke.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )
    monkeypatch.setattr(
        smoke.os, "getpgid",
        lambda pid: 731 if pid == 421 else 732,
    )
    with pytest.raises(smoke.SmokeError, match="isolated_loopback_port_owned_by_other_process"):
        smoke._require_owned_loopback_listener(18311, 731)
    monkeypatch.setattr(smoke.os, "getpgid", lambda pid: 731)
    smoke._require_owned_loopback_listener(18311, 731)
