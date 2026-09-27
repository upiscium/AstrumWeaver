from __future__ import annotations

from collections import deque
from collections.abc import Callable
import tomllib

from astrumweaver.executors.structured_echo import StructuredEchoExecutor
from astrumweaver.runtime import RuntimeHostFacts
from astrumweaver.setup import (
    DeploymentPath,
    DiscoveredGpu,
    PrivilegeMode,
    SetupHostSnapshot,
)
from astrumweaver.setup.first_run import SystemdBootstrapResult
from astrumweaver.setup.tui import (
    TuiRunStatus,
    build_worker_spec,
    run_first_run_tui,
)
from astrumweaver.worker.runtime import require_executor_capabilities


Response = str | Callable[[str], str]


class ScriptedIO:
    def __init__(self, responses: list[Response]) -> None:
        self.responses = deque(responses)
        self.output: list[str] = []

    def write(self, text: str = "") -> None:
        self.output.append(text)

    def ask(self, prompt: str) -> str:
        if not self.responses:
            raise AssertionError(f"unexpected prompt: {prompt}")
        response = self.responses.popleft()
        return response(prompt) if callable(response) else response

    def ask_secret(self, prompt: str) -> str:
        if not self.responses:
            raise AssertionError(f"unexpected secret prompt: {prompt}")
        response = self.responses.popleft()
        return response(prompt) if callable(response) else response

    def clear(self) -> None:
        return None


def _systemd_snapshot() -> SetupHostSnapshot:
    return SetupHostSnapshot(
        runtime_host=RuntimeHostFacts(
            cpu_count=16,
            host_ram_mb=65536,
            architecture="x86_64",
        ),
        deployment_path=DeploymentPath.SYSTEMD,
        os_id="debian",
        os_version="13",
        service_manager="systemd",
        package_manager="apt",
        available_commands=frozenset({"nvidia-smi", "systemctl"}),
        privilege_mode=PrivilegeMode.ROOT,
    )


def _nixos_snapshot() -> SetupHostSnapshot:
    return SetupHostSnapshot(
        runtime_host=RuntimeHostFacts(
            cpu_count=16,
            host_ram_mb=65536,
            architecture="x86_64",
        ),
        deployment_path=DeploymentPath.NIXOS,
        os_id="nixos",
        os_version="26.11",
        service_manager="systemd",
        package_manager="nix",
        available_commands=frozenset({"nix", "nvidia-smi", "systemctl"}),
        privilege_mode=PrivilegeMode.SUDO,
    )


def _one_gpu() -> tuple[DiscoveredGpu, ...]:
    return (DiscoveredGpu("GPU-one", 24576, "8.6"),)


def _exact_token(prompt: str) -> str:
    marker = "Type '"
    assert marker in prompt
    return prompt.split(marker, 1)[1].split("'", 1)[0]


class RecordingInstaller:
    instances: list["RecordingInstaller"] = []

    def __init__(self) -> None:
        self.worker_calls: list[dict[str, object]] = []
        type(self).instances.append(self)

    def install_worker(self, **kwargs: object) -> SystemdBootstrapResult:
        self.worker_calls.append(kwargs)
        return SystemdBootstrapResult(
            worker_installed=True,
            worker_ready=bool(kwargs["start"]),
        )


def test_smoke_capabilities_are_authoritative_before_review_and_in_generic_toml(
    monkeypatch,
) -> None:
    from astrumweaver.setup import tui as tui_module

    RecordingInstaller.instances.clear()
    monkeypatch.setattr(
        tui_module,
        "SystemdFirstRunInstaller",
        RecordingInstaller,
    )

    reviewed_workers = []
    original_review = tui_module._render_first_run_review

    def record_review(io, **kwargs):
        reviewed_workers.append(kwargs["worker"])
        return original_review(io, **kwargs)

    monkeypatch.setattr(tui_module, "_render_first_run_review", record_review)

    io = ScriptedIO(
        [
            "2",  # worker only
            "",  # all GPUs
            "",  # worker ID
            "",  # worker class
            "",  # smoke
            "",  # Control URL
            "worker-token",  # hidden worker token
            _exact_token,
        ]
    )

    result = run_first_run_tui(
        io=io,
        snapshot=_systemd_snapshot(),
        gpus=_one_gpu(),
    )

    assert result.status is TuiRunStatus.APPLIED
    assert len(reviewed_workers) == 1
    reviewed_worker = reviewed_workers[0]
    assert reviewed_worker.capabilities == frozenset({"debug.echo"})
    assert "Capabilities: debug.echo" in io.output
    require_executor_capabilities(
        StructuredEchoExecutor(),
        reviewed_worker.capabilities,
    )

    installer = RecordingInstaller.instances[-1]
    worker_toml = str(installer.worker_calls[0]["worker_toml"])
    parsed = tomllib.loads(worker_toml)
    assert parsed["executor"]["factory"] == (
        "astrumweaver.executors.structured_echo:create_executor"
    )
    assert "runtime" not in parsed
    advertised = frozenset(parsed["worker"]["capabilities"])
    assert advertised == reviewed_worker.capabilities
    assert advertised == StructuredEchoExecutor.capabilities
    require_executor_capabilities(StructuredEchoExecutor(), advertised)


def test_smoke_capabilities_are_rendered_in_nixos_snippet(tmp_path) -> None:
    output_path = tmp_path / "astrumweaver-first-run.nix"
    io = ScriptedIO(
        [
            "2",  # worker only
            "",  # all GPUs
            "",  # worker ID
            "",  # worker class
            "",  # smoke
            "http://control.example:9000",
            "worker-token",  # hidden worker token
            str(output_path),
            _exact_token,
            "n",  # do not write protected env in the test
        ]
    )

    result = run_first_run_tui(
        io=io,
        snapshot=_nixos_snapshot(),
        gpus=_one_gpu(),
    )

    assert result.status is TuiRunStatus.PLANNED
    rendered = output_path.read_text(encoding="utf-8")
    assert 'capabilities = [ "debug.echo" ];' in rendered
    assert '"llm.chat"' not in rendered
    assert '"text.generate"' not in rendered
    assert 'executorFactory = "astrumweaver.executors.structured_echo:create_executor";' in rendered
    require_executor_capabilities(
        StructuredEchoExecutor(),
        frozenset({"debug.echo"}),
    )


def test_runtime_worker_builder_keeps_llm_capabilities() -> None:
    worker = build_worker_spec(
        worker_id="worker-runtime",
        worker_class="gpu-single",
        gpus=_one_gpu(),
    )

    assert worker.capabilities == frozenset({"llm.chat", "text.generate"})
