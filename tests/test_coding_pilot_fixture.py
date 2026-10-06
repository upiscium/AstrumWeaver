from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1] / "validation" / "coding-pilot" / "fixture"
BASELINE = "sha256:35cc690e909857f306f3ccf130ae3af244df1c742470a91a3f7cfd9f2fb289f4"
CHECKERS = {
    "check_discovery.py": "sha256:9dafceeecffecb2887668cad5f6ac93294e3244bfc67e403ef6824d5ed4ea0aa",
    "check_test_addition.py": "sha256:1aa016f109b429a1fe80df3c736f2a2c2a41d55c05d45843b8b0f2f9f9cce9aa",
    "check_scoped_fix.py": "sha256:97e9c5e999794323c12db830c7e0800321874ed51cd01795bb88e0ed6117b82c",
}
BASELINE_FILES = (
    "orbit/__init__.py",
    "orbit/config.py",
    "orbit/math.py",
    "orbit/service.py",
    "tests/test_config.py",
    "tests/test_math.py",
)


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_digest() -> str:
    h = hashlib.sha256()
    for name in sorted(BASELINE_FILES):
        data = (ROOT / name).read_bytes()
        encoded = name.encode()
        h.update(len(encoded).to_bytes(4, "big"))
        h.update(encoded)
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return "sha256:" + h.hexdigest()


def copy_fixture(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    shutil.copytree(ROOT / "orbit", work / "orbit")
    shutil.copytree(ROOT / "tests", work / "tests")
    return work


def run_checker(name: str, work: Path, *, stdin: str | None = None):
    env = {"PYTHONPATH": str(work)}
    return subprocess.run(
        [sys.executable, str(ROOT / name)],
        cwd=work,
        env=env,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )


def test_fixture_and_checker_digests_are_frozen():
    assert baseline_digest() == BASELINE
    assert {name: digest(ROOT / name) for name in CHECKERS} == CHECKERS


def test_discovery_checker_accepts_only_exact_file_line(tmp_path):
    work = copy_fixture(tmp_path)

    assert run_checker(
        "check_discovery.py", work, stdin="orbit/service.py:5"
    ).returncode == 0
    assert run_checker(
        "check_discovery.py", work, stdin="orbit/service.py:4"
    ).returncode != 0


def test_test_addition_checker_requires_boundaries_and_green_test(tmp_path):
    work = copy_fixture(tmp_path)

    assert run_checker("check_test_addition.py", work).returncode != 0
    path = work / "tests" / "test_math.py"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "\n\ndef test_clamp_score_boundaries():\n"
        + "    assert clamp_score(120) == 100\n"
        + "    assert clamp_score(-5) == 0\n",
        encoding="utf-8",
    )
    assert run_checker("check_test_addition.py", work).returncode == 0


def test_scoped_fix_checker_has_negative_control_then_passes(tmp_path):
    work = copy_fixture(tmp_path)

    assert run_checker("check_scoped_fix.py", work).returncode != 0
    (work / "orbit" / "config.py").write_text(
        "def parse_limit(value: str) -> int:\n"
        "    parsed = int(value)\n"
        "    if parsed < 0:\n"
        "        raise ValueError('limit must be non-negative')\n"
        "    return parsed\n",
        encoding="utf-8",
    )
    assert run_checker("check_scoped_fix.py", work).returncode == 0