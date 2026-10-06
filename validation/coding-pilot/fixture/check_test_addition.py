from __future__ import annotations

from pathlib import Path
import subprocess
import sys

path = Path("tests/test_math.py")
text = path.read_text(encoding="utf-8")
required = ("assert clamp_score(120) == 100", "assert clamp_score(-5) == 0")
if not all(item in text for item in required):
    raise SystemExit("required boundary assertions are missing")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests/test_math.py"],
    check=False,
)
raise SystemExit(result.returncode)