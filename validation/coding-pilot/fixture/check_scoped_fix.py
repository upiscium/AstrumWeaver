from __future__ import annotations

import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests/test_config.py"],
    check=False,
)
raise SystemExit(result.returncode)