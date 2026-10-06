from __future__ import annotations

import sys

text = sys.stdin.read().strip()
expected = "orbit/service.py:5"
if text != expected:
    raise SystemExit(f"expected exact discovery evidence {expected!r}")