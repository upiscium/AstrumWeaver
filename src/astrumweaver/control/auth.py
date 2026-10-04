"""Control-plane authentication mode contracts."""

from __future__ import annotations

from enum import StrEnum


class ClientAuthMode(StrEnum):
    """Authentication policy for operator/client job endpoints."""

    BEARER = "bearer"
    NONE = "none"


__all__ = ["ClientAuthMode"]
