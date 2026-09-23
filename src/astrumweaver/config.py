"""Shared runtime configuration helpers."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Mapping


class ConfigurationError(ValueError):
    """Invalid or incomplete runtime configuration."""


def load_toml(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            value = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"configuration file not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"invalid TOML configuration: {config_path}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError("configuration root must be a table")
    return value


def table(
    value: Mapping[str, Any],
    name: str,
    *,
    required: bool = False,
) -> dict[str, Any]:
    raw = value.get(name)
    if raw is None:
        if required:
            raise ConfigurationError(f"missing [{name}] configuration table")
        return {}
    if not isinstance(raw, dict):
        raise ConfigurationError(f"[{name}] must be a TOML table")
    return dict(raw)


def require_string(value: Mapping[str, Any], key: str, *, context: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigurationError(f"{context}.{key} must be a non-blank string")
    return raw.strip()


def optional_string(
    value: Mapping[str, Any],
    key: str,
    *,
    context: str,
    default: str | None = None,
) -> str | None:
    raw = value.get(key, default)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigurationError(f"{context}.{key} must be a non-blank string")
    return raw.strip()


def positive_int(
    value: Mapping[str, Any],
    key: str,
    *,
    context: str,
    default: int,
) -> int:
    raw = value.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ConfigurationError(f"{context}.{key} must be a positive integer")
    return raw


def positive_float(
    value: Mapping[str, Any],
    key: str,
    *,
    context: str,
    default: float,
) -> float:
    raw = value.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        raise ConfigurationError(f"{context}.{key} must be positive")
    return float(raw)


def required_environment(name: str) -> str:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        raise ConfigurationError(f"required environment variable is missing: {name}")
    return raw.strip()
