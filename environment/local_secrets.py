"""Local secret loading helpers for developer-only runtime settings."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import tomllib
from typing import Any


def _secrets_path() -> Path:
    return Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"


@lru_cache(maxsize=1)
def _load_local_secrets() -> dict[str, Any]:
    path = _secrets_path()
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return {str(key): value for key, value in data.items()}


def get_runtime_secret(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value

    secrets = _load_local_secrets()
    for name in names:
        value = secrets.get(name)
        if isinstance(value, str) and value:
            return value

    return default
