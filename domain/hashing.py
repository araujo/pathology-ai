"""Canonical, deterministic content fingerprints (stdlib only)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from typing import Any


def to_canonical(value: Any) -> Any:
    """Convert domain values to a JSON-compatible structure with deterministic ordering."""
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_canonical(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {
            str(k): to_canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, frozenset | set):
        return sorted((to_canonical(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, tuple | list):
        return [to_canonical(v) for v in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not canonicalisable")


def canonical_json(value: Any) -> str:
    return json.dumps(to_canonical(value), sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    """Full SHA-256 hex digest of the canonical JSON form."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def short_digest(value: Any, prefix: str, length: int = 32) -> str:
    """Namespaced identifier `pai-<prefix>-<hex>`; never reuses external/legacy id schemes."""
    return f"pai-{prefix}-{digest(value)[:length]}"
