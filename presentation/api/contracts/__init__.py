"""JSON Schema contract files, loadable as package resources."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


def load_schema(api: str, version: str, name: str) -> dict[str, Any]:
    path = resources.files(__name__).joinpath(api, version, f"{name}.schema.json")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def schema_names(api: str, version: str) -> tuple[str, ...]:
    folder = resources.files(__name__).joinpath(api, version)
    return tuple(
        sorted(
            p.name.removesuffix(".schema.json")
            for p in folder.iterdir()
            if p.name.endswith(".schema.json")
        )
    )
