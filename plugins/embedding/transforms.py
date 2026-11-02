"""NumPy implementations of the declared transform vocabulary (`domain.embedding`).

Each implementation does exactly what its step declares, on one HWC patch at a time; the chain
was composed and validated before any pixel was read. Every implementation is versioned, and
that version is recorded in the transform chain and therefore in the run identity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from domain.embedding import TRANSFORM_PARAMETERS, TransformChain
from domain.errors import ModelInputContractError
from domain.preprocessing import TransformStep
from domain.provenance import ComponentVersion

TRANSFORM_VERSION = "1"
IMPLEMENTATIONS: dict[str, ComponentVersion] = {
    name: ComponentVersion("transform", f"numpy.{name}", TRANSFORM_VERSION)
    for name in sorted(TRANSFORM_PARAMETERS)
}

Array = npt.NDArray[Any]


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split(","))


def _size(text: str) -> tuple[int, int]:
    w, _, h = text.partition("x")
    return int(w), int(h)


def apply_step(step: TransformStep, x: Array) -> Array:
    p = dict(step.params)
    if step.name == "channel_reorder":
        return x[..., ::-1]
    if step.name == "drop_alpha":
        return x[..., :3]
    if step.name == "cast":
        return x.astype(np.dtype(p["to"]))
    if step.name == "scale":
        scaled: Array = np.multiply(x, x.dtype.type(float(p["factor"])), dtype=x.dtype)
        return scaled
    if step.name == "resample":
        (fw, _), (tw, th) = _size(p["from"]), _size(p["to"])
        f = fw // tw
        blocks = x.reshape(th, f, tw, f, x.shape[2])
        return blocks.mean(axis=(1, 3), dtype=np.float64).astype(x.dtype)
    if step.name == "normalize":
        mean = np.asarray(_floats(p["mean"]), dtype=x.dtype)
        std = np.asarray(_floats(p["std"]), dtype=x.dtype)
        normalized: Array = np.divide(np.subtract(x, mean, dtype=x.dtype), std, dtype=x.dtype)
        return normalized
    if step.name == "layout":
        return np.transpose(x, (2, 0, 1)) if p["to"] == "chw" else x
    raise ModelInputContractError(f"no implementation for transform {step.name!r}")


def apply_chain(chain: TransformChain, pixels: Array) -> Array:
    """Apply every declared step in order to one decoded HWC patch."""
    x = pixels
    for t in chain.steps:
        if IMPLEMENTATIONS.get(t.step.name) != t.implementation:
            raise ModelInputContractError(
                f"transform {t.step.name} was declared for {t.implementation}, which is not "
                "this implementation"
            )
        x = apply_step(t.step, x)
    return x
