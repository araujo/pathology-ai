"""A deterministic NumPy reference encoder: an architectural test fixture, nothing more.

`reference-projection` pools each channel of a normalised 256x256 RGB patch on an 8x8 grid, adds
the per-channel standard deviation, and projects these 195 statistics with a fixed pseudo-random
matrix (seeded PCG64) followed by `tanh`. It has no trained weights, downloads nothing and needs
no deep-learning framework.

Its embeddings carry no biological, diagnostic or clinical meaning and must never be used as
evidence about tissue. They exist to exercise streaming, storage, identity and recovery.

Computation runs per sample in float64 with `einsum` (no BLAS), so every row depends only on
its own patch: output is identical for identical input bytes, across batch
sizes, repeated runs and fresh processes on the same platform.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from domain.embedding import (
    EncoderIdentity,
    ModelInputSpec,
    Normalization,
    Precision,
    TensorLayout,
)
from domain.errors import ModelInputContractError, UnsupportedModelInputError
from domain.foundation import (
    DeviceKind,
    EncoderDescriptor,
    EncoderRuntimeReport,
    PreprocessingProfile,
    ProfileStatus,
)
from domain.image import ChannelOrder
from domain.pipeline import ConfigValue
from domain.provenance import ComponentVersion
from plugins.preprocessing.detection import cfg_int
from ports import EncodeContext

REFERENCE_PROJECTION = "reference-projection"
REFERENCE_VERSION = "1"
GRID = 8
SEED = 20261020
MAX_DIM = 4096

INPUT_SPEC = ModelInputSpec(
    width=256,
    height=256,
    channel_order=ChannelOrder.RGB,
    channels=3,
    dtype="float32",
    scaled_range=(0.0, 1.0),
    layout=TensorLayout.CHW,
    normalization=Normalization((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # Explicit: the fixture accepts any (or unknown) spacing; it does not interpret scale.
    pixel_spacing_um=None,
    accepts_variable_spacing=True,
    accepts_unknown_spacing=True,
    size_divisor=GRID,
    precisions=frozenset({Precision.FP32}),
)
_FEATURES = INPUT_SPEC.channels * GRID * GRID + INPUT_SPEC.channels
COMPONENT = ComponentVersion("encoder", REFERENCE_PROJECTION, REFERENCE_VERSION)
DESCRIPTOR = EncoderDescriptor(
    name=REFERENCE_PROJECTION,
    adapter=COMPONENT,
    architecture="pooled-statistics-random-projection",
    # The output dimension is configurable (`embedding_dim`); this is the pipeline default.
    output_dim=64,
    output_dtype="float32",
    devices=frozenset({DeviceKind.CPU}),
    precisions=frozenset({Precision.FP32}),
    profiles=(
        PreprocessingProfile(
            profile_id="reference",
            version="1",
            official=False,
            status=ProfileStatus.AVAILABLE,
            description="256x256 RGB, scaled to [0, 1], mean/std 0.5, CHW; any spacing",
            evidence=("synthetic fixture: the contract is its own definition",),
        ),
    ),
    synthetic=True,
    batch_invariant=True,
)


class ReferenceProjectionEncoder:
    def __init__(self, embedding_dim: int) -> None:
        if not 1 <= embedding_dim <= MAX_DIM:
            raise ModelInputContractError(f"embedding_dim must lie in 1..{MAX_DIM}")
        rng = np.random.Generator(np.random.PCG64(SEED))
        self._weights = rng.standard_normal((_FEATURES, embedding_dim)) / math.sqrt(_FEATURES)
        checksum = hashlib.sha256(self._weights.astype("<f8").tobytes()).hexdigest()
        self._identity = EncoderIdentity(
            component=COMPONENT,
            model=None,
            weights_checksum=f"sha256:{checksum}",
            synthetic=True,
            parameters=(("embedding_dim", embedding_dim), ("grid", GRID), ("seed", SEED)),
            output_dim=embedding_dim,
            output_dtype="float32",
        )
        self._closed = False

    @property
    def identity(self) -> EncoderIdentity:
        return self._identity

    @property
    def input_spec(self) -> ModelInputSpec:
        return INPUT_SPEC

    def encode_batch(self, batch: object, context: EncodeContext) -> npt.NDArray[np.float32]:
        if self._closed:
            raise UnsupportedModelInputError("the encoder is closed")
        x: npt.NDArray[Any] = np.asarray(batch)
        s = INPUT_SPEC
        if x.ndim != 4 or x.shape[1:] != (s.channels, s.height, s.width):
            raise UnsupportedModelInputError(f"expected (b, 3, 256, 256), got {x.shape}")
        if x.dtype != np.float32:
            raise UnsupportedModelInputError("expected float32 input")
        if context.precision is not Precision.FP32 or len(context.patch_ids) != x.shape[0]:
            raise UnsupportedModelInputError("context does not match the batch")
        out = np.empty((x.shape[0], self._identity.output_dim), dtype=np.float32)
        for i in range(x.shape[0]):  # one sample at a time: bounded memory, batch-invariant rows
            out[i] = self._encode_one(x[i])
        return out

    def _encode_one(self, sample: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        if not bool(np.isfinite(sample).all()):
            raise UnsupportedModelInputError("expected finite input")
        c, cell = INPUT_SPEC.channels, INPUT_SPEC.width // GRID
        x = sample.astype(np.float64)
        pooled = x.reshape(c, GRID, cell, GRID, cell).mean(axis=(2, 4))
        spread = x.reshape(c, -1).std(axis=1)
        features = np.concatenate([pooled.reshape(-1), spread])
        projected: npt.NDArray[np.float64] = np.einsum("f,fd->d", features, self._weights)
        return np.tanh(projected).astype(np.float32)

    def working_bytes(self, batch_patches: int) -> int:
        """Per call: one float64 copy of one sample plus its reductions (independent of batch)."""
        return 2 * INPUT_SPEC.channels * INPUT_SPEC.height * INPUT_SPEC.width * 8

    def runtime_report(self) -> EncoderRuntimeReport:
        return EncoderRuntimeReport()

    def close(self) -> None:
        self._closed = True


class ReferenceProjectionFactory:
    @property
    def name(self) -> str:
        return REFERENCE_PROJECTION

    @property
    def descriptor(self) -> EncoderDescriptor:
        return DESCRIPTOR

    def input_spec(self, config: Mapping[str, ConfigValue]) -> ModelInputSpec:
        return INPUT_SPEC

    def create(self, config: Mapping[str, ConfigValue]) -> ReferenceProjectionEncoder:
        return ReferenceProjectionEncoder(cfg_int(config, "embedding_dim", 64))
