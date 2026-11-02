"""Lazy patch reads and the bounded read -> transform -> encode -> write loop.

Memory is bounded by construction, independent of the number of patches:

* exactly one decoded region is alive at a time (`LazyPatchSource.borrow`); it is transformed
  into its slot of one preallocated model-input batch buffer and released;
* the batch buffer is allocated once with `budget.batch_patches` rows and reused;
* one encoder output batch exists at a time and is written, synced and committed to the store
  before the next batch is read.

Everything runs on the calling thread with the caller's image handle: native reader handles are
never shared across threads or forked processes (ADR-005 D10), and the handle is never closed
here (the caller owns it). Chunks that the store already holds as verified commits are skipped
without reading their pixels.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from domain.embedding import (
    BatchBudget,
    BatchPolicy,
    EmbeddingRecord,
    EmbeddingStoreHeader,
    TensorLayout,
    TransformChain,
)
from domain.errors import (
    CorruptImageError,
    EmbeddingBudgetError,
    EmbeddingOutputError,
    ModelInputContractError,
)
from domain.reading import DataValidity, ImageDescription, RegionReadRequest
from plugins.embedding.transforms import apply_chain
from ports import EmbeddingEncoder, EmbeddingStore, EncodeContext, OpenedImage

Array = npt.NDArray[Any]


class LazyPatchSource:
    """Reads one patch at a time from a caller-owned `OpenedImage`. Never caches, never closes."""

    def __init__(
        self, image: OpenedImage, description: ImageDescription, width: int, height: int
    ) -> None:
        self._image = image
        self._d = description
        self._shape = (height, width, description.samples_per_pixel)
        self.reads = 0
        self.partially_acquired = 0
        self.live = 0
        self.max_live = 0

    @contextmanager
    def borrow(self, record: EmbeddingRecord) -> Iterator[Array]:
        """The decoded pixels of `record`, valid only inside the `with` block."""
        lr = record.level_rect
        region = self._image.read_region(
            RegionReadRequest(
                lr.level,
                int(lr.x),
                int(lr.y),
                int(lr.width),
                int(lr.height),
                scene=self._d.scene,
            )
        )
        pixels: Array = np.asarray(region.pixels)
        if pixels.shape != self._shape or str(pixels.dtype) != self._d.dtype:
            raise CorruptImageError(
                f"patch {record.patch_id} decoded as {pixels.shape} {pixels.dtype}, expected "
                f"{self._shape} {self._d.dtype}"
            )
        if region.level0 != record.level0 or region.channel_order is not self._d.channel_order:
            raise CorruptImageError(f"patch {record.patch_id} was read at {region.level0}")
        if region.valid_data.validity in (
            DataValidity.PARTIALLY_ACQUIRED,
            DataValidity.NONE_ACQUIRED,
        ):
            self.partially_acquired += 1
        del region
        self.reads += 1
        self.live += 1
        self.max_live = max(self.max_live, self.live)
        try:
            yield pixels
        finally:
            self.live -= 1
            del pixels


@dataclass(frozen=True, slots=True)
class LoopCounters:
    patches_encoded: int
    patches_resumed: int
    batches_encoded: int
    chunks_committed: int
    chunks_resumed: int


def model_input_shape(encoder: EmbeddingEncoder) -> tuple[int, int, int]:
    s = encoder.input_spec
    if s.layout is TensorLayout.CHW:
        return (s.channels, s.height, s.width)
    return (s.height, s.width, s.channels)


def validate_output(out: object, n: int, encoder: EmbeddingEncoder, budget: BatchBudget) -> Array:
    """Rows of the declared shape and dtype, finite and within the output budget, or an error."""
    identity = encoder.identity
    rows: Array = np.asarray(out)
    if rows.shape != (n, identity.output_dim):
        raise EmbeddingOutputError(
            f"encoder {identity.component.name} returned {rows.shape}, expected "
            f"({n}, {identity.output_dim})"
        )
    if str(rows.dtype) != identity.output_dtype:
        raise EmbeddingOutputError(
            f"encoder returned {rows.dtype}, declared {identity.output_dtype}"
        )
    if not bool(np.isfinite(rows).all()):
        raise EmbeddingOutputError("encoder returned non-finite values")
    if rows.nbytes > budget.output_batch_bytes:
        raise EmbeddingBudgetError(
            f"encoder output of {rows.nbytes} bytes exceeds the planned {budget.output_batch_bytes}"
        )
    return rows


def encode_into_store(
    *,
    source: LazyPatchSource,
    chain: TransformChain,
    encoder: EmbeddingEncoder,
    store: EmbeddingStore,
    header: EmbeddingStoreHeader,
    budget: BatchBudget,
    policy: BatchPolicy,
    skip: frozenset[int],
) -> LoopCounters:
    lay = header.layout
    if lay.chunk_rows != budget.batch_patches:
        raise EmbeddingBudgetError("store chunks must equal the planned batch size")
    shape = model_input_shape(encoder)
    buffer: Array = np.empty((budget.batch_patches, *shape), dtype=encoder.input_spec.dtype)
    if buffer.nbytes != budget.model_input_batch_bytes:
        raise EmbeddingBudgetError("the model-input buffer differs from the planned budget")
    encoded = resumed = batches = commits = chunks_resumed = 0
    for index in range(lay.chunk_count):
        records = header.chunk_records(index)
        if index in skip:
            resumed += len(records)
            chunks_resumed += 1
            continue
        for i, record in enumerate(records):
            with source.borrow(record) as pixels:
                x = apply_chain(chain, pixels)
                if x.shape != shape or str(x.dtype) != encoder.input_spec.dtype:
                    raise ModelInputContractError(
                        f"transforms produced {x.shape} {x.dtype}, the encoder declares {shape}"
                    )
                buffer[i] = x
                del x
            del pixels  # release the decoded region before the next read
        n = len(records)
        context = EncodeContext(
            batch_index=index,
            first_row=records[0].row,
            patch_ids=tuple(r.patch_id for r in records),
            precision=policy.precision,
            device=policy.device,
        )
        rows = validate_output(encoder.encode_batch(buffer[:n], context), n, encoder, budget)
        store.write_chunk(index, rows)
        store.commit_chunk(index)
        del rows
        encoded += n
        batches += 1
        commits += 1
    return LoopCounters(encoded, resumed, batches, commits, chunks_resumed)
