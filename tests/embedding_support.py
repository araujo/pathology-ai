"""Helpers for SPEC-005 tests: embedding runs on in-memory slides, instrumentation and faults.

Content is synthetic. The reference encoder's embeddings carry no medical meaning.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from application.embedding_service import WSI_EMBEDDING_TASK
from application.runner import ExecutionOutcome, ExecutionRequest
from bootstrap import Platform
from domain.embedding import (
    ChunkCommit,
    DurableStoreReference,
    EmbeddingManifest,
    EmbeddingStoreHeader,
    EncoderIdentity,
    ModelInputSpec,
    RecoveryPolicy,
    StoreInspection,
    StoreVerification,
)
from domain.errors import CorruptImageError
from domain.foundation import EncoderDescriptor, EncoderRuntimeReport
from domain.ids import PipelineId, Version
from domain.pipeline import ConfigValue, PipelineDefinition, PipelineRef
from domain.reading import PixelRegion, RegionReadRequest
from infrastructure.embedding_store import NumpyEmbeddingStoreProvider
from plugins.embedding.operations import EMBED_PATCHES, EmbeddingOperation
from plugins.embedding.reference import ReferenceProjectionEncoder, ReferenceProjectionFactory
from ports import EMBEDDING_STORE_EXTRA, EmbeddingBagPayload, EmbeddingStore, EncodeContext
from tests.preprocessing_support import ArrayImage, ArrayReader, slide_rgb

EMBED_PIPELINE = PipelineRef(PipelineId("wsi-patch-embedding"), Version("1.0.0"))
PATCH = 256

# Level 0 (0.25 µm/px) of an in-memory slide; the stride sets the patch count without changing
# the image: stride 256 -> 16 patches, stride 64 -> 169 patches on a 1024 x 1024 slide.
LEVEL0 = (("plan", "level", 0), ("plan", "target_spacing_um", 0.0))


def stride(px: float) -> tuple[tuple[str, str, ConfigValue], ...]:
    return LEVEL0 + (("plan", "stride_x", px), ("plan", "stride_y", px))


class TrackingImage(ArrayImage):
    """Counts live decoded patch arrays (weak references) and can fail on the n-th patch read."""

    def __init__(
        self, *args: Any, fail_on_patch: int | None = None, patch_px: int = PATCH, **kw: Any
    ) -> None:
        super().__init__(*args, **kw)
        self.patch_px = patch_px
        self.patch_reads = 0
        self.live = 0
        self.max_live = 0
        self.fail_on_patch = fail_on_patch

    def _released(self) -> None:
        self.live -= 1

    def read_region(self, request: RegionReadRequest) -> PixelRegion:
        is_patch = (request.width, request.height) == (self.patch_px, self.patch_px)
        if is_patch:
            self.patch_reads += 1
            if self.fail_on_patch is not None and self.patch_reads == self.fail_on_patch:
                raise CorruptImageError(f"injected read failure at patch {self.patch_reads}")
        region = super().read_region(request)
        if is_patch:
            self.live += 1
            self.max_live = max(self.max_live, self.live)
            weakref.finalize(region.pixels, self._released)
        return region


def tissue_slide(image_id: str = "emb-slide", size: int = 1024, **kw: Any) -> TrackingImage:
    """Mostly candidate tissue on acquired glass (every grid patch qualifies), with texture so
    that patches differ."""
    rgb, acquired = slide_rgb(size, size, tissue=((0, 0, size, size - 64),), unacquired=())
    rng = np.random.default_rng(7)
    noise = rng.integers(-12, 13, size=rgb.shape, dtype=np.int16)
    rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return TrackingImage(image_id, rgb, acquired=acquired, **kw)


def ensure_reader(platform: Platform, image: ArrayImage) -> None:
    d = image.describe()
    if platform.readers._by_format.find(d.asset.image_format.code) is None:
        platform.readers.register(ArrayReader(d.asset.image_format.code, d.channel_order))


def with_embed_config(pipeline: PipelineDefinition, **config: ConfigValue) -> PipelineDefinition:
    """`pipeline` with its `embed` stage configuration replaced by `config` merged over it.

    Overrides cannot add keys; an encoder with other keys (SPEC-006) needs its own definition."""
    stages = tuple(
        replace(s, config=tuple(sorted({**s.config_dict(), **config}.items())))
        if str(s.stage_id) == "embed"
        else s
        for s in pipeline.stages
    )
    return replace(pipeline, stages=stages)


def run_embedding(
    platform: Platform,
    image: ArrayImage,
    out: Path | object,
    overrides: Sequence[tuple[str, str, ConfigValue]] = (),
    pipeline: PipelineDefinition | None = None,
) -> ExecutionOutcome:
    """`out` is an output directory or an `EmbeddingStoreProvider`."""
    ensure_reader(platform, image)
    provider = NumpyEmbeddingStoreProvider(out) if isinstance(out, Path) else out
    return platform.runner.run(
        ExecutionRequest(
            pipeline=pipeline or platform.pipelines.get(EMBED_PIPELINE),
            image=image.describe().asset,
            task=WSI_EMBEDDING_TASK,
            models={},
            preprocessing=None,
            research_use_acknowledged=True,
            source=image,
            config_overrides=tuple(overrides),
            extras={EMBEDDING_STORE_EXTRA: provider},
        )
    )


def bag(outcome: ExecutionOutcome) -> EmbeddingBagPayload:
    payload = outcome.outputs["embeddings"].payload
    assert isinstance(payload, EmbeddingBagPayload)
    return payload


def embedding_operation(platform: Platform) -> EmbeddingOperation:
    op = platform.operations.get(EMBED_PATCHES)
    assert isinstance(op, EmbeddingOperation)
    return op


# --------------------------------------------------------------------------- faulty encoders


class FaultyEncoder:
    def __init__(self, inner: ReferenceProjectionEncoder, mode: str, at_batch: int) -> None:
        self.inner, self.mode, self.at_batch = inner, mode, at_batch
        self.closed = False

    @property
    def identity(self) -> EncoderIdentity:
        return self.inner.identity

    @property
    def input_spec(self) -> ModelInputSpec:
        return self.inner.input_spec

    def encode_batch(self, batch: object, context: EncodeContext) -> object:
        out = self.inner.encode_batch(batch, context)
        if context.batch_index != self.at_batch:
            return out
        if self.mode == "raise":
            raise RuntimeError(f"injected encoder failure at batch {context.batch_index}")
        if self.mode == "nan":
            out = out.copy()
            out[0, 0] = np.nan
            return out
        if self.mode == "dim":
            return out[:, :-1]
        if self.mode == "dtype":
            return out.astype(np.float64)
        raise AssertionError(self.mode)

    def working_bytes(self, batch_patches: int) -> int:
        return self.inner.working_bytes(batch_patches)

    def runtime_report(self) -> EncoderRuntimeReport:
        return self.inner.runtime_report()

    def close(self) -> None:
        self.closed = True
        self.inner.close()


class FaultyFactory:
    """Same identity as the reference encoder (so stores are shared), different name."""

    def __init__(self, name: str, mode: str, at_batch: int) -> None:
        self._name, self.mode, self.at_batch = name, mode, at_batch
        self.created: list[FaultyEncoder] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def descriptor(self) -> EncoderDescriptor:
        return replace(ReferenceProjectionFactory().descriptor, name=self._name)

    def input_spec(self, config: Mapping[str, ConfigValue]) -> ModelInputSpec:
        return ReferenceProjectionFactory().input_spec(config)

    def create(self, config: Mapping[str, ConfigValue]) -> FaultyEncoder:
        enc = FaultyEncoder(ReferenceProjectionFactory().create(config), self.mode, self.at_batch)
        self.created.append(enc)
        return enc


# --------------------------------------------------------------------------- faulty stores


class FaultyStore:
    """Delegates to a real store and raises `OSError` at one boundary (simulated crash)."""

    def __init__(self, inner: EmbeddingStore, fail: str, at: int) -> None:
        self.inner, self.fail, self.at = inner, fail, at

    @property
    def header(self) -> EmbeddingStoreHeader:
        return self.inner.header

    @property
    def reference(self) -> DurableStoreReference:
        return self.inner.reference

    def inspect(self) -> StoreInspection:
        return self.inner.inspect()

    def write_chunk(self, index: int, rows: object) -> None:
        if self.fail == "write" and index == self.at:
            raise OSError(f"injected write failure at chunk {index}")
        self.inner.write_chunk(index, rows)

    def commit_chunk(self, index: int) -> ChunkCommit:
        if self.fail == "commit" and index == self.at:
            raise OSError(f"injected crash before committing chunk {index}")
        return self.inner.commit_chunk(index)

    def verify(self) -> StoreVerification:
        return self.inner.verify()

    def finalize(self, manifest: EmbeddingManifest) -> None:
        if self.fail == "finalize":
            raise OSError("injected crash before finalization")
        self.inner.finalize(manifest)

    def read_rows(self, start: int, stop: int) -> object:
        return self.inner.read_rows(start, stop)

    def close(self) -> None:
        self.inner.close()


class FaultyProvider:
    def __init__(self, root: Path, fail: str, at: int = 0) -> None:
        self.inner = NumpyEmbeddingStoreProvider(root)
        self.fail, self.at = fail, at

    @property
    def format(self) -> str:
        return self.inner.format

    def open(self, header: EmbeddingStoreHeader, *, recovery: RecoveryPolicy) -> FaultyStore:
        return FaultyStore(self.inner.open(header, recovery=recovery), self.fail, self.at)

    def open_existing(self, store_id: str) -> EmbeddingStore:
        return self.inner.open_existing(store_id)
