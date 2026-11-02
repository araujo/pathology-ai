"""Deterministic synthetic MIL cohorts written as finalized SPEC-005 embedding stores.

SYNTHETIC: no image is read and no encoder runs. Rows are drawn from a seeded generator, and
the labels are the generator's own classes. A model trained on them demonstrates that the
training, evaluation and prediction path works end to end. It is not evidence of any
biological, diagnostic or biomarker property.

Signal model (controlled separability): every instance is standard normal noise. In a slide of
class `c` (every class for multiclass tasks, the positive class only for binary tasks), a
fraction `witness_fraction` of the instances is shifted by `separability` along a seeded
unit direction `u_c`. Bags vary in length, and groups hold one to several slides with the
same label.

Stores go through the `EmbeddingStoreProvider` port with the usual chunk commits, verification
and final manifest, so they are indistinguishable in format from encoder-produced stores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from domain.embedding import (
    EmbeddingManifest,
    EmbeddingRecord,
    EmbeddingRunIdentity,
    EmbeddingStoreHeader,
    EncoderIdentity,
    ModelInputSpec,
    Precision,
    RecoveryPolicy,
    ResolutionRecord,
    StoreDescriptor,
    StoreLayout,
    StoreState,
    TensorLayout,
    TransformChain,
    patch_ids_digest,
)
from domain.geometry import Level0Rect, LevelRect
from domain.hashing import digest, short_digest
from domain.ids import ImageId
from domain.image import ChannelOrder
from domain.manifest import PixelStorage
from domain.provenance import ComponentVersion
from ports import EmbeddingStoreProvider

PLUGIN_VERSION = "0.1.0"
PATCH_PX = 32
CHUNK_ROWS = 64
SYNTHETIC_LIMITATION = (
    "synthetic MIL bag: no image was read and no encoder ran; rows are seeded noise with an "
    "injected class signal and carry no biological or clinical meaning"
)
_SPEC = ModelInputSpec(
    width=PATCH_PX,
    height=PATCH_PX,
    channel_order=ChannelOrder.RGB,
    channels=3,
    dtype="float32",
    scaled_range=(0.0, 1.0),
    layout=TensorLayout.CHW,
    normalization=None,
    pixel_spacing_um=None,
    accepts_variable_spacing=True,
    accepts_unknown_spacing=True,
)


def synthetic_encoder(dim: int, variant: str = "a") -> EncoderIdentity:
    """The identity synthetic bags claim. Another `variant` is another (incompatible) encoder."""
    return EncoderIdentity(
        component=ComponentVersion("encoder", "synthetic-mil-bag", PLUGIN_VERSION),
        model=None,
        weights_checksum=None,
        synthetic=True,
        parameters=(("variant", variant),),
        output_dim=dim,
        output_dtype="float32",
    )


def _records(image_id: str, n: int) -> tuple[EmbeddingRecord, ...]:
    side = max(1, int(np.ceil(np.sqrt(n))))
    out = []
    for i in range(n):
        x, y = (i % side) * PATCH_PX, (i // side) * PATCH_PX
        out.append(
            EmbeddingRecord(
                row=i,
                patch_id=short_digest((image_id, i), "patch"),
                plan_index=i,
                level_rect=LevelRect(0, x, y, PATCH_PX, PATCH_PX),
                level0=Level0Rect(x, y, PATCH_PX, PATCH_PX),
                effective_spacing_um=None,
                field_of_view_um=None,
            )
        )
    return tuple(out)


def write_bag(
    provider: EmbeddingStoreProvider,
    image_id: str,
    rows: npt.NDArray[np.float32],
    *,
    variant: str = "a",
    device: str = "cpu",
) -> str:
    """Write `rows` as one finalized store and return its id (idempotent for equal input)."""
    n, dim = rows.shape
    records = _records(image_id, n)
    identity = EmbeddingRunIdentity(
        image_id=ImageId(image_id),
        source_integrity=None,
        scene=0,
        plan_digest=digest(("synthetic-plan", image_id, n)),
        patch_manifest_digest=digest(("synthetic-patches", image_id, n)),
        records_digest=patch_ids_digest(records),
        encoder=synthetic_encoder(dim, variant),
        input_spec=_SPEC,
        transforms=TransformChain(),
        precision=Precision.FP32,
        device=device,
    )
    header = EmbeddingStoreHeader(
        identity.store_id, identity, StoreLayout(n, dim, "float32", CHUNK_ROWS), records
    )
    store = provider.open(header, recovery=RecoveryPolicy.RESUME_VERIFIED)
    try:
        inspection = store.inspect()
        for index in inspection.missing:
            start, stop = header.layout.chunk_range(index)
            store.write_chunk(index, np.ascontiguousarray(rows[start:stop], dtype=np.float32))
            store.commit_chunk(index)
        verified = store.verify()
        component = ComponentVersion("synthetic", "synthetic-mil-bag", PLUGIN_VERSION)
        manifest = EmbeddingManifest(
            image_id=identity.image_id,
            source_integrity=None,
            scene=0,
            pipeline=None,
            reader=component,
            backend=component,
            operation=ComponentVersion("operation", "synthetic.mil-bags", "1"),
            plan_digest=identity.plan_digest,
            patch_manifest_digest=identity.patch_manifest_digest,
            patch_manifest_schema="synthetic",
            patch_pixel_storage=PixelStorage.NOT_MATERIALIZED,
            run_identity=identity.value,
            encoder=identity.encoder,
            input_spec=_SPEC,
            transforms=identity.transforms,
            resolution=ResolutionRecord(
                source_spacing_um=None,
                source_spacing_source=None,
                extraction_level=0,
                extraction_downsample=1.0,
                extraction_spacing_um=None,
                requested_spacing_um=None,
                spacing_relative_error=None,
                patch_px=(PATCH_PX, PATCH_PX),
                field_of_view_um=None,
                model_input_px=(PATCH_PX, PATCH_PX),
                model_input_spacing_um=None,
                model_required_spacing_um=None,
                model_accepts_variable_spacing=True,
                resampling="none",
            ),
            precision=Precision.FP32,
            device=device,
            embedding_dim=dim,
            embedding_dtype="float32",
            row_count=n,
            records=records,
            store=StoreDescriptor(
                format=provider.format,
                schema_version=header.schema_version,
                store_id=header.store_id,
                chunk_rows=CHUNK_ROWS,
                chunk_count=header.layout.chunk_count,
                chunks_digest=verified.chunks_digest,
            ),
            content_digest=verified.content_digest,
            status=StoreState.FINALIZED,
            limitations=(SYNTHETIC_LIMITATION,),
        )
        store.finalize(manifest)
    finally:
        store.close()
    return header.store_id


@dataclass(frozen=True, slots=True)
class SyntheticSlide:
    slide_id: str
    group_id: str
    label: str
    rows: npt.NDArray[np.float32]


def synthetic_cohort(
    classes: Sequence[str],
    *,
    binary: bool,
    groups_per_class: int,
    slides_per_group: tuple[int, int] = (1, 3),
    instances: tuple[int, int] = (8, 64),
    dim: int = 32,
    separability: float = 1.5,
    witness_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[SyntheticSlide, ...]:
    """A labelled cohort with groups (patients) of one to several slides each."""
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((len(classes), dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    slides = []
    for c, name in enumerate(classes):
        signal = not binary or c == 1
        for g in range(groups_per_class):
            group_id = f"group-{c:02d}-{g:03d}"
            for s in range(int(rng.integers(slides_per_group[0], slides_per_group[1] + 1))):
                n = int(rng.integers(instances[0], instances[1] + 1))
                rows = rng.standard_normal((n, dim))
                if signal:
                    witnesses = rng.random(n) < witness_fraction
                    witnesses[int(rng.integers(n))] = True
                    rows[witnesses] += separability * directions[c]
                slides.append(
                    SyntheticSlide(
                        f"slide-{c:02d}-{g:03d}-{s}", group_id, name, rows.astype(np.float32)
                    )
                )
    return tuple(slides)
