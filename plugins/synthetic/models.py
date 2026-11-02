"""Synthetic model plugins: random-projection encoder, MIL-like aggregator, direct segmenter.

Weights are derived deterministically from the model artifact checksum. These
are architecture demonstrations; they are not trained and have no medical meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from domain.errors import ExecutionUnsupportedError, UnsupportedModelInputError
from domain.geometry import Level0Rect, LevelRect, PyramidalGeometry, RectangleGeometry
from domain.hashing import digest, short_digest
from domain.ids import EvidenceId
from domain.kinds import SEGMENTATION_2D
from domain.model import ModelDefinition
from domain.prediction import (
    EvidenceSemantics,
    ImageClassification,
    SegmentationReference,
    SpatialEvidence,
)
from domain.provenance import ArtifactIdentity, ComponentVersion
from plugins.synthetic import PLUGIN_VERSION
from plugins.synthetic.imaging import SyntheticSlideReader
from plugins.synthetic.operations import PatchSet
from ports import DataValue, PredictionPayload, StageContext

Floats = npt.NDArray[np.float64]


@dataclass(frozen=True)
class EmbeddingBag:
    rects: tuple[Level0Rect, ...]
    features: Floats  # (N, D)


def _rng(ctx: StageContext, salt: str) -> np.random.Generator:
    checksum = ctx.model_artifact.checksum if ctx.model_artifact else "none"
    return np.random.default_rng(int(digest([checksum, salt, ctx.seed])[:12], 16))


class _Base:
    def __init__(self, definition: ModelDefinition) -> None:
        self._definition = definition

    @property
    def definition(self) -> ModelDefinition:
        return self._definition

    @property
    def component(self) -> ComponentVersion:
        return ComponentVersion("model_plugin", str(self._definition.ref), PLUGIN_VERSION)

    def _require_kind(self, value: DataValue) -> None:
        if value.kind != self._definition.input.kind:
            raise UnsupportedModelInputError(
                f"{self._definition.ref} consumes {self._definition.input.kind}, got {value.kind}"
            )


class SyntheticPatchEncoder(_Base):
    """Per-patch colour statistics projected to `output.feature_dim` dimensions."""

    def predict(self, value: DataValue, ctx: StageContext) -> object:
        self._require_kind(value)
        patches = value.payload
        if not isinstance(patches, PatchSet):
            raise UnsupportedModelInputError("expected a PatchSet payload")
        dim = self._definition.output.feature_dim
        assert dim is not None
        x = patches.pixels.astype(np.float64) / 255.0
        stats = np.concatenate([x.mean(axis=(1, 2)), x.std(axis=(1, 2))], axis=1)  # (N, 6)
        proj = _rng(ctx, "projection").normal(0, 1, size=(stats.shape[1], dim))
        return EmbeddingBag(patches.rects, np.tanh((stats - 0.5) @ proj))


class SyntheticAttentionAggregator(_Base):
    """Attention-pooled linear head. Attention is emitted as ATTENTION_WEIGHT evidence only."""

    def predict(self, value: DataValue, ctx: StageContext) -> object:
        self._require_kind(value)
        bag = value.payload
        if not isinstance(bag, EmbeddingBag):
            raise UnsupportedModelInputError("expected an EmbeddingBag payload")
        dim = self._definition.input.feature_dim
        if dim is None:
            raise UnsupportedModelInputError("aggregator must declare its input feature_dim")
        if bag.features.shape[1] != dim:
            raise UnsupportedModelInputError(
                f"embedding dim {bag.features.shape[1]} != model input dim {dim}"
            )
        schema = self._definition.output_schema
        assert schema is not None
        rng = _rng(ctx, "head")
        w1 = rng.normal(0, 1 / np.sqrt(dim), size=(dim, 8))
        w2 = rng.normal(0, 1, size=8)
        v = rng.normal(0, 1, size=(dim, len(schema.classes)))
        logits_a = np.tanh(bag.features @ w1) @ w2
        attn = np.exp(logits_a - logits_a.max())
        attn /= attn.sum()
        logits = (attn @ bag.features) @ v
        scores = np.exp(logits - logits.max())
        scores /= scores.sum()
        label = schema.classes[int(np.argmax(scores))]
        top_k = int(ctx.stage.config_dict().get("evidence_top_k", 5))
        order = np.argsort(-attn, kind="stable")[:top_k]
        evidence = tuple(
            SpatialEvidence(
                evidence_id=EvidenceId(
                    short_digest([ctx.image.image_id, self._definition.ref, bag.rects[i]], "ev")
                ),
                image_id=ctx.image.image_id,
                geometry=RectangleGeometry(bag.rects[i]),
                semantics=EvidenceSemantics.ATTENTION_WEIGHT,
                producer=self._definition.ref,
                value=float(attn[i]),
                value_is_probability=False,
                rank=rank,
            )
            for rank, i in enumerate(order, start=1)
        )
        output = ImageClassification(
            label=label,
            scores=tuple((c, float(s)) for c, s in zip(schema.classes, scores, strict=True)),
            scores_are_calibrated_probabilities=False,
        )
        return PredictionPayload(outputs=(output,), evidence=evidence)


class SyntheticDirectSegmenter(_Base):
    """Direct image -> segmentation, no patches/embeddings/MIL. Intensity threshold only."""

    def predict(self, value: DataValue, ctx: StageContext) -> object:
        self._require_kind(value)
        if not isinstance(ctx.reader, SyntheticSlideReader):
            raise ExecutionUnsupportedError("synthetic segmenter requires the synthetic reader")
        geom = ctx.image.spatial_reference
        if not isinstance(geom, PyramidalGeometry):
            raise UnsupportedModelInputError("direct 2D segmenter needs a pyramidal image")
        pre = ctx.preprocessing
        assert pre is not None and pre.target_pixel_spacing_um is not None
        spacing = geom.require_pixel_spacing()
        level = next(
            (
                lvl
                for lvl in geom.levels
                if abs(spacing.x_um * lvl.downsample - pre.target_pixel_spacing_um)
                <= 0.01 * pre.target_pixel_spacing_um
            ),
            None,
        )
        if level is None:
            raise ExecutionUnsupportedError("no pyramid level matches the contract spacing")
        rgb = ctx.reader.read_region(
            ctx.image, LevelRect(level.index, 0, 0, level.width, level.height)
        )
        if ctx.channel_conversion is not None:
            rgb = rgb[..., ::-1]
        mask = (rgb.mean(axis=-1) < 200).astype(np.uint8)
        schema = self._definition.output_schema
        assert schema is not None
        identity = ArtifactIdentity.derive(
            SEGMENTATION_2D,
            {
                "model": self._definition.ref,
                "image": ctx.image.image_id,
                "level": level.index,
                "preprocessing": pre.fingerprint,
            },
        )
        ref = SegmentationReference(
            mask_artifact=identity.artifact_id,
            label_map=tuple(enumerate(schema.classes)),
            level=level.index,
            downsample=level.downsample,
        )
        return PredictionPayload(outputs=(ref,), artifacts=((identity, mask),))
